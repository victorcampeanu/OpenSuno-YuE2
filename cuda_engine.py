"""Small studio adapter around the pinned, unmodified upstream YuE2 runtime."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import time
import re
import sys
import json
from functools import wraps
import numpy as np
import torch
import soundfile as sf
from yue2.pipeline import YuE2Pipeline
from yue2.protocol import Sampling
from yue2.pipeline import SymbolicPlan, SemanticResult, SongResult
from yue2.protocol import SongRequest, token_prefixes, negative_prefix, ABC_START, ABC_END, CODEC_OFFSET, EOD, VOCAB_SIZE
from yue2.storage import identity
import song_continuation

# Windows Torch exposes the FlashAttention operator even when it was not built.
# Select upstream's supported cuDNN graph path instead of its operator-only auto check.
if sys.platform == 'win32' and not torch.backends.cuda.is_flash_attention_available():
    from yue2 import nar
    _original_nar_attention = nar.attention

    @wraps(_original_nar_attention)
    def windows_nar_attention(q, k, v, *, query_chunk_size=None, **kwargs):
        if q.device.type == 'cuda' and kwargs.get('backend', 'sdpa') == 'sdpa' and query_chunk_size is None:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            query, key, value = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
            params = torch.backends.cuda.SDPAParams(
                query, key, value, None, 0.0, kwargs.get('causal', False),
                query.shape[1] != key.shape[1])
            if torch.backends.cuda.can_use_cudnn_attention(params):
                # Default SDPA dispatch can select math on Windows despite
                # cuDNN supporting the shape. Require the fused kernel here.
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    return _original_nar_attention(q, k, v, **kwargs)
        # Windows SDPA may fall back to math attention. Bound its temporary
        # score matrix while retaining every key and the original causal mask.
        return _original_nar_attention(q, k, v,
                                       query_chunk_size=query_chunk_size or 256,
                                       **kwargs)

    nar.attention = windows_nar_attention
    from yue2 import cuda_graph
    class WindowsGraphAR(cuda_graph.GraphAR):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('attention_backend', 'cudnn')
            super().__init__(*args, **kwargs)

        def prefill(self):
            # The upstream eager prefix pass otherwise permits a math fallback,
            # even though subsequent captured decode steps already use cuDNN.
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return super().prefill()
    cuda_graph.GraphAR = WindowsGraphAR


def write_wav(path, audio):
    sf.write(str(path), audio, 48000, subtype='PCM_16')


def finish_song(pipe, request, semantic, cancel, **config_extra):
    """Synthesize and decode prepared music tokens into a SongResult."""
    cancel()
    tick = time.perf_counter()
    latents = pipe.synthesize(semantic, cancelled=lambda:(cancel() or False))
    synthesis_seconds = time.perf_counter()-tick
    cancel()
    audio = pipe.decode(latents)
    config = pipe.effective_config(request)
    config.update(config_extra)
    timing = {'nar_seconds':synthesis_seconds, 'vae_seconds':time.perf_counter()-tick-synthesis_seconds, **semantic.timing}
    return SongResult(audio,48000,semantic,latents,config,pipe.weights,timing,
                      identity({'request':request.to_dict(),'prefix':semantic.plan.prefix,'codec':semantic.tokens,'config':config}))


def render_saved(pipe, saved, request, cancel):
    """Run only synthesis and decoding, retaining the original music tokens."""
    abc = saved['abc'] or None
    abc_ids = pipe.tokenizer.encode(abc) if abc else []
    prefix = saved.get('prefix')
    if prefix is not None and request.cot != 'off':
        abc_ids = prefix[prefix.index(ABC_START)+1:prefix.index(ABC_END)]
    prefix = prefix or token_prefixes(request, pipe.tokenizer, abc_ids)
    plan = SymbolicPlan(request, abc, abc_ids, prefix)
    semantic = SemanticResult(plan, saved['codec'], {'reused_tokens':True}, False)
    return finish_song(pipe, request, semantic, cancel, reused_tokens=True)


def continue_plan(pipe, request, abc_prefix, sampling, callbacks):
    """Let the model write the rest of a score that was cut at a bar.

    The prompt ends with ABC_START and the kept bars; the model appends bars until it
    closes the score. Without a plan (cot=off) the plan is the request alone.
    """
    if request.cot == 'off' or not abc_prefix:
        return SymbolicPlan(request, None, [], token_prefixes(request, pipe.tokenizer))
    kept = list(pipe.tokenizer.encode(abc_prefix))
    prefix = token_prefixes(request, pipe.tokenizer) + kept
    limits = song_continuation.budget(asdict(sampling), len(prefix))
    sampling = replace(sampling, max_tokens=limits['max_tokens'], min_tokens=limits['min_tokens'])
    ids, timing, truncated = pipe._generate(prefix, sampling, request.seed, 'abc', **callbacks)
    abc_ids = kept + [int(t) for t in ids]
    return SymbolicPlan(request, pipe.tokenizer.decode(abc_ids), abc_ids,
                        token_prefixes(request, pipe.tokenizer, abc_ids), timing, truncated)


def replan(pipe, plan, abc):
    """The same plan with its score text replaced (instrumental rewrite)."""
    if abc == plan.abc:
        return plan
    abc_ids = list(pipe.tokenizer.encode(abc))
    return SymbolicPlan(plan.request, abc, abc_ids, token_prefixes(plan.request, pipe.tokenizer, abc_ids),
                        plan.timing, plan.truncated)


class _Stop(Exception):
    """Raised from the token callback once ``stop`` is satisfied, ending the upstream loop early."""


class _FirstToken:
    """The upstream loop has no logits hook, so ``distribution`` is wrapped: while a proposal is
    running, the first sampled token is limited to text tokens matching the writer's pattern."""
    mask = None
    installed = False

    @classmethod
    def install(cls):
        if cls.installed:
            return
        from yue2 import sampling as upstream
        original = upstream.distribution

        @wraps(original)
        def distribution(logits, sampling, history, step, phase, *args, **kwargs):
            if step == 0 and cls.mask is not None:
                logits = logits + cls.mask.to(logits.device, logits.dtype)
            return original(logits, sampling, history, step, phase, *args, **kwargs)
        upstream.distribution = distribution
        cls.installed = True


class ScoreWriter:
    """cover_arrangement writer on the upstream runtime: each proposal re-prefills the committed score.

    The upstream generator has no incremental interface, so a proposal is one short ``abc``
    generation after the request text, ABC_START and the committed text. ``stop`` ends the
    generation from the token callback as soon as the decoded proposal satisfies it, and
    ``first`` limits the first token through a wrapped ``distribution``, so the arrangement
    is held to the same constraints as on MLX.
    """
    def __init__(self, pipe, style, lyrics, sampling, seed, callbacks):
        self.pipe, self.sampling, self.seed, self.callbacks = pipe, sampling, seed, callbacks
        self.base = list(token_prefixes(SongRequest(style=style, lyrics=lyrics, cot='full', seed=seed), pipe.tokenizer))
        self.text = ''
        self._first_masks = {}

    def commit(self, text):
        self.text += text

    def first_mask(self, pattern):
        """Additive mask keeping only text tokens whose decoded text fully matches ``pattern``."""
        if pattern not in self._first_masks:
            regex = re.compile(pattern)
            ids = [i for i in range(EOD) if regex.fullmatch(self.pipe.tokenizer.decode([i]))]
            if not ids:
                raise ValueError(f'No token matches {pattern!r}')
            mask = torch.full((VOCAB_SIZE,), float('-inf'))
            mask[ids] = 0
            self._first_masks[pattern] = mask
        return self._first_masks[pattern]

    def propose(self, max_tokens, stop, first=None):
        prefix = self.base + list(self.pipe.tokenizer.encode(self.text))
        if len(prefix) + max_tokens + 64 > song_continuation.CONTEXT:
            raise ValueError('The arranged score does not fit the model context')
        self.seed = (self.seed + 1) % 4294967296
        sampled, report = [], self.callbacks.get('on_token')

        def on_token(phase, token):
            if report is not None:
                report(phase, token)
            if token == ABC_END:
                return
            sampled.append(int(token))
            if stop(self.pipe.tokenizer.decode(sampled)):
                raise _Stop()
        _FirstToken.install()
        _FirstToken.mask = self.first_mask(first) if first else None
        try:
            self.pipe._generate(prefix, replace(self.sampling, min_tokens=0, max_tokens=max_tokens), self.seed, 'abc',
                                **{**self.callbacks, 'on_token': on_token})
        except _Stop:
            pass
        finally:
            _FirstToken.mask = None
        return self.pipe.tokenizer.decode(sampled)


def continue_semantic(pipe, plan, codec_prompt, sampling, callbacks):
    """Generate music tokens after a prompt of saved codec tokens; the prompt is kept so the whole song is synthesized."""
    request = plan.request
    prompt = [int(c) + CODEC_OFFSET for c in codec_prompt]
    prefix = list(plan.prefix) + prompt
    negative = (list(negative_prefix(request, pipe.tokenizer, plan.abc_ids)) + prompt) if request.guidance != 1 else None
    limits = song_continuation.budget(asdict(sampling), len(prefix), len(negative or []))
    sampling = replace(sampling, max_tokens=limits['max_tokens'], min_tokens=limits['min_tokens'])
    ids, timing, truncated = pipe._generate(prefix, sampling, request.seed, 'semantic', negative=negative,
                                            cfg_scale=request.guidance, legacy_off=request.cot == 'off', **callbacks)
    new = [int(t) - CODEC_OFFSET for t in ids]
    if not new:   # same failure as mlx_continuation: a song that ends at once is not a result
        raise RuntimeError('The model ended the song right away. Cut it earlier or try a different seed.')
    tokens = list(codec_prompt) + new
    timing = {**timing, 'prompt_frames': len(codec_prompt)}
    return SemanticResult(plan, tokens, timing, truncated)


class Pipeline(YuE2Pipeline):
    def __init__(self, root, variant, report, cancel, lora=None, drop_ar_file=None):
        # ``lora``: ``[(file, strength), ...]`` — the LoRA and the Sound LoRA folded in together when the model loads.
        # ``drop_ar_file``: on covers, fold that Writes adapter as decoder-only so the transcribed score cannot loop.
        self.report, self.cancel, self.root, self.lora = report, cancel, root, lora or []
        self.drop_ar_file = drop_ar_file or ''
        if not torch.cuda.is_available():
            raise RuntimeError('The official model requires an NVIDIA CUDA GPU.')
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        super().__init__(root/'model/cuda/backbone', root/'model/cuda/vae', device='cuda',
                         memory_budget_gib=total, vae_core_frames=512, offload_ar=True,
                         backend='torch-eager' if variant == 'cuda-fp8' else 'torch',
                         quantization='fp8' if variant == 'cuda-fp8' else 'none')

    def _load_model(self, for_nar=False):
        # Load the BF16 weights on the CPU and fold the chosen LoRAs in before upstream quantizes or moves them.
        if self._model is None and self.lora:
            import loras
            from yue2.modeling_yue2 import YuE2ForCausalLM
            with self._status('Loading model'):
                start = time.perf_counter()
                self._model = YuE2ForCausalLM.from_pretrained(self.model_dir, local_files_only=True,
                                                              torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
                self.load_timing['mot_load_seconds'] = time.perf_counter() - start
            with self._status('Applying the LoRA ' + ' and '.join(name for name, _ in self.lora)):
                dims = loras.dims_from_config(json.loads((self.model_dir/'config.json').read_text(encoding='utf-8')))
                parts = []
                for name, strength in self.lora:
                    delta = loras.check(loras.load(self.root, name, dims), dims)
                    if name == self.drop_ar_file:
                        delta = loras.drop_ar(delta)
                        if not delta['linears'] and not delta['io']:
                            continue
                    parts.append((delta, strength))
                touched = loras.merge_torch(self._model, loras.combine(parts), 1.0) if parts else 0
                for name, strength in self.lora:
                    print(f'[lora] {name} at strength {strength:g}', flush=True)
                if self.drop_ar_file:
                    print('[lora] Cover: decoder half of the Writes adapter only, so the transcribed melody does not loop', flush=True)
                print(f'[lora] {touched} adapted layers', flush=True)
        return super()._load_model(for_nar)

    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        owner = self
        stage = ('synthesis' if label.startswith('Synthesizing') else
                 'decoding_audio' if label.startswith('Decoding') else
                 'planning' if label.startswith('Planning') else
                 'music_tokens' if label.startswith('Generating') else 'loading')
        owner.report(stage=stage, loading_detail=label if stage == 'loading' else None)
        class Status:
            def update(self, completed, total=None):
                owner.cancel()
                if stage == 'synthesis':
                    owner.report(stage=stage, steps=completed, step_total=total or 32)
            def advance(self): owner.cancel()
            def finish(self, **kwargs): owner.cancel()
        yield Status()

    def callbacks(self, request):
        counts, ticks = {}, {}
        def cancelled():
            self.cancel()
            return False
        def token(phase, value):
            self.cancel()
            ticks.setdefault(phase, time.perf_counter())
            counts[phase] = counts.get(phase, 0) + 1
            n = counts[phase]
            if n == 1 or n % 20 == 0:
                self.report(stage='planning' if phase == 'abc' else 'music_tokens', tokens=n,
                            token_limit=request['abc_sampling' if phase == 'abc' else 'semantic_sampling']['max_tokens'],
                            tokens_per_second=round(n/max(.001,time.perf_counter()-ticks[phase]),1))
        return dict(cancelled=cancelled, on_token=token)


def run(root, jobdir, r, cache, report, cancel, save_meta, style, lyrics, start, analysis_elapsed, abc_transform=None,
        prepare_score=None):
    """``prepare_score(pipe, abc)`` rewrites a supplied score once the model is loaded (cover arrangement)."""
    abc_transform = abc_transform or (lambda abc: abc)
    variant = r['model']
    # LoRAs are folded into the weights when the model loads (CUDA graphs capture the weights, so nothing is
    # switched at run time); another adapter or strength means loading the model again.
    import loras
    lora = loras.chosen(r)
    drop_ar_file = r.get('lora') or '' if r['kind'] == 'cover' else ''
    key = variant + ''.join(f'+{name}@{strength:g}' for name, strength in lora) + ('+cover-decoder' if drop_ar_file else '')
    pipe = cache.get(key)
    if pipe is None:
        for old in list(cache.values()):
            if isinstance(old, Pipeline): old.close()
        for k in [k for k, v in cache.items() if isinstance(v, Pipeline)]: del cache[k]
        pipe = cache[key] = Pipeline(root, variant, report, cancel, lora=lora, drop_ar_file=drop_ar_file or None)
    pipe.report, pipe.cancel = report, cancel
    if r.get('_preload'):
        pipe._load_model()
        raise SystemExit(0)
    torch.cuda.reset_peak_memory_stats()
    pipe.generation_config = replace(pipe.generation_config, ode_steps=r['steps'])
    if prepare_score is not None:
        r['abc'] = prepare_score(pipe, r.get('abc', ''))
    def args(seed):
        return dict(style=style, lyrics=lyrics, cot=r['cot'], seed=seed, abc=r['abc'] or None,
                    cfg_scale=r['cfg_scale'], abc_sampling=Sampling(**r['abc_sampling']), **pipe.callbacks(r))
    def write_plan(seed):
        plan = pipe.plan(**args(seed))
        if not plan.abc or not plan.abc.strip(): raise RuntimeError('The model did not produce a score. Try a different seed.')
        return abc_transform(plan.abc), plan.truncated
    if r['kind'] == 'plan':
        abc, truncated = (r['abc'], False) if r.get('abc') else write_plan(r['seed'])
        (jobdir/'score.abc').write_text(abc, encoding='utf-8')
        return dict(abc=abc, elapsed=round(time.perf_counter()-start,2), model=variant, truncated=truncated)
    count = r.get('candidates',1)
    meta = dict(candidates=[], model=variant, requested_candidates=count)
    for index in range(1,count+1):
        cancel()
        seed = (r['seed']+index-1)%4294967296
        destination = jobdir if count == 1 else jobdir/f'candidate-{index:02d}'
        destination.mkdir(exist_ok=True)
        report(candidate=index, candidates=count, tokens=0, tokens_per_second=0, steps=0)
        tick = time.perf_counter()
        callbacks = pipe.callbacks(r)
        if r.get('render_source'):
            saved = json.loads((jobdir/'render-input.json').read_text())
            request = SongRequest(style=style,lyrics=lyrics,cot=r['cot'],seed=seed,
                                  abc=saved['abc'] or None,cfg_scale=r['cfg_scale'])
            report(tokens=len(saved['codec']),tokens_per_second=0)
            result = render_saved(pipe,saved,request,cancel)
        elif (jobdir/'continue-input.json').is_file():
            # Extend / regenerate from here: continue the cut plan, then the music after the kept tokens.
            saved = json.loads((jobdir/'continue-input.json').read_text())
            request = SongRequest(style=style,lyrics=lyrics,cot=r['cot'],seed=seed,cfg_scale=r['cfg_scale'])
            plan = continue_plan(pipe,request,saved.get('abc_prefix',''),Sampling(**r['abc_sampling']),callbacks)
            if plan.abc: plan = replan(pipe,plan,abc_transform(plan.abc))
            semantic = continue_semantic(pipe,plan,saved['codec'],Sampling(**r['semantic_sampling']),callbacks)
            result = finish_song(pipe,request,semantic,cancel,continued_from=saved.get('source'),prompt_frames=len(saved['codec']))
        else:
            settings = args(seed)
            # Instrumental songs plan first so the score can be rewritten before any music tokens are generated.
            if r.get('instrumental') and r['cot'] != 'off' and not settings['abc']:
                settings['abc'], _ = write_plan(seed)
            result = pipe(**settings, semantic_sampling=Sampling(**r['semantic_sampling']))
        write_wav(destination/'audio.wav', result.audio)
        if result.abc: (destination/'score.abc').write_text(result.abc, encoding='utf-8')
        (destination/'tokens.json').write_text(json.dumps(result.semantic.tokens))
        (destination/'render-prefix.json').write_text(json.dumps(result.semantic.plan.prefix))
        settings = {**r, 'seed':seed, 'candidates':1, 'random_seed':False}
        (destination/'settings.json').write_text(json.dumps(settings,indent=2))
        (destination/'cuda-runtime.json').write_text(json.dumps(dict(config=result.config,timing=result.timing,weights=result.weights,
            gpu=torch.cuda.get_device_name(0),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30),indent=2))
        if count > 1: (destination/'request.json').write_text(json.dumps(settings,indent=2))
        item = dict(index=index, seed=seed, model=variant, seconds=round(len(result.audio)/48000,3),
                    elapsed=round(time.perf_counter()-tick,2), abc=result.abc,
                    prefix='' if count == 1 else f'candidate-{index:02d}/',
                    peak_memory_gb=round(torch.cuda.max_memory_allocated()/2**30,2))
        meta['candidates'].append(item)
        if count == 1: meta.update({k:v for k,v in item.items() if k not in {'index','prefix'}})
        meta['elapsed'] = round(analysis_elapsed+time.perf_counter()-start,2)
        save_meta(meta)
        del result
        torch.cuda.empty_cache()
    return meta
