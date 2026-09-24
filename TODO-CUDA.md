# TODO: real-audio decoder on the official CUDA runtime (Windows)

The Mac (MLX) studio can **Continue this recording**: an uploaded recording's music tokens are kept and
rendered together with the model's continuation, using the community NAR adapter from
`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` (`tokens/nar_lora_joint_v9.safetensors`, already
part of the **Audio input** download). The CUDA path does not apply the adapter yet, so the server
rejects `continue_recording` for `cuda-bf16` / `cuda-fp8` (`server.py`, `Job.valid`) and re-rendering
such a song runs on the MLX model that made it.

## What the adapter is

- `checkpoint['lora']`: 392 float32 tensors = 28 layers × 7 linears × (A `[32, in]`, B `[out, 32]`), in
  order per layer: `nar_self_attn.q_proj, k_proj, v_proj, o_proj, nar_mlp.gate_proj, up_proj, down_proj`.
  Scale is 1 (`alpha = rank`). Delta per linear: `W += B @ A`.
- `checkpoint['io']`: full `state_dict`s for `vae2llm` and `llm2vae` (weight + bias, float32) that
  replace the base layers outright.
- `checkpoint['rank']`: 32.
- Reference: `scripts/ar_generate.py` in the HF repo folds both into `pipe._load_model()` weights and
  runs the unmodified pipeline. `scripts/nar_lora.py` documents the training-time module.

## Plan

1. **Load once, fold on demand.** `cuda_engine.Pipeline` keeps the upstream `yue2` model in memory
   across jobs (`MODEL_CACHE`). Folding permanently would change every song, so either:
   - keep the deltas (`B @ A` per linear, ~1.7 GB in bf16 for 28 × 7 matrices — too much) — no; or
   - keep A/B (140 MB) and add/subtract the folded delta around each real-audio synthesis
     (`W += B@A` before, `W -= B@A` after; bf16 rounding drifts a little per toggle — measure); or
   - wrap the seven NAR linears per layer with a `LoRALinear` module like `model/real_audio.py` does
     for MLX. This is the clean option but the upstream CUDA-graph sampler (`yue2.cuda_graph.GraphAR`)
     captures the AR path only; the NAR runs eagerly (`yue2.nar.synthesize`), so wrapping NAR linears
     should be graph-safe. Verify on Windows; the eager `torch-eager` backend used for fp8 is the
     fallback.
2. **Read the checkpoint** with `torch.load(..., weights_only=True)` in the `.cuda-venv` (torch is
   available there; `model/torch_checkpoint.py` is only needed where torch is absent).
3. **Hook points.** `cuda_engine.finish_song` calls `pipe.synthesize(semantic, ...)`; wrap that call with
   the adapter enabled when the job's `continue-input.json` or `render-input.json` has
   `real_audio: true` (the same flags `worker.py` uses on MLX). `io` layers: swap
   `model.vae2llm` / `model.llm2vae` in the same context.
4. **Lift the server guard** in `Job.valid` (`is_cuda(self.model)` check under `continue_recording`) and
   the MLX-only hint in `web/opensuno-app.js` / `web/song-timeline.js` (`continueRecording` button title,
   `submitContinueRecording`). Add the CUDA case to `tests/test_real_audio.py` and
   `tests/test_cuda_continuation.py`.
5. **fp8 variant.** The adapter deltas must be applied to the bf16 weights before quantization or
   kept as a separate bf16 path; check what `quantization='fp8'` does to `nar_*` linears in the pinned
   `yue2` runtime before deciding.

## Out of scope for now

- Training artist AR LoRAs (`ar_lora_cursor.py`): CUDA-only, 24 GB GPU, hours; also needs the
  `Mothersuperior/yue2-minted-corpus` regularizer pack (~100 MB) and Demucs + MMS alignment.
- Loading a user-supplied AR LoRA at inference (`ar_generate.py` `AR_SCALE`): would follow the same
  fold-or-wrap decision as above but on `self_attn` / `mlp`, and needs the MLX equivalent too.
