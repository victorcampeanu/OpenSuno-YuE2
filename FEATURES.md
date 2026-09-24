# OpenSuno feature documentation

OpenSuno is a local music studio built around the [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
song model. It started (12 September 2026) as a browser front end with a resumable model
downloader, and grew into a full production tool: cover analysis, model-written
arrangements, song editing, extension, community LoRAs, workspaces, and a second GPU runtime for Windows.
This document describes every feature, how it works, and what it improves. For installation and day-to-day usage see [README.md](README.md) and
[START-HERE.md](START-HERE.md).

Features are grouped by area rather than by commit. A chronological list is at the end.

---

## 1. How YuE2 generates a song (background for everything below)

YuE2 is a decoder-only transformer with two heads. A song is produced in four stages, and almost
every OpenSuno feature works by shaping the input to one of them:

| Stage | What the model does | What OpenSuno feeds it |
| --- | --- | --- |
| 1. Plan (`cot=melody`/`full`) | Writes an ABC score: a `Vocal` voice (melody, chord symbols) and an `Ins` voice (instrumental lead), grouped in `% intro`, `% verse`, … sections | Instruction, `[Tags]` style, `[Lyrics]`; or a supplied score |
| 2. Semantic | Autoregressively emits 25 music tokens per second from a vocabulary of 32,768 codes, conditioned on the request and the score | The full text prefix, optional classifier-free guidance |
| 3. Acoustic (NAR) | Flow-matching from noise to 64-dimensional audio latents, guided by the music tokens | Midpoint ODE steps (default 2) |
| 4. VAE | Decodes latents to 48 kHz stereo | Tiled decode |

The score is the most powerful lever. The model renders what the score says: a chord on a bar
produces that harmony, a silent `Vocal` voice produces no singing, a bare melody line produces a
bare arrangement. Several features (instrumental mode, cover arrangement, composition editing,
Extend) are score-level manipulations for that reason.

The prompt format is fixed by the model's training:

```
<instruction>
[Tags]
<style>
[Lyrics]
<lyrics with [Verse]/[Chorus] tags>
<abc> …score… </abc> <music> …codec tokens… </music>
```

Every song in `library/` keeps its `score.abc`, `tokens.json` (music tokens), `render-prefix.json`
(the exact text prefix) and `settings.json`, which is what makes re-rendering,
editing and extending possible later.

---

## 2. Studio core

### 2.1 Browser studio and job model

- **What:** A single-page app (`web/`) served by FastAPI (`server.py`) on `127.0.0.1:7862`. Jobs are
  folders under `library/<timestamp>-<hex>/` with `request.json`, `progress.json`, `state.json`,
  `run.log` and results. Kinds: `generate`, `cover`, `transcribe`, `plan`, `tokenize`.
- **How:** The browser polls `/api/jobs`; a worker process writes `progress.json` atomically
  (`runtime_platform.replace_file`). The run panel turns that into plain language: the current
  stage, a progress bar, four steps (getting ready, composing, writing the music, making the
  audio) and one line with the version and the current token speed. It gives no time estimate,
  because a song can stop well before its token cap; exact counts stay in `progress.json` and
  `run.log`. The panel appears only while a job is queued or running, or when one stopped early,
  and disappears once the song is saved. The progress bar maps log markers (`[plan]`, `[semantic]`,
  `[nar]`, `[vae]`) to percentages. The job continues if the page reloads.
- **Improves:** Generation is observable and resumable instead of a black-box command line.

### 2.2 Candidates, seeds and settings

- Up to 8 candidates run sequentially with seeds `base + i`; finished candidates play while the rest
  generate. Seeds are random per generation unless **Lock seed** is on; **Use these settings** locks
  a candidate's seed. `settings.json` per candidate stores the exact request for reproduction.
- Four sliders are the only sampling controls: **Weirdness** maps to temperature / top-p / top-k
  curves around the model defaults (`generation-defaults.json`: 1.0 / 0.95 / 100), **Style influence**
  to `cfg_scale` (1 → 1.2 → 2), **Repetition** to `repetition_penalty` (Normal = 1.2), **Audio steps**
  to the flow-matching steps (default 2). A saved request whose values sit off a curve (older songs,
  CLI-made requests) restores as *Custom* and is sent back verbatim until the slider is moved. The
  raw-value fields and the duplicate *Prompt strength* slider were removed on 2026-09-17.
- **Composition** (2026-09-17) drives the *planner's* sampling — the pass that writes the ABC score
  before any music token exists (`abc_sampling`: temperature 0.5 → 0.7 → 1.0, top-p 0.8 → 0.9 → 0.97,
  top-k 12 → 30 → 64 across the slider). Left is familiar chord choices and melodic turns, right is
  more adventurous writing and a busier instrumental line; unlike Weirdness it changes *what* is
  played rather than how it is rendered. It is greyed out when no score is written (No Plan, or a
  cover that keeps the recording's score without *Arrange with the model*). Saved requests whose
  planner values sit off the curve restore as *Custom* and are sent back verbatim. The plan mode
  select formerly labelled "Composition" is now **Plan**.
- **Duration** is a cap on music tokens (25 per second), not a target.
- **LoRA** (2026-09-17): community adapters dropped into `loras/` (beside `model/`) appear in a
  picker with a 0–2 strength; see 3.5.
- **Improves:** Musically named controls, and every result can be reproduced from `settings.json`.

### 2.3 Job queue

- **What:** Submitting while a job runs queues the new job with a position badge; queued songs can be
  renamed, removed or deleted from the library. The status panel always shows the generation that is
  running (stage, progress, tokens/sec, **Stop generation**); when the song you submitted is waiting,
  a line underneath says so ("Your song is next…" / "2nd of 3 waiting") with its own **Remove from
  queue** button, instead of the panel going blank with "Waiting in queue".
- **How:** `server.py` keeps `queue` and `pending`; `start_next` launches the next ready job when the
  active one finishes. Analysis and tokenize jobs jump to the front because generations depend on
  them. Closing the last page cancels waiting jobs.
- **Improves:** Batch work without babysitting the page.

### 2.4 Voice selection

- **What:** Any / Male / Female / Duet next to the composition mode.
- **How:** The model has no voice input, so `worker.py` appends a tag line (`Male lead vocal.`,
  `Female lead vocal.`, or a duet description) to the style for both planning and music stages.
  Instrumental songs skip it and get `Instrumental.` instead. This is the only place the singer's
  gender enters the prompt: the Style Builder and presets set this control instead of writing
  voice words into the text, and Ask AI is told to describe vocal character without gender, so the
  model never sees two voice lines that disagree.
- **Improves:** A reliable single control for the most-requested style adjustment.

### 2.5 Library: downloads, badges, rename, trash, workspaces

- **Downloads:** WAV directly; MP3 via `POST /api/jobs/{j}/songs/{i}/export?format=mp3`, which
  converts once with FFmpeg at 320 kbps and reuses the file (`audio_export.py`).
- **Badges:** Duration and generation time per version; the seed is in details.
- **Rename / Delete:** Per version (`song_library.py`). Deleted versions move to `.trash/`, never
  erased by the app; uploaded sources stay.
- **Workspaces** (`workspaces.py`, `web/workspaces.js`): Named folders stored in `.workspaces.json`
  as `job:index → workspace` memberships; audio never moves. Bulk select (shift-range, select all
  shown) and **Move to workspace**; **Save to** sets the destination for new songs; deleting a
  workspace returns songs to *My Workspace*. Endpoints: `GET/POST /api/workspaces`,
  `PATCH/DELETE /api/workspaces/{id}`, `POST /api/workspaces/move`.
- **Improves:** A growing library stays organised without touching files on disk.

### 2.6 Album art

- **What:** Each generated song or cover gets a cover image when an OpenAI key is configured.
- **How:** `queue_artwork` asks OpenAI Images for an 816×816 WebP at low quality (about 26× smaller
  than the earlier PNGs) and writes `artwork.webp` + `artwork.json`; `GET/POST /api/jobs/{j}/artwork`.
- **Improves:** The library looks like a music library, at negligible disk cost.

### 2.7 Style helpers

- **Build a style** (`web/style-builder.js`): a client-side composer with genre, voice and delivery,
  instruments, mood, tempo (30–240 BPM), production and intro. Apply appends or replaces the style
  text. Choosing *Instrumental only* toggles the instrumental switch; a vocal type turns it off.
- **Style presets** (`web/style-presets-data.js`, `web/style-presets.js`; 73 presets from SongScribe,
  MIT): the first section of Build a style. Presets are structured fields (genre, bpm, key, mood,
  scene, production, instruments, vocal presence/timbre/delivery), rendered as one YuE2 line in the
  order genre → tempo → mood → instruments → voice → production → scene → language at three detail
  levels (*tags*, *full*, *rich*); articles are stripped and a few prose phrases aliased so the
  output reads as tags. **Blend with** interleaves the list fields of a second preset by a chosen
  weight and switches bpm/key/vocal presence past 50%; **Era**, **Texture** and **Mood shift**
  prepend phrases without replacing. The builder's language folds into the preset line and its
  vocal type sets the song's Voice option (or Instrumental) rather than the text (§2.4); the other
  builder choices follow as sentences on a second line. Grouped by category in the shared searchable dropdown.
- **Ask AI for a style** (`style_ai.py`, `web/style-ai.js`): OpenAI writes one comma-separated style
  line (≤400 characters, no artist names) from a short description; the suggestion is editable
  before applying. Key from **Settings** (kept in this browser's `localStorage`), `.openai-key`, or
  `OPENAI_API_KEY`. `POST /api/style/models` lists the account's chat models; **Test key** checks
  it. Default model `gpt-4o-mini`.
- **Not offered any more (2026-09-17):** *Style reading* (CLAP + MAEST describing an upload as a
  style line) and *Add its style to prompt* (melodic-density / harmony phrases counted from a
  transcription). Both only produced text for this box, and their vocabulary ("stepwise contour",
  "one chord per bar") is not caption language YuE2 was trained on, so they moved the result less
  than a preset or a sentence typed by hand. Ask AI and the presets cover the same need.
- **Improves:** Better prompts are the cheapest quality gain with a prompt-driven model.

### 2.8 Find lyrics

- **What:** Search LRCLIB by title and artist, preview versions, **Use these lyrics**.
- **How:** `GET /api/lyrics/search?q=` (`lyrics_provider.py`, read-only). Synced (LRC) lyrics are
  kept alongside the plain text so covers can place lines by time (see §4.4). The search is
  prefilled from the song title or the uploaded recording.
- **Improves:** Real lyrics in seconds, and timestamped lyrics that make covers line up.

### 2.8b Section tag picker

- **What:** **Add Tag** in the lyrics header (and in the expanded lyrics editor) inserts
  `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Interlude]`,
  `[Instrumental]`, `[Rap]` or `[Outro]` at the caret. Selected lines are wrapped with the tag.
- **How:** The last caret in `#lyrics` / `#expandedLyrics` is remembered before **Add Tag** takes
  focus. The chosen tag is placed on its own line so you do not type the brackets each time.
- **Improves:** Song structure can be built by pointing and tapping instead of typing labels.

### 2.8c Lyrics structure check (`web/lyrics-structure.js`)

- **What:** A note under the lyrics when section tags are not in `[Tag]` form (`(Chorus)`,
  `{hook}`, `Verse 2:`) with a one-click **Fix tags**, and a warning when the lyrics need more
  singing time than the **Maximum duration** cap allows.
- **How:** Bracketed or bare `Word:` lines are mapped through an alias table (hook/refrain →
  Chorus, middle 8 → Bridge, solo/break → Instrumental, …) to the tag set of the picker; numbers
  are kept, case differences are not flagged, unrecognised tags are left alone. Singing time is
  syllables / 3.2 per second plus 4 s per section, reported as a 0.65×–1.6× range; only a cap
  below the low end warns, since the cap is a maximum, not a target. Adapted from SongScribe.
- **Improves:** A malformed tag no longer silently drops structure from a render that takes minutes.

### 2.8d Embedded tags from uploads (`recordings.embedded_tags`)

- **What:** Uploading a recording with ID3 / MP4 / Vorbis / FLAC tags prefills the title (and
  the library entry's name with the artist) and, when the lyrics box is empty, the lyrics.
- **How:** `ffprobe -show_entries format_tags:stream_tags` on the node; `title`, `artist`,
  `lyrics`/`USLT`/`©lyr` are read. LRC time stamps are stripped for the lyrics box and the stamped
  text is kept as `synced_lyrics` for §4.4. The upload response gains `title`, `artist`, `lyrics`,
  `synced_lyrics` (empty strings when absent).
- **Improves:** Covers of tagged files start with the right title and words without a search.

### 2.9 macOS launcher, Windows launcher and hosted mode

- **macOS:** The disk image (`Build OpenSuno Installer.command` → `dist/OpenSuno.dmg`) installs
  **OpenSuno.app** and **OpenSuno Render Node.app**; from a source checkout `Launch Studio.command`
  starts the server through launchd in the background, waits until ready and opens the browser.
  Logs in `~/Library/Logs/OpenSuno/server.log`.
- **Windows:** `Install OpenSuno.ps1` / `Launch Studio.ps1` (see §7).
- **Hosted mode:** With `OPENSUNO_HOSTED` or `VERCEL` set, `server.py` serves the website and the
  GPU-free APIs (workspaces, style key shape) on Vercel; `POST /api/jobs` returns 503 and downloads
  404. Music generation is not connected there yet.

---

## 3. Models, workers and performance

### 3.1 Resumable, verified model downloads

- **What:** Models are downloaded from the Models panel as packages (`bf16`, `cuda-bf16`,
  `cuda-fp8`, `analysis`, `tokens`) with live percentage, bytes and speed; failures resume. LoRA adapters
  (`YuE2 instrumental (Mothersuperior v3)`, `Old School Hip-Hop`, and the becausereasons
  artist packs MLTNT, CHNSN, QWWL / DRKSF and CNZN) are listed under a
  separate LoRAs heading, grouped by family with trigger words and recommended settings, and are not part of
  Install all missing. **Download all** on that heading starts every missing adapter at once.
- **How:** `model-assets.json` pins repository, revision, path, size and SHA-256 for every file;
  `lora-assets.json` does the same for community adapters, which install into `loras/`.
  `model_downloads.py` downloads one file at a time with HTTP Range into `.part`, verifies size and
  SHA-256, and only then replaces the target; servers without range support restart the file safely.
  Several packages can save at once (each LoRA is its own job); a file already transferring is claimed
  until that job finishes so overlapping model files are not double-fetched. Shared files (VAE,
  tokenizer) are symlinked, or hard-linked where Windows denies symlinks. Downloads are blocked while
  a generation is active. Ready models can generate while others download. The Models panel only
  disables the package that is transferring or already installed.
- **Improves:** A 7.5 GB download over a flaky connection never corrupts an install; nothing is in Git.

### 3.2 Resident workers and preloading

- **What:** Models load once and stay in memory between batches; the page shows "ready in memory".
- **How:** `worker_pool.py` keeps one long-lived `resident_worker.py` per model key (and one for
  analysis); each job is a JSON line on stdin, and `worker.py` runs inside that interpreter with
  `MODEL_CACHE` shared. `model_preloads.py` warms a model through a fake job under `.preloads/` when a
  page opens (BF16 on load, SheetSage2/MERT when opening Cover / Remix). A page-connection lease
  (`GET /api/session`, server-sent events) releases workers 20 s after the last page closes.
  Cancellation touches `.cancel-resident`; exit code 2 keeps the models, an unresponsive worker is
  killed after 5 s, and a crashed worker reloads on next use.
- **Improves:** Second and later songs start in seconds instead of re-reading gigabytes of weights.

### 3.3 Semantic decoding speed-ups (MLX)

- **Sliced output head:** The semantic stage can only emit `MUSIC_END` and the 32,768 codec tokens,
  which are contiguous in the 184,704-token vocabulary. `phase_window`/`logits_head` compute logits
  for that ~18 % window only (~17 % faster token generation). ABC planning keeps the full head.
- **Seed-stable sampling:** Gumbel noise is drawn for the whole vocabulary and sliced, so a seed
  produces the same song with or without the sliced head.
- **CFG batching:** Conditional and unconditional branches are prefilled separately, then decoded
  together in one batched step (`pack_cfg_caches`).
- **Host-sync batching:** Tokens are copied to the CPU every 20 steps, not every step.
- **Analysis decoder on CPU:** On Apple Silicon the tiny SheetSage2 BART decoder runs on CPU, where
  it is ~1.9× faster than MPS for incremental decode; MERT stays on the GPU. The analysis loop also
  releases unused Metal cache during long transcriptions.
- **Defaults:** CFG defaults to 1.2 and audio synthesis to 2 midpoint steps (`generation-defaults.json`),
  the fastest settings that held up in listening comparisons; both remain adjustable.

### 3.4 Native instrumental mode (`instrumental.py`)

- **What:** *Instrumental only* produces music without singing, keeping the hook.
- **How:** Asking for "no vocals" in the style is unreliable. Instead the plan is written first, then
  the `Vocal` voice is rewritten as rests of the same durations (chords stay where the dialect
  requires them) and the sung melody is moved into the `Ins` voice. A silent `Vocal` voice is the
  model's own "nobody sings here" signal. Lyrics are reduced to their section tags (or a default
  Intro…Outro structure) and `Instrumental.` is appended to the style; the UI warns about vocal words
  left in the style. The bar grid is verified unchanged.
- **Improves:** Instrumentals that are actually instrumental, with the melody carried by an instrument.

### 3.5 Community LoRAs (`loras.py`, `model/adapters.py`; 2026-09-17)

- **What:** any YuE2 LoRA `.safetensors` placed in `loras/` (or the render node's `loras/`) can be
  chosen per song under *More Options → LoRA* with a strength of 0–2. The file name is stored in the
  request, so **Use these settings** and re-renders pick the same adapter; the library shows a
  `LoRA` badge. `GET /api/loras` lists the folder (the picker refreshes when opened; a node reports
  its list in `/v1/health`); a job that names a file the renderer does not have is refused.
- **Formats:** `loras.py` reads safetensors with the standard library and NumPy (bf16 widened
  exactly) and understands four layouts — upstream / Hugging Face (`layers.N.self_attn.q_proj.lora_A`
  + `lora_B`), PEFT (`base_model.model.model.…lora_A.weight`), ComfyUI, whose fused
  `qkv_proj` / `gate_up_proj` pairs stack the A matrices in `lora_down` and place the B blocks on
  the diagonal of `lora_up`, and Sound & Vision native exports (`sound-vision-yue2-native-export-v1`)
  that keep AR under `text_encoders.model` and NAR under `diffusion_model.model` with a shared `lora_A`
  and stacked `lora_B` on fused projections. `deltas` splits fused pairs back into per-projection
  weights (checked in `tests/test_loras.py`). Per-projection `alpha` scales by `alpha / rank`;
  full `vae2llm` / `llm2vae` replacement weights (as in the real-audio adapter) are applied too.
  Every delta is checked against the model's layer sizes (`dims_from_config`) before it is used.
  Training step and trigger word are read from native-export metadata when present. Download catalog
  notes in `lora-assets.json` overlay those fields for known files (including ComfyUI artist LoRAs that
  do not embed a trigger), and choosing one in the picker fills Style (trigger word), Instrumental,
  strength, Plan, style influence, Composition, Weirdness and duration from the card.
  Old School Hip-Hop uses No Plan and CFG 1.0 as published; the Mothersuperior instrumental adapter
  turns Instrumental on and Melody and Chords at strength 0.7 (1.0 is the author's recipe; 0.7 was
  found to break up repeating instrumental patterns). **Real-audio decoder v9** (2026-09-20) is the
  first sound-only card: `nar_lora_joint_v9` from the real-audio tokenizer repository (the companion
  of the v9 head that Cover/Remix continuation uses, trained from v6 on with an audio-domain loss against the
  original recordings). Its card sets only strength, so Plan, Style and style influence stay as set,
  and it is meant to be heard on ordinary songs as a fuller, more produced render.
- **Branches:** AR adapters (`self_attn`, `mlp`) change how the model *writes* the song — the
  planner and the music tokens; NAR adapters (`nar_self_attn`, `nar_mlp`, io) change how it
  *renders* the sound. The picker keeps a trigger chip next to the name; the line under it is
  strength, plan and CFG (training steps stay on Models cards).
- **Two slots** (2026-09-20): **Writes** (Style LoRA) takes every file that touches the
  AR branch — genre, artist and planner adapters, including ones with a sound half — and its card
  fills Controls; **Renders** (Sound LoRA) lists decoder-only files, so the two pickers
  never overlap and a writing adapter (say the Mothersuperior instrumental planner at 0.7) and a
  decoder adapter (the Real-audio decoder v9) play together, the way the ComfyUI workflows stack
  them on the CLIP and MODEL slots. On a **cover**, the Writes adapter is applied in full while the
  transcribed score is arranged, then only its decoder (NAR) half is kept for music tokens, so a
  style LoRA cannot collapse the cover into a repeating loop. A saved choice that sits in the wrong slot is named as such
  in the picker. The Models page groups the catalog the same way (`slot` in `lora-assets.json`,
  default `style`) and, since 2026-09-20, is a full page without the library: package and LoRA
  cards in a responsive grid, each LoRA card showing its recipe line (trigger, strength, plan,
  cfg, training steps). The request
  carries `sound_lora` / `sound_lora_strength` beside `lora` / `lora_strength`; saved prompts,
  drafts, re-renders and the library badges keep both. The same file cannot sit in both slots
  (the other picker greys it out; the server refuses it). `loras.combine` folds the two into one
  delta with each strength baked into its B columns — deltas on a shared projection are stacked
  along the rank axis so `B @ A` is their sum — and two files that both replace `vae2llm` /
  `llm2vae` are refused, since replacements cannot be summed. MLX applies the combined delta once
  (`adapters.apply`, label = both choices); CUDA folds it in at load and keys the pipeline cache
  on both.
- **MLX:** the resident model is wrapped once (`model/adapters.py`). On the BF16 model the deltas are
  *folded into the weights* (`W += strength · scale · B @ A`), which costs nothing per token; the next
  job without that LoRA reads the original tensors back from `model.safetensors` (verified bit-exact:
  a stock song after a LoRA song reproduces the earlier stock song's tokens and score). Quantized
  linears and the io replacements use switchable `Adapter` wrappers instead (`base(x) + scale · x Aᵀ Bᵀ`
  in float32). The wrapper path was measured first and halved the music-token rate with 196 active
  adapters (98 → 51 tok/s on an M5 Max); merged, the LoRA song runs at 91 tok/s. Applying the same
  LoRA at the same strength for consecutive songs is a no-op. Parsed deltas are cached with the model
  (`MODEL_CACHE['_loras']`). The real-audio decoder's own wrappers coexist (it measures the stock
  layer through `adapters.innermost`).
- **CUDA:** CUDA graphs capture the weights, so nothing is switched at run time: `cuda_engine.Pipeline`
  folds the deltas into the BF16 weights right after `from_pretrained`, before upstream's FP8
  quantization or the move to the GPU (`loras.merge_torch`). The pipeline cache key includes the
  LoRA and strength; another choice reloads the model.
- **First A/B (Mothersuperior's instrumental AR LoRA, rank 64, all 28 layers; same seed, prompt and
  2.5-minute cap, `cot=full`):** the LoRA wrote a denser score (733 instrumental notes over 215 bars
  against 525 over 137) and its render was flatter and brighter — 8.5 dB between the loud and quiet
  fifth-percentiles against 30 dB for the stock model, which followed the prompt's "dramatic build"
  from −34 dB to −8 dB; onsets 2.2/s against 3.1/s. Both hit the length cap, so the LoRA's claimed
  natural endings (3–5 minute songs) were not tested. Its card asks for bare lower-case section tags
  one per line; the Studio's instrumental mode keeps the tags as typed but separates them with blank
  lines and appends `Instrumental.` to the style, so the caption is close to, not identical with,
  its training format.
- **Honest limits:** a LoRA shifts the model's tendencies (instrumental writing, a production
  colour); it does not clone a singer or reproduce a specific recording. Files trained for other
  YuE2 sizes are refused by the shape check.
- **Improves:** community adapters — including ones released in ComfyUI's format — work without
  conversion or a different UI.

---

## 4. Cover / Remix

### 4.1 Recording upload, trim editor and cached analysis

- **What:** Upload WAV/MP3/FLAC/M4A/AIFF/OGG/AAC (≤200 MB, ≤30 min), see a waveform, drag start/end
  handles or type times, preview the selection, then **Analyze selection**. Titles are inherited from
  the file name; drop uploads are supported.
- **How:** `GET /api/uploads/{name}/waveform` returns ≤1000 peaks at 2 kHz; `POST …/trim` writes a new
  PCM WAV under a content hash and leaves the original untouched (`audio_edit.py`). Analysis is a
  `transcribe` job in the separate PyTorch environment: SheetSage2 + MERT-v2 transcribe the melody
  (and chords when *Preserve: Melody + chords*) into the two-voice ABC dialect, with beat and downbeat
  times. `analysis_cache.py` stores the result in `.analysis-cache/<sha256>.json`, keyed by
  `(audio_id, melody_only, source_seconds)`, so every later cover, app restart, or style borrowing reuses
  it. Isolated missing beats in the analysis are repaired.
- **Improves:** Analysis (the slow part) happens once per clip; the cover itself is a normal job.

### 4.2 One-step covers

- **What:** **Create cover** analyses (if needed) and generates in the same job; **Stop** works during
  either phase; progress shows both.
- **How:** The cover job receives `analysis.json` (with its source identity, verified against the
  request), the transcribed ABC replaces the model's own planning step, and `cot` follows the
  Preserve setting. Cached analysis is written into the job as `{'cached': True}`.
- **Improves:** No separate score step or manual copy of a transcription.

### 4.3 Arrange with the model (`cover_arrangement.py`)

- **Problem it solves:** SheetSage2 hears one lead line: the sung melody, and the instrumental lead in
  the gaps between verses. With *Preserve: Melody* it writes no chord symbols at all. YuE2's own plans
  carry a chord on nearly every bar, and the semantic stage renders what the score says, so a raw
  transcription came out as two or three thin instruments under an exposed, pitch-tracked voice.
- **What:** On by default under the recording. The model keeps everything the recording gave (sung
  notes, heard instrumental lines, section labels, meter and key changes, bar grid) and writes the
  rest in the new style: a chord symbol on every bar of the `Vocal` voice that has none, and its own
  instrumental line for sections where nothing melodic was heard. Generation then runs in the
  chord-annotated mode (`cot=full`). *Melody + chords* keeps the transcribed chords and fills only
  unannotated bars.
- **How:** The model writes its own score format token by token while the transcription is forced
  between its choices, so it harmonises exactly the notes that will be sung:
  - `model/generate.py` `ScoreWriter` (MLX) commits forced text and samples short proposals; the
    committed text is always re-encoded with the model's BPE (so a merged `|"` barline-and-quote
    token is fed as the model would have written it) and the KV cache is trimmed back to the last
    agreeing token, so a bar costs about five model steps. `cuda_engine.ScoreWriter` does the same
    on the upstream runtime by re-prefilling per proposal; it ends a proposal from the token
    callback as soon as `stop` is met and limits the first token through a wrapped
    `yue2.sampling.distribution`, so both engines hold the arrangement to the same rules.
  - After a barline the merged `|"` token is committed and the model names the chord; at a line start
    the first sampled token is constrained to `"A`…`"G` tokens. Both choices matter: asked *whether* to
    write a chord, a model that skipped the first bar never added any; forced to continue a lone `"`
    (a token it never saw) it repeated one odd chord for the whole song.
  - Full-bar `Z` rests in the `Vocal` voice become explicit rests (`z16`) so chords can sit on them,
    as in the model's own plans. Model-written `Ins` lines are validated as a tiny two-voice score
    (bar count, durations, no chords) and retried up to three times.
  - **In tune (`in_tune=True`).** Sampling alone drifts: the BPE spells `Ebm` as `E` + `bm`, so a
    stray closing quote leaves an `E` chord in an E-flat song; a major chord lands under a sung minor
    third; and once one wrong chord is in the score the model copies it for the rest of the song
    (one cover had 47 `E` chords in E-flat). Every proposed chord is therefore checked against the
    notes of its bar (what is sung, or where nothing is sung, the instrumental line heard in the
    recording) and the key: all its tones must lie in the key (either parallel mode, since the
    transcription's `K:` is trusted for its tonic, not its mode), it may not carry the opposite
    third to the one sung, it may not dwell on notes a semitone above its tones, and at least a
    third of the bar's sung time must fall on chord tones. A proposal that fails is replaced by its
    root respelled a semitone away (`E` → `Eb`), else by the previous chord if it still fits, else by
    the best-fitting triad in the key, so every bar keeps a chord and the model never sees a wrong
    one. Accepted chords are spelled for the key (`B#` → `C`), and a chord borrowed from outside the
    key's own scale is kept only when it fits the bar better than the best diatonic chord (`G#` under
    D#–G#–C# in E becomes `G#m`). A model-written `Ins` line must put at least half its notes on the
    chords of its bars and keep 90% of them in the key, or it is refused and retried.
  - The result must parse; the bar grid and the sung notes are verified identical to the
    transcription; any failure keeps the raw transcription. Instrumental covers arrange first, then
    silence the voice, so the chords remain under the moved melody.
- **Measured:** 3–8 s per song on Apple Silicon (109–176 bars); every bar received a chord on all ten
  library covers, with key-appropriate, style-aware progressions (a Gm ballad: verse
  `Gm Gm Cm Cm / F F Cm Dm`, chorus `Gm Gm Cm Cm / F Cm Bb Dm`; jazz-funk got `C#m7 F#m7 Amaj7 G#m7`).
  On a 176-bar E-flat cover with a modulating chorus, the checks corrected 61 proposals: bars whose
  chord carried the opposite third to the melody fell from 26% to 0%, and melody notes outside their
  chord from 55% to 32% (the model's own plans sit near 46%).
- **Improves:** Covers get the same harmonic scaffolding as new songs, so the arrangement fills out
  and the voice sits inside harmony instead of on top of a single line, and the added harmony and
  lines are in tune with the melody.

### 4.4 Continue this recording (Audio input package, optional 0.31 GB; MLX models)

- **What:** Keep an uploaded recording up to a chosen point and let the model continue it from your
  style and lyrics, as one song that contains the original.
- **UI:** The same timeline picker as Extend on a saved song (waveform from
  `GET /api/uploads/{id}/timeline`, draggable cursor, play from here, the same fields); recordings
  have no plan, so the cursor is not snapped to bars.
- **How:** YuE2 never shipped an audio-to-token encoder, so `audio_tokens.py` uses the community
  `yue2-mothersuperior-realaudio-tokenizer` v9 head: MERT-v2 layer-20 features at 25 fps, normalised
  per recording, mapped by a small transformer over 20 s windows to YuE2's 32,768 codes. Tokenization
  is a background `tokenize` job (`POST /api/tokens`), cached under `.analysis-cache/`. The tokens are
  cut by time and used as the codec prompt of a score-free (`cot=off`) continuation, exactly like
  Extend on a saved song (`continue_recording` on `POST /api/jobs`,
  `song_continuation.recording_continuation`). Because the head's tokens are approximate, the stock
  flow-matching decoder renders them muddy; the package's second file, `nar_lora_joint_v9.safetensors`, is the
  companion **NAR adapter** (rank-32 LoRA on `nar_self_attn.{q,k,v,o}` and `nar_mlp.{gate,up,down}`
  of all 28 layers plus replacement `vae2llm`/`llm2vae`) trained jointly with the head.
  `model/real_audio.py` reads that safetensors file without torch and wraps the
  MLX model's NAR linears with switchable `LoRALinear`/`SwitchLinear` adapters that are on only during
  synthesis of songs whose `continue-input.json`/`render-input.json` carry `real_audio: true`.
  Ordinary songs are untouched. Attaching takes well under a second and adds ~140 MB.
- **Limits:** MLX only for now (see `TODO-CUDA.md`); no plan, so Instrumental only is unavailable;
  the kept part is a re-synthesis from tokens, close to but not identical with the original audio.
- **Improves:** Real recordings become material the model can build on, not only a style hint.

### 4.5 Uploaded recordings in the library

- **What:** Like Suno, every recording uploaded under Cover / Remix is also kept in the song list,
  badged **UPLOADED** where other songs show their model, in the workspace chosen under *Save to*.
- **How:** After `POST /api/upload`, the page calls `POST /api/uploads/{id}/song`
  (`uploaded_songs.py`), which writes a complete one-version job folder with `kind: upload`:
  `request.json`, `state.json`, `result.json` and `audio.wav` (the recording decoded with ffmpeg to
  the studio's 48 kHz stereo PCM). The original stays in `uploads/` and is referenced by `audio_id`,
  so playback, WAV/MP3 downloads, favorites, rename, workspaces and delete work unchanged. The row's
  menu offers the recording actions instead of the generation ones: *Use as cover source* and
  *Continue this recording* (4.4) on the whole original recording. A library filter **Uploaded** lists
  them alone.
- **Limits:** No album art or generation timings; deleting the entry moves it to trash but keeps the
  file in `uploads/`. Uploading the same file twice makes two entries.
- **Improves:** Recordings are first-class library items you can play, organise and build on later,
  not only a transient source in the Cover panel.

### 4.6 Not offered: making a song sound like another recording

YuE2 has no timbre or reference-audio input. Priming a new song with an excerpt's music tokens
(tried twice, as a *Keep the original's sound* switch and as *personas*) shifts the timeline and makes
the model sing the excerpt's words over the new lyrics; describing the recording in words (CLAP tags,
counted score traits) moved the result less than a well-written style line. What does work: a cover
keeps the melody and chords (4.2–4.3), *Continue this recording* keeps the audio itself up to a point
(4.4), and the style text, Voice, Vocal Range and a LoRA shape the rest.

---

## 5. Working with finished songs

### 5.1 Re-render from saved tokens

- **What:** Render a finished version again at different audio steps without regenerating music tokens.
  The dialog also offers a **Sound LoRA** picker with its strength (2026-09-20): the decoder adapter
  only touches the rendering, so the same tokens and seed can be rendered once with **None** and once
  with e.g. *Real-audio decoder v9*, which is the fair A/B test for a sound adapter. The Style LoRA,
  seed and tokens always come from the source song; the copy's title notes the steps and the adapter.
- **How:** The job carries `render-input.json` with the saved `codec`, `abc` and `prefix`; the worker
  runs only synthesis and VAE (`render_source` path on both engines). `server.py` rebuilds the request
  from the source and takes only `steps`, `sound_lora` and `sound_lora_strength` from the dialog.
- **Improves:** Cheap quality passes; the musical content is preserved exactly.

### 5.2 New arrangement / Edit (`composition_edit.py`)

- **What:** From a finished song: change style (Jazz / Orchestral / Electronic presets or your own),
  tempo (30–300 BPM), harmony (**Keep chords**, **New chords, same melody**, **Jazz seventh chords**),
  and song structure (move, repeat, remove or shorten sections; shorten keeps the first portion at a
  bar boundary). Then **Use edited composition** and **Create**.
- **How:** Edits are applied to that version's saved `score.abc` and validated with the vendored
  two-voice ABC checker (`vendor/yue2_abc.py`): unsupported notation and broken voice timing are
  rejected rather than silently regenerated; when only arrangement, harmony or tempo change, the
  melody is checked unchanged. *New chords* strips chords and plans with `cot=melody`; *Jazz sevenths*
  extends simple chord symbols and requires existing chords. Drafts live in `.song-edits/<id>.json`
  and are referenced by `edit_id`. Maximum 12 minutes. Structure changes on vocal songs need section
  lyrics assigned first.
- **Improves:** Musical edits at the score level, where the model actually follows them.

### 5.3 Extend and Regenerate from here (`song_continuation.py`, `web/song-timeline.js`)

- **What:** Open a song's timeline, drop the cursor at a bar, then either **Extend** (keep everything
  and continue for N seconds, default 30) or **Regenerate from here** (keep the bars before the cut,
  write new music after it). Bars snap (Alt-drag to disable); playback starts at the cut.
- **How:** Every song keeps its music tokens and plan. The semantic stage is a language model whose
  prompt ends with the plan and `MUSIC_START`; appending the saved tokens continues that song. The
  plan is cut at the same bar so the model first continues the score (`continue_plan`) and then the
  music (`continue_semantic`). Nominal bar times come from the score tempo and are scaled to the real
  audio length when they disagree by at most 25 %. Tokens are cut at 25 fps. The context (24,576
  tokens) must leave at least 200 tokens (~8 s), otherwise the studio asks for an earlier cut.
  Available on both engines (`mlx_continuation.py`, `cuda_engine.py`). Job input:
  `continue-input.json`; timeline: `GET /api/jobs/{j}/songs/{i}/timeline`.
- **Improves:** Songs can be lengthened or partially rewritten instead of regenerated from scratch.

---

## 6. Score and ABC tooling

- **`vendor/yue2_abc.py`:** An original, standard-library parser for the limited two-voice ABC
  dialect shared by YuE2 and SheetSage2. It fails closed on unsupported notation and resolves sounding
  notes, bar grids, chords and key changes as exact fractions. Instrumental mode, composition editing,
  cover arrangement and the tests all verify their output with it.
- **`transcriber/`:** Standalone SheetSage2 inference (pinned revision) with MERT-v2, whole-song
  windowed decoding with cached overlap prefixes, validated ABC export, LAB/MIDI outputs, optional
  offline rendering. `melody_only=True` omits chords for covers.

---

## 7. Windows / NVIDIA runtime (`cuda_engine.py`)

- **What:** **PyTorch CUDA BF16** (Windows default) and experimental **CUDA FP8** run the official
  YuE2-3B checkpoint through the unmodified upstream `yue2-infer` package in an isolated
  `.cuda-venv` (PyTorch 2.10 / CUDA 12.8). MLX BF16 remains available for comparison.
- **How:** `cuda_engine.py` is a small adapter, not a fork: it maps studio requests to upstream
  `SongRequest`/`Sampling`, reports progress, supports cancellation, re-render, Extend and
  cover arrangement, reserves 2 GiB VRAM, offloads AR layers during synthesis and decodes audio in
  512-frame tiles. Windows selects cuDNN attention for CUDA graphs because its PyTorch wheel exposes a
  FlashAttention operator without the kernel; unsupported shapes use 256-query chunks so SDPA never
  silently falls back to a huge math-attention buffer. MLX on Windows uses a large-file reader
  workaround (`model/weight_io.py`). One resident backend at a time on Windows.
- **Measured:** RTX 4070 Ti SUPER 16 GB, 40 s BF16 with full planning: 39.75 s, 95 tokens/s, 8.31 GiB
  peak. FP8 was slower in a short test.
- **Improves:** The same studio, library and API on the two common GPU platforms. Seeds are repeatable
  within a backend, not across backends.

---

## 8. Data and privacy

- The server binds to `127.0.0.1` only. Recordings stay in `uploads/`, results in `library/`,
  analysis in `.analysis-cache/`, edits in `.song-edits/`, deleted
  songs in `.trash/`, organisation in `.workspaces.json`. None of these, nor model weights, are in Git.
- The only network calls after installation are: model downloads (Hugging Face, on request), LRCLIB
  lyric search (title/artist only), and OpenAI for style suggestions and album art when a key is set.
  Recordings are never uploaded anywhere.

---

## 9. Testing

`.venv/bin/python -m unittest discover -s tests -v` runs the application tests. They use tiny
local HTTP fixtures and fake models to cover download integrity and resume, first-run behaviour,
worker reuse and cancellation, page lifetime, analysis caching, instrumental rewriting, composition
edits, continuation budgets, workspaces, the style AI, LoRA parsing and application, the CUDA
adapter's prompt assembly and score writer (upstream stubbed), the MLX token-generation batching,
the cover arrangement (scripted writers, KV-cache trimming, first-token constraints) and
embedded-tag reading. `node tests/*.cjs` covers the browser modules (style presets and blending,
lyrics tag normalisation and duration warnings, model availability). Real-model checks used during
development are kept out of the repository.

---

## 10. Chronology

| Date | Change |
| --- | --- |
| 2026-09-12 | Studio with resumable, verified model setup |
| 2026-09-12 | Cached cover analysis, resident workers, preloading |
| 2026-09-12 | Recording trim editor, titles from file names |
| 2026-09-12 | WAV/MP3 downloads, duration and time badges, generate while other models download |
| 2026-09-12 | Manual seed entry, song text copying, collapsible sidebar |
| 2026-09-12 | Song editing (New arrangement / Edit), workspaces, sampling tooltips |
| 2026-09-12 | Build-a-style helper (voice, instruments, mood) |
| 2026-09-12 | Dropdown polish; song menus stay open during generation |
| 2026-09-12 | Re-render from saved tokens, macOS app launcher |
| 2026-09-13 | Repair missing analysis beats; defaults 8 steps / CFG 1.2 |
| 2026-09-13 | Windows CUDA runtime, attention memory work |
| 2026-09-13 | LRCLIB lyrics search, drop uploads |
| 2026-09-13 | Compact studio chrome, lyrics search prefilled from title |
| 2026-09-13 | Tidied cover upload and generation options; 2-step audio synthesis default |
| 2026-09-14 | Sliced output head (+17 % tokens/s), host-sync batching, analysis decoder on CPU |
| 2026-09-14 | Job queue, voice selection, friendly sliders, AI style modal, WebP album art |
| 2026-09-14 | Match cover lyrics to the score and to heard words (Sung words pack) |
| 2026-09-14 (working tree) | Extend / Regenerate from here, hosted mode, native instrumental mode, **Arrange with the model** for covers |
| 2026-09-15 | Plus button inserts `[Verse]` / `[Chorus]` / other section tags at the lyrics caret |
| 2026-09-15 | Lyric matching and the Sung words pack removed: covers send the lyrics as written |
| 2026-09-15 | 73 style presets with blending and modifiers, lyrics tag check with Fix tags and duration warning, title and lyrics from uploaded files' tags |
| 2026-09-17 | Vocal Range on covers; **LoRA picker** for community adapters in HF / PEFT / ComfyUI layouts (3.5); **Composition** slider for planner sampling (2.2) |
| 2026-09-17 | Cleanup: personas, stem splitting, Style reading, Add its style, the raw sampling fields, the decode panel, OpenAI cost tracking and the extra macOS installer scripts removed; CUDA cover writer honours stop/first (4.3) |
| 2026-09-20 | LoRAs heading has Download all for every missing adapter; **Sound LoRA** second slot and the Real-audio decoder v9 card (3.5) |
| 2026-09-21 | DreamPop v2 removed from the LoRA download catalog |
| 2026-09-24 | Audio input uses the Mothersuperior v9 tokenizer head and matching NAR adapter |
