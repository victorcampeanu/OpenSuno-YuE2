For another Mac, build `OpenSuno.dmg` with **scripts/Build OpenSuno Installer.command**; on that Mac, open **Install OpenSuno** in the image and tick Studio, Render node or both. Everything else is in [README.md](../README.md). Model weights are downloaded from the page and are not included in Git.

# OpenSuno

Open **http://127.0.0.1:7862** while the local server is running.

To start it later, double-click **scripts/Launch Studio.command**. It runs the server in the background and opens the browser; `launchctl bootout gui/$(id -u)/local.opensuno.studio` stops it. You can also run `./.venv/bin/python app/server.py` from the project folder.

## Create music

1. Select **YuE2 BF16** (the default). It becomes available after its download completes in **Models**.
2. Enter a title, lyrics and a style. **Add Tag** in Lyrics inserts `[Verse]`, `[Chorus]` and other section tags at the cursor; **Build a style** and the sparkles button (Ask AI, needs an OpenAI key in Settings) help write the style line.
3. For music without singing, enable **Instrumental**. Section tags in the lyrics still shape the song; any words between them are ignored.
4. Open **More Options** for the plan mode, voice, LoRA, number of versions, duration cap, seed and the sliders. Create.
5. Watch the current stage and elapsed time under the Create button. Finished versions play while the rest generate; if a run fails, `run.log` in the song's library folder has the details.

Each new generation chooses a fresh random seed unless **Lock** is on. **Use these settings** on a saved version restores its exact settings and seed. Candidates run with seeds `base seed + candidate number - 1`.

The duration setting caps music tokens at 25 per second. It can cut a song abruptly; it does not force an exact length. The default cap is six minutes.

## Controls

- **Plan:** *Melody and Chords* (default) makes the model write a chord-annotated score before the music; *Melody Only* writes the melody line only; *No Plan* skips the score and is the least predictable.
- **Voice:** Any / Male / Female / Duet, added to the style for the model. Ignored for instrumentals.
- **LoRA:** a community adapter from the `loras/` folder, with a strength from 0 to 2; applies to this song only.
- **Composition:** how adventurous the score writing is (chords, melodic turns, busyness of the instrumental line). 50% is the model default. Inactive with *No Plan*, and for covers that keep the recording's score as heard.
- **Weirdness:** how random the music tokens are (temperature, top-p, top-k together). 50% is the model default.
- **Style Influence:** how hard the style text and lyrics are pushed (classifier-free guidance, 1.0 → 1.2 → 2.0). Raise it if the genre, instruments or voice get ignored.
- **Repetition:** how much the music may repeat what it just played. *More allowed* for steady grooves and held notes; *Less* if a song loops.
- **Audio Steps:** passes of the audio decoder; default 2. More sounds cleaner and takes longer. It does not change the music itself, so a song you like can be re-rendered later with more steps from its row menu.
- **Restore Defaults** puts every control back.

## Cover / Remix an uploaded recording

1. Choose **Cover / Remix** and upload WAV, MP3, FLAC, M4A, AIFF, OGG or AAC audio (up to 200 MB / 30 minutes). A trim editor opens; keep the whole recording or a section, then **Analyze selection**.
2. Choose **Preserve**: melody, or melody and chords.
3. Enter lyrics with section tags that follow the recording (the **Form** card lists its sections), or enable **Instrumental**; choose the new style and the number of versions; **Create**.

**Arrange with the Model** (on by default) keeps the melody and the instrumental lines heard in the recording and lets the model add chords on every bar and write its own lines where nothing was heard, in the new style. Turn it off to sing the transcription exactly as heard.

**Vocal Range** moves the transcribed melody by octaves. *Match the Voice* (default) puts it where the chosen Voice would sing it, which fixes the common case of a male recording transcribed an octave too high and then sung by a female voice; *As Heard* keeps it; the others shift it by one or two octaves.

A cover is a new performance of the transcribed melody. It does not preserve the original singer's voice, does not mix the original audio, and lyrics are not transcribed. Analysis of a selection is cached and reused across generations and restarts.

**Continue this recording** (needs the optional **Audio input** download in Models, plus Cover analysis; MLX models) keeps the recording up to a point you choose and lets the model write what follows from your style and lyrics. The kept part is rendered again from its tokens, so it comes out close to the original rather than identical.

Every uploaded recording is also kept in your song list, badged **UPLOADED**. Its menu offers **Use as cover source** and **Continue this recording**; filter the library by **Uploaded** to see only recordings.

## Downloads and local files

Each song's menu offers **Download WAV** and **Download MP3**. Technical files stay in the song's folder.

- `library/`: results, settings, progress, favorites and job logs.
- `uploads/`: imported recordings.
- `loras/`: community LoRA files (see `loras/README.md`).
- `.venv/`: isolated MLX generator and browser server environment.
- `.transcribe-venv/`: separate SheetSage2/PyTorch environment.
- `model/`, `transcriber/`, `mert/`, `tokens/`: locally installed model files.

The server listens only on this Mac (127.0.0.1). After installation, generation and transcription use local files. Form edits survive browser reloads.

## Sources and versions

Original model: https://huggingface.co/m-a-p/YuE2-3B

Community MLX port: https://huggingface.co/ahmadw/YuE2-3B-MLX · revision `fe0a9050fd658257b486b880422d8872ee1f81e3`

SheetSage2: https://huggingface.co/m-a-p/SheetSage2 · revision `eab522a8168e8b8b8c4856bf8609cd86198f01fe`

MERT-v2 parent: https://huggingface.co/m-a-p/MERT-v2-FullSong · revision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`

Audio-to-token head and real-audio decoder adapter (Audio input, v9 pair): https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 · revision `e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5`

Model weights use CC BY-NC 4.0 (non-commercial). Package versions are recorded in the requirements files.
