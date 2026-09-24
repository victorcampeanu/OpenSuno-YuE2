# LoRA adapters

Drop community YuE2 LoRA files (`.safetensors`) into this folder. They appear in **More Options → LoRA** the next
time the picker is opened; no restart needed. When a render node does the rendering, the files go into the
`loras/` folder of the node instead.

Accepted layouts:

- Hugging Face / upstream YuE2 (`layers.N.self_attn.q_proj.lora_A` / `lora_B`)
- PEFT (`base_model.model.model.layers.N…lora_A.weight`)
- ComfyUI (`…layers.N.self_attn.qkv_proj.lora_down.weight` / `lora_up.weight`, fused q/k/v and gate/up)
- Sound & Vision native (`sound-vision-yue2-native-export-v1`: AR under `text_encoders.model`, NAR under
  `diffusion_model.model`, fused `qkv_proj` / `gate_up_proj` with one `lora_A` and stacked `lora_B`).
  Artist packs that keep that packing under ComfyUI `lora_down` / `lora_up` names (CNZN, MLTNT, CHNSN, QWWL)
  are read the same way.

YuE2 instrumental (Mothersuperior v3), Old School Hip-Hop and the becausereasons artist packs
(MLTNT, CHNSN, QWWL / DRKSF, CNZN) can be downloaded from **Models → LoRAs**; they land in this folder
and appear in the picker. Each documented adapter lists its trigger word and recommended Controls
(strength, plan, style influence). Choosing one fills those in. Both branches are understood: AR adapters (`self_attn`, `mlp`) change how the model writes the song,
NAR adapters (`nar_self_attn`, `nar_mlp`, `vae2llm`, `llm2vae`) change how it renders the sound. `.pt` / `.pth`
checkpoints are not read; convert them to safetensors first.
