# <img src="assets/branding/vimoo-logo.svg" alt="Vimoo Logo" width="32" height="32" align="texttop"> ComfyUI-Vimoo

[![Website](https://img.shields.io/badge/Website-Vimoo-181717?logo=google-chrome)](https://vimoo.ai)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Vimoo--AI-orange?logo=huggingface)](https://huggingface.co/Vimoo-AI)
[![Vimoo-1.0](https://img.shields.io/badge/Vimoo--1.0-Repo-blue?logo=github)](https://github.com/Vimoo-AI/Vimoo-1.0)
[![Vimoo-1.0-MLX](https://img.shields.io/badge/Vimoo--1.0--MLX-Repo-blue?logo=github)](https://github.com/Vimoo-AI/Vimoo-1.0-MLX)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/TBD)

ComfyUI-Vimoo provides Vimoo 1.0 nodes and workflow templates for text-to-video (T2V) and image-to-video (I2V) generation in ComfyUI. Vimoo 1.0 is an open-source video generation model family built on a dual-stream MMDiT. Its lightweight variant, Vimoo 1.0 - Lite, handles both tasks in a single model for faster inference.

See the [Vimoo 1.0 repository](https://github.com/Vimoo-AI/Vimoo-1.0) for the model family, online API, system overview, architecture and evaluation.

## 🔥 News

- **[2026-10]** Release Vimoo 1.0: T2V / I2V models, Diffusers inference examples and ComfyUI workflows.

## 🎬 Showcase

https://github.com/user-attachments/assets/91f0aa20-5f6a-43d7-b76e-a7fdedb6f7ec

## 🧭 Choose a Model and Workflow

| Model | Task | Adapter | Workflow |
| --- | --- | --- | --- |
| Vimoo-1.0 T2V | T2V | None | [vimoo_v1_t2v.json](example_workflows/vimoo_v1_t2v.json) |
| Vimoo-1.0 I2V | I2V | None | [vimoo_v1_i2v.json](example_workflows/vimoo_v1_i2v.json) |
| Vimoo-1.0 T2V Distilled | T2V | T2V distilled LoRA | [vimoo_v1_t2v_distilled.json](example_workflows/vimoo_v1_t2v_distilled.json) |
| Vimoo-1.0 I2V Distilled | I2V | I2V distilled LoRA | [vimoo_v1_i2v_distilled.json](example_workflows/vimoo_v1_i2v_distilled.json) |
| Vimoo-1.0-Lite TI2V | T2V / I2V | None | [vimoo_v1_ti2v_lite.json](example_workflows/vimoo_v1_ti2v_lite.json) |
| Vimoo-1.0-Lite TI2V Distilled | T2V / I2V | Lite TI2V distilled LoRA | [vimoo_v1_ti2v_lite_distilled.json](example_workflows/vimoo_v1_ti2v_lite_distilled.json) |

Vimoo-1.0 T2V and I2V offer the highest quality, Vimoo-1.0-Lite TI2V runs faster with one checkpoint for both tasks, and the Distilled models use matching adapters. After installation, the workflows also appear in ComfyUI's template browser with looping square previews. Restart ComfyUI and refresh the page after updating the nodes.

## 🚀 Quick Start

### Requirements

- **ComfyUI:** 0.34.0 or later.
- **Python:** 3.10 or later; install dependencies in the environment that launches ComfyUI.
- **PyTorch:** 2.9.0 or later, using a CUDA-enabled build compatible with your GPU and driver. Choose the build with the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
- **Hardware:** an NVIDIA CUDA GPU.
- **Attention:** all supplied workflows default to PyTorch SDPA (`attention=pytorch`), with no additional attention package. Install FA2, FA3, FA4 or SageAttention only when selecting it in **Vimoo Performance Options**; see [performance options](docs/performance.md).

Use separate environments for ComfyUI and the Diffusers examples; their Transformers dependencies differ. ComfyUI manages its Torch installation. Vimoo-1.0-Lite TI2V templates use the `im2col_bounded` VAE backend with PyTorch **2.9.0 or later**; see [VAE decoding](docs/performance.md#vae-decoding) for its workspace settings.

### Install the nodes

Install [ComfyUI](https://github.com/Comfy-Org/ComfyUI), then run these commands from its root directory:

```bash
git clone https://github.com/Vimoo-AI/ComfyUI-Vimoo custom_nodes/ComfyUI-Vimoo
python -m pip install -r custom_nodes/ComfyUI-Vimoo/requirements.txt
```

Restart ComfyUI and refresh the browser to load the Vimoo nodes and templates.

### Download weights

Run these commands from the ComfyUI root directory. Install the Hugging Face CLI with `python -m pip install -U huggingface_hub` if `hf` is not available. Vimoo-1.0 and its Distilled models are in [Vimoo-1.0-ComfyUI](https://huggingface.co/Vimoo-AI/Vimoo-1.0-ComfyUI); Vimoo-1.0-Lite models are in [Vimoo-1.0-Lite-ComfyUI](https://huggingface.co/Vimoo-AI/Vimoo-1.0-Lite-ComfyUI). Every workflow needs the shared text encoders and VAE; I2V workflows also use the SigLIP vision encoder. Both repositories include these shared files; if you use only Vimoo-1.0-Lite models, download them from `Vimoo-AI/Vimoo-1.0-Lite-ComfyUI` instead:

```bash
hf download Vimoo-AI/Vimoo-1.0-ComfyUI \
  text_encoders/vimoo_v1_qwen2_5_vl_7b_bf16.safetensors \
  text_encoders/vimoo_v1_glyph_byt5_fp32.safetensors \
  clip_vision/vimoo_v1_siglip_so400m_fp16.safetensors \
  vae/vimoo_v1_wan22_ti2v_vae_fp32.safetensors \
  --local-dir models
```

Then download the files for your model from its repository in the same way:

| Model | Repository | Files |
|---|---|---|
| T2V | Vimoo-1.0-ComfyUI | `diffusion_models/vimoo_v1_t2v_bf16.safetensors`<br>`diffusion_models/vimoo_v1_t2v_bf16.safetensors.vimoo.json` |
| I2V | Vimoo-1.0-ComfyUI | `diffusion_models/vimoo_v1_i2v_bf16.safetensors`<br>`diffusion_models/vimoo_v1_i2v_bf16.safetensors.vimoo.json` |
| T2V Distilled | Vimoo-1.0-ComfyUI | `diffusion_models/vimoo_v1_t2v_distilled_bf16.safetensors`<br>`diffusion_models/vimoo_v1_t2v_distilled_bf16.safetensors.vimoo.json`<br>`loras/vimoo_v1_t2v_distilled_lora_bf16.safetensors`<br>`loras/vimoo_v1_t2v_distilled_lora_bf16.safetensors.json` |
| I2V Distilled | Vimoo-1.0-ComfyUI | `diffusion_models/vimoo_v1_i2v_distilled_bf16.safetensors`<br>`diffusion_models/vimoo_v1_i2v_distilled_bf16.safetensors.vimoo.json`<br>`loras/vimoo_v1_i2v_distilled_lora_bf16.safetensors`<br>`loras/vimoo_v1_i2v_distilled_lora_bf16.safetensors.json` |
| Lite TI2V | Vimoo-1.0-Lite-ComfyUI | `diffusion_models/vimoo_v1_lite_bf16.safetensors`<br>`diffusion_models/vimoo_v1_lite_bf16.safetensors.vimoo.json` |
| Lite TI2V Distilled | Vimoo-1.0-Lite-ComfyUI | The two Lite TI2V files, plus:<br>`loras/vimoo_v1_ti2v_distilled_lora_fp32.safetensors`<br>`loras/vimoo_v1_ti2v_distilled_lora_fp32.safetensors.json` |

For example, for Vimoo-1.0-Lite TI2V:

```bash
hf download Vimoo-AI/Vimoo-1.0-Lite-ComfyUI \
  diffusion_models/vimoo_v1_lite_bf16.safetensors \
  diffusion_models/vimoo_v1_lite_bf16.safetensors.vimoo.json \
  --local-dir models
```

The downloaded files use this layout under `ComfyUI/models/`:

```text
ComfyUI/models/
├── diffusion_models/
│   ├── vimoo_v1_t2v_bf16.safetensors
│   ├── vimoo_v1_t2v_bf16.safetensors.vimoo.json
│   ├── vimoo_v1_i2v_bf16.safetensors
│   ├── vimoo_v1_i2v_bf16.safetensors.vimoo.json
│   ├── vimoo_v1_lite_bf16.safetensors
│   ├── vimoo_v1_lite_bf16.safetensors.vimoo.json
│   ├── vimoo_v1_t2v_distilled_bf16.safetensors
│   ├── vimoo_v1_t2v_distilled_bf16.safetensors.vimoo.json
│   ├── vimoo_v1_i2v_distilled_bf16.safetensors
│   └── vimoo_v1_i2v_distilled_bf16.safetensors.vimoo.json
├── text_encoders/
│   ├── vimoo_v1_qwen2_5_vl_7b_bf16.safetensors
│   └── vimoo_v1_glyph_byt5_fp32.safetensors
├── clip_vision/
│   └── vimoo_v1_siglip_so400m_fp16.safetensors
├── vae/
│   └── vimoo_v1_wan22_ti2v_vae_fp32.safetensors
└── loras/
    ├── vimoo_v1_t2v_distilled_lora_bf16.safetensors
    ├── vimoo_v1_t2v_distilled_lora_bf16.safetensors.json
    ├── vimoo_v1_i2v_distilled_lora_bf16.safetensors
    ├── vimoo_v1_i2v_distilled_lora_bf16.safetensors.json
    ├── vimoo_v1_ti2v_distilled_lora_fp32.safetensors
    └── vimoo_v1_ti2v_distilled_lora_fp32.safetensors.json
```

Keep each `.vimoo.json` and `.json` file next to its weights. The nodes read the supported tasks, the matching base checkpoint and the sampling settings from them.

### Generate video

1. Open the workflow for your model from ComfyUI's template browser, or drag its JSON from `example_workflows/` onto the canvas.
2. Select the checkpoint in **Vimoo Model Loader**. For a distilled workflow, check its adapter inside the generation subgraph. Keep the saved node connections and defaults.
3. Edit the prompt. For I2V, upload a reference image in the input node for Vimoo-1.0-Lite workflows or **Vimoo Load Image** for Vimoo-1.0 I2V; it is scaled and center-cropped to the output size.
4. Click **Queue**. Videos are saved to `ComfyUI/output/`.

Vimoo-1.0-Lite workflows select `t2v` or `i2v` in the input node. Vimoo-1.0 T2V and I2V use separate templates and checkpoints. Vimoo-1.0-Lite TI2V shares its checkpoint across both tasks, and Vimoo-1.0-Lite TI2V Distilled also shares its adapter. Vimoo-1.0-Lite TI2V Distilled defaults to T2V; switch to I2V to use its saved bread image with the same prompt. To use the saved I2V examples, copy the three PNG files from `assets/` to `ComfyUI/input/`, keeping their filenames. Vimoo-1.0 I2V uses [the barn owl](assets/vimoo_v1_i2v.png), Vimoo-1.0 I2V Distilled uses [the train passenger](assets/vimoo_v1_i2v_distilled.png), and Vimoo-1.0-Lite TI2V Distilled uses [the sourdough loaf](assets/vimoo_v1_ti2v_lite_distilled.png). Each reference is a frame from a Vimoo video and matches the first frame of its template preview.

## 🎛️ Settings and Performance

### Workflow defaults

The templates provide model loading, performance, generation and video output nodes; I2V adds an image input. Expand the generation subgraph to select supporting encoder, VAE and adapter files. Ordinary models expose steps and text CFG; Distilled models keep their fixed sampling settings.

`resolution=auto` fits the oriented I2V image ratio to a 1280 × 720 pixel budget, rounding both axes down to multiples of 16. A square image produces 960 × 960; a portrait 9:16 image produces 720 × 1280. Choose `manual` to enter width and height. T2V defaults to manual 1280 × 720; Vimoo-1.0-Lite uses 1280 × 720 when auto is selected in T2V mode. The initial frame count is 121 at 24 fps.

Vimoo-1.0-Lite TI2V and Vimoo-1.0-Lite TI2V Distilled require BF16 compute; select `dtype=bf16` in **Vimoo Model Loader**. FP16 and FP32 compute overrides are rejected for these profiles.

The Distilled workflows use CFG 1 and only the positive prompt; keep their saved sampling grid and matching adapter metadata. Keep the generation FPS and the video output FPS equal.

### Memory and speed

- **Offload:** Start with `--enable-dynamic-vram --async-offload 2`. Vimoo-1.0 T2V / I2V Distilled in BF16 on 16 GB VRAM should start with `--reserve-vram 3` and `batch=1`; keep the supplied performance defaults. See [memory settings](docs/performance.md#16-gb-vram) for cache headroom and optional acceleration.
- **VAE decoding:** Inside the generation subgraph, enable `tiled_decode` in **Vimoo Output VAE** to use the model’s release tiles. With the switch off, decoding tries the whole video and falls back to automatic tiles when needed. Reference encoding has its own loader.
- **Attention and compilation:** Choose them in **Vimoo Performance Options**. Optional backends such as SageAttention need their own dependencies; see [performance options](docs/performance.md).
- **Prompt testing:** For Vimoo-1.0 T2V / I2V and Vimoo-1.0-Lite TI2V, smaller sizes and fewer frames are faster for trying prompts. Width and height must be multiples of 16 and frame counts must be `4n + 1`.

Detailed node settings are in the [workflow guide](example_workflows/README.md).

## 🩺 Troubleshooting

| Issue | Likely cause | Recommended resolution |
|---|---|---|
| Vimoo custom nodes do not appear | Dependencies missing or installed in the wrong environment | Install `requirements.txt` in the Python environment that launches ComfyUI, then restart ComfyUI and refresh the browser. |
| Model dropdown menus are empty | Files are outside the designated directories | Check the matching files under `ComfyUI/models/`, including `diffusion_models/`, and refresh the model list. |
| A checkpoint or adapter is rejected | Base, adapter or metadata mismatch | Use the checkpoint and adapter listed for the workflow, and keep their adjacent JSON files. |
| CUDA out of memory during sampling | DiT temporary allocations or resident weights exceed VRAM | Update the plugin and enable Dynamic VRAM / async offload with `batch=1`. VAE tiling does not affect sampling memory. |
| CUDA out of memory during decoding | VAE tiles or workspace exceed VRAM | Enable `tiled_decode` on the output VAE inside the generation subgraph. |
| Workflow loading fails | Nodes and workflow come from different versions | Update ComfyUI-Vimoo and open the workflow JSON from the same version. |

For other problems, open an [issue](https://github.com/Vimoo-AI/ComfyUI-Vimoo/issues) with your workflow, software versions, GPU model and the full traceback.

## 📝 Prompting Guide

**Use English prompts.** The current open-source models have been validated only with English prompts. Generation quality and prompt adherence in other languages have not been validated. We recommend translating prompts into English before generation.

**Write clear, detailed prompts.** Specific descriptions of the subject, scene, action sequence, and camera movement can help produce results that better match your intent than short, vague prompts. See the [Prompt Guide](https://vimoo.ai/vimoo-docs/api#prompts) for structure and examples, and focus on relevant, consistent visual details.

The Prompt Guide describes automatic prompt enhancement in the hosted service. Local examples use your input prompt directly; follow the guide's writing structure and examples when preparing prompts.

## 🧪 Local Checks

Use the Python environment that runs ComfyUI:

```bash
PYTHONPATH=/path/to/ComfyUI python -m unittest discover -s tests -q
ruff check .
ruff format --check .
```

## 📜 License

Vimoo-authored plugin code in this repository, Vimoo model weights, and official adapters are provided under the [Apache License 2.0](LICENSE).

- Commercial use, modification, and redistribution are permitted under the terms of the license.
- Redistribution must include a copy of the license, retain applicable copyright and attribution notices, and identify modified files.
- Attribution and third-party component notices are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Third-party components remain subject to their own licenses.
