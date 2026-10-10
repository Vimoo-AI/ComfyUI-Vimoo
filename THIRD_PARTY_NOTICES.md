# Third-party components

ComfyUI-Vimoo uses the following upstream software and model components. Their copyright notices and license terms continue to apply.

| Component | Source |
|---|---|
| ComfyUI | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI), [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE); the plugin integrates with ComfyUI's model, tokenizer and attention modules and does not redistribute ComfyUI |
| Qwen2.5-VL text tower and tokenizer | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), text tower and tokenizer |
| Glyph-ByT5 and tokenizer | [nlpcvcode/Glyph-SDXL-v2](https://huggingface.co/nlpcvcode/Glyph-SDXL-v2), fine-tuned encoder weights and font/color vocabulary; base config and tokenizer from [google/byt5-small](https://huggingface.co/google/byt5-small) |
| VAE | [Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), `vae` subfolder |
| SigLIP vision encoder | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384), vision tower and image processor |
| SageAttention workaround | The optional fake-handler fix in `docs/performance.md` comes from [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/commit/d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4), [Apache-2.0](https://github.com/thu-ml/SageAttention/blob/main/LICENSE) |
| SageAttention SM103 build patch | `tools/sageattention-sm103.patch` modifies [thu-ml/SageAttention revision d1a57a5](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5), [Apache-2.0](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/LICENSE) |

Optional FlashAttention and SageAttention backends are installed separately from their upstream packages and keep their own license terms. Refer to the upstream repositories, model cards, and license files for the terms governing each component. The [Vimoo license](LICENSE) does not replace those terms.
