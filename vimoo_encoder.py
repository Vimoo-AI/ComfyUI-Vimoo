import re

import torch

from comfy import sd1_clip
import comfy.text_encoders.llama
import comfy.model_management
from comfy.text_encoders.hunyuan_image import ByT5SmallModel, ByT5SmallTokenizer
from comfy.text_encoders.qwen_image import QwenImageTokenizer


VIMOO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:         "
    "1. The main content and theme of the video.         "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
    "4. background environment, light, style and atmosphere.         "
    "5. camera angles, movements, and transitions used in the video."
)


def glyph_texts(text: str) -> list[str]:
    matches = re.findall(r'"(.*?)"|“(.*?)”', text)
    return list(dict.fromkeys(a or b for a, b in matches))


def crop_start(row: list) -> int:
    for i in range(len(row) - 2):
        if tuple(token[0] for token in row[i : i + 3]) == (151644, 872, 198):
            return i + 3
    return 0


class VimooTokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        tokenizer_data = tokenizer_data or {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.byt5 = ByT5SmallTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = (
            "<|im_start|>system\n" + VIMOO_SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def tokenize_with_weights(
        self,
        text: str,
        return_word_ids=False,
        glyph_conditioning=True,
        max_sequence_length=1000,
        **kwargs,
    ):
        if type(max_sequence_length) is not int or max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        out = super().tokenize_with_weights(text, return_word_ids, prevent_empty_text=True, **kwargs)
        # Both published paths use fixed padding and carry glyph masks into DiT.
        for index, row in enumerate(out["qwen25_7b"]):
            length = crop_start(row) + max_sequence_length
            out["qwen25_7b"][index] = row[:length] + [(151643, 1.0)] * (length - len(row))

        texts = glyph_texts(text) if glyph_conditioning else []
        if texts:
            glyph_prompt = "".join('Text "{}". '.format(glyph) for glyph in texts)
            glyph_options = {**kwargs, "disable_weights": True}
            rows = self.byt5.tokenize_with_weights(glyph_prompt, return_word_ids, **glyph_options)
            out["byt5"] = [row[:255] + [row[-1]] if len(row) > 256 else row[:256] for row in rows]
        if "byt5" in out:
            out["byt5"] = [row + [(0, 1.0)] * (256 - len(row)) for row in out["byt5"]]
        return out


class VimooQwen25Text(comfy.text_encoders.llama.BaseLlama, torch.nn.Module):
    """Text-only Qwen2.5-VL tower.

    Vimoo supplies visual conditioning through its separate SigLIP encoder, so
    constructing Qwen's unused vision tower wastes memory and produces hundreds
    of misleading missing-key warnings for a valid text-only checkpoint.
    """

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = comfy.text_encoders.llama.Qwen25_7BVLI_Config(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = comfy.text_encoders.llama.Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype


class VimooQwen25Model(sd1_clip.SDClipModel):
    def __init__(
        self,
        device="cpu",
        layer="hidden",
        layer_idx=-3,
        dtype=None,
        attention_mask=True,
        model_options=None,
    ):
        model_options = model_options or {}
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=VimooQwen25Text,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


class VimooTextEncoder(sd1_clip.SD1ClipModel):
    def __init__(
        self, device="cpu", dtype=None, model_options=None, dtype_llama=None, llama_quantization_metadata=None
    ):
        model_options = model_options or {}
        if llama_quantization_metadata is not None:
            model_options = {**model_options, "llama_quantization_metadata": llama_quantization_metadata}
        if dtype_llama is not None:
            dtype = dtype_llama
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen25_7b",
            clip_model=VimooQwen25Model,
            model_options=model_options,
        )
        self.byt5_small = ByT5SmallModel(device=device, dtype=torch.float32, model_options=model_options)
        self.byt5_small.zero_out_masked = False
        self.dtypes.add(torch.float32)

    def encode_token_weights(self, token_weight_pairs):
        if "vimoo_cfg_prompts" in token_weight_pairs:
            return self.encode_cfg_batch(token_weight_pairs["vimoo_cfg_prompts"])
        rows = token_weight_pairs["qwen25_7b"]
        cond, pooled, extra = self.qwen25_7b.encode_token_weights(rows)
        start = crop_start(rows[0])
        cond = cond[:, start:]
        extra["attention_mask"] = extra["attention_mask"][:, start:]
        glyphs = token_weight_pairs.get("byt5")
        if glyphs is not None:
            extra["vimoo_byt5_mask"] = torch.tensor(
                [[int(token[0] != 0) for token in row] for row in glyphs],
                device=cond.device,
                dtype=torch.int64,
            )
            glyph_states = self.byt5_small.encode_token_weights(glyphs)[0]
            if getattr(self, "vimoo_precision_profile", None) == "vimoo_bf16":
                glyph_states = glyph_states.bfloat16()
            extra["conditioning_byt5small"] = glyph_states
        else:
            extra["vimoo_byt5_mask"] = torch.zeros(
                (cond.shape[0], 256),
                dtype=torch.int64,
                device=cond.device,
            )
            extra["conditioning_byt5small"] = torch.zeros(
                (cond.shape[0], 256, 1472),
                dtype=torch.bfloat16,
                device=cond.device,
            )
        return cond, pooled, extra

    def encode_cfg_batch(self, prompts):
        """One Qwen forward for the CFG batch; glyph prompts stay independent."""
        native_patchify = getattr(self, "vimoo_precision_profile", None) == "lite_compact"
        rows = [prompt["qwen25_7b"][0] for prompt in prompts]
        if any(len(prompt["qwen25_7b"]) != 1 for prompt in prompts):
            raise ValueError("Lite TI2V expects one fixed-length text row per CFG branch")
        if any(weight != 1.0 for row in rows for _, weight, *_ in row):
            raise ValueError("Lite TI2V reference prompts do not support token weights")
        encoded, _, extra = self.qwen25_7b.encode([[token[0] for token in row] for row in rows])
        starts = [crop_start(row) for row in rows]
        cond = torch.cat([encoded[index : index + 1, start:] for index, start in enumerate(starts)])
        masks = torch.cat(
            [extra["attention_mask"][index : index + 1, start:] for index, start in enumerate(starts)]
        )
        glyph_states, glyph_masks = [], []
        for prompt in prompts:
            glyphs = prompt.get("byt5")
            if glyphs is None:
                if native_patchify:
                    # Evaluate an empty ByT5 row before
                    # zeroing its states and mask. ByT5 uses EOS=1, PAD=0.
                    dummy = [[(1, 1.0)] + [(0, 1.0)] * 255]
                    states = self.byt5_small.encode_token_weights(dummy)[0].cpu().bfloat16()
                    glyph_states.append(torch.zeros_like(states))
                else:
                    glyph_states.append(torch.zeros(1, 256, 1472, dtype=torch.bfloat16))
                glyph_masks.append(torch.zeros(1, 256, dtype=torch.int64))
            else:
                # _cpu_byt5_forward controls execution independently of Qwen's device.
                glyph_states.append(self.byt5_small.encode_token_weights(glyphs)[0].cpu().bfloat16())
                glyph_masks.append(torch.tensor([[int(token[0] != 0) for token in row] for row in glyphs]))
        device = comfy.model_management.intermediate_device()
        mask_dtype = torch.bool if native_patchify else masks.dtype
        return (
            cond.to(device=device, dtype=torch.bfloat16),
            None,
            {
                "attention_mask": masks.to(device=device, dtype=mask_dtype),
                "conditioning_byt5small": torch.cat(glyph_states).to(device),
                "vimoo_byt5_mask": torch.cat(glyph_masks).to(
                    device=device, dtype=torch.bool if native_patchify else torch.int64
                ),
            },
        )

    def set_clip_options(self, options):
        super().set_clip_options(options)
        self.byt5_small.set_clip_options(options)

    def reset_clip_options(self):
        super().reset_clip_options()
        self.byt5_small.reset_clip_options()

    def load_sd(self, sd):
        if "encoder.block.0.layer.0.SelfAttention.o.weight" in sd:
            return self.byt5_small.load_sd(sd)
        return super().load_sd(sd)
