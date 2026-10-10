"""Task selection and lazy conditioning for full and shared Lite TI2V recipes."""

from pathlib import Path

import folder_paths
from PIL import Image

from .vimoo_checkpoint import sha256
from .vimoo_vision import load_pil_image, validate_source_image_decode
from .vimoo_nodes import VimooSampler, _validate_video_inputs
from .vimoo_ti2v import ti2v_guidance_branches
from .vimoo_ti2v import LITE_TI2V_DEFAULTS, validate_guidance_values

LITE_PROMPTS = LITE_TI2V_DEFAULTS
_COMMON = LITE_TI2V_DEFAULTS["common"]


class VimooLiteDistilledSampler(VimooSampler):
    FIXED_NOISE_PROFILE = "lite_ti2v_distilled"
    DESCRIPTION = "Samples Vimoo-1.0-Lite TI2V Distilled with its fixed CPU BF16 noise seed 1234."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        del inputs["required"]["seed"]
        del inputs["required"]["renoise_seed"]
        del inputs["optional"]["add_fps"]
        return inputs

    def sample(self, **kwargs):
        from .vimoo_ti2v import LITE_TI2V_DISTILLED_NOISE_SEED

        recipe = kwargs["model"].model_options.get("vimoo_distilled")
        if not recipe or recipe.get("profile") != "lite_ti2v_distilled":
            raise ValueError(
                "Vimoo Lite Distilled Sampler requires the Vimoo-1.0-Lite TI2V Distilled adapter"
            )
        return super().sample(seed=LITE_TI2V_DISTILLED_NOISE_SEED, **kwargs)


class VimooImageLoader:
    DESCRIPTION = (
        "Loads the reference image for a fixed I2V workflow, using the connected model's image settings. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        files = [p.name for p in Path(folder_paths.get_input_directory()).iterdir() if p.is_file()]
        images = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(images), {"image_upload": True}),
            },
            "optional": {"model": ("MODEL",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, image, model=None):
        if image == "none" or not folder_paths.exists_annotated_filepath(image):
            return "I2V requires an uploaded reference image"
        try:
            with Image.open(folder_paths.get_annotated_filepath(image)) as source:
                source.load()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            return f"Cannot decode reference image {image!r}: {error}"
        return True

    @classmethod
    def IS_CHANGED(cls, image, model=None):
        return sha256(folder_paths.get_annotated_filepath(image))

    def load(self, image, model=None):
        valid = self.VALIDATE_INPUTS(image)
        if valid is not True:
            raise ValueError(valid)
        decoder = validate_source_image_decode(
            None if model is None else model.model_options.get("vimoo_source_image_decode")
        )
        if decoder == "pil":
            return (load_pil_image(folder_paths.get_annotated_filepath(image)),)
        from nodes import LoadImage

        return (LoadImage().load_image(image)[0],)


class VimooTI2VInput:
    DESCRIPTION = "Selects T2V or I2V for Vimoo-1.0-Lite TI2V and Vimoo-1.0-Lite TI2V Distilled. I2V loads the reference image; T2V skips it. "

    @classmethod
    def INPUT_TYPES(cls):
        inputs = VimooImageLoader.INPUT_TYPES()
        images, options = inputs["required"]["image"]
        inputs["required"] = {
            "task": (["t2v", "i2v"],),
            "image": (["none", *images], options),
        }
        return inputs

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("task", "image")
    FUNCTION = "load"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, task, image, model=None):
        if task not in ("t2v", "i2v"):
            return "Choose t2v or i2v"
        return True if task == "t2v" else VimooImageLoader.VALIDATE_INPUTS(image, model)

    @classmethod
    def IS_CHANGED(cls, task, image, model=None):
        return "t2v" if task == "t2v" else VimooImageLoader.IS_CHANGED(image, model)

    def load(self, task, image, model=None):
        valid = self.VALIDATE_INPUTS(task, image)
        if valid is not True:
            raise ValueError(valid)
        if task == "t2v":
            return task, None
        return task, VimooImageLoader().load(image, model)[0]


class VimooDistilledTI2VInput(VimooTI2VInput):
    DESCRIPTION = (
        "Selects T2V or I2V for Vimoo-1.0-Lite TI2V Distilled, using its PIL reference-image decoder."
    )

    def load(self, task, image, model=None):
        valid = self.VALIDATE_INPUTS(task, image)
        if valid is not True:
            raise ValueError(valid)
        if task == "t2v":
            return task, None
        return task, load_pil_image(folder_paths.get_annotated_filepath(image))


class VimooTI2VConditioning:
    DESCRIPTION = (
        "T2V creates an empty video latent; I2V adds VAE reference latents and SigLIP features. "
        "Only I2V evaluates the reference VAE and vision encoder. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": ("STRING", {"forceInput": True}),
                "positive": ("CONDITIONING",),
                "width": ("INT", {"default": _COMMON["width"], "min": 16, "step": 16}),
                "height": ("INT", {"default": _COMMON["height"], "min": 16, "step": 16}),
                "length": ("INT", {"default": _COMMON["num_frames"], "min": 5, "step": 4}),
                "first_frame_replace": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "image": ("IMAGE", {"lazy": True}),
                "vae": (
                    "VAE",
                    {
                        "lazy": True,
                        "tooltip": "I2V reference-image encoding, separate from output video decoding.",
                    },
                ),
                "clip_vision": (
                    "CLIP_VISION",
                    {
                        "lazy": True,
                        "tooltip": "SigLIP visual features; needed only for I2V.",
                    },
                ),
                "attention_backend": (
                    ["comfy", "precise"],
                    {
                        "default": "comfy",
                        "tooltip": "I2V SigLIP vision attention. Text and video-model attention have separate controls.",
                    },
                ),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Vimoo"

    @classmethod
    def VALIDATE_INPUTS(cls, width=None, height=None, length=None):
        return _validate_video_inputs(cls, width, height, length)

    def check_lazy_status(self, task, image=None, vae=None, clip_vision=None, **kwargs):
        if task == "t2v":
            return []
        if task != "i2v":
            raise ValueError("Choose t2v or i2v")
        return [
            key
            for key, value in {"image": image, "vae": vae, "clip_vision": clip_vision}.items()
            if value is None
        ]

    def encode(
        self,
        task,
        positive,
        width,
        height,
        length,
        first_frame_replace=False,
        image=None,
        vae=None,
        clip_vision=None,
        attention_backend="comfy",
        model=None,
        negative=None,
    ):
        from .vimoo_nodes import VimooEmptyLatentVideo, VimooImageToVideo

        if task == "t2v":
            keys = {
                "concat_latent_image",
                "concat_mask",
                "clip_vision_output",
                "vimoo_ref_latent",
                "vimoo_vision_mask",
            }

            def clean(conditioning):
                return [
                    [tensor, {k: v for k, v in extra.items() if k not in keys}]
                    for tensor, extra in conditioning
                ]

            return (
                clean(positive),
                None if negative is None else clean(negative),
                VimooEmptyLatentVideo().generate(width, height, length)[0],
            )
        if task != "i2v" or any(value is None for value in (image, vae, clip_vision)):
            raise ValueError("I2V requires an image, reference VAE and vision encoder")
        return VimooImageToVideo().encode(
            positive,
            vae,
            clip_vision,
            image,
            width,
            height,
            length,
            first_frame_replace,
            attention_backend,
            model=model,
            negative=negative,
        )


class VimooTI2VSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "task": ("STRING", {"forceInput": True}),
                "steps": ("INT", {"default": _COMMON["num_inference_steps"], "min": 1, "max": 200}),
                "flow_shift": (
                    "FLOAT",
                    {
                        "default": _COMMON["flow_shift"],
                        "min": 0.01,
                        "tooltip": "Flow shift for Vimoo-1.0 T2V and Vimoo-1.0-Lite TI2V; Vimoo-1.0 I2V uses full_i2v_flow_shift.",
                    },
                ),
                "t2v_cfg": ("FLOAT", {"default": LITE_TI2V_DEFAULTS["t2v"]["guidance_scale"], "min": 0.0}),
                "i2v_cfg": ("FLOAT", {"default": LITE_TI2V_DEFAULTS["i2v"]["guidance_scale"], "min": 0.0}),
                "t2v_rescale": ("FLOAT", {"default": _COMMON["guidance_rescale"], "min": 0.0}),
                "i2v_rescale": ("FLOAT", {"default": _COMMON["guidance_rescale"], "min": 0.0}),
                "first_frame_replace": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "recipe": (
                    ["lite", "full"],
                    {
                        "default": "lite",
                        "tooltip": "Full selects its task recipe. Lite reads checkpoint metadata: Lite TI2V uses lognorm Euler with anchor I2V.",
                    },
                ),
                "full_i2v_flow_shift": ("FLOAT", {"default": 6.0, "min": 0.01}),
            },
        }

    RETURN_TYPES = (
        "SIGMAS",
        "FLOAT",
        "FLOAT",
        "BOOLEAN",
        "COMBO",
        "INT",
    )
    RETURN_NAMES = (
        "sigmas",
        "cfg",
        "guidance_rescale",
        "first_frame_replace",
        "sampler_name",
        "max_sequence_length",
    )
    FUNCTION = "resolve"
    CATEGORY = "Vimoo"

    def resolve(
        self,
        model,
        task,
        steps,
        flow_shift,
        t2v_cfg,
        i2v_cfg,
        t2v_rescale,
        i2v_rescale,
        first_frame_replace,
        recipe="lite",
        full_i2v_flow_shift=6.0,
    ):
        from .vimoo_sampler import recipe_sigmas
        from .vimoo_nodes import VimooScheduler

        if task not in ("t2v", "i2v"):
            raise ValueError("Choose t2v or i2v")
        if recipe not in ("lite", "full"):
            raise ValueError("Choose lite or full recipe")
        validate_guidance_values(guidance_scale=t2v_cfg, guidance_rescale=t2v_rescale)
        validate_guidance_values(guidance_scale=i2v_cfg, guidance_rescale=i2v_rescale)
        if recipe == "full":
            if model.model_options.get("vimoo_distilled") or getattr(
                model.model.diffusion_model, "hms", False
            ):
                raise ValueError("The full recipe requires a Vimoo-1.0 T2V or I2V checkpoint")
            profile = model.model_options.get("vimoo_sampling_profile")
            if profile in ("standard_t2v", "standard_i2v") and profile != f"standard_{task}":
                raise ValueError(f"{profile} checkpoint does not match task={task}")
            if task == "i2v":
                return (
                    VimooScheduler().get_sigmas("vimoo_i2v_euler", steps, full_i2v_flow_shift)[0],
                    i2v_cfg,
                    i2v_rescale,
                    first_frame_replace,
                    "vimoo_i2v_euler",
                    1000,
                )
            return (
                VimooScheduler().get_sigmas("vimoo", steps, flow_shift)[0],
                t2v_cfg,
                t2v_rescale,
                False,
                "vimoo_uni_pc",
                1000,
            )
        if model.model_options.get("vimoo_sampling_profile") != LITE_TI2V_DEFAULTS["profile"]:
            raise ValueError(
                "Lite recipe requires the Vimoo-1.0-Lite TI2V checkpoint; use full for Vimoo-1.0 T2V and I2V"
            )
        document = model.model_options.get("vimoo_distilled")
        if document:
            if document.get("profile") != "lite_ti2v_distilled":
                raise ValueError("No matching Distilled adapter has been supplied for Vimoo-1.0-Lite TI2V")
            selected = document["recipes"][task]
            if (
                steps != selected["num_inference_steps"]
                or flow_shift != selected["flow_shift"]
                or (i2v_cfg if task == "i2v" else t2v_cfg) != 1
                or (i2v_rescale if task == "i2v" else t2v_rescale) != 0
                or first_frame_replace
            ):
                raise ValueError(
                    "Vimoo-1.0-Lite TI2V Distilled workflow controls must match its adapter recipe"
                )
            return (
                recipe_sigmas({**selected, "task": task}).float(),
                1.0,
                0.0,
                False,
                "vimoo_lite_ti2v_euler",
                selected["max_sequence_length"],
            )
        return (
            VimooScheduler().get_sigmas("vimoo_lite_ti2v_euler", steps, flow_shift)[0],
            i2v_cfg if task == "i2v" else t2v_cfg,
            i2v_rescale if task == "i2v" else t2v_rescale,
            False,
            "vimoo_lite_ti2v_euler",
            _COMMON["max_sequence_length"],
        )


class VimooTI2VTextEncodePair:
    DESCRIPTION = (
        "Vimoo-1.0-Lite TI2V encodes its required positive/negative CFG branches in one text batch. "
        "Text CFG and reference CFG choose the branches; distilled CFG 1 can use a single Text Encode. "
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "task": ("STRING", {"forceInput": True}),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": LITE_PROMPTS["positive_prompt"],
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Wanted scene and motion.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": LITE_PROMPTS["negative_prompt"],
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Unwanted content for CFG branches.",
                    },
                ),
                "glyph_conditioning": ("BOOLEAN", {"default": True}),
                "max_sequence_length": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),
                "reference_guidance_scale": (
                    "FLOAT",
                    {
                        "default": LITE_TI2V_DEFAULTS["i2v"]["reference_guidance_scale"],
                        "min": 0.0,
                        "tooltip": "Lite I2V image guidance, independent of text cfg. A value other than 1 can require negative conditioning even at text CFG 1.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "reference_guidance_scale")
    FUNCTION = "encode"
    CATEGORY = "Vimoo"

    def encode(
        self,
        clip,
        task,
        positive_prompt,
        negative_prompt,
        glyph_conditioning,
        max_sequence_length,
        cfg,
        reference_guidance_scale=LITE_TI2V_DEFAULTS["i2v"]["reference_guidance_scale"],
    ):
        import torch

        if task not in ("t2v", "i2v"):
            raise ValueError("Choose t2v or i2v")
        validate_guidance_values(guidance_scale=cfg, reference_guidance_scale=reference_guidance_scale)
        if getattr(clip.cond_stage_model, "vimoo_precision_profile", None) not in (
            LITE_TI2V_DEFAULTS["profile"],
            "lite_compact",
        ):
            raise ValueError("TI2V paired text encoding requires its bound precision profile")
        reference_scale = reference_guidance_scale if task == "i2v" else 1.0
        names = ti2v_guidance_branches(cfg, reference_scale, task == "i2v")
        positive = clip.tokenize(
            positive_prompt,
            glyph_conditioning=glyph_conditioning,
            max_sequence_length=max_sequence_length,
        )
        negative = (
            clip.tokenize(
                negative_prompt,
                glyph_conditioning=False,
                max_sequence_length=max_sequence_length,
            )
            if len(names) > 1
            else None
        )
        batch = [positive if name == "positive" else negative for name in names]
        tokens = {"qwen25_7b": [prompt["qwen25_7b"][0] for prompt in batch], "vimoo_cfg_prompts": batch}
        encoded = clip.encode_from_tokens_scheduled(tokens)
        positive_output, negative_output = [], []
        for tensor, values in encoded:
            required = {"attention_mask", "conditioning_byt5small", "vimoo_byt5_mask"}
            if not required.issubset(values) or tensor.shape[0] != len(batch):
                raise ValueError(
                    "Paired text encoding lost CFG branches or masks; disable scheduled CLIP hooks"
                )

            def branch(index):
                return [
                    tensor[index : index + 1],
                    {
                        key: value[index : index + 1]
                        if torch.is_tensor(value) and value.ndim and value.shape[0] == len(batch)
                        else value
                        for key, value in values.items()
                    },
                ]

            positive_output.append(branch(len(batch) - 1))
            if len(batch) > 1:
                negative_branch = branch(len(batch) - 2)
                if "unreferenced" in names:
                    negative_branch[1]["vimoo_unreferenced_conditioning"] = branch(0)
                negative_output.append(negative_branch)
        return positive_output, negative_output or None, reference_scale


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (
        VimooImageLoader,
        VimooTI2VInput,
        VimooDistilledTI2VInput,
        VimooLiteDistilledSampler,
        VimooTI2VConditioning,
        VimooTI2VSchedule,
        VimooTI2VTextEncodePair,
    )
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VimooImageLoader": "Vimoo Load Image",
    "VimooTI2VInput": "Vimoo TI2V Input",
    "VimooDistilledTI2VInput": "Vimoo Lite Distilled Input",
    "VimooLiteDistilledSampler": "Vimoo Lite Distilled Sampler",
    "VimooTI2VTextEncodePair": "Vimoo TI2V Paired Text Encode",
    "VimooTI2VConditioning": "Vimoo TI2V Conditioning",
    "VimooTI2VSchedule": "Vimoo TI2V Schedule",
}
