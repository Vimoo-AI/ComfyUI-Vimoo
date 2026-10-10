"""Image-ratio output sizing using the same 720p budget as Vimoo MLX."""

from math import sqrt
import logging


def reference_output_size(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Reference image dimensions must be positive")
    pixel_budget = 1280 * 720
    scale = sqrt(pixel_budget / (width * height))
    maximum = pixel_budget // 16
    return (
        min(maximum, max(16, int(width * scale) // 16 * 16)),
        min(maximum, max(16, int(height * scale) // 16 * 16)),
    )


class VimooResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (["auto", "manual"], {"default": "auto"}),
                "width": ("INT", {"default": 1280, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "task": ("STRING", {"forceInput": True}),
            },
            "optional": {"image": ("IMAGE", {"lazy": True})},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "resolve"
    CATEGORY = "Vimoo"
    DESCRIPTION = "Auto fits the reference image ratio to a 1280×720 pixel budget, aligned to 16."

    def check_lazy_status(self, resolution, task, image=None, **kwargs):
        return ["image"] if resolution == "auto" and task == "i2v" and image is None else []

    @classmethod
    def VALIDATE_INPUTS(cls, resolution, width=None, height=None, task=None):
        try:
            cls._validate(
                resolution,
                16 if width is None else width,
                16 if height is None else height,
                "t2v" if task is None else task,
            )
        except ValueError as error:
            return str(error)
        return True

    @staticmethod
    def _validate(resolution, width, height, task):
        if resolution not in ("auto", "manual") or task not in ("t2v", "i2v"):
            raise ValueError("Choose auto/manual resolution and t2v/i2v task")
        if resolution == "manual" and any(
            type(value) is not int or value % 16 or not 16 <= value <= 8192 for value in (width, height)
        ):
            raise ValueError("Manual dimensions must be multiples of 16 between 16 and 8192")

    def resolve(self, resolution, width, height, task, image=None):
        self._validate(resolution, width, height, task)
        if resolution == "auto":
            if task == "t2v":
                width, height = 1280, 720
            else:
                if image is None or image.ndim != 4 or image.shape[0] < 1:
                    raise ValueError("I2V auto resolution requires a reference image")
                width, height = reference_output_size(image.shape[2], image.shape[1])
            if max(width, height) > 8192:
                raise ValueError(
                    "Auto resolution exceeds the 8192 pixel axis limit; choose manual dimensions"
                )
        logging.info("Vimoo resolved %s %s resolution: %dx%d", task, resolution, width, height)
        return {"ui": {"vimoo_resolution": [width, height]}, "result": (width, height)}


NODE_CLASS_MAPPINGS = {"VimooResolution": VimooResolution}
NODE_DISPLAY_NAME_MAPPINGS = {"VimooResolution": "Vimoo Resolution"}
