"""Checkpoint naming shared by native loading and offline ComfyUI conversion."""

import hashlib
import json
from pathlib import Path
import re

import torch

_BLOCK_SUBS = [
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.norm_added_q\.weight$"),
        r"double_blocks.\1.txt_attn.norm.query_norm.weight",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.norm_added_k\.weight$"),
        r"double_blocks.\1.txt_attn.norm.key_norm.weight",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.norm_q\.weight$"),
        r"double_blocks.\1.img_attn.norm.query_norm.weight",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.norm_k\.weight$"),
        r"double_blocks.\1.img_attn.norm.key_norm.weight",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$"),
        r"double_blocks.\1.img_attn.proj.\2",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.attn\.to_add_out\.(weight|bias)$"),
        r"double_blocks.\1.txt_attn.proj.\2",
    ),
    (re.compile(r"^double_blocks\.(\d+)\.img_mlp\.fc1\.(weight|bias)$"), r"double_blocks.\1.img_mlp.0.\2"),
    (re.compile(r"^double_blocks\.(\d+)\.img_mlp\.fc2\.(weight|bias)$"), r"double_blocks.\1.img_mlp.2.\2"),
    (re.compile(r"^double_blocks\.(\d+)\.txt_mlp\.fc1\.(weight|bias)$"), r"double_blocks.\1.txt_mlp.0.\2"),
    (re.compile(r"^double_blocks\.(\d+)\.txt_mlp\.fc2\.(weight|bias)$"), r"double_blocks.\1.txt_mlp.2.\2"),
    (
        re.compile(r"^double_blocks\.(\d+)\.img_mod\.linear\.(weight|bias)$"),
        r"double_blocks.\1.img_mod.lin.\2",
    ),
    (
        re.compile(r"^double_blocks\.(\d+)\.txt_mod\.linear\.(weight|bias)$"),
        r"double_blocks.\1.txt_mod.lin.\2",
    ),
    (re.compile(r"^time_in\.mlp\.0\.(weight|bias)$"), r"time_in.in_layer.\1"),
    (re.compile(r"^time_in\.mlp\.2\.(weight|bias)$"), r"time_in.out_layer.\1"),
    (re.compile(r"^txt_in\.t_embedder\.mlp\.0\.(weight|bias)$"), r"txt_in.t_embedder.in_layer.\1"),
    (re.compile(r"^txt_in\.t_embedder\.mlp\.2\.(weight|bias)$"), r"txt_in.t_embedder.out_layer.\1"),
    (re.compile(r"^txt_in\.c_embedder\.linear_1\.(weight|bias)$"), r"txt_in.c_embedder.in_layer.\1"),
    (re.compile(r"^txt_in\.c_embedder\.linear_2\.(weight|bias)$"), r"txt_in.c_embedder.out_layer.\1"),
    (
        re.compile(r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(weight|bias)$"),
        r"txt_in.individual_token_refiner.blocks.\1.self_attn.qkv.\2",
    ),
    (
        re.compile(r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_proj\.(weight|bias)$"),
        r"txt_in.individual_token_refiner.blocks.\1.self_attn.proj.\2",
    ),
    (
        re.compile(r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)$"),
        r"txt_in.individual_token_refiner.blocks.\1.mlp.0.\2",
    ),
    (
        re.compile(r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)$"),
        r"txt_in.individual_token_refiner.blocks.\1.mlp.2.\2",
    ),
]

_QKV_PATTERNS = {
    "img": re.compile(r"^double_blocks\.(\d+)\.attn\.to_([qkv])\.(weight|bias)$"),
    "txt": re.compile(r"^double_blocks\.(\d+)\.attn\.add_([qkv])_proj\.(weight|bias)$"),
}

_BACKEND_ATTENTION_SUBS = (
    (".img_attn_q.", ".attn.to_q."),
    (".img_attn_k.", ".attn.to_k."),
    (".img_attn_v.", ".attn.to_v."),
    (".img_attn_q_norm.", ".attn.norm_q."),
    (".img_attn_k_norm.", ".attn.norm_k."),
    (".img_attn_proj.", ".attn.to_out.0."),
    (".txt_attn_q.", ".attn.add_q_proj."),
    (".txt_attn_k.", ".attn.add_k_proj."),
    (".txt_attn_v.", ".attn.add_v_proj."),
    (".txt_attn_q_norm.", ".attn.norm_added_q."),
    (".txt_attn_k_norm.", ".attn.norm_added_k."),
    (".txt_attn_proj.", ".attn.to_add_out."),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checkpoint_provenance(path, *, task=None):
    sidecar = Path(str(path) + ".vimoo.json")
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid checkpoint provenance: {sidecar.name}") from error
    if not isinstance(document, dict):
        raise ValueError("Checkpoint provenance must be a JSON object")
    if (
        document.get("format") != "vimoo-checkpoint-binding-v2"
        or document.get("task") not in ("t2v", "i2v", "ti2v")
        or (task is not None and document["task"] != task)
        or not valid_sha256(document.get("sha256"))
        or "derived_from" in document
    ):
        raise ValueError("Invalid checkpoint provenance format, task, or SHA-256")
    if sha256(path) != document["sha256"]:
        raise ValueError("Checkpoint provenance checksum mismatch")
    base = document.get("base")
    if base is not None:
        if (
            not isinstance(base, dict)
            or not isinstance(base.get("file"), str)
            or Path(base["file"]).name != base["file"]
            or not base["file"].endswith(".safetensors")
            or not valid_sha256(base.get("sha256"))
        ):
            raise ValueError("Checkpoint base binding requires an adjacent public file and SHA-256")
        base_path = sidecar.parent / base["file"]
        if base_path.resolve() == Path(path).resolve() or not base_path.is_file():
            raise ValueError("Checkpoint base binding must reference a separate public checkpoint")
        if sha256(base_path) != base["sha256"]:
            raise ValueError("Checkpoint base binding checksum mismatch")
    return document


def normalize_backend_transformer_key(key: str) -> str:
    """Normalize supported Diffusers generations and native backend names."""
    if key.startswith("condition_embedder."):
        key = key.removeprefix("condition_embedder.")
    if key.startswith("block_modulation_proj."):
        key = "mod_projection." + key.removeprefix("block_modulation_proj.")
    if key.startswith("block_projection."):
        key = "mod_projection." + key.removeprefix("block_projection.")

    for source, target in (
        (".attn.to_out.0.projection.weight", ".attn.to_out.0.weight"),
        (".attn.to_add_out.projection.weight", ".attn.to_add_out.weight"),
        (".net.0.proj.", ".fc1."),
        (".net.2.", ".fc2."),
        (".in_proj.projection.weight", ".fc1.weight"),
        (".in_proj.bias", ".fc1.bias"),
        (".out_proj.projection.weight", ".fc2.weight"),
        (".out_proj.bias", ".fc2.bias"),
    ):
        key = key.replace(source, target)

    key = re.sub(
        r"^(double_blocks\.\d+\.attn)\.norm_qk\.(query|key)_weight$",
        lambda match: f"{match.group(1)}.norm_{'q' if match.group(2) == 'query' else 'k'}.weight",
        key,
    )
    key = re.sub(
        r"^(double_blocks\.\d+\.attn)\.norm_added_qk\.(query|key)_weight$",
        lambda match: f"{match.group(1)}.norm_added_{'q' if match.group(2) == 'query' else 'k'}.weight",
        key,
    )
    key = re.sub(
        r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.attn\.to_qkv\.(weight|bias)$",
        r"\1.self_attn_qkv.\2",
        key,
    )
    key = re.sub(
        r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.attn\.to_out(?:\.projection)?\.(weight|bias)$",
        r"\1.self_attn_proj.\2",
        key,
    )
    for source, target in (
        ("vision_in.norm_in.", "vision_in.proj.0."),
        ("vision_in.linear_1.", "vision_in.proj.1."),
        ("vision_in.linear_2.", "vision_in.proj.3."),
        ("vision_in.norm_out.", "vision_in.proj.4."),
        ("final_layer.output_projection.", "final_layer.linear."),
    ):
        key = key.replace(source, target)
    for source, target in _BACKEND_ATTENTION_SUBS:
        key = key.replace(source, target)
    return key


def remap_transformer(key: str):
    for pattern, repl in _BLOCK_SUBS:
        if pattern.match(key):
            return pattern.sub(repl, key)
    compact = re.fullmatch(r"double_blocks\.(\d+)\.(img|txt)_mod\.compact_(linear\.weight|bias)", key)
    if compact:
        block, stream, part = compact.groups()
        suffix = "weight" if part == "linear.weight" else "bias"
        return f"double_blocks.{block}.{stream}_mod.lin.{suffix}"
    # identity passthrough groups
    for prefix in (
        "byt5_in.",
        "cond_type_embedding.",
        "img_in.proj.",
        "img_in.patch_merge.",
        "txt_in.input_embedder.",
        "txt_in.individual_token_refiner.blocks.",  # norm1/norm2/adaLN_modulation.1 keep their names
        "final_layer.linear.",
        "final_layer.patch_split.",
        "final_layer.adaLN_modulation.",
        "vision_in.proj.",
        "mod_projection.",
    ):
        if key.startswith(prefix):
            return key
    return None


def normalize_transformer_state(tensors, config, *, packed_qkv=True):
    """Convert names losslessly; split storage avoids allocating a second full model."""
    out, groups = {}, {}
    for source_key, tensor in tensors:
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported transformer dtype for {source_key}: {tensor.dtype}")
        key = normalize_backend_transformer_key(source_key)
        image, text = _QKV_PATTERNS["img"].match(key), _QKV_PATTERNS["txt"].match(key)
        if image or text:
            match = image or text
            block, part, suffix = int(match.group(1)), match.group(2), match.group(3)
            stream = "img" if image else "txt"
            group = groups.setdefault((stream, block, suffix), {})
            if part in group:
                raise ValueError(f"Duplicate QKV source tensor: {source_key}")
            group[part] = tensor
            if not packed_qkv:
                name = f"double_blocks.{block}.{stream}_attn.to_{part}.{suffix}"
                if name in out:
                    raise ValueError(f"Duplicate normalized transformer key: {name}")
                out[name] = tensor
            continue
        mapped = remap_transformer(key)
        if mapped is None:
            raise KeyError(f"unmapped transformer key: {source_key}")
        if mapped in out:
            raise KeyError(f"duplicate key after remap: {mapped}")
        out[mapped] = tensor
    depth = config.get("mm_double_blocks_depth", 40)
    expected = {
        (stream, block, suffix)
        for stream in ("img", "txt")
        for block in range(depth)
        for suffix in (("weight", "bias") if config.get("qkv_bias", True) else ("weight",))
    }
    if set(groups) != expected:
        raise ValueError(f"expected complete weight and bias QKV groups for {depth} img/txt blocks")
    for (stream, block, suffix), parts in groups.items():
        if set(parts) != {"q", "k", "v"}:
            raise KeyError(f"incomplete qkv for {stream} block {block}: {sorted(parts)}")
        if len({part.dtype for part in parts.values()}) != 1:
            raise ValueError("QKV parts must have identical dtypes; refusing implicit promotion")
        shapes = [tuple(parts[part].shape) for part in "qkv"]
        if any(shape != shapes[0] for shape in shapes) or not shapes[0]:
            raise ValueError(f"incompatible qkv shapes: {shapes}")
        if packed_qkv:
            out[f"double_blocks.{block}.{stream}_attn.qkv.{suffix}"] = torch.cat(
                [parts[p] for p in "qkv"], dim=0
            ).contiguous()
    hms_keys = {"img_in.patch_merge.weight", "final_layer.patch_split.weight"}
    present = hms_keys.intersection(out)
    if present and present != hms_keys:
        raise ValueError(f"Incomplete hidden_merge_split checkpoint; missing {sorted(hms_keys - present)}")
    architecture = config.get("patch_architecture", "direct")
    if architecture not in ("direct", "hidden_merge_split"):
        raise ValueError(f"Unsupported Vimoo patch architecture: {architecture}")
    if bool(present) != (architecture == "hidden_merge_split"):
        raise ValueError(
            "Vimoo patch architecture metadata/tensor mismatch; convert a configured Diffusers model"
        )
    if present and config.get("patch_size") != [1, 2, 2]:
        raise ValueError("hidden_merge_split requires patch_size=[1, 2, 2]")
    return out


def validate_modulation_state(state, hidden, depth, rank, precision):
    """Validate every expansion, its shared projection and the declared storage policy."""
    if precision not in ("fp32", "bf16_expand_fp32_bias"):
        raise ValueError(f"Unsupported block modulation precision: {precision!r}")
    if type(rank) is not int or rank < 1:
        raise ValueError("block_modulation_rank must be a positive integer")
    expected = {"mod_projection.weight": [rank, hidden], "mod_projection.bias": [rank]}
    for index in range(depth):
        for stream in ("img", "txt"):
            expected[f"double_blocks.{index}.{stream}_mod.lin.weight"] = [6 * hidden, rank]
            expected[f"double_blocks.{index}.{stream}_mod.lin.bias"] = [6 * hidden]
    for key, shape in expected.items():
        if key not in state:
            raise ValueError(f"Missing factorized Vimoo modulation tensor {key!r}")
        if list(state[key].shape) != shape:
            raise ValueError(f"Factorized Vimoo modulation tensor {key!r} is not {shape}")
        if precision == "bf16_expand_fp32_bias":
            dtype = torch.bfloat16 if key.endswith("lin.weight") else torch.float32
            if state[key].dtype != dtype:
                raise ValueError(f"Factorized Vimoo modulation tensor {key!r} must be {dtype}")
