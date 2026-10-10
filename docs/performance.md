# Attention, compilation and memory

Start with a supplied workflow: all templates use PyTorch SDPA
(`attention=pytorch`) and `compile=false`, with no extra attention package.
Adjust acceleration in **Vimoo Performance Options**, connected between the
model loader and sampler.

## Optional attention backends

| `attention` | Dependencies and requirements |
| --- | --- |
| `pytorch` | Default SDPA; no extra package. |
| `fa2` | Install `requirements-flash-attn.txt` for Hub kernels, or native `flash-attn`. Hub kernels also support Vimoo-1.0-Lite TI2V Distilled. |
| `fa3` | Install `requirements-flash-attn.txt`; requires NVIDIA Hopper (compute capability 9.x). |
| `fa4` | Install `requirements-fa4.txt`. |
| `sage` | Install `requirements-sageattention.txt`; uses quantized attention and can change the result. |

Run only the command for your chosen backend, from the ComfyUI root, using the
Python environment that launches ComfyUI:

```bash
# FA2 / FA3 Hub kernels
python -m pip install -r custom_nodes/ComfyUI-Vimoo/requirements-flash-attn.txt
# Optional native FA2 instead of Hub kernels
python -m pip install --no-build-isolation -r custom_nodes/ComfyUI-Vimoo/requirements-native-fa2.txt
# FA4
python -m pip install -r custom_nodes/ComfyUI-Vimoo/requirements-fa4.txt
# SageAttention 2
python -m pip install --no-build-isolation -r custom_nodes/ComfyUI-Vimoo/requirements-sageattention.txt
```

Hub kernels need network access or cached binaries matching Torch, CUDA, your OS
and GPU. Native FA2 must also match Torch, CUDA and Python. SageAttention is built
from source; it needs Git, a matching CUDA Toolkit and a C++ compiler. Native FA2
needs the same build tools when no compatible wheel is available. Restart ComfyUI
after installation.

Selecting `sage` uses the release configuration automatically: quantized image attention with exact text attention and PyTorch fallback. There are no separate Sage precision or kernel controls.

<details>
<summary>SageAttention 2.2.0 on Hopper/SM90</summary>

Apply the [upstream fake-handler fix](https://github.com/thu-ml/SageAttention/commit/d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4)
to the installed `sageattention/sm90_compile.py`, then restart ComfyUI.

</details>

## Compilation and memory

The `compile` switch is off by default. Turn it on to compile repeated Transformer
blocks with the release settings. Native weight staging and release stay outside
the compiled tensor graphs, preserving dynamic VRAM and asynchronous offload.
The first compiled run takes extra time;
changing the output size can trigger recompilation. If compilation fails,
turn the switch off. Compilation and quantized attention can change output pixels.

Checkpoint layout, precise arithmetic, prefetch and memory estimation are selected
by the release configuration. The model loader detects split or packed QKV from
the checkpoint.

### 16 GB VRAM

Use one ComfyUI process and `batch=1`. For Vimoo-1.0 T2V / I2V Distilled in BF16, start with
the supplied release settings. From the ComfyUI root:

```bash
python main.py --enable-dynamic-vram --async-offload 2 \
  --reserve-vram 3 --cache-ram 6 8
```

`--cache-ram` values reserve available system-memory headroom; they do not cap
process RAM. Increasing VRAM reserve can increase transfers and does not solve
every sampling OOM. If sampling runs out of memory, update the plugin and check
offload and batch size. VAE tiles only reduce decoding memory.

After the baseline works, Vimoo-1.0 I2V Distilled can use `attention=fa2` with
`compile=true`. Install the FA2 dependency
listed above. The same Hub FA2 kernel is shared by ordinary and distilled models.

## VAE decoding

Keep the selected workflow's VAE settings initially. Its saved decode modes,
tiles and video output are listed in the
[workflow guide](../example_workflows/README.md#decode-and-output).

- Enable `tiled_decode` in **Vimoo Output VAE** inside the generation subgraph
  to use the model's release tiles.
- With the switch off, decoding tries the whole video and falls back to automatic
  tiles after OOM. Automatic fallback selects its own tile sizes.
- Reference-image encoding uses a separate loader and keeps its release settings.

The convolution backend, tile sizes, strides and workspace budget are fixed by
the selected release profile. Changing the decode switch can change the result.

Vimoo-1.0-Lite TI2V uses `im2col_bounded`, which requires an NVIDIA GPU and
**PyTorch 2.9.0 or later**. It selects the bounded im2col convolution directly,
independently of Torch/cuDNN convolution dispatch. Choose a compatible CUDA build with
the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Its convolution workspace budget is **2 GiB**;
weights and activations need additional memory. Enable tiled decoding if the whole video does not fit. The `comfy` backend
follows the general runtime requirements.
