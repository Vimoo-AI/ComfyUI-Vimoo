"""Real Dynamo regressions for native casting state outside regional graphs."""

from contextlib import contextmanager
from functools import partial
import ctypes
import gc
import sys
from types import MethodType
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import torch

from test_nodes_runtime import nodes, spec
from test_model_runtime import packed_tiny_copy, tiny_vimoo

compilation = sys.modules[f"{spec.name}.vimoo_compile"]
precision = sys.modules[f"{spec.name}.vimoo_precision"]


class RegionalCompilationTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        self.graphs = []
        self.events = []

    def tearDown(self):
        torch._dynamo.reset()

    def backend(self, graph, inputs):
        self.graphs.append(graph)
        return graph.forward

    @contextmanager
    def native_context(self, module, value, **kwargs):
        self.assertFalse(torch.compiler.is_compiling())
        # Native DynamicVRAM uses ctypes and mutable prefetched/resident state.
        # Both must execute every call without becoming Dynamo guards.
        ctypes.c_void_p(id(module))
        self.events.append(("cast", dict(getattr(module, "_prefetch", {}))))
        try:
            yield module.weight, module.bias
        finally:
            self.assertFalse(torch.compiler.is_compiling())
            self.events.append(("release", None))

    def test_cast_state_changes_do_not_recompile_tensor_math(self):
        for forward in (compilation._cast, precision._anchor_linear_forward):
            with self.subTest(forward=forward.__name__):
                torch._dynamo.reset()
                self.graphs.clear()
                module = nodes.comfy.ops.manual_cast.Linear(4, 4)
                torch.nn.init.normal_(module.weight)
                torch.nn.init.normal_(module.bias)
                module.vimoo_upcast_bias = False
                specialized = compilation.specialize_cast(forward, module)
                function = torch.compile(MethodType(specialized, module), backend=self.backend)
                value = torch.randn(2, 4)
                with patch.object(nodes.comfy.ops, "CastBiasWeightContext", self.native_context):
                    expected = forward(module, value)
                    torch.testing.assert_close(function(value), expected, rtol=0, atol=0)
                    cold_graphs = len(self.graphs)
                    self.assertGreater(cold_graphs, 0)
                    self.assertTrue(any("linear" in graph.code for graph in self.graphs))
                    for index in range(12):
                        module._prefetch = {"resident": bool(index % 2), "signature": index}
                        if index % 3:
                            module._prefetch["extra"] = index
                        torch.testing.assert_close(function(value), expected, rtol=0, atol=0)
                    self.assertEqual(len(self.graphs), cold_graphs)
                    del module._prefetch
                    torch.testing.assert_close(function(value), expected, rtol=0, atol=0)
                    self.assertEqual(len(self.graphs), cold_graphs)
                self.assertEqual(
                    sum(event[0] == "cast" for event in self.events),
                    sum(event[0] == "release" for event in self.events),
                )

    def test_operator_shapes_have_independent_continuation_caches(self):
        functions = []
        with patch.object(nodes.comfy.ops, "CastBiasWeightContext", self.native_context):
            for size in range(4, 16):
                module = nodes.comfy.ops.manual_cast.Linear(size, size)
                torch.nn.init.normal_(module.weight)
                torch.nn.init.normal_(module.bias)
                function = compilation.specialize_cast(compilation._cast, module)
                compiled = torch.compile(MethodType(function, module), backend=self.backend)
                value = torch.randn(2, size)
                torch.testing.assert_close(compiled(value), function(module, value))
                functions.append((compiled, value))
            self.assertEqual(len(self.graphs), 12)
            for compiled, value in functions:
                compiled(value)
            self.assertEqual(len(self.graphs), 12)
            self.assertFalse(torch._dynamo.utils.counters["unimplemented"].get("recompile_limit reached", 0))

    def test_exception_releases_native_cast_buffer(self):
        module = nodes.comfy.ops.manual_cast.Linear(4, 4)
        with patch.object(nodes.comfy.ops, "CastBiasWeightContext", self.native_context):
            with self.assertRaises(RuntimeError):
                compilation._cast(module, torch.ones(2, 3))
        self.assertEqual([event[0] for event in self.events], ["cast", "release"])

    def test_native_dispatch_shapes_and_dtypes_keep_compiling_and_restore(self):
        modules = torch.nn.ModuleList(
            [
                nodes.comfy.ops.manual_cast.Linear(size, size).to(dtype)
                for size in range(4, 16)
                for dtype in (torch.float32, torch.bfloat16)
            ]
        )
        for module in modules:
            torch.nn.init.normal_(module.weight)
            torch.nn.init.normal_(module.bias)
        patcher = SimpleNamespace(model=torch.nn.Module())
        patcher.model.diffusion_model = SimpleNamespace(double_blocks=modules)
        originals = {module: module.forward.__func__ for module in modules}
        compilation._inject_casting(patcher)
        functions = []
        try:
            with patch.object(nodes.comfy.ops, "CastBiasWeightContext", self.native_context):
                for module in modules:
                    compiled = torch.compile(module, backend=self.backend)
                    value = torch.randn(2, module.in_features, dtype=module.weight.dtype)
                    expected = torch.nn.functional.linear(value, module.weight, module.bias)
                    before = len(self.graphs)
                    torch.testing.assert_close(compiled(value), expected, rtol=0, atol=0)
                    self.assertGreater(len(self.graphs), before)
                    functions.append((compiled, value, expected))
                cold_graphs = len(self.graphs)
                for compiled, value, expected in functions:
                    torch.testing.assert_close(compiled(value), expected, rtol=0, atol=0)
                self.assertEqual(len(self.graphs), cold_graphs)
                self.assertFalse(
                    torch._dynamo.utils.counters["unimplemented"].get("recompile_limit reached", 0)
                )
        finally:
            compilation._eject_casting(patcher)
        for module in modules:
            self.assertNotIn("forward", module.__dict__)
            self.assertIs(module.forward.__func__, originals[module])

    def test_flash_mask_cache_changes_do_not_guard_compiled_packing(self):
        attention = sys.modules[f"{spec.name}.vimoo_attention"]
        model = tiny_vimoo()

        def forward(value, mask, **kwargs):
            metadata = attention._cached_flash_metadata(
                mask, mask[:, 0], 1, 5, 5, value.device, kwargs["transformer_options"]
            )
            return attention._pack_valid_tensor(value, metadata) + 1

        patcher = nodes.comfy.model_patcher.ModelPatcher(
            torch.nn.Module(), torch.device("cpu"), torch.device("cpu")
        )
        patcher.set_model_optimized_attention(forward)
        compiled = torch.compile(
            patcher.model_options["transformer_options"]["optimized_attention_override"], backend=self.backend
        )
        value = torch.randn(1, 5, 2, 4)
        cold_graphs = None
        for index in range(16):
            mask = (torch.arange(5)[None, None] < (index % 5 + 1)).clone()
            options = {
                "vimoo_attention_metadata": {},
                "vimoo_symmetric_self_attention": True,
                "vimoo_flash_padded_capacity": True,
                "vimoo_math_graph": True,
            }
            # Token refinement fills the eager cache before compiled blocks.
            forward(value, mask, transformer_options=options)
            block_options, _ = model._block_options(options, mask, 5, ())
            self.assertNotIn("vimoo_attention_metadata", block_options)
            torch.testing.assert_close(
                compiled(None, value, mask, transformer_options=block_options),
                forward(value, mask, transformer_options=options),
            )
            if cold_graphs is None:
                cold_graphs = len(self.graphs)
                self.assertGreater(cold_graphs, 0)
            self.assertEqual(len(self.graphs), cold_graphs)
        self.assertFalse(torch._dynamo.utils.counters["unimplemented"].get("recompile_limit reached", 0))

    def test_flash_packing_and_unpacking_keep_graphs_when_prompt_lengths_change(self):
        attention = sys.modules[f"{spec.name}.vimoo_attention"]

        @torch.compiler.disable
        def kernel(value):
            return value

        def forward(value, mask, **kwargs):
            metadata = attention._cached_flash_metadata(
                mask, mask[:, 0], 1, 20, 20, value.device, kwargs["transformer_options"]
            )
            packed = attention._pack_valid_tensor(value, metadata)
            return attention._unpack_valid_tensor(kernel(packed), mask[:, 0], metadata)

        patcher = nodes.comfy.model_patcher.ModelPatcher(
            torch.nn.Module(), torch.device("cpu"), torch.device("cpu")
        )
        patcher.set_model_optimized_attention(forward)
        compiled = torch.compile(
            patcher.model_options["transformer_options"]["optimized_attention_override"],
            backend=self.backend,
            dynamic=False,
        )
        options = {"vimoo_symmetric_self_attention": True, "vimoo_flash_padded_capacity": True}
        value = torch.randn(1, 20, 2, 4)
        cold_graphs = None
        for length in range(1, 17):
            mask = (torch.arange(20)[None, None] < length).clone()
            expected = value.masked_fill(~mask.reshape(1, 20, 1, 1), 0)
            torch.testing.assert_close(
                compiled(None, value, mask, transformer_options=options), expected, rtol=0, atol=0
            )
            if cold_graphs is None:
                cold_graphs = len(self.graphs)
                self.assertGreater(cold_graphs, 0)
            self.assertEqual(len(self.graphs), cold_graphs)
        self.assertFalse(torch._dynamo.utils.counters["unimplemented"].get("recompile_limit reached", 0))

    def test_flash_unpacking_discards_undefined_kernel_capacity_rows(self):
        attention = sys.modules[f"{spec.name}.vimoo_attention"]
        valid = torch.tensor([[True, False, True, False, False], [False, True, False, False, True]])
        value = torch.randn(2, 5, 2, 4)
        metadata = attention._prepare_flash_metadata(
            valid, 2, 5, 5, value.device, symmetric=True, padded_capacity=True
        )
        packed = attention._pack_valid_tensor(value, metadata)
        packed[metadata["indices"] == 10] = float("nan")
        actual = attention._unpack_valid_tensor(packed, valid, metadata)
        torch.testing.assert_close(actual, value.masked_fill(~valid[:, :, None, None], 0), rtol=0, atol=0)

    def test_flash_weak_cache_reuses_metadata_without_retaining_masks(self):
        attention = sys.modules[f"{spec.name}.vimoo_attention"]
        options = {"vimoo_symmetric_self_attention": True, "vimoo_flash_padded_capacity": True}
        mask = torch.ones(1, 1, 5, dtype=torch.bool)
        reference = weakref.ref(mask)
        with patch.object(
            attention, "_prepare_flash_metadata", wraps=attention._prepare_flash_metadata
        ) as prepare:
            metadata = attention._cached_flash_metadata(mask, mask[:, 0], 1, 5, 5, mask.device, options)
            self.assertIs(
                metadata, attention._cached_flash_metadata(mask, mask[:, 0], 1, 5, 5, mask.device, options)
            )
            self.assertEqual(prepare.call_count, 1)
            # Mock call history owns the valid_keys view and its mask base.
            prepare.reset_mock()
        del mask
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(any(entry[1] is metadata for entry in attention._flash_metadata_cache.values()))

    def make_lora_patcher(self, packed=False):
        lora = sys.modules[f"{spec.name}.vimoo_lora"]
        model = torch.nn.Module()
        model.diffusion_model = packed_tiny_copy(tiny_vimoo()) if packed else tiny_vimoo()
        patcher = nodes.comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        path = "diffusion_model.double_blocks.0." + ("img_attn.qkv" if packed else "img_mlp.0")
        base = patcher.get_model_object(path)
        if packed:
            pairs = [
                (torch.randn(2, base.in_features), torch.randn(base.out_features // 3, 2)) for _ in range(3)
            ]
            adapter = lora.PackedLoRALinear(base, pairs)
        else:
            adapter = lora.LoRALinear(
                base, torch.randn(2, base.in_features), torch.randn(base.out_features, 2)
            )
        values = pairs if packed else (adapter.down.weight, adapter.up.weight)
        patcher.add_callback_with_key(
            nodes.comfy.patcher_extension.CallbacksMP.ON_CLONE,
            "vimoo_distilled_lora",
            partial(lora._rebind_lora_modules, adapters={path: values}),
        )
        return patcher, path, adapter

    def test_lora_regional_load_forward_unload_and_reload(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                patcher, path, adapter = self.make_lora_patcher(packed)
                base = patcher.get_model_object(path)
                patcher.add_object_patch(path, adapter)
                value = torch.randn(2, base.in_features)
                expected = adapter(value)
                original_methods = {
                    module: module.forward_comfy_cast_weights.__func__
                    for module in adapter.modules()
                    if hasattr(module, "forward_comfy_cast_weights")
                }
                compilation.configure_regional_casting(patcher, True)
                self.assertEqual(patcher.object_patches, {path: adapter})
                for _ in range(2):
                    try:
                        patcher.patch_model(load_weights=False)
                        torch.testing.assert_close(
                            patcher.get_model_object(path)(value), expected, rtol=0, atol=0
                        )
                        for module in original_methods:
                            self.assertIs(module.forward_comfy_cast_weights.__self__, module)
                            self.assertIsNot(
                                module.forward_comfy_cast_weights.__func__, original_methods[module]
                            )
                    finally:
                        patcher.unpatch_model(unpatch_weights=False)
                    self.assertIs(patcher.model.get_submodule(path), base)
                    self.assertFalse(patcher.object_patches_backup)
                    for module, function in original_methods.items():
                        self.assertIs(module.forward_comfy_cast_weights.__func__, function)

    def test_resident_lora_does_not_contaminate_ordinary_regional_branch(self):
        base, path, adapter = self.make_lora_patcher()
        original = base.get_model_object(path)
        resident = base.clone()
        resident.add_object_patch(path, adapter)
        resident.patch_model(load_weights=False)
        try:
            ordinary = base.clone()
            compilation.configure_regional_casting(ordinary, True)
            self.assertFalse(ordinary.object_patches)
            self.assertIs(resident.model.get_submodule(path), adapter)
        finally:
            resident.unpatch_model(unpatch_weights=False)
        try:
            ordinary.patch_model(load_weights=False)
            value = torch.randn(2, original.in_features)
            torch.testing.assert_close(
                original(value), torch.nn.functional.linear(value, original.weight, original.bias)
            )
        finally:
            ordinary.unpatch_model(unpatch_weights=False)
        self.assertIs(base.model.get_submodule(path), original)
        self.assertFalse(base.object_patches_backup)
        self.assertNotIn("forward_comfy_cast_weights", original.__dict__)

    def test_regional_scope_toggle_reapply_and_fresh_clone(self):
        base, path, adapter = self.make_lora_patcher()
        source = base.clone()
        source.add_object_patch(path, adapter)
        compilation.configure_regional_casting(source, True)
        native_function = nodes.comfy.ops.manual_cast.Linear.forward_comfy_cast_weights
        try:
            source.patch_model(load_weights=False)
            installed = adapter.base.forward_comfy_cast_weights
            compilation.configure_regional_casting(source, True)
            disabled = source.clone()
            compilation.configure_regional_casting(disabled, False)
            self.assertIs(adapter.base.forward_comfy_cast_weights, installed)
            self.assertIs(nodes.comfy.ops.manual_cast.Linear.forward_comfy_cast_weights, native_function)
            # A distinct model must receive methods bound to its own operators.
            fresh, _, _ = self.make_lora_patcher()
            clone = source.clone(model_override=fresh.get_clone_model_override())
            clone.patch_model(load_weights=False)
            owner = clone.model.get_submodule(path).base
            self.assertIs(owner.forward_comfy_cast_weights.__self__, owner)
            clone.unpatch_model(unpatch_weights=False)
            self.assertIs(adapter.base.forward_comfy_cast_weights, installed)
        finally:
            source.unpatch_model(unpatch_weights=False)
        disabled.patch_model(load_weights=False)
        try:
            self.assertIs(adapter.base.forward_comfy_cast_weights.__func__, native_function)
        finally:
            disabled.unpatch_model(unpatch_weights=False)


if __name__ == "__main__":
    unittest.main()
