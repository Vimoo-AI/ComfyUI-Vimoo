"""Text adapter isolation and mixed-precision loading contracts."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import torch
from test_nodes_runtime import spec
import comfy.model_management
import comfy.ops
from comfy.text_encoders import llama, t5

adapters = sys.modules[spec.name + ".vimoo_text"]
encoder_module = sys.modules[spec.name + ".vimoo_encoder"]


class TextTests(unittest.TestCase):
    def test_vimoo_bf16_profile_rounds_glyph_conditioning_and_keeps_other_profiles(self):
        glyph = torch.tensor([1.003, -0.12345], dtype=torch.float32).reshape(1, 1, 2)
        for profile, expected_dtype in (
            ("vimoo_bf16", torch.bfloat16),
            ("standard_i2v", torch.float32),
        ):
            with self.subTest(profile=profile):
                encoder = SimpleNamespace(
                    vimoo_precision_profile=profile,
                    qwen25_7b=SimpleNamespace(
                        encode_token_weights=Mock(
                            return_value=(torch.zeros(1, 1, 2), None, {"attention_mask": torch.ones(1, 1)})
                        )
                    ),
                    byt5_small=SimpleNamespace(encode_token_weights=Mock(return_value=(glyph, None, {}))),
                )
                with patch.object(encoder_module, "crop_start", return_value=0):
                    _, _, extra = encoder_module.VimooTextEncoder.encode_token_weights(
                        encoder, {"qwen25_7b": [[(1, 1.0)]], "byt5": [[(1, 1.0)]]}
                    )
                actual = extra["conditioning_byt5small"]
                self.assertEqual(actual.dtype, expected_dtype)
                torch.testing.assert_close(actual, glyph.to(expected_dtype), rtol=0, atol=0)
                torch.testing.assert_close(extra["vimoo_byt5_mask"], torch.ones(1, 1, dtype=torch.int64))

    def test_standard_profiles_keep_qwen_precision_and_fp32_glyph_strides(self):
        from comfy.model_patcher import ModelPatcher

        for profile, compute_dtype in (("standard_t2v", torch.bfloat16), ("standard_i2v", torch.float16)):
            with self.subTest(profile=profile):
                encoder = self.fixture()
                qwen = encoder.qwen25_7b.transformer.model
                patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
                adapters.patch_vimoo_text(
                    SimpleNamespace(cond_stage_model=encoder, patcher=patcher), "precise", profile
                )
                self.assertTrue(patcher.weight_wrapper_patches)
                for _ in range(2):
                    try:
                        patcher.patch_model(device_to=torch.device("cpu"), load_weights=True)
                        self.assertEqual(qwen.forward.__func__.keywords["compute_dtype"], compute_dtype)
                        self.assertEqual(
                            qwen.compute_freqs_cis.__func__.keywords["compute_dtype"], compute_dtype
                        )
                        self.assertTrue(
                            all(
                                parameter.dtype == torch.float32
                                for parameter in encoder.byt5_small.parameters()
                            )
                        )
                        for name in patcher.weight_wrapper_patches:
                            layer = encoder.get_submodule(name.rpartition(".")[0])
                            with comfy.ops.CastBiasWeightContext(layer, torch.ones(1, layer.in_features)) as (
                                weight,
                                _,
                            ):
                                self.assertEqual(weight.dtype, torch.float32)
                                self.assertEqual(weight.stride(), (1, layer.out_features))
                    finally:
                        patcher.unpatch_model(unpatch_weights=True)

    def fixture(self):
        config = llama.Llama2Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        config.head_dim = 4
        qwen = llama.Llama2_(config, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
        byt5 = t5.T5(
            dict(
                num_layers=2,
                d_model=8,
                d_kv=64,
                num_heads=1,
                d_ff=12,
                dense_act_fn="gelu_pytorch_tanh",
                is_gated_act=True,
                model_type="t5",
                vocab_size=16,
            ),
            torch.float32,
            "cpu",
            comfy.ops.manual_cast,
        )
        encoder = torch.nn.Module()
        encoder.qwen25_7b = torch.nn.Module()
        encoder.qwen25_7b.transformer = torch.nn.Module()
        encoder.qwen25_7b.transformer.model = qwen
        encoder.byt5_small = byt5
        return encoder

    def test_patcher_restores_text_math_and_keeps_native_attention(self):
        from comfy.model_patcher import ModelPatcher

        encoder = self.fixture()
        qwen = encoder.qwen25_7b.transformer.model
        patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
        original_forward = qwen.forward.__func__
        original_norm = qwen.norm.forward.__func__
        original_attention = qwen.layers[0].self_attn.forward.__func__
        parameters = list(encoder.parameters())
        adapters.patch_vimoo_text(SimpleNamespace(cond_stage_model=encoder, patcher=patcher))
        self.assertIs(qwen.norm.forward.__func__, original_norm)
        clone = patcher.clone()
        for _ in range(2):
            try:
                clone.patch_model(load_weights=False)
                self.assertIs(qwen.norm.forward.__func__, adapters._qwen_norm)
                self.assertIs(
                    qwen.layers[0].self_attn.forward.__func__.func,
                    adapters._qwen_attention,
                )
                self.assertIsNone(qwen.layers[0].self_attn.forward.__func__.keywords["attention_backend"])
                self.assertIsInstance(encoder.byt5_small, t5.T5)
            finally:
                clone.unpatch_model(unpatch_weights=False)
            self.assertIs(qwen.layers[0].self_attn.forward.__func__, original_attention)
            self.assertIs(qwen.forward.__func__, original_forward)
            self.assertIs(qwen.norm.forward.__func__, original_norm)
            self.assertTrue(all(a is b for a, b in zip(parameters, encoder.parameters())))

    def test_glyph_weight_layout_survives_first_load_and_reload(self):
        from comfy.model_patcher import ModelPatcher

        encoder = self.fixture()
        patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
        adapters.patch_vimoo_text(SimpleNamespace(cond_stage_model=encoder, patcher=patcher))
        names = [name for name in patcher.weight_wrapper_patches if name.endswith("wi_0.weight")]
        self.assertTrue(names)
        clone = patcher.clone()
        for _ in range(2):
            try:
                clone.patch_model(device_to=torch.device("cpu"), load_weights=True)
                for name in names:
                    layer = encoder.get_submodule(name.rpartition(".")[0])
                    self.assertEqual(layer.weight_function, [adapters._glyph_layout])
                    with comfy.ops.CastBiasWeightContext(layer, torch.ones(1, layer.in_features)) as (
                        weight,
                        _,
                    ):
                        self.assertEqual(weight.stride(), (1, layer.out_features))
            finally:
                clone.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=True)

    def test_patchify_glyph_matches_hf_eager_with_full_mask_and_reloads(self):
        from comfy.model_patcher import ModelPatcher
        from transformers import T5Config, T5EncoderModel

        torch.manual_seed(712)
        source = T5EncoderModel(
            T5Config(
                vocab_size=16,
                d_model=8,
                d_kv=64,
                d_ff=12,
                num_layers=2,
                num_heads=1,
                dropout_rate=0,
                feed_forward_proj="gated-gelu",
                _attn_implementation="eager",
            )
        ).eval()
        encoder = self.fixture()
        native = encoder.byt5_small
        native.load_state_dict(
            {
                key: value
                for key, value in source.state_dict().items()
                if key != "encoder.embed_tokens.weight"
            },
            strict=True,
        )
        inputs = torch.arange(256).reshape(1, -1) % 16
        mask = torch.arange(256).reshape(1, -1) < 21
        with torch.inference_mode():
            expected = source(inputs, attention_mask=mask.float()).last_hidden_state
        patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
        attention = native.encoder.block[0].layer[0].SelfAttention
        original_forward = attention.forward.__func__
        parameters = list(encoder.parameters())
        adapters.patch_vimoo_text(
            SimpleNamespace(cond_stage_model=encoder, patcher=patcher),
            "precise",
            "lite_compact",
        )
        self.assertFalse(patcher.weight_wrapper_patches)
        clone = patcher.clone()
        for _ in range(2):
            try:
                clone.patch_model(device_to=torch.device("cpu"), load_weights=True)
                self.assertIs(attention.forward.__func__, adapters._byt5_eager_attention)
                with torch.inference_mode():
                    # The fixture holds raw T5; the CPU wrapper targets the
                    # production CLIP container surrounding this native model.
                    actual = type(native).forward(native, inputs, attention_mask=mask.float())[0]
                self.assertTrue(
                    torch.equal(
                        actual.contiguous().view(torch.uint8),
                        expected.view(torch.uint8),
                    )
                )
                for block in native.encoder.block:
                    for name in ("wi_0", "wi_1", "wo"):
                        layer = getattr(block.layer[1].DenseReluDense, name)
                        with comfy.ops.CastBiasWeightContext(layer, torch.ones(1, layer.in_features)) as (
                            weight,
                            _,
                        ):
                            self.assertEqual(weight.stride(), (layer.in_features, 1))
            finally:
                clone.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=True)
            self.assertIs(attention.forward.__func__, original_forward)
            self.assertTrue(all(a is b for a, b in zip(parameters, encoder.parameters())))

    def test_lite_glyph_keeps_selected_sdpa_and_layout(self):
        from comfy.model_patcher import ModelPatcher

        encoder = self.fixture()
        patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
        attention = encoder.byt5_small.encoder.block[0].layer[0].SelfAttention
        adapters.patch_vimoo_text(
            SimpleNamespace(cond_stage_model=encoder, patcher=patcher),
            "precise",
            "lite_ti2v_anchor",
        )
        try:
            patcher.patch_model(load_weights=False)
            self.assertIs(attention.forward.__func__.func, adapters._byt5_attention)
            self.assertTrue(patcher.weight_wrapper_patches)
        finally:
            patcher.unpatch_model(unpatch_weights=False)

    def test_qwen_math_preserves_selected_attention_and_cache_contract(self):
        module = self.fixture().qwen25_7b.transformer.model.layers[0].self_attn
        for parameter in module.parameters():
            parameter.data.normal_(0, 0.1)
        x = torch.randn(1, 3, 8).bfloat16()
        cos = torch.full((1, 1, 3, 4), 0.7).bfloat16()
        sin = torch.full((1, 1, 3, 2), 0.3).bfloat16()
        mask = torch.zeros(3, 3)
        backend = Mock(return_value=torch.zeros_like(x))
        adapters._qwen_attention(module, x, mask, (cos, sin, -sin), backend)
        backend.assert_called_once()
        self.assertIs(backend.call_args.kwargs["mask"], mask)
        self.assertTrue(backend.call_args.kwargs["enable_gqa"])
        q = module.q_proj(x).view(1, 3, 2, 4).transpose(1, 2)
        expected = q * cos + torch.cat((-q[..., 2:], q[..., :2]), -1) * torch.cat((sin, sin), -1)
        torch.testing.assert_close(backend.call_args.args[0], expected, rtol=0, atol=0)
        cache = (torch.zeros(1),)
        with patch.object(type(module), "forward", return_value=(x, cache)) as original:
            result = adapters._qwen_attention(module, x, mask, (cos, sin, -sin), backend, cache)
        self.assertIs(result[1], cache)
        self.assertIs(original.call_args.args[-2], cache)

    def test_bf16_adapter_preserves_bias_even_with_zero_last_row(self):
        module = self.fixture().qwen25_7b.transformer.model.layers[0].self_attn
        for parameter in module.parameters():
            parameter.data.zero_()
        x = torch.zeros(1, 3, 8).bfloat16()
        freqs = (
            torch.ones(1, 1, 3, 4),
            torch.zeros(1, 1, 3, 2),
            torch.zeros(1, 1, 3, 2),
        )
        backend = Mock(return_value=x)
        mask = torch.tensor([[0.0, -2.0, 0.0], [0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0]])
        adapters._qwen_attention(module, x, mask, freqs, attention_backend=backend)
        self.assertIs(backend.call_args.kwargs["mask"], mask)
        self.assertNotIn("is_causal", backend.call_args.kwargs)
        backend.reset_mock()
        causal = torch.full((3, 3), torch.finfo(torch.bfloat16).min / 4, dtype=torch.bfloat16).triu(1)
        adapters._qwen_attention(module, x, causal, freqs, attention_backend=backend)
        self.assertIsNone(backend.call_args.kwargs["mask"])
        self.assertTrue(backend.call_args.kwargs["is_causal"])

    def test_sdpa_backend_policy_is_active_at_sdpa_and_restored(self):
        attention = sys.modules[spec.name + ".vimoo_attention"]
        q = torch.zeros(1, 3, 8)
        before = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        for function, expected in (
            (attention.siglip_attention, (False, False, True)),
            (attention.byt5_attention, (False, True, False)),
            (attention.byt5_cpu_attention, (True, True, False)),
        ):

            def execute(q, k, v, **kwargs):
                self.assertEqual(
                    (
                        torch.backends.cuda.flash_sdp_enabled(),
                        torch.backends.cuda.math_sdp_enabled(),
                        torch.backends.cuda.cudnn_sdp_enabled(),
                    ),
                    expected,
                )
                raise RuntimeError("interrupted")

            with patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    function(q, q, q, 2)
            self.assertEqual(
                (
                    torch.backends.cuda.flash_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled(),
                    torch.backends.cuda.cudnn_sdp_enabled(),
                ),
                before,
            )

    def test_qwen_backend_keeps_noncausal_masks(self):
        attention = sys.modules[spec.name + ".vimoo_attention"]
        q, k, v = [torch.randn(1, 2, 3, 4) for _ in range(3)]
        mask = torch.tensor([[0.0, -2.0, 0.0], [0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0]])
        expected = attention.attention_pytorch(q, k, v, 2, mask=mask, skip_reshape=True)
        actual = attention.qwen_attention(q, k, v, 2, mask=mask, skip_reshape=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_qwen_keeps_explicit_single_query_causality(self):
        attention = sys.modules[spec.name + ".vimoo_attention"]
        q = torch.zeros(2, 2, 1, 4)
        k = torch.zeros(2, 2, 3, 4)
        v = torch.arange(48).reshape(2, 2, 3, 4).float()
        actual = attention.qwen_attention(
            q, k, v, 2, skip_reshape=True, skip_output_reshape=True, is_causal=True
        )
        torch.testing.assert_close(actual, v[:, :, :1], rtol=0, atol=0)

    def test_byt5_fp32_is_archived_before_state_dict_loading(self):
        encoder = encoder_module.VimooTextEncoder(device="meta", dtype=torch.bfloat16)
        comfy.model_management.archive_model_dtypes(encoder)
        self.assertEqual(encoder.dtypes, {torch.float32, torch.bfloat16})
        for module in encoder.byt5_small.modules():
            for name, param in module.named_parameters(recurse=False):
                self.assertEqual(param.dtype, torch.float32)
                self.assertEqual(getattr(module, name + "_comfy_model_dtype"), torch.float32)
        self.assertEqual(next(encoder.qwen25_7b.transformer.model.parameters()).dtype, torch.bfloat16)

    def test_glyph_parentheses_are_literal_in_both_profiles(self):
        tokenizer = encoder_module.VimooTokenizer()
        prompt = 'A poster reads "Hello (world)".'
        before = tokenizer.tokenize_with_weights(prompt)
        self.assertTrue(all(token[1] == 1.0 for row in before["byt5"] for token in row))
        rows = tokenizer.tokenize_with_weights(prompt)["byt5"]
        self.assertTrue(all(token[1] == 1.0 for row in rows for token in row))
        self.assertEqual(before, tokenizer.tokenize_with_weights(prompt))

    def test_long_glyph_retains_huggingface_eos_on_truncation(self):
        tokenizer = encoder_module.VimooTokenizer()
        glyph = "Vimoo " * 60
        actual = tokenizer.tokenize_with_weights(f'A sign reads "{glyph}".')["byt5"][0]
        expected = tokenizer.byt5.tokenizer(
            f'Text "{glyph}". ', truncation=True, max_length=256, padding="max_length"
        )["input_ids"]
        self.assertEqual([token[0] for token in actual], expected)


class ScheduledTextTests(unittest.TestCase):
    def test_clip_hooks_preserve_conditioning_or_raise(self):
        import comfy.hooks
        import comfy.sd
        from comfy_extras.nodes_hooks import SetClipHooks
        from test_nodes_runtime import nodes

        class Tokenizer:
            def __init__(self, **kwargs):
                pass

            def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
                return {
                    "qwen25_7b": [[(2, 1.0), (3, 1.0), (151643, 1.0)]],
                    "byt5": [[(4, 1.0), (5, 1.0), (1, 1.0), (0, 1.0)]],
                }

        options = {
            "dtype": torch.bfloat16,
            "load_device": torch.device("cpu"),
            "offload_device": torch.device("cpu"),
            "qwen25_7b_model_config": dict(
                hidden_size=128,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=1,
                num_key_value_heads=1,
                vocab_size=151644,
            ),
            "_model_config": dict(num_layers=2, d_model=8, d_kv=8, num_heads=1, d_ff=12, vocab_size=16),
        }
        clip = comfy.sd.CLIP(
            SimpleNamespace(params={}, clip=encoder_module.VimooTextEncoder, tokenizer=Tokenizer),
            model_options=options,
            disable_dynamic=True,
        )
        with torch.no_grad():
            for parameter in clip.cond_stage_model.parameters():
                parameter.normal_(0, 0.1)
        adapters.patch_vimoo_text(clip)
        plain = nodes.VimooTextEncode().encode(clip, 'render "Vimoo"')[0]
        required = ("attention_mask", "conditioning_byt5small", "vimoo_byt5_mask")

        def assert_matches_plain(conditioning):
            self.assertEqual(len(conditioning), 1)
            torch.testing.assert_close(conditioning[0][0], plain[0][0], rtol=0, atol=0)
            for key in required:
                torch.testing.assert_close(conditioning[0][1][key], plain[0][1][key], rtol=0, atol=0)

        hook = comfy.hooks.WeightHook()
        hook.weights = {}
        group = comfy.hooks.HookGroup()
        group.add(hook)
        for schedule_clip, hooks in ((False, None), (True, None), (False, group)):
            with self.subTest(schedule_clip=schedule_clip, has_hooks=hooks is not None):
                hooked_clip = SetClipHooks().apply_hooks(
                    clip, schedule_clip=schedule_clip, apply_to_conds=False, hooks=hooks
                )[0]
                assert_matches_plain(nodes.VimooTextEncode().encode(hooked_clip, 'render "Vimoo"')[0])

        scheduled_clip = SetClipHooks().apply_hooks(
            clip, schedule_clip=True, apply_to_conds=False, hooks=group
        )[0]
        native = scheduled_clip.encode_from_tokens_scheduled(scheduled_clip.tokenize('render "Vimoo"'))
        self.assertEqual(len(native), 1)
        # Stock ComfyUI currently drops encoder extras here. Also accept an
        # upstream fix, without making the plugin depend on a patched core.
        if all(key in native[0][1] for key in required):
            scheduled = nodes.VimooTextEncode().encode(scheduled_clip, 'render "Vimoo"')[0]
            assert_matches_plain(scheduled)
            self.assertEqual(scheduled[0][1]["clip_start_percent"], 0.0)
            self.assertEqual(scheduled[0][1]["clip_end_percent"], 1.0)
        else:
            with self.assertRaisesRegex(ValueError, "disable schedule_clip"):
                nodes.VimooTextEncode().encode(scheduled_clip, 'render "Vimoo"')
        self.assertTrue(scheduled_clip.use_clip_schedule)
        self.assertIsNotNone(scheduled_clip.patcher.forced_hooks)

    def test_missing_extras_raise_before_sampling(self):
        from test_nodes_runtime import nodes

        clip = Mock()
        clip.encode_from_tokens_scheduled.return_value = [[torch.ones(1, 2, 4), {}]]
        with self.assertRaisesRegex(ValueError, "schedule_clip"):
            nodes.VimooTextEncode().encode(clip, "test")


if __name__ == "__main__":
    unittest.main()
