"""Check saved graph integrity and template thumbnail validity."""

import json
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def check_workflow(self, workflow):
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {
            (e["id"] if isinstance(e, dict) else e[0]): (
                [e["id"], e["origin_id"], e["origin_slot"], e["target_id"], e["target_slot"], e["type"]]
                if isinstance(e, dict)
                else e
            )
            for e in workflow["links"]
        }
        self.assertEqual(len(nodes), len(workflow["nodes"]))
        self.assertEqual(len(links), len(workflow["links"]))
        state = workflow.get("state", {})
        self.assertGreaterEqual(workflow.get("last_node_id", state.get("lastNodeId")), max(nodes))
        self.assertGreaterEqual(workflow.get("last_link_id", state.get("lastLinkId")), max(links))
        if "inputNode" in workflow:
            nodes[-10] = {
                "id": -10,
                "outputs": [{"type": p["type"], "links": p["linkIds"]} for p in workflow["inputs"]],
                "properties": {},
                "type": "boundary",
            }
            nodes[-20] = {
                "id": -20,
                "inputs": [{"type": p["type"], "link": p["linkIds"][0]} for p in workflow["outputs"]],
                "properties": {},
                "type": "boundary",
            }
        for lid, source, slot, target, target_slot, type_ in links.values():
            output = nodes[source]["outputs"][slot]
            input_ = nodes[target]["inputs"][target_slot]
            self.assertIn(lid, output["links"])
            self.assertEqual(input_["link"], lid)
            self.assertEqual(output["type"], type_)
            self.assertEqual(input_["type"], type_)
        for node in nodes.values():
            for output in node.get("outputs", []):
                for lid in output.get("links") or []:
                    self.assertEqual(links[lid][1], node["id"])
            for input_ in node.get("inputs", []):
                if input_.get("link") is not None:
                    self.assertEqual(links[input_["link"]][3], node["id"])
            self.assertIn("properties", node)

    def test_saved_examples(self):
        for path in (ROOT / "example_workflows").rglob("*.json"):
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text())
                self.check_workflow(workflow)
                for graph in workflow.get("definitions", {}).get("subgraphs", []):
                    self.check_workflow(graph)

    def test_saved_examples_have_square_animated_previews(self):
        directory = ROOT / "example_workflows"
        stems = {path.stem for path in directory.glob("*.json")}
        self.assertEqual(stems, {path.stem for path in directory.glob("*.webp")})
        for stem in stems:
            with self.subTest(workflow=stem):
                with Image.open(directory / f"{stem}.webp") as animation:
                    self.assertEqual(animation.format, "WEBP")
                    self.assertEqual(animation.size, (512, 512))
                    self.assertTrue(animation.is_animated)
                    self.assertGreaterEqual(animation.n_frames, 40)
                    self.assertEqual(animation.info["loop"], 0)
                    first = animation.convert("RGB")
                    animation.seek(animation.n_frames // 2)
                    difference = ImageChops.difference(first, animation.convert("RGB"))
                    self.assertGreater(max(ImageStat.Stat(difference).mean), 0.5)

    def test_preview_catalog_matches_saved_prompts_and_reference_images(self):
        catalog = json.loads((ROOT / "web" / "workflow_previews.json").read_text())
        self.assertEqual(catalog["moduleName"], "ComfyUI-Vimoo")
        entries = {entry["name"]: entry for entry in catalog["templates"]}
        self.assertEqual(len(entries), len(catalog["templates"]))
        self.assertEqual(set(entries), {p.stem for p in (ROOT / "example_workflows").glob("*.json")})
        references = []
        for stem, entry in entries.items():
            with self.subTest(workflow=stem):
                self.assertEqual(entry["mediaSubtype"], "webp")
                workflow = json.loads((ROOT / "example_workflows" / f"{stem}.json").read_text())
                graphs = [workflow, *workflow.get("definitions", {}).get("subgraphs", [])]
                prompts = []
                inputs = []
                for graph in graphs:
                    for node in graph["nodes"]:
                        values = node.get("widgets_values", [])
                        if node["type"] == "VimooTextEncode" and values[1]:
                            prompts.append(values[0])
                        elif node["type"] == "VimooTI2VTextEncodePair":
                            prompts.append(values[0])
                        elif (
                            node["type"] in ("VimooTI2VInput", "VimooDistilledTI2VInput")
                            and values[1] != "none"
                        ):
                            inputs.append(values[1])
                        elif node["type"] == "VimooImageLoader":
                            inputs.append(values[0])
                self.assertTrue(prompts)
                self.assertEqual(set(prompts), {entry["prompt"]})
                reference = entry["referenceImage"]
                needs_reference = any(
                    node["type"] == "VimooImageLoader"
                    or (
                        node["type"] in ("VimooTI2VInput", "VimooDistilledTI2VInput")
                        and node["widgets_values"][0] == "i2v"
                    )
                    for graph in graphs
                    for node in graph["nodes"]
                )
                self.assertEqual(inputs, [reference] if needs_reference else [])
                if reference:
                    references.append(reference)
                    with Image.open(ROOT / "assets" / reference) as image:
                        self.assertEqual(image.size, (1280, 720))
                        square = ImageOps.fit(image.convert("RGB"), (512, 512), Image.Resampling.LANCZOS)
                    with Image.open(ROOT / "example_workflows" / f"{stem}.webp") as animation:
                        difference = ImageChops.difference(square, animation.convert("RGB"))
                        self.assertLess(max(ImageStat.Stat(difference).mean), 6)
        self.assertEqual(len(references), 3)
        self.assertEqual(set(references), {p.name for p in (ROOT / "assets").glob("*.png")})

    def test_compact_templates_and_generation_controls(self):
        for path in (ROOT / "example_workflows").glob("*.json"):
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text())
                self.assertIn(len(workflow["nodes"]), (4, 5))
                graph = workflow["definitions"]["subgraphs"][0]
                instance = next(n for n in workflow["nodes"] if n["type"] == graph["id"])
                controls = {p["name"] for p in instance["inputs"] if "widget" in p}
                self.assertTrue({"prompt", "resolution", "width", "height", "length", "fps"} <= controls)
                self.assertEqual("steps" in controls, "distilled" not in path.stem)
                self.assertEqual("seed" in controls, path.stem != "vimoo_v1_ti2v_lite_distilled")
                self.assertEqual(sum(n["type"] == "VimooResolution" for n in graph["nodes"]), 1)
                performance = next(n for n in workflow["nodes"] if n["type"] == "VimooPerformanceOptions")
                self.assertEqual(performance["widgets_values"], ["pytorch", False])
                resolver = next(n for n in graph["nodes"] if n["type"] == "VimooResolution")
                for node in graph["nodes"]:
                    if node["type"] in (
                        "VimooTI2VConditioning",
                        "VimooImageToVideo",
                        "VimooEmptyLatentVideo",
                    ):
                        for name, slot in (("width", 0), ("height", 1)):
                            port = next(p for p in node["inputs"] if p["name"] == name)
                            edge = next(e for e in graph["links"] if e["id"] == port["link"])
                            self.assertEqual((edge["origin_id"], edge["origin_slot"]), (resolver["id"], slot))

    def test_connected_sampling_widgets_keep_scalar_saved_values(self):
        for path in (ROOT / "example_workflows").glob("*.json"):
            graph = json.loads(path.read_text())["definitions"]["subgraphs"][0]
            for node in graph["nodes"]:
                if node["type"] in (
                    "VimooSampler",
                    "VimooLiteDistilledSampler",
                    "VimooTI2VSchedule",
                    "VimooTI2VConditioning",
                ):
                    with self.subTest(workflow=path.name, node=node["type"]):
                        self.assertFalse(
                            any(isinstance(value, (list, dict)) for value in node["widgets_values"])
                        )

    def test_distilled_examples_need_one_prompt_and_no_negative_branch(self):
        for path in (ROOT / "example_workflows").glob("*distilled*.json"):
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text())
                all_nodes = [*workflow["nodes"], *workflow["definitions"]["subgraphs"][0]["nodes"]]
                text_nodes = [
                    node
                    for node in all_nodes
                    if node["type"] in ("VimooTextEncode", "VimooTI2VTextEncodePair")
                ]
                self.assertEqual(len(text_nodes), 1)
                self.assertEqual(text_nodes[0]["type"], "VimooTextEncode")
                expected_length = 1400 if "ti2v" in path.stem else 1000
                self.assertEqual(text_nodes[0]["widgets_values"][2], expected_length)
                for node in all_nodes:
                    self.assertFalse(any(port["name"] == "negative" for port in node.get("inputs", [])))

                if path.stem == "vimoo_v1_ti2v_lite_distilled":
                    samplers = [node for node in all_nodes if node["type"] == "VimooLiteDistilledSampler"]
                    self.assertEqual(len(samplers), 1)
                    self.assertFalse(any(node["type"] == "VimooSampler" for node in all_nodes))
                    self.assertFalse(
                        any(
                            port["name"] in ("seed", "renoise_seed", "add_fps")
                            for port in samplers[0]["inputs"]
                        )
                    )
                    self.assertNotIn("fixed", samplers[0]["widgets_values"])
                    image_nodes = [
                        node
                        for node in all_nodes
                        if node["type"] in ("VimooTI2VInput", "VimooDistilledTI2VInput")
                    ]
                    self.assertEqual(len(image_nodes), 1)
                    self.assertEqual(image_nodes[0]["widgets_values"], ["t2v", "none"])
                    self.assertEqual(sum(node["type"] == "VimooTI2VConditioning" for node in all_nodes), 1)
                    self.assertEqual(sum(node["type"] == "VimooTI2VSchedule" for node in all_nodes), 1)
                    self.assertFalse(any(node["type"] == "VimooImageLoader" for node in all_nodes))
                elif path.stem == "vimoo_v1_t2v_distilled":
                    self.assertFalse(
                        any(
                            node["type"]
                            in (
                                "VimooImageLoader",
                                "VimooTI2VInput",
                                "VimooTI2VConditioning",
                                "CLIPVisionLoader",
                            )
                            for node in all_nodes
                        )
                    )
                else:
                    image_nodes = [node for node in all_nodes if node["type"] == "VimooImageLoader"]
                    self.assertEqual(len(image_nodes), 1)
                    self.assertEqual(len(image_nodes[0]["widgets_values"]), 1)
                    self.assertFalse(
                        any(
                            node["type"] in ("VimooTI2VInput", "VimooDistilledTI2VInput")
                            for node in all_nodes
                        )
                    )

    def test_lite_distilled_has_independent_model_bound_output_vae(self):
        workflow = json.loads((ROOT / "example_workflows" / "vimoo_v1_ti2v_lite_distilled.json").read_text())
        graph = workflow["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in graph["nodes"]}
        links = {edge["id"]: edge for edge in graph["links"]}

        def source(node, input_name):
            port = next(port for port in node["inputs"] if port["name"] == input_name)
            return nodes[links[port["link"]]["origin_id"]]

        conditioning = next(node for node in nodes.values() if node["type"] == "VimooTI2VConditioning")
        decoder = next(node for node in nodes.values() if node["type"] == "VAEDecode")
        reference_vae, output_vae = source(conditioning, "vae"), source(decoder, "vae")
        self.assertNotEqual(reference_vae["id"], output_vae["id"])
        self.assertEqual(reference_vae["type"], "VimooReferenceVAELoader")
        self.assertEqual(
            output_vae["widgets_values"],
            [reference_vae["widgets_values"][0], True],
        )
        self.assertEqual(source(reference_vae, "model"), source(output_vae, "model"))
        self.assertEqual(source(output_vae, "model")["type"], "VimooDistilledLoRALoader")


if __name__ == "__main__":
    unittest.main()
