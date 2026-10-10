import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import { configure } from "./generation_controls_state.js";

app.registerExtension({
    name: "Vimoo.GenerationControls",
    nodeCreated(node) {
        queueMicrotask(() => configure(node));
    },
    loadedGraphNode(node) {
        queueMicrotask(() => configure(node));
    },
    afterConfigureGraph() {
        for (const node of app.graph.nodes ?? app.graph._nodes ?? []) configure(node);
    },
    setup() {
        queueMicrotask(() => {
            for (const node of app.rootGraph?.nodes ?? app.graph.nodes ?? []) configure(node);
        });
        api.addEventListener("executed", ({ detail }) => {
            const size = detail.output?.vimoo_resolution;
            if (!size) return;
            const displayId = String(detail.display_node ?? detail.node).split(":")[0];
            const node = app.graph.getNodeById(displayId);
            if (!node) return;
            node.properties.vimoo_base_title ??= node.title;
            node.title = `${node.properties.vimoo_base_title} · ${size[0]}×${size[1]}`;
            app.graph.setDirtyCanvas(true, true);
        });
    },
});