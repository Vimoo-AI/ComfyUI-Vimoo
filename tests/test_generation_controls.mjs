import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The state module has no app imports and runs directly in Node as well as the browser.
const source = await readFile(new URL("../web/generation_controls_state.js", import.meta.url), "utf8");
const { configure } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

function fixture() {
    const original = function () { this.originalCalls = (this.originalCalls ?? 0) + 1; return "original"; };
    const task = { name: "task", value: "t2v", callback: original };
    const otherTask = { name: "task", value: "i2v", callback: original };
    const graph = {
        links: { 1: { origin_id: 1 }, 2: { origin_id: 2 } },
        getNodeById(id) { return { widgets: [id === 1 ? task : otherTask] }; },
        setDirtyCanvas() {},
    };
    const node = {
        properties: { vimoo_generation: true },
        graph,
        inputs: [{ name: "task", link: 1 }],
        widgets: [
            { name: "resolution", value: "auto", callback: original },
            { name: "width", value: 1232 }, { name: "height", value: 688 },
            { name: "t2v_cfg", value: 4 }, { name: "i2v_cfg", value: 5 },
        ],
        size: [490, 610],
        computeSize() { return [490, 300]; },
        setSize() { this.resizes = (this.resizes ?? 0) + 1; },
    };
    return { node, task, otherTask, original, graph };
}

function change(widget, value) {
    widget.value = value;
    return widget.callback?.call(widget, value);
}

const flush = () => new Promise(resolve => queueMicrotask(resolve));

test("repeated configuration keeps one subscription and preserves original callbacks", () => {
    const { node, task } = fixture();
    configure(node); configure(node); configure(node);
    node.resizes = 0;
    assert.equal(change(task, "i2v"), "original");
    assert.equal(node.resizes, 1);
    assert.equal(task.originalCalls, 1);
    assert.equal(node.widgets.find(w => w.name === "t2v_cfg").hidden, true);
    assert.equal(node.widgets.find(w => w.name === "i2v_cfg").hidden, false);
});

test("rewiring and disconnecting release the previous task subscription", async () => {
    const { node, task, otherTask, original } = fixture();
    configure(node);
    node.inputs[0].link = 2;
    node.onConnectionsChange(); await flush();
    assert.equal(task.callback, original);
    node.resizes = 0;
    change(task, "i2v"); assert.equal(node.resizes, 0);
    change(otherTask, "t2v"); assert.equal(node.resizes, 1);
    node.inputs[0].link = null;
    node.onConnectionsChange(); await flush();
    assert.equal(otherTask.callback, original);
    node.resizes = 0;
    change(otherTask, "i2v"); assert.equal(node.resizes, 0);
});

test("removing one of two consumers leaves only the live consumer subscribed", () => {
    const { node, task, original } = fixture();
    const second = { ...node, widgets: node.widgets.map(w => ({ ...w })) };
    configure(node); configure(second);
    node.onRemoved(); node.graph = null;
    node.resizes = 0; second.resizes = 0;
    change(task, "i2v");
    assert.equal(node.resizes, 0); assert.equal(second.resizes, 1);
    second.onRemoved(); second.graph = null;
    assert.equal(task.callback, original);
});

test("remove and re-add bind once and preserve lifecycle callbacks", async () => {
    const { node, task, graph } = fixture();
    let removed = 0; let added = 0;
    node.onRemoved = () => { removed++; return "removed"; };
    node.onAdded = () => { added++; return "added"; };
    configure(node);
    assert.equal(node.onRemoved(), "removed"); node.graph = null;
    node.resizes = 0; change(task, "i2v"); assert.equal(node.resizes, 0);
    node.graph = graph;
    assert.equal(node.onAdded(), "added"); await flush(); configure(node);
    node.resizes = 0; change(task, "t2v"); assert.equal(node.resizes, 1);
    assert.equal(removed, 1); assert.equal(added, 1);
});

test("manual size values survive visibility changes", () => {
    const { node } = fixture(); configure(node);
    const resolution = node.widgets[0]; const width = node.widgets[1]; const height = node.widgets[2];
    assert.equal(width.hidden, true); assert.equal(height.hidden, true);
    change(resolution, "manual"); assert.equal(width.hidden, false); assert.equal(height.hidden, false);
    change(resolution, "auto"); assert.equal(width.hidden, true); assert.equal(height.hidden, true);
    assert.equal(width.value, 1232); assert.equal(height.value, 688);
});

test("recreated native widget projections are hidden and old callbacks are detached", async () => {
    const { node } = fixture(); const raw = node.widgets; let projected;
    delete node.widgets; node.isSubgraphNode = () => true;
    Object.defineProperty(node, "widgets", { configurable: true, get() {
        return projected ??= raw.map(widget => ({ ...widget, options: {} }));
    } });
    configure(node); await flush();
    const previous = node.widgets; projected = null;
    const current = node.widgets;
    assert.notEqual(current, previous); assert.equal(current[1].hidden, true);
    await flush(); node.resizes = 0;
    change(previous[0], "manual"); assert.equal(node.resizes, 0);
    change(current[0], "manual"); assert.equal(current[1].hidden, false);
    await flush(); assert.equal(node.resizes, 1);
    node.onRemoved(); node.graph = null; node.resizes = 0;
    await flush(); assert.equal(node.resizes, 0);
});

test("T2V ignores a saved missing image and valid uploads survive task changes", () => {
    for (const comfyClass of ["VimooTI2VInput", "VimooDistilledTI2VInput"]) {
        const { node, original } = fixture();
        const task = { name: "task", value: "t2v", callback: original };
        const image = { name: "image", value: "missing-reference.png", callback: original,
            options: { values: ["none", "uploaded.png"] } };
        Object.assign(node, { comfyClass, properties: {}, inputs: [], widgets: [task, image] });
        configure(node); configure(node);
        assert.equal(image.value, "none");
        assert.equal(image.originalCalls, 1);
        change(task, "i2v");
        assert.equal(image.value, "none");
        change(image, "uploaded.png");
        change(task, "t2v");
        assert.equal(image.value, "uploaded.png");
        change(task, "i2v");
        assert.equal(image.value, "uploaded.png");
        change(image, "none");
        assert.equal(image.value, "none");
        change(task, "t2v"); change(task, "i2v");
        assert.equal(image.value, "none");
        node.onRemoved(); node.graph = null;
        assert.equal(task.callback, original);
        assert.equal(image.callback, original);
    }
});

test("uploading an available image in T2V retains the selection without changing the task", () => {
    const { node, original } = fixture();
    const task = { name: "task", value: "t2v", callback: original };
    const image = { name: "image", value: "none", callback: original,
        options: { values: () => ["none", "uploaded.png"] } };
    Object.assign(node, { comfyClass: "VimooDistilledTI2VInput", properties: {}, inputs: [], widgets: [task, image] });
    configure(node);
    change(image, "uploaded.png");
    assert.equal(image.value, "uploaded.png");
    assert.equal(task.value, "t2v");
    change(task, "i2v");
    assert.equal(image.value, "uploaded.png");
});
