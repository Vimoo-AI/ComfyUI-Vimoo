/* Exercise the shipped extension against ComfyUI's template-loading API. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const vm = require("node:vm");

async function main() {
    const root = path.resolve(__dirname, "..");
    const manifest = JSON.parse(await fs.readFile(path.join(root, "web/workflow_previews.json"), "utf8"));
    let extension;
    let fetchCount = 0;
    const languages = [];
    const app = { registerExtension(value) { extension = value; } };
    const api = {
        async getCoreWorkflowTemplates(language) {
            languages.push(language);
            return [{ moduleName: "default", title: "Video", templates: [{ name: "wan" }] }];
        },
        async getWorkflowTemplates() {
            return { "ComfyUI-Vimoo": ["vimoo_v1_t2v"], OtherNodes: ["other_workflow"] };
        },
    };
    const context = vm.createContext({
        URL,
        async fetch(url) {
            assert.equal(url.href, "http://localhost/proxy/extensions/ComfyUI-Vimoo/workflow_previews.json");
            fetchCount++;
            return { ok: true, async json() { return manifest; } };
        },
    });
    const source = await fs.readFile(path.join(root, "web/workflow_previews.js"), "utf8");
    const module = new vm.SourceTextModule(source, {
        context,
        initializeImportMeta(meta) {
            meta.url = "http://localhost/proxy/extensions/ComfyUI-Vimoo/workflow_previews.js";
        },
    });
    await module.link(async (specifier) => {
        const value = specifier.endsWith("app.js") ? { app } : { api };
        return new vm.SyntheticModule(Object.keys(value), function () {
            for (const [name, item] of Object.entries(value)) this.setExport(name, item);
        }, { context });
    });
    await module.evaluate();
    extension.init();

    const custom = await api.getWorkflowTemplates();
    assert.deepEqual(custom, { OtherNodes: ["other_workflow"] });
    const [english, chinese] = await Promise.all([
        api.getCoreWorkflowTemplates("en"), api.getCoreWorkflowTemplates("zh"),
    ]);
    assert.deepEqual(languages, ["en", "zh"]);
    assert.equal(fetchCount, 1);
    for (const categories of [english, chinese]) {
        assert.equal(categories.length, 2);
        assert.equal(categories[0].templates[0].name, "wan");
        assert.equal(categories[1], manifest);
    }
    assert.equal(manifest.moduleName, "ComfyUI-Vimoo");
    assert.equal(manifest.templates.length, 6);
    for (const template of manifest.templates) {
        assert.equal(template.mediaType, "image");
        assert.equal(template.mediaSubtype, "webp");
        const workflow = path.join(root, "example_workflows", `${template.name}.json`);
        const preview = path.join(root, "example_workflows", `${template.name}.webp`);
        await fs.access(workflow);
        await fs.access(preview);
        const thumbnailUrl = `/api/workflow_templates/${manifest.moduleName}/${template.name}.${template.mediaSubtype}`;
        assert.ok(thumbnailUrl.endsWith(`${template.name}.webp`));
    }
    console.log("Vimoo animated template registration passed.");
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
