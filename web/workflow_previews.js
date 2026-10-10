import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Vimoo.WorkflowPreviews",
    init() {
        const getCoreTemplates = api.getCoreWorkflowTemplates.bind(api);
        const getCustomTemplates = api.getWorkflowTemplates.bind(api);
        let previews;

        const loadPreviews = () => {
            previews ??= fetch(new URL("./workflow_previews.json", import.meta.url)).then(async (response) => {
                if (!response.ok) {
                    throw new Error(`Unable to load Vimoo workflow previews: ${response.status}`);
                }
                return response.json();
            });
            return previews;
        };

        api.getCoreWorkflowTemplates = async (...args) => {
            const [categories, vimoo] = await Promise.all([getCoreTemplates(...args), loadPreviews()]);
            return [...categories, vimoo];
        };

        api.getWorkflowTemplates = async (...args) => {
            const templates = await getCustomTemplates(...args);
            delete templates["ComfyUI-Vimoo"];
            return templates;
        };
    },
});
