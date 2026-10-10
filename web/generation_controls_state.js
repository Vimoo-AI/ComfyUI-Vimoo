// Node state is private to this extension and does not enter workflow serialization.
const states = new WeakMap();
const subscriptions = new WeakMap();

function subscribe(widget, refresh) {
    if (!widget) return;
    let entry = subscriptions.get(widget);
    if (!entry) {
        const original = widget.callback;
        entry = { original, listeners: new Set() };
        entry.callback = function (...args) {
            const result = original?.apply(this, args);
            for (const listener of [...entry.listeners]) {
                if (entry.listeners.has(listener)) listener();
            }
            return result;
        };
        subscriptions.set(widget, entry);
        widget.callback = entry.callback;
    }
    entry.listeners.add(refresh);
}

function unsubscribe(widget, refresh) {
    const entry = widget && subscriptions.get(widget);
    if (!entry) return;
    entry.listeners.delete(refresh);
    if (entry.listeners.size === 0) {
        if (widget.callback === entry.callback) widget.callback = entry.original;
        subscriptions.delete(widget);
    }
}

function taskWidget(node) {
    const port = node.inputs?.find((input) => input.name === "task");
    const link = port?.link != null ? node.graph?.links?.[port.link] : null;
    return link ? node.graph?.getNodeById(link.origin_id)?.widgets?.find((widget) => widget.name === "task") : null;
}

function bindWidget(state, key, widget) {
    if (state[key] === widget) return;
    unsubscribe(state[key], state.refresh);
    state[key] = widget;
    subscribe(widget, state.refresh);
}

function showWidget(widget, visible) {
    if (!widget) return false;
    const changed = widget.hidden !== !visible || widget.options?.hidden !== !visible;
    widget.options ??= {};
    widget.options.hidden = !visible;
    widget.hidden = !visible;
    const element = widget.element ?? widget.inputEl;
    if (element?.style) element.style.display = visible ? "" : "none";
    return changed;
}

function updateVisibility(node, widgets) {
    const find = (name) => widgets?.find((widget) => widget.name === name);
    const resolution = find("resolution");
    if (!resolution) return false;
    const manual = resolution.value === "manual";
    let changed = showWidget(find("width"), manual);
    changed = showWidget(find("height"), manual) || changed;
    const task = taskWidget(node)?.value ?? "t2v";
    changed = showWidget(find("t2v_cfg"), task === "t2v") || changed;
    return showWidget(find("i2v_cfg"), task === "i2v") || changed;
}

function updateBindings(node, state, widgets) {
    bindWidget(state, "resolution", widgets?.find((widget) => widget.name === "resolution"));
    bindWidget(state, "task", taskWidget(node));
}

function observePromotedWidgets(node, state) {
    if (!node.isSubgraphNode?.()) return;
    const descriptor = Object.getOwnPropertyDescriptor(node, "widgets");
    if (!descriptor?.get || !descriptor.configurable) return;
    // Native subgraphs rebuild widget projections; update every new projection.
    Object.defineProperty(node, "widgets", {
        ...descriptor,
        get() {
            const widgets = descriptor.get.call(this);
            if (state.removed || !this.graph) return widgets;
            updateBindings(this, state, widgets);
            if (updateVisibility(this, widgets) && !state.refreshing && !state.resizePending) {
                state.resizePending = true;
                queueMicrotask(() => {
                    if (state.resizePending) state.refresh();
                });
            }
            return widgets;
        },
    });
}

function createState(node) {
    const state = { removed: false };
    state.refresh = () => {
        if (state.refreshing) return;
        state.resizePending = false;
        if (state.removed || !node.graph) {
            bindWidget(state, "task", null);
            bindWidget(state, "resolution", null);
            return;
        }
        state.refreshing = true;
        try {
            const widgets = node.widgets;
            updateBindings(node, state, widgets);
            updateVisibility(node, widgets);
            node.setSize?.([node.size[0], node.computeSize()[1]]);
            node.graph.setDirtyCanvas?.(true, true);
        } finally {
            state.refreshing = false;
        }
    };
    const onWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (...args) {
        const result = onWidgetChanged?.apply(this, args);
        state.refresh();
        return result;
    };
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = onConnectionsChange?.apply(this, args);
        // The graph finishes changing its links after notifying the node.
        queueMicrotask(state.refresh);
        return result;
    };
    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        state.removed = true;
        bindWidget(state, "task", null);
        bindWidget(state, "resolution", null);
        return onRemoved?.apply(this, args);
    };
    const onAdded = node.onAdded;
    node.onAdded = function (...args) {
        const result = onAdded?.apply(this, args);
        state.removed = false;
        queueMicrotask(state.refresh);
        return result;
    };
    observePromotedWidgets(node, state);
    return state;
}

export function configure(node) {
    if (!node.properties?.vimoo_generation && node.comfyClass !== "VimooResolution") return;
    let state = states.get(node);
    if (!state) {
        state = createState(node);
        states.set(node, state);
    }
    state.refresh();
}
