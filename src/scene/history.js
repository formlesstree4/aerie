/**
 * Undo/redo over Scene.captureState()/restoreState(). The `baseline` is the
 * last committed state; `past`/`future` hold states to step back/forward to.
 *
 * Discrete actions (add, delete, transform, bake) call commit() immediately;
 * continuous edits (slider/colour drags) call commitSoon(), which coalesces a
 * burst into a single entry. Selection is captured so undo restores it too.
 */
export class History {
    scene;
    getSelection;
    setSelection;
    onChange;
    past = [];
    future = [];
    baseline;
    timer = null;
    limit = 100;
    constructor(scene, getSelection, setSelection, onChange) {
        this.scene = scene;
        this.getSelection = getSelection;
        this.setSelection = setSelection;
        this.onChange = onChange;
        this.baseline = this.snap();
    }
    snap() {
        return { state: this.scene.captureState(), selection: this.getSelection().map((s) => s.id) };
    }
    /** Record the current state as a new undo step (cancels any pending coalesce). */
    commit() {
        if (this.timer !== null) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        this.past.push(this.baseline);
        if (this.past.length > this.limit)
            this.past.shift();
        this.baseline = this.snap();
        this.future.length = 0;
    }
    /** Drop all undo/redo history and re-baseline to the current scene (e.g. on
     *  New Scene, where resurrecting the old document would be confusing). */
    reset() {
        if (this.timer !== null) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        this.past.length = 0;
        this.future.length = 0;
        this.baseline = this.snap();
    }
    /** Coalesce a burst of continuous edits into one commit ~400ms after the last. */
    commitSoon() {
        if (this.timer !== null)
            clearTimeout(this.timer);
        this.timer = setTimeout(() => { this.timer = null; this.commit(); }, 400);
    }
    apply(s) {
        this.scene.restoreState(s.state);
        const byId = new Map(this.scene.selectables.map((it) => [it.id, it]));
        this.setSelection(s.selection.map((id) => byId.get(id)).filter((x) => !!x));
        this.onChange();
    }
    undo() {
        // Flush a pending coalesced edit so it becomes its own undo step first.
        if (this.timer !== null) {
            clearTimeout(this.timer);
            this.timer = null;
            this.commit();
        }
        if (this.past.length === 0)
            return false;
        this.future.push(this.baseline);
        this.baseline = this.past.pop();
        this.apply(this.baseline);
        return true;
    }
    redo() {
        if (this.future.length === 0)
            return false;
        this.past.push(this.baseline);
        this.baseline = this.future.pop();
        this.apply(this.baseline);
        return true;
    }
    canUndo() {
        return this.past.length > 0 || this.timer !== null;
    }
    canRedo() {
        return this.future.length > 0;
    }
}
