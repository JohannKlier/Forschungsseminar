export type SelectionMode = "contiguous" | "free";

const uniq = (indices: number[]) => Array.from(new Set(indices));

const normalize = (indices: number[], mode: SelectionMode) => {
  if (!indices.length) return [];
  const unique = uniq(indices);
  if (mode === "free") return unique;
  const min = Math.min(...unique);
  const max = Math.max(...unique);
  return Array.from({ length: max - min + 1 }, (_, i) => min + i);
};

type ClickSelectionArgs = {
  current: number[];
  idx: number;
  multi: boolean;
  mode: SelectionMode;
};

export const applyClickSelection = ({ current, idx, multi, mode }: ClickSelectionArgs) => {
  if (multi) {
    const next = current.includes(idx) ? current.filter((i) => i !== idx) : [...current, idx];
    return normalize(next, mode);
  }
  if (current.includes(idx)) return normalize(current, mode);
  return normalize([idx], mode);
};

type DragSelectionArgs = {
  current: number[];
  idx: number;
  multi: boolean;
  mode: SelectionMode;
};

export const resolveDragSelection = ({ current, idx, multi, mode }: DragSelectionArgs) => {
  const next = applyClickSelection({ current, idx, multi, mode });
  const targets = next.length ? next : [idx];
  return { next, targets };
};

type BrushSelectionArgs = {
  current: number[];
  selected: number[];
  multi: boolean;
  mode: SelectionMode;
};

export const applyBrushSelection = ({ current, selected, multi, mode }: BrushSelectionArgs) => {
  const merged = multi ? uniq([...current, ...selected]) : selected;
  return normalize(merged, mode);
};
