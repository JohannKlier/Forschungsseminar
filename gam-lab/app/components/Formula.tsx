"use client";

import { useEffect, useRef } from "react";
import styles from "./Formula.module.css";

declare global {
  interface Window {
    katex?: {
      render: (
        expression: string,
        element: HTMLElement,
        options?: { displayMode?: boolean; throwOnError?: boolean }
      ) => void;
    };
  }
}

type FormulaProps = {
  expression: string;
  displayMode?: boolean;
};

export default function Formula({
  expression,
  displayMode = true,
}: FormulaProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!window.katex || !ref.current) return;

    try {
      window.katex.render(expression, ref.current, {
        displayMode,
        throwOnError: false,
      });
    } catch (error) {
      console.error("KaTeX render error", error);
    }
  }, [expression, displayMode]);

  return <span className={styles.formula} ref={ref} aria-label={expression} />;
}
