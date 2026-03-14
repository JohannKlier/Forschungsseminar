"use client";

import { useEffect } from "react";
import { sanitizeAuditText, type AuditLogFn } from "../lib/audit";

const pickInteractiveTarget = (target: EventTarget | null) => {
  if (!(target instanceof Element)) return null;
  return target.closest("button,a,input,select,textarea,[role='button'],[data-audit]") as HTMLElement | null;
};

const describeElement = (element: HTMLElement) => {
  const input = element as HTMLInputElement;
  return {
    tag: element.tagName.toLowerCase(),
    id: sanitizeAuditText(element.id, 80),
    name: sanitizeAuditText(input.name, 80),
    role: sanitizeAuditText(element.getAttribute("role"), 80),
    type: sanitizeAuditText(input.type, 40),
    ariaLabel: sanitizeAuditText(element.getAttribute("aria-label"), 120),
    text: sanitizeAuditText(element.textContent, 120),
  };
};

const getChangeValue = (element: HTMLElement) => {
  if (element instanceof HTMLInputElement) {
    if (element.type === "checkbox" || element.type === "radio") {
      return { checked: element.checked };
    }
    return { value: sanitizeAuditText(element.value, 200) };
  }
  if (element instanceof HTMLSelectElement || element instanceof HTMLTextAreaElement) {
    return { value: sanitizeAuditText(element.value, 200) };
  }
  return {};
};

export function useUiAuditLogger(logEvent: AuditLogFn) {
  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      const target = pickInteractiveTarget(event.target);
      if (!target) return;
      logEvent({
        category: "ui",
        action: "ui.click",
        detail: describeElement(target),
      });
    };

    const handleChange = (event: Event) => {
      const target = pickInteractiveTarget(event.target);
      if (!target) return;
      logEvent({
        category: "ui",
        action: "ui.change",
        detail: {
          ...describeElement(target),
          ...getChangeValue(target),
        },
      });
    };

    document.addEventListener("click", handleClick, true);
    document.addEventListener("change", handleChange, true);
    return () => {
      document.removeEventListener("click", handleClick, true);
      document.removeEventListener("change", handleChange, true);
    };
  }, [logEvent]);
}
