"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import styles from "../page.module.css";

type TourLabelPlacement =
  | "top-left"
  | "top-right"
  | "mid-right"
  | "bottom-left"
  | "bottom-right";

type Props = {
  label: string;
  title: string;
  description: string;
  details?: string[];
  placement?: TourLabelPlacement;
};

const placementClassName: Record<TourLabelPlacement, string> = {
  "top-left": "tourLabelPlacementTopLeft",
  "top-right": "tourLabelPlacementTopRight",
  "mid-right": "tourLabelPlacementMidRight",
  "bottom-left": "tourLabelPlacementBottomLeft",
  "bottom-right": "tourLabelPlacementBottomRight",
};

export default function TourLabel({
  label,
  title,
  description,
  details = [],
  placement = "top-left",
}: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [portalReady, setPortalReady] = useState(false);
  const [popoverStyle, setPopoverStyle] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    setPortalReady(true);
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!rootRef.current?.contains(target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [isOpen]);

  useLayoutEffect(() => {
    if (!isOpen) return;
    if (!buttonRef.current) return;

    const updatePosition = () => {
      const rect = buttonRef.current?.getBoundingClientRect();
      const popoverRect = popoverRef.current?.getBoundingClientRect();
      if (!rect || !popoverRect) return;
      const gap = 8;
      const margin = 12;
      const prefersAbove = placement === "bottom-left" || placement === "bottom-right";
      const opensToLeft = placement === "top-left" || placement === "bottom-left";
      const preferredTop = prefersAbove
        ? rect.top - popoverRect.height - gap
        : rect.bottom + gap;
      const preferredLeft = opensToLeft
        ? rect.left - popoverRect.width - gap
        : rect.right + gap;
      const clampedTop = Math.min(
        window.innerHeight - popoverRect.height - margin,
        Math.max(margin, preferredTop),
      );
      const clampedLeft = Math.min(
        window.innerWidth - popoverRect.width - margin,
        Math.max(margin, preferredLeft),
      );
      setPopoverStyle({
        top: clampedTop,
        left: clampedLeft,
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [isOpen, placement]);

  const popover = (
    <div
      id={panelId}
      ref={popoverRef}
      className={`${styles.tourLabelPopover} ${styles.tourLabelPopoverFixed} ${isOpen ? styles.tourLabelPopoverOpen : ""} ${placement === "bottom-left" || placement === "bottom-right" ? styles.tourLabelPopoverAbove : ""}`}
      role="note"
      style={popoverStyle}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <p className={styles.tourLabelTitle}>{title}</p>
      <p className={styles.tourLabelDescription}>{description}</p>
      {details.length ? (
        <ul className={styles.tourLabelList}>
          {details.map((detail) => (
            <li key={detail} className={styles.tourLabelListItem}>
              {detail}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );

  return (
    <div
      ref={rootRef}
      className={`${styles.tourLabelWrap} ${styles[placementClassName[placement]]}`}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => {
        window.setTimeout(() => {
          if (!popoverRef.current?.matches(":hover")) {
            setIsOpen(false);
          }
        }, 20);
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        className={`${styles.tourLabelButton} ${isOpen ? styles.tourLabelButtonActive : ""}`}
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={() => setIsOpen((prev) => !prev)}
        onFocus={() => setIsOpen(true)}
        onBlur={(event) => {
          const related = event.relatedTarget as Node | null;
          if (!rootRef.current?.contains(related)) {
            setIsOpen(false);
          }
        }}
      >
        <span>{label}</span>
        <span className={styles.tourLabelGlyph} aria-hidden="true">
          ?
        </span>
      </button>
      {portalReady ? createPortal(popover, document.body) : null}
    </div>
  );
}
