import styles from "./Sidebar.module.css";

type SidebarItem = {
  id: string;
  label: string;
  subsections?: { id: string; label: string }[];
};

type SidebarProps = {
  items: SidebarItem[];
  activeItemId: string;
  onSelect: (itemId: string) => void;
  activeSubsectionId?: string;
  onSelectSubsection?: (itemId: string) => void;
  onOpenInspector?: () => void;
  inspectorAvailable?: boolean;
};

export default function Sidebar({
  items,
  activeItemId,
  onSelect,
  activeSubsectionId,
  onSelectSubsection,
  onOpenInspector,
  inspectorAvailable,
}: SidebarProps) {

  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <div>
          <p className={styles.brandLabel}>Johann Klier</p>
          <p className={styles.brandTitle}>Statistik</p>
          <p className={styles.brandSub}>Interactive studio</p>
        </div>
        <p className={styles.brandSummary}>Pick a section and dial its quick settings.</p>
      </div>

      <nav className={styles.nav}>
        {items.map((item) => {
          const isActive = item.id === activeItemId;
          const subsections = item.subsections ?? [];
          return (
            <div key={item.id} className={styles.navGroup}>
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                aria-current={isActive ? "page" : undefined}
                className={`${styles.navButton} ${isActive ? styles.navButtonActive : ""}`}
              >
                <span className={styles.navLabel}>{item.label}</span>
              </button>
              {isActive && subsections.length ? (
                <nav className={styles.subnav}>
                  {subsections.map((subsection) => {
                    const isSubActive = subsection.id === activeSubsectionId && isActive;
                    return (
                      <button
                        key={subsection.id}
                        type="button"
                        onClick={() => {
                          if (!isActive) onSelect(item.id);
                          onSelectSubsection?.(subsection.id);
                        }}
                        className={`${styles.subnavButton} ${isSubActive ? styles.subnavButtonActive : ""}`}
                      >
                        <span>{subsection.label}</span>
                      </button>
                    );
                  })}
                </nav>
              ) : null}
            </div>
          );
        })}
      </nav>

      {onOpenInspector ? (
        <button type="button" className={styles.inspectorButton} onClick={onOpenInspector} disabled={!inspectorAvailable}>
          {inspectorAvailable ? "Open inspector" : "No quick settings"}
        </button>
      ) : null}
    </aside>
  );
}
