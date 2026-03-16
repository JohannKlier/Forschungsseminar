import Link from "next/link";
import { listAuditDays, readAuditRecords, type AuditRecord } from "../api/audit/_lib";
import styles from "./page.module.css";

export const dynamic = "force-dynamic";

type PageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

const takeFirst = (value: string | string[] | undefined) => {
  if (Array.isArray(value)) return value[0];
  return value;
};

const formatTimestamp = (value?: string) => {
  if (!value) return "n/a";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const formatJson = (value: AuditRecord["detail"] | AuditRecord["request"]) => {
  if (value == null) return "";
  return JSON.stringify(value, null, 2);
};

export default async function LogsPage({ searchParams }: PageProps) {
  const params = await searchParams;
  const day = takeFirst(params.day)?.trim() || undefined;
  const userId = takeFirst(params.userId)?.trim() || undefined;
  const sessionId = takeFirst(params.sessionId)?.trim() || undefined;
  const category = takeFirst(params.category)?.trim() || undefined;
  const action = takeFirst(params.action)?.trim() || undefined;
  const query = takeFirst(params.query)?.trim() || undefined;
  const limitValue = Number.parseInt(takeFirst(params.limit) ?? "200", 10);
  const limit = Number.isFinite(limitValue) ? limitValue : 200;

  const [days, result] = await Promise.all([
    listAuditDays(),
    readAuditRecords({
      day,
      userId,
      sessionId,
      category,
      action,
      query,
      limit,
    }),
  ]);

  const categoryOptions = [...new Set(result.records.map((record) => record.category).filter(Boolean))];
  const actionOptions = [...new Set(result.records.map((record) => record.action).filter(Boolean))];
  const userOptions = [...new Set(result.records.map((record) => record.userId).filter(Boolean))];

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <div>
            <p className={styles.eyebrow}>Audit Inspector</p>
            <h1 className={styles.title}>Inspect all user logs</h1>
            <p className={styles.subtitle}>
              Review recorded user sessions, filter by identity or event type, and inspect the full payload for each
              entry.
            </p>
          </div>
          <Link className={styles.backLink} href="/">
            Back to study home
          </Link>
        </section>

        <section className={styles.summaryGrid}>
          <article className={styles.summaryCard}>
            <span className={styles.summaryLabel}>Visible entries</span>
            <strong className={styles.summaryValue}>{result.records.length}</strong>
          </article>
          <article className={styles.summaryCard}>
            <span className={styles.summaryLabel}>Audit files</span>
            <strong className={styles.summaryValue}>{days.length}</strong>
          </article>
          <article className={styles.summaryCard}>
            <span className={styles.summaryLabel}>Users in result</span>
            <strong className={styles.summaryValue}>{userOptions.length}</strong>
          </article>
          <article className={styles.summaryCard}>
            <span className={styles.summaryLabel}>Actions in result</span>
            <strong className={styles.summaryValue}>{actionOptions.length}</strong>
          </article>
        </section>

        <section className={styles.panel}>
          <form className={styles.filters} method="get">
            <label className={styles.field}>
              <span>Day</span>
              <select name="day" defaultValue={day ?? ""}>
                <option value="">All days</option>
                {days.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span>User</span>
              <input list="log-users" name="userId" defaultValue={userId ?? ""} placeholder="anon-..." />
            </label>

            <label className={styles.field}>
              <span>Session</span>
              <input name="sessionId" defaultValue={sessionId ?? ""} placeholder="session-..." />
            </label>

            <label className={styles.field}>
              <span>Category</span>
              <select name="category" defaultValue={category ?? ""}>
                <option value="">All categories</option>
                {categoryOptions.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span>Action</span>
              <select name="action" defaultValue={action ?? ""}>
                <option value="">All actions</option>
                {actionOptions.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span>Search</span>
              <input name="query" defaultValue={query ?? ""} placeholder="text inside detail or request" />
            </label>

            <label className={styles.field}>
              <span>Limit</span>
              <input name="limit" type="number" min="1" max="5000" defaultValue={String(limit)} />
            </label>

            <div className={styles.actions}>
              <button className={styles.primary} type="submit">
                Apply filters
              </button>
              <Link className={styles.secondary} href="/logs">
                Reset
              </Link>
            </div>

            <datalist id="log-users">
              {userOptions.map((value) => (
                <option key={value} value={value} />
              ))}
            </datalist>
          </form>
        </section>

        <section className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Recorded</th>
                <th>User</th>
                <th>Session</th>
                <th>Category</th>
                <th>Action</th>
                <th>Page</th>
                <th>Payload</th>
              </tr>
            </thead>
            <tbody>
              {result.records.length ? (
                result.records.map((record, index) => (
                  <tr key={record.eventId ?? `${record.recordedAt ?? "record"}-${index}`}>
                    <td>
                      <div className={styles.timestamp}>{formatTimestamp(record.occurredAt ?? record.recordedAt)}</div>
                      <div className={styles.meta}>{record.eventId ?? "system event"}</div>
                    </td>
                    <td>
                      <div>{record.userId ?? "n/a"}</div>
                      <div className={styles.meta}>{record.participantId ?? "no participant id"}</div>
                    </td>
                    <td className={styles.wrap}>{record.sessionId ?? "n/a"}</td>
                    <td>{record.category ?? "n/a"}</td>
                    <td className={styles.wrap}>{record.action ?? "n/a"}</td>
                    <td className={styles.wrap}>{record.page ?? "n/a"}</td>
                    <td>
                      <details className={styles.details}>
                        <summary>Open payload</summary>
                        <div className={styles.payloadGroup}>
                          {record.detail != null ? (
                            <pre className={styles.payload}>{formatJson(record.detail)}</pre>
                          ) : (
                            <p className={styles.emptyPayload}>No detail payload.</p>
                          )}
                          {record.request != null ? (
                            <pre className={styles.payload}>{formatJson(record.request)}</pre>
                          ) : null}
                        </div>
                      </details>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className={styles.empty} colSpan={7}>
                    No log entries matched the current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
}
