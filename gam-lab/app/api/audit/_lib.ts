import { appendFile, mkdir, readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { Pool } from "pg";

const AUDIT_DIR = path.join(process.cwd(), "data", "audit");
const AUDIT_USERS_DIR = path.join(AUDIT_DIR, "users");
const AUDIT_TABLE = "audit_events";
const AUDIT_STORAGE = process.env.AUDIT_STORAGE?.trim().toLowerCase() ?? (process.env.DATABASE_URL ? "postgres" : "file");
const AUDIT_DATABASE_SSL = process.env.AUDIT_DATABASE_SSL?.trim().toLowerCase() ?? "disable";

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && Object.getPrototypeOf(value) === Object.prototype;

const toJsonValue = (value: unknown): JsonValue => {
  if (value == null) return null;
  if (typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (Array.isArray(value)) return value.map((item) => toJsonValue(item));
  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [key, toJsonValue(nested)]),
    );
  }
  return String(value);
};

export const sanitizeUserId = (value: string | undefined | null) => {
  const cleaned = value?.trim().replace(/[^a-zA-Z0-9._-]/g, "-").slice(0, 128);
  return cleaned ? cleaned : undefined;
};

export const createServerUserId = () => {
  return `anon-${crypto.randomUUID()}`;
};

export type AuditRecord = {
  recordedAt?: string;
  occurredAt?: string;
  eventId?: string;
  userId?: string;
  participantId?: string | null;
  sessionId?: string;
  category?: string;
  action?: string;
  featureKey?: string | null;
  page?: string | null;
  detail?: JsonValue;
  request?: JsonValue;
};

export type AuditRecordFilter = {
  day?: string;
  userId?: string;
  sessionId?: string;
  category?: string;
  action?: string;
  query?: string;
  limit?: number;
};

const isAuditRecord = (value: JsonValue): value is AuditRecord => {
  return typeof value === "object" && value !== null && !Array.isArray(value);
};

const normalizeForSearch = (value: AuditRecord) => JSON.stringify(value).toLowerCase();

const getAuditRecordTimestamp = (value: AuditRecord) => value.occurredAt ?? value.recordedAt ?? "";

const compareAuditRecords = (left: AuditRecord, right: AuditRecord) =>
  getAuditRecordTimestamp(right).localeCompare(getAuditRecordTimestamp(left));

const getConfiguredAuditStorage = () => {
  if (AUDIT_STORAGE === "postgres" || AUDIT_STORAGE === "file") {
    return AUDIT_STORAGE;
  }
  return process.env.DATABASE_URL ? "postgres" : "file";
};

const toTrimmedString = (value: unknown) => {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
};

const toTimestampString = (value: unknown) => {
  const trimmed = toTrimmedString(value);
  if (!trimmed) return undefined;
  const parsed = new Date(trimmed);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
};

const normalizeAuditRecord = (record: unknown): AuditRecord => {
  const source = isPlainObject(record) ? record : {};
  const recordedAt = toTimestampString(source.recordedAt) ?? new Date().toISOString();
  const occurredAt = toTimestampString(source.occurredAt);
  const userId = sanitizeUserId(toTrimmedString(source.userId)) ?? "unknown-user";

  return {
    recordedAt,
    occurredAt,
    eventId: toTrimmedString(source.eventId),
    userId,
    participantId: sanitizeUserId(toTrimmedString(source.participantId)) ?? null,
    sessionId: sanitizeUserId(toTrimmedString(source.sessionId)),
    category: toTrimmedString(source.category),
    action: toTrimmedString(source.action),
    featureKey: toTrimmedString(source.featureKey) ?? null,
    page: toTrimmedString(source.page) ?? null,
    detail: "detail" in source ? toJsonValue(source.detail) : null,
    request: "request" in source ? toJsonValue(source.request) : null,
  };
};

const toDayBounds = (day: string) => {
  const start = new Date(`${day}T00:00:00.000Z`);
  if (Number.isNaN(start.getTime())) return undefined;
  const end = new Date(start);
  end.setUTCDate(end.getUTCDate() + 1);
  return { start: start.toISOString(), end: end.toISOString() };
};

let auditPool: Pool | null = null;
let auditSchemaPromise: Promise<void> | null = null;

const getAuditPool = () => {
  if (!process.env.DATABASE_URL) {
    throw new Error("AUDIT_STORAGE=postgres requires DATABASE_URL.");
  }
  if (auditPool) return auditPool;

  const ssl =
    AUDIT_DATABASE_SSL === "require"
      ? {
          rejectUnauthorized: false,
        }
      : undefined;

  auditPool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl,
  });
  return auditPool;
};

const ensureAuditSchema = async () => {
  if (getConfiguredAuditStorage() !== "postgres") return;
  if (!auditSchemaPromise) {
    const pool = getAuditPool();
    auditSchemaPromise = (async () => {
      await pool.query(`
        CREATE TABLE IF NOT EXISTS ${AUDIT_TABLE} (
          id BIGSERIAL PRIMARY KEY,
          recorded_at TIMESTAMPTZ NOT NULL,
          occurred_at TIMESTAMPTZ NULL,
          event_id TEXT NULL,
          user_id TEXT NOT NULL,
          participant_id TEXT NULL,
          session_id TEXT NULL,
          category TEXT NULL,
          action TEXT NULL,
          feature_key TEXT NULL,
          page TEXT NULL,
          detail JSONB NULL,
          request JSONB NULL
        )
      `);
      await Promise.all([
        pool.query(`CREATE INDEX IF NOT EXISTS ${AUDIT_TABLE}_recorded_at_idx ON ${AUDIT_TABLE} (recorded_at DESC)`),
        pool.query(`CREATE INDEX IF NOT EXISTS ${AUDIT_TABLE}_user_id_idx ON ${AUDIT_TABLE} (user_id)`),
        pool.query(`CREATE INDEX IF NOT EXISTS ${AUDIT_TABLE}_session_id_idx ON ${AUDIT_TABLE} (session_id)`),
        pool.query(`CREATE INDEX IF NOT EXISTS ${AUDIT_TABLE}_category_action_idx ON ${AUDIT_TABLE} (category, action)`),
      ]);
    })().catch((error) => {
      auditSchemaPromise = null;
      throw error;
    });
  }

  await auditSchemaPromise;
};

const collectAuditFiles = async (rootDir: string, prefix = ""): Promise<string[]> => {
  const entries = await readdir(rootDir, { withFileTypes: true }).catch(() => []);
  const files: string[] = [];

  for (const entry of entries) {
    const relativePath = prefix ? path.join(prefix, entry.name) : entry.name;
    const absolutePath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectAuditFiles(absolutePath, relativePath)));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".ndjson")) {
      files.push(relativePath);
    }
  }

  return files;
};

const appendAuditRecordsToFiles = async (records: AuditRecord[]) => {
  if (!records.length) return;
  const grouped = new Map<string, AuditRecord[]>();

  for (const record of records) {
    const day = (record.recordedAt ?? new Date().toISOString()).slice(0, 10);
    const key = `${record.userId ?? "unknown-user"}/${day}`;
    const bucket = grouped.get(key);
    if (bucket) {
      bucket.push(record);
    } else {
      grouped.set(key, [record]);
    }
  }

  for (const [key, bucket] of grouped) {
    const [userId, day] = key.split("/");
    const directory = path.join(AUDIT_USERS_DIR, userId);
    const filePath = path.join(directory, `${day}.ndjson`);
    await mkdir(directory, { recursive: true });
    const lines = bucket.map((item) => JSON.stringify(item)).join("\n");
    await appendFile(filePath, `${lines}\n`, "utf8");
  }
};

const appendAuditRecordsToPostgres = async (records: AuditRecord[]) => {
  if (!records.length) return;
  await ensureAuditSchema();
  const pool = getAuditPool();
  const values: unknown[] = [];
  const rows = records.map((record, index) => {
    const offset = index * 12;
    values.push(
      record.recordedAt ?? new Date().toISOString(),
      record.occurredAt ?? null,
      record.eventId ?? null,
      record.userId ?? "unknown-user",
      record.participantId ?? null,
      record.sessionId ?? null,
      record.category ?? null,
      record.action ?? null,
      record.featureKey ?? null,
      record.page ?? null,
      record.detail == null ? null : JSON.stringify(record.detail),
      record.request == null ? null : JSON.stringify(record.request),
    );
    return `($${offset + 1}, $${offset + 2}, $${offset + 3}, $${offset + 4}, $${offset + 5}, $${offset + 6}, $${offset + 7}, $${offset + 8}, $${offset + 9}, $${offset + 10}, $${offset + 11}::jsonb, $${offset + 12}::jsonb)`;
  });

  await pool.query(
    `
      INSERT INTO ${AUDIT_TABLE} (
        recorded_at,
        occurred_at,
        event_id,
        user_id,
        participant_id,
        session_id,
        category,
        action,
        feature_key,
        page,
        detail,
        request
      )
      VALUES ${rows.join(", ")}
    `,
    values,
  );
};

export const appendAuditRecords = async (records: unknown[]) => {
  if (!records.length) return;
  const normalized = records.map((record) => normalizeAuditRecord(record));
  if (getConfiguredAuditStorage() === "postgres") {
    await appendAuditRecordsToPostgres(normalized);
    return;
  }
  await appendAuditRecordsToFiles(normalized);
};

const readAuditRecordsFromFiles = async (filter: AuditRecordFilter = {}) => {
  await mkdir(AUDIT_DIR, { recursive: true });
  const legacyFiles = (await collectAuditFiles(AUDIT_DIR)).filter((filePath) => !filePath.startsWith("users/"));
  const userFiles = (await collectAuditFiles(AUDIT_USERS_DIR)).map((filePath) => path.join("users", filePath));
  const fileNames = [...legacyFiles, ...userFiles].sort().reverse();
  const selectedFiles = filter.day ? fileNames.filter((name) => name.endsWith(`${filter.day}.ndjson`)) : fileNames;
  const records: AuditRecord[] = [];
  const limit = Math.min(Math.max(filter.limit ?? 500, 1), 5000);
  const query = filter.query?.trim().toLowerCase();

  for (const fileName of selectedFiles) {
    const filePath = path.join(AUDIT_DIR, fileName);
    const raw = await readFile(filePath, "utf8");
    const lines = raw.split("\n").filter(Boolean).reverse();

    for (const line of lines) {
      const parsed = JSON.parse(line) as JsonValue;
      if (!isAuditRecord(parsed)) continue;
      if (filter.userId && parsed.userId !== filter.userId) continue;
      if (filter.sessionId && parsed.sessionId !== filter.sessionId) continue;
      if (filter.category && parsed.category !== filter.category) continue;
      if (filter.action && parsed.action !== filter.action) continue;
      if (query && !normalizeForSearch(parsed).includes(query)) continue;

      records.push(parsed);
    }
  }

  records.sort(compareAuditRecords);

  return {
    files: selectedFiles,
    total: records.length,
    records: records.slice(0, limit),
  };
};

const readAuditRecordsFromPostgres = async (filter: AuditRecordFilter = {}) => {
  await ensureAuditSchema();
  const pool = getAuditPool();
  const limit = Math.min(Math.max(filter.limit ?? 500, 1), 5000);
  const clauses: string[] = [];
  const values: unknown[] = [];

  if (filter.day) {
    const bounds = toDayBounds(filter.day);
    if (!bounds) {
      return { files: [], total: 0, records: [] as AuditRecord[] };
    }
    values.push(bounds.start, bounds.end);
    clauses.push(`recorded_at >= $${values.length - 1} AND recorded_at < $${values.length}`);
  }
  if (filter.userId) {
    values.push(filter.userId);
    clauses.push(`user_id = $${values.length}`);
  }
  if (filter.sessionId) {
    values.push(filter.sessionId);
    clauses.push(`session_id = $${values.length}`);
  }
  if (filter.category) {
    values.push(filter.category);
    clauses.push(`category = $${values.length}`);
  }
  if (filter.action) {
    values.push(filter.action);
    clauses.push(`action = $${values.length}`);
  }
  if (filter.query?.trim()) {
    values.push(`%${filter.query.trim().toLowerCase()}%`);
    clauses.push(
      `LOWER(CONCAT_WS(' ', COALESCE(event_id, ''), COALESCE(user_id, ''), COALESCE(participant_id, ''), COALESCE(session_id, ''), COALESCE(category, ''), COALESCE(action, ''), COALESCE(feature_key, ''), COALESCE(page, ''), COALESCE(detail::text, ''), COALESCE(request::text, ''))) LIKE $${values.length}`,
    );
  }

  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  const totalResult = await pool.query<{ count: string }>(`SELECT COUNT(*)::text AS count FROM ${AUDIT_TABLE} ${where}`, values);

  values.push(limit);
  const recordsResult = await pool.query<{
    recorded_at: Date | string;
    occurred_at: Date | string | null;
    event_id: string | null;
    user_id: string;
    participant_id: string | null;
    session_id: string | null;
    category: string | null;
    action: string | null;
    feature_key: string | null;
    page: string | null;
    detail: JsonValue | null;
    request: JsonValue | null;
  }>(
    `
      SELECT
        recorded_at,
        occurred_at,
        event_id,
        user_id,
        participant_id,
        session_id,
        category,
        action,
        feature_key,
        page,
        detail,
        request
      FROM ${AUDIT_TABLE}
      ${where}
      ORDER BY COALESCE(occurred_at, recorded_at) DESC
      LIMIT $${values.length}
    `,
    values,
  );

  const records = recordsResult.rows.map<AuditRecord>((row) => ({
    recordedAt:
      row.recorded_at instanceof Date ? row.recorded_at.toISOString() : new Date(row.recorded_at).toISOString(),
    occurredAt:
      row.occurred_at == null
        ? undefined
        : row.occurred_at instanceof Date
          ? row.occurred_at.toISOString()
          : new Date(row.occurred_at).toISOString(),
    eventId: row.event_id ?? undefined,
    userId: row.user_id,
    participantId: row.participant_id,
    sessionId: row.session_id ?? undefined,
    category: row.category ?? undefined,
    action: row.action ?? undefined,
    featureKey: row.feature_key ?? null,
    page: row.page ?? null,
    detail: row.detail ?? null,
    request: row.request ?? null,
  }));

  return {
    files: [] as string[],
    total: Number.parseInt(totalResult.rows[0]?.count ?? "0", 10),
    records,
  };
};

export const readAuditRecords = async (filter: AuditRecordFilter = {}) => {
  if (getConfiguredAuditStorage() === "postgres") {
    return readAuditRecordsFromPostgres(filter);
  }
  return readAuditRecordsFromFiles(filter);
};

const listAuditDaysFromFiles = async () => {
  await mkdir(AUDIT_DIR, { recursive: true });
  const legacyFiles = (await collectAuditFiles(AUDIT_DIR)).filter((filePath) => !filePath.startsWith("users/"));
  const userFiles = await collectAuditFiles(AUDIT_USERS_DIR);
  return [...legacyFiles, ...userFiles]
    .map((filePath) => path.basename(filePath).replace(/\.ndjson$/, ""))
    .filter(Boolean)
    .filter((value, index, all) => all.indexOf(value) === index)
    .sort()
    .reverse();
};

const listAuditDaysFromPostgres = async () => {
  await ensureAuditSchema();
  const pool = getAuditPool();
  const result = await pool.query<{ day: string }>(`
    SELECT DISTINCT TO_CHAR(recorded_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day
    FROM ${AUDIT_TABLE}
    ORDER BY day DESC
  `);
  return result.rows.map((row) => row.day).filter(Boolean);
};

export const listAuditDays = async () => {
  if (getConfiguredAuditStorage() === "postgres") {
    return listAuditDaysFromPostgres();
  }
  return listAuditDaysFromFiles();
};
