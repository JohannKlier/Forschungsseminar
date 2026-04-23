import { appendFile, mkdir, readdir, readFile } from "node:fs/promises";
import path from "node:path";

const AUDIT_DIR = path.join(process.cwd(), "data", "audit");
const AUDIT_USERS_DIR = path.join(AUDIT_DIR, "users");

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
  const userId = sanitizeUserId(toTrimmedString(source.userId)) ?? "unknown";

  return {
    recordedAt,
    occurredAt,
    eventId: toTrimmedString(source.eventId),
    userId,
    sessionId: sanitizeUserId(toTrimmedString(source.sessionId)),
    category: toTrimmedString(source.category),
    action: toTrimmedString(source.action),
    featureKey: toTrimmedString(source.featureKey) ?? null,
    page: toTrimmedString(source.page) ?? null,
    detail: "detail" in source ? toJsonValue(source.detail) : null,
    request: "request" in source ? toJsonValue(source.request) : null,
  };
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

export const appendAuditRecords = async (records: unknown[]) => {
  if (!records.length) return;
  const normalized = records.map((record) => normalizeAuditRecord(record));
  const grouped = new Map<string, AuditRecord[]>();

  for (const record of normalized) {
    const day = (record.recordedAt ?? new Date().toISOString()).slice(0, 10);
    const key = `${record.userId ?? "unknown"}/${day}`;
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

export const readAuditRecords = async (filter: AuditRecordFilter = {}) => {
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

export const listAuditDays = async () => {
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
