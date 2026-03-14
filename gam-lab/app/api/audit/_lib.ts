import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";

const AUDIT_DIR = path.join(process.cwd(), "data", "audit");

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

export const appendAuditRecords = async (records: unknown[]) => {
  if (!records.length) return;
  await mkdir(AUDIT_DIR, { recursive: true });
  const day = new Date().toISOString().slice(0, 10);
  const filePath = path.join(AUDIT_DIR, `${day}.ndjson`);
  const lines = records.map((record) => JSON.stringify(toJsonValue(record))).join("\n");
  await appendFile(filePath, `${lines}\n`, "utf8");
};
