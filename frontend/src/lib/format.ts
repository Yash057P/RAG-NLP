/** Small display helpers shared across the dashboard. */

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "-";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

/** "3 minutes ago" style relative label. */
export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "never";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "never";

  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 45) return "just now";
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ["minute", 60],
    ["hour", 3600],
    ["day", 86400],
    ["week", 604800],
    ["month", 2592000],
    ["year", 31536000],
  ];

  const formatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  let previous = 1;
  for (const [unit, size] of units) {
    if (Math.abs(seconds) < size) return formatter.format(-Math.round(seconds / previous), unit);
    previous = size;
  }
  return formatter.format(-Math.round(seconds / 31536000), "year");
}

/** Colour a cosine similarity score: 0.5+ strong, 0.3+ decent, below weak. */
export function scoreTone(score: number): "strong" | "decent" | "weak" {
  if (score >= 0.5) return "strong";
  if (score >= 0.3) return "decent";
  return "weak";
}

export function scoreColor(score: number): string {
  const tone = scoreTone(score);
  if (tone === "strong") return "text-emerald-400";
  if (tone === "decent") return "text-sky-400";
  return "text-amber-400";
}

export function scoreBarColor(score: number): string {
  const tone = scoreTone(score);
  if (tone === "strong") return "bg-emerald-400";
  if (tone === "decent") return "bg-sky-400";
  return "bg-amber-400";
}

/** Highlight question terms inside a chunk snippet. */
export function splitHighlight(text: string, query: string): Array<{ text: string; hit: boolean }> {
  const terms = Array.from(
    new Set(
      query
        .toLowerCase()
        .split(/[^a-z0-9]+/i)
        .filter((term) => term.length > 2),
    ),
  );
  if (!terms.length) return [{ text, hit: false }];

  // `split` on a capturing group interleaves the separators, so every odd part
  // is an exact match of one of the query terms.
  const lookup = new Set(terms);
  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  return text
    .split(pattern)
    .filter((part) => part !== "")
    .map((part) => ({ text: part, hit: lookup.has(part.toLowerCase()) }));
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Shorten a filename while keeping the extension visible. */
export function truncateFilename(name: string, max = 28): string {
  if (name.length <= max) return name;
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 ? name.slice(dot) : "";
  const stem = dot > 0 ? name.slice(0, dot) : name;
  return `${stem.slice(0, Math.max(6, max - ext.length - 1))}...${ext}`;
}
