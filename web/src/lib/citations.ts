// Splits answer text on citation markers like [2] or [1, 3].
export type Segment = { text: string } | { cite: number[] };

// Same pattern as CITATION in src/copilot/answer.py.
const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export function segments(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index) });
    out.push({ cite: m[1].split(",").map(Number) });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

// Model answers arrive as light markdown: blank-line paragraphs, "*" or "-" bullets,
// **bold** and $math$. The page shows plain paragraphs and lists, so the markup is dropped.
export type Block = { kind: "p" | "li"; text: string };

const BULLET = /^\s*(?:[*-]|\d+\.)\s+/;

export function blocks(text: string): Block[] {
  return text
    .split("\n")
    .map((line) => line.replace(/\*\*(.+?)\*\*/g, "$1").replace(/\$([^$]+)\$/g, "$1").trim())
    .filter(Boolean)
    .map((line) => (BULLET.test(line) ? { kind: "li", text: line.replace(BULLET, "") } : { kind: "p", text: line }));
}
