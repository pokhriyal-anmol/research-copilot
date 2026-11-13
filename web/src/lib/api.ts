// Client for the Research Copilot API (src/copilot/app.py, POST /answer).
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Citation = {
  number: number;
  arxiv_id: string;
  title: string;
  page: number;
  url: string;
  passage: string;
};

export type Answer = {
  outcome: "answered" | "refused" | "citation_check_failed";
  text: string;
  citations: Citation[];
};

export class AskError extends Error {}

export async function ask(question: string): Promise<Answer> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new AskError("The request failed. Please retry.");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    // FastAPI sends a string detail for its own errors and a list for validation errors.
    throw new AskError(typeof body?.detail === "string" ? body.detail : "Please check your question.");
  }
  return body as Answer;
}

// Render's free plan sleeps after 15 idle minutes and takes up to a minute to wake.
// While it boots, /health can time out or return an error, so keep asking until the deadline.
const WAKE_DEADLINE_MS = 120_000;
const WAKE_RETRY_MS = 3_000;

type WakeOptions = { fetcher?: typeof fetch; retryMs?: number; deadlineMs?: number };

export async function wake({
  fetcher = fetch,
  retryMs = WAKE_RETRY_MS,
  deadlineMs = WAKE_DEADLINE_MS,
}: WakeOptions = {}): Promise<boolean> {
  const deadline = Date.now() + deadlineMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetcher(`${API_URL}/health`, { signal: AbortSignal.timeout(deadline - Date.now()) });
      if (res.ok) return true;
    } catch {
      // Timed out or refused while the server boots: try again below.
    }
    await new Promise((resolve) => setTimeout(resolve, retryMs));
  }
  return false;
}
