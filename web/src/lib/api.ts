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
