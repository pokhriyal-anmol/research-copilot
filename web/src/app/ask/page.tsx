"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { PolishedToaster, showPolishedToast } from "@/components/ui/halo-toast";
import { PromptComposer } from "@/components/ui/prompt-composer";
import { SidePanel } from "@/components/ui/side-panel";
import { type Answer, AskError, type Citation, ask, wake } from "@/lib/api";

import { AnswerView } from "./answer-view";

// Mirrors the Question model's limits in src/copilot/app.py.
const MIN_LENGTH = 3;
const MAX_LENGTH = 500;
// Questions from evals/questions-v2.csv that the indexed papers answer with citations.
const EXAMPLES = [
  "In what way does UltraGCN manage to avoid traditional message passing?",
  "What impact can score ties have on ranking metrics like NDCG@10?",
  "What is the purpose of using negative sampling in recommender systems that rely on implicit feedback?",
];
// Below this, a health check means the server was already awake, so the page says nothing.
const SLOW_WAKE_MS = 1500;

type ServerState = "checking" | "waking" | "ready" | "down";

const SERVER_NOTE: Record<ServerState, string | null> = {
  checking: null,
  waking: "Waking the free server. The first answer can take up to a minute.",
  ready: null,
  down: "The server is slow to wake. Your question may still work, or try again in a minute.",
};

function useServerState(): [ServerState, (s: ServerState) => void] {
  const [state, setState] = useState<ServerState>("checking");
  useEffect(() => {
    const slow = setTimeout(() => setState("waking"), SLOW_WAKE_MS);
    wake().then((ok) => {
      clearTimeout(slow);
      setState(ok ? "ready" : "down");
    });
    return () => clearTimeout(slow);
  }, []);
  return [state, setState];
}

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [passage, setPassage] = useState<Citation | null>(null);
  const [server, setServer] = useServerState();

  async function send(text = question) {
    const q = text.trim();
    if (q.length < MIN_LENGTH) return;
    setLoading(true);
    try {
      setAnswer(await ask(q));
      setPassage(null);
      setServer("ready");
    } catch (e) {
      showPolishedToast({
        title: "No answer this time",
        description: e instanceof AskError ? e.message : "Something went wrong. Please retry.",
        rimVariant: "destructive",
        politeness: "alert",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 items-start gap-6 py-10 md:pr-4">
      {answer && answer.citations.length > 0 && (
        // Phones show the passage under its citation instead (AnswerView).
        // Fixed width: the panel opens to 97% of this box, so a content-sized box would grow with the passage.
        <div className="hidden md:block md:w-[350px] md:shrink-0">
        <SidePanel
          panelOpen={passage !== null}
          handlePanelOpen={() => setPassage(passage ? null : answer.citations[0])}
          className="sticky top-10 md:w-[340px]"
          renderButton={(toggle) => (
            <button type="button" onClick={toggle} className="text-sm text-muted-foreground hover:text-foreground">
              {passage ? "Hide passage" : "Show passage"}
            </button>
          )}
        >
          {passage && (
            <figure className="px-4 pb-5 text-sm">
              <figcaption className="mb-2 text-muted-foreground">
                [{passage.number}] {passage.title}, p. {passage.page}
              </figcaption>
              <blockquote className="max-h-[60vh] overflow-y-auto leading-6 text-foreground/90">{passage.passage}</blockquote>
            </figure>
          )}
        </SidePanel>
        </div>
      )}
      <main className="mx-auto flex w-full max-w-2xl flex-col gap-8 px-4 md:pr-0">
        <header className="flex flex-col gap-3">
          <h1 className="font-heading text-3xl md:text-4xl">Ask the papers</h1>
          <p className="text-muted-foreground">
            Answers come from openly licensed arXiv papers on information retrieval. Each citation links to its
            paper on arXiv, and the code withholds any answer whose citations don&apos;t match the retrieved passages.
          </p>
          {SERVER_NOTE[server] && (
            <p role="status" className="text-sm text-amber-300">
              {SERVER_NOTE[server]}
            </p>
          )}
        </header>
        <PromptComposer
          aria-label="Question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onSend={() => send()}
          isLoading={loading}
          maxLength={MAX_LENGTH}
          placeholder="Ask about an openly licensed IR paper…"
          loadingText={server === "ready" ? "Searching papers…" : "Waking the server…"}
        />
        {!answer && (
          <div className="flex flex-col items-start gap-2">
            <p className="text-sm text-muted-foreground">Try one:</p>
            {EXAMPLES.map((q) => (
              <Button
                key={q}
                variant="outline"
                disabled={loading}
                className="h-auto whitespace-normal py-1.5 text-left"
                onClick={() => {
                  setQuestion(q);
                  send(q);
                }}
              >
                {q}
              </Button>
            ))}
          </div>
        )}
        {answer && <AnswerView answer={answer} open={passage} onOpen={setPassage} />}
      </main>
      <PolishedToaster />
    </div>
  );
}
