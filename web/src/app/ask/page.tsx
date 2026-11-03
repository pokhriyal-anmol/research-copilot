"use client";

import { useState } from "react";

import { PolishedToaster, showPolishedToast } from "@/components/ui/halo-toast";
import { PromptComposer } from "@/components/ui/prompt-composer";
import { SidePanel } from "@/components/ui/side-panel";
import { type Answer, AskError, type Citation, ask } from "@/lib/api";

import { AnswerView } from "./answer-view";

// Mirrors the Question model's limits in src/copilot/app.py.
const MIN_LENGTH = 3;
const MAX_LENGTH = 500;

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [passage, setPassage] = useState<Citation | null>(null);

  async function send() {
    const q = question.trim();
    if (q.length < MIN_LENGTH) return;
    setLoading(true);
    try {
      setAnswer(await ask(q));
      setPassage(null);
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
    <div className="flex flex-1 items-start gap-6 py-10 pr-4">
      {answer && answer.citations.length > 0 && (
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
      )}
      <main className="mx-auto flex w-full max-w-2xl flex-col gap-8 pl-4">
        <h1 className="font-heading text-4xl">Ask the papers</h1>
        <PromptComposer
          aria-label="Question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onSend={send}
          isLoading={loading}
          maxLength={MAX_LENGTH}
          placeholder="Ask about an openly licensed IR paper…"
          loadingText="Searching papers…"
        />
        {answer && <AnswerView answer={answer} onOpen={setPassage} />}
      </main>
      <PolishedToaster />
    </div>
  );
}
