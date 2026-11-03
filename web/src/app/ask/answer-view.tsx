"use client";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/skiper-ui/skiper101";
import { MinimalCard, MinimalCardDescription, MinimalCardTitle } from "@/components/ui/minimal-card";
import type { Answer, Citation } from "@/lib/api";
import { segments } from "@/lib/citations";

type Props = { answer: Answer; onOpen: (citation: Citation) => void };

const NOTICE: Record<Answer["outcome"], string | null> = {
  answered: null,
  refused: "No indexed passage supports an answer to this question.",
  citation_check_failed: "The answer was withheld because a citation did not match its source.",
};

function Marker({ citation, onOpen }: { citation: Citation; onOpen: Props["onOpen"] }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => onOpen(citation)}
          className="mx-0.5 rounded px-1 font-mono text-xs text-sky-300 underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-sky-300"
          aria-label={`Citation ${citation.number}: ${citation.title}, page ${citation.page}`}
        >
          [{citation.number}]
        </button>
      </TooltipTrigger>
      <TooltipContent>
        {citation.title}, p. {citation.page}
      </TooltipContent>
    </Tooltip>
  );
}

export function AnswerView({ answer, onOpen }: Props) {
  const byNumber = new Map(answer.citations.map((c) => [c.number, c]));
  const notice = NOTICE[answer.outcome];
  return (
    <section aria-live="polite" className="flex flex-col gap-6">
      {notice && <p className="text-sm text-amber-300">{notice}</p>}
      <p className="leading-7 text-foreground/90">
        {segments(answer.text).map((s, i) =>
          "text" in s
            ? s.text
            : s.cite.map((n) => {
                const c = byNumber.get(n);
                return c ? <Marker key={`${i}-${n}`} citation={c} onOpen={onOpen} /> : `[${n}]`;
              }),
        )}
      </p>
      <ol className="flex flex-col gap-3">
        {answer.citations.map((c) => (
          <li key={c.number}>
            <MinimalCard className="h-full p-4">
              <MinimalCardTitle className="mt-0 text-sm">
                [{c.number}] {c.title}
              </MinimalCardTitle>
              <MinimalCardDescription className="flex flex-wrap gap-x-4 pb-0">
                <a href={c.url} target="_blank" rel="noopener noreferrer" className="underline">
                  arXiv:{c.arxiv_id}, p. {c.page}
                </a>
                <button type="button" onClick={() => onOpen(c)} className="underline">
                  Show passage
                </button>
              </MinimalCardDescription>
            </MinimalCard>
          </li>
        ))}
      </ol>
    </section>
  );
}
