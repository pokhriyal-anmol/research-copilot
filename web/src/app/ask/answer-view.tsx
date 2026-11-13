"use client";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/skiper-ui/skiper101";
import { MinimalCard, MinimalCardDescription, MinimalCardTitle } from "@/components/ui/minimal-card";
import type { Answer, Citation } from "@/lib/api";
import { blocks, segments } from "@/lib/citations";

type Props = { answer: Answer; open: Citation | null; onOpen: (citation: Citation | null) => void };

const NOTICE: Record<Answer["outcome"], string | null> = {
  answered: null,
  refused: "No indexed passage supports an answer to this question.",
  citation_check_failed: "The answer was withheld because a citation did not match its source.",
};

function Marker({ citation, onOpen }: { citation: Citation; onOpen: (c: Citation) => void }) {
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

function Rich({ text, byNumber, onOpen }: { text: string; byNumber: Map<number, Citation>; onOpen: (c: Citation) => void }) {
  return segments(text).map((s, i) =>
    "text" in s
      ? s.text
      : s.cite.map((n) => {
          const c = byNumber.get(n);
          return c ? <Marker key={`${i}-${n}`} citation={c} onOpen={onOpen} /> : `[${n}]`;
        }),
  );
}

// Consecutive list items become one <ul>; everything else is a paragraph.
function Body({ text, byNumber, onOpen }: { text: string; byNumber: Map<number, Citation>; onOpen: (c: Citation) => void }) {
  const groups: { kind: "p" | "ul"; lines: string[] }[] = [];
  for (const b of blocks(text)) {
    const last = groups.at(-1);
    if (b.kind === "li" && last?.kind === "ul") last.lines.push(b.text);
    else groups.push({ kind: b.kind === "li" ? "ul" : "p", lines: [b.text] });
  }
  return (
    <div className="flex flex-col gap-3 leading-7 text-foreground/90">
      {groups.map((g, i) =>
        g.kind === "p" ? (
          <p key={i}>
            <Rich text={g.lines[0]} byNumber={byNumber} onOpen={onOpen} />
          </p>
        ) : (
          <ul key={i} className="flex list-disc flex-col gap-2 pl-5">
            {g.lines.map((line, j) => (
              <li key={j}>
                <Rich text={line} byNumber={byNumber} onOpen={onOpen} />
              </li>
            ))}
          </ul>
        ),
      )}
    </div>
  );
}

export function AnswerView({ answer, open, onOpen }: Props) {
  const byNumber = new Map(answer.citations.map((c) => [c.number, c]));
  const notice = NOTICE[answer.outcome];
  return (
    <section aria-live="polite" className="flex flex-col gap-6">
      {notice && <p className="text-sm text-amber-300">{notice}</p>}
      <Body text={answer.text} byNumber={byNumber} onOpen={onOpen} />
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
                <button type="button" onClick={() => onOpen(open === c ? null : c)} className="underline">
                  {open === c ? "Hide passage" : "Show passage"}
                </button>
              </MinimalCardDescription>
              {open === c && (
                <blockquote className="mt-3 max-h-[50vh] overflow-y-auto text-sm leading-6 text-foreground/80 md:hidden">
                  {c.passage}
                </blockquote>
              )}
            </MinimalCard>
          </li>
        ))}
      </ol>
    </section>
  );
}
