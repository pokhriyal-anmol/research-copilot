# Answering questions that don't name their paper

Recorded 2026-11-04, before the change was written or any model call was made.

## Why

Of the 120 reviewed eval questions, two AI labellers agree that 64 do not say which paper they mean, for example "How many A800 graphics processing units were used to run the experiments?" ([2026-11-04-refusals.md](2026-11-04-refusals.md)). Real users ask this way too. On 2026-11-03 the demo refused 23 of those 64 (36%), against 2 of the 44 that name their paper.

## Change (prompt v2)

The system prompt gains one instruction: when the question does not say which paper it means and the passages come from papers that answer it differently, give each paper's answer, naming the paper and citing its passage, instead of picking one or refusing. Refusal stays for passages that answer nothing. Retrieval, the model (gemini-3.5-flash-lite, OpenRouter fallback off) and the citation check are unchanged.

## Measurement

All 120 questions are answered again with v2. As a control for run-to-run variation, the 64 vague questions are also answered again with v1, if the free quota allows; without it, the 2026-11-03 v1 answers are the baseline and that is stated.

Two AI labellers, working separately, judge up to 50 sentences sampled at random from v2 answers to the 64 vague questions, at sentence level against all the passages each cites. They use the earlier rubric plus one rule: a sentence that names a paper is supported only if its cited passage comes from that paper.

## Pass rule

v2 is kept only if all three hold:

1. Refusals on the 64 vague questions fall by at least 8 questions against the v1 baseline (the v1 control if run, else 23).
2. Refusals on the 44 identifiable questions rise by no more than 2 (to at most 4 against 2).
3. At least 90% of the sampled v2 sentences are fully supported according to both labellers.

Also reported, not part of the rule: how many v2 answers to vague questions cite more than one paper.

## Result: v2 fails rule 1 and is not live

Both runs finished with no provider failures (120 v2 answers, 64 v1 control answers).

| Refused | v1, 2026-11-03 | v1 control, 2026-11-04 | v2 |
|---|---|---|---|
| 64 vague questions | 23 | 23 | 23 |
| 44 identifiable questions | 2 | | 2 |

Rule 1 needed refusals on vague questions to fall by at least 8; they did not move. Rule 2 holds. Since all three rules must hold, the sentence labelling for rule 3 was not run. Between the v1 control and v2, 6 of the 64 vague questions changed outcome, 3 each way, which is the size of run-to-run variation. 3 of 41 v2 answers to vague questions cite more than one paper.

Why: of the 23 vague questions v2 refused, the known passage was outside the 30-passage rerank pool for 16 and on another page of the right paper for 6; it was sent for 1. For a question that never names its paper, the passages retrieved rarely contain an answer, so there are no differing per-paper answers to report. The instruction targets a case that almost never arises. A fix for vague questions has to act before the model: on retrieval, or on what the reply says when nothing answers.

The live prompt stays v1. v2 is kept in `copilot/answer.py`, marked as dropped, so this run can be repeated.

## Second attempt: suggest the closest papers on a refusal

Recorded 2026-11-04, before it was written or measured.

When the model refuses, the reply names the 3 closest distinct papers among the passages retrieval found, in retrieval order, and asks the user to name one. No model call is added and nothing is generated: the titles come from the database. Answers and the citation check are unchanged.

Measure, from the deterministic retrieval replay: for each question the demo refused on 2026-11-03, whether the paper the question was written from is among the 3 suggested, reported separately for vague and identifiable questions. Kept if the right paper is suggested for at least a third of the refused vague questions; below that the suggestions would mostly point users the wrong way.

### Result: below the bar, not kept

For the 26 questions the demo refused on 2026-11-03, the paper each question was written from was among the 3 suggested for 3 of the 23 vague questions (13%), 0 of the 2 identifiable ones and 1 of the 1 split. The bar was a third of the vague questions, so the suggestions would mostly have pointed users at the wrong paper, and the change was removed. Offering more papers was not tried: the number was fixed before measuring, and choosing it after seeing this result would be tuning to these 23 questions.

## Where this leaves vague questions

Two fixes that act after retrieval both failed for the same reason: for a question that never names its paper, retrieval rarely finds that paper at all. On these 64 eval questions, the paper behind the question is not identifiable from the question alone, so refusing is the honest answer. The questions came from a generator told not to name the paper; how often real users ask that way is unknown. The next eval set, drafted with the fixed generator, will measure what matters more: answers to questions that do name their subject.
