# Numbers check on cited sentences

Recorded 2026-11-04, before the check was written or run.

## Why

Of 48 sampled answer sentences judged against all their cited passages, 3 were not fully supported ([2026-11-03-sentence-labels.json](2026-11-03-sentence-labels.json)). One, item 38, states "approximately 2% to 4%" where its passage reports a 1.2-point gap: a figure the source never gives. A check in code can catch that kind of error with no model call.

## The check

After the existing citation check, every sentence that cites a passage has its numbers compared with the numbers in the passages it cites. A number in the sentence must equal, as a value, some number in those passages (so 2 matches 2.0, 1,200 matches 1200, and 3% matches 3). A sentence with a number not found there is removed from the answer, and the reply says how many sentences were removed and why. If no cited sentence is left, the answer is withheld as a failed citation check.

To avoid removing correct sentences, only standalone numbers in the sentence are checked: digits inside names (BM25, GPT-4, Qwen3-8B), citation markers like [2] and list numbering are skipped, and numbers in the passage are read loosely, including digits run into words by PDF extraction. Numbers written as words are not checked.

## Pass rule, measured offline on existing answers

The check is kept only if, on the 48 labelled sentences:

1. it removes item 38, and
2. it removes none of the 45 sentences both labellers judged fully supported.

Also reported, not part of the rule: how many sentences it would remove across all 94 answered questions of 2026-11-03, each one listed for inspection.

## Result: fails rule 1, not kept

Measured offline on 2026-11-04, with no model call.

- Item 38 was **not** flagged. Its figures are 2, 3 and 4, and its passage, like most 2,000-character passages, contains each of those digits somewhere (figure numbers, fold labels, counts). Matching values cannot tell "2% to 4% of targets" from "Figure 4".
- None of the 45 supported sentences was flagged, so rule 2 held, but rule 1 failed.
- Across all 94 answers of 2026-11-03 it would have removed 5 sentences. At least 3 look like false removals: two are list numbers split off by the sentence splitter ("**The examples** [2]\n2."), and one cites "0.723" where the passage most likely gives the same value as a percentage.

The check misses the error it was built for, because small numbers are where invented figures sit and small numbers are in every passage, and it removes correct sentences over formatting. It was removed. A version that matches a number together with its unit or surrounding words could do better, but it would need its own rule, set before measuring, and the error it targets was 1 of 48 sentences.
