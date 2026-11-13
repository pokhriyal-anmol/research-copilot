# Research Copilot

Finds the paper a researcher means and answers questions about it, with every citation checked against its source. Built on open models and free compute (a MacBook and a free Kaggle GPU).

**Try it: [research-copilot-black.vercel.app](https://research-copilot-black.vercel.app).** Pick an example question and the answer comes back with each citation linking to its paper on arxiv.org. A question outside the papers gets no answer. The free server sleeps after 15 idle minutes, and the page says so while it wakes (up to a minute). Each visitor gets 10 answers an hour and the service 200 a day, so a shared link can't use up the free model quota.

## Results

| What was measured | Result | Best comparison | Why it matters |
|---|---|---|---|
| Right paper in the top 20, [LitSearch](https://arxiv.org/abs/2407.18940) held-out test (397 questions, 64,183 papers) | **85.3%** | 81.8%, best published (PairSem, [arXiv 2510.09897](https://arxiv.org/abs/2510.09897)) | A paper that isn't retrieved can't be read or cited. Search sets the ceiling for everything after it. |
| Ranking quality, same test (nDCG@10) | **64.1** | 61.1, same paper | Researchers read the first few results, so the right paper has to be near the top as well as in the list. |
| Answer sentences fully backed by the passages they cite | **45 of 48** (94%, 95% interval 83 to 98%) | | A citation that doesn't support its sentence is the error that costs a researcher credibility, and the hardest one for them to spot. |
| Right page in the top 20 for the demo's own questions (56 reviewed) | **89.3%** (MRR 0.826) | | The demo's search over 299 openly licensed papers works on questions that name what they ask about. |

The first two rows used rules written down before any run ([bench/BENCHMARK.md](bench/BENCHMARK.md)): every choice was made on 200 development questions, and the 397 test questions were scored once, at the end. All four numbers were recomputed from saved results on 2026-11-04.

```mermaid
flowchart TB
  subgraph find["Finding papers (benchmark)"]
    direction LR
    q[Researcher's question] -->|matched against 64,183 abstracts| embed[Search model<br/>Qwen3-Embedding-8B]
    embed -->|100 closest papers| judge[Judge model<br/>Qwen3-Reranker-4B]
    judge -->|reorders by fit| ranked[Ranked papers]
  end
  subgraph answer["Answering with citations (demo)"]
    direction LR
    ask[Question] -->|search and rerank| passages[(299 open papers,<br/>16,290 passages)]
    passages -->|top passages| llm[Language model]
    llm -->|draft with numbered citations| check{Citation check<br/>in code}
    check -->|every citation matches a given passage| reply[Answer with sources]
    check -->|any citation doesn't| withheld[Answer withheld]
  end
  find ~~~ answer
  classDef model fill:#e8f0fe,stroke:#4a6fa5
  classDef store fill:#fff4e0,stroke:#b8860b
  class embed,judge,llm model
  class passages store
```

## How the numbers were kept honest

Methods that failed their pre-written rule were dropped and recorded, not tuned until they passed. Three published ways of using the citation graph all failed on development questions: adding papers one citation away cut top-20 recall from 85.2% to 72.0%. A free-tier Gemini reranker was stopped because its 20 requests a day would have made a full run take 30 days. The [change log](bench/BENCHMARK.md#change-log) has every attempt with its date and outcome.

Answer quality was judged by two AI labellers working separately, who agreed on all 48 sentences. The three failures are a figure the passage doesn't give and two overstatements; their quoted reasons are in [evals/results](evals/results/2026-11-03-sentence-labels.json).

## Limits

- The nDCG@10 lead on test is 2.99 points, just under the 3-point margin set in advance as "clearly better". Recall@20 clears it.
- PairSem's own baseline doesn't reproduce here: the same search model scores higher in this setup (81.6 against their 79.2 on all 597 questions), so part of the lead may be setup rather than method.
- Both Qwen3 models were released after LitSearch was public and may have seen its papers.
- Against LitSearch's own GPT-4o reranker, this pipeline wins on broad questions (79.1% against 75.3% in the top 20) and loses on specific ones (76.8% against 79.2% in the top 5).
- Answer quality was judged by AI labellers, not people.

The full write-up, with breakdowns and the next experiment, is in [docs/litsearch.md](docs/litsearch.md).

## Repository

| Path | What's there |
|---|---|
| `src/copilot/litsearch/` | Benchmark pipeline: data loading, search, reranking, metrics |
| `bench/` | Pre-registered protocol, run scripts and result files |
| `src/copilot/` | The demo: arXiv ingest, hybrid search, answering, citation check, API |
| `evals/` | Question sets, answers and labels for the demo |
| `web/` | Next.js front end for the demo |

## Run it

```bash
uv sync
docker compose up -d
cp .env.example .env
TEST_DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot_test uv run pytest
```

To reproduce the benchmark (the paper vectors were built on a free Kaggle GPU, and reranking the test split took 4.1 hours on a MacBook):

```bash
uv sync --group bench
HF_HUB_OFFLINE=1 uv run --group bench python bench/rerank/run.py --reranker qwen3-4b --split test
HF_HUB_OFFLINE=1 uv run --group bench python bench/rerank/native.py
```

Papers come from arXiv under CC BY 4.0 or CC0, and citations link back to arxiv.org. This project is not endorsed by arXiv. LitSearch states no licence, so it is used for offline benchmarking only.
