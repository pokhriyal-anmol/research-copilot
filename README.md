# Research Copilot

Ask a question about openly licensed information retrieval papers. Answers cite the paper and page, and code withholds any answer whose citations don't match retrieved passages.

Status: live demo in progress. No production claim yet.

```mermaid
flowchart LR
  user[Reader] -->|asks a question| api[Copilot web app]
  api -->|finds matching passages| db[(Paper database)]
  api -->|sends passages and question| llm[Free model providers, with fallback]
  llm -->|returns a draft answer| api
  api -->|checks citations, then replies| user
  arxiv[arXiv open papers] -->|CC BY and CC0 papers only| db
  classDef store fill:#e8f0fe,stroke:#4a6fa5
  classDef external fill:#fff4e0,stroke:#b8860b
  class db store
  class llm,arxiv external
```

Papers come from arXiv under CC BY 4.0 or CC0. This project is not endorsed by arXiv.
