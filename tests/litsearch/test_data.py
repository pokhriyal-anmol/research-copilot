"""Loading rules from bench/BENCHMARK.md, checked on tiny parquet files shaped like LitSearch."""
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from copilot.litsearch.data import doc_text, load


def write(root, queries, corpus):
    (root / "query").mkdir()
    (root / "corpus_clean").mkdir()
    pd.DataFrame(queries).to_parquet(root / "query" / "full-00000-of-00001.parquet")
    pd.DataFrame(corpus).to_parquet(root / "corpus_clean" / "full-00000-of-00001.parquet")


def query(text, ids, qs="manual_acl", sp=1):
    return {"query_set": qs, "query": text, "specificity": sp, "quality": 2, "corpusids": ids}


def paper(cid, title, abstract, cites=()):
    return {"corpusid": cid, "title": title, "abstract": abstract, "citations": list(cites)}


def test_doc_text_joins_title_and_abstract_with_a_blank_line():
    assert doc_text("Title", "Abstract") == "Title\n\nAbstract"


@pytest.mark.parametrize("missing", ["", None, "   "])
def test_doc_text_is_the_title_alone_without_an_abstract(missing):
    assert doc_text("Title", missing) == "Title"


def test_load_returns_questions_docs_and_citations(tmp_path):
    write(tmp_path, [query("q one", [1])], [paper(1, "A", "x", cites=[2]), paper(2, "B", "")])
    data = load(tmp_path)
    assert data.docs == {1: "A\n\nx", 2: "B"}
    assert data.citations == {1: [2], 2: []}
    [q] = data.questions
    assert (q.text, q.gold, q.query_set, q.specificity) == ("q one", {1}, "manual_acl", 1)
    assert data.no_abstract == {2}


def test_question_ids_are_stable_and_unique(tmp_path):
    write(tmp_path, [query("same", [1]), query("same", [1])], [paper(1, "A", "x")])
    ids = [q.id for q in load(tmp_path).questions]
    assert len(set(ids)) == 2 and ids == [q.id for q in load(tmp_path).questions]


def test_duplicate_paper_ids_are_an_error(tmp_path):
    write(tmp_path, [query("q", [1])], [paper(1, "A", "x"), paper(1, "A again", "y")])
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path)


def test_gold_papers_missing_from_the_corpus_are_reported_not_dropped(tmp_path):
    write(tmp_path, [query("q", [1, 99])], [paper(1, "A", "x")])
    data = load(tmp_path)
    assert data.questions[0].gold == {1, 99}
    assert data.missing_gold == {99}
