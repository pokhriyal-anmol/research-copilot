from copilot.arxiv import CHUNK_WORDS, chunk_words, parse_records, resumption_token

XML = b"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
<id>1</id><title>Open
  Paper</title><authors>A</authors><license>http://creativecommons.org/licenses/by/4.0/</license>
</arXivRaw></metadata></record>
<record><metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
<id>2</id><title>Closed</title><authors>B</authors>
<license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
</arXivRaw></metadata></record>
<record><metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
<id>3</id><title>No licence</title><authors>C</authors>
</arXivRaw></metadata></record>
</ListRecords></OAI-PMH>"""


def test_only_allowed_licences_are_kept():
    papers = parse_records(XML)
    assert [p.arxiv_id for p in papers] == ["1"]
    assert papers[0].title == "Open Paper"


def test_chunks_overlap_and_lose_no_words():
    words = [str(i) for i in range(CHUNK_WORDS + 100)]
    chunks = chunk_words(" ".join(words))
    assert len(chunks) == 2
    assert set(" ".join(chunks).split()) == set(words)


def test_empty_page_has_no_chunks():
    assert chunk_words("  ") == []


def test_page_number_alone_is_not_a_chunk():
    assert chunk_words("15") == []


def test_resumption_token_found_and_absent():
    wrap = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">%s</OAI-PMH>'
    assert resumption_token(wrap % b"<resumptionToken>abc</resumptionToken>") == "abc"
    assert resumption_token(wrap % b"<resumptionToken/>") is None
    assert resumption_token(XML) is None


def test_unencodable_pdf_characters_are_removed():
    text = " ".join(["word"] * 30) + " bad\ud835char nul\x00byte"
    chunk = chunk_words(text)[0]
    chunk.encode("utf-8")
    assert "badchar" in chunk and "\x00" not in chunk
