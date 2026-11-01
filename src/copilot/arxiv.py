import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}
# arXiv forbids serving e-prints unless the licence allows it, so only these get stored.
ALLOWED_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
}
CHUNK_WORDS = 250
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 20  # page numbers and stray headers are not evidence


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: str
    license: str


def parse_records(xml: bytes) -> list[Paper]:
    papers = []
    for meta in ET.fromstring(xml).iterfind(".//raw:arXivRaw", NS):
        license_url = meta.findtext("raw:license", "", NS).strip()
        if license_url in ALLOWED_LICENSES:
            papers.append(Paper(
                meta.findtext("raw:id", "", NS).strip(),
                " ".join(meta.findtext("raw:title", "", NS).split()),
                " ".join(meta.findtext("raw:authors", "", NS).split()),
                license_url,
            ))
    return papers


def resumption_token(xml: bytes) -> str | None:
    token = ET.fromstring(xml).findtext(".//oai:resumptionToken", None, NS)
    return token.strip() if token and token.strip() else None


def chunk_words(text: str) -> list[str]:
    # PDFs can yield NULs and lone surrogates, which Postgres and the tokenizer reject.
    clean = text.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")
    words = clean.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = [words[i:i + CHUNK_WORDS] for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]
    return [" ".join(c) for c in chunks if len(c) >= MIN_CHUNK_WORDS]


def chunk_pdf(pdf: bytes) -> list[tuple[int, str]]:
    pages = PdfReader(BytesIO(pdf)).pages
    return [(n, c) for n, p in enumerate(pages, 1) for c in chunk_words(p.extract_text() or "")]
