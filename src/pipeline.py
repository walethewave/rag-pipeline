"""STEP 1 of the RAG pipeline: turn a raw PDF into small, searchable pieces of text ("chunks").

RAG stands for Retrieval-Augmented Generation. Instead of asking an AI model a question and
hoping it "remembers" the right answer, we first search our own documents for the relevant
piece of text, then hand that piece to the AI and say "answer using only this". This file is
the "getting documents ready to be searched" half of that - the other half (actually searching,
and asking the AI) lives in storage.py and ai.py.

Three steps happen here, in order, every time someone uploads a PDF:
1. load_pdf      - read the raw text out of the PDF file, one page at a time
2. clean_pages   - throw away junk (repeated headers, page numbers, broken line breaks)
3. chunk_document - cut the clean text into small pieces, ideally one Question+Answer per piece

Called by: main.py's _ingest() function, right after a user uploads a file.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import pdfplumber
from pypdf import PdfReader

from src.errors import UnsupportedPDFError

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 15

# Three different ways a question can be written in a document. We try all three so we can
# handle PDFs that use "Q: ... A: ...", numbered lists like "1. ... ?", or just a plain sentence
# ending in a question mark with no special formatting at all.
_QA_PATTERNS = (
    re.compile(r"\b(?:Q|Question)\s*[:.]\s*(?P<question>.+?\??)\s*\b(?:A|Answer)\s*[:.]\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?<!\d)(?:\d{1,3})\.\s*(?P<question>[^.?!]{3,200}\?)\s*", re.DOTALL),
    re.compile(r"(?P<question>[A-Z][^.!?]{3,200}\?)\s*", re.DOTALL),
)


@dataclass
class Page:
    """One page of a PDF: which page number it was, and the plain text pdfplumber/pypdf pulled
    out of it. Produced by load_pdf(), consumed by clean_pages()."""
    page_number: int
    text: str


@dataclass
class Chunk:
    """One small, self-contained piece of a document - usually one Question + its Answer.
    `id` is how we find it again later (format: '<doc_id>-<0001>'). `vector` starts empty and
    gets filled in afterwards by ai.py's Embedder - a "vector" here just means a list of numbers
    that represents what the text *means*, so a computer can compare two pieces of text for
    similarity by comparing their numbers instead of their words."""
    id: str
    text: str
    vector: list[float] = field(default_factory=list)


def count_tokens(text: str) -> int:
    """A rough stand-in for 'how much text is this, in units an AI model understands'. A real
    token counter is more precise, but counting words is close enough to budget chunk sizes by,
    and needs no extra library. Used throughout this file to decide when a piece of text is
    'too big' and needs to be split further."""
    return len(text.split())


# ==================================================================== 1. Load PDF


def load_pdf(path: str) -> list[Page]:
    """Reads a PDF file off disk and returns its text, one Page per page of the PDF.

    Tries pdfplumber first (it's generally better at reading layout/columns correctly). If
    pdfplumber chokes on a specific page, we fall back to pypdf just for that one page, so one
    broken page doesn't ruin the whole upload. If NO page produced any text at all - which
    usually means the PDF is a scanned image with no real text layer - we raise
    UnsupportedPDFError instead of silently returning nothing, so the person uploading finds out
    immediately rather than getting a document that can never be found by search later.

    Called by: main.py's _ingest(). Calls into: pdfplumber, and pypdf as a fallback.
    """
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        reader = None
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdfplumber failed on page %d (%s), using pypdf fallback", i, exc)
                if reader is None:
                    reader = PdfReader(path)
                text = reader.pages[i - 1].extract_text() or "" if (i - 1) < len(reader.pages) else ""
            pages.append(Page(page_number=i, text=text))

    if not any(p.text.strip() for p in pages):
        raise UnsupportedPDFError()
    logger.info("Loaded %d pages from %s", len(pages), path)
    return pages


# ==================================================================== 2. Clean Pages


def clean_pages(pages: list[Page]) -> str:
    """Tidies up the raw text before we chunk it, and joins every page into one long string.

    Three kinds of "noise" get removed, because if we left them in, they would end up stuffed
    into random chunks and waste space that should be real content:
    - Running headers/footers: a line (like a company name or document title) that repeats on
      at least half the pages is almost certainly decoration, not content, so we drop it
      everywhere it appears. We only do this with 3+ pages, since with fewer pages we can't tell
      a real repeat from a coincidence.
    - Page-number-only lines: things like "3" or "Page 3 of 10" on their own line.
    - Broken formatting: PDFs often hyphenate words across a line break ("informa-\\ntion"), and
      wrap a single sentence across several short lines. We rejoin both of these so sentences
      read naturally again, which matters because chunk_document() looks for whole sentences.

    Called by: main.py's _ingest(), right after load_pdf(). Returns one big string, which is
    what chunk_document() then cuts into pieces.
    """
    # Running headers/footers: lines repeated across >= 50% of pages (only if 3+ pages)
    header_footer = set()
    if len(pages) >= 3:
        counts = Counter()
        for p in pages:
            counts.update({line.strip() for line in p.text.splitlines() if line.strip()})
        threshold = max(2, int(len(pages) * 0.5))
        header_footer = {line for line, c in counts.items() if c >= threshold}

    is_page_num = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)

    cleaned_pages = []
    for p in pages:
        lines = []
        for raw in p.text.splitlines():
            line = raw.strip()
            if not line:
                lines.append("")
            elif line not in header_footer and not is_page_num.match(line):
                lines.append(line)

        joined = "\n".join(lines)
        joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)  # "informa-\ntion" -> "information"
        joined = re.sub(r"(?<!\n)\n(?!\n)", " ", joined)  # single newline -> space
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        if joined.strip():
            cleaned_pages.append(joined.strip())

    return "\n\n".join(cleaned_pages)


# ==================================================================== 3. Recursive Split


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """A fallback splitter for text that is too big to be one chunk on its own - e.g. one Q&A
    whose answer alone is longer than chunk_size tokens, or (rarely) a whole document with no
    detectable questions in it at all.

    "chunk_size" is the biggest a piece is allowed to be (measured by count_tokens). "overlap" is
    how many words from the END of one piece get repeated at the START of the next one - this
    matters because if we cut a sentence exactly in half with no overlap, a search later might
    only find one half and miss the full meaning; repeating a little context across the cut
    reduces that risk.

    How it splits: it tries the biggest, most natural break first (a blank line between
    paragraphs), and only falls back to smaller breaks (single line, then sentence, then word)
    for whichever piece is still too big after the bigger break. This keeps pieces reading like
    real sentences/paragraphs instead of getting chopped mid-word.

    Called by: chunk_document(), for any answer that's too long to fit in one chunk, and as the
    last-resort path when a document has no Q&A structure the detector recognizes at all.
    """
    if count_tokens(text) <= chunk_size or not text.strip():
        return [text] if text.strip() else []

    def _split(s: str, seps: list[str]) -> list[str]:
        """Tries the biggest separator first (paragraph, then line, then sentence, then word),
        recursing into whichever separator is next in `seps` only for a piece that's still too
        big - this is what keeps pieces as readable as possible instead of always cutting on
        single words. Defined inside recursive_split() since it's only ever called from here
        and needs chunk_size from the enclosing scope."""
        if count_tokens(s) <= chunk_size or not s.strip():
            return [s] if s.strip() else []
        sep = seps[0] if seps else " "
        remaining = seps[1:]
        parts = s.split(sep)
        pieces: list[str] = []
        curr = ""
        for part in parts:
            cand = f"{curr}{sep}{part}" if curr else part
            if count_tokens(cand) <= chunk_size:
                curr = cand
            else:
                if curr:
                    pieces.append(curr)
                if count_tokens(part) > chunk_size and remaining:
                    pieces.extend(_split(part, remaining))
                    curr = ""
                else:
                    curr = part
        if curr:
            pieces.append(curr)
        return pieces

    pieces = _split(text, ["\n\n", "\n", ". ", " "])

    # Apply overlap across chunk cuts: repeat the last `overlap` words of each piece at the
    # start of the next one, so a reader (or search) never sees a piece totally cut off from
    # what came right before it.
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    result = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = " ".join(pieces[i - 1].split()[-overlap:])
        result.append(f"{tail} {pieces[i]}".strip())
    return result


# ==================================================================== 4. Chunk Document


def chunk_document(text: str, doc_id: str, max_tokens: int = 500, overlap: int = 50) -> list[Chunk]:
    """The main chunking function - turns one cleaned document into a list of Chunk objects
    ready to be embedded and stored. This is the single most important design decision in the
    whole RAG pipeline: **one chunk = one Question and its Answer**.

    Why one chunk per Q&A, instead of just splitting every 500 words? Because later, when a user
    asks a question, we search for the single chunk whose *meaning* is closest to that question.
    If a chunk mixed together three unrelated questions, the search would get confused about
    which part of the chunk actually matches. Keeping each chunk to one self-contained idea
    makes that search far more accurate.

    How it works, step by step:
    1. Scan the text with the three question patterns from _QA_PATTERNS (see top of file) and
       record where every question starts. If the same text could match two different patterns,
       whichever pattern found it first wins (patterns are tried in priority order), so we never
       detect the same question twice.
    2. For each question found, its answer is simply "everything between the end of this
       question and the start of the next one". This naturally handles answers of any length
       without needing special-case logic.
    3. Format each pair as "Question: ...\\nAnswer: ..." - writing it this way (rather than just
       the raw answer) helps the search step later, because the question's own wording becomes
       part of what gets compared against a user's question.
    4. If a formatted Q&A is still bigger than max_tokens, its answer gets cut up by
       recursive_split() into several chunks, with the *same question repeated* at the top of
       each one - so every chunk is still self-contained and makes sense on its own.
    5. If no questions were found anywhere in the whole document, we give up on the Q&A
       structure and just run recursive_split() on the entire text instead, so the document is
       still searchable even if it isn't formatted as a FAQ.

    Called by: main.py's _ingest(), right after clean_pages(). Calls into: recursive_split() for
    oversized answers or documents with no detected structure.
    """
    # 1. Detect question positions
    claimed: list[tuple[int, int]] = []
    events: list[dict] = []
    for pattern in _QA_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end in claimed):
                continue
            q = m.group("question").strip()
            if q:
                events.append({"start": m.start(), "ans_start": m.end(), "question": q})
                claimed.append((m.start(), m.end()))

    events.sort(key=lambda e: e["start"])

    # 2. Extract Q&A text spans
    raw_pieces: list[str] = []
    if not events:
        logger.warning("No Q&A structure detected in doc %s, using recursive split", doc_id)
        raw_pieces = recursive_split(text, max_tokens, overlap)
    else:
        for i, ev in enumerate(events):
            ans_end = events[i + 1]["start"] if i + 1 < len(events) else len(text)
            ans = text[ev["ans_start"]:ans_end].strip()
            if not ans:
                continue
            q = ev["question"].strip().rstrip("?.") + "?"
            full_qa = f"Question: {q}\nAnswer: {ans}"
            if count_tokens(full_qa) <= max_tokens:
                raw_pieces.append(full_qa)
            else:
                header = f"Question: {q}\nAnswer: "
                budget = max(max_tokens - count_tokens(header), 50)
                raw_pieces.extend(f"{header}{piece}" for piece in recursive_split(ans, budget, overlap))

    # 3. Build Chunk objects (drop fragments < MIN_CHUNK_CHARS - too short to be useful)
    return [
        Chunk(id=f"{doc_id}-{idx:04d}", text=p.strip())
        for idx, p in enumerate(p for p in raw_pieces if len(p.strip()) >= MIN_CHUNK_CHARS)
    ]
