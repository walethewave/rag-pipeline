"""Tests for pipeline.py's clean_pages() step."""

from src.pipeline import Page, clean_pages


def test_repeated_header_and_footer_are_removed():
    """A line that repeats on most pages (a running header/footer) is stripped from every page."""
    pages = [
        Page(page_number=1, text="Acme Inc - FAQ\nFirst question here.\nPage 1"),
        Page(page_number=2, text="Acme Inc - FAQ\nSecond question here.\nPage 2"),
        Page(page_number=3, text="Acme Inc - FAQ\nThird question here.\nPage 3"),
    ]
    cleaned = clean_pages(pages)
    assert "Acme Inc - FAQ" not in cleaned
    assert "Page 1" not in cleaned and "Page 2" not in cleaned
    assert "First question here." in cleaned
    assert "Second question here." in cleaned


def test_hyphenated_line_break_is_joined():
    """A word split across a line break with a hyphen is rejoined into one word."""
    pages = [Page(page_number=1, text="This is informa-\ntion that matters.")]
    cleaned = clean_pages(pages)
    assert "information that matters" in cleaned


def test_short_lines_inside_a_paragraph_are_joined_with_spaces():
    """Soft-wrapped lines within one paragraph become one line, not three."""
    pages = [Page(page_number=1, text="This sentence\nwas wrapped\nacross lines.")]
    cleaned = clean_pages(pages)
    assert "This sentence was wrapped across lines." in cleaned


def test_too_few_pages_skips_header_footer_detection():
    """With fewer than 3 pages we cannot reliably tell a real header from a coincidence, so
    nothing is treated as a header/footer."""
    pages = [Page(page_number=1, text="Title\nSome content.")]
    cleaned = clean_pages(pages)
    assert "Title" in cleaned
