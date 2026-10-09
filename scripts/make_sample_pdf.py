"""Creates a fake FAQ PDF for manual testing and for the automated end-to-end run.

Run with: python scripts/make_sample_pdf.py [output_path]
"""

import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# 12 Q&As covering different formats the chunker must detect, plus one long answer
# (to exercise the recursive-splitter fallback) and one numbered section.
FAQS = [
    ("Q: How do I reset my password?",
     "A: Go to the login page and click 'Forgot password'. Enter your work email and we will "
     "send a reset link that is valid for 30 minutes. If the email does not arrive, check your "
     "spam folder before contacting IT support."),
    ("Q: What are the support hours?",
     "A: Our support team is available Monday to Friday, 9am to 6pm West Africa Time. We are "
     "closed on public holidays."),
    ("Question: How do I update my billing information?",
     "Answer: Open Settings, then Billing, then click 'Edit payment method'. Changes take effect "
     "on your next billing cycle."),
    ("Question: Can I cancel my subscription at any time?",
     "Answer: Yes. Cancel from Settings > Subscription > Cancel plan. You keep access until the "
     "end of the period you already paid for."),
    ("1. How do I invite a teammate?",
     "From the Team page, click 'Invite member', enter their email, and choose a role. They "
     "receive an invite email with a link that expires after 7 days."),
    ("2. How do I remove a teammate?",
     "Go to Team, find the member, and click 'Remove'. This cannot be undone and their data "
     "stays, but their account access is revoked immediately."),
    ("What file types can I upload?",
     "You can upload PDF, DOCX and TXT files up to 20MB each. Scanned image-only PDFs are not "
     "supported yet because we do not run OCR on them."),
    ("Is my data encrypted?",
     "Yes. All data is encrypted in transit with TLS and at rest with AES-256. We never share "
     "customer data with third parties without consent."),
    ("Do you offer a free trial?",
     "Yes, every new account gets a 14-day free trial with full access to all features. No "
     "credit card is required to start the trial."),
    ("How do I export my data?",
     "Go to Settings > Data > Export. You will receive a downloadable ZIP file by email within "
     "a few minutes, containing your data in CSV format."),
    ("What happens to my data if I cancel?",
     # Deliberately long answer (~700+ tokens) to force the recursive-splitter fallback.
     "Your account data is retained for 90 days after cancellation in case you want to "
     "reactivate your subscription and pick up exactly where you left off. " * 20 +
     "After the 90 day window closes, all of your data is permanently and irreversibly deleted "
     "from our production systems, and within a further 30 days it is also purged from backups. "
     "We recommend exporting your data before cancelling if you think you may need it later, "
     "since support cannot recover data once the retention window has closed. " * 10),
    ("Who do I contact for billing questions?",
     "Email billing@example.com and include your account email and invoice number. We reply "
     "within one business day."),
]


def build_pdf(output_path: str) -> None:
    """Writes a 2-page FAQ PDF with a repeated header/footer, so the cleaner has something to strip."""
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 20 * mm

    def draw_header_footer(page_num: int):
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(margin, height - 12 * mm, "Acme Inc - Frequently Asked Questions")
        c.drawString(margin, 10 * mm, f"Page {page_num}")

    y = height - margin
    page_num = 1
    draw_header_footer(page_num)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Frequently Asked Questions")
    y -= 14 * mm

    for question, answer in FAQS:
        block_lines = _wrap(question, 90) + _wrap(answer, 90) + [""]
        needed = len(block_lines) * 5 * mm
        if y - needed < margin:
            c.showPage()
            page_num += 1
            draw_header_footer(page_num)
            y = height - margin - 10 * mm
        c.setFont("Helvetica-Bold", 11)
        for line in _wrap(question, 90):
            c.drawString(margin, y, line)
            y -= 5 * mm
        c.setFont("Helvetica", 10)
        for line in _wrap(answer, 90):
            c.drawString(margin, y, line)
            y -= 5 * mm
        y -= 4 * mm

    c.save()


def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap so long lines do not run off the page edge."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_faq.pdf"
    build_pdf(out)
    print(f"Wrote sample FAQ PDF to {out}")
