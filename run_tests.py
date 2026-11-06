"""Runs the three required tests and saves the evidence in /evidencias/.

For each test it writes a .txt file (input, Top-5 with scores, comment)
and a .png console capture.

    python run_tests.py
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from main import run_by_id, run_by_text
from recommender import TfidfRecommender, load_corpus

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidencias"
FONT_CANDIDATES = ("DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "cour.ttf")

TESTS = [
    {
        "file": "test1_policy_process",
        "title": "Test 1 - Seed document from Policy/Process",
        "mode": "id",
        "value": "DOC001",
        "comment": (
            "Coherent. The top result (DOC034, FAQ: How Do I Get an Expense Reimbursed?) "
            "is the most useful match: it shares the key terms expense, receipts, approval "
            "and reimbursed with the travel expense policy. DOC004 (accounting close) is "
            "also finance-related. DOC003, DOC017 and DOC039 have low scores (< 0.10); they "
            "appear because they share generic policy phrases such as 'approved by the "
            "direct manager' and 'employees', not the same topic."
        ),
    },
    {
        "file": "test2_support_incidents",
        "title": "Test 2 - Seed document from Support/Incidents",
        "mode": "id",
        "value": "DOC009",
        "comment": (
            "Coherent. 4 of the 5 results are Support/Incidents tickets, and the FAQ on "
            "how to connect to the VPN (DOC035) is exactly the knowledge an agent would "
            "need to solve this ticket. DOC015 ranks first because it shares 'access' and "
            "'error' plus the whole ticket template ('user reports', 'support verified', "
            "'status: closed', 'priority: medium'). The lower results (printer, laptop) match "
            "mainly that template, which shows that TF-IDF also picks up the document "
            "format, not only the topic."
        ),
    },
    {
        "file": "test3_free_text",
        "title": "Test 3 - Free-text query",
        "mode": "text",
        "value": "I received a suspicious email asking me to confirm my password, what should I do?",
        "comment": (
            "Mostly coherent. The phishing alert (DOC026), the MFA policy (DOC030) and the "
            "password policy (DOC025) are the right documents for this question. However, "
            "the VPN FAQ (DOC035) sneaks in only because it mentions 'password', and "
            "the first result is the password reset ticket (DOC012), because it repeats "
            "the literal words 'email' and 'password' more often. The query never says "
            "'phishing', so TF-IDF cannot know that 'suspicious email' means phishing: a "
            "clear example of the synonym limitation that embeddings would reduce."
        ),
    },
    {
        "file": "extra_short_document",
        "title": "Extra - Limitation with a short, ambiguous document",
        "mode": "id",
        "value": "DOC016",
        "comment": (
            "Not coherent, on purpose. DOC016 ('Ticket: Not Working') has only 9 words and "
            "no specific system named. All scores are below 0.04, so the ranking is close "
            "to random and even includes a vacation FAQ. This shows the limitation of very "
            "short documents: a real system should flag low-confidence results (e.g., "
            "score < 0.10) instead of showing them as recommendations."
        ),
    },
]


def capture(rec: TfidfRecommender, test: dict) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if test["mode"] == "id":
            print(f"$ python main.py --id {test['value']}")
            run_by_id(rec, test["value"], 5)
        else:
            print(f'$ python main.py --text "{test["value"]}"')
            run_by_text(rec, test["value"], 5)
    return buffer.getvalue().rstrip()


def wrap(text: str, width: int = 96) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def load_font(size: int) -> ImageFont.ImageFont:
    """Use a monospace font available on Linux, Windows or macOS."""
    for name in FONT_CANDIDATES:
        with contextlib.suppress(OSError):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def render_png(title: str, console: str, comment: str, path: Path) -> None:
    font = load_font(22)
    lines = [(f"# {title}", "#7fd1ff"), ("", None)]
    lines += [(line, "#e6e6e6") for line in console.splitlines()]
    lines += [("", None), ("# Comment:", "#9be37a")]
    lines += [(line, "#9be37a") for line in wrap(comment)]
    width, line_h, pad = 1500, 30, 40
    img = Image.new("RGB", (width, pad * 2 + 40 + line_h * len(lines)), "#1e1e1e")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 40], fill="#333333")
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        draw.ellipse([16 + i * 28, 12, 32 + i * 28, 28], fill=color)
    draw.text((120, 8), "Terminal - document-recommender", font=font, fill="#bbbbbb")
    for i, (line, color) in enumerate(lines):
        if line:
            draw.text((pad, 40 + pad + i * line_h), line, font=font, fill=color)
    img.save(path)


def main() -> None:
    EVIDENCE_DIR.mkdir(exist_ok=True)
    rec = TfidfRecommender(load_corpus())
    for test in TESTS:
        console = capture(rec, test)
        report = f"{test['title']}\n{'=' * len(test['title'])}\n\n{console}\n\nComment:\n{test['comment']}\n"
        (EVIDENCE_DIR / f"{test['file']}.txt").write_text(report, encoding="utf-8")
        render_png(test["title"], console, test["comment"], EVIDENCE_DIR / f"{test['file']}.png")
        print(report)


if __name__ == "__main__":
    main()
