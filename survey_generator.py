#!/usr/bin/env python3
"""
survey_generator.py
====================
Generates a mixed-type survey questionnaire from pasted PDF text using
rule-based extraction (no LLM / API keys required).

Usage
-----
  # Interactive (paste text at the prompt):
  python survey_generator.py

  # From a file:
  python survey_generator.py my_report.txt

Output
------
  survey_questions.json  — structured list of survey questions
"""

import io
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

# ---------------------------------------------------------------------------
# NLTK bootstrap (auto-downloads required data on first run)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    import nltk  # noqa: PLC0415

    for resource in [
        "tokenizers/punkt_tab",
        "tokenizers/punkt",
        "corpora/stopwords",
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            name = resource.split("/")[-1]
            try:
                with redirect_stderr(io.StringIO()):
                    nltk.download(name, quiet=True)
            except Exception:
                pass  # SSL / network unavailable — regex fallbacks will be used


def _get_sentences(text: str) -> list[str]:
    try:
        import nltk  # noqa: PLC0415
        _ensure_nltk()
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 20]


def _get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords  # noqa: PLC0415
        _ensure_nltk()
        words = set(stopwords.words("english"))
        if not words:
            raise ValueError("empty")
        return words
    except Exception:
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "this", "that",
            "these", "those", "it", "its", "they", "them", "their", "we", "our",
            "you", "your", "he", "she", "his", "her", "i", "my", "me", "also",
            "such", "than", "more", "can", "all", "one", "two", "three", "use",
            "used", "using", "including", "based", "report", "study", "research",
        }


# ---------------------------------------------------------------------------
# Signal words that flag sentences containing key findings
# ---------------------------------------------------------------------------

_FINDING_SIGNALS: set[str] = {
    "found", "find", "shows", "show", "indicates", "indicate",
    "suggests", "suggest", "reveals", "reveal", "demonstrates",
    "demonstrate", "reports", "report", "notes", "note",
    "highlights", "highlight", "emphasizes", "emphasize",
    "concludes", "conclude", "recommends", "recommend",
    "identified", "identify", "observed", "observe",
    "determined", "determine", "established", "establish",
    "increased", "decreased", "improved", "reduced", "significant",
    "majority", "minority", "average", "total", "overall",
}

_NUMERIC_RE = re.compile(
    r"\b\d+\.?\d*\s*%|\b\d+\s+(?:percent|percentage)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Keyword / keyphrase extraction (frequency-based, no external NLP deps)
# ---------------------------------------------------------------------------

def _extract_keyphrases(text: str, top_n: int = 12) -> list[str]:
    """Return the most significant two-word and three-word phrases."""
    stop = _get_stopwords()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    bigrams = [
        f"{words[i]} {words[i + 1]}"
        for i in range(len(words) - 1)
        if words[i] not in stop and words[i + 1] not in stop
    ]
    trigrams = [
        f"{words[i]} {words[i + 1]} {words[i + 2]}"
        for i in range(len(words) - 2)
        if words[i] not in stop and words[i + 2] not in stop
    ]

    freq = Counter(bigrams + trigrams)
    # Filter out phrases that appear only once — likely noise
    return [phrase for phrase, count in freq.most_common(top_n) if count > 1]


def _extract_single_keywords(text: str, top_n: int = 10) -> list[str]:
    """Return the most frequent meaningful single words."""
    stop = _get_stopwords()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in stop]
    return [word for word, _ in Counter(filtered).most_common(top_n)]


# ---------------------------------------------------------------------------
# Finding sentence extraction
# ---------------------------------------------------------------------------

def _extract_finding_sentences(sentences: list[str], max_findings: int = 8) -> list[str]:
    """Return sentences that contain signal words or numeric statistics."""
    findings: list[str] = []
    seen: set[str] = set()

    for sent in sentences:
        lower = sent.lower()
        words_in_sent = set(re.findall(r"\b[a-z]+\b", lower))
        has_signal = bool(words_in_sent & _FINDING_SIGNALS)
        has_numeric = bool(_NUMERIC_RE.search(sent))

        if (has_signal or has_numeric) and sent not in seen and len(sent) > 30:
            findings.append(sent)
            seen.add(sent)

        if len(findings) >= max_findings:
            break

    return findings


# ---------------------------------------------------------------------------
# Question option sets
# ---------------------------------------------------------------------------

_LIKERT_AGREE = [
    "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"
]
_LIKERT_IMPORTANCE = [
    "Very Important", "Important", "Neutral", "Unimportant", "Very Unimportant"
]
_LIKERT_SATISFACTION = [
    "Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"
]
_LIKERT_FREQUENCY = ["Always", "Often", "Sometimes", "Rarely", "Never"]
_LIKERT_EFFECTIVENESS = [
    "Very Effective", "Effective", "Neutral", "Ineffective", "Very Ineffective"
]
_LIKERT_LIKELIHOOD = [
    "Very Likely", "Likely", "Unsure", "Unlikely", "Very Unlikely"
]

_RELEVANCE_OPTIONS = [
    "Highly Relevant", "Relevant", "Somewhat Relevant", "Not Relevant"
]

_ROLE_OPTIONS = [
    "Executive / Senior Leadership",
    "Manager / Team Lead",
    "Individual Contributor",
    "Consultant / Advisor",
    "Other",
]

_SECTOR_OPTIONS = [
    "Public Sector / Government",
    "Private Sector",
    "Non-profit / NGO",
    "Academia / Research",
    "Other",
]


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------

def _q(q_id: int, q_type: str, question: str, options: list[str] | None = None) -> dict:
    entry: dict = {"id": q_id, "type": q_type, "question": question}
    if options:
        entry["options"] = options
    return entry


def _truncate(text: str, max_len: int = 200) -> str:
    return text if len(text) <= max_len else text[:max_len - 3].rstrip() + "..."


def build_survey(text: str) -> list[dict]:
    """Main entry point: analyse text and return a list of question dicts."""
    sentences = _get_sentences(text)
    findings = _extract_finding_sentences(sentences)
    keyphrases = _extract_keyphrases(text)
    keywords = _extract_single_keywords(text)

    # Fall back to single keywords if keyphrase extraction found nothing
    topics = keyphrases if keyphrases else keywords

    questions: list[dict] = []
    q_id = 1

    # ------------------------------------------------------------------
    # Section 1 — Respondent background
    # ------------------------------------------------------------------
    questions.append(_q(q_id, "multiple_choice",
                        "Which of the following best describes your role?", _ROLE_OPTIONS))
    q_id += 1

    questions.append(_q(q_id, "multiple_choice",
                        "Which sector does your organization operate in?", _SECTOR_OPTIONS))
    q_id += 1

    questions.append(_q(q_id, "multiple_choice",
                        "How long have you been working in your current field?",
                        ["Less than 2 years", "2–5 years", "6–10 years", "More than 10 years"]))
    q_id += 1

    # ------------------------------------------------------------------
    # Section 2 — Likert scale questions derived from key findings
    # ------------------------------------------------------------------
    for finding in findings[:6]:
        questions.append(_q(
            q_id, "likert",
            f'To what extent do you agree with the following statement: '
            f'"{_truncate(finding)}"',
            _LIKERT_AGREE,
        ))
        q_id += 1

    # ------------------------------------------------------------------
    # Section 3 — Importance ratings for key topics
    # ------------------------------------------------------------------
    for phrase in topics[:4]:
        questions.append(_q(
            q_id, "likert",
            f"How important is {phrase.title()} to your organization?",
            _LIKERT_IMPORTANCE,
        ))
        q_id += 1

    # ------------------------------------------------------------------
    # Section 4 — Satisfaction and effectiveness
    # ------------------------------------------------------------------
    if topics:
        primary = topics[0].title()
        questions.append(_q(
            q_id, "likert",
            f"How satisfied are you with how your organization currently "
            f"addresses {primary}?",
            _LIKERT_SATISFACTION,
        ))
        q_id += 1

    if len(topics) > 1:
        second = topics[1].title()
        questions.append(_q(
            q_id, "likert",
            f"How effective are your organization's current strategies "
            f"related to {second}?",
            _LIKERT_EFFECTIVENESS,
        ))
        q_id += 1

    # ------------------------------------------------------------------
    # Section 5 — Frequency and likelihood
    # ------------------------------------------------------------------
    if len(topics) > 2:
        third = topics[2].title()
        questions.append(_q(
            q_id, "multiple_choice",
            f"How frequently does your organization review or update its "
            f"approach to {third}?",
            _LIKERT_FREQUENCY,
        ))
        q_id += 1

    if len(topics) > 3:
        fourth = topics[3].title()
        questions.append(_q(
            q_id, "likert",
            f"How likely is your organization to take action on "
            f"{fourth} in the next 12 months?",
            _LIKERT_LIKELIHOOD,
        ))
        q_id += 1

    # ------------------------------------------------------------------
    # Section 6 — Open-ended questions
    # ------------------------------------------------------------------
    open_ended = [
        "What are the most significant challenges your organization faces "
        "that relate to the topics covered in this report?",
        "What actions has your organization already taken in response to "
        "findings or recommendations similar to those in this report?",
        "What additional resources, tools, or support would help your "
        "organization improve in the areas highlighted by this report?",
        "Is there anything important about your organization's context "
        "that is not captured by the questions above? Please describe.",
    ]
    for prompt in open_ended:
        questions.append(_q(q_id, "open_ended", prompt))
        q_id += 1

    # ------------------------------------------------------------------
    # Section 7 — Overall rating (closer)
    # ------------------------------------------------------------------
    questions.append(_q(
        q_id, "multiple_choice",
        "Overall, how would you rate the relevance of this report to "
        "your organization's current priorities?",
        _RELEVANCE_OPTIONS,
    ))

    return questions


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _build_output(questions: list[dict], source_hint: str) -> dict:
    return {
        "metadata": {
            "title": "Survey Questionnaire",
            "source": source_hint,
            "generated_by": "survey_generator.py (rule-based)",
            "total_questions": len(questions),
            "question_types": {
                "likert": sum(1 for q in questions if q["type"] == "likert"),
                "multiple_choice": sum(1 for q in questions if q["type"] == "multiple_choice"),
                "open_ended": sum(1 for q in questions if q["type"] == "open_ended"),
            },
        },
        "questions": questions,
    }


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Survey saved to: {path.resolve()}")
    print(f"  Total questions : {data['metadata']['total_questions']}")
    types = data["metadata"]["question_types"]
    print(f"  Likert scale    : {types['likert']}")
    print(f"  Multiple choice : {types['multiple_choice']}")
    print(f"  Open-ended      : {types['open_ended']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _read_input() -> tuple[str, str]:
    """Return (text, source_hint)."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Error: file not found — {path}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, path.name

    print("Paste your PDF text below.")
    print("When finished, press Enter then Ctrl+Z (Windows) or Ctrl+D (Mac/Linux) and Enter.\n")
    lines = sys.stdin.readlines()
    text = "".join(lines)
    return text, "pasted text"


def main() -> None:
    text, source = _read_input()

    if not text.strip():
        print("Error: no text provided.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalysing text ({len(text):,} characters) …")
    questions = build_survey(text)

    output = _build_output(questions, source)
    out_path = Path("survey_questions.json")
    _save_json(output, out_path)


if __name__ == "__main__":
    main()
