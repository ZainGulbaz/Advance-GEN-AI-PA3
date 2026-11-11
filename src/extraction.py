"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    # isinstance first: bool() on garbage like numpy arrays can itself raise
    if not isinstance(completion_text, str) or not completion_text:
        return None

    try:
        # Normalize unicode minus and LaTeX thousands separator '1{,}234'
        text = completion_text.replace("\u2212", "-").replace("{,}", ",")

        # A label counts if it starts a line (after markdown decoration) or is followed by a
        # colon, so prose like 'I hope this final answer helps' is not a label
        labels = list(re.finditer(
            r"^[ \t#*`_>]*final\s*answer|final\s*answer[ \t*_`]*:",
            text, flags=re.IGNORECASE | re.MULTILINE,
        ))
        if not labels:
            return None

        # Read only the rest of the LAST label's line; never fall through to the next line
        answer_line = text[labels[-1].end():].split("\n", 1)[0]

        # 'Final Answer: 9 * 2 = 18' -> the result is after the last '='
        if "=" in answer_line:
            answer_line = answer_line.rsplit("=", 1)[1]

        # First number: optional sign around '$', proper thousands groups or plain digits,
        # optional decimals. Not preceded by a digit/'.'/',' (rejects '.5'), and the
        # integer part may not continue as ',<digit>' (rejects '18,5')
        match = re.search(
            r"(-?)\$?(-?)(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?!\d|,\d)(?:\.(\d+))?",
            answer_line,
        )
        if not match:
            return None

        sign_before, sign_after, int_part, frac_part = match.groups()
        if frac_part and frac_part.strip("0"):
            return None  # non-integer like 18.5

        value = int(int_part.replace(",", ""))
        return -value if (sign_before or sign_after) else value
    except Exception:
        return None
