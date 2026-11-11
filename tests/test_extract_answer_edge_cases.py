"""
tests/test_extract_answer_edge_cases.py - Edge cases beyond the PA3 section 4.3 table.

Expected values follow the spec's intent: use the last 'Final Answer' line, never raise,
and no fallback to arbitrary integers (format failures surface as None).

Known limitation (not tested): the first number on the answer line wins, so
'Final Answer: In 2 days she makes 36 dollars.' -> 2. Any simple rule that fixes this
breaks 'Final Answer: 18 dollars, 2 left' instead; analyze such cases in report Section 3.

Usage:
  python tests/test_extract_answer_edge_cases.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import submission_adapter as adapter


class _AmbiguousTruth:
    """Mimics numpy arrays / pandas Series: bool() raises."""
    def __bool__(self):
        raise ValueError("truth value is ambiguous")


# (label, input, expected)
CASES = [
    # --- Raises instead of returning None ---
    ("garbage whose bool() raises", _AmbiguousTruth(), None),

    # --- Prose mention of 'final answer' after the real line ---
    ("trailing prose mention",
     "Step 1: 9 * 2 = 18.\nFinal Answer: 18\nI hope this final answer helps!", 18),
    ("trailing note mentions final answer",
     "Final Answer: 18\n\nNote: the final answer is in dollars.", 18),

    # --- First number on the line is not the answer ---
    ("equation on answer line", "Final Answer: 9 * 2 = 18", 18),
    ("label inline after last step", "Step 4: 9 * 2 = 18. Final Answer: 18", 18),
    ("prose without delimiter is not a label", "So the final answer is 18.", None),

    # --- Label line empty: falls through to the next line's numbers ---
    ("empty label, next line is a step",
     "Step 1: 16 - 7 = 9.\nFinal Answer:\nStep 2: 9 * 2 = 18", None),
    ("heading label, next line is prose",
     "## Final Answer\nShe uses 3 + 4 = 7 eggs and earns 18 dollars.", None),
    ("bold label, answer on next line", "**Final Answer:**\n\n18", None),

    # --- Number formats parsed wrongly ---
    ("bare decimal", "Final Answer: .5", None),
    ("unicode minus sign", "Final Answer: −5", -5),
    ("LaTeX thousands separator", "Final Answer: $1{,}234$", 1234),
    ("comma as decimal point", "Final Answer: 18,5", None),
    ("thousands with decimals", "Final Answer: $1,234,567.00", 1234567),
    ("comma as list separator", "Final Answer: 18, not 20", 18),
    ("bold answer value", "Final Answer: **18**", 18),
    ("bold whole line", "**Final Answer: 18**", 18),
    ("boxed", "Final Answer: $\\boxed{18}$", 18),
    ("negative dollars", "Final Answer: -$5", -5),
    ("plural label", "Final Answers: 18", 18),
]


def test_extract_answer_edge_cases():
    failures = []
    for label, text, expected in CASES:
        try:
            actual = adapter.extract_answer(text)
        except Exception as e:  # spec: must never raise
            actual = f"RAISED {type(e).__name__}: {e}"
        if actual != expected:
            failures.append(f"{label}: expected {expected!r}, got {actual!r}")
    assert not failures, "\n    " + "\n    ".join(failures)


if __name__ == "__main__":
    passed = 0
    for label, text, expected in CASES:
        try:
            actual = adapter.extract_answer(text)
        except Exception as e:
            actual = f"RAISED {type(e).__name__}: {e}"
        ok = actual == expected
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: expected {expected!r}, got {actual!r}")
    print("=" * 45)
    print(f"{passed}/{len(CASES)} edge cases passed.")
    sys.exit(0 if passed == len(CASES) else 1)
