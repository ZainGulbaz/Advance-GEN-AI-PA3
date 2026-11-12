"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

from .utils import make_hf_judge_fn
import re

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    if preceding_steps:
        prev = "\n".join(f"Step {i}: {s}" for i, s in enumerate(preceding_steps, 1))
    else:
        prev = "(no preceding steps)"
    
    return (
        "<role: you verify one step of a math solution>\n"
        "<rule: judge only the Current Step; assume all preceding steps are correct ...>\n"
        "<what makes a step VALID / INVALID; recompute the arithmetic>\n\n"
        f"Problem: {question}\n\n"
        f"Preceding steps:\n{prev}\n\n"
        f"Current Step: {current_step}\n\n"
        "<answer in exactly this format:>\n"
        "Verdict: VALID or Verdict: INVALID\n"
        "Rationale: <one sentence>\n"
        )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not isinstance(judge_output, str) or not judge_output:
        return -1.0, ""

    try:
        # Rule 1: first explicit 'Verdict: VALID|INVALID' line wins
        verdict = re.search(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", judge_output, flags=re.IGNORECASE)
        if verdict:
            score = 1.0 if verdict.group(1).upper() == "VALID" else -1.0
        # Rules 2-3: word boundaries so 'INVALID' never counts as 'VALID'; check negatives first
        elif re.search(r"\bINVALID\b|\bNOT\s+VALID\b", judge_output, flags=re.IGNORECASE):
            score = -1.0
        elif re.search(r"\bVALID\b", judge_output, flags=re.IGNORECASE):
            score = 1.0
        # Rule 4: unparseable defaults to INVALID
        else:
            score = -1.0

        rationale = ""
        label = re.search(r"rationale\s*:", judge_output, flags=re.IGNORECASE)
        if label:
            # Stop at the newline first, then at a sentence end that is followed by
            # whitespace or end of line, so '2.5' is not split
            line = judge_output[label.end():].split("\n", 1)[0]
            end = re.search(r"[.!?](?=\s|$)", line)
            rationale = (line[:end.end()] if end else line).strip()

        return score, rationale
    except Exception:
        return -1.0, ""


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    if judge_fn is None:
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    # read-only access: the candidate dict is never modified
    steps = (candidate or {}).get("steps") or []

    step_scores = []
    rationales = []
    judge_tokens = 0
    for t, step in enumerate(steps):
        # local validity: the judge sees only the steps before this one
        prompt = build_prm_judge_prompt(question, steps[:t], step)
        output = judge_fn(prompt)
        score, rationale = parse_judge_decision(output["text"])
        step_scores.append(score)
        rationales.append(rationale)
        judge_tokens += output["num_tokens"]

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
    }
