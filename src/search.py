"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from .generation import compute_majority_vote, run_self_consistency
from .prm_judge import score_path
from .utils import make_hf_judge_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    # validate first so a misspelled strategy is caught even on an empty list
    if strategy not in ("min", "product", "step_penalty"):
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    if not step_scores:
        return -1.0

    if strategy == "min":
        return float(min(step_scores))

    if strategy == "product":
        score = 1.0
        for r in step_scores:
            score *= min(max((r + 1) / 2, 0.01), 0.99)
        return score

    # step_penalty
    mean = sum(step_scores) / len(step_scores)
    num_invalid = step_scores.count(-1.0)
    return mean - penalty_lambda * num_invalid


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    if not candidate_paths:
        raise ValueError("select_best_path needs at least one candidate")

    # Rule 1: score every candidate (aggregate_path_score maps [] to -1.0)
    scores = [
        aggregate_path_score(c.get("step_scores") or [], aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]

    # Rule 2: drop answer=None candidates unless that would leave none
    eligible = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if not eligible:
        eligible = list(range(len(candidate_paths)))

    # Rule 3: tied set = eligible candidates within 1e-9 of the max, kept in index order
    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if abs(scores[i] - best) <= 1e-9]

    # Rule 4: break ties by majority vote, then lowest index
    winner = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            winner = next(i for i in tied if candidate_paths[i].get("answer") == vote)

    # Rule 5: shallow copy so the caller's dict is untouched
    chosen = dict(candidate_paths[winner])
    chosen["path_score"] = scores[winner]
    chosen["num_tied"] = len(tied)
    return chosen


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()

    generation_calls = 0
    generated_tokens = 0
    if candidates is None:
        # standalone: draw our own pool; it counts toward this call's generation cost
        sc = run_self_consistency(
            model=model, tokenizer=gen_tokenizer, question=question, n=n, generate_fn=generate_fn
        )
        candidates = sc["candidates"]
        generation_calls = sc["generation_calls"]
        generated_tokens = sc["generated_tokens"]

    if judge_fn is None:
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    # judge every step of every candidate (no early stopping); copies keep a shared pool untouched
    scored = []
    judge_calls = 0
    judge_tokens = 0
    for cand in candidates:
        res = score_path(judge_model, judge_tokenizer, question, cand, judge_fn)
        scored.append({**cand, "step_scores": res["step_scores"], "rationales": res["rationales"]})
        judge_calls += res["judge_calls"]
        judge_tokens += res["judge_tokens"]

    selected = select_best_path(scored, aggregation_strategy, penalty_lambda)

    elapsed = time.perf_counter() - start

    return {
        "selected": selected,
        "answer": selected["answer"],
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": elapsed,
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()

    # reuse the shared pool first; never more than target_calls of it
    reused = list(existing_candidates or [])[:target_calls]
    deficit = target_calls - len(reused)

    new_candidates = []
    generation_calls = 0
    generated_tokens = 0
    if deficit > 0:
        sc = run_self_consistency(
            model=model, tokenizer=tokenizer, question=question,
            n=deficit, temperature=temperature, generate_fn=generate_fn,
        )
        new_candidates = sc["candidates"]
        generation_calls = sc["generation_calls"]
        generated_tokens = sc["generated_tokens"]

    # existing first, then new, so the first N candidates match the SC@N scaling curve
    candidates = reused + new_candidates
    answer = compute_majority_vote([c.get("answer") for c in candidates])

    elapsed = time.perf_counter() - start

    return {
        "candidates": candidates,
        "answer": answer,
        "generation_calls": generation_calls,
        "total_samples": len(candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": elapsed,
    }
