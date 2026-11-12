"""
tests/test_spec_gaps.py - Spec requirements (PA3 sections 4.3, 5, 6) that the public smoke
tests do not check.

Tests prefixed `test_advisory_` check prompt quality, not the adapter contract: the
autograder probably can't test them, but the rubric grades them.

Usage:
  .venv/bin/python -m pytest -v tests/test_spec_gaps.py
  .venv/bin/python tests/test_spec_gaps.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import submission_adapter as adapter
import src.prm_judge as prm_judge_module
from src.utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS


class _AmbiguousTruth:
    """Mimics numpy arrays / pandas Series: bool() raises."""
    def __bool__(self):
        raise ValueError("truth value is ambiguous")


def make_gen(responses, token_counts=None, mutate_kwargs=False):
    calls = {"n": 0, "kwargs": [], "num_samples": []}

    def generate_fn(prompt, num_samples, gen_kwargs):
        calls["kwargs"].append(copy.deepcopy(gen_kwargs))
        calls["num_samples"].append(num_samples)
        if mutate_kwargs:
            gen_kwargs["temperature"] = 99.0
        out = []
        for _ in range(num_samples):
            i = calls["n"] % len(responses)
            calls["n"] += 1
            out.append({"text": responses[i], "num_tokens": token_counts[i] if token_counts else 7})
        return out

    generate_fn.calls = calls
    return generate_fn


def make_judge(texts, token_counts=None):
    seen = []

    def judge_fn(prompt):
        i = len(seen) % len(texts)
        seen.append(prompt)
        return {"text": texts[i], "num_tokens": token_counts[i] if token_counts else 5}

    judge_fn.seen = seen
    return judge_fn


# =====================================================================
# 4.3 / 1.1  parse_steps, extract_answer: never raise, 0 is a real answer
# =====================================================================
def test_parse_steps_never_raises():
    for bad in [None, 123, 4.5, b"Step 1: x", ["Step 1: x"], {}, _AmbiguousTruth()]:
        assert adapter.parse_steps(bad) == [], f"parse_steps({bad!r}) should return []"


def test_parse_steps_no_labels_and_final_answer_only():
    assert adapter.parse_steps("Just prose, no steps.") == []
    assert adapter.parse_steps("Final Answer: 5") == []
    assert adapter.parse_steps("") == []


def test_extract_answer_zero_is_valid():
    assert adapter.extract_answer("Final Answer: 0") == 0
    assert adapter.extract_answer("Final Answer: 0.00") == 0
    assert adapter.extract_answer("Final Answer: $0") == 0


def test_advisory_step_prompt_forbids_markdown_latex_boxed():
    # Spec 4.2 Format Compliance Warning: the prompt must explicitly prevent these
    p = adapter.format_step_prompt("Q?").lower()
    assert "markdown" in p or "heading" in p, "prompt should forbid markdown headings"
    assert "latex" in p, "prompt should forbid LaTeX"
    assert "boxed" in p, "prompt should forbid \\boxed{}"


# =====================================================================
# 1.3  compute_majority_vote
# =====================================================================
def test_majority_vote_zero_is_not_dropped():
    assert adapter.compute_majority_vote([0, 0, 5]) == 0
    assert adapter.compute_majority_vote([None, 0]) == 0
    assert adapter.compute_majority_vote([5, 0, 0, 5]) == 5  # tie -> earliest


def test_majority_vote_does_not_mutate_input():
    answers = [3, None, 3, 7]
    adapter.compute_majority_vote(answers)
    assert answers == [3, None, 3, 7]


# =====================================================================
# 1.2  run_greedy
# =====================================================================
def test_greedy_exact_keys_and_full_kwargs():
    gen = make_gen(["Step 1: 0 apples.\nFinal Answer: 0"], [9])
    res = adapter.run_greedy(None, None, "Q?", generate_fn=gen)
    assert set(res) == {"prompt", "completion", "steps", "answer",
                        "generation_calls", "generated_tokens", "time_seconds"}
    assert res["answer"] == 0  # 0 survives
    assert gen.calls["num_samples"] == [1]
    kw = gen.calls["kwargs"][0]
    assert kw["do_sample"] is False
    assert kw["temperature"] is None and kw["top_p"] is None and kw["top_k"] is None
    assert kw["max_new_tokens"] == 512
    assert isinstance(res["time_seconds"], float)


def test_greedy_shared_kwargs_not_mutated_by_generate_fn():
    before = dict(GREEDY_GEN_KWARGS)
    gen = make_gen(["Final Answer: 1"], mutate_kwargs=True)
    adapter.run_greedy(None, None, "Q?", generate_fn=gen)
    assert GREEDY_GEN_KWARGS == before


# =====================================================================
# 1.3  run_self_consistency
# =====================================================================
def test_sc_exact_keys_and_candidate_keys():
    gen = make_gen(["Step 1: a.\nFinal Answer: 4"])
    res = adapter.run_self_consistency(None, None, "Q?", n=2, generate_fn=gen)
    assert set(res) == {"candidates", "answer", "generation_calls",
                        "generated_tokens", "time_seconds"}
    for c in res["candidates"]:
        assert set(c) == {"completion", "steps", "answer", "num_tokens"}


def test_sc_full_sampling_kwargs():
    gen = make_gen(["Final Answer: 4"])
    adapter.run_self_consistency(None, None, "Q?", n=5, generate_fn=gen)
    kw = gen.calls["kwargs"][0]
    assert gen.calls["num_samples"] == [5]
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 1.0
    assert kw["top_k"] == 0
    assert kw["repetition_penalty"] == 1.0
    assert kw["max_new_tokens"] == 512


def test_sc_temperature_argument_is_used():
    gen = make_gen(["Final Answer: 4"])
    adapter.run_self_consistency(None, None, "Q?", n=1, temperature=0.3, generate_fn=gen)
    assert gen.calls["kwargs"][0]["temperature"] == 0.3


def test_sc_shared_kwargs_not_mutated_by_generate_fn():
    before = dict(SAMPLING_GEN_KWARGS)
    gen = make_gen(["Final Answer: 1"], mutate_kwargs=True)
    adapter.run_self_consistency(None, None, "Q?", n=2, generate_fn=gen)
    assert SAMPLING_GEN_KWARGS == before


def test_sc_no_valid_answer_returns_none():
    gen = make_gen(["Step 1: I give up.", "\\boxed{7}"])
    res = adapter.run_self_consistency(None, None, "Q?", n=2, generate_fn=gen)
    assert res["answer"] is None
    assert res["generation_calls"] == 2


def test_sc_zero_majority():
    gen = make_gen(["Final Answer: 0", "Final Answer: 3", "Final Answer: 0"])
    res = adapter.run_self_consistency(None, None, "Q?", n=3, generate_fn=gen)
    assert res["answer"] == 0


# =====================================================================
# 2.1  build_prm_judge_prompt, score_path
# =====================================================================
def test_judge_prompt_order_and_no_input_mutation():
    preceding = ["a = 1", "b = 2"]
    p = adapter.build_prm_judge_prompt("Q?", preceding, "a + b = 3")
    assert preceding == ["a = 1", "b = 2"]
    assert p.index("Q?") < p.index("Step 1: a = 1") < p.index("Step 2: b = 2") < p.index("Current Step:")
    assert "(no preceding steps)" not in p


def test_advisory_judge_prompt_has_no_placeholders():
    p = adapter.build_prm_judge_prompt("Q?", [], "x = 1")
    assert "<role:" not in p and "<rule:" not in p and "..." not in p, \
        "judge prompt still contains outline placeholders"


def test_advisory_judge_prompt_verdicts_on_separate_lines():
    p = adapter.build_prm_judge_prompt("Q?", [], "x = 1")
    lines = [line.strip() for line in p.splitlines()]
    assert "Verdict: VALID" in lines and "Verdict: INVALID" in lines, \
        "a judge that copies 'Verdict: VALID or Verdict: INVALID' is parsed as VALID"


def test_advisory_judge_prompt_states_earlier_errors_dont_propagate():
    p = adapter.build_prm_judge_prompt("Q?", ["x = 2"], "y = x + 1").lower()
    assert "earlier" in p or "previous" in p or "preceding" in p
    assert "does not" in p or "do not" in p or "don't" in p, \
        "prompt should say an earlier error does not by itself make this step INVALID"


def test_score_path_exact_keys_and_prompts():
    steps = ["x = 1", "y = 2", "x + y = 3"]
    cand = {"steps": steps, "answer": 3}
    orig = copy.deepcopy(cand)
    judge = make_judge(["Verdict: VALID\nRationale: Ok."], [4])
    res = adapter.score_path(None, None, "Q?", cand, judge_fn=judge)
    assert set(res) == {"step_scores", "rationales", "judge_calls", "judge_tokens"}
    assert cand == orig
    assert res["judge_calls"] == 3 == len(judge.seen)
    assert res["judge_tokens"] == 12
    for t, step in enumerate(steps):
        assert judge.seen[t] == adapter.build_prm_judge_prompt("Q?", steps[:t], step)


def test_score_path_empty_steps():
    judge = make_judge(["Verdict: VALID"])
    for cand in [{"steps": []}, {"answer": 5}, None]:
        res = adapter.score_path(None, None, "Q?", cand, judge_fn=judge)
        assert res == {"step_scores": [], "rationales": [], "judge_calls": 0, "judge_tokens": 0}
    assert judge.seen == []


def test_score_path_garbage_judge_output_scores_invalid():
    judge = make_judge(["", "I am not sure", "Verdict: NOT VALID"])
    res = adapter.score_path(None, None, "Q?", {"steps": ["a", "b", "c"]}, judge_fn=judge)
    assert res["step_scores"] == [-1.0, -1.0, -1.0]


def test_score_path_calls_required_helpers():
    """Spec: score_path MUST call build_prm_judge_prompt and parse_judge_decision."""
    calls = {"build": 0, "parse": 0}
    orig_build = prm_judge_module.build_prm_judge_prompt
    orig_parse = prm_judge_module.parse_judge_decision

    def spy_build(*a, **k):
        calls["build"] += 1
        return orig_build(*a, **k)

    def spy_parse(*a, **k):
        calls["parse"] += 1
        return orig_parse(*a, **k)

    prm_judge_module.build_prm_judge_prompt = spy_build
    prm_judge_module.parse_judge_decision = spy_parse
    try:
        adapter.score_path(None, None, "Q?", {"steps": ["a", "b"]},
                           judge_fn=make_judge(["Verdict: VALID"]))
    finally:
        prm_judge_module.build_prm_judge_prompt = orig_build
        prm_judge_module.parse_judge_decision = orig_parse
    assert calls == {"build": 2, "parse": 2}


# =====================================================================
# 2.3  parse_judge_decision
# =====================================================================
def test_parse_judge_never_raises_and_types():
    for bad in [None, 123, b"Verdict: VALID", ["Verdict: VALID"], _AmbiguousTruth()]:
        assert adapter.parse_judge_decision(bad) == (-1.0, "")
    score, rat = adapter.parse_judge_decision("Verdict: VALID\nRationale: Fine.")
    assert type(score) is float and type(rat) is str


def test_parse_judge_extra_cases():
    cases = [
        ("Verdict: NOT VALID\nRationale: Wrong.", -1.0, "Wrong."),
        ("VERDICT : valid", 1.0, ""),
        ("Verdict:**INVALID**", -1.0, ""),
        ("Verdict: VALIDATED", -1.0, ""),          # not the word VALID
        ("This step INVALIDATES the claim", -1.0, ""),
        ("Verdict: INVALID\nRationale: It says VALID elsewhere.", -1.0, "It says VALID elsewhere."),
        ("Verdict: VALID\nRationale: 3.14 * 2 = 6.28 holds", 1.0, "3.14 * 2 = 6.28 holds"),
        ("Verdict: VALID\nRationale:\nNext line.", 1.0, ""),
        ("Verdict: VALID\r\nRationale: Ok.\r\n", 1.0, "Ok."),
    ]
    for text, score, rat in cases:
        assert adapter.parse_judge_decision(text) == (score, rat), text


# =====================================================================
# 2.2  aggregate_path_score
# =====================================================================
def test_aggregate_returns_float_and_lambda_zero():
    for strat in ["min", "product", "step_penalty"]:
        assert type(adapter.aggregate_path_score([1.0, 1.0], strategy=strat)) is float
    assert adapter.aggregate_path_score([1.0, -1.0], "step_penalty", penalty_lambda=0.0) == 0.0
    assert abs(adapter.aggregate_path_score([1.0] * 4, "product") - 0.99 ** 4) < 1e-12


def test_aggregate_does_not_mutate_input():
    s = [1.0, -1.0, 1.0]
    for strat in ["min", "product", "step_penalty"]:
        adapter.aggregate_path_score(s, strat)
    assert s == [1.0, -1.0, 1.0]


def test_aggregate_unknown_strategy_on_empty_list_raises():
    try:
        adapter.aggregate_path_score([], strategy="mean")
        assert False, "unknown strategy should raise even for []"
    except ValueError:
        pass


# =====================================================================
# 3.1  select_best_path
# =====================================================================
def test_select_float_tolerance_ties():
    # Same multiset of step scores in a different order gives products that differ by 1 ulp
    a = [1, 1, 1, 1, -1, 1, -1]
    b = [1, 1, 1, -1, -1, 1, 1]
    cands = [
        {"step_scores": a, "answer": 7},
        {"step_scores": b, "answer": 9},
        {"step_scores": a, "answer": 9},
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="product")
    assert win["num_tied"] == 3, "1-ulp differences must count as a tie (tolerance 1e-9)"
    assert win["answer"] == 9


def test_select_beyond_tolerance_is_not_a_tie():
    cands = [
        {"step_scores": [1.0] * 4, "answer": 1},  # 0.99**4
        {"step_scores": [1.0] * 3, "answer": 2},  # 0.99**3, higher
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="product")
    assert win["answer"] == 2 and win["num_tied"] == 1


def test_select_zero_answer_wins_vote():
    cands = [
        {"step_scores": [1.0], "answer": 5},
        {"step_scores": [1.0], "answer": 0},
        {"step_scores": [1.0], "answer": 0},
    ]
    win = adapter.select_best_path(cands)
    assert win["answer"] == 0 and win is not cands[1]


def test_select_none_excluded_from_tied_set():
    cands = [
        {"step_scores": [1.0], "answer": None},
        {"step_scores": [1.0], "answer": 5},
        {"step_scores": [1.0], "answer": 7},
    ]
    win = adapter.select_best_path(cands)
    assert win["num_tied"] == 2
    assert win["answer"] == 5


def test_select_all_none_tie_returns_lowest_index():
    cands = [
        {"step_scores": [1.0], "answer": None, "tag": "a"},
        {"step_scores": [1.0], "answer": None, "tag": "b"},
    ]
    win = adapter.select_best_path(cands)
    assert win["tag"] == "a" and win["num_tied"] == 2


def test_select_empty_steps_score_minus_one():
    cands = [
        {"step_scores": [], "answer": 1},
        {"step_scores": [-1.0, -1.0], "answer": 2},
    ]
    win = adapter.select_best_path(cands, aggregation_strategy="step_penalty")
    assert win["answer"] == 1 and win["path_score"] == -1.0


def test_select_penalty_lambda_is_passed_through():
    a = {"step_scores": [1.0, -1.0], "answer": 1}               # mean 0.0, 1 error
    b = {"step_scores": [1.0] * 18 + [-1.0, -1.0], "answer": 2}  # mean 0.8, 2 errors
    assert adapter.select_best_path([a, b], "step_penalty", penalty_lambda=0.0)["answer"] == 2
    assert adapter.select_best_path([a, b], "step_penalty", penalty_lambda=2.0)["answer"] == 1


def test_select_does_not_mutate_and_types():
    cands = [{"step_scores": [1.0], "answer": 3, "steps": ["s"]}]
    orig = copy.deepcopy(cands)
    win = adapter.select_best_path(cands)
    assert cands == orig and win is not cands[0]
    assert type(win["path_score"]) is float and type(win["num_tied"]) is int


# =====================================================================
# 3.1 / 3.2  run_prm_best_of_n
# =====================================================================
def test_bon_exact_keys_and_shared_pool_untouched():
    shared = [
        {"completion": "c0", "steps": ["a", "b"], "answer": 0, "num_tokens": 10},
        {"completion": "c1", "steps": [], "answer": 4, "num_tokens": 10},
        {"completion": "c2", "steps": ["a"], "answer": None, "num_tokens": 10},
    ]
    orig = copy.deepcopy(shared)
    gen = make_gen(["Final Answer: 99"])
    judge = make_judge(["Verdict: VALID\nRationale: Ok."], [3])
    res = adapter.run_prm_best_of_n(None, None, None, None, "Q?", n=3,
                                    candidates=shared, generate_fn=gen, judge_fn=judge)
    assert set(res) == {"selected", "answer", "candidates", "generation_calls", "judge_calls",
                        "generated_tokens", "judge_tokens", "time_seconds"}
    assert shared == orig, "shared candidate pool was mutated"
    assert gen.calls["n"] == 0, "generate_fn called despite shared candidates"
    assert res["generation_calls"] == 0 and res["generated_tokens"] == 0
    assert res["judge_calls"] == 3 == len(judge.seen)  # K = 2 + 0 + 1
    assert res["judge_tokens"] == 9
    assert res["answer"] == 0  # answer 0 path, all VALID
    assert [c["completion"] for c in res["candidates"]] == ["c0", "c1", "c2"]
    assert res["candidates"][1]["step_scores"] == []


def test_bon_standalone_uses_sampling_kwargs():
    gen = make_gen(["Step 1: a.\nFinal Answer: 1"], [11])
    judge = make_judge(["Verdict: VALID"], [2])
    res = adapter.run_prm_best_of_n(None, None, None, None, "Q?", n=5,
                                    generate_fn=gen, judge_fn=judge)
    kw = gen.calls["kwargs"][0]
    assert kw["do_sample"] is True and kw["temperature"] == 0.7 and kw["top_k"] == 0
    assert res["generation_calls"] == 5 and res["generated_tokens"] == 55
    assert res["judge_calls"] == 5 and res["judge_tokens"] == 10


def test_bon_strategy_and_lambda_reach_selection():
    shared = [
        {"steps": ["a", "b"], "answer": 1},
        {"steps": ["a"] * 20, "answer": 2},
    ]
    # cand 0: [VALID, INVALID]; cand 1: 18 VALID then 2 INVALID
    texts = ["Verdict: VALID", "Verdict: INVALID"] + ["Verdict: VALID"] * 18 + ["Verdict: INVALID"] * 2
    r0 = adapter.run_prm_best_of_n(None, None, None, None, "Q?", aggregation_strategy="step_penalty",
                                   penalty_lambda=0.0, candidates=shared, judge_fn=make_judge(texts))
    r2 = adapter.run_prm_best_of_n(None, None, None, None, "Q?", aggregation_strategy="step_penalty",
                                   penalty_lambda=2.0, candidates=shared, judge_fn=make_judge(texts))
    assert r0["answer"] == 2 and r2["answer"] == 1


# =====================================================================
# 3.3  run_compute_matched_self_consistency
# =====================================================================
def test_cmsc_exact_keys_no_existing():
    gen = make_gen(["Final Answer: 3"], [4])
    res = adapter.run_compute_matched_self_consistency(None, None, "Q?", target_calls=15, generate_fn=gen)
    assert set(res) == {"candidates", "answer", "generation_calls", "total_samples",
                        "generated_tokens", "time_seconds"}
    assert res["generation_calls"] == 15 and res["total_samples"] == 15
    assert res["generated_tokens"] == 60


def test_cmsc_no_deficit_never_calls_generate():
    existing = [{"completion": "", "steps": [], "answer": 0, "num_tokens": 5}] * 15
    gen = make_gen(["Final Answer: 1"])
    res = adapter.run_compute_matched_self_consistency(None, None, "Q?", target_calls=15,
                                                       existing_candidates=existing, generate_fn=gen)
    assert gen.calls["n"] == 0
    assert res["generation_calls"] == 0 and res["generated_tokens"] == 0
    assert res["answer"] == 0


def test_cmsc_order_kwargs_and_no_mutation():
    existing = [{"completion": f"e{i}", "steps": [], "answer": 10, "num_tokens": 5} for i in range(5)]
    orig = copy.deepcopy(existing)
    gen = make_gen(["Final Answer: 20"], [6])
    res = adapter.run_compute_matched_self_consistency(None, None, "Q?", target_calls=15, temperature=0.7,
                                                       existing_candidates=existing, generate_fn=gen)
    assert existing == orig and len(existing) == 5
    assert [c["completion"] for c in res["candidates"][:5]] == [f"e{i}" for i in range(5)]
    assert gen.calls["num_samples"] == [10]
    kw = gen.calls["kwargs"][0]
    assert kw["do_sample"] is True and kw["temperature"] == 0.7 and kw["top_p"] == 1.0
    assert res["answer"] == 20  # 10 x 20 beats 5 x 10
    # scaling-curve consistency: N=15 point equals this row
    assert adapter.compute_majority_vote([c["answer"] for c in res["candidates"]]) == res["answer"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
