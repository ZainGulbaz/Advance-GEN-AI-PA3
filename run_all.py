"""
run_all.py - End-to-end PA3 benchmark pipeline (all 6 methods, ablation, scaling curve).

Per problem i (0-based line index in the dataset):
  1. set_seed(5326 + i)
  2. Greedy (N=1)
  3. SC@5: 5 sampled candidates = the shared pool
  4. PRM min: judge the pool once via run_prm_best_of_n
  5. PRM product / step_penalty (and the lambda ablation): select_best_path on the
     already-judged pool, no extra judge calls
  6. SC@15: reuse the pool, sample 10 more
  7. Scaling curve: majority vote over the first N of the 15 answers, N = 1..15
  8. Append one line to per_problem.jsonl immediately (resumable after a disconnect)
Summary JSONs are rebuilt from per_problem.jsonl at the end, so a resumed run reports
all 50 problems.

Usage:
  python run_all.py                      # full run (Kaggle T4x2)
  python run_all.py --limit 3            # first 3 problems only
  python run_all.py --fake               # offline fake models, writes to outputs_fake/
  python run_all.py --summarize-only     # rebuild summary JSONs from per_problem.jsonl
"""
import argparse
import json
import os
import random
import time

from src.generation import compute_majority_vote, run_greedy, run_self_consistency
from src.search import run_compute_matched_self_consistency, run_prm_best_of_n, select_best_path
from src.utils import (
    save_aggregation_ablation,
    save_results_summary,
    save_sc_scaling_curve,
    set_seed_for_problem,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

DATA_PATH = "data/gsm8k_test_50.jsonl"
POOL_SIZE = 5
SC_TARGET = 15
ABLATION = [("min", None), ("product", None),
            ("step_penalty", 0.0), ("step_penalty", 0.5), ("step_penalty", 2.0)]
# (summary method name, key in record["methods"])
METHODS = [
    ("Greedy (N=1)", "greedy"),
    ("Self-Consistency (N=5)", "sc5"),
    ("Compute-Matched SC (N=15)", "sc15"),
    ("PRM Best-of-5 (min)", "prm_min"),
    ("PRM Best-of-5 (product)", "prm_product"),
    ("PRM Best-of-5 (step_penalty, lambda=2.0)", "prm_step_penalty_2.0"),
]


def ablation_key(strategy, lam):
    return f"prm_{strategy}" if lam is None else f"prm_{strategy}_{lam}"


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def load_problems(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_records(path):
    """Reads finished records; drops a half-written last line left by a crash."""
    if not os.path.exists(path):
        return []
    records, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        # rewrite without the broken line so the next append starts on a clean line
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Dropped {dropped} malformed line(s) from {path}; those problems will rerun.")
    return records


def append_record(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------
# One problem
# ---------------------------------------------------------------------
def run_problem(index, problem, generate_fn, judge_fn):
    question = problem["question"]
    set_seed_for_problem(index)

    greedy = run_greedy(question=question, generate_fn=generate_fn)

    sc5 = run_self_consistency(question=question, n=POOL_SIZE, temperature=0.7, generate_fn=generate_fn)
    pool = sc5["candidates"]

    # Judge the pool exactly once. 'index' rides along so we can record which candidate won.
    indexed_pool = [{**c, "index": i} for i, c in enumerate(pool)]
    bon = run_prm_best_of_n(question=question, n=POOL_SIZE, aggregation_strategy="min",
                            candidates=indexed_pool, judge_fn=judge_fn)
    judged = bon["candidates"]

    # bon time = judging + min selection; split them so each PRM row adds only its own selection
    selections, select_times = {}, {}
    for strategy, lam in ABLATION:
        t0 = time.perf_counter()
        sel = select_best_path(judged, strategy, 2.0 if lam is None else lam)
        select_times[ablation_key(strategy, lam)] = time.perf_counter() - t0
        selections[ablation_key(strategy, lam)] = sel
    judge_time = max(bon["time_seconds"] - select_times["prm_min"], 0.0)

    sc15 = run_compute_matched_self_consistency(question=question, target_calls=SC_TARGET, temperature=0.7,
                                                existing_candidates=pool, generate_fn=generate_fn)
    all_answers = [c["answer"] for c in sc15["candidates"]]
    all_tokens = [c["num_tokens"] for c in sc15["candidates"]]

    pool_tokens = sc5["generated_tokens"]
    methods = {
        "greedy": {"answer": greedy["answer"], "gen_calls": 1, "judge_calls": 0,
                   "gen_tokens": greedy["generated_tokens"], "judge_tokens": 0,
                   "time": greedy["time_seconds"]},
        "sc5": {"answer": sc5["answer"], "gen_calls": POOL_SIZE, "judge_calls": 0,
                "gen_tokens": pool_tokens, "judge_tokens": 0, "time": sc5["time_seconds"]},
        "sc15": {"answer": sc15["answer"], "gen_calls": POOL_SIZE + sc15["generation_calls"],
                 "judge_calls": 0, "gen_tokens": pool_tokens + sc15["generated_tokens"],
                 "judge_tokens": 0, "time": sc5["time_seconds"] + sc15["time_seconds"]},
    }
    for key, sel in selections.items():
        methods[key] = {
            "answer": sel["answer"], "gen_calls": POOL_SIZE, "judge_calls": bon["judge_calls"],
            "gen_tokens": pool_tokens, "judge_tokens": bon["judge_tokens"],
            "time": sc5["time_seconds"] + judge_time + select_times[key],
            "selected_index": sel["index"], "path_score": sel["path_score"], "num_tied": sel["num_tied"],
        }

    candidates = []
    for i, c in enumerate(sc15["candidates"]):
        entry = {"index": i, "completion": c["completion"], "steps": c["steps"],
                 "answer": c["answer"], "num_tokens": c["num_tokens"]}
        if i < POOL_SIZE:
            entry["step_scores"] = judged[i]["step_scores"]
            entry["rationales"] = judged[i]["rationales"]
        candidates.append(entry)

    return {
        "id": problem["id"],
        "problem_index": index,
        "question": question,
        "ground_truth": problem["ground_truth_answer"],
        "greedy": {"completion": greedy["completion"], "steps": greedy["steps"],
                   "answer": greedy["answer"], "num_tokens": greedy["generated_tokens"]},
        "candidates": candidates,
        "methods": methods,
        "sc_curve_answers": [compute_majority_vote(all_answers[:n]) for n in range(1, SC_TARGET + 1)],
        "sc_curve_tokens": [sum(all_tokens[:n]) for n in range(1, SC_TARGET + 1)],
        "timing": {"greedy": greedy["time_seconds"], "sc5_sampling": sc5["time_seconds"],
                   "judging": judge_time, "sc15_extra_sampling": sc15["time_seconds"]},
    }


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
def pct(numerator, denominator):
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def is_correct(answer, truth):
    return answer is not None and answer == truth


def build_summaries(records, outputs_dir):
    n = len(records)
    if n == 0:
        print("No records to summarize.")
        return

    summary = []
    for name, key in METHODS:
        rows = [r["methods"][key] for r in records]
        gen_calls = sum(m["gen_calls"] for m in rows)
        judge_calls = sum(m["judge_calls"] for m in rows)
        summary.append({
            "Method": name,
            "Accuracy (%)": pct(sum(is_correct(m["answer"], r["ground_truth"]) for m, r in zip(rows, records)), n),
            "Generation Calls": gen_calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": gen_calls + judge_calls,
            "Avg Gen Tokens / Problem": round(sum(m["gen_tokens"] for m in rows) / n, 2),
            "Avg Judge Tokens / Problem": round(sum(m["judge_tokens"] for m in rows) / n, 2),
            "Total Time (s)": round(sum(m["time"] for m in rows), 2),
        })
    save_results_summary(summary, os.path.join(outputs_dir, "results_summary.json"))

    ablation = []
    for strategy, lam in ABLATION:
        rows = [r["methods"][ablation_key(strategy, lam)] for r in records]
        ablation.append({
            "Strategy": strategy,
            "Lambda": lam,
            "Accuracy (%)": pct(sum(is_correct(m["answer"], r["ground_truth"]) for m, r in zip(rows, records)), n),
            "Tie Rate (%)": pct(sum(m["num_tied"] > 1 for m in rows), n),
        })
    save_aggregation_ablation(ablation, os.path.join(outputs_dir, "aggregation_ablation.json"))

    curve = []
    for k in range(SC_TARGET):
        curve.append({
            "N": k + 1,
            "Accuracy (%)": pct(sum(is_correct(r["sc_curve_answers"][k], r["ground_truth"]) for r in records), n),
            "Avg Gen Tokens / Problem": round(sum(r["sc_curve_tokens"][k] for r in records) / n, 2),
        })
    save_sc_scaling_curve(curve, os.path.join(outputs_dir, "sc_scaling_curve.json"))

    # By construction these must match; a mismatch means the pool order was broken
    assert curve[POOL_SIZE - 1]["Accuracy (%)"] == summary[1]["Accuracy (%)"], "N=5 point != SC@5 row"
    assert curve[SC_TARGET - 1]["Accuracy (%)"] == summary[2]["Accuracy (%)"], "N=15 point != SC@15 row"

    greedy_none = sum(r["greedy"]["answer"] is None for r in records)
    print(f"\nProblems: {n}   greedy format failures (answer=None): {greedy_none} ({pct(greedy_none, n)}%)")
    for row in summary:
        print(f"  {row['Method']:<42} acc {row['Accuracy (%)']:>6}%   calls {row['Total LLM Calls']:>6}"
              f"   time {row['Total Time (s)']:>9}s")
    print(f"Wrote summaries to {outputs_dir}/")


# ---------------------------------------------------------------------
# Fake models (offline smoke run on a laptop; numbers are meaningless)
# ---------------------------------------------------------------------
def make_fake_fns(problems):
    truths = {p["question"]: p["ground_truth_answer"] for p in problems}
    # the global RNG, so set_seed_for_problem reseeds the fakes too and resumed runs match
    rng = random

    def generate_fn(prompt, num_samples, gen_kwargs):
        truth = next((t for q, t in truths.items() if q in prompt), 0)
        out = []
        for _ in range(num_samples):
            roll = rng.random()
            answer = truth if roll < 0.6 else truth + rng.choice([-2, -1, 1, 3])
            k = rng.randint(2, 5)
            text = "\n".join(f"Step {j}: compute part {j}." for j in range(1, k + 1))
            if roll > 0.95:
                text += f"\n\\boxed{{{answer}}}"  # format failure -> answer None
            else:
                text += f"\nFinal Answer: {answer}"
            out.append({"text": text, "num_tokens": 20 * k + rng.randint(0, 9)})
        return out

    # varied phrasings so parse_judge_decision's fallback rules get exercised too
    valid_texts = [
        "Verdict: VALID\nRationale: The arithmetic checks out.",
        "Verdict: **VALID**\nRationale: Follows from the previous step.",
        "The step is valid.",
    ]
    invalid_texts = [
        "Verdict: INVALID\nRationale: 2.5 * 4 is 10, not 12.",
        "Verdict: **INVALID**\nRationale: Wrong quantity used!",
        "This step is NOT VALID because it misreads the problem.",
        "I cannot decide.",  # unparseable -> INVALID
    ]

    def judge_fn(judge_prompt):
        texts = valid_texts if rng.random() < 0.8 else invalid_texts
        return {"text": rng.choice(texts), "num_tokens": rng.randint(8, 20)}

    return generate_fn, judge_fn


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PA3 benchmark pipeline")
    parser.add_argument("--fake", action="store_true", help="offline fake generate_fn/judge_fn (writes outputs_fake/)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N problems")
    parser.add_argument("--outputs", default=None, help="output directory (default outputs/)")
    parser.add_argument("--summarize-only", action="store_true", help="skip generation, rebuild JSONs")
    args = parser.parse_args()

    outputs_dir = args.outputs or ("outputs_fake" if args.fake else "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    per_problem_path = os.path.join(outputs_dir, "per_problem.jsonl")

    problems = load_problems(DATA_PATH)
    if args.limit is not None:
        problems = problems[:args.limit]

    records = load_done_records(per_problem_path)
    done_ids = {r["id"] for r in records}

    if not args.summarize_only:
        todo = [(i, p) for i, p in enumerate(problems) if p["id"] not in done_ids]
        print(f"{len(done_ids)} problem(s) already done, {len(todo)} to run.")
        if todo:
            if args.fake:
                generate_fn, judge_fn = make_fake_fns(problems)
            else:
                from src.utils import load_models, make_hf_generate_fn, make_hf_judge_fn
                gen_model, gen_tok, judge_model, judge_tok = load_models("cuda:0", "cuda:1")
                generate_fn = make_hf_generate_fn(gen_model, gen_tok)
                judge_fn = make_hf_judge_fn(judge_model, judge_tok)

            for i, problem in tqdm(todo, desc="problems"):
                record = run_problem(i, problem, generate_fn, judge_fn)
                append_record(per_problem_path, record)
                records.append(record)

    wanted = {p["id"] for p in problems}
    records = sorted((r for r in records if r["id"] in wanted), key=lambda r: r["problem_index"])
    build_summaries(records, outputs_dir)


if __name__ == "__main__":
    main()
