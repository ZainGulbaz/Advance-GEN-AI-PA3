"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps
from .utils import GREEDY_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    return (
        "Solve the following math problem step by step.\n"
        "Write each reasoning step on its own line, numbered as 'Step 1:', 'Step 2:', and so on.\n"
        "Each step should do one calculation or deduction.\n"
        "After the last step, write the final answer on its own line in exactly this form:\n"
        "Final Answer: <integer>\n"
        "Write only the integer after 'Final Answer:', with no units, symbols or extra words.\n"
        "\n"
        f"Problem: {question}\n"
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    # TODO: Implement deterministic majority voting

    if answers is None or len(answers)==0:
        return None;

    ans_count={};
    previous_ans=[];

    for index, a in answers:
        
        if a not in previous_ans:
            ans_count[a]=1;
        else:
           ans_count =  ans_count[a] + 1 ;

    max_ans=0;    
    for ans in ans_count.keys():
        if ans_count[ans] > max_ans:
            max_ans=ans_count[ans];
            ans_majority=ans;
    
    return ans_majority;a
    
    raise NotImplementedError("Implement compute_majority_vote in src/generation.py")



def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()

    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)

    prompt = format_step_prompt(question)
    outputs = generate_fn(prompt, 1, dict(GREEDY_GEN_KWARGS))
    completion = outputs[0]["text"]
    generated_tokens = outputs[0]["num_tokens"]

    steps = parse_steps(completion)
    answer = extract_answer(completion)

    elapsed = time.perf_counter() - start

    return {
        "prompt": prompt,
        "completion": completion,
        "steps": steps,
        "answer": answer,
        "generation_calls": 1,
        "generated_tokens": generated_tokens,
        "time_seconds": elapsed,
    }


def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    # TODO: Implement Self-Consistency sampling and majority voting
    raise NotImplementedError("Implement run_self_consistency in src/generation.py")
