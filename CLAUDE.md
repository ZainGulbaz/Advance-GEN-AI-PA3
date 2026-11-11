# CS 5326 PA3: Search over Reasoning & Step-Level PRMs

Assignment spec: the PA3 PDF in the parent folder. Read it when unsure.

## How to help me
- I'm learning this. Explain and review; write graded logic only when I ask.
- When I share code, point out bugs and edge cases the autograder might test.
- Plotting, JSON writing and boilerplate: fine to write fully.

## Environment
- Phase 1 runs on a MacBook with no GPU. Test everything with mock generate_fn / judge_fn.
- Full run happens later on Kaggle T4x2.
- Run tests: `python tests/test_public_adapter_smoke.py`

## Hard rules (from the spec)
- Function signatures and return keys in submission_adapter.py must match the spec exactly.
- Only torch, transformers, accelerate, bitsandbytes, numpy, tqdm, re, json.
  No vLLM, TGI, SGLang, LangChain, LlamaIndex, DSPy, or PRM libraries.
- Token counts from token IDs: len(output_ids[len(input_ids):]). Never split strings.
- num_return_sequences=5 counts as 5 generation calls.
- Seed: transformers.set_seed(5326 + problem_index) before each problem.
- Never mutate input dicts; return shallow copies.
- Parsers must never raise; return None (answers) or -1.0 (judge) on garbage.