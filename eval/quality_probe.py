import json, os, sys, urllib.request
out = sys.argv[1]
prompts = [
 "Explain in three sentences why the sky is blue.",
 "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
 "What is 17 * 23 + 456 / 12? Show the steps.",
 "Tóm tắt ngắn gọn lịch sử thành phố Hà Nội trong 4 câu.",
 "List five differences between TCP and UDP.",
 "Translate to French: 'The quick brown fox jumps over the lazy dog, and then goes home to sleep.'",
 "Solve for x: 3x^2 - 12x + 9 = 0.",
 "Write a SQL query to find the second highest salary from a table Employees(id, name, salary).",
 "Explain what a mixture-of-experts layer is and why it saves compute.",
 "Giải thích sự khác nhau giữa prefill và decode trong suy luận LLM.",
 "Write a haiku about GPUs.",
 "What are the main causes of the French Revolution? Answer in a bulleted list.",
]
res = []
for p in prompts:
    body = json.dumps({"model": "bench", "prompt": "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                       "max_tokens": 160, "temperature": 0, "logprobs": 1}).encode()
    r = urllib.request.urlopen(urllib.request.Request(os.environ.get("BENCH_URL", "http://127.0.0.1:8100") + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=300)
    j = json.loads(r.read())["choices"][0]
    res.append({"prompt": p, "text": j["text"], "tokens": j["logprobs"]["tokens"], "lp": j["logprobs"]["token_logprobs"]})
json.dump(res, open(out, "w"), ensure_ascii=False)
print("saved", len(res))
