"""Agentic coding workload: multi-turn tool calling with a growing context.

  record <out.json> [short|long]    run 8 tasks live (chat completions + tools) and store the trajectories, the
                                    exact prompt token ids of every turn, and functional stats. short: coding tasks
                                    of up to 16 turns (10-30k tokens); long: whole-package reviews that grow the
                                    context to the 131072-token limit
  replay <traj.json> <out.json> [conc,...]
                                    replay the recorded turns token-for-token (same prefill and decode lengths on
                                    every config) with 1..N agents in parallel; per-turn TTFT, task time, cache hits

Replay metric version 2: TTFT starts at the first nonempty text/token payload, not a metadata SSE frame.
Output throughput counts only completed, verified turns (server usage, or a length finish plus [DONE]
under the fixed min/max-token request). Failures retain partial results, mark the summary invalid, and
raise in the caller. The report is checkpointed after each concurrency and on interruption. Old summary
keys are retained; their values on successful runs use the stricter token and first-token definitions.

Tools are served by this harness, read-only over the Python stdlib sources. Model-written code is never executed:
run_tests returns a canned failure on the first call and a pass afterwards.
"""
import glob, json, lzma, os, re, statistics, sys, threading, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8100")
ROOT = sorted(glob.glob("/usr/lib/python3.[0-9]*"))[-1]
MAX_TURNS = {"short": 16, "long": 120}
MAX_TOKENS = 4096  # a file plus its tests must fit in one tool call, or the agent loops on truncated writes
MAX_MODEL_LEN = 131072
TMPL = {"enable_thinking": False}


def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=900)
    return r if stream else json.loads(r.read())


def metrics():
    m = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    get = lambda name: sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith(name))
    return get("vllm:prefix_cache_hits_total"), get("vllm:prefix_cache_queries_total")


# ---------------------------------------------------------------- tools (harness side, read-only)
def _safe(path):
    p = os.path.realpath(os.path.join(ROOT, path.lstrip("/")))
    if not p.startswith(ROOT):
        raise ValueError("path outside the repository root")
    return p


def t_list_dir(path="."):
    p = _safe(path)
    return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in os.listdir(p))[:200])


def t_read_file(path, start_line=1, end_line=None):
    lines = open(_safe(path), errors="ignore").read().splitlines()
    s = max(int(start_line), 1); e = min(int(end_line) if end_line else s + 249, s + 249, len(lines))
    return "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(s, e + 1)) + f"\n[{len(lines)} lines total]"


def t_grep(pattern, path="."):
    rx, out = re.compile(pattern), []
    files = [_safe(path)] if os.path.isfile(_safe(path)) else sorted(glob.glob(_safe(path) + "/**/*.py", recursive=True))
    for f in files:
        for i, line in enumerate(open(f, errors="ignore"), 1):
            if rx.search(line):
                out.append(f"{os.path.relpath(f, ROOT)}:{i}: {line.rstrip()[:160]}")
                if len(out) >= 40:
                    return "\n".join(out) + "\n[truncated at 40 matches]"
    return "\n".join(out) or "no matches"


FAIL = """============================= test session starts ==============================
collected 4 items

tests/test_solution.py ..F.                                              [100%]

=================================== FAILURES ===================================
______________________________ test_edge_cases _________________________________

    def test_edge_cases():
>       assert solve(EDGE_INPUT) == EXPECTED_EDGE
E       AssertionError: assert None == ['expected', 'value']
E        +  where None = solve('')

tests/test_solution.py:41: AssertionError
=========================== short test summary info ============================
FAILED tests/test_solution.py::test_edge_cases - AssertionError: empty input returns None
========================= 1 failed, 3 passed in 0.21s ========================="""


class Session:
    def __init__(self):
        self.files, self.test_runs = {}, 0

    def workspace(self, path):
        path = path.strip("/").removeprefix("./")
        if path in self.files:
            return path
        return None

    def call(self, name, args):
        try:
            # files the agent wrote shadow the repository, so it can read back and grep its own work
            if name == "list_dir":
                prefix = args.get("path", ".").strip("/").removeprefix(".").strip("/")
                own = sorted(p for p in self.files if p.startswith(prefix))
                if own:
                    return "\n".join(own)
                return t_list_dir(**args)
            if name == "read_file":
                own = self.workspace(args.get("path", ""))
                if own:
                    lines = self.files[own].splitlines()
                    return "\n".join(f"{i:5d}  {l}" for i, l in enumerate(lines, 1)) + f"\n[{len(lines)} lines total]"
                return t_read_file(**args)
            if name == "grep":
                own = self.workspace(args.get("path", ""))
                if own:
                    rx = re.compile(args["pattern"])
                    hits = [f"{own}:{i}: {l}" for i, l in enumerate(self.files[own].splitlines(), 1) if rx.search(l)]
                    return "\n".join(hits) or "no matches"
                return t_grep(**args)
            if name == "write_file":
                self.files[args["path"]] = args["content"]
                return f"wrote {len(args['content'])} bytes to {args['path']}"
            if name == "run_tests":
                self.test_runs += 1
                return FAIL if self.test_runs == 1 else "4 passed in 0.19s"
            if name == "finish": return "done"
            return f"error: unknown tool {name}"
        except Exception as ex:
            return f"error: {type(ex).__name__}: {ex}"


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


S = {"type": "string"}; I = {"type": "integer"}
TOOLS = [
    fn("list_dir", "List a directory of the repository (paths are relative to the repository root).", {"path": S}, ["path"]),
    fn("read_file", "Read up to 250 lines of a file, with line numbers.", {"path": S, "start_line": I, "end_line": I}, ["path"]),
    fn("grep", "Search Python files under a path with a regular expression.", {"pattern": S, "path": S}, ["pattern", "path"]),
    fn("write_file", "Create or overwrite a file in the workspace.", {"path": S, "content": S}, ["path", "content"]),
    fn("run_tests", "Run the test suite of the workspace and return the output.", {}, []),
    fn("finish", "Finish the task with a short summary of what was done.", {"summary": S}, ["summary"]),
]

SYSTEM = """You are an autonomous software engineering agent working in a sandboxed repository that contains the
CPython standard library sources. You solve the user's task by calling tools; you cannot ask the user questions.

How to work:
1. Explore before you write. Use list_dir, grep and read_file to find the relevant code, and read enough of it to
   understand the behaviour you depend on. Prefer reading the actual implementation over guessing.
2. Plan briefly in plain text (two or three sentences at most), then act. Do not narrate every step.
3. Write complete, working files with write_file. Never leave placeholders, TODOs or elided code. Include type hints
   and docstrings for public functions. Keep modules small and focused.
4. Always add tests (pytest style, in tests/test_solution.py) that cover normal cases and edge cases such as empty
   input, very long input, unicode, and invalid values.
5. Run the tests with run_tests. If something fails, read the failure carefully, fix the code, and run again.
6. When the tests pass, call finish with a short summary. Call exactly one tool per turn.

Coding standards for this repository:
- Python 3.12, standard library only, no third-party packages.
- Follow PEP 8; 4-space indentation; maximum line length 100.
- Public functions validate their arguments and raise ValueError or TypeError with a clear message.
- No global mutable state; no network access; no filesystem access outside the workspace.
- Prefer clarity over cleverness; small pure functions are easier to test.
- Errors must never be silently swallowed. Catch only the exceptions you can handle.
- Document non-obvious decisions in comments that explain why, not what.
- Tests must be deterministic and fast (under one second in total).

Repository layout notes:
- The repository root is the stdlib directory (for example email/, asyncio/, json/, logging/, http/, urllib/).
- Your own files go to the workspace (any relative path such as solution.py or tests/test_solution.py).
- Large files: read them in windows with start_line/end_line instead of all at once.
"""

TASKS = [
    "Study how the email package folds long header lines (look at email/_header_value_parser.py and email/policy.py). "
    "Write fold_utils.py with a function fold_header(name: str, value: str, max_len: int = 78) -> str that folds "
    "an unstructured header value the same way for plain ASCII text, plus tests.",
    "Read how asyncio.Queue and asyncio.PriorityQueue are implemented (asyncio/queues.py). Implement a "
    "BoundedPriorityQueue in solution.py that drops the lowest-priority item when full instead of blocking, plus tests.",
    "Find where the json module encodes floats (json/encoder.py). Write solution.py with a JSONEncoder subclass that "
    "rounds floats to a configurable number of decimals and encodes NaN/Infinity as null, plus tests.",
    "Read logging.handlers.RotatingFileHandler and TimedRotatingFileHandler (logging/handlers.py). Write solution.py "
    "with a pure function should_rollover(size: int, max_bytes: int, last: float, now: float, interval: float) -> "
    "bool that combines both policies, plus tests.",
    "Study http/cookies.py. Write solution.py with parse_cookie_header(header: str) -> dict[str, str] that tolerates "
    "malformed segments (missing '=', stray semicolons, quoted values) instead of raising, plus tests.",
    "Read textwrap.py and find how long words are broken. Write solution.py with wrap_preserving_urls(text: str, "
    "width: int) -> list[str] that never breaks inside a URL but otherwise matches textwrap.wrap, plus tests.",
    "Study csv.py and the csv.Sniffer class. Write solution.py with dedupe_csv(lines: list[str], key_columns: "
    "list[str]) -> list[str] that keeps the first row for each key and preserves the header and dialect, plus tests.",
    "Read urllib/parse.py. Write solution.py with normalize_url(url: str) -> str that lowercases scheme and host, "
    "removes default ports, sorts query parameters, and removes fragments, plus tests.",
]


LONG = ("This is a long, thorough review. Read EVERY module of the package completely, in windows of up to 250 "
        "lines, before writing anything; do not skip files and do not stop early. Then write REVIEW.md with one line "
        "per public function or class and every bug or risky construct you found, write tests for five of the "
        "trickiest functions, run the tests, fix failures, and finish.")
LONG_TASKS = [f"Package to review: {p}. {LONG}" for p in
              ["email/", "asyncio/", "http/ (client.py, server.py, cookiejar.py, cookies.py)", "logging/",
               "multiprocessing/", "xml/ (etree and dom)", "concurrent/ and unittest/", "importlib/ and zipfile/"]]


# ---------------------------------------------------------------- record (live)
def record(out, kind="short"):
    tasks = TASKS if kind == "short" else LONG_TASKS
    trajs, stats = [], {"turns": 0, "tool_calls": 0, "bad_tool_calls": 0, "tool_errors": 0, "truncated_turns": 0,
                        "no_tool_turns": 0, "finished": 0, "test_fix_cycles": 0, "wall_s": 0.0}
    t_all = time.time()
    for ti, task in enumerate(tasks):
        sess, msgs, turns = Session(), [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}], []
        for _ in range(MAX_TURNS[kind]):
            tok = post("/tokenize", {"model": "bench", "messages": msgs, "tools": TOOLS, "add_generation_prompt": True,
                                     "chat_template_kwargs": TMPL})["tokens"]
            if len(tok) + MAX_TOKENS > MAX_MODEL_LEN:
                stats["context_full"] = stats.get("context_full", 0) + 1
                break
            t0 = time.time()
            j = post("/v1/chat/completions", {"model": "bench", "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                                              "temperature": 0, "max_tokens": MAX_TOKENS, "chat_template_kwargs": TMPL})
            dt = time.time() - t0
            m, u = j["choices"][0]["message"], j["usage"]
            stats["truncated_turns"] += j["choices"][0]["finish_reason"] == "length"
            if u["prompt_tokens"] != len(tok):
                print(f"  warning: /tokenize gave {len(tok)} tokens, chat used {u['prompt_tokens']}", flush=True)
            turns.append({"prompt_ids": tok, "completion_tokens": u["completion_tokens"], "live_s": dt})
            stats["turns"] += 1
            calls = m.get("tool_calls") or []
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls") and v is not None})
            if not calls:
                stats["no_tool_turns"] += 1
                break
            done = False
            for c in calls:
                stats["tool_calls"] += 1
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    result = sess.call(c["function"]["name"], args)
                except (json.JSONDecodeError, TypeError) as ex:
                    stats["bad_tool_calls"] += 1
                    result = f"error: invalid arguments: {ex}"
                if result.startswith("error:"):
                    stats["tool_errors"] += 1  # unknown tool, missing/extra arguments, bad path
                    if len(stats.setdefault("error_samples", [])) < 12:
                        stats["error_samples"].append(f"{c['function']['name']}({c['function']['arguments'][:120]}) -> {result[:160]}")
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                done |= c["function"]["name"] == "finish"
            if done:
                stats["finished"] += 1
                break
        stats["test_fix_cycles"] += max(sess.test_runs - 1, 0)
        trajs.append({"task": ti, "turns": turns})
        print(f"task {ti}: {len(turns)} turns, final prompt {len(turns[-1]['prompt_ids'])} tok, "
              f"{sum(t['completion_tokens'] for t in turns)} completion tok, live {sum(t['live_s'] for t in turns):.1f}s",
              flush=True)
    stats["wall_s"] = time.time() - t_all
    json.dump({"stats": stats, "trajectories": trajs}, open(out, "w"))
    print("STATS", json.dumps(stats))


# ---------------------------------------------------------------- replay (token-exact)
def replay_agent(traj, salt, res, stop=None, result_lock=None):
    for k, t in enumerate(traj["turns"]):
        if stop is not None and stop.is_set():
            return
        requested = t["completion_tokens"]
        t0 = time.perf_counter()
        row = {"task": traj["task"], "turn": k, "prompt": len(t["prompt_ids"]),
               "requested_completion": requested, "completion": 0, "completion_source": None,
               "ttft": None, "finish_reason": None, "stream_done": False, "success": False}
        try:
            with post("/v1/completions", {
                "model": "bench", "prompt": t["prompt_ids"], "max_tokens": requested,
                "min_tokens": requested, "ignore_eos": True, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, "cache_salt": salt,
            }, stream=True) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        row["stream_done"] = True
                        break
                    event = json.loads(data)
                    if event.get("error") is not None:
                        raise ValueError(f"server stream error: {event['error']}")
                    usage = event.get("usage")
                    if usage is not None:
                        count = usage.get("completion_tokens")
                        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                            raise ValueError(f"invalid completion usage: {count!r}")
                        row["completion"] = count
                        row["completion_source"] = "usage"
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("unexpected additional completion choice")
                        tokens = choice.get("token_ids") or (choice.get("logprobs") or {}).get("tokens")
                        if row["ttft"] is None and (choice.get("text") or tokens):
                            row["ttft"] = time.perf_counter() - t0
                        if choice.get("finish_reason") is not None:
                            row["finish_reason"] = choice["finish_reason"]
            if not row["stream_done"] or row["finish_reason"] is None:
                raise ValueError("truncated stream: missing finish record or [DONE]")
            if row["finish_reason"] not in ("length", "stop"):
                raise ValueError(f"unsuccessful finish reason: {row['finish_reason']}")
            if row["ttft"] is None:
                raise ValueError("stream completed without a token payload")
            if row["completion_source"] is None and row["finish_reason"] == "length":
                # Equal min/max and ignore_eos make a length finish a count check.
                row["completion"] = requested
                row["completion_source"] = "length_finish"
            if row["completion_source"] is None or row["completion"] != requested:
                raise ValueError(f"completion length mismatch: requested {requested}, "
                                 f"verified {row['completion']} ({row['completion_source']})")
            row["success"] = True
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"
            raise RuntimeError(f"task {traj['task']} turn {k}: {row['error']}") from ex
        finally:
            row["e2e"] = time.perf_counter() - t0
            if result_lock is None:
                res.append(row)
            else:
                with result_lock:
                    res.append(row)


def _save_replay_report(out, report):
    # Atomic replacement preserves the preceding concurrency if writing is interrupted.
    temporary = f"{out}.tmp-{os.getpid()}"
    with open(temporary, "w") as f:
        json.dump(report, f)
    os.replace(temporary, out)


def _replay_summary(c, expected, res, wall, hit_rate, valid):
    successful = [r for r in res if r["success"]]
    ttfts = sorted(r["ttft"] for r in successful if r["ttft"] is not None)
    first = [r["ttft"] for r in successful if r["turn"] == 0 and r["ttft"] is not None]
    later = sorted(r["ttft"] for r in successful if r["turn"] > 0 and r["ttft"] is not None)
    completed_tasks = [task for task, turns in expected.items()
                       if sum(r["task"] == task for r in successful) == turns]
    task_s = [sum(r["e2e"] for r in successful if r["task"] == task) for task in completed_tasks]
    percentile_ms = lambda values, p: 1000 * values[int(len(values) * p)] if values else None
    return {"conc": c, "agents": len(completed_tasks), "wall_s": wall,
            "turns": len(successful), "turns_per_s": len(successful) / max(wall, 1e-9),
            "out_tok_s": sum(r["completion"] for r in successful) / max(wall, 1e-9),
            "prefix_hit_rate": hit_rate,
            "ttft_first_turn_ms": 1000 * statistics.mean(first) if first else None,
            "ttft_later_p50_ms": percentile_ms(later, .5),
            "ttft_later_p90_ms": percentile_ms(later, .9), "ttft_p50_ms": percentile_ms(ttfts, .5),
            "task_mean_s": statistics.mean(task_s) if task_s else None,
            "metric_version": 2, "valid": valid, "expected_agents": len(expected),
            "expected_turns": sum(expected.values()), "attempted_turns": len(res),
            "failed_turns": len(res) - len(successful),
            "not_run_turns": sum(expected.values()) - len(res)}


def replay(traj_file, out, concs):
    with (lzma.open if traj_file.endswith(".xz") else open)(traj_file) as f:
        trajs = json.load(f)["trajectories"]
    if not trajs or any(not tr["turns"] for tr in trajs):
        raise ValueError("replay requires nonempty trajectories and turns")
    report = {}
    for c in concs:
        if c < 1:
            raise ValueError("concurrency must be positive")
        # More agents than recorded trajectories: replay copies, each under its own cache salt so a copy cannot
        # reuse another copy's prefix cache (the shared system prompt is then cached per copy, a slight pessimism).
        copies = -(-c // len(trajs))
        base = f"agentic-c{c}-{os.getpid()}"
        queue = [(dict(tr, task=k * len(trajs) + tr["task"]), f"{base}-copy{k}") for k in range(copies) for tr in trajs]
        expected = {tr["task"]: len(tr["turns"]) for tr, _ in queue}
        if len(expected) != len(queue):
            raise ValueError("trajectory task IDs must be unique across replay copies")
        h0, q0 = metrics(); t0 = time.perf_counter(); res = []; errors = []
        lock = threading.Lock()
        stop = threading.Event()

        def worker():
            while True:
                with lock:
                    if stop.is_set() or not queue:
                        return
                    tr, salt = queue.pop(0)
                try:
                    replay_agent(tr, salt, res, stop=stop, result_lock=lock)
                except Exception as ex:
                    with lock:
                        errors.append({"task": tr["task"], "error": f"{type(ex).__name__}: {ex}"})
                        stop.set()
                    return

        ths = [threading.Thread(target=worker, daemon=True) for _ in range(c)]
        hit_rate, interrupted = None, False
        try:
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            wall = time.perf_counter() - t0
            h1, q1 = metrics()
            hit_rate = (h1 - h0) / max(q1 - q0, 1)
        except BaseException as ex:
            interrupted = True
            stop.set()
            with lock:
                errors.append({"error": f"{type(ex).__name__}: {ex}"})
            raise
        finally:
            with lock:
                wall = time.perf_counter() - t0 if interrupted else wall
                valid = not errors and len(res) == sum(expected.values()) and all(r["success"] for r in res)
                rep = _replay_summary(c, expected, res, wall, hit_rate, valid)
                report[c] = {"summary": rep, "turns": list(res), "errors": list(errors)}
                _save_replay_report(out, report)
        print(json.dumps(rep), flush=True)
        if not rep["valid"]:
            raise RuntimeError(f"replay c={c} incomplete: {rep['turns']}/{rep['expected_turns']} "
                               f"turns verified; see {out}")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "short")
    elif sys.argv[1] == "replay":
        replay(sys.argv[2], sys.argv[3], [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,4,8").split(",")])
