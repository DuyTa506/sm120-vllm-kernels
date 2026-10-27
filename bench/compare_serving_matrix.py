"""Print paired serving tables only for complete, compatible experiment runs.

Usage: python3 bench/compare_serving_matrix.py BASE_RUN CANDIDATE_RUN
Inputs are qwen_experiment.sh result directories; no server/GPU access.
"""
import json
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("serving_pair_greedy",HERE/"compare_greedy_decode.py")
GREEDY=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(GREEDY)
MATRIX,IDENTITY=GREEDY.MATRIX,GREEDY.IDENTITY
require=IDENTITY.require


def validate_matrix(data, manifest, read, text):
    args=dict(manifest["arguments"]);args.setdefault("temperature",None)
    require(args["suite"]=="all", "not a full serving suite")
    cases=MATRIX.scenarios("all",args["seed_offset"])
    require(manifest["scenarios"]==json.loads(json.dumps([asdict(c) for c in cases])),
            "unexpected full-matrix shapes/counts/seeds")
    require(data["status"]=="passed" and not data.get("error") and len(data["results"])==len(cases),
            "incomplete serving matrix")
    summaries={r["scenario"]:r for r in data["results"]}
    require(len(summaries)==len(cases) and set(summaries)=={c.name for c in cases}, "duplicate/missing serving rows")
    commands=manifest["commands"]
    require(len(commands)==2*len(cases), "missing matrix phases")
    require(len({x["cache_salt"] for x in commands})==len(commands), "reused phase cache salt")
    targets=read("expected_output_lengths.json")
    events=[json.loads(line) for line in text("events.jsonl").splitlines() if line]
    normalized=[]
    for i,case in enumerate(cases):
        for warmup in (True,False):
            phase=MATRIX.phase_name(case,warmup)
            saved=commands[2*i+(0 if warmup else 1)]
            argv=MATRIX.bench_command(SimpleNamespace(**args),case,warmup,manifest["container_result_dir"],manifest["run_nonce"])
            require(saved["scenario"]==case.name and saved["warmup"] is warmup and saved["command"]==argv,
                    "benchmark command differs from declared scenario: "+phase)
            require(saved["cache_salt"]==MATRIX.cache_salt(manifest["run_nonce"],case,warmup), "phase salt mismatch")
            starts=[e for e in events if e.get("kind")=="command_start" and e.get("log")==phase+".log"]
            ends=[e for e in events if e.get("kind")=="command_end" and e.get("log")==phase+".log"]
            require(len(starts)==len(ends)==1 and starts[0]["argv"]==argv and ends[0]["returncode"]==0,
                    "actual command/exit evidence mismatch: "+phase)
            result=read(phase+".json")
            row=MATRIX.result_row(result,case,warmup,targets["cases"][phase]["output_lens"])
            require(result["total_input_tokens"]==sum(result["input_lens"])
                    and result["total_output_tokens"]==sum(result["output_lens"]), "inconsistent token totals")
            for field,count in (("req_s",row["completed"]),("out_tok_s",result["total_output_tokens"]),
                                ("total_tok_s",result["total_input_tokens"]+result["total_output_tokens"])):
                require(math.isclose(row[field],count/row["duration_s"],rel_tol=1e-8), "inconsistent timing/rate totals")
            if not warmup:require(row==summaries[case.name],"summary differs from detailed result")
            norm=argv[:];norm[2]="CONTAINER"
            for flag in ("--base-url","--result-dir","--extra-body"):
                norm[norm.index(flag)+1]="RUN_SPECIFIC"
            normalized.append(norm)
    return {"expected_lengths":targets,"normalized_commands":normalized}


def load(path):
    root=Path(path); hashes={}
    def text(relative):
        content=(root/relative).read_bytes();hashes[relative]=hashlib.sha256(content).hexdigest();return content.decode()
    def read(relative):return json.loads(text(relative))
    require(text("exit_code.txt").strip()=="0",f"experiment did not complete successfully: {root}")
    prefix="serving/serving_matrix_all/"
    data,manifest=read(prefix+"summary.json"),read(prefix+"manifest.json")
    runtime=IDENTITY.load(root,text,read,allow_legacy_sources=True)
    require(manifest["arguments"]["container"]=="qwen-exp-"+root.name,"request/container identity mismatch")
    validated=validate_matrix(data,manifest,lambda name:read(prefix+name),lambda name:text(prefix+name))
    data["comparison_validation"]={**validated,"runtime":runtime,"input_sha256":hashes}
    return root,root/prefix,data,manifest,runtime["config"]


def counters(path):
    series=GREEDY.mtp_series(path.read_text())
    return {name:sum(values.values()) for name,values in series.items()}


def acceptance(matrix, scenario, hashes=None):
    def series(phase):
        path=matrix/f"{scenario}.measured.metrics_{phase}.txt"
        content=path.read_bytes()
        if hashes is not None:hashes[str(path.relative_to(matrix.parent.parent))]=hashlib.sha256(content).hexdigest()
        return GREEDY.mtp_series(content.decode())
    pre,post=series("before"),series("after")
    delta={}
    for name in GREEDY.COUNTERS:
        require(pre[name].keys()==post[name].keys(),"MTP series changed during phase")
        changes=[post[name][key]-value for key,value in pre[name].items()]
        require(all(value>=0 for value in changes),"MTP counters reset during measured phase")
        delta[name]=sum(changes)
    require(delta["num_accepted_tokens"]<=delta["num_draft_tokens"],"invalid accepted token count")
    return (delta["num_accepted_tokens"] / delta["num_draft_tokens"]
            if delta["num_draft_tokens"] else None)


def change(before, after):
    return f"{100 * (after / before - 1):+.1f}%" if before else "—"


def preflight():
    import copy
    files={};cases=MATRIX.scenarios("all",0)
    args=MATRIX.parser().parse_args(["--container","qwen-exp-fixture","--output-dir","/fixture","--suite","all"])
    remote="/bench-results/serving_matrix_all";nonce="fixture"
    manifest={"arguments":vars(args),"scenarios":json.loads(json.dumps([asdict(c) for c in cases])),
              "run_nonce":nonce,"container_result_dir":remote,"commands":[]}
    summaries=[];events=[];targets={}
    for case in cases:
        for warmup in (True,False):
            phase=MATRIX.phase_name(case,warmup);n=case.warmups if warmup else case.prompts
            argv=MATRIX.bench_command(args,case,warmup,remote,nonce)
            manifest["commands"].append({"scenario":case.name,"warmup":warmup,"command":argv,
                                         "cache_salt":MATRIX.cache_salt(nonce,case,warmup)})
            events += [{"kind":"command_start","log":phase+".log","argv":argv},
                       {"kind":"command_end","log":phase+".log","returncode":0}]
            result={"num_prompts":n,"max_concurrency":case.concurrency,"completed":n,"failed":0,
                    **{field:1.0 for field in MATRIX.METRICS.values()},
                    **{field:[1.0]*n for field in MATRIX.DETAILS},
                    "input_lens":[case.input_len]*n,"output_lens":[case.output_len]*n,
                    "errors":[""]*n,"generated_texts":["fixture"]*n,"itls":[[] if case.output_len==1 else [.001]]*n,
                    "duration":2.,"total_input_tokens":case.input_len*n,"total_output_tokens":case.output_len*n,
                    "request_throughput":n/2,"output_throughput":case.output_len*n/2,
                    "total_token_throughput":(case.input_len+case.output_len)*n/2}
            files[phase+".json"]=result;targets[phase]={"output_lens":[case.output_len]*n}
            if not warmup:summaries.append(MATRIX.result_row(result,case,warmup,targets[phase]["output_lens"]))
    files["events.jsonl"]="\n".join(json.dumps(x) for x in events)
    files["expected_output_lengths.json"]={"cases":targets}
    data={"status":"passed","results":summaries,"error":None}
    validate_matrix(data,manifest,files.__getitem__,files.__getitem__)
    rejected=0
    fixtures=[("summary",lambda d:d["results"].__setitem__(1,d["results"][0])),
              ("manifest",lambda d:d["scenarios"][0].update(prompts=1)),
              ("manifest",lambda d:d["commands"][1]["command"].append("--temperature")),
              ("files",lambda d:d["long-doc.measured.json"].update(output_lens=[1]*8)),
              ("files",lambda d:d["mixed-len.measured.json"]["output_lens"].__setitem__(0,1)),
              ("files",lambda d:d["prefill-8192.measured.json"].update(completed=7)),
              ("files",lambda d:d["events.jsonl"] and d.update({"events.jsonl":d["events.jsonl"].replace('"returncode": 0','"returncode": 1',1)}))]
    for target,mutate in fixtures:
        values={"summary":copy.deepcopy(data),"manifest":copy.deepcopy(manifest),"files":copy.deepcopy(files)}
        mutate(values[target])
        try:validate_matrix(values["summary"],values["manifest"],values["files"].__getitem__,values["files"].__getitem__)
        except ValueError:rejected+=1
        else:raise AssertionError("malformed full-matrix fixture accepted")
    print(json.dumps({"status":"passed","mode":"cpu_preflight","scenarios":17,
                      "malformed_fixtures_rejected":rejected,"gpu_work":False,"http_requests":0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",nargs="?");parser.add_argument("candidate",nargs="?")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--output-prefix",type=Path)
    args=parser.parse_args()
    if args.preflight:
        preflight();return
    require(args.baseline and args.candidate,"baseline and candidate required")
    a,b=load(args.baseline),load(args.candidate)
    av,bv=a[2]["comparison_validation"],b[2]["comparison_validation"]
    changes=IDENTITY.compare(av["runtime"],bv["runtime"],allow_legacy_sources=True)
    for key in ("expected_lengths","normalized_commands"):
        require(av[key]==bv[key],"incompatible "+key)
    require(a[3]["run_nonce"]!=b[3]["run_nonce"],"runs reused cache namespace")
    require(a[3]["scenarios"]==b[3]["scenarios"],"different workload definitions/seeds")
    lines=[];paired_rows=[]
    def emit(value=""):
        lines.append(value)
        print(value)
    rows_b = {r["scenario"]: r for r in b[2]["results"]}
    emit(f"Baseline `{a[0].name}`; candidate `{b[0].name}`; GPU utilization {a[4]['gpu_util']}.")
    emit("\nPositive rate changes mean faster; negative latency changes mean faster.")
    emit("\nRuntime identity: " + changes["source_validation"] + "; FI control=" + str(changes["fi_control"]) + ".")
    emit("Validated source/config differences: `" + json.dumps(changes,sort_keys=True) + "`")
    for suite in ("throughput", "prefill", "decode"):
        emit(f"\n**{suite}**\n")
        emit("| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |")
        emit("|---|---:|---:|---:|---:|---:|---:|---:|")
        for x in a[2]["results"]:
            if x["suite"] != suite:
                continue
            y = rows_b[x["scenario"]]
            rates = [acceptance(run[1], x["scenario"],run[2]["comparison_validation"]["input_sha256"]) for run in (a, b)]
            paired_rows.append({"scenario":x["scenario"],"baseline":x,"candidate":y,
                                "mtp_acceptance":{"baseline":rates[0],"candidate":rates[1]}})
            mtp = " / ".join(f"{100*r:.1f}%" if r is not None else "—" for r in rates)
            emit(f"| {x['scenario']} | {x['total_tok_s']:.1f} / {y['total_tok_s']:.1f} | "
                  f"{change(x['total_tok_s'], y['total_tok_s'])} | "
                  f"{x['med_ttft_ms']:.1f} / {y['med_ttft_ms']:.1f} | "
                  f"{change(x['med_ttft_ms'], y['med_ttft_ms'])} | "
                  f"{x['med_itl_ms']:.2f} / {y['med_itl_ms']:.2f} | "
                  f"{change(x['out_tok_s'], y['out_tok_s'])} | {mtp} |")
    emit("\nThese are single-run task comparisons. Dataset seeds fix prompts; sampling follows the request/model settings recorded with each run.")

    if args.output_prefix is not None:
        report={"status":"complete_compatible_pair" if changes["source_validation"]=="verified" else "complete_legacy_pair_source_unverified", "runtime_changes":changes, "rows":paired_rows,
                "baseline":{"name":a[0].name,"validation":av,"manifest":a[3]},
                "candidate":{"name":b[0].name,"validation":bv,"manifest":b[3]},
                "analysis_source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__),HERE/"compare_runtime_identity.py",HERE/"compare_greedy_decode.py",HERE/"serving_matrix.py")}}
        args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
        Path(str(args.output_prefix)+".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        Path(str(args.output_prefix)+".md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
