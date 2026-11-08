"""Tokenize the existing stress generator's actual prompt assignments, CPU only."""
import ast,hashlib,json,random
from pathlib import Path
from transformers import AutoTokenizer
import torch
here=Path(__file__).resolve().parent
source=here.parent/'tests/stress.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='one')
assignments=[n for n in function.body if isinstance(n,ast.Assign)]
assert len(assignments)==3
words_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='words' for t in n.targets))
code=compile(ast.fix_missing_locations(ast.Module(body=[words_node]+assignments,type_ignores=[])),str(source),'exec')
model='/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
records=[]
for i in range(128):
    env=dict(random=random,i=i,MAXP=8000)
    exec(code,env)
    body=env['body'];aborted=env['r'].random()<.2
    if aborted:body.update(stream=True,max_tokens=400)
    ids=tokenizer.encode(body['prompt'],add_special_tokens=True)
    bound=len(ids)+body['max_tokens']+3
    assert bound<=16384
    records.append(dict(index=i,prompt_sha256=hashlib.sha256(body['prompt'].encode()).hexdigest(),token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt_tokens=len(ids),requested_output=body['max_tokens'],abort=aborted,with_draft_bound=bound))
assert sum(x['abort'] for x in records)==37
print(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model=model,max_model_len=16384,max_bound=max(x['with_draft_bound'] for x in records),cuda_initialized=torch.cuda.is_initialized(),records=records),indent=2))
