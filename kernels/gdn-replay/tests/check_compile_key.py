"""The flag must change vllm.envs.compile_factors() (and so the torch.compile cache key). CPU only."""
import hashlib, os, subprocess, sys, json
code = ("import json,hashlib;from vllm import envs;f=envs.compile_factors();"
        "print(json.dumps({'has':'VLLM_GDN_STATE_REPLAY' in f,'val':repr(f.get('VLLM_GDN_STATE_REPLAY')),"
        "'hash':hashlib.sha256(repr(sorted(f.items(),key=lambda kv:kv[0])).encode()).hexdigest()[:16]}))")
for v in ("0", "1"):
    env = dict(os.environ, VLLM_GDN_STATE_REPLAY=v)
    print(v, subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1])
