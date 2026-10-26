#!/bin/bash
# Regenerate patch/04-gdn-state-replay.patch (a/ = vLLM v0.32.0 originals, b/ = modified) and check it applies.
set -e
cd "$(dirname "$0")/patch"
G=b/vllm/model_executor/layers/mamba/gdn
cp ../gdn_replay.cu $G/gdn_replay.cu
rm -f 04-gdn-state-replay.patch
for f in $(cd b && find vllm -type f | sort); do
  if [ -f a/$f ]; then diff -u --label a/$f --label b/$f a/$f b/$f || true
  else diff -u --label /dev/null --label b/$f /dev/null b/$f || true; fi
done > 04-gdn-state-replay.patch
wc -l 04-gdn-state-replay.patch
# applicability check on a scratch copy of the pristine files
T=$(mktemp -d); cp -r a/vllm $T/; mkdir -p $T/vllm/model_executor/layers/mamba/gdn
(cd $T && patch -p1 --dry-run < $OLDPWD/04-gdn-state-replay.patch)
rm -rf $T
