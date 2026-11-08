#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?unique family required}
TARGET=${2:?bf16 fp8 or both required}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case $TARGET in bf16|fp8|both) ;; *) exit 2;; esac
# Same bytes and same explicit cache root across transitions; never clear caches.
for item in 1:off 2:$TARGET 3:off 4:$TARGET; do
 index=${item%%:*}; profile=${item#*:}
 CACHE_FAMILY=$TAG bash "$HERE/run.sh" "${TAG}_${index}_${profile}" "$profile"
done
