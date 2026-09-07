#!/usr/bin/env bash
# Fetch the GPT-2 BPE artifacts into partA/vendor/ and verify their sha256.
#
# Primary source is OpenAI's blob store. The audit sandbox blocks it (HTTP 403),
# so a GitHub mirror is used as fallback. The sha256 check is what makes the
# mirror safe to use: the files are byte-identical to OpenAI's or the script fails.
set -euo pipefail
cd "$(dirname "$0")/../partA/vendor"

PRIMARY="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main"
MIRROR="https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2"

fetch () {  # $1 = filename
  curl -fsSL -o "$1" "$PRIMARY/$1" 2>/dev/null \
    || curl -fsSL -o "$1" "$MIRROR/$1"
}

fetch encoder.json
fetch vocab.bpe

echo "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783  encoder.json" | sha256sum -c -
echo "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5  vocab.bpe"   | sha256sum -c -
echo "OK: vendored GPT-2 vocab verified."
