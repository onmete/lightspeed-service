#!/bin/bash
# CI job: run LSEval evaluations against OLS using OpenAI GPT-4o-mini + GPT-4.1-mini judge.
#
# Runs two suites in sequence against the same OLS deployment:
#   1. lseval_periodic          — full 797-question QnA dataset
#   2. lseval_troubleshooting   — scenario-based and MCP troubleshooting evals
#
# After both suites complete, summary JSONs are recorded into
# eval/results/history.csv and weekly trend plots are written to ARTIFACT_DIR.
#
# Input environment variables:
#   OPENAI_PROVIDER_KEY_PATH  - path to file containing the OpenAI API key
#   OLS_IMAGE                 - pullspec for the OLS container image to deploy

set -eou pipefail

make install-deps && make install-deps-test

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# Install operator-sdk
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; *) echo -n $(uname -m) ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.36.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
mkdir -p $HOME/.local/bin
chmod +x operator-sdk_${OS}_${ARCH} && mv operator-sdk_${OS}_${ARCH} $HOME/.local/bin/operator-sdk
export PATH=$HOME/.local/bin:$PATH
operator-sdk version

# Export OpenAI key so the judge LLM can authenticate
export OPENAI_API_KEY=$(cat "$OPENAI_PROVIDER_KEY_PATH")

function run_suites() {
  local rc=0

  set +e
  # Deploy OLS with OpenAI GPT-4o-mini.
  # run_suite arguments: suiteid test_tags provider provider_keypath model ols_image ols_config_suffix
  # OLS_CONFIG_SUFFIX="lseval" → ols_installer builds: olsconfig.crd.openai_lseval.yaml

  # Suite 1: full 797-question QnA dataset
  SUITE_ID="lseval_periodic" run_suite \
    "lseval_periodic" "lseval" "openai" "$OPENAI_PROVIDER_KEY_PATH" "gpt-4o-mini" "$OLS_IMAGE" "lseval"
  (( rc = rc || $? ))

  # Suite 2: scenario-based and MCP troubleshooting evals (same OLS deployment)
  SUITE_ID="lseval_troubleshooting" make test-lseval-troubleshooting
  (( rc = rc || $? ))

  set -e

  cleanup_ols_operator
  return $rc
}

function record_trends() {
  # Collect run summaries and regenerate trend plots.
  # Missing JSONs are silently skipped (suite may have errored before producing output).
  uv run --extra evaluation python eval/scripts/update_eval_trends.py \
    --suite lseval_periodic \
    --summary-json "${ARTIFACT_DIR}/lseval/openai/evaluation_summary.json" \
    --suite lseval_troubleshooting_scenarios \
    --summary-json "${ARTIFACT_DIR}/troubleshooting/scenarios/evaluation_summary.json" \
    --suite lseval_troubleshooting_mcp \
    --summary-json "${ARTIFACT_DIR}/troubleshooting/mcp/evaluation_summary.json" \
    --history-csv eval/score_history.csv \
    --output-dir "${ARTIFACT_DIR}" \
    || echo "WARNING: trend update failed (non-fatal)"
}

function finish() {
  record_trends
  if [ "${LOCAL_MODE:-0}" -eq 1 ]; then
    rm -rf "$ARTIFACT_DIR"
  fi
}
trap finish EXIT

# ARTIFACT_DIR is set automatically in Prow; fall back to a temp dir locally
if [ -z "${ARTIFACT_DIR:-}" ]; then
  export ARTIFACT_DIR=$(mktemp -d)
  readonly LOCAL_MODE=1
fi

run_suites
