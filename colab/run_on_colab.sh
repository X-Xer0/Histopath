#!/usr/bin/env bash
# =============================================================================
#  Run the MoNuSeg training pipeline on a Google Colab GPU from this machine.
#
#  Uses the official Google Colab CLI (google-colab-cli).
#
#  First-time setup (once, done by you in your own terminal):
#      ./colab/run_on_colab.sh auth
#  That prints a Google sign-in URL; sign in, paste the code back.
#
#  Then, in order:
#      ./colab/run_on_colab.sh up          # allocate a T4 runtime
#      ./colab/run_on_colab.sh push-data   # upload the dataset zip and unpack it
#      ./colab/run_on_colab.sh deps        # install python packages on the VM
#      ./colab/run_on_colab.sh smoke       # ~5 min end-to-end pipeline check
#      ./colab/run_on_colab.sh train       # the real 50-epoch run (hours)
#      ./colab/run_on_colab.sh fetch       # download model + results
#      ./colab/run_on_colab.sh down        # stop the VM (always do this)
#
#  `./colab/run_on_colab.sh status` shows the current session.
# =============================================================================
set -euo pipefail

SESSION="${COLAB_SESSION:-histopath}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="${COLAB_BIN:-$ROOT/colab_venv/bin/colab}"

ZIP="$ROOT/colab/_data_zip/monuseg_local.zip"
NOTEBOOK="$ROOT/colab/Histopathology_Tumor_Segmentation_and_Grading.ipynb"
SMOKE="$ROOT/colab/smoke_test.py"
RESULTS="$ROOT/colab/runtime"

REMOTE_DATA="/content/monuseg_local.zip"
TRAIN_TIMEOUT=21600     # 6 h ceiling for the full run
SMOKE_TIMEOUT=3600      # 1 h ceiling for the smoke test

if [[ ! -x "$COLAB" ]]; then
  echo "ERROR: colab CLI not found at $COLAB"
  echo "Install it with:  python3 -m venv colab_venv && ./colab_venv/bin/pip install google-colab-cli"
  exit 1
fi

cmd="${1:-help}"

case "$cmd" in

  auth)
    # One-time Google sign-in. Must be run by you: it prints a URL and waits
    # for you to paste the authorization code back.
    echo "Starting Google sign-in. Open the URL below, sign in, and paste the code back."
    echo
    "$COLAB" sessions
    echo
    echo "If that printed your session list (or 'No active sessions'), auth is done."
    ;;

  status)
    "$COLAB" sessions
    "$COLAB" status -s "$SESSION" 2>/dev/null || true
    ;;

  up)
    echo "Allocating a T4 GPU runtime named '$SESSION' ..."
    "$COLAB" new -s "$SESSION" --gpu T4
    "$COLAB" status -s "$SESSION"
    ;;

  push-data)
    [[ -f "$ZIP" ]] || { echo "ERROR: $ZIP missing. Build it first."; exit 1; }
    echo "Uploading $(du -h "$ZIP" | cut -f1) dataset zip ... (this takes a few minutes)"
    "$COLAB" upload -s "$SESSION" "$ZIP" "$REMOTE_DATA"
    echo "Unpacking on the VM ..."
    "$COLAB" exec -s "$SESSION" --timeout 1800 <<'PY'
import zipfile, pathlib
src = pathlib.Path("/content/monuseg_local.zip")
dst = pathlib.Path("/content/monuseg_upload")
dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(src) as zf:
    zf.extractall(dst)
n = sum(1 for _ in dst.rglob("*") if _.is_file())
print(f"[OK] extracted {n} files into {dst}")
PY
    ;;

  deps)
    echo "Installing python packages on the VM ..."
    "$COLAB" install -s "$SESSION" \
      albumentations onnx reportlab opencv-python-headless scikit-image matplotlib pandas scikit-learn requests
    ;;

  smoke)
    echo "Running the end-to-end smoke test (a few minutes) ..."
    "$COLAB" exec -s "$SESSION" -f "$SMOKE" --timeout "$SMOKE_TIMEOUT"
    ;;

  train)
    echo "Running the full notebook. This is the long step (hours)."
    echo "The keep-alive daemon holds the VM open; you do not need a browser tab."
    "$COLAB" exec -s "$SESSION" -f "$NOTEBOOK" --timeout "$TRAIN_TIMEOUT"
    ;;

  fetch)
    mkdir -p "$RESULTS"
    echo "Downloading results into $RESULTS ..."
    for f in tumor_unet.onnx best_monuseg_unet.pth; do
      if "$COLAB" download -s "$SESSION" "$f" "$RESULTS/$f" 2>/dev/null; then
        echo "  [OK] $f"
      else
        echo "  [--] $f not found on the VM"
      fi
    done
    # The executed notebook (with all plots and printed tables) is kept as history
    "$COLAB" log -s "$SESSION" -o "$RESULTS/execution_log.md" 2>/dev/null \
      && echo "  [OK] execution_log.md" || echo "  [--] no log available"
    ls -la "$RESULTS" || true
    ;;

  down)
    echo "Stopping session '$SESSION' ..."
    "$COLAB" stop -s "$SESSION"
    "$COLAB" sessions
    echo "Done. Confirm no session is left above."
    ;;

  *)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    ;;

esac
