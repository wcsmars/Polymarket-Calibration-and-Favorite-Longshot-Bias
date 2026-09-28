"""Run the early pipeline stages on the synthetic sample in this folder.

Copies code/ and the two sample inputs into a separate project root, then runs
00_setup, 03_build_sample, 04_build_panel, 05_analysis, 07_tables and
08_features there. Every script locates the project from its own path, so all
outputs land under that root's data/processed/ and results/. This folder's
archived results/ are never written. Stages 06 and 09-12 need a larger sample:
06 plots 30- and 90-day calibration bins and 09 fits text features on many
months of markets.

Usage:
  python sample/run_sample.py                 # runs in a new temporary directory
  python sample/run_sample.py --workdir DIR   # DIR must be new or empty
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = ("markets_meta.jsonl", "price_histories.jsonl")
STAGES = ("03_build_sample", "04_build_panel", "05_analysis", "07_tables", "08_features")


def run(workdir, stages=STAGES, echo=False):
    """Copy the pipeline and sample into workdir, run the stages, return their stdout."""
    workdir = Path(workdir).resolve()
    if workdir == ROOT:
        raise ValueError("workdir must be separate from the project root")
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError(f"workdir is not empty: {workdir}")
    (workdir / "code").mkdir(parents=True, exist_ok=True)
    for script in sorted((ROOT / "code").glob("*.py")):
        shutil.copy2(script, workdir / "code" / script.name)

    def call(stage):
        # The stages print Greek letters and symbols; keep piped output UTF-8
        # on platforms whose default encoding is not (for example Windows).
        done = subprocess.run([sys.executable, str(workdir / "code" / f"{stage}.py")],
                              cwd=workdir, capture_output=True, encoding="utf-8",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if done.returncode:
            raise RuntimeError(f"{stage} failed:\n{done.stdout}{done.stderr}")
        if echo:
            print(f"== {stage}\n{done.stdout}", flush=True)
        return done.stdout

    logs = {"00_setup": call("00_setup")}
    for name in INPUTS:
        shutil.copy2(HERE / name, workdir / "data" / "raw" / name)
    for stage in stages:
        logs[stage] = call(stage)
    return logs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path,
                        help="new or empty directory for the run (default: a temporary one)")
    args = parser.parse_args()
    # The echoed stage output includes Greek letters; do not fail on consoles
    # or redirected output that cannot encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="polymarket-sample-"))
    run(workdir, echo=True)
    print(f"Sample run complete. Outputs are in {workdir}")


if __name__ == "__main__":
    main()
