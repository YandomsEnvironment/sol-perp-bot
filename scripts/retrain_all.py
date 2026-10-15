"""Retrain all horizons sequentially and append a cycle summary to the
improvement log. Used by the weekly improvement loop (and by hand).

Usage: python scripts/retrain_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "improvement_log.md"


def main() -> None:
    results = {}
    for h in ("fast", "mid", "slow"):
        print(f"=== training {h} ===", flush=True)
        p = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--horizon", h],
            cwd=ROOT, capture_output=True, text=True)
        out = p.stdout
        agg_line = [ln for ln in out.splitlines() if "maker entry" in ln]
        if p.returncode == 0 and agg_line:
            results[h] = agg_line[-1].strip()
        else:
            err_tail = "\n".join(p.stderr.splitlines()[-6:])
            results[h] = f"FAILED rc={p.returncode}: {err_tail}"
        print(f"{h}: {results[h]}", flush=True)
    if not LOG.exists():
        LOG.write_text("# Improvement log — every cycle must beat or explain "
                       "itself vs the previous\n\n", encoding="utf-8")
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## cycle {stamp}\n")
        for h, r in results.items():
            f.write(f"- {h}: {r}\n")
    print(f"appended -> {LOG}", flush=True)

    # Training writes research candidates, never silently promotes to live.
    print("Candidates saved separately; live model and dashboard unchanged.", flush=True)


if __name__ == "__main__":
    main()
