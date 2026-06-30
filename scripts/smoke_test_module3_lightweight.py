import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]

    input_csv = repo_root / "examples" / "sample_module2_for_module3.csv"
    module3_script = repo_root / "scripts" / "run_module3_hpo_prioritization.py"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing sample input: {input_csv}")
    if not module3_script.exists():
        raise FileNotFoundError(f"Missing Module 3 script: {module3_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "module3_smoke_output.csv"

        cmd = [
            sys.executable,
            str(module3_script),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--max-candidate-rank",
            "3",
            "--top-n",
            "10",
        ]

        subprocess.run(cmd, check=True)

        rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))

    expected_top_hits = {
        "P001": "HP:0001263",
        "P002": "HP:0002205",
    }

    observed = {
        row["UID"]: row["hpo_id"]
        for row in rows
        if row["priority_rank"] == "1"
    }

    failures = []
    for uid, expected_hpo in expected_top_hits.items():
        observed_hpo = observed.get(uid)
        if observed_hpo != expected_hpo:
            failures.append((uid, expected_hpo, observed_hpo))

    if failures:
        print("Module 3 smoke test failed:")
        for uid, expected_hpo, observed_hpo in failures:
            print(f"  {uid}: expected top HPO {expected_hpo}, observed {observed_hpo}")
        raise SystemExit(1)

    print("SUCCESS: lightweight Module 3 smoke test passed")
    print(f"Checked {len(expected_top_hits)} patient-level top-ranked HPO terms")


if __name__ == "__main__":
    main()
