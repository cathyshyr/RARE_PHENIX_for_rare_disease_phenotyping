import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]

    input_csv = repo_root / "examples" / "sample_module1_for_module2.csv"
    hpo_terms = repo_root / "data" / "HPO_ID_TERM_DEFN.xlsx"
    module2_script = repo_root / "scripts" / "run_module2_hpo_standardization.py"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing sample input: {input_csv}")
    if not hpo_terms.exists():
        raise FileNotFoundError(f"Missing HPO terms file: {hpo_terms}")
    if not module2_script.exists():
        raise FileNotFoundError(f"Missing Module 2 script: {module2_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "module2_smoke_output.csv"

        cmd = [
            sys.executable,
            str(module2_script),
            "--input",
            str(input_csv),
            "--id-column",
            "UID",
            "--phenotype-column",
            "Step1_Clean_Split",
            "--hpo-terms",
            str(hpo_terms),
            "--output",
            str(output_csv),
            "--method",
            "lexical",
            "--top-k",
            "1",
        ]

        subprocess.run(cmd, check=True)

        rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))

    expected_top_hits = {
        "global developmental delay": "HP:0001263",
        "hypotonia": "HP:0001252",
        "microcephaly": "HP:0000252",
        "feeding difficulties": "HP:0011968",
        "short stature": "HP:0004322",
        "scoliosis": "HP:0002650",
        "myopia": "HP:0000545",
        "recurrent respiratory infections": "HP:0002205",
    }

    observed = {
        row["Step1_Clean_Split"]: row["hpo_id"]
        for row in rows
        if row["rank"] == "1"
    }

    failures = []
    for phenotype, expected_hpo in expected_top_hits.items():
        observed_hpo = observed.get(phenotype)
        if observed_hpo != expected_hpo:
            failures.append((phenotype, expected_hpo, observed_hpo))

    if failures:
        print("Module 2 smoke test failed:")
        for phenotype, expected_hpo, observed_hpo in failures:
            print(f"  {phenotype}: expected {expected_hpo}, observed {observed_hpo}")
        raise SystemExit(1)

    print("SUCCESS: lightweight Module 2 smoke test passed")
    print(f"Checked {len(expected_top_hits)} exact phenotype-to-HPO mappings")


if __name__ == "__main__":
    main()
