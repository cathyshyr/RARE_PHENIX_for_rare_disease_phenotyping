import argparse
import math
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "UID",
    "Step1_Clean_Split",
    "rank",
    "match_score",
    "hpo_id",
    "hpo_label",
    "hpo_definition",
}


def clean_pipe_join(values):
    vals = []
    seen = set()
    for value in values:
        value = str(value).strip()
        if not value or value.lower() == "nan":
            continue
        key = value.lower()
        if key not in seen:
            vals.append(value)
            seen.add(key)
    return "|".join(vals)


def load_module2_candidates(path, max_candidate_rank=None, min_match_score=None):
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce")

    df = df.dropna(subset=["UID", "Step1_Clean_Split", "rank", "match_score", "hpo_id"])
    df["rank"] = df["rank"].astype(int)

    if max_candidate_rank is not None:
        df = df[df["rank"] <= max_candidate_rank].copy()

    if min_match_score is not None:
        df = df[df["match_score"] >= min_match_score].copy()

    if df.empty:
        raise ValueError("No Module 2 candidates remain after filtering.")

    return df


def prioritize_patient_hpos(df):
    grouped_rows = []

    for (uid, hpo_id), g in df.groupby(["UID", "hpo_id"], sort=False):
        best_rank = int(g["rank"].min())
        best_match_score = float(g["match_score"].max())
        mean_match_score = float(g["match_score"].mean())
        mention_count = int(len(g))
        phenotype_count = int(g["Step1_Clean_Split"].nunique())

        ranking_score = (
            best_match_score
            + 10.0 / max(best_rank, 1)
            + 2.0 * math.log1p(mention_count)
            + 2.0 * math.log1p(phenotype_count)
        )

        grouped_rows.append({
            "UID": uid,
            "hpo_id": hpo_id,
            "hpo_label": g["hpo_label"].iloc[0],
            "ranking_score": round(ranking_score, 3),
            "best_candidate_rank": best_rank,
            "best_match_score": round(best_match_score, 3),
            "mean_match_score": round(mean_match_score, 3),
            "mention_count": mention_count,
            "phenotype_count": phenotype_count,
            "source_phenotypes": clean_pipe_join(g["Step1_Clean_Split"].tolist()),
            "hpo_definition": g["hpo_definition"].iloc[0],
        })

    out = pd.DataFrame(grouped_rows)

    out = out.sort_values(
        by=[
            "UID",
            "ranking_score",
            "best_candidate_rank",
            "best_match_score",
            "phenotype_count",
            "mention_count",
        ],
        ascending=[True, False, True, False, False, False],
    ).reset_index(drop=True)

    out["priority_rank"] = out.groupby("UID").cumcount() + 1

    cols = [
        "UID",
        "priority_rank",
        "ranking_score",
        "hpo_id",
        "hpo_label",
        "best_candidate_rank",
        "best_match_score",
        "mean_match_score",
        "mention_count",
        "phenotype_count",
        "source_phenotypes",
        "hpo_definition",
    ]

    return out[cols]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight RARE-PHENIX Module 3 HPO prioritization. "
            "This aggregates Module 2 HPO candidates into a patient-level ranked HPO list."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV from lightweight Module 2.")
    parser.add_argument("--output", required=True, help="Output CSV with prioritized patient-level HPO terms.")
    parser.add_argument(
        "--max-candidate-rank",
        type=int,
        default=5,
        help="Keep only Module 2 candidates with rank <= this value.",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=None,
        help="Optional minimum Module 2 match_score to keep.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional maximum number of prioritized HPO terms to keep per UID.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Module 2 candidates from: {input_path}")
    df = load_module2_candidates(
        input_path,
        max_candidate_rank=args.max_candidate_rank,
        min_match_score=args.min_match_score,
    )
    print(f"Loaded {len(df)} candidate rows across {df['UID'].nunique()} UID(s)")

    out = prioritize_patient_hpos(df)

    if args.top_n is not None:
        out = out[out["priority_rank"] <= args.top_n].copy()

    out.to_csv(output_path, index=False)

    print(f"Wrote prioritized HPO terms: {output_path}")
    print(f"Output rows: {len(out)}")


if __name__ == "__main__":
    main()
