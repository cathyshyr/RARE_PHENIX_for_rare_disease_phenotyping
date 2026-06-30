import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_hpo_id(hpo_id: str) -> str:
    return str(hpo_id).replace("HP_", "HP:")


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_hpo_terms(path: Path, include_obsolete: bool = False) -> pd.DataFrame:
    df = pd.read_excel(path)

    required = {"id", "lbl", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HPO file is missing required columns: {sorted(missing)}")

    df = df.copy()

    if not include_obsolete:
        df = df[~df["lbl"].astype(str).str.lower().str.startswith("obsolete")].copy()

    df["hpo_id"] = df["id"].map(normalize_hpo_id)
    df["hpo_label"] = df["lbl"].astype(str)
    df["hpo_definition"] = df["definition"].fillna("").astype(str)
    df["normalized_label"] = df["hpo_label"].map(normalize_text)

    return df[["hpo_id", "hpo_label", "hpo_definition", "normalized_label"]]


def read_input_csv(path: Path, id_column: str, phenotype_column: str):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if id_column not in reader.fieldnames:
        raise ValueError(f"ID column '{id_column}' not found. Available columns: {reader.fieldnames}")
    if phenotype_column not in reader.fieldnames:
        raise ValueError(f"Phenotype column '{phenotype_column}' not found. Available columns: {reader.fieldnames}")

    return rows


def match_phenotype(phenotype: str, hpo_df: pd.DataFrame, top_k: int):
    query_norm = normalize_text(phenotype)
    choices = hpo_df["normalized_label"].tolist()

    matches = process.extract(
        query_norm,
        choices,
        scorer=fuzz.WRatio,
        limit=top_k,
    )

    output = []
    for rank, (_matched_norm_label, score, idx) in enumerate(matches, start=1):
        hpo_row = hpo_df.iloc[idx]
        output.append({
            "rank": rank,
            "match_score": round(float(score), 3),
            "hpo_id": hpo_row["hpo_id"],
            "hpo_label": hpo_row["hpo_label"],
            "hpo_definition": hpo_row["hpo_definition"],
        })

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run lightweight RARE-PHENIX Module 2 HPO standardization using lexical/fuzzy HPO label matching."
    )
    parser.add_argument("--input", required=True, help="Input CSV from Module 1 long-format output.")
    parser.add_argument("--output", required=True, help="Output CSV with HPO candidates.")
    parser.add_argument("--hpo-terms", default="data/HPO_ID_TERM_DEFN.xlsx", help="HPO dictionary file.")
    parser.add_argument("--id-column", default="UID", help="Column containing patient/note IDs.")
    parser.add_argument("--phenotype-column", default="Step1_Clean_Split", help="Column containing extracted phenotype strings.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of HPO candidates to return per phenotype.")
    parser.add_argument("--include-obsolete", action="store_true", help="Include obsolete HPO terms in candidate matching.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    hpo_path = Path(args.hpo_terms)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading HPO terms from: {hpo_path}")
    hpo_df = load_hpo_terms(hpo_path, include_obsolete=args.include_obsolete)
    print(f"Loaded {len(hpo_df)} HPO terms")

    rows = read_input_csv(input_path, args.id_column, args.phenotype_column)
    print(f"Loaded {len(rows)} phenotype rows from: {input_path}")

    output_fields = [
        args.id_column,
        args.phenotype_column,
        "rank",
        "match_score",
        "hpo_id",
        "hpo_label",
        "hpo_definition",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            uid = row[args.id_column]
            phenotype = row[args.phenotype_column]

            print(f"Processing {i}/{len(rows)}: {uid} | {phenotype}")

            matches = match_phenotype(
                phenotype=phenotype,
                hpo_df=hpo_df,
                top_k=args.top_k,
            )

            for match in matches:
                writer.writerow({
                    args.id_column: uid,
                    args.phenotype_column: phenotype,
                    **match,
                })

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
