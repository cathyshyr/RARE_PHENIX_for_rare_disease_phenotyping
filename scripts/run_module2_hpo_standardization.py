import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def normalize_hpo_id(x):
    return str(x).replace("HP_", "HP:")


def normalize_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def load_hpo_terms(path, include_obsolete=False):
    df = pd.read_excel(path)

    required = {"id", "lbl", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HPO file missing columns: {sorted(missing)}")

    df = df.copy()

    if not include_obsolete:
        df = df[~df["lbl"].astype(str).str.lower().str.startswith("obsolete")].copy()

    df["hpo_id"] = df["id"].map(normalize_hpo_id)
    df["hpo_label"] = df["lbl"].astype(str)
    df["hpo_definition"] = df["definition"].fillna("").astype(str)
    df["normalized_label"] = df["hpo_label"].map(normalize_text)
    df["retrieval_text"] = (
        "HPO label: " + df["hpo_label"] + ". Definition: " + df["hpo_definition"]
    )

    return df.reset_index(drop=True)


def read_rows(path, id_column, phenotype_column):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if id_column not in reader.fieldnames:
        raise ValueError(f"Missing ID column: {id_column}. Available: {reader.fieldnames}")
    if phenotype_column not in reader.fieldnames:
        raise ValueError(f"Missing phenotype column: {phenotype_column}. Available: {reader.fieldnames}")

    return rows


def lexical_candidates(phenotype, hpo_df, limit):
    query = normalize_text(phenotype)
    choices = hpo_df["normalized_label"].tolist()
    matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)

    out = []
    for _, score, idx in matches:
        row = hpo_df.iloc[idx]
        out.append({
            "idx": int(idx),
            "score": float(score),
            "hpo_id": row["hpo_id"],
            "hpo_label": row["hpo_label"],
            "hpo_definition": row["hpo_definition"],
            "normalized_label": row["normalized_label"],
        })
    return out


def choose_device(user_device=None):
    if user_device:
        return user_device

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def build_semantic_index(hpo_df, model_name, device, batch_size):
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    print(f"Embedding device: {device}")

    model = SentenceTransformer(model_name, device=device)
    corpus = hpo_df["retrieval_text"].tolist()

    print(f"Embedding {len(corpus)} HPO terms...")
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return model, np.asarray(embeddings, dtype=np.float32)


def semantic_candidates(phenotype, hpo_df, model, embeddings, limit):
    query = f"Clinical phenotype: {phenotype}"

    q = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    q = np.asarray(q, dtype=np.float32)[0]

    scores = embeddings @ q
    top_idx = np.argsort(scores)[::-1][:limit]

    out = []
    for idx in top_idx:
        row = hpo_df.iloc[int(idx)]
        out.append({
            "idx": int(idx),
            "score": float(scores[idx]) * 100.0,
            "hpo_id": row["hpo_id"],
            "hpo_label": row["hpo_label"],
            "hpo_definition": row["hpo_definition"],
            "normalized_label": row["normalized_label"],
        })
    return out


def format_ranked(method, ranked):
    out = []
    for rank, item in enumerate(ranked, start=1):
        out.append({
            "method": method,
            "rank": rank,
            "match_score": round(float(item["match_score"]), 3),
            "hpo_id": item["hpo_id"],
            "hpo_label": item["hpo_label"],
            "hpo_definition": item["hpo_definition"],
        })
    return out


def lexical_match(phenotype, hpo_df, top_k):
    candidates = lexical_candidates(phenotype, hpo_df, top_k)
    ranked = []
    for c in candidates:
        c["match_score"] = c["score"]
        ranked.append(c)
    return format_ranked("lexical", ranked)


def semantic_match(phenotype, hpo_df, model, embeddings, top_k):
    candidates = semantic_candidates(phenotype, hpo_df, model, embeddings, top_k)
    ranked = []
    for c in candidates:
        c["match_score"] = c["score"]
        ranked.append(c)
    return format_ranked("semantic", ranked)


def combined_match(phenotype, hpo_df, model, embeddings, top_k):
    query_norm = normalize_text(phenotype)
    candidate_limit = max(top_k * 10, 50)

    lexical = lexical_candidates(phenotype, hpo_df, candidate_limit)
    semantic = semantic_candidates(phenotype, hpo_df, model, embeddings, candidate_limit)

    merged = {}

    for c in lexical:
        key = c["hpo_id"]
        merged.setdefault(key, {
            "hpo_id": c["hpo_id"],
            "hpo_label": c["hpo_label"],
            "hpo_definition": c["hpo_definition"],
            "normalized_label": c["normalized_label"],
            "lexical_score": 0.0,
            "semantic_score": 0.0,
        })
        merged[key]["lexical_score"] = max(merged[key]["lexical_score"], c["score"])

    for c in semantic:
        key = c["hpo_id"]
        merged.setdefault(key, {
            "hpo_id": c["hpo_id"],
            "hpo_label": c["hpo_label"],
            "hpo_definition": c["hpo_definition"],
            "normalized_label": c["normalized_label"],
            "lexical_score": 0.0,
            "semantic_score": 0.0,
        })
        merged[key]["semantic_score"] = max(merged[key]["semantic_score"], c["score"])

    ranked = []
    for item in merged.values():
        exact_match = int(item["normalized_label"] == query_norm)

        # Exact HPO label matches should win. Otherwise blend lexical and semantic scores.
        combined_score = (
            0.65 * item["lexical_score"]
            + 0.35 * item["semantic_score"]
            + 25.0 * exact_match
        )

        item["exact_match"] = exact_match
        item["match_score"] = combined_score
        ranked.append(item)

    ranked = sorted(
        ranked,
        key=lambda x: (x["exact_match"], x["match_score"], x["lexical_score"], x["semantic_score"]),
        reverse=True,
    )[:top_k]

    return format_ranked("combined", ranked)


def main():
    parser = argparse.ArgumentParser(
        description="Run lightweight RARE-PHENIX Module 2 HPO standardization."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hpo-terms", default="data/HPO_ID_TERM_DEFN.xlsx")
    parser.add_argument("--id-column", default="UID")
    parser.add_argument("--phenotype-column", default="Step1_Clean_Split")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", choices=["lexical", "semantic", "combined"], default="combined")
    parser.add_argument("--include-obsolete", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading HPO terms from: {args.hpo_terms}")
    hpo_df = load_hpo_terms(
        path=args.hpo_terms,
        include_obsolete=args.include_obsolete,
    )
    print(f"Loaded {len(hpo_df)} HPO terms")

    rows = read_rows(input_path, args.id_column, args.phenotype_column)
    print(f"Loaded {len(rows)} phenotype rows from: {input_path}")

    semantic_model = None
    semantic_embeddings = None

    if args.method in {"semantic", "combined"}:
        device = choose_device(args.embedding_device)
        semantic_model, semantic_embeddings = build_semantic_index(
            hpo_df=hpo_df,
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embedding_batch_size,
        )

    fields = [
        args.id_column,
        args.phenotype_column,
        "method",
        "rank",
        "match_score",
        "hpo_id",
        "hpo_label",
        "hpo_definition",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            uid = row[args.id_column]
            phenotype = row[args.phenotype_column]
            print(f"Processing {i}/{len(rows)}: {uid} | {phenotype}")

            if args.method == "lexical":
                matches = lexical_match(phenotype, hpo_df, args.top_k)
            elif args.method == "semantic":
                matches = semantic_match(
                    phenotype,
                    hpo_df,
                    semantic_model,
                    semantic_embeddings,
                    args.top_k,
                )
            else:
                matches = combined_match(
                    phenotype,
                    hpo_df,
                    semantic_model,
                    semantic_embeddings,
                    args.top_k,
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
