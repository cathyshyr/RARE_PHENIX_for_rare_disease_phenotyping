"""
Train an HPO term ranking model (XGBoost pairwise ranker) using:
- Positives: physician-curated HPO terms per patient (from 'HPO term IDs')
- Negatives: the four negative sets generated per patient from RARE_PHENIX_Module3_Preprocess.py:
    neg_hpo_ids_hard / medium / easy / implausible

Outputs:
- XGBoost model file (.xgb)
- Preprocessing bundle (feature columns, one-hot columns, fill values) as JSON
- (Optional) training dataframe parquet/csv for auditing
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict, deque
from functools import lru_cache
from sklearn.model_selection import GroupShuffleSplit

from pyhpo import Ontology

# ============================================================
# 0) Configuration
# ============================================================
RANDOM_SEED = 33

# Input training table (with negatives already created)
NEG_DATA_PATH = "UDN_patients_with_negative_hpo_sets.csv"

# Demographics
DEMOGRAPHICS_CSV = "./Demographics_Report_2024-07-14T10-30-39.737Z.csv"  # set None to disable
UDN_ID_MAP_TSV   = "./UDN ID map.txt"                                    # set None to disable
DEMOG_UID_COL_IN_NEGDATA = "UDN ID"  # column in NEG_DATA_PATH
DEMOG_UID_COL_IN_DEMOG   = "UID"     # after merging UDN map, we expect a "UID" column

# Optional OMIM/Orphanet usage table
OMIM_ORPHA_USAGE_TSV = "./RankingAlgorithm/HPO terms used in OMIM or Orpha.txt"  # set None to disable

# Where to save model + preprocessing bundle
OUT_DIR = "./RankingAlgorithm"
MODEL_PATH = os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026.xgb")
PREP_PATH  = os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_preprocess.json")

# Audit outputs (optional)
SAVE_TRAIN_TABLE = True
TRAIN_TABLE_PATH = os.path.join(OUT_DIR, "UDN_HPO_Ranker_TrainingTable_2026.parquet")

# Negative sampling ratios
NEG_CAPS = {
    "hard": 10,
    "medium": 15,
    "easy": 15,
    "implausible": 10,
}

# XGBoost tuning grid 
ETA_VALS       = [0.03, 0.07, 0.1]
MAX_DEPTH_VALS = [3, 6]
ALPHA_VALS     = [0.0, 0.5, 1.0]
LAMBDA_VALS    = [1.0, 5.0, 10.0]

EARLY_STOPPING_ROUNDS = 25
MAX_BOOST_ROUNDS      = 2000
EVAL_METRIC           = "map@30"  # good default for ranking

# ============================================================
# 1) Helpers: HPO parsing + Ontology-safe IDs
# ============================================================
Ontology()  # initialize HPO

HPO_RE = re.compile(r"HP:\d{7}")
EPS = 1e-12

def parse_hpo_ids(s):
    if pd.isna(s):
        return []
    return HPO_RE.findall(str(s))

def as_int_hpo_id(x):
    """Normalize to integer id (e.g., 421) from term obj or string 'HP:0000421'."""
    if hasattr(x, "id"):
        x = x.id
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        t = x.strip()
        if t.startswith("HP:"):
            return int(t.split(":")[1])
        if t.isdigit():
            return int(t)
        m = re.search(r"(\d{1,7})", t)
        if m:
            return int(m.group(1))
    raise ValueError(f"Can't parse HPO id from: {x} ({type(x)})")

def int_to_hp(i):
    return f"HP:{as_int_hpo_id(i):07d}"

def get_term_obj(hp_str):
    return Ontology.get_hpo_object(as_int_hpo_id(hp_str))

# ============================================================
# 2) Optional OMIM/Orphanet usage dict
# ============================================================
def load_omim_orpha_usage(path):
    """
    Expected columns:
      term_id
      N_OMIM_diseases_with_term, frac_OMIM_diseases_with_term
      N_Orpha_diseases_with_term, frac_Orpha_diseases_with_term
    """
    if path is None or (not os.path.exists(path)):
        return None

    usage = pd.read_csv(path, delimiter="\t")
    if "term_id" not in usage.columns:
        raise ValueError("OMIM/Orpha usage table must have 'term_id' column")

    usage["term_id"] = pd.to_numeric(usage["term_id"], errors="coerce")
    usage = usage.dropna(subset=["term_id"]).copy()
    usage["term_id"] = usage["term_id"].astype(int)

    needed = [
        "N_OMIM_diseases_with_term", "frac_OMIM_diseases_with_term",
        "N_Orpha_diseases_with_term", "frac_Orpha_diseases_with_term",
    ]
    missing = [c for c in needed if c not in usage.columns]
    if missing:
        raise ValueError(f"Missing columns in OMIM/Orpha usage table: {missing}")

    return usage.set_index("term_id")[needed].to_dict(orient="index")

USAGE_DICT = load_omim_orpha_usage(OMIM_ORPHA_USAGE_TSV)

# ============================================================
# 3) Optional demographics merge
# ============================================================
def load_demographics(demo_csv, udn_map_tsv):
    if demo_csv is None or udn_map_tsv is None:
        return None
    if not os.path.exists(demo_csv) or not os.path.exists(udn_map_tsv):
        print("Demographics files not found; skipping demographics.")
        return None

    dem = pd.read_csv(demo_csv)
    dem = dem.rename(columns={"UDN ID": "UID_long"})  
    m = pd.read_csv(udn_map_tsv, delimiter="\t")

    dem = pd.merge(m, dem, on="UID_long", how="inner")

    # We expect columns like:
    #  "UID" (short)
    #  "Age at Application"
    #  "Sex"
    #  "Primary Symptom Category (App Review)"
    return dem

DEMOG = load_demographics(DEMOGRAPHICS_CSV, UDN_ID_MAP_TSV)

# ============================================================
# 4) Build training table: one row per (patient, candidate HPO term)
# ============================================================
def cap_list(lst, k, rng):
    if k is None:
        return lst
    if len(lst) <= k:
        return lst
    return rng.sample(lst, k)

def build_training_table(df_negs, rng):
    """
    Returns a dataframe with columns:
      UID, term_hp, label,
      neg_type (pos/hard/medium/easy/implausible),
      plus optional demographics columns
    """
    rows = []
    for i, row in df_negs.iterrows():
        uid = row.get("UDN ID", None)
        if pd.isna(uid):
            continue

        pos = parse_hpo_ids(row.get("HPO term IDs", ""))

        hard = parse_hpo_ids(row.get("neg_hpo_ids_hard", ""))
        med  = parse_hpo_ids(row.get("neg_hpo_ids_medium", ""))
        easy = parse_hpo_ids(row.get("neg_hpo_ids_easy", ""))
        impl = parse_hpo_ids(row.get("neg_hpo_ids_implausible", ""))

        # cap negatives per type (keeps training balanced & predictable)
        hard = cap_list(hard, NEG_CAPS["hard"], rng)
        med  = cap_list(med,  NEG_CAPS["medium"], rng)
        easy = cap_list(easy, NEG_CAPS["easy"], rng)
        impl = cap_list(impl, NEG_CAPS["implausible"], rng)

        # Avoid duplicates & any accidental collisions with positives
        pos_set = set(pos)
        hard = [t for t in hard if t not in pos_set]
        med  = [t for t in med  if t not in pos_set]
        easy = [t for t in easy if t not in pos_set]
        impl = [t for t in impl if t not in pos_set]

        # Add positives
        for t in pos:
            rows.append({"UID": uid, "term_hp": t, "label": 1, "neg_type": "pos"})

        # Add negatives
        for t in hard:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "hard"})
        for t in med:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "medium"})
        for t in easy:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "easy"})
        for t in impl:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "implausible"})

    out = pd.DataFrame(rows)

    # Optional: merge demographics
    if DEMOG is not None:
  
        # We'll try two merges:
        #   1) if df_negs UID matches DEMOG["UID_long"], use that.
        #   2) else, if df_negs UID matches DEMOG["UID"], use that.
        if "UID_long" in DEMOG.columns:
            # attempt merge by UID_long first
            dem1 = DEMOG[["UID_long", "UID", "Age at Application", "Sex", "Primary Symptom Category (App Review)"]].drop_duplicates()
            out = out.merge(dem1, left_on="UID", right_on="UID_long", how="left")
            # if that didn't hit, try by short UID
            miss = out["Age at Application"].isna().mean()
            if miss > 0.5 and "UID" in DEMOG.columns:
                dem2 = DEMOG[["UID", "Age at Application", "Sex", "Primary Symptom Category (App Review)"]].drop_duplicates()
                out = out.drop(columns=["UID_long", "UID_y"], errors="ignore")
                out = out.rename(columns={"UID_x": "UID"})
                out = out.merge(dem2, on="UID", how="left")
            else:
                out = out.rename(columns={"UID_x": "UID"}).drop(columns=["UID_long", "UID_y"], errors="ignore")

        # rename to stable feature names
        out = out.rename(columns={
            "Age at Application": "Age",
            "Primary Symptom Category (App Review)": "Primary_Category"
        })
        # normalize Sex
        if "Sex" in out.columns:
            out["Sex"] = out["Sex"].astype(str).str.lower()

    return out

rng = random.Random(RANDOM_SEED)
df_negs = pd.read_csv(NEG_DATA_PATH)
train_df = build_training_table(df_negs, rng)

# Drop patients with no positives (shouldn't happen, but safe)
pos_counts = train_df.groupby("UID")["label"].sum()
keep_uids = pos_counts[pos_counts > 0].index
train_df = train_df[train_df["UID"].isin(keep_uids)].reset_index(drop=True)

print("Training rows:", len(train_df), "Patients:", train_df["UID"].nunique(),
      "Positives:", int(train_df["label"].sum()))

# ============================================================
# 5) Feature engineering per term
# ============================================================
def compute_term_features(term_hp):
    """
    Returns a dict of term-level features (floats/ints).
    Uses pyhpo Ontology (IC, genes, OMIM diseases) and optional OMIM/Orpha usage table.
    """
    try:
        term = get_term_obj(term_hp)
    except Exception:
        term = None

    feats = {}

    # --- IC (OMIM) ---
    ic = 0.0
    n_genes = 0
    n_omim_diseases = 0
    has_known_gene = 0

    if term is not None:
        # IC access differs across pyhpo versions; try a couple patterns
        try:
            # many versions: term.information_content.omim
            ic = float(term.information_content.omim)
        except Exception:
            try:
                # some versions: term.information_content['omim']
                ic = float(term.information_content["omim"])
            except Exception:
                ic = 0.0

        try:
            n_genes = len(list(term.genes))
        except Exception:
            n_genes = 0

        try:
            n_omim_diseases = len(list(term.omim_diseases))
        except Exception:
            n_omim_diseases = 0

        has_known_gene = 1 if n_genes > 0 else 0

    feats["IC"] = ic
    feats["n_genes"] = n_genes
    feats["n_omim_diseases"] = n_omim_diseases
    feats["has_known_gene"] = has_known_gene

    # --- OMIM/Orpha usage table features (optional) ---
    if USAGE_DICT is None:
        feats["N_OMIM_diseases_with_term"] = 0.0
        feats["frac_OMIM_diseases_with_term"] = 0.0
        feats["N_Orpha_diseases_with_term"] = 0.0
        feats["frac_Orpha_diseases_with_term"] = 0.0
        feats["Any_frac_OMIM_Orpha"] = 0.0
        feats["OMIM_idf"] = 0.0
        feats["Orpha_idf"] = 0.0
        feats["Any_idf_OMIM_Orpha"] = 0.0
        return feats

    tid = as_int_hpo_id(term_hp)
    u = USAGE_DICT.get(tid, None)

    omim_n = float(u["N_OMIM_diseases_with_term"]) if u is not None else 0.0
    omim_f = float(u["frac_OMIM_diseases_with_term"]) if u is not None else 0.0
    orph_n = float(u["N_Orpha_diseases_with_term"]) if u is not None else 0.0
    orph_f = float(u["frac_Orpha_diseases_with_term"]) if u is not None else 0.0

    any_f = 1.0 - (1.0 - omim_f) * (1.0 - orph_f)

    feats["N_OMIM_diseases_with_term"] = omim_n
    feats["frac_OMIM_diseases_with_term"] = omim_f
    feats["N_Orpha_diseases_with_term"] = orph_n
    feats["frac_Orpha_diseases_with_term"] = orph_f
    feats["Any_frac_OMIM_Orpha"] = any_f

    feats["OMIM_idf"] = -np.log10(omim_f + EPS) if omim_f > 0 else 0.0
    feats["Orpha_idf"] = -np.log10(orph_f + EPS) if orph_f > 0 else 0.0
    feats["Any_idf_OMIM_Orpha"] = -np.log10(any_f + EPS) if any_f > 0 else 0.0

    return feats

# Compute features with a simple cache for speed
_feature_cache = {}

def get_features_cached(term_hp):
    if term_hp in _feature_cache:
        return _feature_cache[term_hp]
    feats = compute_term_features(term_hp)
    _feature_cache[term_hp] = feats
    return feats

print("Computing term-level features...")
feat_dicts = [get_features_cached(t) for t in train_df["term_hp"].tolist()]
feat_df = pd.DataFrame(feat_dicts)
train_df = pd.concat([train_df, feat_df], axis=1)

# Patient-level covariates (optional)
# If demographics weren't merged, these may not exist; create safe defaults.
if "Age" not in train_df.columns:
    train_df["Age"] = 0.0
if "Sex" not in train_df.columns:
    train_df["Sex"] = "unknown"
if "Primary_Category" not in train_df.columns:
    train_df["Primary_Category"] = "unknown"

# One-hot encode
train_df["Sex"] = train_df["Sex"].fillna("unknown").astype(str).str.lower()
train_df["Primary_Category"] = train_df["Primary_Category"].fillna("unknown").astype(str)

train_df = pd.get_dummies(train_df, columns=["Sex", "Primary_Category"], drop_first=True)

# Fill NaNs in numeric
numeric_cols = [
    "Age", "IC", "n_genes", "n_omim_diseases", "has_known_gene",
    "N_OMIM_diseases_with_term", "frac_OMIM_diseases_with_term",
    "N_Orpha_diseases_with_term", "frac_Orpha_diseases_with_term",
    "Any_frac_OMIM_Orpha", "OMIM_idf", "Orpha_idf", "Any_idf_OMIM_Orpha",
]
for c in numeric_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0.0)

# Feature columns used for training
feature_cols = (
    numeric_cols
    + [c for c in train_df.columns if c.startswith("Sex_")]
    + [c for c in train_df.columns if c.startswith("Primary_Category_")]
)

# ============================================================
# 6) Build group-aware train/val split + train XGBoost ranker
# ============================================================
X = train_df[feature_cols].values
y = train_df["label"].values
groups = train_df["UID"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss.split(X, y, groups))

dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
_, grp_counts_train = np.unique(groups[train_idx], return_counts=True)
dtrain.set_group(grp_counts_train)

dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
_, grp_counts_val = np.unique(groups[val_idx], return_counts=True)
dval.set_group(grp_counts_val)

def tune_and_train():
    best_score = -1
    best_params = None
    best_rounds = None

    for eta in ETA_VALS:
        for md in MAX_DEPTH_VALS:
            for alpha in ALPHA_VALS:
                for lambd in LAMBDA_VALS:
                    params = {
                        "objective": "rank:pairwise",
                        "eval_metric": EVAL_METRIC,
                        "eta": eta,
                        "max_depth": md,
                        "min_child_weight": 1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "tree_method": "hist",
                        "alpha": alpha,
                        "lambda": lambd,
                        "verbosity": 0,
                        "seed": RANDOM_SEED,
                    }

                    bst = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=MAX_BOOST_ROUNDS,
                        evals=[(dval, "val")],
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose_eval=False,
                    )

                    # XGBoost stores best_score for the first eval_metric; for map@30 higher is better
                    score = float(bst.best_score) if bst.best_score is not None else -1

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_rounds = bst.best_iteration

                        print(f"New best {EVAL_METRIC}={best_score:.4f} "
                              f"(eta={eta}, max_depth={md}, alpha={alpha}, lambda={lambd}, rounds={best_rounds})")

    # Train final model on train+val using best rounds
    full_idx = np.concatenate([train_idx, val_idx])
    dfull = xgb.DMatrix(X[full_idx], label=y[full_idx])
    _, grp_counts_full = np.unique(groups[full_idx], return_counts=True)
    dfull.set_group(grp_counts_full)

    final_bst = xgb.train(best_params, dfull, num_boost_round=best_rounds, verbose_eval=False)
    return final_bst, best_params, best_rounds, best_score

os.makedirs(OUT_DIR, exist_ok=True)

print("Tuning + training ranker...")
bst, best_params, best_rounds, best_score = tune_and_train()

bst.save_model(MODEL_PATH)
print("Saved model:", MODEL_PATH)

# Save preprocessing bundle 
prep = {
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "best_params": best_params,
    "best_rounds": int(best_rounds),
    "best_score": float(best_score),
    "seed": RANDOM_SEED,
    "notes": {
        "labels": "1=physician-curated, 0=constructed negatives",
        "objective": "rank:pairwise",
        "eval_metric": EVAL_METRIC,
        "neg_caps": NEG_CAPS,
        "uses_demographics": DEMOG is not None,
        "uses_omim_orpha_usage": USAGE_DICT is not None,
    },
}
with open(PREP_PATH, "w") as f:
    json.dump(prep, f, indent=2)

print("Saved preprocessing bundle:", PREP_PATH)

# Optional audit table (handy for debugging/inspection)
if SAVE_TRAIN_TABLE:
    try:
        train_df.to_parquet(TRAIN_TABLE_PATH, index=False)
        print("Saved training table:", TRAIN_TABLE_PATH)
    except Exception:
        # parquet may fail if pyarrow not installed; fall back to csv
        csv_path = TRAIN_TABLE_PATH.replace(".parquet", ".csv")
        train_df.to_csv(csv_path, index=False)
        print("Saved training table (CSV fallback):", csv_path)

print("Done.")
