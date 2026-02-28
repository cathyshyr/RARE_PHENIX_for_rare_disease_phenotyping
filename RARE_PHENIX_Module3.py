"""
Train HPO term ranking models using:
- Positives: physician-curated HPO terms per patient (from 'HPO term IDs')
- Negatives: pre-generated negative sets per patient:
    neg_hpo_ids_hard / medium / easy / implausible

This script trains and SAVES multiple models (for later evaluation in a separate script):
- XGBoost rank:pairwise (LTR)
- LightGBM LambdaRank (LTR)  [optional if installed]
- CatBoost YetiRankPairwise (LTR) [optional if installed]
- Logistic Regression (pointwise baseline)

It ALSO:
- Computes validation MAP@30 for each trained model on the held-out validation split
- Prints a results table

Outputs:
- Model files under OUT_DIR
- Preprocessing bundle JSON (feature columns, fill defaults, model paths, params, validation results)
- (Optional) training dataframe parquet/csv for auditing
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from pyhpo import Ontology

# Optional deps (skip gracefully if not installed)
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostRanker, Pool
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# ============================================================
# 0) Configuration
# ============================================================
RANDOM_SEED = 33

NEG_DATA_PATH = "UDN_patients_with_negative_hpo_sets.csv"

DEMOGRAPHICS_CSV = "./Demographics_Report_2024-07-14T10-30-39.737Z.csv"  # set None to disable (UDN patients' demographics file - please see README for access to UDN data)
UDN_ID_MAP_TSV   = "./UDN ID map.txt"                                    # set None to disable (File for mapping VUMC internal patient IDs to UDN long IDs; not used for model training)

OMIM_ORPHA_USAGE_TSV = "./data/HPO terms used in OMIM or Orpha.txt"  # set None to disable (contains OMIM and Orphanet-drived ontology information used for training the learning-to-rank model)

OUT_DIR = "./RankingAlgorithm"
MODEL_PATHS = {
    "xgboost_pairwise": os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_xgb.xgb"),
    "lightgbm_lambdarank": os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_lgb.txt"),
    "catboost_yetirank": os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_cat.cbm"),
    "logreg_pointwise": os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_logreg.joblib"),
}

VAL_RESULTS_CSV = os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_validation_results.csv")

MODELS_TO_TRAIN = [
    "xgboost_pairwise",
    "lightgbm_lambdarank",
    "catboost_yetirank",
    "logreg_pointwise",
]

PREP_PATH = os.path.join(OUT_DIR, "UDN_HPO_Ranker_2026_preprocess.json")

SAVE_TRAIN_TABLE = True
TRAIN_TABLE_PATH = os.path.join(OUT_DIR, "UDN_HPO_Ranker_TrainingTable_2026.parquet")

NEG_CAPS = {"hard": 10, "medium": 15, "easy": 15, "implausible": 10}

ETA_VALS       = [0.03, 0.07, 0.1]
MAX_DEPTH_VALS = [3, 6]
ALPHA_VALS     = [0.0, 0.5, 1.0]
LAMBDA_VALS    = [1.0, 5.0, 10.0]

EARLY_STOPPING_ROUNDS = 25
MAX_BOOST_ROUNDS      = 2000
EVAL_METRIC           = "map@30"

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

def get_term_obj(hp_str):
    return Ontology.get_hpo_object(as_int_hpo_id(hp_str))

# ============================================================
# 2) Optional OMIM/Orphanet usage dict
# ============================================================
def load_omim_orpha_usage(path):
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
    return dem

DEMOG = load_demographics(DEMOGRAPHICS_CSV, UDN_ID_MAP_TSV)

# ============================================================
# 4) Build training table
# ============================================================
def cap_list(lst, k, rng):
    if k is None or len(lst) <= k:
        return lst
    return rng.sample(lst, k)

def build_training_table(df_negs, rng):
    rows = []
    for _, row in df_negs.iterrows():
        uid = row.get("UDN ID", None)
        if pd.isna(uid):
            continue

        pos  = parse_hpo_ids(row.get("HPO term IDs", ""))
        hard = parse_hpo_ids(row.get("neg_hpo_ids_hard", ""))
        med  = parse_hpo_ids(row.get("neg_hpo_ids_medium", ""))
        easy = parse_hpo_ids(row.get("neg_hpo_ids_easy", ""))
        impl = parse_hpo_ids(row.get("neg_hpo_ids_implausible", ""))

        hard = cap_list(hard, NEG_CAPS["hard"], rng)
        med  = cap_list(med,  NEG_CAPS["medium"], rng)
        easy = cap_list(easy, NEG_CAPS["easy"], rng)
        impl = cap_list(impl, NEG_CAPS["implausible"], rng)

        pos_set = set(pos)
        hard = [t for t in hard if t not in pos_set]
        med  = [t for t in med  if t not in pos_set]
        easy = [t for t in easy if t not in pos_set]
        impl = [t for t in impl if t not in pos_set]

        for t in pos:
            rows.append({"UID": uid, "term_hp": t, "label": 1, "neg_type": "pos"})
        for t in hard:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "hard"})
        for t in med:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "medium"})
        for t in easy:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "easy"})
        for t in impl:
            rows.append({"UID": uid, "term_hp": t, "label": 0, "neg_type": "implausible"})

    out = pd.DataFrame(rows)

    if DEMOG is not None and len(out) > 0 and "UID_long" in DEMOG.columns:
        dem1 = DEMOG[["UID_long", "UID", "Age at Application", "Sex",
                      "Primary Symptom Category (App Review)"]].drop_duplicates()
        out = out.merge(dem1, left_on="UID", right_on="UID_long", how="left")

        miss = out["Age at Application"].isna().mean()
        if miss > 0.5 and "UID" in DEMOG.columns:
            dem2 = DEMOG[["UID", "Age at Application", "Sex",
                          "Primary Symptom Category (App Review)"]].drop_duplicates()
            out = out.drop(columns=["UID_long", "UID_y"], errors="ignore")
            out = out.rename(columns={"UID_x": "UID"})
            out = out.merge(dem2, on="UID", how="left")
        else:
            out = out.rename(columns={"UID_x": "UID"}).drop(columns=["UID_long", "UID_y"], errors="ignore")

        out = out.rename(columns={
            "Age at Application": "Age",
            "Primary Symptom Category (App Review)": "Primary_Category"
        })
        if "Sex" in out.columns:
            out["Sex"] = out["Sex"].astype(str).str.lower()

    return out

rng = random.Random(RANDOM_SEED)
df_negs = pd.read_csv(NEG_DATA_PATH)
train_df = build_training_table(df_negs, rng)

pos_counts = train_df.groupby("UID")["label"].sum()
keep_uids = pos_counts[pos_counts > 0].index
train_df = train_df[train_df["UID"].isin(keep_uids)].reset_index(drop=True)

print("Training rows:", len(train_df), "Patients:", train_df["UID"].nunique(),
      "Positives:", int(train_df["label"].sum()))

# ============================================================
# 5) Feature engineering per term
# ============================================================
def compute_term_features(term_hp):
    try:
        term = get_term_obj(term_hp)
    except Exception:
        term = None

    feats = {}
    ic = 0.0
    n_genes = 0
    n_omim_diseases = 0
    has_known_gene = 0

    if term is not None:
        try:
            ic = float(term.information_content.omim)
        except Exception:
            try:
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

    if USAGE_DICT is None:
        feats.update({
            "N_OMIM_diseases_with_term": 0.0,
            "frac_OMIM_diseases_with_term": 0.0,
            "N_Orpha_diseases_with_term": 0.0,
            "frac_Orpha_diseases_with_term": 0.0,
            "Any_frac_OMIM_Orpha": 0.0,
            "OMIM_idf": 0.0,
            "Orpha_idf": 0.0,
            "Any_idf_OMIM_Orpha": 0.0,
        })
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

_feature_cache = {}
def get_features_cached(term_hp):
    if term_hp in _feature_cache:
        return _feature_cache[term_hp]
    feats = compute_term_features(term_hp)
    _feature_cache[term_hp] = feats
    return feats

print("Computing term-level features...")
feat_df = pd.DataFrame([get_features_cached(t) for t in train_df["term_hp"].tolist()])
train_df = pd.concat([train_df, feat_df], axis=1)

# Safe defaults
if "Age" not in train_df.columns:
    train_df["Age"] = 0.0
if "Sex" not in train_df.columns:
    train_df["Sex"] = "unknown"
if "Primary_Category" not in train_df.columns:
    train_df["Primary_Category"] = "unknown"

train_df["Sex"] = train_df["Sex"].fillna("unknown").astype(str).str.lower()
train_df["Primary_Category"] = train_df["Primary_Category"].fillna("unknown").astype(str)

train_df = pd.get_dummies(train_df, columns=["Sex", "Primary_Category"], drop_first=True)

numeric_cols = [
    "Age", "IC", "n_genes", "n_omim_diseases", "has_known_gene",
    "N_OMIM_diseases_with_term", "frac_OMIM_diseases_with_term",
    "N_Orpha_diseases_with_term", "frac_Orpha_diseases_with_term",
    "Any_frac_OMIM_Orpha", "OMIM_idf", "Orpha_idf", "Any_idf_OMIM_Orpha",
]
for c in numeric_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0.0)

feature_cols = (
    numeric_cols
    + [c for c in train_df.columns if c.startswith("Sex_")]
    + [c for c in train_df.columns if c.startswith("Primary_Category_")]
)

# --- CRITICAL: robustly eliminate any remaining NaNs in features (fixes LR crash) ---
train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ============================================================
# 6) Group-aware train/val split
# ============================================================
X = train_df[feature_cols].values
y = train_df["label"].values.astype(int)
groups = train_df["UID"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss.split(X, y, groups))

def group_sizes_from_group_ids(group_ids):
    _, counts = np.unique(group_ids, return_counts=True)
    return counts

grp_counts_train = group_sizes_from_group_ids(groups[train_idx])
grp_counts_val   = group_sizes_from_group_ids(groups[val_idx])

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 7) Train + save models
# ============================================================
trained = {}

# ----------------------------
# (A) XGBoost pairwise ranker
# ----------------------------
def tune_and_train_xgb():
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dtrain.set_group(grp_counts_train)

    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
    dval.set_group(grp_counts_val)

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

                    score = float(bst.best_score) if bst.best_score is not None else -1
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_rounds = bst.best_iteration
                        print(f"[XGB] New best {EVAL_METRIC}={best_score:.4f} "
                              f"(eta={eta}, max_depth={md}, alpha={alpha}, lambda={lambd}, rounds={best_rounds})")

    full_idx = np.concatenate([train_idx, val_idx])
    dfull = xgb.DMatrix(X[full_idx], label=y[full_idx])
    grp_counts_full = group_sizes_from_group_ids(groups[full_idx])
    dfull.set_group(grp_counts_full)

    final_bst = xgb.train(best_params, dfull, num_boost_round=best_rounds, verbose_eval=False)
    return final_bst, best_params, int(best_rounds), float(best_score)

if "xgboost_pairwise" in MODELS_TO_TRAIN:
    print("Training XGBoost ranker...")
    bst, best_params, best_rounds, best_score = tune_and_train_xgb()
    bst.save_model(MODEL_PATHS["xgboost_pairwise"])
    trained["xgboost_pairwise"] = {
        "model_path": MODEL_PATHS["xgboost_pairwise"],
        "best_params": best_params,
        "best_rounds": best_rounds,
        "best_score_internal_val": best_score,  # from xgb's eval
        "notes": {"objective": "rank:pairwise", "eval_metric": EVAL_METRIC},
    }
    print("Saved:", MODEL_PATHS["xgboost_pairwise"])

# ----------------------------
# (B) LightGBM LambdaRank
# ----------------------------
def train_lgb_lambdarank():
    if not HAS_LGB:
        print("LightGBM not installed; skipping LightGBM.")
        return None

    lgb_train = lgb.Dataset(X[train_idx], label=y[train_idx], group=grp_counts_train, free_raw_data=False)
    lgb_val   = lgb.Dataset(X[val_idx], label=y[val_idx], group=grp_counts_val, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "map",
        "map_eval_at": [30],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    best_iter = booster.best_iteration
    best_score = None
    try:
        best_score = booster.best_score["val"]["map@30"]
    except Exception:
        pass

    return booster, params, int(best_iter), (float(best_score) if best_score is not None else None)

if "lightgbm_lambdarank" in MODELS_TO_TRAIN:
    print("Training LightGBM LambdaRank...")
    out = train_lgb_lambdarank()
    if out is not None:
        booster, params, best_iter, best_score = out
        booster.save_model(MODEL_PATHS["lightgbm_lambdarank"])
        trained["lightgbm_lambdarank"] = {
            "model_path": MODEL_PATHS["lightgbm_lambdarank"],
            "best_params": params,
            "best_rounds": best_iter,
            "best_score_internal_val": best_score,
            "notes": {"objective": "lambdarank", "metric": "map@30"},
        }
        print("Saved:", MODEL_PATHS["lightgbm_lambdarank"])

# ----------------------------
# (C) CatBoost YetiRank (pairwise)
# ----------------------------
def train_cat_yetirank():
    if not HAS_CAT:
        print("CatBoost not installed; skipping CatBoost.")
        return None

    train_pool = Pool(X[train_idx], label=y[train_idx], group_id=groups[train_idx])
    val_pool   = Pool(X[val_idx], label=y[val_idx], group_id=groups[val_idx])

    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="MAP:top=30",
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_SEED,
        verbose=False,
        od_type="Iter",
        od_wait=50,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    best_iter = int(model.get_best_iteration()) if model.get_best_iteration() is not None else None
    best_score = None
    try:
        best_score = float(model.get_best_score().get("validation", {}).get("MAP:top=30", None))
    except Exception:
        pass

    params = model.get_params()
    return model, params, best_iter, best_score

if "catboost_yetirank" in MODELS_TO_TRAIN:
    print("Training CatBoost YetiRankPairwise...")
    out = train_cat_yetirank()
    if out is not None:
        model, params, best_iter, best_score = out
        model.save_model(MODEL_PATHS["catboost_yetirank"])
        trained["catboost_yetirank"] = {
            "model_path": MODEL_PATHS["catboost_yetirank"],
            "best_params": params,
            "best_rounds": best_iter,
            "best_score_internal_val": best_score,
            "notes": {"loss_function": "YetiRankPairwise", "eval_metric": "MAP:top=30"},
        }
        print("Saved:", MODEL_PATHS["catboost_yetirank"])

# ----------------------------
# (D) Logistic regression baseline (pointwise)
# ----------------------------
def train_logreg_pointwise():
    # Imputer prevents any NaN crash; scaling helps LR stability.
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])
    clf.fit(X[train_idx], y[train_idx])
    return clf, {"type": "Pipeline(Imputer+StandardScaler+LogReg)", "class_weight": "balanced"}

if "logreg_pointwise" in MODELS_TO_TRAIN:
    print("Training LogisticRegression baseline (pointwise)...")
    clf, params = train_logreg_pointwise()
    joblib.dump(clf, MODEL_PATHS["logreg_pointwise"])
    trained["logreg_pointwise"] = {
        "model_path": MODEL_PATHS["logreg_pointwise"],
        "best_params": params,
        "best_rounds": None,
        "best_score_internal_val": None,
        "notes": {"objective": "pointwise binary classification"},
    }
    print("Saved:", MODEL_PATHS["logreg_pointwise"])

# ============================================================
# 8) Validation scoring: MAP@30 computed uniformly for all models
# ============================================================
def average_precision_at_k(y_true_sorted, k=30):
    """y_true_sorted is a list/array of 0/1 in predicted rank order (best first)."""
    y_true_sorted = np.asarray(y_true_sorted)[:k]
    if y_true_sorted.size == 0:
        return 0.0
    n_pos = int(np.sum(y_true_sorted))
    if n_pos == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(y_true_sorted, start=1):
        if rel == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.sum(precisions) / n_pos)

def map_at_k(df_val, score_col, k=30):
    """Compute MAP@k grouped by UID."""
    aps = []
    for _, g in df_val.groupby("UID", sort=False):
        g2 = g.sort_values(score_col, ascending=False)
        aps.append(average_precision_at_k(g2["label"].values, k=k))
    return float(np.mean(aps)) if len(aps) else 0.0

# Build a validation dataframe to score
val_df = train_df.iloc[val_idx].copy()
X_val = X[val_idx]
y_val = y[val_idx]
groups_val = groups[val_idx]

# Score each trained model
val_results = []

# XGBoost score
if "xgboost_pairwise" in trained:
    booster = xgb.Booster()
    booster.load_model(trained["xgboost_pairwise"]["model_path"])
    dval = xgb.DMatrix(X_val)
    val_df["_score_xgb"] = booster.predict(dval)
    m = map_at_k(val_df, "_score_xgb", k=30)
    trained["xgboost_pairwise"]["val_map@30"] = m
    val_results.append({"model": "xgboost_pairwise", "val_map@30": m})

# LightGBM score
if "lightgbm_lambdarank" in trained and HAS_LGB:
    booster = lgb.Booster(model_file=trained["lightgbm_lambdarank"]["model_path"])
    val_df["_score_lgb"] = booster.predict(X_val)
    m = map_at_k(val_df, "_score_lgb", k=30)
    trained["lightgbm_lambdarank"]["val_map@30"] = m
    val_results.append({"model": "lightgbm_lambdarank", "val_map@30": m})

# CatBoost score
if "catboost_yetirank" in trained and HAS_CAT:
    model = CatBoostRanker()
    model.load_model(trained["catboost_yetirank"]["model_path"])
    val_df["_score_cat"] = model.predict(X_val)
    m = map_at_k(val_df, "_score_cat", k=30)
    trained["catboost_yetirank"]["val_map@30"] = m
    val_results.append({"model": "catboost_yetirank", "val_map@30": m})

# Logistic regression score
if "logreg_pointwise" in trained:
    clf = joblib.load(trained["logreg_pointwise"]["model_path"])
    # Prefer predict_proba if available
    if hasattr(clf, "predict_proba"):
        val_df["_score_lr"] = clf.predict_proba(X_val)[:, 1]
    else:
        val_df["_score_lr"] = clf.decision_function(X_val)
    m = map_at_k(val_df, "_score_lr", k=30)
    trained["logreg_pointwise"]["val_map@30"] = m
    val_results.append({"model": "logreg_pointwise", "val_map@30": m})

# Results table + best model
results_df = pd.DataFrame(val_results).sort_values("val_map@30", ascending=False).reset_index(drop=True)
os.makedirs(OUT_DIR, exist_ok=True)
results_df.to_csv(VAL_RESULTS_CSV, index=False)
print("Saved validation results table:", VAL_RESULTS_CSV)

print("\n================ Validation Results ================")
if len(results_df) == 0:
    print("No models were trained/scored.")
    best_model = None
else:
    print(results_df.to_string(index=False))
    best_model = results_df.loc[0, "model"]
    best_score = float(results_df.loc[0, "val_map@30"])
    print("----------------------------------------------------")
    print(f"Best model on validation (MAP@30): {best_model}  ({best_score:.4f})")
print("====================================================\n")

# ============================================================
# 9) Save preprocessing bundle (shared across models)
# ============================================================
prep = {
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "seed": RANDOM_SEED,
    "models_trained": trained,
    "validation": {
        "metric": "MAP@30",
        "results_table": val_results,
        "best_model": best_model,
        "best_model_val_map@30": (best_score if len(val_results) else None),
        "split": {"method": "GroupShuffleSplit", "test_size": 0.2},
    },
    "notes": {
        "labels": "1=physician-curated, 0=constructed negatives",
        "neg_caps": NEG_CAPS,
        "uses_demographics": DEMOG is not None,
        "uses_omim_orpha_usage": USAGE_DICT is not None,
        "ranking_eval_note": "Final evaluation should be run on external test set with the chosen model.",
        "optional_libs": {"lightgbm_installed": HAS_LGB, "catboost_installed": HAS_CAT},
    },
}
with open(PREP_PATH, "w") as f:
    json.dump(prep, f, indent=2)

print("Saved preprocessing bundle:", PREP_PATH)

if SAVE_TRAIN_TABLE:
    try:
        train_df.to_parquet(TRAIN_TABLE_PATH, index=False)
        print("Saved training table:", TRAIN_TABLE_PATH)
    except Exception:
        csv_path = TRAIN_TABLE_PATH.replace(".parquet", ".csv")
        train_df.to_csv(csv_path, index=False)
        print("Saved training table (CSV fallback):", csv_path)

print("Done.")

