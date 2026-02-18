import re
import random
import pandas as pd
from collections import defaultdict, deque
from functools import lru_cache

from pyhpo import Ontology

# ============================================================
# 0) Load ontology + data
# ============================================================
Ontology()  # loads HPO (safe to call multiple times; may print "built before already")

# Load in follow-up report from Gateway
df = pd.read_csv("follow_up_report_2024-05-05.csv")

# Filter patients: exclude Vanderbilt, exclude Not Accepted, require non-empty HPO term IDs
df_filtered = df[
    (df["Clinical Site"] != "Vanderbilt")
    & (df["Clinical Site"].notna())
    & (df["Clinical Site"] != "")
    & (df["Status"] != "Not Accepted")
]
df_filtered = df_filtered[~(df_filtered["HPO term IDs"].isna() | (df_filtered["HPO term IDs"] == ""))].reset_index(drop=True)

ROOT_HP = "HP:0000001"  # Phenotypic abnormality
HPO_RE = re.compile(r"HP:\d{7}")

# ============================================================
# 1) ID helpers (robust to pyhpo term.id being a string like "HP:0000421")
# ============================================================
def as_int_hpo_id(x):
    """
    Normalize x to an integer HPO id (e.g., 421).

    Accepts:
      - pyhpo term objects (term.id like "HP:0000421")
      - "HP:0000421"
      - "0000421" or "421"
      - int
    """
    if hasattr(x, "id"):
        x = x.id

    if isinstance(x, int):
        return x

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("HP:"):
            return int(s.split(":")[1])
        if s.isdigit():
            return int(s)
        m = re.search(r"(\d{1,7})", s)
        if m:
            return int(m.group(1))

    raise ValueError(f"Can't parse HPO id from: {x} ({type(x)})")

def int_to_hp(x) -> str:
    """Always returns canonical 'HP:xxxxxxx'."""
    return f"HP:{as_int_hpo_id(x):07d}"

def parse_hpo_ids(hpo_id_string):
    """Extracts ['HP:0000421', ...] from a semicolon-separated string."""
    if pd.isna(hpo_id_string):
        return []
    return HPO_RE.findall(str(hpo_id_string))

def get_term(hp_str):
    """hp_str is 'HP:xxxxxxx'."""
    return Ontology.get_hpo_object(as_int_hpo_id(hp_str))

def parents(term):
    return list(term.parents) if term is not None else []

def children(term):
    return list(term.children) if term is not None else []

_root_term = get_term(ROOT_HP)
_root_int = as_int_hpo_id(_root_term)

# ============================================================
# 2) Enumerate all phenotype terms under HP:0000118 (robust, no Ontology.terms needed)
# ============================================================
def enumerate_terms_under_root(root_term):
    """BFS down the ontology from root_term via children, returning term objects (excluding root)."""
    visited = set([as_int_hpo_id(root_term)])
    q = deque([root_term])
    out = []
    while q:
        cur = q.popleft()
        for ch in children(cur):
            cid = as_int_hpo_id(ch)
            if cid not in visited:
                visited.add(cid)
                out.append(ch)
                q.append(ch)
    return out

ALL_TERMS = enumerate_terms_under_root(_root_term)
ALL_TERM_IDS = [as_int_hpo_id(t) for t in ALL_TERMS]  # IMPORTANT: ints only

# ============================================================
# 3) Top-level branch mapping (child of HP:0000118) with caching
# ============================================================
@lru_cache(maxsize=None)
def top_level_branch_id(term_id):
    """
    Return integer id of the top-level branch for term_id:
    the child of ROOT on a path from term to ROOT.
    If multiple, returns the smallest id (deterministic).
    """
    term_int_id = as_int_hpo_id(term_id)

    if term_int_id == _root_int:
        return None

    try:
        term = Ontology.get_hpo_object(term_int_id)
    except Exception:
        return None

    branches = set()
    q = deque([(term, None)])  # (current, child_below_current)
    visited = set([as_int_hpo_id(term)])

    while q:
        cur, child_below = q.popleft()
        cur_id = as_int_hpo_id(cur)

        if cur_id == _root_int:
            if child_below is not None:
                branches.add(as_int_hpo_id(child_below))
            continue

        for p in parents(cur):
            pid = as_int_hpo_id(p)
            if pid not in visited:
                visited.add(pid)
                next_child_below = cur if child_below is None else child_below
                q.append((p, next_child_below))

    return min(branches) if branches else None

# Branch -> list[int term_id]
TERMS_BY_BRANCH = defaultdict(list)
for t in ALL_TERMS:
    tid = as_int_hpo_id(t)
    bid = top_level_branch_id(tid)
    if bid is not None:
        TERMS_BY_BRANCH[bid].append(tid)

# ============================================================
# 4) Neighborhood utilities (all return SETS OF INTS)
# ============================================================
def ancestors_upto(term, depth=3):
    out = set()
    current = [term]
    for _ in range(depth):
        nxt = []
        for t in current:
            for p in parents(t):
                pid = as_int_hpo_id(p)
                if pid not in out:
                    out.add(pid)
                    nxt.append(p)
        current = nxt
        if not current:
            break
    return out

def descendants_upto(term, depth=3):
    out = set()
    current = [term]
    for _ in range(depth):
        nxt = []
        for t in current:
            for c in children(t):
                cid = as_int_hpo_id(c)
                if cid not in out:
                    out.add(cid)
                    nxt.append(c)
        current = nxt
        if not current:
            break
    return out

def siblings(term):
    sibs = set()
    tid = as_int_hpo_id(term)
    for p in parents(term):
        for c in children(p):
            cid = as_int_hpo_id(c)
            if cid != tid:
                sibs.add(cid)
    return sibs

def cousins(term):
    cous = set()
    tid = as_int_hpo_id(term)
    for p in parents(term):
        for gp in parents(p):
            for child_of_gp in children(gp):
                for gc in children(child_of_gp):
                    cid = as_int_hpo_id(gc)
                    if cid != tid:
                        cous.add(cid)
    return cous


def neighbors(term):
    """1-hop neighbors in the HPO graph: parents + children (term objects)."""
    return parents(term) + children(term)

def nodes_at_distance_band(pos_terms, dmin=3, dmax=5, per_term_cap=5000):
    """
    Collect candidate node IDs whose shortest-path distance from ANY positive term
    is in [dmin, dmax] using a BFS expansion from positives.
    Returns a set of int term IDs.
    """
    # Multi-source BFS from all positives
    start_ids = [as_int_hpo_id(t) for t in pos_terms]
    visited = set(start_ids)
    q = deque()

    # queue entries: (term_object, distance)
    for t in pos_terms:
        q.append((t, 0))

    band = set()
    expansions = 0

    while q:
        cur, dist = q.popleft()
        if dist >= dmax:
            continue

        for nb in neighbors(cur):
            nid = as_int_hpo_id(nb)
            if nid in visited:
                continue
            visited.add(nid)
            ndist = dist + 1

            if dmin <= ndist <= dmax:
                band.add(nid)

            q.append((nb, ndist))

        expansions += 1
        if expansions >= per_term_cap:
            # safety cap to avoid runaway on huge graphs
            break

    return band

# ============================================================
# 5) Negative sampling per patient
# ============================================================
def construct_patient_negatives(
    pos_hp_list,
    n_hard=10,
    n_medium=15,
    n_easy=15,
    n_implausible=10,
    exclude_depth=3,
    seed=0
):
    """
    pos_hp_list: list of 'HP:xxxxxxx' positives for a patient.
    Returns dict of lists of 'HP:xxxxxxx' negatives in 4 categories.
    """
    rng = random.Random(seed)

    # Convert positives to term objects
    pos_terms = []
    for hp in pos_hp_list:
        try:
            pos_terms.append(get_term(hp))
        except Exception:
            continue
    pos_terms = [t for t in pos_terms if t is not None]
    if not pos_terms:
        return {"hard": [], "medium": [], "easy": [], "implausible": []}

    pos_ids = set(as_int_hpo_id(t) for t in pos_terms)

    # Patient branch set
    patient_branches = set()
    for t in pos_terms:
        bid = top_level_branch_id(as_int_hpo_id(t))
        if bid is not None:
            patient_branches.add(bid)

    # Global exclusions used for hard/easy/implausible:
    # Keep ancestors (avoid trivial generalizations) but only shallow descendants
    excl = set(pos_ids)
    for t in pos_terms:
        excl |= ancestors_upto(t, depth=3)
        excl |= descendants_upto(t, depth=1)  # was 3; depth=1 prevents wiping out huge areas

    # Hard pool: siblings + cousins (near-miss)
    hard_pool = set()
    for t in pos_terms:
        hard_pool |= siblings(t)
        hard_pool |= cousins(t)
    hard_pool -= excl

    # MEDIUM pool: graph-distance band from positives (same neighborhood, but not too close)
    # Distance band chosen to be "a couple of degrees off"
    medium_pool = nodes_at_distance_band(pos_terms, dmin=3, dmax=5)

    # Exclude ambiguous + too-close items:
    # - positives
    # - ancestors (generalizations)
    # - descendants depth 1 (optional; prevents trivially-more-specific)
    # - hard_pool (siblings/cousins)
    excl_medium = set(pos_ids)
    for t in pos_terms:
        excl_medium |= ancestors_upto(t, depth=3)
        excl_medium |= descendants_upto(t, depth=1)  # set to 0 by removing this line if you want

    medium_pool -= excl_medium
    medium_pool -= hard_pool

    # Easy pool: different branches (out-of-domain)
    easy_pool = set()
    for tid in ALL_TERM_IDS:
        if tid in excl:
            continue
        bid = top_level_branch_id(tid)
        if bid is None:
            continue
        if bid not in patient_branches:
            easy_pool.add(tid)

    # Implausible pool: different branches + no shared close ancestors (depth=2)
    pos_anc2 = set()
    for t in pos_terms:
        pos_anc2 |= ancestors_upto(t, depth=2)

    implausible_pool = set()
    for tid in easy_pool:
        try:
            term = Ontology.get_hpo_object(tid)
        except Exception:
            continue
        anc2 = ancestors_upto(term, depth=2)
        if anc2.isdisjoint(pos_anc2):
            implausible_pool.add(tid)

    def sample(pool, k):
        pool = list(pool)
        if not pool:
            return []
        if len(pool) <= k:
            rng.shuffle(pool)
            return pool
        return rng.sample(pool, k)

    def ids_to_hp(ids_):
        return [int_to_hp(i) for i in ids_]

    return {
        "hard": ids_to_hp(sample(hard_pool, n_hard)),
        "medium": ids_to_hp(sample(medium_pool, n_medium)),
        "easy": ids_to_hp(sample(easy_pool, n_easy)),
        "implausible": ids_to_hp(sample(implausible_pool, n_implausible)),
    }

def add_negative_sets(df_in, seed=33, print_every=50):
    out = df_in.copy()
    out["pos_hpo_ids"] = out["HPO term IDs"].apply(parse_hpo_ids)

    hard_col, med_col, easy_col, impl_col = [], [], [], []

    n = len(out)

    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        pos_list = row["pos_hpo_ids"]

        if i == 1 or i % print_every == 0 or i == n:
            uid = row.get("UDN ID", "NA")
            print(f"Processing patient {i}/{n} (UDN ID = {uid})")

        negs = construct_patient_negatives(
            pos_list,
            n_hard=10,
            n_medium=15,
            n_easy=15,
            n_implausible=10,
            exclude_depth=3,
            seed=seed + i
        )

        hard_col.append("; ".join(negs["hard"]) + (";" if negs["hard"] else ""))
        med_col.append("; ".join(negs["medium"]) + (";" if negs["medium"] else ""))
        easy_col.append("; ".join(negs["easy"]) + (";" if negs["easy"] else ""))
        impl_col.append("; ".join(negs["implausible"]) + (";" if negs["implausible"] else ""))

    out["neg_hpo_ids_hard"] = hard_col
    out["neg_hpo_ids_medium"] = med_col
    out["neg_hpo_ids_easy"] = easy_col
    out["neg_hpo_ids_implausible"] = impl_col
    return out


# ============================================================
# 6) Run + save
# ============================================================
df_with_negs = add_negative_sets(df_filtered, seed=33)
df_with_negs.to_csv("UDN_patients_with_negative_hpo_sets.csv", index=False)

print("Saved: UDN_patients_with_negative_hpo_sets.csv")
