import os
import gc
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

parser = argparse.ArgumentParser("HCDR LightGBM")
parser.add_argument("--data-dir", default=r"D:\home_credit\data\raw", help="Absolute path to Kaggle CSVs")
parser.add_argument("--out-dir",  default=r"D:\home_credit\outputs",  help="Absolute path to save outputs")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--early-stopping", type=int, default=400)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2027, 777, 1337])

args, _ = parser.parse_known_args()

DATA_DIR = os.path.abspath(args.data_dir)
OUT_DIR  = os.path.abspath(args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 2501
N_FOLDS = int(args.folds)
EARLY_STOPPING_ROUNDS = int(args.early_stopping)
N_ESTIMATORS = 20000
LEARNING_RATE = 0.015
SEEDS = list(args.seeds)


# Helpers
def _rank_pct(arr):
    return pd.Series(arr).rank(pct=True, method="average").to_numpy(dtype=np.float32)

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_columns]
    return df, new_cols

def agg_numeric(df, parent_key, df_name, skip_cols=None):
    if skip_cols is None:
        skip_cols = []
    numeric_df = df.drop(columns=[c for c in df.columns if c in skip_cols])
    numeric_df = numeric_df.select_dtypes(include=[np.number]).copy()
    numeric_df[parent_key] = df[parent_key]
    agg = numeric_df.groupby(parent_key).agg(["count", "mean", "max", "min", "sum", "var"])
    agg.columns = [f"{df_name}_{e[0]}_{e[1]}".upper() for e in agg.columns.tolist()]
    agg.reset_index(inplace=True)
    return agg

def agg_categorical(df, parent_key, df_name):
    cat_cols = [c for c in df.columns if (df[c].dtype == "uint8" or df[c].dtype == "int8") and c != parent_key]
    if len(cat_cols) == 0:
        return None
    group = df.groupby(parent_key)[cat_cols].agg(["sum", "mean"])
    group.columns = [f"{df_name}_{c}_{stat}".upper() for (c, stat) in group.columns]
    group.reset_index(inplace=True)
    return group

def days_fix(df):
    day_cols = [c for c in df.columns if c.startswith("DAYS_")]
    for c in day_cols:
        df.loc[df[c] == 365243, c] = np.nan
    return df

def sanitize_columns(cols):
    sanitized = []
    seen = set()
    for c in cols:
        s = str(c)
        s = re.sub(r'[\[\]\{\}"\'\\/:<>,;=\+\*\%\|\?\!\@\#\$\^\&`]', '_', s)  # risky chars
        s = re.sub(r'\s+', '_', s.strip())   # spaces -> underscore
        s = re.sub(r'_+', '_', s)            # collapse underscores
        if s == "":
            s = "f"
        base = s
        k = 1
        while s in seen:
            k += 1
            s = f"{base}__{k}"
        seen.add(s)
        sanitized.append(s)
    return dict(zip(cols, sanitized)), sanitized


def add_more_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9

    def sdiv(a, b):  # safe divide
        return a / (b + eps)

    def sum_if_exists(cols):
        cols = [c for c in cols if c in out.columns]
        return out[cols].sum(axis=1) if cols else 0.0

    # Basic hygiene
    # Age & employment in years (values are inverted because of them being stored as negative offsets in the original dataset)
    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = (-out["DAYS_BIRTH"]) / 365.25
    if "DAYS_EMPLOYED" in out.columns:
        out["EMPLOYED_YEARS"] = (-out["DAYS_EMPLOYED"]) / 365.25

    # Social circle outlier
    for sc in ["OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"]:
        if sc in out.columns:
            out.loc[out[sc] > 30, sc] = np.nan

    # Count of missing values per application
    out["MISSING_VALS_TOTAL_APP"] = out.isna().sum(axis=1)

    #Basic features
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)

    #Income / credit / annuity / goods
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO2"] = sdiv(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(out.columns):
        out["CREDIT_ANNUITY_RATIO"] = sdiv(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO2"] = sdiv(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
        out["INCOME_ANNUITY_DIFF"] = out["AMT_INCOME_TOTAL"] - out["AMT_ANNUITY"]
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(out.columns):
        out["CREDIT_GOODS_RATIO"] = sdiv(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
        out["CREDIT_GOODS_DIFF"]  = out["AMT_CREDIT"] - out["AMT_GOODS_PRICE"]
    if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["GOODS_INCOME_RATIO"] = sdiv(out["AMT_GOODS_PRICE"], out["AMT_INCOME_TOTAL"])

    # EXT_SOURCE stuff
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        ext = out[ext_cols].copy()
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        out["EXT_SOURCE_MIN"]  = ext.min(axis=1)
        out["EXT_SOURCE_MAX"]  = ext.max(axis=1)
        out["EXT_SOURCE_VAR"]  = ext.var(axis=1, ddof=0)
        # weighted
        w = {"EXT_SOURCE_1": 2.0, "EXT_SOURCE_2": 3.0, "EXT_SOURCE_3": 4.0}
        out["WEIGHTED_EXT_SOURCE"] = sum(ext[c] * w[c] for c in ext.columns)
        if "EXT_SOURCE_3" in out.columns:
            if "AMT_INCOME_TOTAL" in out.columns:
                out["INCOME_EXT_RATIO"] = sdiv(out["AMT_INCOME_TOTAL"], out["EXT_SOURCE_3"])
            if "AMT_CREDIT" in out.columns:
                out["CREDIT_EXT_RATIO"] = sdiv(out["AMT_CREDIT"], out["EXT_SOURCE_3"])

    #Age / employment combos
    if {"DAYS_BIRTH", "DAYS_EMPLOYED"}.issubset(out.columns):
        out["AGE_EMPLOYED_DIFF"]    = out["DAYS_BIRTH"] - out["DAYS_EMPLOYED"]
        out["EMPLOYED_TO_AGE_RATIO"] = sdiv(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])

    #Car features
    if {"OWN_CAR_AGE", "DAYS_EMPLOYED"}.issubset(out.columns):
        out["CAR_EMPLOYED_DIFF"]  = out["OWN_CAR_AGE"] - out["DAYS_EMPLOYED"]
        out["CAR_EMPLOYED_RATIO"] = sdiv(out["OWN_CAR_AGE"], out["DAYS_EMPLOYED"])
    if {"OWN_CAR_AGE", "DAYS_BIRTH"}.issubset(out.columns):
        out["CAR_AGE_DIFF"]  = out["DAYS_BIRTH"] - out["OWN_CAR_AGE"]
        out["CAR_AGE_RATIO"] = sdiv(out["OWN_CAR_AGE"], out["DAYS_BIRTH"])

    #Contacts / family
    contact_flags = [c for c in ["FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL"] if c in out.columns]
    out["FLAG_CONTACTS_SUM"] = out[contact_flags].sum(axis=1) if contact_flags else 0
    if {"CNT_FAM_MEMBERS", "CNT_CHILDREN"}.issubset(out.columns):
        out["CNT_NON_CHILDREN"] = out["CNT_FAM_MEMBERS"] - out["CNT_CHILDREN"]
    if {"CNT_CHILDREN", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CHILDREN_INCOME_RATIO"] = sdiv(out["CNT_CHILDREN"], out["AMT_INCOME_TOTAL"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["PER_CAPITA_INCOME2"] = sdiv(out["AMT_INCOME_TOTAL"], (out["CNT_FAM_MEMBERS"] + 1))

    #Hour × credit interaction
    if {"AMT_CREDIT","HOUR_APPR_PROCESS_START"}.issubset(out.columns):
        out["HOUR_PROCESS_CREDIT_MUL"] = out["AMT_CREDIT"] * out["HOUR_APPR_PROCESS_START"]

    #Region ratings interactions (keep numeric, before OHE)
    if {"REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY"}.issubset(out.columns):
        r1 = out["REGION_RATING_CLIENT"].astype(float)
        r2 = out["REGION_RATING_CLIENT_W_CITY"].astype(float)
        out["REGIONS_RATING_INCOME_MUL"] = (r1 + r2) * (out["AMT_INCOME_TOTAL"] if "AMT_INCOME_TOTAL" in out.columns else 1.0) / 2.0
        out["REGION_RATING_MAX"]  = np.maximum(r1, r2)
        out["REGION_RATING_MIN"]  = np.minimum(r1, r2)
        out["REGION_RATING_MEAN"] = (r1 + r2) / 2.0
        out["REGION_RATING_MUL"]  = r1 * r2

    #Region/city mismatch flags sum
    region_flags = [c for c in [
        "REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY"
    ] if c in out.columns]
    out["FLAG_REGIONS"] = out[region_flags].sum(axis=1) if region_flags else 0

    #Apartment / property composites (AVG/MODE/MEDI groups)
    avg_cols  = [c for c in ["APARTMENTS_AVG","BASEMENTAREA_AVG","YEARS_BEGINEXPLUATATION_AVG","YEARS_BUILD_AVG","COMMONAREA_AVG",
                             "ELEVATORS_AVG","ENTRANCES_AVG","FLOORSMAX_AVG","FLOORSMIN_AVG","LANDAREA_AVG","LIVINGAPARTMENTS_AVG",
                             "LIVINGAREA_AVG","NONLIVINGAPARTMENTS_AVG","NONLIVINGAREA_AVG","TOTALAREA_MODE"] if c in out.columns]
    mode_cols = [c for c in ["APARTMENTS_MODE","BASEMENTAREA_MODE","YEARS_BEGINEXPLUATATION_MODE","YEARS_BUILD_MODE","COMMONAREA_MODE",
                             "ELEVATORS_MODE","ENTRANCES_MODE","FLOORSMAX_MODE","FLOORSMIN_MODE","LANDAREA_MODE","LIVINGAPARTMENTS_MODE",
                             "LIVINGAREA_MODE","NONLIVINGAPARTMENTS_MODE","NONLIVINGAREA_MODE","TOTALAREA_MODE"] if c in out.columns]
    medi_cols = [c for c in ["APARTMENTS_MEDI","BASEMENTAREA_MEDI","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BUILD_MEDI","COMMONAREA_MEDI",
                             "ELEVATORS_MEDI","ENTRANCES_MEDI","FLOORSMAX_MEDI","FLOORSMIN_MEDI","LANDAREA_MEDI","LIVINGAPARTMENTS_MEDI",
                             "LIVINGAREA_MEDI","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAREA_MEDI"] if c in out.columns]

    out["APARTMENTS_SUM_AVG"]  = out[avg_cols].sum(axis=1)  if avg_cols  else 0.0
    out["APARTMENTS_SUM_MODE"] = out[mode_cols].sum(axis=1) if mode_cols else 0.0
    out["APARTMENTS_SUM_MEDI"] = out[medi_cols].sum(axis=1) if medi_cols else 0.0

    if "AMT_INCOME_TOTAL" in out.columns:
        out["INCOME_APARTMENT_AVG_MUL"]  = out["APARTMENTS_SUM_AVG"]  * out["AMT_INCOME_TOTAL"]
        out["INCOME_APARTMENT_MODE_MUL"] = out["APARTMENTS_SUM_MODE"] * out["AMT_INCOME_TOTAL"]
        out["INCOME_APARTMENT_MEDI_MUL"] = out["APARTMENTS_SUM_MEDI"] * out["AMT_INCOME_TOTAL"]

    #OBS/DEF social circle combos & credit ratios
    for a, b, name in [
        ("OBS_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","OBS_30_60_SUM"),
        ("DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","DEF_30_60_SUM"),
    ]:
        if {a,b}.issubset(out.columns):
            out[name] = out[a].fillna(0) + out[b].fillna(0)

    if {"OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE"}.issubset(out.columns):
        out["OBS_DEF_30_MUL"] = out["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0) * out["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0)
    if {"OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE"}.issubset(out.columns):
        out["OBS_DEF_60_MUL"] = out["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0) * out["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0)

    sc_all = [c for c in ["OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE"] if c in out.columns]
    out["SUM_OBS_DEF_ALL"] = out[sc_all].sum(axis=1) if sc_all else 0.0

    if "AMT_CREDIT" in out.columns:
        for c, name in [
            ("OBS_30_CNT_SOCIAL_CIRCLE","OBS_30_CREDIT_RATIO"),
            ("OBS_60_CNT_SOCIAL_CIRCLE","OBS_60_CREDIT_RATIO"),
            ("DEF_30_CNT_SOCIAL_CIRCLE","DEF_30_CREDIT_RATIO"),
            ("DEF_60_CNT_SOCIAL_CIRCLE","DEF_60_CREDIT_RATIO"),
        ]:
            if c in out.columns:
                out[name] = sdiv(out["AMT_CREDIT"], out[c])

    #Combined document flags
    doc_flags = [c for c in [
        "FLAG_DOCUMENT_3","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9","FLAG_DOCUMENT_11","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_21"
    ] if c in out.columns]
    out["SUM_FLAGS_DOCUMENTS"] = out[doc_flags].sum(axis=1) if doc_flags else 0

    #“Details change” combos (ID publish, registration, last phone change)
    if {"DAYS_LAST_PHONE_CHANGE","DAYS_REGISTRATION","DAYS_ID_PUBLISH"}.issubset(out.columns):
        a = out["DAYS_LAST_PHONE_CHANGE"]; b = out["DAYS_REGISTRATION"]; c = out["DAYS_ID_PUBLISH"]
        out["DAYS_DETAILS_CHANGE_MUL"] = a * b * c
        out["DAYS_DETAILS_CHANGE_SUM"] = a + b + c

    #Enquiries sums & ratios
    enq_cols = [c for c in [
        "AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR"
    ] if c in out.columns]
    out["AMT_ENQ_SUM"] = out[enq_cols].sum(axis=1) if enq_cols else 0.0
    if "AMT_CREDIT" in out.columns:
        out["ENQ_CREDIT_RATIO"] = sdiv(out["AMT_ENQ_SUM"], out["AMT_CREDIT"])

    return out


# Load Application Data
print("Reading application files...")
app_train = pd.read_csv(os.path.join(DATA_DIR, "application_train.csv"))
app_test  = pd.read_csv(os.path.join(DATA_DIR, "application_test.csv"))

print("Initial shapes:", app_train.shape, app_test.shape)

y = app_train["TARGET"].astype(int).values
train_id = app_train["SK_ID_CURR"].values
test_id  = app_test["SK_ID_CURR"].values

app = pd.concat([app_train.drop(columns=["TARGET"]), app_test], axis=0, ignore_index=True)

# Clean/engineer application-level features
app = days_fix(app)
app = add_more_application_features(app)
app, app_cat_cols = one_hot_encoder(app, nan_as_category=True)

# Bureau + Bureau Balance
print("Processing bureau & bureau_balance...")
bureau = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"))
bb = pd.read_csv(os.path.join(DATA_DIR, "bureau_balance.csv"))

# One-hot bureau_balance, aggregate to SK_ID_BUREAU
bb, bb_cat_cols = one_hot_encoder(bb, nan_as_category=True)

# Last-12-months arrears features from bureau_balance
bb_recent = bb[bb["MONTHS_BALANCE"] >= -12].copy()
bad_cols = [c for c in ["STATUS_1","STATUS_2","STATUS_3","STATUS_4","STATUS_5"] if c in bb_recent.columns]
if bad_cols:
    bb_recent["BAD_ROW"] = bb_recent[bad_cols].sum(axis=1)
else:
    bb_recent["BAD_ROW"] = 0
bb12 = bb_recent.groupby("SK_ID_BUREAU").agg(
    MB12_SIZE=("MONTHS_BALANCE", "size"),
    MB12_BAD_SUM=("BAD_ROW", "sum"),
).reset_index()
bb12["MB12_BAD_RATE"] = bb12["MB12_BAD_SUM"] / (bb12["MB12_SIZE"] + 1e-9)

# Full-history compact aggregation including status counts
bb_agg_dict = {"MONTHS_BALANCE": ["min", "max", "size"]}
for label in ["0", "1", "2", "3", "4", "5", "C", "X"]:
    col = f"STATUS_{label}"
    if col in bb.columns:
        bb_agg_dict[col] = ["sum"]
bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_agg_dict)
bb_agg.columns = [f"BB_{c}_{stat}".upper() for c, stat in bb_agg.columns]
bb_agg.reset_index(inplace=True)

# Merge bb features into bureau rows
bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
bureau = bureau.merge(bb12, on="SK_ID_BUREAU", how="left")
del bb, bb_agg, bb12, bb_recent
gc.collect()

# One-hot bureau and aggregate to SK_ID_CURR
bureau = days_fix(bureau)
bureau, bureau_cat_cols = one_hot_encoder(bureau, nan_as_category=True)

bureau_agg_num = agg_numeric(bureau, "SK_ID_CURR", "BUREAU", skip_cols=["SK_ID_BUREAU"])
bureau_agg_cat = agg_categorical(bureau, "SK_ID_CURR", "BUREAU")
bureau_agg = bureau_agg_num if bureau_agg_cat is None else bureau_agg_num.merge(bureau_agg_cat, on="SK_ID_CURR", how="left")

# Active / Closed subsets
for status in ["CREDIT_ACTIVE_Active", "CREDIT_ACTIVE_Closed"]:
    if status in bureau.columns:
        subset = bureau[bureau[status] == 1]
        sub_agg = agg_numeric(subset, "SK_ID_CURR", f"BUREAU_{status.split('_')[-1]}", skip_cols=["SK_ID_BUREAU"])
        bureau_agg = bureau_agg.merge(sub_agg, on="SK_ID_CURR", how="left")
        del subset, sub_agg
        gc.collect()

del bureau
gc.collect()

# Previous Application
print("Processing previous_application...")
prev = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"))
prev = days_fix(prev)
prev, prev_cat_cols = one_hot_encoder(prev, nan_as_category=True)

prev_agg_num = agg_numeric(prev, "SK_ID_CURR", "PREV")
prev_agg_cat = agg_categorical(prev, "SK_ID_CURR", "PREV")
prev_agg = prev_agg_num if prev_agg_cat is None else prev_agg_num.merge(prev_agg_cat, on="SK_ID_CURR", how="left")

# POS_CASH_balance
print("Processing POS_CASH_balance...")
pos = pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"))
pos, pos_cat_cols = one_hot_encoder(pos, nan_as_category=True)

pos_agg_num = agg_numeric(pos, "SK_ID_CURR", "POS")
pos_agg_cat = agg_categorical(pos, "SK_ID_CURR", "POS")
pos_agg = pos_agg_num if pos_agg_cat is None else pos_agg_num.merge(pos_agg_cat, on="SK_ID_CURR", how="left")
del pos
gc.collect()

# Installments Payments
print("Processing installments_payments...")
ins = pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"))
ins = days_fix(ins)
# helpful deltas
ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1e-9)
ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
# Late flags
ins["LATE_FLAG"] = (ins["DPD"] > 0).astype(np.uint8)

ins_agg = ins.groupby("SK_ID_CURR").agg(
    INSTAL_COUNT=("NUM_INSTALMENT_VERSION", "size"),
    INSTAL_AMT_PAY_SUM=("AMT_PAYMENT", "sum"),
    INSTAL_AMT_PAY_MEAN=("AMT_PAYMENT", "mean"),
    INSTAL_AMT_INST_SUM=("AMT_INSTALMENT", "sum"),
    INSTAL_PAY_PERC_MEAN=("PAYMENT_PERC", "mean"),
    INSTAL_PAY_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
    INSTAL_DPD_MEAN=("DPD", "mean"),
    INSTAL_DPD_MAX=("DPD", "max"),
    INSTAL_DBD_MEAN=("DBD", "mean"),
    INSTAL_DBD_MAX=("DBD", "max"),
    INSTAL_LATE_RATE=("LATE_FLAG", "mean"),
    INSTAL_LATE_SUM=("LATE_FLAG", "sum"),
).reset_index()
del ins
gc.collect()

# Credit Card Balance
print("Processing credit_card_balance...")
ccb = pd.read_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"))
# Utilization & payment ratios
eps = 1e-9
if "AMT_CREDIT_LIMIT_ACTUAL" in ccb.columns:
    ccb["BALANCE_LIMIT_RATIO"]  = ccb["AMT_BALANCE"] / (ccb["AMT_CREDIT_LIMIT_ACTUAL"] + eps)
    ccb["DRAWINGS_LIMIT_RATIO"] = ccb["AMT_DRAWINGS_CURRENT"] / (ccb["AMT_CREDIT_LIMIT_ACTUAL"] + eps)
if "AMT_INST_MIN_REGULARITY" in ccb.columns:
    ccb["PAYMENT_MIN_RATIO"]    = ccb["AMT_PAYMENT_TOTAL_CURRENT"] / (ccb["AMT_INST_MIN_REGULARITY"] + eps)
if "AMT_TOTAL_RECEIVABLE" in ccb.columns:
    ccb["PAYMENT_TOTAL_RATIO"]  = ccb["AMT_PAYMENT_TOTAL_CURRENT"] / (ccb["AMT_TOTAL_RECEIVABLE"] + eps)

ccb, ccb_cat_cols = one_hot_encoder(ccb, nan_as_category=True)

ccb_agg_num = agg_numeric(ccb, "SK_ID_CURR", "CCB", skip_cols=["SK_ID_PREV"])
ccb_agg_cat = agg_categorical(ccb, "SK_ID_CURR", "CCB")
ccb_agg = ccb_agg_num if ccb_agg_cat is None else ccb_agg_num.merge(ccb_agg_cat, on="SK_ID_CURR", how="left")
del ccb
gc.collect()

# Merge all engineered tables into applications
print("Merging engineered features...")
for block in [bureau_agg, prev_agg, pos_agg, ins_agg, ccb_agg]:
    app = app.merge(block, on="SK_ID_CURR", how="left")
    del block
    gc.collect()

# Split back into train/test
X = app.iloc[: len(train_id), :].copy()
X_test = app.iloc[len(train_id):, :].copy()
del app, app_train, app_test
gc.collect()

# Remove identifier columns from features
drop_cols = ["SK_ID_CURR"]
feature_cols = [c for c in X.columns if c not in drop_cols]

# Fill NA
X[feature_cols] = X[feature_cols].fillna(0)
X_test[feature_cols] = X_test[feature_cols].fillna(0)

# Ensure numeric types for LightGBM
for df_ in (X, X_test):
    for c in feature_cols:
        if df_[c].dtype == "bool":
            df_[c] = df_[c].astype(np.uint8)
        elif str(df_[c].dtype) not in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "float16", "float32", "float64"):
            df_[c] = df_[c].astype(np.float32)

# Sanitize feature names
rename_map, sanitized_cols = sanitize_columns(feature_cols)
if rename_map:
    X.rename(columns=rename_map, inplace=True)
    X_test.rename(columns=rename_map, inplace=True)
    feature_cols = sanitized_cols


def train_lgb_one_seed(seed, X, y, feature_cols, X_test, n_folds=5, esr=200, base_params=None):
    params = dict(base_params)
    params["random_state"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    skf_local = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_s = np.zeros(X.shape[0], dtype=np.float32)
    test_s = np.zeros(X_test.shape[0], dtype=np.float64)
    gain_s = defaultdict(float)

    for trn_idx, val_idx in skf_local.split(X[feature_cols], y):
        X_trn, y_trn = X.iloc[trn_idx][feature_cols], y[trn_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(esr, first_metric_only=True),
                       lgb.log_evaluation(50)],
        )
        best_iter = getattr(model, "best_iteration_", params["n_estimators"])

        oof_s[val_idx] = model.predict_proba(X_val, num_iteration=best_iter)[:, 1].astype(np.float32)
        test_s += model.predict_proba(X_test[feature_cols], num_iteration=best_iter)[:, 1] / n_folds

        names = model.booster_.feature_name()
        gains = model.booster_.feature_importance(importance_type="gain")
        for n, g in zip(names, gains):
            gain_s[n] += float(g)
    return oof_s, test_s, gain_s


print("Starting training (multi-seed, rank-bagging)...")


# class imbalance calc (in case of using them)
pos = int(y.sum())
neg = len(y) - pos
scale_pos_weight = neg / max(pos, 1)

#LightGBM params
lgb_params = dict(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    num_leaves=128,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.6,
    min_child_samples=50,
    objective="binary",
    metric="auc",
    first_metric_only=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    verbosity=-1,
)

# Unbalanced logs toggler
USE_SPW = False
if not USE_SPW:
    lgb_params.pop("scale_pos_weight", None)

oof_seeds = []
test_seeds = []
feat_gain_total = defaultdict(float)

for s in SEEDS:
    oof_s, test_s, gain_s = train_lgb_one_seed(
        seed=s,
        X=X, y=y, feature_cols=feature_cols, X_test=X_test,
        n_folds=N_FOLDS, esr=EARLY_STOPPING_ROUNDS,
        base_params=lgb_params,
    )
    oof_seeds.append(oof_s)
    test_seeds.append(test_s)
    for k, v in gain_s.items():
        feat_gain_total[k] += v / len(SEEDS)

# rank bagging aggregation
oof_ranked = np.stack([_rank_pct(o) for o in oof_seeds], axis=1)
oof_pred = oof_ranked.mean(axis=1).astype(np.float32)

test_ranked = np.stack([_rank_pct(t) for t in test_seeds], axis=1)
test_pred = test_ranked.mean(axis=1).astype(np.float32)

cv_auc = roc_auc_score(y, oof_pred)
print(f"\nMulti-seed Rank-Bagged OOF AUC: {cv_auc:.6f}")


# Submission
pred_for_submit = test_pred
submission = pd.DataFrame({"SK_ID_CURR": test_id, "TARGET": pred_for_submit.astype(np.float32)})
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(f"\nWrote {os.path.join(OUT_DIR, 'submission.csv')}")

# Save OOF for ensembling (should be deprecated but I'll keep it cuz why not)
oof_df = pd.DataFrame({"SK_ID_CURR": train_id, "OOF_PRED": oof_pred.astype(np.float32), "TARGET": y})
oof_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)
print(f"Wrote {os.path.join(OUT_DIR, 'oof_predictions.csv')}")

print("\nDone.")