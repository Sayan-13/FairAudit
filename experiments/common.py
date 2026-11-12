# things that both experiment scripts need: data loading, split, preprocessing, fairness metrics
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
RES = os.path.join(ROOT, "experiments", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

MODELS = {
    "LogReg": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "LinearSVM": LinearSVC(dual="auto", random_state=RANDOM_STATE, max_iter=5000),
    "RandForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}


def load_data():
    df = fetch_openml("adult", version=2, as_frame=True).frame.copy()
    text_cols = df.select_dtypes(include=["category", "object"]).columns
    for c in text_cols:
        df[c] = df[c].astype(object).apply(lambda v: v.strip() if isinstance(v, str) else v)
    df[text_cols] = df[text_cols].replace("?", np.nan)

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["y"] = df["class"].astype(str).str.contains(">50K").astype(int)
    df["age_group"] = pd.cut(df["age"], [0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
    return df, n_dup


def split_data(df):
    # 25% test first, then 20% of the rest as validation -> 60/15/25
    y = df["y"]
    s = df[["sex", "race", "age_group"]]
    X = df.drop(columns=["class", "y", "fnlwgt", "age_group"])
    X_rest, X_test, y_rest, y_test, s_rest, s_test = train_test_split(
        X, y, s, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_rest, y_rest, s_rest, test_size=0.20, stratify=y_rest, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test


def make_preprocessor(X, drop=()):
    num_cols = [c for c in X.select_dtypes(include="number").columns if c not in drop]
    cat_cols = [c for c in X.select_dtypes(exclude="number").columns if c not in drop]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def get_scores(model, X):
    # the SVM has no predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def overall(y_true, y_pred, score=None):
    d = {"accuracy": accuracy_score(y_true, y_pred),
         "f1": f1_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred),
         "recall": recall_score(y_true, y_pred)}
    if score is not None:
        d["auc"] = roc_auc_score(y_true, score)
    return d


def group_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        yt = y_true[groups == g]
        yp = y_pred[groups == g]
        pos = (yt == 1).sum()
        neg = (yt == 0).sum()
        pred_pos = (yp == 1).sum()
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        rows.append({"group": g, "n": len(yt),
                     "base_rate": yt.mean(),
                     "selection_rate": yp.mean(),
                     "TPR": tp / pos if pos else np.nan,
                     "FPR": fp / neg if neg else np.nan,
                     "FNR": fn / pos if pos else np.nan,
                     "PPV": tp / pred_pos if pred_pos else np.nan,
                     "accuracy": (yp == yt).mean()})
    return pd.DataFrame(rows).set_index("group")


def gaps(t):
    # gap = biggest group value - smallest group value
    return {"DP": t["selection_rate"].max() - t["selection_rate"].min(),
            "EO": t["TPR"].max() - t["TPR"].min(),
            "FPR": t["FPR"].max() - t["FPR"].min(),
            "FNR": t["FNR"].max() - t["FNR"].min(),
            "PPV": t["PPV"].max() - t["PPV"].min()}


def bootstrap_gaps(y_true, y_pred, groups, n_boot):
    # resample the test set n_boot times and take the 2.5 / 97.5 percentiles of each gap
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    n = len(y_true)
    out = {"DP": [], "EO": [], "FPR": [], "PPV": []}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        g = gaps(group_metrics(y_true[idx], y_pred[idx], groups[idx]))
        for k in out:
            out[k].append(g[k])
    ci = {}
    for k in out:
        ci[k] = (float(np.nanpercentile(out[k], 2.5)), float(np.nanpercentile(out[k], 97.5)))
    return ci


def chi2_test(y_pred, groups):
    table = pd.crosstab(np.asarray(groups), np.asarray(y_pred))
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return float(chi2), float(p), int(dof)


def evaluate(model, setting, y_test, y_pred, s_test, score, n_boot):
    # one row of a mitigation table + the per-group tables
    row = {"model": model, "setting": setting}
    row.update(overall(y_test, y_pred, score))
    tables = {}
    for attr in s_test.columns:
        t = group_metrics(y_test, y_pred, s_test[attr])
        tables[attr] = t
        for k, v in gaps(t).items():
            row[attr + "_" + k] = v
    ci = bootstrap_gaps(y_test, y_pred, s_test["sex"], n_boot)
    for k in ci:
        row["sex_" + k + "_lo"] = ci[k][0]
        row["sex_" + k + "_hi"] = ci[k][1]
    print(model, setting, "acc", round(row["accuracy"], 4), "DP", round(row["sex_DP"], 3), "EO", round(row["sex_EO"], 3))
    return row, tables
