# Part 1: data, 4 models, loss curves, fairness audit, significance tests, reweighing, group thresholds
# run: .venv/bin/python experiments/run_experiments.py
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

from common import (RANDOM_STATE, ROOT, FIG, RES, MODELS, load_data, split_data, make_preprocessor,
                    get_scores, overall, group_metrics, gaps, bootstrap_gaps, chi2_test, evaluate)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "figure.dpi": 150, "savefig.bbox": "tight"})
summary = {}

# 1. load and clean
df, n_dup = load_data()
print("rows after cleaning:", len(df), "duplicates dropped:", n_dup)

missing = df.isna().sum()
missing = missing[missing > 0]
summary["data"] = {
    "raw_rows": len(df) + n_dup,
    "duplicates_dropped": n_dup,
    "clean_rows": len(df),
    "missing_per_col": {k: int(v) for k, v in missing.items()},
    "pos_rate": float(df["y"].mean()),
    "sex_counts": df["sex"].value_counts().to_dict(),
    "race_counts": df["race"].value_counts().to_dict(),
    "income_rate_by_sex": df.groupby("sex")["y"].mean().round(4).to_dict(),
    "income_rate_by_race": df.groupby("race")["y"].mean().round(4).to_dict(),
}

# 5 example rows for the report
sample_cols = ["age", "workclass", "education", "marital-status", "occupation", "race", "sex",
               "hours-per-week", "native-country", "class"]
df[sample_cols].sample(5, random_state=RANDOM_STATE).to_csv(os.path.join(RES, "sample_rows.csv"), index=False)

# 2. split
X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test = split_data(df)
summary["split"] = {
    "total": len(df), "train": len(X_train), "val": len(X_val), "test": len(X_test),
    "train_full": len(X_train) + len(X_val),
    "pos_rate": {"train": float(y_train.mean()), "val": float(y_val.mean()), "test": float(y_test.mean())},
    "test_sex_counts": s_test["sex"].value_counts().to_dict(),
    "test_race_counts": s_test["race"].value_counts().to_dict(),
}
print(summary["split"])

# 3. the fairness functions are in common.py

# 4. train the 4 models, with and without sex/race, tuned with 3-fold CV on the training set
GRIDS = {
    "LogReg": {"clf__C": [0.01, 0.1, 1, 10]},
    "LinearSVM": {"clf__C": [0.01, 0.1, 1]},
    "RandForest": {"clf__n_estimators": [200], "clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [1, 5]},
    "HistGB": {"clf__learning_rate": [0.05, 0.1], "clf__max_iter": [200, 400], "clf__max_leaf_nodes": [15, 31]},
}
DROP = {"with": (), "without": ("sex", "race")}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

results = []
per_group = []
preds, scores, fitted, best_params = {}, {}, {}, {}
for name in MODELS:
    for variant in ["with", "without"]:
        pipe = Pipeline([("pre", make_preprocessor(X_train, DROP[variant])), ("clf", MODELS[name])])
        gs = GridSearchCV(pipe, GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
        start = time.time()
        gs.fit(X_train, y_train)
        fit_time = time.time() - start

        model = gs.best_estimator_
        y_pred_val = model.predict(X_val)
        y_pred = model.predict(X_test)
        score = get_scores(model, X_test)
        preds[(name, variant)] = y_pred
        scores[(name, variant)] = score
        fitted[(name, variant)] = model
        best_params[name + "_" + variant] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}

        row = {"model": name, "variant": variant, "cv_f1": gs.best_score_,
               "val_accuracy": accuracy_score(y_val, y_pred_val), "val_f1": f1_score(y_val, y_pred_val),
               "fit_time_s": fit_time}
        row.update(overall(y_test, y_pred, score))
        for attr in ["sex", "race"]:
            t = group_metrics(y_test, y_pred, s_test[attr])
            for k, v in gaps(t).items():
                row[attr + "_" + k] = v
            t = t.reset_index()
            t.insert(0, "attr", attr)
            t.insert(0, "variant", variant)
            t.insert(0, "model", name)
            per_group.append(t)
        results.append(row)
        print(name, variant, gs.best_params_, "acc", round(row["accuracy"], 4), "f1", round(row["f1"], 4),
              "sex DP", round(row["sex_DP"], 4), "sex EO", round(row["sex_EO"], 4))

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(RES, "model_results.csv"), index=False)
pd.concat(per_group).to_csv(os.path.join(RES, "per_group_metrics.csv"), index=False)
summary["best_params"] = best_params

# 5. loss curves
# logistic regression with SGD, one pass over the training data = one epoch
pre = make_preprocessor(X_train).fit(X_train)
Xtr = pre.transform(X_train)
Xva = pre.transform(X_val)
sgd = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal", random_state=RANDOM_STATE)
EPOCHS = 40
tr_loss, va_loss, va_acc = [], [], []
rng = np.random.default_rng(RANDOM_STATE)
for ep in range(EPOCHS):
    order = rng.permutation(len(Xtr))
    sgd.partial_fit(Xtr[order], y_train.values[order], classes=[0, 1])
    tr_loss.append(log_loss(y_train, sgd.predict_proba(Xtr)[:, 1]))
    va_loss.append(log_loss(y_val, sgd.predict_proba(Xva)[:, 1]))
    va_acc.append(accuracy_score(y_val, sgd.predict(Xva)))
summary["sgd_curve"] = {"epochs": EPOCHS, "train_loss_first": tr_loss[0], "train_loss_last": tr_loss[-1],
                        "val_loss_first": va_loss[0], "val_loss_last": va_loss[-1],
                        "val_loss_min": float(np.min(va_loss)), "val_loss_argmin": int(np.argmin(va_loss)) + 1,
                        "val_acc_last": va_acc[-1]}

# gradient boosting, loss after every boosting iteration
hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=400, max_leaf_nodes=31,
                                     early_stopping=False, random_state=RANDOM_STATE)
hgb.fit(Xtr, y_train)
g_tr = [log_loss(y_train, p[:, 1]) for p in hgb.staged_predict_proba(Xtr)]
g_va = [log_loss(y_val, p[:, 1]) for p in hgb.staged_predict_proba(Xva)]
summary["hgb_curve"] = {"iters": len(g_tr), "train_first": g_tr[0], "train_last": g_tr[-1],
                        "val_first": g_va[0], "val_last": g_va[-1],
                        "val_min": float(np.min(g_va)), "val_argmin": int(np.argmin(g_va)) + 1}
print("sgd", summary["sgd_curve"])
print("hgb", summary["hgb_curve"])

epochs = np.arange(1, EPOCHS + 1)
iters = np.arange(1, len(g_tr) + 1)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
axes[0].plot(epochs, tr_loss, color="#2563eb", label="training loss")
axes[0].plot(epochs, va_loss, color="#ea580c", ls="--", label="validation loss")
axes[0].set_xlabel("epoch (one SGD pass over the training set)")
axes[0].set_title("(a) Logistic regression trained by SGD")
axes[1].plot(iters, g_tr, color="#2563eb", label="training loss")
axes[1].plot(iters, g_va, color="#ea580c", ls="--", label="validation loss")
axes[1].axvline(summary["hgb_curve"]["val_argmin"], color="#6b7280", lw=0.8, ls=":")
axes[1].set_xlabel("boosting iteration")
axes[1].set_title("(b) Gradient boosting")
for ax in axes:
    ax.set_ylabel("log loss")
    ax.legend()
    ax.grid(alpha=.3)
fig.savefig(os.path.join(FIG, "loss_curve.pdf"))
plt.close(fig)

pd.DataFrame({"epoch": epochs, "train_loss": tr_loss, "val_loss": va_loss, "val_acc": va_acc}).to_csv(
    os.path.join(RES, "sgd_loss_curve.csv"), index=False)
pd.DataFrame({"iter": iters, "train_loss": g_tr, "val_loss": g_va}).to_csv(
    os.path.join(RES, "hgb_loss_curve.csv"), index=False)

# 6. are the gaps significant? bootstrap CI, z-test and chi-square test for every model
def ztest(y_pred, groups, a, b):
    # two-proportion z-test on the selection rates of group a and group b
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    xa, na = y_pred[groups == a].sum(), (groups == a).sum()
    xb, nb = y_pred[groups == b].sum(), (groups == b).sum()
    p = (xa + xb) / (na + nb)
    z = (xa / na - xb / nb) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return float(z), float(2 * stats.norm.sf(abs(z)))

stat_rows = []
for (name, variant), y_pred in preds.items():
    for attr in ["sex", "race"]:
        chi2, chi2_p, dof = chi2_test(y_pred, s_test[attr])
        row = {"model": name, "variant": variant, "attr": attr, "chi2": chi2, "chi2_p": chi2_p, "dof": dof}
        ci = bootstrap_gaps(y_test, y_pred, s_test[attr], 1000)
        for k in ci:
            row[k + "_lo"] = ci[k][0]
            row[k + "_hi"] = ci[k][1]
        if attr == "sex":
            row["z"], row["z_p"] = ztest(y_pred, s_test[attr], "Male", "Female")
        else:
            row["z"], row["z_p"] = ztest(y_pred, s_test[attr], "White", "Black")
        stat_rows.append(row)
pd.DataFrame(stat_rows).to_csv(os.path.join(RES, "significance.csv"), index=False)
print("significance tests done")

# 7. mitigation for sex on gradient boosting and logistic regression (starting from the "without" model)
def reweigh_weights(groups, labels):
    # Kamiran & Calders: w(g, y) = P(g) * P(y) / P(g, y)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    w = np.ones(len(labels))
    for g in np.unique(groups):
        for l in np.unique(labels):
            m = (groups == g) & (labels == l)
            w[m] = (groups == g).mean() * (labels == l).mean() / m.mean()
    return w

def group_thresholds(score, y, groups):
    # one threshold per group so that each group gets the same TPR as the whole validation set has at 0.5
    score = np.asarray(score)
    y = np.asarray(y)
    groups = np.asarray(groups)
    target_tpr = ((score >= 0.5) & (y == 1)).sum() / (y == 1).sum()
    th = {}
    for g in np.unique(groups):
        pos_scores = score[(groups == g) & (y == 1)]
        th[g] = float(np.quantile(pos_scores, 1 - target_tpr))
    return th, float(target_tpr)

def apply_thresholds(score, groups, th):
    return np.array([int(s >= th[g]) for s, g in zip(score, groups)])

mit_rows = []
mit_tables = []
def record(name, setting, y_pred, score):
    row, tables = evaluate(name, setting, y_test, y_pred, s_test[["sex", "race"]], score, 500)
    mit_rows.append(row)
    for attr in tables:
        mit_tables.append(tables[attr].reset_index().assign(model=name, setting=setting, attr=attr))

w = reweigh_weights(s_train["sex"], y_train)
for name in ["HistGB", "LogReg"]:
    base = fitted[(name, "without")]
    base_score = scores[(name, "without")]
    record(name, "with sex/race", preds[(name, "with")], scores[(name, "with")])
    record(name, "without sex/race", preds[(name, "without")], base_score)

    # reweighing: same model and hyperparameters, trained with sample weights
    est = clone(MODELS[name]).set_params(**best_params[name + "_without"])
    rw = Pipeline([("pre", make_preprocessor(X_train, ("sex", "race"))), ("clf", est)])
    rw.fit(X_train, y_train, clf__sample_weight=w)
    rw_score = rw.predict_proba(X_test)[:, 1]
    record(name, "+ reweighing", rw.predict(X_test), rw_score)
    if name == "HistGB":
        rw_hgb = rw

    # group thresholds, picked on the validation set
    th, target = group_thresholds(base.predict_proba(X_val)[:, 1], y_val, s_val["sex"])
    record(name, "+ group thresholds", apply_thresholds(base_score, s_test["sex"], th), base_score)
    summary["thresholds_" + name] = {"thresholds": th, "target_tpr": target}

    # both together
    th2, _ = group_thresholds(rw.predict_proba(X_val)[:, 1], y_val, s_val["sex"])
    record(name, "+ reweighing + thresholds", apply_thresholds(rw_score, s_test["sex"], th2), rw_score)

    weights = {}
    for g in ["Male", "Female"]:
        for l in [0, 1]:
            weights[g + "_" + str(l)] = float(w[(s_train["sex"].values == g) & (y_train.values == l)][0])
    summary["reweigh_weights_" + name] = weights

pd.DataFrame(mit_rows).to_csv(os.path.join(RES, "mitigation.csv"), index=False)
pd.concat(mit_tables).to_csv(os.path.join(RES, "mitigation_per_group.csv"), index=False)

# confusion matrix of the best model (by F1)
best = res_df.sort_values("f1", ascending=False).iloc[0]
cm = confusion_matrix(y_test, preds[(best["model"], best["variant"])])
summary["best_model"] = {"model": best["model"], "variant": best["variant"], "cm": cm.tolist()}
print("best model:", best["model"], best["variant"], cm.tolist())

# save the gradient boosting models for the demo
cat_cols = X_train.select_dtypes(exclude="number").columns
demo_models = {
    "with": fitted[("HistGB", "with")],
    "without": fitted[("HistGB", "without")],
    "reweighed": rw_hgb,
    "columns": X_train.columns.tolist(),
    "example": X_test.iloc[:200].reset_index(drop=True),
    "example_y": y_test.iloc[:200].reset_index(drop=True),
    "choices": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
}
os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
joblib.dump(demo_models, os.path.join(ROOT, "demo", "models.joblib"))

with open(os.path.join(RES, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("done, results are in", RES)
