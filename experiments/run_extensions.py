# Part 2: age groups, PPV gap, sex x race groups, and three more mitigation methods
# (disparate impact remover, equalized odds post-processing, exponentiated gradient)
# run after run_experiments.py: .venv/bin/python experiments/run_extensions.py
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

from common import (RANDOM_STATE, FIG, RES, MODELS, load_data, split_data, make_preprocessor,
                    get_scores, group_metrics, gaps, bootstrap_gaps, chi2_test, evaluate)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6, "figure.dpi": 150, "savefig.bbox": "tight"})
ext = {}

# same data and same split as part 1
df, n_dup = load_data()
X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test = split_data(df)
sex_race = s_test["sex"] + "/" + s_test["race"]
ext["age_group_counts_test"] = s_test["age_group"].value_counts().to_dict()
ext["age_income_rate"] = df.groupby("age_group")["y"].mean().round(4).to_dict()
ext["intersectional_counts_test"] = sex_race.value_counts().to_dict()
print(ext["age_group_counts_test"])

# A. train the 8 models again with the hyperparameters found in part 1, audit by sex, race and age group
with open(os.path.join(RES, "summary.json")) as f:
    best_params = json.load(f)["best_params"]

rows = []
per_group = []
preds, scores, fitted = {}, {}, {}
for name in MODELS:
    for variant in ["with", "without"]:
        drop = () if variant == "with" else ("sex", "race")
        est = clone(MODELS[name]).set_params(**best_params[name + "_" + variant])
        model = Pipeline([("pre", make_preprocessor(X_train, drop)), ("clf", est)]).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted[(name, variant)] = model
        preds[(name, variant)] = y_pred
        scores[(name, variant)] = get_scores(model, X_test)

        row = {"model": name, "variant": variant, "accuracy": accuracy_score(y_test, y_pred)}
        for attr in ["sex", "race", "age_group"]:
            t = group_metrics(y_test, y_pred, s_test[attr])
            for k, v in gaps(t).items():
                row[attr + "_" + k] = v
            chi2, p, dof = chi2_test(y_pred, s_test[attr])
            row[attr + "_chi2"] = chi2
            row[attr + "_chi2_p"] = p
            t = t.reset_index()
            t.insert(0, "attr", attr)
            t.insert(0, "variant", variant)
            t.insert(0, "model", name)
            per_group.append(t)
        ci = bootstrap_gaps(y_test, y_pred, s_test["age_group"], 300)
        row["age_DP_lo"], row["age_DP_hi"] = ci["DP"]
        row["age_EO_lo"], row["age_EO_hi"] = ci["EO"]
        rows.append(row)
        print(name, variant, "acc", round(row["accuracy"], 4), "age DP", round(row["age_group_DP"], 3),
              "age EO", round(row["age_group_EO"], 3), "sex PPV gap", round(row["sex_PPV"], 3))

ext_df = pd.DataFrame(rows)
ext_df.to_csv(os.path.join(RES, "ext_model_results.csv"), index=False)
pd.concat(per_group).to_csv(os.path.join(RES, "ext_per_group_metrics.csv"), index=False)

# check that the retrained models give the same accuracy as in part 1
part1 = pd.read_csv(os.path.join(RES, "model_results.csv"))
assert np.allclose(ext_df["accuracy"], part1["accuracy"]), "models do not match part 1"

# B. sex x race groups for gradient boosting
inter = []
for variant in ["with", "without"]:
    t = group_metrics(y_test, preds[("HistGB", variant)], sex_race).reset_index()
    t.insert(0, "variant", variant)
    inter.append(t)
inter = pd.concat(inter)
inter.to_csv(os.path.join(RES, "ext_intersectional.csv"), index=False)

# only groups with at least 100 test rows
big = inter[(inter["variant"] == "with") & (inter["n"] >= 100)]
ext["intersectional_with_n>=100"] = {
    "DP": float(big["selection_rate"].max() - big["selection_rate"].min()),
    "EO": float(big["TPR"].max() - big["TPR"].min()),
    "max_group": big.loc[big["selection_rate"].idxmax(), "group"],
    "min_group": big.loc[big["selection_rate"].idxmin(), "group"],
}
print("intersectional", ext["intersectional_with_n>=100"])

# C. more mitigation methods (for sex)
class DisparateImpactRemover:
    # Feldman et al. 2015, numeric columns only.
    # A value at quantile q inside its own group is replaced by the value at quantile q
    # of the "median distribution" (median of the group quantiles).
    def __init__(self, cols, repair_level=1.0):
        self.cols = cols
        self.level = repair_level
        self.q = np.linspace(0, 1, 101)

    def fit(self, X, groups):
        groups = np.asarray(groups)
        self.groups = np.unique(groups)
        self.group_q = {}
        self.median_q = {}
        for c in self.cols:
            v = X[c].astype(float).values
            self.group_q[c] = {g: np.nanquantile(v[groups == g], self.q) for g in self.groups}
            self.median_q[c] = np.median(np.vstack(list(self.group_q[c].values())), axis=0)
        return self

    def transform(self, X, groups):
        X = X.copy()
        groups = np.asarray(groups)
        for c in self.cols:
            v = X[c].astype(float).values
            out = v.copy()
            for g in self.groups:
                m = (groups == g) & ~np.isnan(v)
                if not m.any():
                    continue
                rank = np.searchsorted(self.group_q[c][g], v[m], side="right") / len(self.q)
                rank = np.clip(rank, 0, 1)
                repaired = np.interp(rank, self.q, self.median_q[c])
                out[m] = (1 - self.level) * v[m] + self.level * repaired
            X[c] = out
        return X

mit_rows = []
mit_tables = []
def record(name, setting, y_pred, score):
    row, tables = evaluate(name, setting, y_test, y_pred, s_test, score, 300)
    mit_rows.append(row)
    for attr in tables:
        mit_tables.append(tables[attr].reset_index().assign(model=name, setting=setting, attr=attr))

num_cols = X_train.select_dtypes(include="number").columns.tolist()
for name in ["HistGB", "LogReg"]:
    base = fitted[(name, "without")]
    base_est = base.named_steps["clf"]
    record(name, "without sex/race", preds[(name, "without")], scores[(name, "without")])

    # disparate impact remover: repair the numeric columns, then train again
    dir_ = DisparateImpactRemover(num_cols).fit(X_train, s_train["sex"])
    X_train_rep = dir_.transform(X_train, s_train["sex"])
    X_test_rep = dir_.transform(X_test, s_test["sex"])
    model = Pipeline([("pre", make_preprocessor(X_train, ("sex", "race"))), ("clf", clone(base_est))])
    model.fit(X_train_rep, y_train)
    record(name, "+ DI remover (lambda=1)", model.predict(X_test_rep), model.predict_proba(X_test_rep)[:, 1])

    # equalized odds post-processing (Hardt et al.), fitted on the validation set
    to = ThresholdOptimizer(estimator=base, constraints="equalized_odds", objective="accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(X_val, y_val, sensitive_features=s_val["sex"])
    y_pred = to.predict(X_test, sensitive_features=s_test["sex"], random_state=RANDOM_STATE)
    record(name, "+ EO post-processing (Hardt)", y_pred, scores[(name, "without")])

    # in-processing: exponentiated gradient (Agarwal et al.) with a DP constraint and with an EO constraint
    pre = make_preprocessor(X_train, ("sex", "race")).fit(X_train)
    Ztr = pre.transform(X_train)
    Zte = pre.transform(X_test)
    constraints = {"DP": DemographicParity(difference_bound=0.01), "EO": EqualizedOdds(difference_bound=0.01)}
    for cname in constraints:
        max_iter = 50 if name == "LogReg" else 25
        eg = ExponentiatedGradient(clone(base_est), constraints=constraints[cname], eps=0.01, max_iter=max_iter)
        eg.fit(Ztr, y_train, sensitive_features=s_train["sex"])
        y_pred = np.asarray(eg.predict(Zte, random_state=RANDOM_STATE))
        score = eg._pmf_predict(Zte)[:, 1]
        record(name, "+ in-processing EG (" + cname + ")", y_pred, score)
        print("   classifiers in the ensemble:", len(eg.predictors_))

mit_df = pd.DataFrame(mit_rows)
mit_df.to_csv(os.path.join(RES, "ext_mitigation.csv"), index=False)
pd.concat(mit_tables).to_csv(os.path.join(RES, "ext_mitigation_per_group.csv"), index=False)

# figure: gradient boosting by age group
t = group_metrics(y_test, preds[("HistGB", "with")], s_test["age_group"]).loc[["<30", "30-44", "45-59", "60+"]]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
t[["base_rate", "selection_rate", "TPR", "FPR"]].plot.bar(
    ax=ax, rot=0, color=["#6b7280", "#2563eb", "#16a34a", "#ea580c"], width=0.8)
ax.set_ylabel("rate")
ax.set_xlabel("age group")
ax.legend(["true >50K rate", "selection rate", "TPR", "FPR"], ncol=2)
ax.grid(axis="y", alpha=.3)
ax.set_title("Gradient boosting by age group (test set)")
fig.savefig(os.path.join(FIG, "age_groups.pdf"))
plt.close(fig)

# figure: accuracy vs gap for every mitigation setting (part 1 and part 2 together)
old = pd.read_csv(os.path.join(RES, "mitigation.csv"))
allm = pd.concat([old[old["setting"] != "without sex/race"], mit_df]).reset_index(drop=True)
colors = {"with sex/race": "#6b7280", "without sex/race": "#2563eb", "+ reweighing": "#dc2626",
          "+ group thresholds": "#16a34a", "+ reweighing + thresholds": "#7c3aed",
          "+ DI remover (lambda=1)": "#92400e", "+ EO post-processing (Hardt)": "#0d9488",
          "+ in-processing EG (DP)": "#be185d", "+ in-processing EG (EO)": "#eab308"}
markers = {"HistGB": "D", "LogReg": "o"}
fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
for ax, col, label in zip(axes, ["sex_DP", "sex_EO"], ["DP gap (sex)", "EO gap (sex)"]):
    for _, r in allm.iterrows():
        ax.scatter(r[col], r["accuracy"], marker=markers[r["model"]], s=24, color=colors[r["setting"]],
                   edgecolor="k", lw=0.3)
    ax.set_xlabel(label)
    ax.grid(alpha=.3)
    ax.tick_params(labelsize=6)
axes[0].set_ylabel("test accuracy")
handles = [Line2D([], [], marker="D", color="k", ls="", ms=4, label="Grad. boosting"),
           Line2D([], [], marker="o", color="k", ls="", ms=4, label="LogReg")]
for setting in colors:
    label = setting.replace("(lambda=1)", "").replace("in-processing ", "")
    handles.append(Line2D([], [], marker="o", color=colors[setting], ls="", ms=5, label=label))
fig.tight_layout()
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.2, frameon=False, bbox_to_anchor=(0.5, -0.27))
fig.savefig(os.path.join(FIG, "tradeoff_ext.pdf"))
plt.close(fig)

with open(os.path.join(RES, "summary_ext.json"), "w") as f:
    json.dump(ext, f, indent=2, default=str)
print("done")
