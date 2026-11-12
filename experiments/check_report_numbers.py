# Checks that the numbers typed in report/main.tex are the ones in experiments/results.
# run: .venv/bin/python experiments/check_report_numbers.py
import os
import re
import sys
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")

with open(os.path.join(ROOT, "report", "main.tex")) as f:
    tex = re.sub(r"\s+", " ", f.read())
with open(os.path.join(RES, "summary.json")) as f:
    S = json.load(f)
with open(os.path.join(RES, "summary_ext.json")) as f:
    E = json.load(f)
res = pd.read_csv(os.path.join(RES, "model_results.csv"))
mit = pd.read_csv(os.path.join(RES, "mitigation.csv"))
ext = pd.read_csv(os.path.join(RES, "ext_mitigation.csv"))
extm = pd.read_csv(os.path.join(RES, "ext_model_results.csv")).set_index(["model", "variant"])
pg = pd.read_csv(os.path.join(RES, "per_group_metrics.csv"))

total = 0
bad = 0
def expect(label, value, fmt="{:.3f}"):
    # the value has to appear somewhere in the tex. the tables write 0.123 as .123 in some columns
    global total, bad
    s = value if isinstance(value, str) else fmt.format(value)
    found = s in tex
    if s.startswith("0."):
        found = found or (" " + s[1:]) in tex or ("/ " + s[1:]) in tex
    total += 1
    if not found:
        bad += 1
        print("MISMATCH", label, "expected", s)

# data and split
expect("clean rows", "48{,}790")
expect("duplicates", "52 exact duplicate")
expect("train size", "29{,}273")
expect("val size", "7{,}319")
expect("test size", "12{,}198")
expect("pos rate", S["data"]["pos_rate"] * 100, "{:.1f}\\%")

# main table
for _, r in res.iterrows():
    for col in ["accuracy", "f1", "auc", "sex_DP", "sex_EO", "race_DP", "race_EO"]:
        expect(r["model"] + " " + r["variant"] + " " + col, r[col])

# loss curves
sgd = S["sgd_curve"]
hgb = S["hgb_curve"]
expect("sgd train first", sgd["train_loss_first"])
expect("sgd train last", sgd["train_loss_last"])
expect("sgd val min", sgd["val_loss_min"])
expect("sgd best epoch", "epoch " + str(sgd["val_loss_argmin"]))
expect("hgb val min", hgb["val_min"])
expect("hgb best iteration", "iteration " + str(hgb["val_argmin"]))
expect("hgb train last", hgb["train_last"])
expect("hgb val last", hgb["val_last"])

# gradient boosting (with) by sex
g = pg[(pg["model"] == "HistGB") & (pg["variant"] == "with") & (pg["attr"] == "sex")].set_index("group")
for sex in ["Male", "Female"]:
    expect(sex + " selection rate", g.loc[sex, "selection_rate"] * 100, "{:.1f}\\%")
    expect(sex + " TPR", g.loc[sex, "TPR"] * 100, "{:.1f}\\%")

# thresholds and reweighing weights
th = S["thresholds_HistGB"]["thresholds"]
w = S["reweigh_weights_HistGB"]
expect("threshold male", th["Male"])
expect("threshold female", th["Female"])
expect("weight female >50K", w["Female_1"], "{:.2f}")
expect("weight male >50K", w["Male_1"], "{:.2f}")

# mitigation table
for _, r in mit.iterrows():
    for col in ["accuracy", "sex_DP", "sex_EO"]:
        expect(r["model"] + " " + r["setting"] + " " + col, r[col])
for _, r in ext.iterrows():
    if r["setting"] == "without sex/race":
        continue
    for col in ["accuracy", "sex_DP", "sex_EO", "sex_PPV"]:
        expect(r["model"] + " " + r["setting"] + " " + col, r[col])

# age groups and sex x race
expect("age DP", extm.loc[("HistGB", "with"), "age_group_DP"])
expect("age EO", extm.loc[("HistGB", "with"), "age_group_EO"])
expect("intersectional DP", E["intersectional_with_n>=100"]["DP"])

print(total - bad, "of", total, "checks OK")
sys.exit(1 if bad else 0)
