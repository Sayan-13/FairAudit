# Live demo. Run: .venv/bin/streamlit run demo/app.py
# All tables come from experiments/results, tab 5 uses the models saved by run_experiments.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
MODELS = os.path.join(ROOT, "demo", "models.joblib")

st.set_page_config(page_title="Fairness Audit: Adult Income", layout="wide")
st.title("Auditing Fairness in Classical ML Income Prediction")
st.caption("UCI Adult | LogReg, Linear SVM, Random Forest, Gradient Boosting | seed 42 | all numbers from experiments/results")


@st.cache_data
def load():
    names = ["model_results", "significance", "mitigation", "mitigation_per_group", "sgd_loss_curve",
             "hgb_loss_curve", "sample_rows", "ext_per_group_metrics", "ext_model_results",
             "ext_mitigation", "ext_intersectional"]
    d = {}
    for name in names:
        d[name] = pd.read_csv(os.path.join(RES, name + ".csv"))
    with open(os.path.join(RES, "summary.json")) as f:
        d["summary"] = json.load(f)
    return d


@st.cache_resource
def get_models():
    return joblib.load(MODELS)


D = load()
S = D["summary"]
tabs = st.tabs(["1. Dataset", "2. Models & loss curves", "3. Fairness audit", "4. Mitigation", "5. Live prediction"])

# tab 1: dataset
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records (clean)", f"{S['data']['clean_rows']:,}", f"-{S['data']['duplicates_dropped']} duplicates")
    c2.metric("Share earning >50K", f"{S['data']['pos_rate']*100:.1f}%")
    c3.metric(">50K rate: men", f"{S['data']['income_rate_by_sex']['Male']*100:.1f}%")
    c4.metric(">50K rate: women", f"{S['data']['income_rate_by_sex']['Female']*100:.1f}%")

    st.subheader("Representative rows")
    st.dataframe(D["sample_rows"], width='stretch')

    st.subheader("Split (stratified, seed 42)")
    sp = S["split"]
    st.table(pd.DataFrame({
        "split": ["train", "validation", "test"],
        "records": [sp["train"], sp["val"], sp["test"]],
        "share": ["60%", "15%", "25%"],
        "positives": [f"{sp['pos_rate'][k]*100:.1f}%" for k in ["train", "val", "test"]],
    }))

    st.subheader("Income rate by group (the bias is already in the labels)")
    a, b = st.columns(2)
    a.bar_chart(pd.Series(S["data"]["income_rate_by_sex"]), height=250)
    b.bar_chart(pd.Series(S["data"]["income_rate_by_race"]), height=250)

# tab 2: models and loss curves
with tabs[1]:
    cols = ["model", "variant", "cv_f1", "val_accuracy", "accuracy", "precision", "recall", "f1", "auc", "fit_time_s"]
    show = D["model_results"][cols].round(4)
    st.subheader("Test-set performance (tuned by 3-fold CV on the training split)")
    st.dataframe(show.style.highlight_max(subset=["accuracy", "f1", "auc"], color="#bbf7d0"),
                 width='stretch', hide_index=True)

    st.subheader("Training and validation loss")
    a, b = st.columns(2)
    a.markdown("**Logistic regression by SGD (log loss per epoch)**")
    a.line_chart(D["sgd_loss_curve"].set_index("epoch")[["train_loss", "val_loss"]], height=280)
    b.markdown("**Gradient boosting (log loss per boosting iteration)**")
    b.line_chart(D["hgb_loss_curve"].set_index("iter")[["train_loss", "val_loss"]], height=280)

    sgd = S["sgd_curve"]
    hgb = S["hgb_curve"]
    st.info(f"SGD: train {sgd['train_loss_first']:.3f} -> {sgd['train_loss_last']:.3f}, "
            f"val min {sgd['val_loss_min']:.3f} at epoch {sgd['val_loss_argmin']} (no overfitting). "
            f"HGB: val loss minimum {hgb['val_min']:.3f} at iteration {hgb['val_argmin']}, "
            f"then rises to {hgb['val_last']:.3f} while train keeps falling to {hgb['train_last']:.3f} "
            f"(mild overfitting beyond ~120 iterations).")

# tab 3: fairness audit
with tabs[2]:
    pg = D["ext_per_group_metrics"]
    sig = D["significance"]
    ex = D["ext_model_results"]
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", pg["model"].unique(), index=3)
    variant = c2.selectbox("Sensitive attributes as inputs", ["with", "without"])
    attr = c3.selectbox("Audit by", ["sex", "race", "age_group"])

    t = pg[(pg["model"] == model) & (pg["variant"] == variant) & (pg["attr"] == attr)].set_index("group")
    t = t[["n", "base_rate", "selection_rate", "TPR", "FPR", "FNR", "PPV", "accuracy"]]
    if attr == "age_group":
        t = t.loc[["<30", "30-44", "45-59", "60+"]]
    st.dataframe(t.round(3), width='stretch')

    gap_values = {
        "Demographic parity gap": t["selection_rate"].max() - t["selection_rate"].min(),
        "Equal opportunity gap": t["TPR"].max() - t["TPR"].min(),
        "FPR gap": t["FPR"].max() - t["FPR"].min(),
        "Predictive parity (PPV) gap": t["PPV"].max() - t["PPV"].min(),
    }

    # confidence intervals and tests (age group only has DP and EO intervals)
    if attr == "age_group":
        r = ex[(ex["model"] == model) & (ex["variant"] == variant)].iloc[0]
        ci = [(r["age_DP_lo"], r["age_DP_hi"]), (r["age_EO_lo"], r["age_EO_hi"]), None, None]
        st.markdown(f"**Significance**: chi-square test of independence between age group and prediction: "
                    f"chi2 = {r['age_group_chi2']:.1f} (3 d.o.f.), p = {r['age_group_chi2_p']:.1e}.")
    else:
        r = sig[(sig["model"] == model) & (sig["variant"] == variant) & (sig["attr"] == attr)].iloc[0]
        ci = [(r["DP_lo"], r["DP_hi"]), (r["EO_lo"], r["EO_hi"]), (r["FPR_lo"], r["FPR_hi"]), None]
        pair = "Male vs Female" if attr == "sex" else "White vs Black"
        st.markdown(f"**Significance**: two-proportion z-test on selection rates ({pair}): "
                    f"z = {r['z']:.1f}, p = {r['z_p']:.1e}; chi-square test of independence: "
                    f"chi2 = {r['chi2']:.1f} ({int(r['dof'])} d.o.f.), p = {r['chi2_p']:.1e}.")

    cols = st.columns(4)
    for i, name in enumerate(gap_values):
        text = ""
        if ci[i] is not None:
            text = f"95% CI [{ci[i][0]:.3f}, {ci[i][1]:.3f}]"
        cols[i].metric(name, f"{gap_values[name]:.3f}", text, delta_color="off")

    with st.expander("Intersectional sex x race audit (gradient boosting)"):
        it = D["ext_intersectional"]
        iv = st.radio("variant", ["with", "without"], horizontal=True, key="iv")
        it = it[it["variant"] == iv].sort_values("n", ascending=False).set_index("group")
        st.dataframe(it[["n", "base_rate", "selection_rate", "TPR", "FPR", "PPV", "accuracy"]].round(3), width='stretch')

    fig, ax = plt.subplots(figsize=(7, 2.8))
    t[["selection_rate", "TPR", "FPR"]].plot.bar(ax=ax, rot=0, color=["#2563eb", "#16a34a", "#ea580c"])
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=.3)
    st.pyplot(fig)
    st.caption("Note how per-group accuracy is *higher* for the disadvantaged group (fewer positives) even though "
               "its selection rate and TPR are lower: accuracy alone hides the bias.")

# tab 4: mitigation
with tabs[3]:
    mit = D["mitigation"]
    mpg = D["mitigation_per_group"]
    model = st.radio("Model", ["HistGB", "LogReg"], horizontal=True)
    cols = ["setting", "accuracy", "f1", "sex_DP", "sex_DP_lo", "sex_DP_hi", "sex_EO", "sex_EO_lo", "sex_EO_hi",
            "sex_FPR", "race_DP", "race_EO"]
    st.dataframe(mit[mit["model"] == model][cols].round(3), width='stretch', hide_index=True)

    # selection rate and TPR of men and women for each setting
    settings = ["with sex/race", "without sex/race", "+ reweighing", "+ group thresholds", "+ reweighing + thresholds"]
    sub = mpg[(mpg["model"] == model) & (mpg["attr"] == "sex")]
    xs = np.arange(len(settings))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, metric in zip(axes, ["selection_rate", "TPR"]):
        male = [sub[(sub["setting"] == s) & (sub["group"] == "Male")][metric].item() for s in settings]
        female = [sub[(sub["setting"] == s) & (sub["group"] == "Female")][metric].item() for s in settings]
        ax.bar(xs - .18, male, .36, label="Male", color="#2563eb")
        ax.bar(xs + .18, female, .36, label="Female", color="#ea580c")
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in settings], fontsize=7)
        ax.set_title(metric)
        ax.legend()
        ax.grid(axis="y", alpha=.3)
    st.pyplot(fig)

    with st.expander("More methods: disparate impact remover, equalized-odds post-processing (Hardt), "
                     "in-processing (exponentiated gradient)", expanded=True):
        em = D["ext_mitigation"]
        cols = ["setting", "accuracy", "f1", "sex_DP", "sex_DP_lo", "sex_DP_hi", "sex_EO", "sex_EO_lo", "sex_EO_hi",
                "sex_FPR", "sex_PPV", "race_DP", "age_group_DP"]
        st.dataframe(em[em["model"] == model][cols].round(3), width='stretch', hide_index=True)
        st.caption("No method improves every gap: equalizing TPR/FPR or selection rates (EG, Hardt) blows up the "
                   "predictive-parity (PPV) gap, which is the impossibility theorem in practice.")

    th = S["thresholds_" + model]
    weights = []
    for k, v in S["reweigh_weights_HistGB"].items():
        weights.append(k.replace("_1", " >50K").replace("_0", " <=50K") + f"={v:.2f}")
    st.markdown(f"**Group thresholds** (chosen on validation, target TPR {th['target_tpr']:.3f}): "
                f"Male {th['thresholds']['Male']:.3f}, Female {th['thresholds']['Female']:.3f}. "
                f"**Reweighing weights** (train): " + ", ".join(weights))
    st.success("Reweighing targets demographic parity (selection rates move together); group thresholds target "
               "equal opportunity (TPRs become equal). Post-processing is the cheaper one in accuracy.")

# tab 5: live prediction
with tabs[4]:
    if not os.path.exists(MODELS):
        st.error("demo/models.joblib not found. Run experiments/run_experiments.py first.")
        st.stop()
    M = get_models()
    st.markdown("""**How this tab works.** Below is one real person from the test set. Change any field (age, education, hours, sex, ...) and the
**Result** panel underneath updates immediately -- there is no Run button. The three gradient boosting models are the ones from the report.
Tip: use a *borderline* person (probability near 0.5): the default record is chosen for that. Records with a Doctorate or a large capital gain are
almost certainly >50K for every model, so changing other fields will not move them.""")

    # start with the record whose probability is closest to 0.5
    if "default_idx" not in st.session_state:
        p = M["with"].predict_proba(M["example"])[:, 1]
        st.session_state["default_idx"] = int(np.argmin(np.abs(p - 0.5)))
    idx = int(st.number_input("Test record index (0-199)", 0, len(M["example"]) - 1, st.session_state["default_idx"]))
    row = M["example"].iloc[idx]

    st.markdown("#### Inputs (edit freely)")
    c = st.columns(4)
    edited = {}
    for i, col in enumerate(M["columns"]):
        box = c[i % 4]
        if col in M["choices"]:
            opts = M["choices"][col]
            cur = row[col] if row[col] in opts else opts[0]
            edited[col] = box.selectbox(col, opts, index=opts.index(cur), key=f"f_{col}_{idx}")
        else:
            edited[col] = box.number_input(col, value=int(round(float(row[col]))), step=1, key=f"f_{col}_{idx}")
    true_label = ">50K" if M["example_y"].iloc[idx] == 1 else "<=50K"
    st.caption(f"True label of this record in the census: **{true_label}**")

    # same person with the other sex
    sex = edited["sex"]
    flip = "Female" if sex == "Male" else "Male"
    x = pd.DataFrame([edited])
    x_flip = x.copy()
    x_flip["sex"] = flip
    th = S["thresholds_HistGB"]["thresholds"]

    st.markdown("---\n#### Result")
    p_with = M["with"].predict_proba(x)[0, 1]
    p_with_flip = M["with"].predict_proba(x_flip)[0, 1]
    p_without = M["without"].predict_proba(x)[0, 1]
    p_rw = M["reweighed"].predict_proba(x)[0, 1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"P(>50K) as {sex} (model WITH sex/race)", f"{p_with:.1%}")
    k2.metric(f"same person as {flip}", f"{p_with_flip:.1%}", f"{(p_with_flip - p_with)*100:+.1f} pts", delta_color="off")
    k3.metric("model WITHOUT sex/race", f"{p_without:.1%}", "identical for both sexes", delta_color="off")
    k4.metric("reweighed model (no sex/race)", f"{p_rw:.1%}")

    rows = []
    for name, key in [("with sex/race", "with"), ("without sex/race", "without"), ("reweighed (no sex/race)", "reweighed")]:
        p = M[key].predict_proba(x)[0, 1]
        p_flip = M[key].predict_proba(x_flip)[0, 1]
        rows.append({
            "model": name,
            f"P(>50K) as {sex}": f"{p:.3f}",
            f"P(>50K) as {flip}": f"{p_flip:.3f}",
            "decision @ 0.5": ">50K" if p >= 0.5 else "<=50K",
            f"decision @ group threshold ({sex}: {th[sex]:.3f})": ">50K" if p >= th[sex] else "<=50K",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption("Models trained without sex/race give identical scores for both sexes by construction, yet their "
               "*group-level* gap is still ~0.17 because other features (relationship, marital status, occupation, "
               "hours) act as proxies. The group threshold is the post-processing mitigation from the report: "
               "women are approved from a lower score (0.412) than men (0.513).")
