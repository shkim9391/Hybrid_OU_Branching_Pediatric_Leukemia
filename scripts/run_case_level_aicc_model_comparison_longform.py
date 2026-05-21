import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

IN_XLSX = "/Revision/Ahlgren_2025_Supplementary_Data_1-18.xlsx"
OUT_DIR = "/Revision"

# Set to True to update/replace the S8 sheet in the supplementary workbook.
UPDATE_SUPPLEMENTARY_WORKBOOK = True
SUPPLEMENTARY_XLSX = os.path.join(OUT_DIR, "Supplementary_Data.xlsx")

EPS = 1e-5
SOURCE_SCRIPT = Path(__file__).name
MODEL_ORDER = ["Brownian", "Branching-only drift", "OU", "Markov-emission", "OU-Branching jump"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def normal_logpdf(x, mean, var):
    var = np.maximum(var, 1e-10)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def aicc_from_ll(ll, k, n):
    aic = 2 * k - 2 * ll
    return aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.nan


def bic_from_ll(ll, k, n):
    return k * np.log(n) - 2 * ll if n > 0 else np.nan


def fit_brownian(t, y):
    dt, dy = np.diff(t), np.diff(y)

    def nll(par):
        sigma = np.exp(par[0])
        return -np.sum(normal_logpdf(dy, 0.0, sigma**2 * dt))

    init_sigma = np.sqrt(np.mean((dy**2) / np.maximum(dt, 1e-6)))
    res = minimize(nll, [np.log(max(init_sigma, 1e-4))], bounds=[(-10, 5)], method="L-BFGS-B")
    return {
        "model": "Brownian",
        "ll": -res.fun,
        "k": 1,
        "n": len(dy),
        "sigma": float(np.exp(res.x[0])),
        "success": bool(res.success),
    }


def fit_branching_drift(t, y):
    dt, dy = np.diff(t), np.diff(y)

    def nll(par):
        beta, sigma = par[0], np.exp(par[1])
        return -np.sum(normal_logpdf(dy, beta * dt, sigma**2 * dt))

    beta0 = np.sum(dy) / np.sum(dt)
    sigma0 = np.sqrt(np.mean(((dy - beta0 * dt) ** 2) / np.maximum(dt, 1e-6)))
    res = minimize(
        nll,
        [beta0, np.log(max(sigma0, 1e-4))],
        bounds=[(-50, 50), (-10, 5)],
        method="L-BFGS-B",
    )
    return {
        "model": "Branching-only drift",
        "ll": -res.fun,
        "k": 2,
        "n": len(dy),
        "beta": float(res.x[0]),
        "sigma": float(np.exp(res.x[1])),
        "success": bool(res.success),
    }


def fit_ou(t, y):
    dt, yprev, ynext = np.diff(t), y[:-1], y[1:]
    ymean, ystd = float(np.mean(y)), max(float(np.std(y)), 1e-3)
    bounds = [(-6, 6), (float(np.min(y) - 5), float(np.max(y) + 5)), (-10, 5)]

    def nll(par):
        theta, mu, sigma = np.exp(par[0]), par[1], np.exp(par[2])
        e = np.exp(-theta * dt)
        mean = mu + (yprev - mu) * e
        var = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        return -np.sum(normal_logpdf(ynext, mean, var))

    starts = [[np.log(t0), ymean, np.log(ystd)] for t0 in [0.1, 1.0, 5.0]]
    best = None
    for st in starts:
        res = minimize(nll, st, bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    res = best
    return {
        "model": "OU",
        "ll": -res.fun,
        "k": 3,
        "n": len(dt),
        "theta": float(np.exp(res.x[0])),
        "mu": float(res.x[1]),
        "sigma": float(np.exp(res.x[2])),
        "success": bool(res.success),
    }


def fit_markov_emission(t, y, q1, q2):
    yprev, ynext = y[:-1], y[1:]
    prev_state = np.digitize(yprev, [q1, q2])
    states = sorted(set(prev_state))
    init_means = [float(np.mean(ynext[prev_state == s])) for s in states]
    resid = ynext - np.array([init_means[states.index(s)] for s in prev_state])
    init_sigma = max(float(np.std(resid)), 1e-3)
    bounds = [(float(np.min(y) - 5), float(np.max(y) + 5))] * len(states) + [(-10, 5)]

    def nll(par):
        means = dict(zip(states, par[:-1]))
        sigma = np.exp(par[-1])
        pred = np.array([means[s] for s in prev_state])
        return -np.sum(normal_logpdf(ynext, pred, sigma**2))

    res = minimize(nll, init_means + [np.log(init_sigma)], bounds=bounds, method="L-BFGS-B")
    return {
        "model": "Markov-emission",
        "ll": -res.fun,
        "k": len(states) + 1,
        "n": len(ynext),
        "n_states": len(states),
        "sigma": float(np.exp(res.x[-1])),
        "success": bool(res.success),
    }


def fit_ou_branching_jump(t, y):
    dt, yprev, ynext = np.diff(t), y[:-1], y[1:]
    ymean, ystd = float(np.mean(y)), max(float(np.std(y)), 1e-3)
    bounds = [(-6, 6), (float(np.min(y) - 5), float(np.max(y) + 5)), (-10, 5), (-6, 6), (-10, 5)]

    def nll(par):
        theta, mu, sigma = np.exp(par[0]), par[1], np.exp(par[2])
        p = 1 / (1 + np.exp(-par[3]))
        tau = np.exp(par[4])
        e = np.exp(-theta * dt)
        mean = mu + (yprev - mu) * e
        var_base = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        ll0 = np.log(1 - p) + normal_logpdf(ynext, mean, var_base)
        ll1 = np.log(p) + normal_logpdf(ynext, mean, var_base + tau**2)
        return -np.sum(logsumexp(np.vstack([ll0, ll1]), axis=0))

    starts = []
    for theta0 in [0.1, 1.0, 5.0]:
        for p0 in [0.1, 0.25, 0.5]:
            starts.append([
                np.log(theta0),
                ymean,
                np.log(max(ystd / 2, 1e-3)),
                np.log(p0 / (1 - p0)),
                np.log(ystd),
            ])
    best = None
    for st in starts:
        res = minimize(nll, st, bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    res = best
    theta, mu, sigma = np.exp(res.x[0]), res.x[1], np.exp(res.x[2])
    p, tau = 1 / (1 + np.exp(-res.x[3])), np.exp(res.x[4])
    return {
        "model": "OU-Branching jump",
        "ll": -res.fun,
        "k": 5,
        "n": len(dt),
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "branch_prob": float(p),
        "jump_sd": float(tau),
        "success": bool(res.success),
    }


def safe_clinical_metadata(clin):
    """Return one clinical metadata row per patient, preserving expected manuscript fields."""
    clin = clin.rename(columns={"Patient_ID": "Patient"})
    wanted = ["Patient", "Disease", "Group", "FusionGeneatDiagnosis", "Infant/Child", "Survival"]
    existing = [c for c in wanted if c in clin.columns]
    out = clin[existing].copy()
    if "Patient" not in out.columns:
        raise ValueError("Clinical sheet must contain Patient_ID or Patient column.")
    return out.drop_duplicates("Patient")


def ordered_columns(df, preferred):
    """Ensure preferred columns exist and order them first, preserving any additional columns after."""
    out = df.copy()
    for col in preferred:
        if col not in out.columns:
            out[col] = np.nan
    return out[preferred + [c for c in out.columns if c not in preferred]]


def write_excel_outputs(fits_long, case_table, collapsed_eligible):
    comparison_xlsx = os.path.join(OUT_DIR, "iScience_case_level_AICc_model_comparison.xlsx")
    with pd.ExcelWriter(comparison_xlsx, engine="openpyxl") as writer:
        fits_long.to_excel(writer, sheet_name="Case_Level_Model_Comparison", index=False)
        case_table.to_excel(writer, sheet_name="Case_Level_Model_Summary", index=False)
        collapsed_eligible.to_excel(writer, sheet_name="VAF_Timeseries_Collapsed", index=False)

    if UPDATE_SUPPLEMENTARY_WORKBOOK:
        mode = "a" if os.path.exists(SUPPLEMENTARY_XLSX) else "w"
        writer_kwargs = {"engine": "openpyxl", "mode": mode}
        if mode == "a":
            writer_kwargs["if_sheet_exists"] = "replace"
        with pd.ExcelWriter(SUPPLEMENTARY_XLSX, **writer_kwargs) as writer:
            fits_long.to_excel(writer, sheet_name="Case_Level_Model_Comparison", index=False)
            case_table.to_excel(writer, sheet_name="Case_Level_Model_Summary", index=False)
            collapsed_eligible.to_excel(writer, sheet_name="VAF_Timeseries_Collapsed", index=False)

    return comparison_xlsx


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df13 = pd.read_excel(IN_XLSX, sheet_name="SuppData13_CoverageTargetSeq.", header=3)
    df13.columns = [str(c).strip() for c in df13.columns]
    df13["Corrected-VAF"] = pd.to_numeric(df13["Corrected-VAF"], errors="coerce")
    df13["Coverage"] = pd.to_numeric(df13["Coverage"], errors="coerce")
    df13["day"] = pd.to_numeric(df13["Sample"].astype(str).str.extract(r"-d(\d+)")[0], errors="coerce")
    df13 = df13[
        np.isfinite(df13["Corrected-VAF"])
        & np.isfinite(df13["day"])
        & (df13["Coverage"].fillna(0) >= 100)
    ].copy()

    clin = pd.read_excel(IN_XLSX, sheet_name="SuppData1_ClinicalData", header=3)
    clin.columns = [str(c).strip() for c in clin.columns]
    clin_meta = safe_clinical_metadata(clin)

    collapsed = (
        df13.groupby(["Patient", "day"], as_index=False)
        .agg(
            mean_vaf=("Corrected-VAF", "mean"),
            median_vaf=("Corrected-VAF", "median"),
            max_vaf=("Corrected-VAF", "max"),
            sd_vaf=("Corrected-VAF", "std"),
            n_variants=("Corrected-VAF", "count"),
            detected_variants=("Corrected-VAF", lambda x: int((x > 1e-4).sum())),
        )
    )
    collapsed["detected_frac"] = collapsed["detected_variants"] / collapsed["n_variants"]
    collapsed = collapsed.merge(clin_meta, on="Patient", how="left")
    collapsed["z"] = logit(collapsed["mean_vaf"])
    q1, q2 = collapsed["z"].quantile([1 / 3, 2 / 3])

    counts = collapsed.groupby("Patient")["day"].nunique()
    eligible = counts[counts >= 8].index.tolist()

    fits = []
    for pid in eligible:
        s = collapsed[collapsed.Patient == pid].sort_values("day")
        t = s["day"].to_numpy(float) / 365.25
        y = logit(s["mean_vaf"].to_numpy(float))
        fitters = [
            fit_brownian,
            fit_branching_drift,
            fit_ou,
            lambda tt, yy: fit_markov_emission(tt, yy, q1, q2),
            fit_ou_branching_jump,
        ]
        for fitter in fitters:
            result = fitter(t, y)
            result["Patient"] = pid
            result["n_timepoints"] = len(y)
            result["min_day"] = float(s["day"].min())
            result["max_day"] = float(s["day"].max())
            fits.append(result)

    fits_df = pd.DataFrame(fits)
    fits_df["AIC"] = 2 * fits_df["k"] - 2 * fits_df["ll"]
    fits_df["AICc"] = fits_df.apply(lambda r: aicc_from_ll(r["ll"], int(r["k"]), int(r["n"])), axis=1)
    fits_df["BIC"] = fits_df.apply(lambda r: bic_from_ll(r["ll"], int(r["k"]), int(r["n"])), axis=1)
    fits_df["delta_AICc"] = fits_df.groupby("Patient")["AICc"].transform(lambda x: x - np.nanmin(x))
    fits_df["akaike_weight"] = fits_df.groupby("Patient")["delta_AICc"].transform(
        lambda d: np.exp(-0.5 * d) / np.nansum(np.exp(-0.5 * d))
    )

    # Add disease/subtype/relapse metadata and explicit best-model annotation to every long-form row.
    fits_df = fits_df.merge(clin_meta, on="Patient", how="left")
    best_idx = fits_df.groupby("Patient")["AICc"].idxmin()
    best = fits_df.loc[
        best_idx,
        ["Patient", "model", "AICc", "BIC", "akaike_weight"],
    ].rename(
        columns={
            "model": "best_supported_model",
            "AICc": "best_AICc",
            "BIC": "best_BIC",
            "akaike_weight": "best_weight",
        }
    )
    fits_df = fits_df.merge(best[["Patient", "best_supported_model"]], on="Patient", how="left")
    fits_df["is_best_model"] = fits_df["model"].eq(fits_df["best_supported_model"])
    fits_df["source_script"] = SOURCE_SCRIPT

    long_cols = [
        "Patient",
        "Disease",
        "Group",
        "model",
        "best_supported_model",
        "is_best_model",
        "n",
        "n_timepoints",
        "k",
        "ll",
        "AIC",
        "AICc",
        "BIC",
        "delta_AICc",
        "akaike_weight",
        "min_day",
        "max_day",
        "source_script",
        "FusionGeneatDiagnosis",
        "Infant/Child",
        "Survival",
        "sigma",
        "beta",
        "theta",
        "mu",
        "n_states",
        "branch_prob",
        "jump_sd",
        "success",
    ]
    fits_long = ordered_columns(fits_df, long_cols)
    fits_long.to_csv(os.path.join(OUT_DIR, "iScience_case_level_model_fits_long.csv"), index=False)

    collapsed_eligible = collapsed[collapsed.Patient.isin(eligible)].copy()
    collapsed_eligible["source_script"] = SOURCE_SCRIPT
    collapsed_eligible.to_csv(os.path.join(OUT_DIR, "iScience_case_level_VAF_timeseries_collapsed.csv"), index=False)

    # Compact patient-level summary table.
    wide_delta = fits_df.pivot_table(index="Patient", columns="model", values="delta_AICc", aggfunc="first").reindex(columns=MODEL_ORDER)
    wide_weight = fits_df.pivot_table(index="Patient", columns="model", values="akaike_weight", aggfunc="first").reindex(columns=MODEL_ORDER)
    wide_aicc = fits_df.pivot_table(index="Patient", columns="model", values="AICc", aggfunc="first").reindex(columns=MODEL_ORDER)
    wide_bic = fits_df.pivot_table(index="Patient", columns="model", values="BIC", aggfunc="first").reindex(columns=MODEL_ORDER)

    meta = (
        collapsed.groupby("Patient")
        .agg(
            n_timepoints=("day", "nunique"),
            min_day=("day", "min"),
            max_day=("day", "max"),
            n_variants_median=("n_variants", "median"),
            mean_vaf_diagnosis=("mean_vaf", "first"),
            mean_vaf_final=("mean_vaf", "last"),
        )
        .reset_index()
    )
    case_table = (
        meta[meta.Patient.isin(eligible)]
        .merge(clin_meta, on="Patient", how="left")
        .merge(best, on="Patient", how="left")
    )
    for m in MODEL_ORDER:
        safe_m = m.replace(" ", "_").replace("-", "_")
        case_table[f"delta_AICc_{safe_m}"] = case_table["Patient"].map(wide_delta[m])
        case_table[f"akaike_weight_{safe_m}"] = case_table["Patient"].map(wide_weight[m])
        case_table[f"AICc_{safe_m}"] = case_table["Patient"].map(wide_aicc[m])
        case_table[f"BIC_{safe_m}"] = case_table["Patient"].map(wide_bic[m])
    case_table["source_script"] = SOURCE_SCRIPT
    case_table.to_csv(os.path.join(OUT_DIR, "iScience_case_level_AICc_model_comparison.csv"), index=False)

    write_excel_outputs(fits_long, case_table, collapsed_eligible)

    # Figure 5 heatmap.
    group_order = {
        "Very early": 0,
        "Very early/refractory": 1,
        "Early": 2,
        "Early/refractory": 3,
        "Late": 4,
        "Remission": 5,
    }
    case_table["_order"] = case_table["Group"].map(group_order).fillna(9)
    case_table = case_table.sort_values(["_order", "Disease", "Patient"])
    plot_patients = case_table["Patient"].tolist()
    plot_delta = wide_delta.loc[plot_patients, MODEL_ORDER]
    mat = np.clip(plot_delta.values.astype(float), 0, 25)
    patient_labels = [
        f"{r.Patient}  {r.Disease}  {str(r.Group).replace('Very early/refractory','VE/R').replace('Very early','VE').replace('Remission','REM')}"
        for r in case_table.itertuples()
    ]
    model_labels = ["Brownian", "Branching\ndrift", "OU", "Markov\nemission", "OU-Branching\njump"]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(patient_labels)))
    ax.set_yticklabels(patient_labels)
    ax.set_ylabel("Patient / disease / relapse group")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔAICc (capped at 25)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = plot_delta.values[i, j]
            ax.text(j, i, ">25" if val > 25 else f"{val:.1f}", ha="center", va="center", fontsize=6)

    for i, pid in enumerate(plot_patients):
        best_model = best.loc[best.Patient == pid, "best_supported_model"].iloc[0]
        j = MODEL_ORDER.index(best_model)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, linewidth=1.6))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Figure5_case_level_model_comparison_AICc.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Figure5_case_level_model_comparison_AICc.pdf"), bbox_inches="tight")

    print("Done.")
    print(f"Eligible patients: {len(eligible)}")
    print(f"Long-form S8 rows: {len(fits_long)}")
    print("Best-supported model counts:")
    print(best["best_supported_model"].value_counts().reindex(MODEL_ORDER, fill_value=0).to_string())
    print(f"Updated long-form CSV: {os.path.join(OUT_DIR, 'iScience_case_level_model_fits_long.csv')}")
    print(f"Updated compact CSV: {os.path.join(OUT_DIR, 'iScience_case_level_AICc_model_comparison.csv')}")
    print(f"Updated Excel workbook: {os.path.join(OUT_DIR, 'iScience_case_level_AICc_model_comparison.xlsx')}")
    if UPDATE_SUPPLEMENTARY_WORKBOOK:
        print(f"Updated supplementary workbook: {SUPPLEMENTARY_XLSX}")


if __name__ == "__main__":
    main()
