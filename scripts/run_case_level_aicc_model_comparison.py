import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

IN_XLSX = "/Ahlgren_2025_Supplementary_Data_1-18.xlsx"
OUT_DIR = "/Revision"
EPS = 1e-5
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


def fit_brownian(t, y):
    dt, dy = np.diff(t), np.diff(y)
    def nll(par):
        sigma = np.exp(par[0])
        return -np.sum(normal_logpdf(dy, 0.0, sigma**2 * dt))
    init_sigma = np.sqrt(np.mean((dy**2) / np.maximum(dt, 1e-6)))
    res = minimize(nll, [np.log(max(init_sigma, 1e-4))], bounds=[(-10, 5)], method="L-BFGS-B")
    return {"model": "Brownian", "ll": -res.fun, "k": 1, "n": len(dy), "sigma": float(np.exp(res.x[0])), "success": res.success}


def fit_branching_drift(t, y):
    dt, dy = np.diff(t), np.diff(y)
    def nll(par):
        beta, sigma = par[0], np.exp(par[1])
        return -np.sum(normal_logpdf(dy, beta * dt, sigma**2 * dt))
    beta0 = np.sum(dy) / np.sum(dt)
    sigma0 = np.sqrt(np.mean(((dy - beta0 * dt) ** 2) / np.maximum(dt, 1e-6)))
    res = minimize(nll, [beta0, np.log(max(sigma0, 1e-4))], bounds=[(-50, 50), (-10, 5)], method="L-BFGS-B")
    return {"model": "Branching-only drift", "ll": -res.fun, "k": 2, "n": len(dy), "beta": float(res.x[0]), "sigma": float(np.exp(res.x[1])), "success": res.success}


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
    return {"model": "OU", "ll": -res.fun, "k": 3, "n": len(dt), "theta": float(np.exp(res.x[0])), "mu": float(res.x[1]), "sigma": float(np.exp(res.x[2])), "success": res.success}


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
    return {"model": "Markov-emission", "ll": -res.fun, "k": len(states) + 1, "n": len(ynext), "n_states": len(states), "sigma": float(np.exp(res.x[-1])), "success": res.success}


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
            starts.append([np.log(theta0), ymean, np.log(max(ystd / 2, 1e-3)), np.log(p0 / (1 - p0)), np.log(ystd)])
    best = None
    for st in starts:
        res = minimize(nll, st, bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    res = best
    theta, mu, sigma = np.exp(res.x[0]), res.x[1], np.exp(res.x[2])
    p, tau = 1 / (1 + np.exp(-res.x[3])), np.exp(res.x[4])
    return {"model": "OU-Branching jump", "ll": -res.fun, "k": 5, "n": len(dt), "theta": float(theta), "mu": float(mu), "sigma": float(sigma), "branch_prob": float(p), "jump_sd": float(tau), "success": res.success}


def main():
    df13 = pd.read_excel(IN_XLSX, sheet_name="SuppData13_CoverageTargetSeq.", header=3)
    df13.columns = [str(c).strip() for c in df13.columns]
    df13["Corrected-VAF"] = pd.to_numeric(df13["Corrected-VAF"], errors="coerce")
    df13["Coverage"] = pd.to_numeric(df13["Coverage"], errors="coerce")
    df13["day"] = df13["Sample"].astype(str).str.extract(r"-d(\d+)").astype(float)
    df13 = df13[np.isfinite(df13["Corrected-VAF"]) & np.isfinite(df13["day"]) & (df13["Coverage"].fillna(0) >= 100)]

    clin = pd.read_excel(IN_XLSX, sheet_name="SuppData1_ClinicalData", header=3)
    clin.columns = [str(c).strip() for c in clin.columns]
    clin = clin.rename(columns={"Patient_ID": "Patient"})

    collapsed = (df13.groupby(["Patient", "day"], as_index=False)
                 .agg(mean_vaf=("Corrected-VAF", "mean"), median_vaf=("Corrected-VAF", "median"), max_vaf=("Corrected-VAF", "max"), sd_vaf=("Corrected-VAF", "std"), n_variants=("Corrected-VAF", "count"), detected_variants=("Corrected-VAF", lambda x: int((x > 1e-4).sum()))))
    collapsed["detected_frac"] = collapsed["detected_variants"] / collapsed["n_variants"]
    collapsed = collapsed.merge(clin[["Patient", "Disease", "Group", "FusionGeneatDiagnosis", "Infant/Child", "Survival"]], on="Patient", how="left")
    collapsed["z"] = logit(collapsed["mean_vaf"])
    q1, q2 = collapsed["z"].quantile([1 / 3, 2 / 3])

    counts = collapsed.groupby("Patient")["day"].nunique()
    eligible = counts[counts >= 8].index.tolist()
    fits = []
    for pid in eligible:
        s = collapsed[collapsed.Patient == pid].sort_values("day")
        t = s["day"].to_numpy(float) / 365.25
        y = logit(s["mean_vaf"].to_numpy(float))
        for fitter in [fit_brownian, fit_branching_drift, fit_ou, lambda tt, yy: fit_markov_emission(tt, yy, q1, q2), fit_ou_branching_jump]:
            result = fitter(t, y)
            result["Patient"] = pid
            result["n_timepoints"] = len(y)
            result["min_day"] = float(s["day"].min())
            result["max_day"] = float(s["day"].max())
            fits.append(result)
    fits_df = pd.DataFrame(fits)
    fits_df["AIC"] = 2 * fits_df["k"] - 2 * fits_df["ll"]
    fits_df["AICc"] = fits_df.apply(lambda r: aicc_from_ll(r["ll"], int(r["k"]), int(r["n"])), axis=1)
    fits_df["delta_AICc"] = fits_df.groupby("Patient")["AICc"].transform(lambda x: x - np.nanmin(x))
    fits_df["akaike_weight"] = fits_df.groupby("Patient")["delta_AICc"].transform(lambda d: np.exp(-0.5 * d) / np.nansum(np.exp(-0.5 * d)))

    fits_df.to_csv(os.path.join(OUT_DIR, "iScience_case_level_model_fits_long.csv"), index=False)
    collapsed[collapsed.Patient.isin(eligible)].to_csv(os.path.join(OUT_DIR, "iScience_case_level_VAF_timeseries_collapsed.csv"), index=False)

    wide_delta = fits_df.pivot(index="Patient", columns="model", values="delta_AICc").reindex(columns=MODEL_ORDER)
    best_idx = fits_df.groupby("Patient")["AICc"].idxmin()
    best = fits_df.loc[best_idx, ["Patient", "model", "AICc", "akaike_weight"]].rename(columns={"model": "best_model", "AICc": "best_AICc", "akaike_weight": "best_weight"})
    meta = collapsed.groupby("Patient").agg(n_timepoints=("day", "nunique"), min_day=("day", "min"), max_day=("day", "max"), n_variants_median=("n_variants", "median"), mean_vaf_diagnosis=("mean_vaf", "first"), mean_vaf_final=("mean_vaf", "last")).reset_index()
    case_table = meta[meta.Patient.isin(eligible)].merge(clin[["Patient", "Disease", "Group", "FusionGeneatDiagnosis", "Infant/Child", "Survival"]], on="Patient", how="left").merge(best, on="Patient", how="left")
    for m in MODEL_ORDER:
        case_table[f"delta_AICc_{m}"] = case_table["Patient"].map(wide_delta[m])
    case_table.to_csv(os.path.join(OUT_DIR, "iScience_case_level_AICc_model_comparison.csv"), index=False)

    # Figure 5 heatmap
    group_order = {"Very early": 0, "Very early/refractory": 1, "Early": 2, "Early/refractory": 3, "Late": 4, "Remission": 5}
    case_table["_order"] = case_table["Group"].map(group_order).fillna(9)
    case_table = case_table.sort_values(["_order", "Disease", "Patient"])
    plot_patients = case_table["Patient"].tolist()
    plot_delta = wide_delta.loc[plot_patients, MODEL_ORDER]
    mat = np.clip(plot_delta.values.astype(float), 0, 25)
    patient_labels = [f"{r.Patient}  {r.Disease}  {str(r.Group).replace('Very early/refractory','VE/R').replace('Very early','VE').replace('Remission','REM')}" for r in case_table.itertuples()]
    model_labels = ["Brownian", "Branching\ndrift", "OU", "Markov\nemission", "OU-Branching\njump"]
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(patient_labels)))
    ax.set_yticklabels(patient_labels)
    #ax.set_title("Figure 5. Case-level model comparison using longitudinal VAF trajectories")
    #ax.set_xlabel("Candidate model")
    ax.set_ylabel("Patient / disease / relapse group")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔAICc (capped at 25)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = plot_delta.values[i, j]
            ax.text(j, i, ">25" if val > 25 else f"{val:.1f}", ha="center", va="center", fontsize=6)
    for i, pid in enumerate(plot_patients):
        best_model = best.loc[best.Patient == pid, "best_model"].iloc[0]
        j = MODEL_ORDER.index(best_model)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, linewidth=1.6))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Figure5_case_level_model_comparison_AICc.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Figure5_case_level_model_comparison_AICc.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
