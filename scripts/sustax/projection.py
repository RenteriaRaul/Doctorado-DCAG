import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# HELPERS
# =========================================================
def normalize_date_series(s: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt = d1 if d1.notna().sum() > d0.notna().sum() else d0
    return dt.dt.normalize()


def pick_precip_col(df: pd.DataFrame) -> str:
    meta_cols = {"sustax_total", "lat", "lon", "date", "year", "month"}
    candidates = [c for c in df.columns if c not in meta_cols]

    for c in candidates:
        cl = c.lower()
        if ("tp" in cl) and ("precip" in cl):
            return c
    for c in candidates:
        cl = c.lower()
        if ("pr" in cl) and ("precip" in cl):
            return c
    for c in candidates:
        if "precip" in c.lower():
            return c

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"No se pudo detectar columna de precipitación en {df.columns.tolist()}")


def get_scenario_files(folder: str, ssp_list: list[str]) -> list[tuple[str, str, str]]:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    out = []

    for fp in files:
        name = os.path.basename(fp)
        if "summary" in name.lower():
            continue

        parts = name.replace(".csv", "").split("__")
        if len(parts) < 2:
            continue

        sustax_total = parts[0]
        scenario = parts[1]

        if scenario in ssp_list:
            out.append((sustax_total, scenario, fp))

    return out


def get_scenario_file(sustax_total: str, scenario: str, folder: str) -> Optional[str]:
    pattern = os.path.join(folder, f"{sustax_total}__{scenario}*.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def monthly_max_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out["year_month"] = out["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        out.groupby("year_month")[value_col]
        .max()
        .reset_index()
        .rename(columns={value_col: "monthly_max_mm"})
    )
    monthly["year"] = monthly["year_month"].dt.year

    return monthly


def summarize_period(df: pd.DataFrame, value_col: str, y0: int, y1: int) -> dict:
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()

    if sub.empty:
        return {
            "n": 0,
            "mean_daily_mm": np.nan,
            "p95_daily_mm": np.nan,
            "p99_daily_mm": np.nan,
            "max_daily_mm": np.nan,
        }

    vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
    if len(vals) == 0:
        return {
            "n": 0,
            "mean_daily_mm": np.nan,
            "p95_daily_mm": np.nan,
            "p99_daily_mm": np.nan,
            "max_daily_mm": np.nan,
        }

    return {
        "n": len(vals),
        "mean_daily_mm": float(vals.mean()),
        "p95_daily_mm": float(vals.quantile(0.95)),
        "p99_daily_mm": float(vals.quantile(0.99)),
        "max_daily_mm": float(vals.max()),
    }


def summarize_monthly_max(monthly_df: pd.DataFrame, y0: int, y1: int) -> dict:
    sub = monthly_df[(monthly_df["year"] >= y0) & (monthly_df["year"] <= y1)].copy()

    if sub.empty:
        return {
            "n_months": 0,
            "mean_monthly_max_mm": np.nan,
            "p95_monthly_max_mm": np.nan,
            "max_monthly_max_mm": np.nan,
        }

    vals = pd.to_numeric(sub["monthly_max_mm"], errors="coerce").dropna()
    if len(vals) == 0:
        return {
            "n_months": 0,
            "mean_monthly_max_mm": np.nan,
            "p95_monthly_max_mm": np.nan,
            "max_monthly_max_mm": np.nan,
        }

    return {
        "n_months": len(vals),
        "mean_monthly_max_mm": float(vals.mean()),
        "p95_monthly_max_mm": float(vals.quantile(0.95)),
        "max_monthly_max_mm": float(vals.max()),
    }


def compute_change(fut: float, hist: float) -> tuple[float, float]:
    if pd.isna(hist) or pd.isna(fut):
        return np.nan, np.nan

    delta = fut - hist
    pct = 100.0 * delta / hist if hist != 0 else np.nan
    return delta, pct


# =========================================================
# RESUMEN HISTÓRICO VS FUTURO
# =========================================================
def build_future_projection_summary(
    byscen_total_folder: str,
    out_folder: str,
    ssp_list: list[str],
    hist_start: int = 1980,
    hist_end: int = 2014,
    fut1_start: int = 2015,
    fut1_end: int = 2049,
    fut2_start: int = 2050,
    fut2_end: int = 2080,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_folder, exist_ok=True)

    rows = []
    triplets = get_scenario_files(byscen_total_folder, ssp_list)

    for sustax_total, scenario, fp in triplets:
        df = pd.read_csv(fp)

        if "date" not in df.columns:
            continue

        df["date"] = normalize_date_series(df["date"])
        df = df.dropna(subset=["date"]).copy()

        pp_col = pick_precip_col(df)
        df[pp_col] = pd.to_numeric(df[pp_col], errors="coerce")
        df = df.dropna(subset=[pp_col]).sort_values("date").reset_index(drop=True)

        df["year"] = df["date"].dt.year

        hist = summarize_period(df, pp_col, hist_start, hist_end)
        fut1 = summarize_period(df, pp_col, fut1_start, fut1_end)
        fut2 = summarize_period(df, pp_col, fut2_start, fut2_end)

        monthly = monthly_max_series(df, pp_col)
        hist_m = summarize_monthly_max(monthly, hist_start, hist_end)
        fut1_m = summarize_monthly_max(monthly, fut1_start, fut1_end)
        fut2_m = summarize_monthly_max(monthly, fut2_start, fut2_end)

        d_mean_daily_1, p_mean_daily_1 = compute_change(fut1["mean_daily_mm"], hist["mean_daily_mm"])
        d_p95_daily_1, p_p95_daily_1 = compute_change(fut1["p95_daily_mm"], hist["p95_daily_mm"])
        d_p99_daily_1, p_p99_daily_1 = compute_change(fut1["p99_daily_mm"], hist["p99_daily_mm"])
        d_mean_mmax_1, p_mean_mmax_1 = compute_change(fut1_m["mean_monthly_max_mm"], hist_m["mean_monthly_max_mm"])
        d_p95_mmax_1, p_p95_mmax_1 = compute_change(fut1_m["p95_monthly_max_mm"], hist_m["p95_monthly_max_mm"])

        d_mean_daily_2, p_mean_daily_2 = compute_change(fut2["mean_daily_mm"], hist["mean_daily_mm"])
        d_p95_daily_2, p_p95_daily_2 = compute_change(fut2["p95_daily_mm"], hist["p95_daily_mm"])
        d_p99_daily_2, p_p99_daily_2 = compute_change(fut2["p99_daily_mm"], hist["p99_daily_mm"])
        d_mean_mmax_2, p_mean_mmax_2 = compute_change(fut2_m["mean_monthly_max_mm"], hist_m["mean_monthly_max_mm"])
        d_p95_mmax_2, p_p95_mmax_2 = compute_change(fut2_m["p95_monthly_max_mm"], hist_m["p95_monthly_max_mm"])

        rows.append(
            {
                "sustax_total": sustax_total,
                "scenario": scenario,
                "hist_mean_daily_mm": hist["mean_daily_mm"],
                "hist_p95_daily_mm": hist["p95_daily_mm"],
                "hist_p99_daily_mm": hist["p99_daily_mm"],
                "hist_max_daily_mm": hist["max_daily_mm"],
                "fut1_mean_daily_mm": fut1["mean_daily_mm"],
                "fut1_p95_daily_mm": fut1["p95_daily_mm"],
                "fut1_p99_daily_mm": fut1["p99_daily_mm"],
                "fut1_max_daily_mm": fut1["max_daily_mm"],
                "fut2_mean_daily_mm": fut2["mean_daily_mm"],
                "fut2_p95_daily_mm": fut2["p95_daily_mm"],
                "fut2_p99_daily_mm": fut2["p99_daily_mm"],
                "fut2_max_daily_mm": fut2["max_daily_mm"],
                "hist_mean_monthly_max_mm": hist_m["mean_monthly_max_mm"],
                "hist_p95_monthly_max_mm": hist_m["p95_monthly_max_mm"],
                "hist_max_monthly_max_mm": hist_m["max_monthly_max_mm"],
                "fut1_mean_monthly_max_mm": fut1_m["mean_monthly_max_mm"],
                "fut1_p95_monthly_max_mm": fut1_m["p95_monthly_max_mm"],
                "fut1_max_monthly_max_mm": fut1_m["max_monthly_max_mm"],
                "fut2_mean_monthly_max_mm": fut2_m["mean_monthly_max_mm"],
                "fut2_p95_monthly_max_mm": fut2_m["p95_monthly_max_mm"],
                "fut2_max_monthly_max_mm": fut2_m["max_monthly_max_mm"],
                "delta_fut1_mean_daily_mm": d_mean_daily_1,
                "pct_fut1_mean_daily": p_mean_daily_1,
                "delta_fut1_p95_daily_mm": d_p95_daily_1,
                "pct_fut1_p95_daily": p_p95_daily_1,
                "delta_fut1_p99_daily_mm": d_p99_daily_1,
                "pct_fut1_p99_daily": p_p99_daily_1,
                "delta_fut1_mean_monthly_max_mm": d_mean_mmax_1,
                "pct_fut1_mean_monthly_max": p_mean_mmax_1,
                "delta_fut1_p95_monthly_max_mm": d_p95_mmax_1,
                "pct_fut1_p95_monthly_max": p_p95_mmax_1,
                "delta_fut2_mean_daily_mm": d_mean_daily_2,
                "pct_fut2_mean_daily": p_mean_daily_2,
                "delta_fut2_p95_daily_mm": d_p95_daily_2,
                "pct_fut2_p95_daily": p_p95_daily_2,
                "delta_fut2_p99_daily_mm": d_p99_daily_2,
                "pct_fut2_p99_daily": p_p99_daily_2,
                "delta_fut2_mean_monthly_max_mm": d_mean_mmax_2,
                "pct_fut2_mean_monthly_max": p_mean_mmax_2,
                "delta_fut2_p95_monthly_max_mm": d_p95_mmax_2,
                "pct_fut2_p95_monthly_max": p_p95_mmax_2,
            }
        )

    df_future_ssp = pd.DataFrame(rows).sort_values(["sustax_total", "scenario"]).reset_index(drop=True)

    out_csv = os.path.join(out_folder, "df_future_ssp_projection_summary.csv")
    df_future_ssp.to_csv(out_csv, index=False)

    table_top_intensification = (
        df_future_ssp[
            [
                "sustax_total",
                "scenario",
                "hist_p95_monthly_max_mm",
                "fut2_p95_monthly_max_mm",
                "delta_fut2_p95_monthly_max_mm",
                "pct_fut2_p95_monthly_max",
            ]
        ]
        .sort_values("pct_fut2_p95_monthly_max", ascending=False)
        .reset_index(drop=True)
    )

    table_scenario_signal = (
        df_future_ssp.groupby("scenario")[
            [
                "pct_fut1_p95_daily",
                "pct_fut2_p95_daily",
                "pct_fut1_p95_monthly_max",
                "pct_fut2_p95_monthly_max",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("pct_fut2_p95_monthly_max", ascending=False)
    )

    table_point_signal = (
        df_future_ssp.groupby("sustax_total")[
            [
                "pct_fut1_p95_daily",
                "pct_fut2_p95_daily",
                "pct_fut1_p95_monthly_max",
                "pct_fut2_p95_monthly_max",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("pct_fut2_p95_monthly_max", ascending=False)
    )

    table_top_intensification.to_csv(os.path.join(out_folder, "table_top_intensification.csv"), index=False)
    table_scenario_signal.to_csv(os.path.join(out_folder, "table_scenario_signal.csv"), index=False)
    table_point_signal.to_csv(os.path.join(out_folder, "table_point_signal.csv"), index=False)

    return df_future_ssp, table_top_intensification, table_scenario_signal, table_point_signal


# =========================================================
# PROYECCIÓN ANUAL 2027–2040
# =========================================================
def build_annual_projection_2027_2040(
    byscen_total_folder: str,
    out_folder: str,
    points_to_use: list[str],
    ssp_list: list[str],
    year_start: int = 2027,
    year_end: int = 2040,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_folder, exist_ok=True)

    rows = []

    for point in points_to_use:
        for scenario in ssp_list:
            fp = get_scenario_file(point, scenario, byscen_total_folder)

            if fp is None:
                print(f"⚠️ No encontrado: {point} - {scenario}")
                continue

            df = pd.read_csv(fp)

            if "date" not in df.columns:
                print(f"⚠️ Sin columna date: {fp}")
                continue

            df["date"] = normalize_date_series(df["date"])
            df = df.dropna(subset=["date"]).copy()

            pp_col = pick_precip_col(df)
            df[pp_col] = pd.to_numeric(df[pp_col], errors="coerce")
            df = df.dropna(subset=[pp_col]).copy()

            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            df = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()
            if df.empty:
                continue

            monthly_max = (
                df.groupby(["year", "month"])[pp_col]
                .max()
                .reset_index()
                .rename(columns={pp_col: "monthly_max_mm"})
            )

            for year, sub in df.groupby("year"):
                vals = pd.to_numeric(sub[pp_col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue

                monthly_sub = monthly_max[monthly_max["year"] == year]["monthly_max_mm"].dropna()

                rows.append(
                    {
                        "sustax_total": point,
                        "scenario": scenario,
                        "year": int(year),
                        "annual_total_mm": float(vals.sum()),
                        "annual_mean_daily_mm": float(vals.mean()),
                        "annual_p95_daily_mm": float(vals.quantile(0.95)),
                        "annual_p99_daily_mm": float(vals.quantile(0.99)),
                        "annual_max_daily_mm": float(vals.max()),
                        "annual_mean_monthly_max_mm": float(monthly_sub.mean()) if len(monthly_sub) else np.nan,
                        "annual_max_monthly_max_mm": float(monthly_sub.max()) if len(monthly_sub) else np.nan,
                        "n_days": int(len(vals)),
                    }
                )

    df_proj_annual = pd.DataFrame(rows).sort_values(["sustax_total", "scenario", "year"]).reset_index(drop=True)

    out_csv = os.path.join(out_folder, "df_projection_annual_2027_2040.csv")
    df_proj_annual.to_csv(out_csv, index=False)

    df_proj_scenario_year = (
        df_proj_annual.groupby(["scenario", "year"])[
            [
                "annual_total_mm",
                "annual_mean_daily_mm",
                "annual_p95_daily_mm",
                "annual_p99_daily_mm",
                "annual_max_daily_mm",
                "annual_mean_monthly_max_mm",
                "annual_max_monthly_max_mm",
            ]
        ]
        .mean()
        .reset_index()
    )

    df_proj_point_scenario = (
        df_proj_annual.groupby(["sustax_total", "scenario"])[
            [
                "annual_total_mm",
                "annual_mean_daily_mm",
                "annual_p95_daily_mm",
                "annual_p99_daily_mm",
                "annual_max_daily_mm",
                "annual_mean_monthly_max_mm",
                "annual_max_monthly_max_mm",
            ]
        ]
        .mean()
        .reset_index()
    )

    df_proj_scenario_year.to_csv(
        os.path.join(out_folder, "df_projection_scenario_year_mean_2027_2040.csv"),
        index=False,
    )
    df_proj_point_scenario.to_csv(
        os.path.join(out_folder, "df_projection_point_scenario_mean_2027_2040.csv"),
        index=False,
    )

    return df_proj_annual, df_proj_scenario_year, df_proj_point_scenario


# =========================================================
# FIGURAS
# =========================================================
def plot_future_signal_by_scenario(
    table_scenario_signal: pd.DataFrame,
    out_fp: str,
):
    plt.figure(figsize=(9, 5))
    x = np.arange(len(table_scenario_signal))
    plt.bar(x, table_scenario_signal["pct_fut2_p95_monthly_max"])
    plt.xticks(x, table_scenario_signal["scenario"], rotation=45)
    plt.ylabel("% cambio futuro lejano vs histórico")
    plt.title("Cambio promedio por escenario\nP95 de máximos mensuales")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()


def plot_future_signal_by_point(
    table_point_signal: pd.DataFrame,
    out_fp: str,
):
    plt.figure(figsize=(11, 5))
    x = np.arange(len(table_point_signal))
    plt.bar(x, table_point_signal["pct_fut2_p95_monthly_max"])
    plt.xticks(x, table_point_signal["sustax_total"], rotation=90)
    plt.ylabel("% cambio futuro lejano vs histórico")
    plt.title("Cambio promedio por punto\nP95 de máximos mensuales")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()


def plot_annual_series_by_point(
    df_proj_annual: pd.DataFrame,
    out_folder: str,
    ssp_list: list[str],
    scenario_colors: dict[str, str],
):
    os.makedirs(out_folder, exist_ok=True)

    for point in sorted(df_proj_annual["sustax_total"].unique()):
        sub = df_proj_annual[df_proj_annual["sustax_total"] == point].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(11, 5))
        for sc in ssp_list:
            s = sub[sub["scenario"] == sc]
            if s.empty:
                continue

            plt.plot(
                s["year"],
                s["annual_max_daily_mm"],
                marker="o",
                linewidth=1.8,
                label=sc,
                color=scenario_colors.get(sc, None),
            )

        plt.title(f"Proyección anual 2027–2040\nMáximo diario anual – {point}")
        plt.xlabel("Año")
        plt.ylabel("Máximo diario anual (mm)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_folder, f"{point}__annual_max_daily_2027_2040.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(11, 5))
        for sc in ssp_list:
            s = sub[sub["scenario"] == sc]
            if s.empty:
                continue

            plt.plot(
                s["year"],
                s["annual_p95_daily_mm"],
                marker="o",
                linewidth=1.8,
                label=sc,
                color=scenario_colors.get(sc, None),
            )

        plt.title(f"Proyección anual 2027–2040\nP95 diario anual – {point}")
        plt.xlabel("Año")
        plt.ylabel("P95 diario anual (mm)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_folder, f"{point}__annual_p95_daily_2027_2040.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_regional_mean_annual_max(
    df_proj_scenario_year: pd.DataFrame,
    out_fp: str,
    ssp_list: list[str],
    scenario_colors: dict[str, str],
):
    plt.figure(figsize=(12, 5))

    for sc in ssp_list:
        s = df_proj_scenario_year[df_proj_scenario_year["scenario"] == sc]
        if s.empty:
            continue

        plt.plot(
            s["year"],
            s["annual_max_daily_mm"],
            marker="o",
            linewidth=2,
            label=sc,
            color=scenario_colors.get(sc, None),
        )

    plt.title("Promedio regional 2027–2040\nMáximo diario anual por escenario SSP")
    plt.xlabel("Año")
    plt.ylabel("Máximo diario anual promedio (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()
