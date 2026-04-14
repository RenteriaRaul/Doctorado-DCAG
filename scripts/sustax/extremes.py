import os
import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================================================
# HELPERS
# =========================================================
def normalize_date_series_to_iso(s: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt = d1 if d1.notna().sum() > d0.notna().sum() else d0
    dt = dt.dt.normalize()
    return dt.dt.strftime("%Y-%m-%d")


def normalize_date_series_to_dt(s: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt = d1 if d1.notna().sum() > d0.notna().sum() else d0
    return dt.dt.normalize()


def pick_precip_col(df: pd.DataFrame) -> str:
    meta_cols = {"sustax_total", "lat", "lon", "date"}
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

    raise ValueError(f"No se pudo identificar columna de precipitación en {df.columns.tolist()}")


def get_scenario_file(sustax_total: str, scenario: str, folder: str) -> Optional[str]:
    pattern = os.path.join(folder, f"{sustax_total}__{scenario}*.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def get_available_scenarios_for_point(sustax_total: str, folder: str) -> list[str]:
    files = glob.glob(os.path.join(folder, f"{sustax_total}__*.csv"))
    scenarios = []
    for fp in files:
        name = os.path.basename(fp).replace(".csv", "")
        parts = name.split("__")
        if len(parts) >= 2 and "summary" not in name.lower():
            scenarios.append(parts[1])
    return sorted(list(set(scenarios)))


# =========================================================
# EVENTO 2015 POR ESCENARIO
# =========================================================
def build_event_scenarios_table(
    byscen_total_folder: str,
    target_date: str = "2015-10-23",
    target_month: int = 10,
) -> pd.DataFrame:
    scenario_files = glob.glob(os.path.join(byscen_total_folder, "*.csv"))
    rows = []

    for fp in sorted(scenario_files):
        name = os.path.basename(fp)

        if "summary" in name.lower():
            continue

        parts = name.replace(".csv", "").split("__")
        if len(parts) < 2:
            continue

        sustax_total = parts[0]
        scenario = parts[1]

        df = pd.read_csv(fp)
        if "date" not in df.columns:
            continue

        df["date"] = normalize_date_series_to_dt(df["date"])
        df = df.dropna(subset=["date"]).copy()

        pp_col = pick_precip_col(df)
        df[pp_col] = pd.to_numeric(df[pp_col], errors="coerce")

        target_dt = pd.Timestamp(target_date)
        event_match = df.loc[df["date"] == target_dt, pp_col]
        event_value = float(event_match.iloc[0]) if len(event_match) else np.nan

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        month_2015 = df[(df["year"] == 2015) & (df["month"] == target_month)]
        monthly_max_2015_10 = float(month_2015[pp_col].max()) if not month_2015.empty else np.nan

        monthly_max_oct = (
            df[df["month"] == target_month]
            .groupby(["year", "month"])[pp_col]
            .max()
            .reset_index()
        )

        if monthly_max_oct.empty or np.isnan(monthly_max_2015_10):
            percentile = np.nan
            rank = np.nan
            n_oct = len(monthly_max_oct)
        else:
            arr = monthly_max_oct[pp_col].dropna().values
            n_oct = len(arr)
            rank = int(np.sum(arr <= monthly_max_2015_10))
            percentile = 100.0 * rank / n_oct if n_oct > 0 else np.nan

        rows.append(
            {
                "sustax_total": sustax_total,
                "scenario": scenario,
                "event_value_mm": event_value,
                "monthly_max_2015_10_mm": monthly_max_2015_10,
                "october_percentile": percentile,
                "october_rank": rank,
                "n_october_years": n_oct,
            }
        )

    return pd.DataFrame(rows).sort_values(["sustax_total", "scenario"]).reset_index(drop=True)


def build_event_validation_summary(
    obs_simple_total_dict: dict[str, pd.DataFrame],
    obs_idw_total_dict: dict[str, pd.DataFrame],
    df_event_scenarios: pd.DataFrame,
    target_date: str = "2015-10-23",
) -> pd.DataFrame:
    rows = []

    all_keys = sorted(set(obs_simple_total_dict.keys()) | set(obs_idw_total_dict.keys()))

    for sustax_total in all_keys:
        df_sim = obs_simple_total_dict.get(sustax_total)
        df_idw = obs_idw_total_dict.get(sustax_total)

        v_sim = (
            df_sim.loc[df_sim["date"] == target_date, "pp_mm"]
            if df_sim is not None
            else pd.Series(dtype=float)
        )
        v_idw = (
            df_idw.loc[df_idw["date"] == target_date, "pp_mm"]
            if df_idw is not None
            else pd.Series(dtype=float)
        )

        obs_simple = float(v_sim.iloc[0]) if len(v_sim) else np.nan
        obs_idw = float(v_idw.iloc[0]) if len(v_idw) else np.nan

        df_sc = df_event_scenarios[df_event_scenarios["sustax_total"] == sustax_total].copy()

        if df_sc.empty:
            rows.append(
                {
                    "sustax_total": sustax_total,
                    "scenario": np.nan,
                    "obs_simple_mm": obs_simple,
                    "obs_idw_mm": obs_idw,
                    "era5_mm": np.nan,
                    "era5_obs_ratio": np.nan,
                    "scenario_event_mm": np.nan,
                    "scenario_monthly_max_mm": np.nan,
                    "scenario_oct_percentile": np.nan,
                }
            )
            continue

        era5_row = df_sc[df_sc["scenario"] == "ERA5"]
        era5_val = float(era5_row["event_value_mm"].iloc[0]) if not era5_row.empty else np.nan
        ratio = era5_val / obs_idw if (pd.notna(obs_idw) and obs_idw > 0 and pd.notna(era5_val)) else np.nan

        for _, r in df_sc.iterrows():
            rows.append(
                {
                    "sustax_total": sustax_total,
                    "scenario": r["scenario"],
                    "obs_simple_mm": obs_simple,
                    "obs_idw_mm": obs_idw,
                    "era5_mm": era5_val,
                    "era5_obs_ratio": ratio,
                    "scenario_event_mm": r["event_value_mm"],
                    "scenario_monthly_max_mm": r["monthly_max_2015_10_mm"],
                    "scenario_oct_percentile": r["october_percentile"],
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# SERIES SEMANALES
# =========================================================
def compute_monthly_max_weekly(df: pd.DataFrame, value_col: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["date"] = normalize_date_series_to_dt(out["date"])
    out = out.dropna(subset=["date"]).copy()

    if value_col is None:
        value_col = pick_precip_col(out)

    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)

    out["weekly_accum_7d_mm"] = out[value_col].rolling(window=7, min_periods=7).sum()
    out["year_month"] = out["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        out.groupby("year_month")["weekly_accum_7d_mm"]
        .max()
        .reset_index()
        .rename(columns={"weekly_accum_7d_mm": "monthly_max_weekly_mm"})
    )

    return out, monthly


# =========================================================
# HEATMAPS
# =========================================================
def plot_percentile_heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    title: str,
    out_fp: str,
    vmin: float = 0,
    vmax: float = 100,
    cmap_name: str = "viridis",
):
    heat = df.pivot(index=index_col, columns=columns_col, values=values_col)

    desired_cols = ["ERA5", "SSP119", "SSP126", "SSP245", "SSP370", "SSP434", "SSP460", "SSP585"]
    heat = heat.reindex(columns=[c for c in desired_cols if c in heat.columns])

    data = heat.values.astype(float)

    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(11, 7))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticklabels(heat.index)

    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="black")
            else:
                txt_color = "white" if val < 55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentil octubre 2015")

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()


def plot_percentile_heatmap_ssp(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_fp: str,
    points_to_use: list[str],
    ssp_list: list[str],
):
    heat = df.pivot(index="sustax_total", columns="scenario", values=value_col)
    heat = heat.reindex(index=points_to_use, columns=ssp_list)

    data = heat.values.astype(float)

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticklabels(heat.index)

    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="black")
            else:
                txt_color = "white" if val < 55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentil octubre 2015")

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# FIGURAS DE COMPARACIÓN
# =========================================================
def plot_daily_vs_weekly(
    point_info: pd.DataFrame,
    sustax_total: str,
    out_fp: str,
):
    col_daily = None
    col_weekly = None

    possible_daily = [
        "monthly_max_daily_2015_10_mm",
        "monthly_max_2015_10_mm",
        "scenario_monthly_max_mm",
    ]
    possible_weekly = [
        "monthly_max_weekly_2015_10_mm",
        "event_value_weekly_7d_mm",
    ]

    for c in possible_daily:
        if c in point_info.columns:
            col_daily = c
            break

    for c in possible_weekly:
        if c in point_info.columns:
            col_weekly = c
            break

    if col_daily is None or col_weekly is None:
        raise ValueError(
            f"No se encontraron columnas adecuadas para diario/semanal en {sustax_total}"
        )

    point_info[col_daily] = pd.to_numeric(point_info[col_daily], errors="coerce")
    point_info[col_weekly] = pd.to_numeric(point_info[col_weekly], errors="coerce")

    x = np.arange(len(point_info))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, point_info[col_daily], width=width, label=f"Diario ({col_daily})")
    plt.bar(x + width / 2, point_info[col_weekly], width=width, label=f"Semanal ({col_weekly})")

    plt.xticks(x, point_info["scenario"], rotation=45)
    plt.ylabel("Precipitación (mm)")
    plt.title(f"Comparación diario vs semanal – {sustax_total}\nOctubre 2015")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()
