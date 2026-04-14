import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot


# =========================================================
# HELPERS
# =========================================================
def normalize_date(s: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt = d1 if d1.notna().sum() > d0.notna().sum() else d0
    return dt.dt.normalize()


def pick_precip_col(df: pd.DataFrame) -> str:
    meta = {"date", "sustax_total", "lat", "lon"}
    candidates = [c for c in df.columns if c not in meta]

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


# =========================================================
# UNIR OBS Y ERA5 POR PUNTO
# =========================================================
def build_obs_vs_era5_dataframe(
    obs_csv_path: str,
    era5_csv_path: str
) -> pd.DataFrame:
    obs = pd.read_csv(obs_csv_path)
    obs["date"] = normalize_date(obs["date"])
    obs["obs_mm"] = pd.to_numeric(obs["pp_mm"], errors="coerce")
    obs = obs[["date", "obs_mm"]].dropna()

    era5 = pd.read_csv(era5_csv_path)
    era5["date"] = normalize_date(era5["date"])
    pp_col = pick_precip_col(era5)
    era5["era5_mm"] = pd.to_numeric(era5[pp_col], errors="coerce")
    era5 = era5[["date", "era5_mm"]].dropna()

    df = pd.merge(obs, era5, on="date", how="inner").dropna()
    if not df.empty:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

    return df


# =========================================================
# MÉTRICAS
# =========================================================
def compute_validation_metrics(df: pd.DataFrame, point_name: Optional[str] = None) -> dict:
    if df.empty:
        return {
            "point": point_name,
            "bias_mm": np.nan,
            "rmse_mm": np.nan,
            "correlation": np.nan,
            "n_obs": 0,
        }

    bias = (df["era5_mm"] - df["obs_mm"]).mean()
    rmse = np.sqrt(((df["era5_mm"] - df["obs_mm"]) ** 2).mean())
    corr = df["obs_mm"].corr(df["era5_mm"])

    return {
        "point": point_name,
        "bias_mm": bias,
        "rmse_mm": rmse,
        "correlation": corr,
        "n_obs": len(df),
    }


def compute_amax(df: pd.DataFrame, point_name: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["point", "year", "obs_amax", "era5_amax"])

    amax = df.groupby("year")[["obs_mm", "era5_mm"]].max().reset_index()
    amax = amax.rename(columns={"obs_mm": "obs_amax", "era5_mm": "era5_amax"})
    amax.insert(0, "point", point_name)

    return amax


# =========================================================
# FIGURAS POR PUNTO
# =========================================================
def plot_exceedance_curve(df: pd.DataFrame, point: str, out_path: str):
    obs_sorted = np.sort(df["obs_mm"].values)[::-1]
    era5_sorted = np.sort(df["era5_mm"].values)[::-1]

    rank_obs = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    rank_era5 = np.arange(1, len(era5_sorted) + 1) / len(era5_sorted)

    plt.figure(figsize=(7, 5))
    plt.plot(obs_sorted, rank_obs, label="OBS")
    plt.plot(era5_sorted, rank_era5, label="ERA5")
    plt.xlabel("Precipitación (mm)")
    plt.ylabel("Probabilidad de excedencia")
    plt.title(f"Exceedance Curve\n{point}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_qq(df: pd.DataFrame, column: str, title: str, out_path: str):
    plt.figure(figsize=(6, 6))
    probplot(df[column], dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_amax_series(amax_df: pd.DataFrame, point: str, out_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(amax_df["year"], amax_df["obs_amax"], label="OBS", marker="o")
    plt.plot(amax_df["year"], amax_df["era5_amax"], label="ERA5", marker="o")
    plt.title(f"Annual Maximum Precipitation\n{point}")
    plt.ylabel("mm")
    plt.xlabel("year")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# FIGURA GLOBAL
# =========================================================
def plot_global_validation_figure(
    df_all: pd.DataFrame,
    out_path: str,
    title: str = "Fase A – Validación histórica de precipitación extrema\nOBS_IDW vs ERA5 en puntos Sustax Total"
):
    global_bias = (df_all["era5_mm"] - df_all["obs_mm"]).mean()
    global_rmse = np.sqrt(((df_all["era5_mm"] - df_all["obs_mm"]) ** 2).mean())
    global_corr = df_all["obs_mm"].corr(df_all["era5_mm"])
    n_total = len(df_all)

    points = sorted(df_all["point"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(points)))
    color_map = dict(zip(points, colors))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    for p in points:
        sub = df_all[df_all["point"] == p]
        ax.scatter(
            sub["obs_mm"],
            sub["era5_mm"],
            s=16,
            alpha=0.65,
            color=color_map[p],
            label=p
        )

    xy_max = np.nanmax([df_all["obs_mm"].max(), df_all["era5_mm"].max()])
    ax.plot([0, xy_max], [0, xy_max], linestyle="--", color="black", linewidth=1.5)

    ax.set_xlabel("OBS_IDW (mm)")
    ax.set_ylabel("ERA5 (mm)")
    ax.set_title("A) Validación global OBS vs ERA5")

    text_metrics = (
        f"N = {n_total}\n"
        f"Bias = {global_bias:.2f} mm\n"
        f"RMSE = {global_rmse:.2f} mm\n"
        f"r = {global_corr:.2f}"
    )

    ax.text(
        0.04, 0.96, text_metrics,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    data_box = [df_all.loc[df_all["point"] == p, "residual_mm"].dropna().values for p in points]

    bp = ax2.boxplot(
        data_box,
        patch_artist=True,
        labels=points,
        vert=True,
        showfliers=False
    )

    for patch, p in zip(bp["boxes"], points):
        patch.set_facecolor(color_map[p])
        patch.set_alpha(0.75)

    ax2.axhline(0, linestyle="--", color="black", linewidth=1.2)
    ax2.set_ylabel("Residuo (ERA5 - OBS) [mm]")
    ax2.set_title("B) Distribución de residuos por punto")
    ax2.tick_params(axis="x", rotation=90)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=15, y=1.02)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# LOOP COMPLETO POR CARPETA
# =========================================================
def run_validation_pipeline(
    obs_folder: str,
    era5_folder: str,
    out_folder: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_folder, exist_ok=True)

    obs_files = [f for f in os.listdir(obs_folder) if f.startswith("OBS_IDW__")]
    points = [f.replace("OBS_IDW__", "").replace(".csv", "") for f in obs_files]

    metrics_rows = []
    amax_parts = []
    all_parts = []

    for point in points:
        obs_path = os.path.join(obs_folder, f"OBS_IDW__{point}.csv")
        era5_path = os.path.join(era5_folder, f"{point}__ERA5__mm.csv")

        if not os.path.exists(era5_path):
            continue

        df = build_obs_vs_era5_dataframe(obs_path, era5_path)

        if len(df) < 10:
            continue

        metrics_rows.append(compute_validation_metrics(df, point_name=point))

        amax_df = compute_amax(df, point_name=point)
        amax_parts.append(amax_df)

        plot_exceedance_curve(
            df, point,
            os.path.join(out_folder, f"{point}_exceedance.png")
        )
        plot_qq(
            df, "obs_mm",
            f"QQ Plot OBS\n{point}",
            os.path.join(out_folder, f"{point}_qq_obs.png")
        )
        plot_qq(
            df, "era5_mm",
            f"QQ Plot ERA5\n{point}",
            os.path.join(out_folder, f"{point}_qq_era5.png")
        )
        plot_amax_series(
            amax_df, point,
            os.path.join(out_folder, f"{point}_amax_series.png")
        )

        df = df.copy()
        df["point"] = point
        df["residual_mm"] = df["era5_mm"] - df["obs_mm"]
        all_parts.append(df)

    df_metrics = pd.DataFrame(metrics_rows)
    df_amax = pd.concat(amax_parts, ignore_index=True) if amax_parts else pd.DataFrame()
    df_all = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

    if not df_metrics.empty:
        df_metrics.to_csv(os.path.join(out_folder, "metrics_OBS_vs_ERA5.csv"), index=False)
    if not df_amax.empty:
        df_amax.to_csv(os.path.join(out_folder, "AMAX_OBS_vs_ERA5.csv"), index=False)
    if not df_all.empty:
        plot_global_validation_figure(
            df_all,
            os.path.join(out_folder, "FINAL_validation_OBS_vs_ERA5.png")
        )

    return df_metrics, df_amax, df_all
