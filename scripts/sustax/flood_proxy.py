import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# HELPERS
# =========================================================
def normalize_date(s: pd.Series) -> pd.Series:
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


def get_scenario_file(point: str, scenario: str, byscen_total_folder: str) -> Optional[str]:
    pattern = os.path.join(byscen_total_folder, f"{point}__{scenario}*.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


# =========================================================
# DETECCIÓN DE EVENTOS EXTREMOS FUTUROS
# =========================================================
def detect_future_extreme_events(
    byscen_total_folder: str,
    points_to_use: list[str],
    ssp_list: list[str],
    year_start: int = 2027,
    year_end: int = 2040,
) -> pd.DataFrame:
    events = []

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

            df["date"] = normalize_date(df["date"])
            df = df.dropna(subset=["date"]).copy()

            precip_col = pick_precip_col(df)
            df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")
            df = df.dropna(subset=[precip_col]).copy()

            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            hist = df[df["year"] <= 2014].copy()
            fut = df[(df["year"] >= year_start) & (df["year"] <= year_end)].copy()

            if hist.empty or fut.empty:
                print(f"⚠️ Sin histórico o futuro para: {point} - {scenario}")
                continue

            # 1) Evento diario extremo
            p99_daily = hist[precip_col].quantile(0.99)
            fut_daily_extreme = fut[fut[precip_col] > p99_daily].copy()

            for _, r in fut_daily_extreme.iterrows():
                events.append(
                    {
                        "point": point,
                        "scenario": scenario,
                        "date": r["date"],
                        "year": int(r["year"]),
                        "month": int(r["month"]),
                        "event_type": "daily_extreme",
                        "value_mm": float(r[precip_col]),
                        "threshold_mm": float(p99_daily),
                    }
                )

            # 2) Evento semanal extremo
            hist = hist.sort_values("date").copy()
            fut = fut.sort_values("date").copy()

            hist["weekly_7d_mm"] = hist[precip_col].rolling(window=7, min_periods=7).sum()
            fut["weekly_7d_mm"] = fut[precip_col].rolling(window=7, min_periods=7).sum()

            p99_weekly = hist["weekly_7d_mm"].quantile(0.99)
            fut_weekly_extreme = fut[fut["weekly_7d_mm"] > p99_weekly].copy()

            for _, r in fut_weekly_extreme.iterrows():
                events.append(
                    {
                        "point": point,
                        "scenario": scenario,
                        "date": r["date"],
                        "year": int(r["year"]),
                        "month": int(r["month"]),
                        "event_type": "weekly_extreme",
                        "value_mm": float(r["weekly_7d_mm"]),
                        "threshold_mm": float(p99_weekly),
                    }
                )

            # 3) Evento mensual extremo
            hist_monthly = (
                hist.groupby(["year", "month"])[precip_col]
                .max()
                .reset_index()
                .rename(columns={precip_col: "monthly_max_mm"})
            )

            fut_monthly = (
                fut.groupby(["year", "month"])[precip_col]
                .max()
                .reset_index()
                .rename(columns={precip_col: "monthly_max_mm"})
            )

            p99_monthly = hist_monthly["monthly_max_mm"].quantile(0.99)
            fut_monthly_extreme = fut_monthly[fut_monthly["monthly_max_mm"] > p99_monthly].copy()

            for _, r in fut_monthly_extreme.iterrows():
                events.append(
                    {
                        "point": point,
                        "scenario": scenario,
                        "date": pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1),
                        "year": int(r["year"]),
                        "month": int(r["month"]),
                        "event_type": "monthly_extreme",
                        "value_mm": float(r["monthly_max_mm"]),
                        "threshold_mm": float(p99_monthly),
                    }
                )

    expected_cols = [
        "point",
        "scenario",
        "date",
        "year",
        "month",
        "event_type",
        "value_mm",
        "threshold_mm",
    ]

    df_future_events = pd.DataFrame(events, columns=expected_cols)

    if not df_future_events.empty:
        df_future_events = df_future_events.sort_values(
            ["scenario", "event_type", "date"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    return df_future_events


def export_future_events(df_future_events: pd.DataFrame, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_future_events.to_csv(out_csv, index=False)


# =========================================================
# RESÚMENES
# =========================================================
def build_summary_by_scenario_and_type(df_future_events: pd.DataFrame) -> pd.DataFrame:
    return (
        df_future_events.groupby(["scenario", "event_type"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["event_type", "n_events"], ascending=[True, False])
    )


def build_timeline_by_scenario(df_future_events: pd.DataFrame) -> pd.DataFrame:
    return (
        df_future_events.groupby(["scenario", "year"])
        .size()
        .reset_index(name="n_events")
    )


def build_timeline_by_event_type(df_future_events: pd.DataFrame) -> pd.DataFrame:
    return (
        df_future_events.groupby(["event_type", "year"])
        .size()
        .reset_index(name="n_events")
    )


def build_station_counts(df_future_events: pd.DataFrame) -> pd.Series:
    return df_future_events.groupby("point").size().sort_values(ascending=True)


def build_month_counts(df_future_events: pd.DataFrame) -> pd.Series:
    return df_future_events.groupby("month").size().reindex(range(1, 13), fill_value=0)


# =========================================================
# FIGURAS
# =========================================================
def plot_heatmap_by_scenario(
    df_future_events: pd.DataFrame,
    out_fp: str,
):
    timeline = build_timeline_by_scenario(df_future_events)
    heat = timeline.pivot(index="scenario", columns="year", values="n_events")

    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(
        heat,
        cmap="Reds",
        linewidths=0.3,
        annot=False,
        cbar_kws={"label": "Número de eventos extremos"},
    )

    years = list(heat.columns)
    step = 2 if len(years) > 10 else 1
    ax.set_xticks(np.arange(0.5, len(years), step))
    ax.set_xticklabels(years[::step], rotation=45)

    plt.title("Eventos extremos proyectados por escenario")
    plt.xlabel("Año")
    plt.ylabel("Escenario SSP")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap_by_event_type(
    df_future_events: pd.DataFrame,
    out_fp: str,
):
    timeline_type = build_timeline_by_event_type(df_future_events)
    heat_type = timeline_type.pivot(index="event_type", columns="year", values="n_events")

    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(
        heat_type,
        cmap="Blues",
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Número de eventos"},
    )

    years = list(heat_type.columns)
    step = 2 if len(years) > 10 else 1
    ax.set_xticks(np.arange(0.5, len(years), step))
    ax.set_xticklabels(years[::step], rotation=45)

    plt.title("Frecuencia anual de eventos extremos por tipo")
    plt.xlabel("Año")
    plt.ylabel("Tipo de evento")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()


def plot_station_counts(
    df_future_events: pd.DataFrame,
    out_fp: str,
):
    station_counts = build_station_counts(df_future_events)

    plt.figure(figsize=(8, 5))
    station_counts.plot(kind="barh")

    plt.title("Eventos extremos proyectados por estación")
    plt.xlabel("Número de eventos")
    plt.ylabel("Estación")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()


def plot_month_distribution(
    df_future_events: pd.DataFrame,
    out_fp: str,
):
    month_counts = build_month_counts(df_future_events)

    plt.figure(figsize=(8, 4))
    month_counts.plot(kind="bar")

    plt.title("Distribución mensual de eventos extremos")
    plt.xlabel("Mes")
    plt.ylabel("Número de eventos")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()
