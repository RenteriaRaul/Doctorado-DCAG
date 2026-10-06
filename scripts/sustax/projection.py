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
# PROYECCIONES DE CAMBIO CLIMÁTICO — SSP
# =========================================================

def _validar_periodo(
    periodo: tuple[int, int],
    nombre: str,
) -> tuple[int, int]:
    """
    Valida un periodo expresado como (año_inicial, año_final).
    """
    if len(periodo) != 2:
        raise ValueError(
            f"{nombre} debe tener exactamente dos años: "
            "(inicio, fin)."
        )

    y0, y1 = int(periodo[0]), int(periodo[1])

    if y0 > y1:
        raise ValueError(
            f"Periodo inválido para {nombre}: {y0}–{y1}."
        )

    return y0, y1


def _preparar_serie_proyeccion(
    fp: str,
) -> tuple[pd.DataFrame, str]:
    """
    Lee y normaliza una serie diaria BY_SCENARIO.

    La precipitación se conserva en las unidades del archivo
    BY_SCENARIO, que en nuestro flujo ya corresponden a mm.
    """
    df = pd.read_csv(fp)

    if "date" not in df.columns:
        raise ValueError(
            f"El archivo no contiene columna date: {fp}"
        )

    df["date"] = normalize_date_series(df["date"])
    df = df.dropna(subset=["date"]).copy()

    pp_col = pick_precip_col(df)

    df[pp_col] = pd.to_numeric(
        df[pp_col],
        errors="coerce",
    )

    # La precipitación negativa no tiene interpretación física.
    df.loc[df[pp_col] < 0, pp_col] = np.nan

    df = (
        df.dropna(subset=[pp_col])
        .sort_values("date")
        .reset_index(drop=True)
    )

    df["year"] = df["date"].dt.year

    return df, pp_col


def _resumir_extremos_periodo(
    df: pd.DataFrame,
    value_col: str,
    y0: int,
    y1: int,
    threshold_mm: float = 50.0,
) -> dict:
    """
    Resume indicadores de precipitación extrema para un periodo.

    Indicadores:
    - P95 diario
    - P99 diario
    - Rx1day: promedio de los máximos diarios anuales
    - R50mm: promedio anual de días con PP >= threshold_mm

    También conserva métricas de cobertura para evitar interpretar
    periodos incompletos como climatologías completas.
    """
    sub = df[
        (df["year"] >= y0)
        & (df["year"] <= y1)
    ].copy()

    expected_years = y1 - y0 + 1

    if sub.empty:
        return {
            "period_start": y0,
            "period_end": y1,
            "expected_years": expected_years,
            "available_years": 0,
            "coverage_pct": 0.0,
            "n_days": 0,
            "p95_daily_mm": np.nan,
            "p99_daily_mm": np.nan,
            "rx1day_mm": np.nan,
            "r50mm_days_per_year": np.nan,
        }

    vals = pd.to_numeric(
        sub[value_col],
        errors="coerce",
    )

    valid = sub.loc[vals.notna()].copy()
    valid[value_col] = vals.loc[vals.notna()]

    if valid.empty:
        return {
            "period_start": y0,
            "period_end": y1,
            "expected_years": expected_years,
            "available_years": 0,
            "coverage_pct": 0.0,
            "n_days": 0,
            "p95_daily_mm": np.nan,
            "p99_daily_mm": np.nan,
            "rx1day_mm": np.nan,
            "r50mm_days_per_year": np.nan,
        }

    available_years = int(
        valid["year"].nunique()
    )

    coverage_pct = (
        100.0
        * available_years
        / expected_years
        if expected_years > 0
        else np.nan
    )

    # Percentiles diarios del periodo completo.
    p95 = float(
        valid[value_col].quantile(0.95)
    )

    p99 = float(
        valid[value_col].quantile(0.99)
    )

    # Rx1day:
    # máximo diario de cada año y posteriormente
    # promedio de esos máximos anuales.
    annual_rx1day = (
        valid.groupby("year")[value_col]
        .max()
    )

    rx1day = (
        float(annual_rx1day.mean())
        if len(annual_rx1day)
        else np.nan
    )

    # R50mm:
    # número de días >= umbral para cada año,
    # seguido de la frecuencia anual promedio.
    annual_r50 = (
        valid.assign(
            exceed=valid[value_col] >= threshold_mm
        )
        .groupby("year")["exceed"]
        .sum()
    )

    r50 = (
        float(annual_r50.mean())
        if len(annual_r50)
        else np.nan
    )

    return {
        "period_start": y0,
        "period_end": y1,
        "expected_years": expected_years,
        "available_years": available_years,
        "coverage_pct": coverage_pct,
        "n_days": int(len(valid)),
        "p95_daily_mm": p95,
        "p99_daily_mm": p99,
        "rx1day_mm": rx1day,
        "r50mm_days_per_year": r50,
    }

# =========================================================
# COBERTURA TEMPORAL PARA PROYECCIONES
# =========================================================

def detect_projection_coverage(
    byscen_total_folder: str,
    ssp_list: list[str],
) -> pd.DataFrame:
    """
    Detecta la cobertura temporal real de cada punto y escenario
    disponible en BY_SCENARIO.

    No calcula indicadores climáticos. Su función es describir
    qué años están realmente disponibles antes de ejecutar una
    proyección.

    Retorna una fila por punto y escenario.
    """
    triplets = get_scenario_files(
        byscen_total_folder,
        ssp_list,
    )

    rows = []

    for sustax_total, scenario, fp in triplets:

        try:
            df, _ = _preparar_serie_proyeccion(fp)

            if df.empty:
                continue

            year_min = int(df["year"].min())
            year_max = int(df["year"].max())

            years_present = sorted(
                df["year"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            expected_years = set(
                range(year_min, year_max + 1)
            )

            missing_years = sorted(
                expected_years - set(years_present)
            )

            rows.append(
                {
                    "sustax_total": sustax_total,
                    "scenario": scenario,
                    "year_min": year_min,
                    "year_max": year_max,
                    "n_years": len(years_present),
                    "n_missing_years": len(missing_years),
                    "missing_years": missing_years,
                    "continuous": len(missing_years) == 0,
                    "file_path": str(fp),
                }
            )

        except Exception as e:
            print(
                f"⚠️ No fue posible revisar cobertura de "
                f"{sustax_total} - {scenario}: {e}"
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sustax_total",
                "scenario",
                "year_min",
                "year_max",
                "n_years",
                "n_missing_years",
                "missing_years",
                "continuous",
                "file_path",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["scenario", "sustax_total"]
        )
        .reset_index(drop=True)
    )


def summarize_projection_coverage(
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Obtiene la cobertura COMÚN entre todos los puntos de cada
    escenario.

    common_start:
        año inicial más restrictivo entre los puntos.

    common_end:
        año final más restrictivo entre los puntos.

    Esta cobertura común es la que debe utilizar Streamlit para
    permitir horizontes comparables espacialmente.
    """
    if coverage_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "n_points",
                "common_start",
                "common_end",
                "common_years",
                "all_continuous",
            ]
        )

    rows = []

    for scenario, group in coverage_df.groupby("scenario"):

        common_start = int(
            group["year_min"].max()
        )

        common_end = int(
            group["year_max"].min()
        )

        if common_end >= common_start:
            common_years = (
                common_end - common_start + 1
            )
        else:
            common_years = 0

        rows.append(
            {
                "scenario": scenario,
                "n_points": int(
                    group["sustax_total"].nunique()
                ),
                "common_start": common_start,
                "common_end": common_end,
                "common_years": common_years,
                "all_continuous": bool(
                    group["continuous"].all()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("scenario")
        .reset_index(drop=True)
    )


def evaluate_projection_period(
    coverage_summary: pd.DataFrame,
    scenario: str,
    period: tuple[int, int],
) -> dict:
    """
    Evalúa si un periodo solicitado está completamente contenido
    en la cobertura común de un escenario.

    Esta función será utilizada posteriormente por Streamlit para
    habilitar, advertir o bloquear horizontes.
    """
    y0, y1 = _validar_periodo(
        period,
        "period",
    )

    row = coverage_summary[
        coverage_summary["scenario"] == scenario
    ]

    if row.empty:
        return {
            "scenario": scenario,
            "period_start": y0,
            "period_end": y1,
            "status": "NOT_AVAILABLE",
            "available_start": None,
            "available_end": None,
            "overlap_start": None,
            "overlap_end": None,
            "available_years": 0,
            "expected_years": y1 - y0 + 1,
            "coverage_pct": 0.0,
        }

    row = row.iloc[0]

    available_start = int(
        row["common_start"]
    )

    available_end = int(
        row["common_end"]
    )

    expected_years = y1 - y0 + 1

    overlap_start = max(
        y0,
        available_start,
    )

    overlap_end = min(
        y1,
        available_end,
    )

    if overlap_end < overlap_start:

        available_years = 0
        coverage_pct = 0.0
        status = "NOT_AVAILABLE"

        overlap_start_out = None
        overlap_end_out = None

    else:

        available_years = (
            overlap_end
            - overlap_start
            + 1
        )

        coverage_pct = (
            100.0
            * available_years
            / expected_years
        )

        overlap_start_out = overlap_start
        overlap_end_out = overlap_end

        if (
            overlap_start == y0
            and overlap_end == y1
        ):
            status = "COMPLETE"
        else:
            status = "PARTIAL"

    return {
        "scenario": scenario,
        "period_start": y0,
        "period_end": y1,
        "status": status,
        "available_start": available_start,
        "available_end": available_end,
        "overlap_start": overlap_start_out,
        "overlap_end": overlap_end_out,
        "available_years": available_years,
        "expected_years": expected_years,
        "coverage_pct": coverage_pct,
    }


def get_compatible_scenarios(
    coverage_summary: pd.DataFrame,
    historical_period: tuple[int, int],
    future_period: tuple[int, int],
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    Evalúa todos los escenarios y determina cuáles son compatibles
    con los periodos histórico y futuro solicitados.

    Si require_complete=True, ambos periodos deben tener cobertura
    completa.
    """
    rows = []

    if coverage_summary.empty:
        return pd.DataFrame()

    for scenario in coverage_summary["scenario"]:

        hist = evaluate_projection_period(
            coverage_summary,
            scenario,
            historical_period,
        )

        future = evaluate_projection_period(
            coverage_summary,
            scenario,
            future_period,
        )

        if require_complete:
            compatible = (
                hist["status"] == "COMPLETE"
                and future["status"] == "COMPLETE"
            )
        else:
            compatible = (
                hist["status"] != "NOT_AVAILABLE"
                and future["status"] != "NOT_AVAILABLE"
            )

        rows.append(
            {
                "scenario": scenario,

                "historical_status":
                    hist["status"],

                "historical_coverage_pct":
                    hist["coverage_pct"],

                "future_status":
                    future["status"],

                "future_coverage_pct":
                    future["coverage_pct"],

                "compatible":
                    compatible,
            }
        )

    return pd.DataFrame(rows)

def build_climate_change_projection(
    byscen_total_folder: str,
    out_folder: str,
    ssp_list: list[str],
    historical_period: tuple[int, int] = (1981, 2010),
    future_period: tuple[int, int] = (2026, 2055),
    threshold_mm: float = 50.0,
    require_complete_period: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula la señal de cambio climático de precipitación
    para cada punto Sustax y escenario SSP.

    El periodo futuro se compara contra el periodo histórico
    DEL MISMO SSP. ERA5 no se utiliza para calcular la anomalía
    SSP, evitando mezclar directamente productos distintos.

    Parámetros
    ----------
    byscen_total_folder:
        Carpeta con archivos BY_SCENARIO.

    out_folder:
        Carpeta de resultados.

    ssp_list:
        Escenarios SSP a procesar.

    historical_period:
        Periodo histórico de referencia.

    future_period:
        Horizonte futuro.

    threshold_mm:
        Umbral de precipitación para R50mm.

    require_complete_period:
        Si True, solo calcula cambios cuando ambos periodos
        tienen todos los años solicitados.

        Si False, calcula con la cobertura disponible y deja
        registrada explícitamente dicha cobertura.

    Retorna
    -------
    df_projection:
        Resultado por punto y escenario.

    table_scenario_signal:
        Promedio espacial de la señal por escenario.

    table_point_signal:
        Promedio entre escenarios por punto.
    """
    os.makedirs(out_folder, exist_ok=True)

    hist_start, hist_end = _validar_periodo(
        historical_period,
        "historical_period",
    )

    fut_start, fut_end = _validar_periodo(
        future_period,
        "future_period",
    )

    if threshold_mm <= 0:
        raise ValueError(
            "threshold_mm debe ser mayor que cero."
        )

    triplets = get_scenario_files(
        byscen_total_folder,
        ssp_list,
    )

    rows = []

    for sustax_total, scenario, fp in triplets:

        try:
            df, pp_col = _preparar_serie_proyeccion(fp)

            hist = _resumir_extremos_periodo(
                df=df,
                value_col=pp_col,
                y0=hist_start,
                y1=hist_end,
                threshold_mm=threshold_mm,
            )

            future = _resumir_extremos_periodo(
                df=df,
                value_col=pp_col,
                y0=fut_start,
                y1=fut_end,
                threshold_mm=threshold_mm,
            )

            hist_complete = (
                hist["available_years"]
                == hist["expected_years"]
            )

            future_complete = (
                future["available_years"]
                == future["expected_years"]
            )

            can_compute = (
                hist["available_years"] > 0
                and future["available_years"] > 0
            )

            if require_complete_period:
                can_compute = (
                    can_compute
                    and hist_complete
                    and future_complete
                )

            metrics = [
                "p95_daily_mm",
                "p99_daily_mm",
                "rx1day_mm",
                "r50mm_days_per_year",
            ]

            changes = {}

            for metric in metrics:

                if can_compute:
                    delta, pct = compute_change(
                        future[metric],
                        hist[metric],
                    )
                else:
                    delta, pct = np.nan, np.nan

                changes[
                    f"delta_{metric}"
                ] = delta

                changes[
                    f"pct_change_{metric}"
                ] = pct

            rows.append(
                {
                    "sustax_total": sustax_total,
                    "scenario": scenario,

                    "hist_start": hist_start,
                    "hist_end": hist_end,
                    "future_start": fut_start,
                    "future_end": fut_end,

                    "threshold_mm": threshold_mm,

                    "hist_expected_years":
                        hist["expected_years"],

                    "hist_available_years":
                        hist["available_years"],

                    "hist_coverage_pct":
                        hist["coverage_pct"],

                    "future_expected_years":
                        future["expected_years"],

                    "future_available_years":
                        future["available_years"],

                    "future_coverage_pct":
                        future["coverage_pct"],

                    "hist_complete":
                        hist_complete,

                    "future_complete":
                        future_complete,

                    "change_computed":
                        can_compute,

                    "hist_p95_daily_mm":
                        hist["p95_daily_mm"],

                    "future_p95_daily_mm":
                        future["p95_daily_mm"],

                    "hist_p99_daily_mm":
                        hist["p99_daily_mm"],

                    "future_p99_daily_mm":
                        future["p99_daily_mm"],

                    "hist_rx1day_mm":
                        hist["rx1day_mm"],

                    "future_rx1day_mm":
                        future["rx1day_mm"],

                    "hist_r50mm_days_per_year":
                        hist["r50mm_days_per_year"],

                    "future_r50mm_days_per_year":
                        future["r50mm_days_per_year"],

                    **changes,
                }
            )

        except Exception as e:
            print(
                f"⚠️ Error en "
                f"{sustax_total} - {scenario}: {e}"
            )

    df_projection = pd.DataFrame(rows)

    if df_projection.empty:
        raise ValueError(
            "No fue posible generar resultados de "
            "proyección climática."
        )

    df_projection = (
        df_projection
        .sort_values(
            ["sustax_total", "scenario"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Resumen por escenario
    # -----------------------------------------------------

    change_cols = [
        "pct_change_p95_daily_mm",
        "pct_change_p99_daily_mm",
        "pct_change_rx1day_mm",
        "pct_change_r50mm_days_per_year",
    ]

    table_scenario_signal = (
        df_projection
        .groupby("scenario")[change_cols]
        .mean()
        .reset_index()
    )

    # -----------------------------------------------------
    # Resumen por punto
    # -----------------------------------------------------

    table_point_signal = (
        df_projection
        .groupby("sustax_total")[change_cols]
        .mean()
        .reset_index()
    )

    # -----------------------------------------------------
    # Guardado
    # -----------------------------------------------------

    df_projection.to_csv(
        os.path.join(
            out_folder,
            "climate_change_projection.csv",
        ),
        index=False,
    )

    table_scenario_signal.to_csv(
        os.path.join(
            out_folder,
            "climate_change_signal_by_scenario.csv",
        ),
        index=False,
    )

    table_point_signal.to_csv(
        os.path.join(
            out_folder,
            "climate_change_signal_by_point.csv",
        ),
        index=False,
    )

    return (
        df_projection,
        table_scenario_signal,
        table_point_signal,
    )
    
    
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
