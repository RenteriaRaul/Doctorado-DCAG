from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sustax.projection import (
    build_climate_change_projection,
)


# =========================================================
# CONFIGURACIÓN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BYSCEN_FOLDER = (
    PROJECT_ROOT
    / "results"
    / "tests"
    / "sustax_by_scenario"
)

OUT_FOLDER = (
    PROJECT_ROOT
    / "results"
    / "tests"
    / "sustax_climate_projection"
)

SSP_LIST = [
    "SSP245",
    "SSP370",
]

HISTORICAL_PERIOD = (1981, 2000)
FUTURE_PERIOD = (2011, 2030)

THRESHOLD_MM = 50.0


# =========================================================
# INICIO
# =========================================================

print("=" * 80)
print("PRUEBA — PROYECCIONES DE CAMBIO CLIMÁTICO SUSTAX")
print("=" * 80)

print("\n1. CONFIGURACIÓN")
print("-" * 80)

print(f"BY_SCENARIO:       {BYSCEN_FOLDER}")
print(f"Salida:            {OUT_FOLDER}")
print(
    f"Periodo histórico: "
    f"{HISTORICAL_PERIOD[0]}–{HISTORICAL_PERIOD[1]}"
)
print(
    f"Periodo futuro:    "
    f"{FUTURE_PERIOD[0]}–{FUTURE_PERIOD[1]}"
)
print(f"Umbral R50mm:      {THRESHOLD_MM:.1f} mm")
print(f"Escenarios:        {', '.join(SSP_LIST)}")

if not BYSCEN_FOLDER.exists():
    raise FileNotFoundError(
        f"No existe la carpeta:\n{BYSCEN_FOLDER}"
    )


# =========================================================
# EJECUCIÓN
# =========================================================

print("\n2. EJECUCIÓN DEL MOTOR")
print("-" * 80)

(
    df_projection,
    table_scenario_signal,
    table_point_signal,
) = build_climate_change_projection(
    byscen_total_folder=str(BYSCEN_FOLDER),
    out_folder=str(OUT_FOLDER),
    ssp_list=SSP_LIST,
    historical_period=HISTORICAL_PERIOD,
    future_period=FUTURE_PERIOD,
    threshold_mm=THRESHOLD_MM,
    require_complete_period=True,
)

print("Motor ejecutado correctamente.")
print(f"Resultados generados: {len(df_projection)}")


# =========================================================
# ESTRUCTURA GENERAL
# =========================================================

print("\n3. ESTRUCTURA GENERAL")
print("-" * 80)

print(
    df_projection[
        [
            "sustax_total",
            "scenario",
            "hist_available_years",
            "future_available_years",
            "hist_coverage_pct",
            "future_coverage_pct",
            "hist_complete",
            "future_complete",
            "change_computed",
        ]
    ]
    .sort_values(["scenario", "sustax_total"])
    .to_string(index=False)
)


# =========================================================
# VALIDACIÓN DE COBERTURA
# =========================================================

print("\n4. VALIDACIÓN DE COBERTURA")
print("-" * 80)

expected_years_hist = (
    HISTORICAL_PERIOD[1]
    - HISTORICAL_PERIOD[0]
    + 1
)

expected_years_future = (
    FUTURE_PERIOD[1]
    - FUTURE_PERIOD[0]
    + 1
)

problems = []

for _, row in df_projection.iterrows():

    if row["hist_available_years"] != expected_years_hist:
        problems.append(
            f"{row['sustax_total']} "
            f"{row['scenario']}: "
            "histórico incompleto"
        )

    if row["future_available_years"] != expected_years_future:
        problems.append(
            f"{row['sustax_total']} "
            f"{row['scenario']}: "
            "futuro incompleto"
        )

    if not bool(row["hist_complete"]):
        problems.append(
            f"{row['sustax_total']} "
            f"{row['scenario']}: "
            "hist_complete=False"
        )

    if not bool(row["future_complete"]):
        problems.append(
            f"{row['sustax_total']} "
            f"{row['scenario']}: "
            "future_complete=False"
        )

    if not bool(row["change_computed"]):
        problems.append(
            f"{row['sustax_total']} "
            f"{row['scenario']}: "
            "cambio no calculado"
        )

if problems:
    print("ADVERTENCIAS:")
    for problem in problems:
        print(f"  - {problem}")
else:
    print(
        "OK: todos los puntos tienen "
        "20 años históricos y 20 años futuros."
    )
    print(
        "OK: todos los cambios fueron calculados."
    )


# =========================================================
# INDICADORES
# =========================================================

print("\n5. INDICADORES DE CAMBIO")
print("-" * 80)

indicator_cols = [
    "sustax_total",
    "scenario",

    "hist_p95_daily_mm",
    "future_p95_daily_mm",
    "pct_change_p95_daily_mm",

    "hist_p99_daily_mm",
    "future_p99_daily_mm",
    "pct_change_p99_daily_mm",

    "hist_rx1day_mm",
    "future_rx1day_mm",
    "pct_change_rx1day_mm",

    "hist_r50mm_days_per_year",
    "future_r50mm_days_per_year",
    "pct_change_r50mm_days_per_year",
]

print(
    df_projection[indicator_cols]
    .sort_values(["scenario", "sustax_total"])
    .round(3)
    .to_string(index=False)
)


# =========================================================
# VALIDACIÓN NUMÉRICA
# =========================================================

print("\n6. VALIDACIÓN NUMÉRICA")
print("-" * 80)

numeric_cols = [
    "hist_p95_daily_mm",
    "future_p95_daily_mm",

    "hist_p99_daily_mm",
    "future_p99_daily_mm",

    "hist_rx1day_mm",
    "future_rx1day_mm",

    "hist_r50mm_days_per_year",
    "future_r50mm_days_per_year",
]

invalid_negative = []

for col in numeric_cols:

    values = pd.to_numeric(
        df_projection[col],
        errors="coerce",
    )

    if (values < 0).any():
        invalid_negative.append(col)

if invalid_negative:
    print(
        "ERROR: se encontraron valores negativos en:"
    )
    for col in invalid_negative:
        print(f"  - {col}")
else:
    print(
        "OK: no existen valores negativos "
        "en los indicadores físicos."
    )


# =========================================================
# VALIDACIÓN DE CAMBIOS
# =========================================================

print("\n7. COMPROBACIÓN DE Δ Y %")
print("-" * 80)

metric_pairs = [
    (
        "p95_daily_mm",
        "hist_p95_daily_mm",
        "future_p95_daily_mm",
    ),
    (
        "p99_daily_mm",
        "hist_p99_daily_mm",
        "future_p99_daily_mm",
    ),
    (
        "rx1day_mm",
        "hist_rx1day_mm",
        "future_rx1day_mm",
    ),
    (
        "r50mm_days_per_year",
        "hist_r50mm_days_per_year",
        "future_r50mm_days_per_year",
    ),
]

calculation_errors = []

for _, row in df_projection.iterrows():

    for metric, hist_col, future_col in metric_pairs:

        hist = row[hist_col]
        future = row[future_col]

        delta_saved = row[f"delta_{metric}"]
        pct_saved = row[f"pct_change_{metric}"]

        if pd.isna(hist) or pd.isna(future):
            continue

        delta_expected = future - hist

        if not np.isclose(
            delta_saved,
            delta_expected,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ):
            calculation_errors.append(
                f"{row['sustax_total']} "
                f"{row['scenario']} "
                f"{metric}: Δ incorrecto"
            )

        if hist != 0:

            pct_expected = (
                100.0
                * (future - hist)
                / hist
            )

            if not np.isclose(
                pct_saved,
                pct_expected,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            ):
                calculation_errors.append(
                    f"{row['sustax_total']} "
                    f"{row['scenario']} "
                    f"{metric}: % incorrecto"
                )

if calculation_errors:

    print("ERROR EN CÁLCULOS:")

    for error in calculation_errors:
        print(f"  - {error}")

else:
    print(
        "OK: todos los cambios absolutos "
        "y porcentuales fueron verificados."
    )


# =========================================================
# RESUMEN POR ESCENARIO
# =========================================================

print("\n8. SEÑAL PROMEDIO POR ESCENARIO")
print("-" * 80)

print(
    table_scenario_signal
    .round(3)
    .to_string(index=False)
)


# =========================================================
# RESUMEN POR PUNTO
# =========================================================

print("\n9. SEÑAL PROMEDIO POR PUNTO")
print("-" * 80)

print(
    table_point_signal
    .round(3)
    .to_string(index=False)
)


# =========================================================
# ARCHIVOS GENERADOS
# =========================================================

print("\n10. ARCHIVOS GENERADOS")
print("-" * 80)

expected_files = [
    OUT_FOLDER / "climate_change_projection.csv",
    OUT_FOLDER / "climate_change_signal_by_scenario.csv",
    OUT_FOLDER / "climate_change_signal_by_point.csv",
]

missing_files = []

for fp in expected_files:

    if fp.exists():
        print(f"OK: {fp.name}")
    else:
        print(f"FALTA: {fp.name}")
        missing_files.append(fp.name)


# =========================================================
# RESULTADO FINAL
# =========================================================

print("\n" + "=" * 80)

if (
    not problems
    and not invalid_negative
    and not calculation_errors
    and not missing_files
):
    print("PRUEBA SUPERADA")
    print(
        "El motor de cambio climático Sustax "
        "funciona correctamente."
    )
else:
    print("PRUEBA FINALIZADA CON ADVERTENCIAS")

print("=" * 80)