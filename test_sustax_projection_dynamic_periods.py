from pathlib import Path

from scripts.sustax.projection import (
    detect_projection_coverage,
    summarize_projection_coverage,
    evaluate_projection_period,
    get_compatible_scenarios,
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

SCENARIOS = [
    "SSP119",
    "SSP126",
    "SSP245",
    "SSP370",
    "SSP434",
    "SSP460",
    "SSP585",
]


print("=" * 80)
print("PRUEBA — CONTROL DINÁMICO DE PERIODOS SUSTAX")
print("=" * 80)


# =========================================================
# 1. COBERTURA
# =========================================================

coverage = detect_projection_coverage(
    byscen_total_folder=str(BYSCEN_FOLDER),
    ssp_list=SCENARIOS,
)

summary = summarize_projection_coverage(
    coverage
)

print("\n1. COBERTURA COMÚN POR ESCENARIO")
print("-" * 80)

print(
    summary.to_string(index=False)
)


# =========================================================
# 2. PERIODO QUE SABEMOS QUE DEBE FUNCIONAR
# =========================================================

print("\n2. PRUEBA — PERIODOS COMPLETOS")
print("-" * 80)

hist_test = (1981, 2000)
future_test = (2011, 2030)

compatible_test = get_compatible_scenarios(
    coverage_summary=summary,
    historical_period=hist_test,
    future_period=future_test,
    require_complete=True,
)

print(
    compatible_test.to_string(index=False)
)


# =========================================================
# 3. HORIZONTE CIENTÍFICO PROPUESTO
# =========================================================

print("\n3. PRUEBA — HORIZONTE 2026–2055")
print("-" * 80)

scientific_test = get_compatible_scenarios(
    coverage_summary=summary,
    historical_period=(1981, 2010),
    future_period=(2026, 2055),
    require_complete=True,
)

print(
    scientific_test.to_string(index=False)
)


# =========================================================
# 4. FUTURO LEJANO
# =========================================================

print("\n4. PRUEBA — HORIZONTE 2051–2080")
print("-" * 80)

far_future = get_compatible_scenarios(
    coverage_summary=summary,
    historical_period=(1981, 2010),
    future_period=(2051, 2080),
    require_complete=True,
)

print(
    far_future.to_string(index=False)
)


# =========================================================
# 5. DETALLE DE UN ESCENARIO
# =========================================================

print("\n5. DETALLE SSP245")
print("-" * 80)

for period in [
    (1981, 2010),
    (2011, 2030),
    (2026, 2055),
    (2051, 2080),
]:

    result = evaluate_projection_period(
        coverage_summary=summary,
        scenario="SSP245",
        period=period,
    )

    print(
        f"{period[0]}–{period[1]} | "
        f"{result['status']} | "
        f"{result['available_years']}/"
        f"{result['expected_years']} años | "
        f"{result['coverage_pct']:.1f}%"
    )


# =========================================================
# 6. VALIDACIONES AUTOMÁTICAS
# =========================================================

print("\n6. VALIDACIÓN")
print("-" * 80)

errors = []


# Debemos encontrar los dos SSP presentes.
found = set(summary["scenario"])

if found != {"SSP245", "SSP370"}:
    errors.append(
        f"Escenarios inesperados: {found}"
    )


# Ambos deben cubrir 1980–2030.
for _, row in summary.iterrows():

    if row["common_start"] != 1980:
        errors.append(
            f"{row['scenario']}: "
            "inicio común distinto de 1980"
        )

    if row["common_end"] != 2030:
        errors.append(
            f"{row['scenario']}: "
            "fin común distinto de 2030"
        )

    if not bool(row["all_continuous"]):
        errors.append(
            f"{row['scenario']}: "
            "serie temporal no continua"
        )


# Periodo técnico debe ser compatible.
if not compatible_test["compatible"].all():
    errors.append(
        "El periodo técnico 1981–2000 / "
        "2011–2030 debería ser compatible."
    )


# 2026–2055 NO debe ser compatible todavía.
if scientific_test["compatible"].any():
    errors.append(
        "2026–2055 no debería aparecer "
        "como completo con estos datos."
    )


# 2051–2080 tampoco.
if far_future["compatible"].any():
    errors.append(
        "2051–2080 no debería estar disponible."
    )


if errors:

    print("PRUEBA CON ERRORES:")

    for error in errors:
        print(f"  - {error}")

else:

    print(
        "OK: cobertura común detectada correctamente."
    )

    print(
        "OK: periodos completos, parciales y "
        "no disponibles fueron diferenciados."
    )

    print(
        "OK: la plataforma puede controlar "
        "dinámicamente los horizontes."
    )


print("\n" + "=" * 80)

if not errors:
    print("PRUEBA SUPERADA")
else:
    print("PRUEBA FINALIZADA CON ERRORES")

print("=" * 80)