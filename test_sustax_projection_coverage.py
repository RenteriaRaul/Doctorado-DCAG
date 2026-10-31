from pathlib import Path

import pandas as pd

from scripts.sustax.projection import (
    get_scenario_files,
    normalize_date_series,
)


# =========================================================
# CONFIGURACIÓN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ajusta SOLO si tu carpeta BY_SCENARIO tiene otro nombre/ruta.
BYSCEN_FOLDER = Path(
    r"G:\My Drive\Doctorado-DCAG\results\tests\sustax_by_scenario"
)

# Escenarios que queremos detectar.
# ERA5 se revisa aparte porque get_scenario_files() acepta
# cualquier lista de escenarios.
SCENARIOS = [
    "ERA5",
    "SSP119",
    "SSP126",
    "SSP245",
    "SSP370",
    "SSP434",
    "SSP460",
    "SSP585",
]


# =========================================================
# HELPERS
# =========================================================

def revisar_archivo(sustax_total: str, scenario: str, fp: str) -> dict:
    """
    Revisa únicamente cobertura temporal y estructura básica.
    No realiza cálculos de proyección.
    """
    df = pd.read_csv(fp)

    if "date" not in df.columns:
        return {
            "sustax_total": sustax_total,
            "scenario": scenario,
            "archivo": Path(fp).name,
            "n_registros": len(df),
            "fecha_ini": pd.NaT,
            "fecha_fin": pd.NaT,
            "anio_ini": pd.NA,
            "anio_fin": pd.NA,
            "n_anios": 0,
            "dias_unicos": 0,
            "duplicados_fecha": pd.NA,
            "estado": "SIN COLUMNA DATE",
        }

    fechas = normalize_date_series(df["date"])
    fechas_validas = fechas.dropna()

    if fechas_validas.empty:
        return {
            "sustax_total": sustax_total,
            "scenario": scenario,
            "archivo": Path(fp).name,
            "n_registros": len(df),
            "fecha_ini": pd.NaT,
            "fecha_fin": pd.NaT,
            "anio_ini": pd.NA,
            "anio_fin": pd.NA,
            "n_anios": 0,
            "dias_unicos": 0,
            "duplicados_fecha": 0,
            "estado": "SIN FECHAS VÁLIDAS",
        }

    fecha_ini = fechas_validas.min()
    fecha_fin = fechas_validas.max()

    return {
        "sustax_total": sustax_total,
        "scenario": scenario,
        "archivo": Path(fp).name,
        "n_registros": len(df),
        "fecha_ini": fecha_ini,
        "fecha_fin": fecha_fin,
        "anio_ini": int(fecha_ini.year),
        "anio_fin": int(fecha_fin.year),
        "n_anios": int(fechas_validas.dt.year.nunique()),
        "dias_unicos": int(fechas_validas.nunique()),
        "duplicados_fecha": int(fechas_validas.duplicated().sum()),
        "estado": "OK",
    }


# =========================================================
# PRUEBA
# =========================================================

print("=" * 80)
print("PRUEBA — COBERTURA TEMPORAL DE PROYECCIONES SUSTAX")
print("=" * 80)

print("\n1. FUENTE")
print("-" * 80)
print(f"Carpeta BY_SCENARIO: {BYSCEN_FOLDER}")

if not BYSCEN_FOLDER.exists():
    raise FileNotFoundError(
        f"No existe la carpeta BY_SCENARIO:\n{BYSCEN_FOLDER}\n\n"
        "Ajusta BYSCEN_FOLDER en el script de prueba."
    )


# ---------------------------------------------------------
# Detectar archivos
# ---------------------------------------------------------

triplets = get_scenario_files(
    str(BYSCEN_FOLDER),
    SCENARIOS,
)

print(f"Archivos detectados: {len(triplets)}")

if not triplets:
    raise RuntimeError(
        "No se detectaron archivos por escenario. "
        "Revisa la ruta y los nombres de los archivos."
    )


# ---------------------------------------------------------
# Revisar cada archivo
# ---------------------------------------------------------

rows = []

for sustax_total, scenario, fp in triplets:
    try:
        rows.append(
            revisar_archivo(
                sustax_total=sustax_total,
                scenario=scenario,
                fp=fp,
            )
        )
    except Exception as e:
        rows.append(
            {
                "sustax_total": sustax_total,
                "scenario": scenario,
                "archivo": Path(fp).name,
                "n_registros": pd.NA,
                "fecha_ini": pd.NaT,
                "fecha_fin": pd.NaT,
                "anio_ini": pd.NA,
                "anio_fin": pd.NA,
                "n_anios": 0,
                "dias_unicos": 0,
                "duplicados_fecha": pd.NA,
                "estado": f"ERROR: {e}",
            }
        )

detalle = pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. Escenarios encontrados
# ---------------------------------------------------------

print("\n2. ESCENARIOS ENCONTRADOS")
print("-" * 80)

escenarios_encontrados = sorted(detalle["scenario"].dropna().unique())

for scenario in escenarios_encontrados:
    n = detalle.loc[
        detalle["scenario"] == scenario,
        "sustax_total"
    ].nunique()

    print(f"{scenario}: {n} punto(s)")


# ---------------------------------------------------------
# 3. Resumen temporal por escenario
# ---------------------------------------------------------

print("\n3. COBERTURA TEMPORAL POR ESCENARIO")
print("-" * 80)

ok = detalle[detalle["estado"] == "OK"].copy()

resumen = (
    ok.groupby("scenario")
    .agg(
        puntos=("sustax_total", "nunique"),
        archivos=("archivo", "count"),
        fecha_mas_antigua=("fecha_ini", "min"),
        fecha_mas_reciente=("fecha_fin", "max"),
        anio_min=("anio_ini", "min"),
        anio_max=("anio_fin", "max"),
        dias_min=("dias_unicos", "min"),
        dias_max=("dias_unicos", "max"),
    )
    .reset_index()
    .sort_values("scenario")
)

print(resumen.to_string(index=False))


# ---------------------------------------------------------
# 4. Cobertura individual
# ---------------------------------------------------------

print("\n4. COBERTURA POR PUNTO Y ESCENARIO")
print("-" * 80)

cols = [
    "sustax_total",
    "scenario",
    "anio_ini",
    "anio_fin",
    "n_anios",
    "dias_unicos",
    "duplicados_fecha",
    "estado",
]

print(
    detalle[cols]
    .sort_values(["scenario", "sustax_total"])
    .to_string(index=False)
)


# ---------------------------------------------------------
# 5. Homogeneidad temporal dentro de cada escenario
# ---------------------------------------------------------

print("\n5. HOMOGENEIDAD TEMPORAL")
print("-" * 80)

for scenario in escenarios_encontrados:

    sub = ok[ok["scenario"] == scenario]

    if sub.empty:
        print(f"{scenario}: sin archivos válidos.")
        continue

    periodos = (
        sub[["anio_ini", "anio_fin"]]
        .drop_duplicates()
        .sort_values(["anio_ini", "anio_fin"])
    )

    if len(periodos) == 1:
        row = periodos.iloc[0]
        print(
            f"{scenario}: OK — todos los puntos cubren "
            f"{int(row['anio_ini'])}–{int(row['anio_fin'])}"
        )
    else:
        print(
            f"{scenario}: ADVERTENCIA — existen "
            f"{len(periodos)} coberturas temporales distintas:"
        )
        print(periodos.to_string(index=False))


# ---------------------------------------------------------
# 6. Periodos del projection.py
# ---------------------------------------------------------

print("\n6. COMPATIBILIDAD CON PERIODOS ACTUALES DE projection.py")
print("-" * 80)

periodos_projection = [
    ("Histórico", 1980, 2014),
    ("Futuro 1", 2015, 2049),
    ("Futuro 2", 2050, 2080),
    ("Proyección anual", 2027, 2040),
]

for scenario in escenarios_encontrados:

    sub = ok[ok["scenario"] == scenario]

    if sub.empty:
        continue

    ini_comun = int(sub["anio_ini"].max())
    fin_comun = int(sub["anio_fin"].min())

    print(f"\n{scenario}")
    print(f"Periodo común entre sus puntos: {ini_comun}–{fin_comun}")

    for nombre, y0, y1 in periodos_projection:

        if ini_comun <= y0 and fin_comun >= y1:
            estado = "COMPLETO"

        elif fin_comun < y0 or ini_comun > y1:
            estado = "NO DISPONIBLE"

        else:
            real_ini = max(ini_comun, y0)
            real_fin = min(fin_comun, y1)

            estado = (
                f"PARCIAL ({real_ini}–{real_fin})"
            )

        print(
            f"  {nombre:18s} "
            f"{y0}–{y1}: {estado}"
        )


# ---------------------------------------------------------
# 7. Problemas estructurales
# ---------------------------------------------------------

print("\n7. VALIDACIÓN")
print("-" * 80)

errores = detalle[detalle["estado"] != "OK"]

duplicados = detalle[
    pd.to_numeric(
        detalle["duplicados_fecha"],
        errors="coerce"
    ).fillna(0) > 0
]

if errores.empty:
    print("OK: todos los archivos tienen fechas válidas.")
else:
    print(
        f"ADVERTENCIA: {len(errores)} archivo(s) "
        "presentan problemas."
    )
    print(
        errores[
            ["archivo", "scenario", "estado"]
        ].to_string(index=False)
    )

if duplicados.empty:
    print("OK: no se detectaron fechas duplicadas.")
else:
    print(
        f"ADVERTENCIA: {len(duplicados)} archivo(s) "
        "tienen fechas duplicadas."
    )
    print(
        duplicados[
            [
                "archivo",
                "scenario",
                "duplicados_fecha",
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 80)
print("PRUEBA DE COBERTURA FINALIZADA")
print("=" * 80)