
from pathlib import Path
import json
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from scripts.sustax.inventory import build_sustax_inventory
from scripts.sustax.loader import load_sustax_file_safe
from scripts.sustax.merge import split_var_scenario

# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_ORIGEN = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\Datos Sustax"
)

CARPETA_SALIDA = Path("results/tests/sustax_consolidacion")
DECIMALES = 4
TOLERANCIA_MM = 1e-6

ESCENARIOS_VALIDOS = {
    "ERA5", "SSP119", "SSP126", "SSP245",
    "SSP370", "SSP434", "SSP460", "SSP585"
}


# ============================================================
# IDENTIFICADORES
# ============================================================

def crear_punto_id(lat, lon):
    return f"{float(lat):.{DECIMALES}f}_{float(lon):.{DECIMALES}f}"


def coordenadas_validas(lat, lon):
    return (
        pd.notna(lat)
        and pd.notna(lon)
        and -90 <= float(lat) <= 90
        and -180 <= float(lon) <= 180
    )


# ============================================================
# INVENTARIO ESPACIAL
# ============================================================

def preparar_inventario(carpeta):
    inventario, errores = build_sustax_inventory(str(carpeta))

    if inventario.empty:
        raise ValueError("No se encontraron archivos Sustax compatibles.")

    inventario = inventario.copy()

    inventario["lat"] = pd.to_numeric(
        inventario["lat"], errors="coerce"
    ).round(DECIMALES)

    inventario["lon"] = pd.to_numeric(
        inventario["lon"], errors="coerce"
    ).round(DECIMALES)

    validos = inventario.apply(
        lambda r: coordenadas_validas(r["lat"], r["lon"]),
        axis=1
    )

    invalidos = inventario.loc[~validos].copy()
    inventario = inventario.loc[validos].copy()

    inventario["punto_id"] = inventario.apply(
        lambda r: crear_punto_id(r["lat"], r["lon"]),
        axis=1
    )

    return inventario, errores, invalidos


# ============================================================
# LECTURA DE OBSERVACIONES
# ============================================================

def leer_observaciones(inventario):
    partes = []
    errores = []

    for _, registro in inventario.iterrows():
        try:
            datos, _, _ = load_sustax_file_safe(
                registro["path"],
                return_pandas_df=True,
                return_metadata=True
            )

            datos = datos.copy()
            datos.index = pd.to_datetime(datos.index, errors="coerce")
            datos = datos.loc[~datos.index.isna()]
            datos.index = datos.index.normalize()

            for columna in datos.columns:
                variable, escenario = split_var_scenario(columna)

                if not variable or not escenario:
                    continue

                escenario = escenario.upper()

                if escenario not in ESCENARIOS_VALIDOS:
                    continue

                valores = pd.to_numeric(
                    datos[columna], errors="coerce"
                ) * 1000.0

                parte = pd.DataFrame({
                    "punto_id": registro["punto_id"],
                    "lat": registro["lat"],
                    "lon": registro["lon"],
                    "date": datos.index,
                    "variable": variable,
                    "scenario": escenario,
                    "value_mm": valores.to_numpy(),
                    "source_file": registro["sustax_file"]
                })

                partes.append(parte)

        except Exception as exc:
            errores.append({
                "archivo": registro["sustax_file"],
                "error": str(exc)
            })

    if not partes:
        raise ValueError("No se obtuvieron observaciones válidas.")

    observaciones = pd.concat(partes, ignore_index=True)

    observaciones = observaciones.dropna(
        subset=["date", "value_mm"]
    ).reset_index(drop=True)

    return observaciones, pd.DataFrame(errores)


# ============================================================
# DETECCIÓN DE DUPLICADOS Y CONFLICTOS
# ============================================================

def consolidar_observaciones(observaciones):
    claves = ["punto_id", "scenario", "variable", "date"]

    diagnostico = (
        observaciones.groupby(claves, as_index=False)
        .agg(
            n_registros=("value_mm", "size"),
            minimo_mm=("value_mm", "min"),
            maximo_mm=("value_mm", "max"),
            archivos=("source_file", lambda x: " | ".join(
                sorted(set(x.astype(str)))
            ))
        )
    )

    diagnostico["diferencia_mm"] = (
        diagnostico["maximo_mm"] - diagnostico["minimo_mm"]
    )

    diagnostico["estado"] = np.select(
        [
            diagnostico["diferencia_mm"] > TOLERANCIA_MM,
            diagnostico["n_registros"] > 1
        ],
        ["CONFLICTO", "DUPLICADO_IDENTICO"],
        default="UNICO"
    )

    conflictos = diagnostico.loc[
        diagnostico["estado"] == "CONFLICTO"
    ].copy()

    duplicados = diagnostico.loc[
        diagnostico["estado"] == "DUPLICADO_IDENTICO"
    ].copy()

    # Se bloquea toda la serie punto/escenario/variable
    # cuando existe un conflicto en alguna fecha.
    series_conflictivas = conflictos[
        ["punto_id", "scenario", "variable"]
    ].drop_duplicates()

    if not series_conflictivas.empty:
        series_conflictivas = series_conflictivas.assign(
            bloqueada=True
        )

        observaciones = observaciones.merge(
            series_conflictivas,
            on=["punto_id", "scenario", "variable"],
            how="left"
        )

        observaciones = observaciones.loc[
            observaciones["bloqueada"].isna()
        ].drop(columns="bloqueada")

    depuradas = (
        observaciones.sort_values(
            ["punto_id", "scenario", "variable", "date", "source_file"]
        )
        .drop_duplicates(subset=claves, keep="first")
        .reset_index(drop=True)
    )

    return depuradas, diagnostico, duplicados, conflictos


# ============================================================
# GENERACIÓN LONG
# ============================================================

def generar_long(depuradas, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = []

    for punto_id, grupo in depuradas.groupby("punto_id"):
        sustax_total = f"{punto_id}Total"

        salida = grupo[
            ["lat", "lon", "source_file", "date",
             "variable", "scenario", "value_mm"]
        ].copy()

        salida.insert(0, "sustax_total", sustax_total)
        salida["date"] = salida["date"].dt.strftime("%Y-%m-%d")

        archivo = carpeta / f"{sustax_total}_LONG_mm.csv"
        salida.to_csv(archivo, index=False)

        resumen.append({
            "sustax_total": sustax_total,
            "lat": grupo["lat"].iloc[0],
            "lon": grupo["lon"].iloc[0],
            "n_rows_merged": len(salida),
            "scenarios_merged": ", ".join(
                sorted(grupo["scenario"].unique())
            ),
            "out_file": str(archivo)
        })

    return pd.DataFrame(resumen)


# ============================================================
# GENERACIÓN BY_SCENARIO
# ============================================================

def generar_by_scenario(depuradas, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = []

    claves = ["punto_id", "scenario"]

    for (punto_id, escenario), grupo in depuradas.groupby(claves):
        sustax_total = f"{punto_id}Total"

        tabla = grupo.pivot(
            index="date",
            columns="variable",
            values="value_mm"
        ).reset_index()

        tabla.insert(0, "sustax_total", sustax_total)
        tabla.insert(1, "lat", grupo["lat"].iloc[0])
        tabla.insert(2, "lon", grupo["lon"].iloc[0])

        tabla["date"] = tabla["date"].dt.strftime("%Y-%m-%d")

        archivo = carpeta / f"{sustax_total}__{escenario}__mm.csv"
        tabla.to_csv(archivo, index=False)

        resumen.append({
            "sustax_total": sustax_total,
            "lat": grupo["lat"].iloc[0],
            "lon": grupo["lon"].iloc[0],
            "scenario": escenario,
            "n_dates": len(tabla),
            "date_start": tabla["date"].min(),
            "date_end": tabla["date"].max(),
            "out_file": str(archivo)
        })

    return pd.DataFrame(resumen)


# ============================================================
# RESUMEN DE PUNTOS
# ============================================================

def resumir_puntos(inventario, depuradas):
    filas = []

    for punto_id, grupo in inventario.groupby("punto_id"):
        datos = depuradas.loc[
            depuradas["punto_id"] == punto_id
        ]

        escenarios = sorted(datos["scenario"].unique())

        filas.append({
            "punto_id": punto_id,
            "lat": grupo["lat"].iloc[0],
            "lon": grupo["lon"].iloc[0],
            "archivos_originales": len(grupo),
            "n_escenarios": len(escenarios),
            "escenarios": ", ".join(escenarios),
            "observaciones": len(datos)
        })

    return pd.DataFrame(filas).sort_values(
        ["lat", "lon"]
    ).reset_index(drop=True)


# ============================================================
# EJECUCIÓN AISLADA
# ============================================================

def main():
    if not CARPETA_ORIGEN.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_ORIGEN}"
        )

    inventario, errores_inventario, invalidos = preparar_inventario(
        CARPETA_ORIGEN
    )

    observaciones, errores_lectura = leer_observaciones(inventario)

    depuradas, diagnostico, duplicados, conflictos = (
        consolidar_observaciones(observaciones)
    )

    # Se prepara todo en una carpeta temporal.
    # La colección anterior no se toca si algo falla.
    CARPETA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="sustax_consolidacion_",
        dir=str(CARPETA_SALIDA.parent.resolve())
    ) as temporal:
        temporal = Path(temporal)

        long_dir = temporal / "LONG"
        byscen_dir = temporal / "BY_SCENARIO"

        resumen_long = generar_long(depuradas, long_dir)
        resumen_byscen = generar_by_scenario(
            depuradas, byscen_dir
        )

        puntos = resumir_puntos(inventario, depuradas)

        reportes = {
            "inventario.csv": inventario,
            "puntos_unicos.csv": puntos,
            "diagnostico_registros.csv": diagnostico,
            "duplicados_identicos.csv": duplicados,
            "conflictos.csv": conflictos,
            "coordenadas_invalidas.csv": invalidos,
            "errores_inventario.csv": errores_inventario,
            "errores_lectura.csv": errores_lectura,
            "resumen_long.csv": resumen_long,
            "resumen_by_scenario.csv": resumen_byscen
        }

        for nombre, tabla in reportes.items():
            tabla.to_csv(
                temporal / nombre,
                index=False,
                encoding="utf-8-sig"
            )

        metadata = {
            "archivos_originales": len(inventario),
            "puntos_geograficos": len(puntos),
            "escenarios": sorted(
                observaciones["scenario"].unique().tolist()
            ),
            "observaciones_originales": len(observaciones),
            "observaciones_depuradas": len(depuradas),
            "grupos_duplicados": len(duplicados),
            "filas_duplicadas_excedentes": int(
                (duplicados["n_registros"] - 1).sum()
            ),
            "grupos_conflictivos": len(conflictos),
            "series_generadas": len(resumen_byscen),
            "archivos_con_error": (
                len(errores_inventario) + len(errores_lectura)
            )
        }

        with (temporal / "metadata.json").open(
            "w", encoding="utf-8"
        ) as archivo:
            json.dump(metadata, archivo, indent=2, ensure_ascii=False)

        # La prueba nunca sobrescribe resultados anteriores.
        destino = CARPETA_SALIDA
        contador = 1

        while destino.exists():
            destino = Path(f"{CARPETA_SALIDA}_{contador:02d}")
            contador += 1

        shutil.copytree(temporal, destino)

    print("\n" + "=" * 65)
    print("CONSOLIDACIÓN ESPACIAL SUSTAX")
    print("=" * 65)

    for clave, valor in metadata.items():
        print(f"{clave}: {valor}")

    print("\nESCENARIOS POR PUNTO")
    print(
        puntos[
            ["punto_id", "n_escenarios", "escenarios"]
        ].to_string(index=False)
    )

    print("\nVALIDACIÓN DE ARCHIVOS GENERADOS")
    print(f"LONG: {len(list((destino / 'LONG').glob('*.csv')))}")
    print(
        "BY_SCENARIO:",
        len(list((destino / "BY_SCENARIO").glob("*.csv")))
    )

    print(f"\nResultados guardados en: {destino.resolve()}")

    if not conflictos.empty:
        print(
            "\nATENCIÓN: existen series bloqueadas por conflictos. "
            "Revisar conflictos.csv."
        )
    else:
        print("\nSin conflictos de valores.")

    print("\nPRUEBA FINALIZADA")


if __name__ == "__main__":
    main()
