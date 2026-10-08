
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.sustax.inventory import build_sustax_inventory
from scripts.sustax.loader import load_sustax_file_safe
from scripts.sustax.merge import split_var_scenario

# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_SUSTAX = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\Datos Sustax"
)
CARPETA_RESULTADOS = Path("results/tests/sustax_coordenadas")
DECIMALES_COORDENADAS = 4
TOLERANCIA_MM = 1e-6

# ============================================================
# INVENTARIO
# ============================================================

def preparar_inventario(carpeta):
    inventario, errores = build_sustax_inventory(str(carpeta))

    if inventario.empty:
        raise ValueError("No se encontraron archivos Sustax compatibles.")

    inventario = inventario.copy()
    inventario["lat"] = pd.to_numeric(
        inventario["lat"], errors="coerce"
    ).round(DECIMALES_COORDENADAS)
    inventario["lon"] = pd.to_numeric(
        inventario["lon"], errors="coerce"
    ).round(DECIMALES_COORDENADAS)

    invalidas = inventario[
        inventario["lat"].isna()
        | inventario["lon"].isna()
        | ~inventario["lat"].between(-90, 90)
        | ~inventario["lon"].between(-180, 180)
    ]

    inventario = inventario.drop(index=invalidas.index).copy()

    inventario["punto_id"] = inventario.apply(
        lambda r: f"{r['lat']:.4f}_{r['lon']:.4f}",
        axis=1
    )

    return inventario, errores, invalidas


# ============================================================
# LECTURA DE OBSERVACIONES
# ============================================================

def leer_observaciones(inventario):
    partes = []
    errores = []

    for _, registro in inventario.iterrows():
        archivo = registro["path"]

        try:
            datos, _, _ = load_sustax_file_safe(
                archivo,
                return_pandas_df=True,
                return_metadata=True
            )

            datos = datos.copy()
            datos.index = pd.to_datetime(datos.index, errors="coerce")
            datos = datos[~datos.index.isna()]
            datos.index = datos.index.normalize()

            for columna in datos.columns:
                variable, escenario = split_var_scenario(columna)

                if variable is None or escenario is None:
                    continue

                valores = pd.to_numeric(
                    datos[columna], errors="coerce"
                ) * 1000.0

                parte = pd.DataFrame({
                    "punto_id": registro["punto_id"],
                    "lat": registro["lat"],
                    "lon": registro["lon"],
                    "scenario": escenario.upper(),
                    "variable": variable,
                    "date": datos.index,
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
        raise ValueError("No se obtuvieron observaciones Sustax.")

    observaciones = pd.concat(partes, ignore_index=True)
    observaciones = observaciones.dropna(
        subset=["date", "value_mm"]
    )

    return observaciones, pd.DataFrame(errores)


# ============================================================
# VALIDACIÓN DE DUPLICADOS
# ============================================================

def validar_observaciones(observaciones):
    claves = ["punto_id", "scenario", "variable", "date"]

    grupos = observaciones.groupby(
        claves, dropna=False, sort=False
    )

    diagnostico = grupos.agg(
        registros=("value_mm", "size"),
        valor_min_mm=("value_mm", "min"),
        valor_max_mm=("value_mm", "max"),
        archivos=("source_file", lambda s: " | ".join(
            sorted(set(s.astype(str)))
        ))
    ).reset_index()

    diagnostico["diferencia_mm"] = (
        diagnostico["valor_max_mm"]
        - diagnostico["valor_min_mm"]
    )

    diagnostico["estado"] = np.select(
        [
            diagnostico["diferencia_mm"] > TOLERANCIA_MM,
            diagnostico["registros"] > 1
        ],
        ["CONFLICTO", "DUPLICADO_IDENTICO"],
        default="UNICO"
    )

    duplicados = diagnostico[
        diagnostico["estado"] == "DUPLICADO_IDENTICO"
    ].copy()

    conflictos = diagnostico[
        diagnostico["estado"] == "CONFLICTO"
    ].copy()

    # No se elige arbitrariamente un valor ante un conflicto.
    claves_conflicto = conflictos[claves].copy()

    sin_conflictos = observaciones.merge(
        claves_conflicto.assign(_conflicto=True),
        on=claves,
        how="left"
    )

    sin_conflictos = sin_conflictos[
        sin_conflictos["_conflicto"].isna()
    ].drop(columns="_conflicto")

    depuradas = (
        sin_conflictos
        .sort_values(["punto_id", "scenario", "date", "source_file"])
        .drop_duplicates(subset=claves, keep="first")
        .reset_index(drop=True)
    )

    return diagnostico, duplicados, conflictos, depuradas


# ============================================================
# RESUMEN ESPACIAL
# ============================================================

def resumir_puntos(inventario, observaciones):
    resumen = []

    for punto_id, grupo in inventario.groupby("punto_id"):
        series = observaciones[
            observaciones["punto_id"] == punto_id
        ]

        escenarios = sorted(series["scenario"].dropna().unique())

        resumen.append({
            "punto_id": punto_id,
            "lat": grupo["lat"].iloc[0],
            "lon": grupo["lon"].iloc[0],
            "archivos_originales": len(grupo),
            "nombres_base": " | ".join(
                sorted(set(grupo["base_name"].astype(str)))
            ),
            "n_escenarios": len(escenarios),
            "escenarios": ", ".join(escenarios),
            "fecha_inicio": series["date"].min(),
            "fecha_fin": series["date"].max()
        })

    return pd.DataFrame(resumen).sort_values(
        ["lat", "lon"]
    ).reset_index(drop=True)


# ============================================================
# EJECUCIÓN
# ============================================================

def main():
    if not CARPETA_SUSTAX.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_SUSTAX}"
        )

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    inventario, errores_inventario, coordenadas_invalidas = (
        preparar_inventario(CARPETA_SUSTAX)
    )

    observaciones, errores_lectura = leer_observaciones(inventario)

    diagnostico, duplicados, conflictos, depuradas = (
        validar_observaciones(observaciones)
    )

    puntos = resumir_puntos(inventario, observaciones)

    archivos = {
        "inventario": inventario,
        "puntos_unicos": puntos,
        "diagnostico_registros": diagnostico,
        "duplicados_identicos": duplicados,
        "conflictos": conflictos,
        "coordenadas_invalidas": coordenadas_invalidas,
        "errores_inventario": errores_inventario,
        "errores_lectura": errores_lectura
    }

    for nombre, tabla in archivos.items():
        tabla.to_csv(
            CARPETA_RESULTADOS / f"{nombre}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n" + "=" * 65)
    print("VALIDACIÓN ESPACIAL DE ARCHIVOS SUSTAX")
    print("=" * 65)
    print(f"Archivos inventariados: {len(inventario)}")
    print(f"Puntos geográficos únicos: {len(puntos)}")
    print(f"Escenarios detectados: {observaciones['scenario'].nunique()}")
    print(f"Observaciones originales válidas: {len(observaciones):,}")
    print(f"Registros duplicados idénticos: {len(duplicados):,}")
    print(f"Registros conflictivos: {len(conflictos):,}")
    print(f"Observaciones depuradas sin conflictos: {len(depuradas):,}")
    print(f"Archivos con error de inventario: {len(errores_inventario)}")
    print(f"Archivos con error de lectura: {len(errores_lectura)}")

    print("\nESCENARIOS POR PUNTO")
    print(
        puntos[
            ["punto_id", "archivos_originales", "n_escenarios", "escenarios"]
        ].to_string(index=False)
    )

    print(f"\nDiagnósticos guardados en: {CARPETA_RESULTADOS.resolve()}")

    if not conflictos.empty:
        print("\nATENCIÓN: existen conflictos que requieren revisión.")
        print("No deben fusionarse automáticamente esos registros.")
    else:
        print("\nNo se detectaron valores contradictorios.")

    print("\nPRUEBA FINALIZADA")


if __name__ == "__main__":
    main()
