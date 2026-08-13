from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from scripts.exceedance import (
    procesar_excedencia_batch_conagua,
)

from scripts.interpolation import (
    interpolar_superficie_observacional,
)

from scripts.raster_export import (
    exportar_desde_puntos_a_geotiff,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_CONAGUA = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\estaciones_conagua_excel\Colima"
)

CARPETA_SALIDA = Path(
    r"G:\My Drive\Doctorado-DCAG\results"
    r"\test_pipeline_conagua"
)

PATRON = "*.xlsx"

THRESHOLD = 50.0

COL_LON = "longitud"
COL_LAT = "latitud"
COL_VAL = "prob_excedencia"

RESOLUCION = 300

CRS = "EPSG:4326"


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA PIPELINE COMPLETO — CONAGUA")
    print("=" * 72)

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. PROCESAR EXCEDENCIAS
    # ========================================================

    print()
    print("=" * 72)
    print("1. EXCEDENCIAS")
    print("=" * 72)

    (
        resultados,
        log_df,
        out_master,
        out_log,
    ) = procesar_excedencia_batch_conagua(
        carpeta_estaciones=CARPETA_CONAGUA,
        patron=PATRON,
        threshold=THRESHOLD,
        consolidar_duplicados=True,
        exportar=False,
    )

    if resultados.empty:
        raise ValueError(
            "No se generaron resultados de excedencia."
        )

    print()
    print(
        f"Estaciones procesadas: {len(resultados)}"
    )

    print(
        "Probabilidad mínima observada:",
        resultados[
            COL_VAL
        ].min(),
    )

    print(
        "Probabilidad máxima observada:",
        resultados[
            COL_VAL
        ].max(),
    )

    print(
        "Probabilidad promedio:",
        resultados[
            COL_VAL
        ].mean(),
    )

    # --------------------------------------------------------
    # GUARDAR TABLA MAESTRA ESPACIAL
    # --------------------------------------------------------

    csv_resultados = (
        CARPETA_SALIDA
        / "excedencia_conagua_50mm.csv"
    )

    resultados.to_csv(
        csv_resultados,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Tabla espacial guardada:")
    print(
        csv_resultados
    )

    # ========================================================
    # 2. VALIDAR DATOS ESPACIALES
    # ========================================================

    print()
    print("=" * 72)
    print("2. VALIDACIÓN ESPACIAL")
    print("=" * 72)

    columnas_requeridas = [
        COL_LON,
        COL_LAT,
        COL_VAL,
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in resultados.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas espaciales: {faltantes}"
        )

    datos_validos = resultados.dropna(
        subset=columnas_requeridas
    ).copy()

    print()
    print(
        f"Estaciones con coordenadas y valor válido: "
        f"{len(datos_validos)}"
    )

    if len(datos_validos) < 3:
        raise ValueError(
            "No existen suficientes estaciones para interpolar."
        )

    # ========================================================
    # 3. INTERPOLACIÓN OBSERVACIONAL
    # ========================================================

    print()
    print("=" * 72)
    print("3. INTERPOLACIÓN OBSERVACIONAL")
    print("=" * 72)

    resultado_interp = (
        interpolar_superficie_observacional(
            df=datos_validos,
            col_lon=COL_LON,
            col_lat=COL_LAT,
            col_val=COL_VAL,
            resolucion=RESOLUCION,
            eliminar_duplicados=True,
        )
    )

    calidad = resultado_interp[
        "calidad"
    ]

    print()
    print("CONTROL DE CALIDAD")
    print("-" * 72)

    for clave, valor in calidad.items():
        print(
            f"{clave}: {valor}"
        )

    # ========================================================
    # 4. EXPORTAR GEOTIFF
    # ========================================================

    print()
    print("=" * 72)
    print("4. EXPORTACIÓN GEOTIFF")
    print("=" * 72)

    tif_path = (
        CARPETA_SALIDA
        / "excedencia_conagua_50mm_interpolada.tif"
    )

    resultado_tif = (
        exportar_desde_puntos_a_geotiff(
            df=datos_validos,
            out_tif=str(
                tif_path
            ),
            col_lon=COL_LON,
            col_lat=COL_LAT,
            col_val=COL_VAL,
            margin=0.0,
            nx=RESOLUCION,
            ny=RESOLUCION,
            method="linear",
            fill_nearest=False,
            nodata=-9999.0,
            crs=CRS,
            dtype="float32",
            eliminar_duplicados=True,
        )
    )

    print()
    print("GeoTIFF generado:")
    print(
        resultado_tif[
            "out_tif"
        ]
    )

    # ========================================================
    # 5. VALIDACIÓN DEL GEOTIFF
    # ========================================================

    print()
    print("=" * 72)
    print("5. VALIDACIÓN GEOTIFF")
    print("=" * 72)

    with rasterio.open(
        resultado_tif[
            "out_tif"
        ]
    ) as src:

        arr = src.read(
            1
        )

        nodata = src.nodata

        if nodata is not None:

            arr_validos = arr[
                arr != nodata
            ]

        else:

            arr_validos = arr[
                np.isfinite(
                    arr
                )
            ]

        print()
        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Dimensiones: "
            f"{src.width} × {src.height}"
        )

        print(
            f"Bounds: {src.bounds}"
        )

        print(
            f"Nodata: {src.nodata}"
        )

        print(
            f"Orientación norte-arriba: "
            f"{src.transform.e < 0}"
        )

        print()
        print(
            "Valor mínimo GeoTIFF:",
            float(
                np.min(
                    arr_validos
                )
            ),
        )

        print(
            "Valor máximo GeoTIFF:",
            float(
                np.max(
                    arr_validos
                )
            ),
        )

        print(
            "Valor promedio GeoTIFF:",
            float(
                np.mean(
                    arr_validos
                )
            ),
        )

        if src.crs.to_string() != CRS:
            raise ValueError(
                "El CRS del GeoTIFF no corresponde a EPSG:4326."
            )

        if src.width != RESOLUCION:
            raise ValueError(
                "El ancho del GeoTIFF no corresponde "
                "a la resolución esperada."
            )

        if src.height != RESOLUCION:
            raise ValueError(
                "La altura del GeoTIFF no corresponde "
                "a la resolución esperada."
            )

        if src.transform.e >= 0:
            raise ValueError(
                "El GeoTIFF no tiene orientación norte-arriba."
            )

    # ========================================================
    # 6. COMPARAR SUPERFICIES
    # ========================================================

    print()
    print("=" * 72)
    print("6. COMPARACIÓN DE SUPERFICIES")
    print("=" * 72)

    grid_z = resultado_interp[
        "grid_z"
    ]

    minimo_interp = float(
        np.nanmin(
            grid_z
        )
    )

    maximo_interp = float(
        np.nanmax(
            grid_z
        )
    )

    minimo_tif = float(
        np.min(
            arr_validos
        )
    )

    maximo_tif = float(
        np.max(
            arr_validos
        )
    )

    print()
    print(
        "Interpolación — mínimo:",
        minimo_interp,
    )

    print(
        "GeoTIFF       — mínimo:",
        minimo_tif,
    )

    print()
    print(
        "Interpolación — máximo:",
        maximo_interp,
    )

    print(
        "GeoTIFF       — máximo:",
        maximo_tif,
    )

    diferencia_min = abs(
        minimo_interp
        - minimo_tif
    )

    diferencia_max = abs(
        maximo_interp
        - maximo_tif
    )

    print()
    print(
        "Diferencia mínima:",
        diferencia_min,
    )

    print(
        "Diferencia máxima:",
        diferencia_max,
    )

    # float32 puede introducir diferencias numéricas pequeñas.
    tolerancia = 1e-6

    if diferencia_min > tolerancia:
        raise ValueError(
            "El mínimo del GeoTIFF no coincide "
            "con la interpolación observacional."
        )

    if diferencia_max > tolerancia:
        raise ValueError(
            "El máximo del GeoTIFF no coincide "
            "con la interpolación observacional."
        )

    # ========================================================
    # 7. VALIDACIONES FINALES
    # ========================================================

    print()
    print("=" * 72)
    print("7. VALIDACIONES FINALES")
    print("=" * 72)

    if len(resultados) != 37:
        raise ValueError(
            "Se esperaban 37 estaciones de Colima."
        )

    if (
        resultados[
            COL_VAL
        ].lt(0).any()
        or resultados[
            COL_VAL
        ].gt(1).any()
    ):
        raise ValueError(
            "Existen probabilidades fuera de [0, 1]."
        )

    if not tif_path.exists():
        raise FileNotFoundError(
            "No se generó el GeoTIFF esperado."
        )

    print()
    print(
        "Estaciones: OK"
    )

    print(
        "Coordenadas: OK"
    )

    print(
        "Probabilidades: OK"
    )

    print(
        "Interpolación observacional: OK"
    )

    print(
        "GeoTIFF: OK"
    )

    print(
        "Orientación north-up: OK"
    )

    print(
        "Correspondencia interpolación ↔ GeoTIFF: OK"
    )

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)

    print()
    print("Archivos generados:")

    print(
        f"1. {csv_resultados}"
    )

    print(
        f"2. {tif_path}"
    )


if __name__ == "__main__":
    main()