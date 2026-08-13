from pathlib import Path

import pandas as pd

from scripts.conagua_reader import (
    leer_lote_conagua,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_CONAGUA = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\estaciones_conagua_excel\Colima"
)

PATRON = "*.xlsx"


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA LECTOR CONAGUA POR LOTE")
    print("=" * 72)

    # --------------------------------------------------------
    # LEER LOTE
    # --------------------------------------------------------

    (
        estaciones,
        metadata_df,
        log_df,
    ) = leer_lote_conagua(
        carpeta=CARPETA_CONAGUA,
        patron=PATRON,
    )

    # --------------------------------------------------------
    # RESUMEN GENERAL
    # --------------------------------------------------------

    total_archivos = len(
        log_df
    )

    total_ok = int(
        (
            log_df["status"] == "ok"
        ).sum()
    )

    total_error = int(
        (
            log_df["status"] == "error"
        ).sum()
    )

    print()
    print("RESUMEN GENERAL")
    print("-" * 72)

    print(
        f"Archivos encontrados: {total_archivos}"
    )

    print(
        f"Estaciones leídas correctamente: {total_ok}"
    )

    print(
        f"Archivos con error: {total_error}"
    )

    print(
        f"Estaciones cargadas en memoria: {len(estaciones)}"
    )

    # --------------------------------------------------------
    # METADATOS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("METADATOS DE ESTACIONES")
    print("=" * 72)

    if metadata_df.empty:

        print(
            "No se generaron metadatos."
        )

    else:

        columnas_mostrar = [
            columna
            for columna in [
                "station",
                "nombre",
                "estado",
                "municipio",
                "situacion",
                "latitud",
                "longitud",
                "altitud_msnm",
            ]
            if columna in metadata_df.columns
        ]

        print()
        print(
            metadata_df[
                columnas_mostrar
            ]
            .sort_values(
                "station"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CALIDAD DE SERIES
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CALIDAD DE SERIES CLIMÁTICAS")
    print("=" * 72)

    calidad_rows = []

    for estacion in estaciones:

        metadata = estacion[
            "metadata"
        ]

        calidad = estacion[
            "calidad"
        ]

        calidad_rows.append(
            {
                "station": metadata.get(
                    "station"
                ),
                "nombre": metadata.get(
                    "nombre"
                ),
                "total_registros": calidad.get(
                    "total_registros"
                ),
                "fecha_inicio": calidad.get(
                    "fecha_inicio"
                ),
                "fecha_fin": calidad.get(
                    "fecha_fin"
                ),
                "precipitacion_valida": calidad.get(
                    "precipitacion_valida"
                ),
                "precipitacion_nula": calidad.get(
                    "precipitacion_nula"
                ),
                "latitud": calidad.get(
                    "latitud"
                ),
                "longitud": calidad.get(
                    "longitud"
                ),
            }
        )

    calidad_df = pd.DataFrame(
        calidad_rows
    )

    if calidad_df.empty:

        print(
            "No existe información de calidad."
        )

    else:

        print()
        print(
            calidad_df
            .sort_values(
                "station"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------

    errores = log_df[
        log_df["status"] == "error"
    ].copy()

    print()
    print("=" * 72)
    print("ARCHIVOS CON ERROR")
    print("=" * 72)

    if errores.empty:

        print()
        print(
            "No se detectaron errores."
        )

    else:

        print()
        print(
            errores[
                [
                    "archivo",
                    "mensaje",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # VALIDACIONES GENERALES
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("VALIDACIONES AUTOMÁTICAS")
    print("=" * 72)

    if total_archivos == 0:
        raise ValueError(
            "No se encontraron archivos CONAGUA."
        )

    if total_ok == 0:
        raise ValueError(
            "No fue posible leer ninguna estación."
        )

    if metadata_df.empty:
        raise ValueError(
            "No se generó la tabla maestra de metadatos."
        )

    if calidad_df.empty:
        raise ValueError(
            "No se generó información de calidad."
        )

    # --------------------------------------------------------
    # CLAVES DUPLICADAS
    # --------------------------------------------------------

    duplicados_station = (
        metadata_df[
            "station"
        ]
        .duplicated(
            keep=False
        )
    )

    total_duplicados = int(
        duplicados_station.sum()
    )

    print()
    print(
        f"Claves de estación duplicadas: "
        f"{total_duplicados}"
    )

    if total_duplicados > 0:

        print()
        print(
            "Estaciones duplicadas:"
        )

        print(
            metadata_df[
                duplicados_station
            ][
                [
                    "station",
                    "nombre",
                    "archivo",
                ]
            ]
            .sort_values(
                "station"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # COORDENADAS INVÁLIDAS
    # --------------------------------------------------------

    coordenadas_invalidas = metadata_df[
        metadata_df["latitud"].isna()
        | metadata_df["longitud"].isna()
    ]

    print()
    print(
        "Estaciones sin coordenadas válidas: "
        f"{len(coordenadas_invalidas)}"
    )

    if not coordenadas_invalidas.empty:

        print()
        print(
            coordenadas_invalidas[
                [
                    "station",
                    "nombre",
                    "latitud",
                    "longitud",
                    "archivo",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # ESTADOS DETECTADOS
    # --------------------------------------------------------

    estados_detectados = sorted(
        metadata_df[
            "estado"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print()
    print(
        "Estados detectados:"
    )

    for estado in estados_detectados:
        print(
            f"  - {estado}"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA")
    print("=" * 72)

    print()
    print(
        "El lector CONAGUA por lote terminó correctamente."
    )


if __name__ == "__main__":
    main()