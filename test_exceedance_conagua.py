from pathlib import Path

import pandas as pd

from scripts.exceedance import (
    procesar_excedencia_batch_conagua,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_CONAGUA = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\estaciones_conagua_excel\Colima"
)

PATRON = "*.xlsx"

THRESHOLD = 50.0


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA EXCEDENCIA — ARCHIVOS ORIGINALES CONAGUA")
    print("=" * 72)

    # --------------------------------------------------------
    # PROCESAMIENTO
    # --------------------------------------------------------

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
        exportar=True,
    )

    # --------------------------------------------------------
    # RESUMEN GENERAL
    # --------------------------------------------------------

    print()
    print("RESUMEN GENERAL")
    print("-" * 72)

    print(
        f"Estaciones con resultado: {len(resultados)}"
    )

    if (
        log_df is not None
        and not log_df.empty
    ):

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

    else:

        total_ok = 0
        total_error = 0

    print(
        f"Procesadas correctamente: {total_ok}"
    )

    print(
        f"Con error: {total_error}"
    )

    # --------------------------------------------------------
    # COLUMNAS GENERADAS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("COLUMNAS DE RESULTADOS")
    print("=" * 72)

    print(
        list(
            resultados.columns
        )
    )

    # --------------------------------------------------------
    # TABLA PRINCIPAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESULTADOS DE EXCEDENCIA")
    print("=" * 72)

    columnas_mostrar = [
        columna
        for columna in [
            "station",
            "nombre",
            "estado",
            "municipio",
            "latitud",
            "longitud",
            "total_dias",
            "dias_excedencia",
            "prob_excedencia",
            "EXCEDENCIA_50MM",
            "fecha_inicio",
            "fecha_fin",
            "n_anios",
        ]
        if columna in resultados.columns
    ]

    print()
    print(
        resultados[
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
    # ESTADÍSTICAS DE EXCEDENCIA
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 72)

    if (
        "prob_excedencia"
        in resultados.columns
        and not resultados.empty
    ):

        probabilidades = pd.to_numeric(
            resultados[
                "prob_excedencia"
            ],
            errors="coerce",
        )

        print()
        print(
            "Probabilidad mínima:",
            probabilidades.min(),
        )

        print(
            "Probabilidad máxima:",
            probabilidades.max(),
        )

        print(
            "Probabilidad promedio:",
            probabilidades.mean(),
        )

    # --------------------------------------------------------
    # VALIDAR COORDENADAS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("VALIDACIÓN ESPACIAL")
    print("=" * 72)

    sin_coordenadas = resultados[
        resultados[
            [
                "latitud",
                "longitud",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    print()
    print(
        f"Estaciones sin coordenadas: "
        f"{len(sin_coordenadas)}"
    )

    if not sin_coordenadas.empty:

        print()
        print(
            sin_coordenadas[
                [
                    "station",
                    "nombre",
                    "latitud",
                    "longitud",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # VALIDAR CLAVES DUPLICADAS
    # --------------------------------------------------------

    duplicadas = resultados[
        resultados[
            "station"
        ]
        .duplicated(
            keep=False
        )
    ]

    print()
    print(
        f"Claves duplicadas: "
        f"{len(duplicadas)}"
    )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("LOG DEL PROCESO")
    print("=" * 72)

    if (
        log_df is not None
        and not log_df.empty
    ):

        print()
        print(
            log_df.to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "No se generó log."
        )

    # --------------------------------------------------------
    # ARCHIVOS GENERADOS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ARCHIVOS GENERADOS")
    print("=" * 72)

    print()
    print(
        "Tabla maestra:"
    )

    print(
        out_master
    )

    print()
    print(
        "Log:"
    )

    print(
        out_log
    )

    # ========================================================
    # VALIDACIONES AUTOMÁTICAS
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIONES AUTOMÁTICAS")
    print("=" * 72)

    if resultados.empty:
        raise ValueError(
            "No se generaron resultados de excedencia."
        )

    if len(resultados) != 37:
        raise ValueError(
            "Se esperaban 37 estaciones válidas para Colima, "
            f"pero se obtuvieron {len(resultados)}."
        )

    if "prob_excedencia" not in resultados.columns:
        raise ValueError(
            "No existe la columna prob_excedencia."
        )

    if "EXCEDENCIA_50MM" not in resultados.columns:
        raise ValueError(
            "No existe la columna EXCEDENCIA_50MM."
        )

    if len(sin_coordenadas) > 0:
        raise ValueError(
            "Existen estaciones sin coordenadas válidas."
        )

    if len(duplicadas) > 0:
        raise ValueError(
            "Existen claves de estación duplicadas."
        )

    # --------------------------------------------------------
    # RANGO DE PROBABILIDAD
    # --------------------------------------------------------

    probabilidades_validas = pd.to_numeric(
        resultados[
            "prob_excedencia"
        ],
        errors="coerce",
    ).dropna()

    if (
        probabilidades_validas.lt(
            0
        ).any()
        or probabilidades_validas.gt(
            1
        ).any()
    ):
        raise ValueError(
            "Existen probabilidades fuera del intervalo [0, 1]."
        )

    # --------------------------------------------------------
    # COMPROBAR ESTACIÓN 6001
    # --------------------------------------------------------

    estacion_6001 = resultados[
        resultados[
            "station"
        ].astype(str) == "6001"
    ]

    if estacion_6001.empty:
        raise ValueError(
            "No se encontró la estación 6001."
        )

    fila_6001 = estacion_6001.iloc[
        0
    ]

    print()
    print("CONTROL ESTACIÓN 6001")
    print("-" * 72)

    print(
        "Nombre:",
        fila_6001[
            "nombre"
        ],
    )

    print(
        "Días válidos:",
        fila_6001[
            "total_dias"
        ],
    )

    print(
        "Días con excedencia:",
        fila_6001[
            "dias_excedencia"
        ],
    )

    print(
        "Probabilidad:",
        fila_6001[
            "prob_excedencia"
        ],
    )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()