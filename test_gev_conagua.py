from pathlib import Path

import numpy as np
import pandas as pd

from scripts.batch_return_levels import (
    ejecutar_proceso_batch_conagua,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_CONAGUA = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación"
    r"\estaciones_conagua_excel\Colima"
)

PATRON = "*.xlsx"

N_MIN_ANIOS = 10

NIVELES_RETORNO = np.array(
    [
        2,
        5,
        10,
        25,
        50,
        100,
    ],
    dtype=float,
)

# Para la prueba inicial usamos 100 réplicas para no hacer
# demasiado lenta la validación. Después puede aumentarse.
N_BOOT = 100

ALPHA = 0.05

SEED = 42

USAR_BOOT_PARAMETRICO = True


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA GEV — ARCHIVOS ORIGINALES CONAGUA")
    print("=" * 72)

    (
        maestro,
        log_df,
        out_master,
        out_log,
        metadata_df,
    ) = ejecutar_proceso_batch_conagua(
        dir_in=CARPETA_CONAGUA,
        patron=PATRON,
        n_min_anios=N_MIN_ANIOS,
        niveles_retorno=NIVELES_RETORNO,
        n_boot=N_BOOT,
        alpha=ALPHA,
        seed=SEED,
        usar_boot_parametrico=USAR_BOOT_PARAMETRICO,
        plot_max_t=float(
            NIVELES_RETORNO.max()
        ),
    )

    # ========================================================
    # VALIDAR SALIDA GENERAL
    # ========================================================

    print()
    print("=" * 72)
    print("RESUMEN GENERAL")
    print("=" * 72)

    if maestro is None or maestro.empty:
        raise ValueError(
            "No se generó la tabla maestra GEV."
        )

    total_estaciones = int(
        maestro[
            "station"
        ]
        .astype(str)
        .nunique()
    )

    total_filas = len(
        maestro
    )

    print()
    print(
        f"Estaciones con resultados GEV: "
        f"{total_estaciones}"
    )

    print(
        f"Filas en tabla maestra: "
        f"{total_filas}"
    )

    print(
        f"Periodos de retorno esperados: "
        f"{len(NIVELES_RETORNO)}"
    )

    print(
        f"Réplicas bootstrap solicitadas: "
        f"{N_BOOT}"
    )

    # ========================================================
    # VALIDAR COLUMNAS
    # ========================================================

    print()
    print("=" * 72)
    print("COLUMNAS GENERADAS")
    print("=" * 72)

    print()
    print(
        list(
            maestro.columns
        )
    )

    columnas_requeridas = [
        "station",
        "nombre",
        "estado",
        "municipio",
        "latitud",
        "longitud",
        "T_years",
        "level_mm",
        "CI_low95_bootA",
        "CI_high95_bootA",
        "CI_low95_bootB",
        "CI_high95_bootB",
        "gev_shape",
        "gev_loc",
        "gev_scale",
        "n_years",
        "trend_slope_mm_per_year",
        "note",
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in maestro.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas esperadas: {faltantes}"
        )

    # ========================================================
    # VALIDAR PERIODOS POR ESTACIÓN
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIÓN DE PERIODOS DE RETORNO")
    print("=" * 72)

    errores_periodos = []

    for station, grupo in maestro.groupby(
        "station"
    ):

        periodos = np.sort(
            pd.to_numeric(
                grupo[
                    "T_years"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        if not np.array_equal(
            periodos,
            NIVELES_RETORNO,
        ):

            errores_periodos.append(
                {
                    "station": station,
                    "periodos": periodos.tolist(),
                }
            )

    print()
    print(
        f"Estaciones con periodos incompletos: "
        f"{len(errores_periodos)}"
    )

    if errores_periodos:

        print()
        print(
            pd.DataFrame(
                errores_periodos
            ).to_string(
                index=False
            )
        )

    # ========================================================
    # VALIDAR PARÁMETROS GEV
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIÓN DE PARÁMETROS GEV")
    print("=" * 72)

    resumen_estaciones = (
        maestro[
            [
                "station",
                "nombre",
                "estado",
                "municipio",
                "latitud",
                "longitud",
                "n_years",
                "gev_shape",
                "gev_loc",
                "gev_scale",
                "trend_slope_mm_per_year",
                "bootA_naccepted",
                "bootB_naccepted",
                "note",
            ]
        ]
        .drop_duplicates(
            subset=[
                "station"
            ]
        )
        .sort_values(
            "station"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        resumen_estaciones.to_string(
            index=False
        )
    )

    parametros_invalidos = resumen_estaciones[
        ~np.isfinite(
            pd.to_numeric(
                resumen_estaciones[
                    "gev_shape"
                ],
                errors="coerce",
            )
        )
        |
        ~np.isfinite(
            pd.to_numeric(
                resumen_estaciones[
                    "gev_loc"
                ],
                errors="coerce",
            )
        )
        |
        ~np.isfinite(
            pd.to_numeric(
                resumen_estaciones[
                    "gev_scale"
                ],
                errors="coerce",
            )
        )
        |
        (
            pd.to_numeric(
                resumen_estaciones[
                    "gev_scale"
                ],
                errors="coerce",
            )
            <= 0
        )
    ]

    print()
    print(
        f"Estaciones con parámetros GEV inválidos: "
        f"{len(parametros_invalidos)}"
    )

    # ========================================================
    # VALIDAR NIVELES DE RETORNO
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIÓN DE NIVELES DE RETORNO")
    print("=" * 72)

    nivel = pd.to_numeric(
        maestro[
            "level_mm"
        ],
        errors="coerce",
    )

    niveles_invalidos = maestro[
        (~np.isfinite(nivel))
        |
        (nivel <= 0)
    ]

    print()
    print(
        f"Filas con niveles inválidos: "
        f"{len(niveles_invalidos)}"
    )

    # ========================================================
    # CONTROL DE LA ESTACIÓN 6001
    # ========================================================

    print()
    print("=" * 72)
    print("CONTROL ESTACIÓN 6001 — ARMERIA")
    print("=" * 72)

    estacion_6001 = (
        maestro[
            maestro[
                "station"
            ].astype(str)
            == "6001"
        ]
        .copy()
        .sort_values(
            "T_years"
        )
    )

    if estacion_6001.empty:
        raise ValueError(
            "No se encontró la estación 6001."
        )

    columnas_6001 = [
        "station",
        "nombre",
        "T_years",
        "level_mm",
        "CI_low95_bootA",
        "CI_high95_bootA",
        "CI_low95_bootB",
        "CI_high95_bootB",
        "gev_shape",
        "gev_loc",
        "gev_scale",
        "n_years",
        "trend_slope_mm_per_year",
    ]

    print()
    print(
        estacion_6001[
            columnas_6001
        ]
        .round(
            {
                "level_mm": 4,
                "CI_low95_bootA": 4,
                "CI_high95_bootA": 4,
                "CI_low95_bootB": 4,
                "CI_high95_bootB": 4,
                "gev_shape": 6,
                "gev_loc": 4,
                "gev_scale": 4,
                "trend_slope_mm_per_year": 6,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # REVISAR ESTACIONES CON POCOS AÑOS
    # ========================================================

    print()
    print("=" * 72)
    print("ESTACIONES CON COBERTURA TEMPORAL LIMITADA")
    print("=" * 72)

    cobertura_limitada = (
        resumen_estaciones[
            pd.to_numeric(
                resumen_estaciones[
                    "n_years"
                ],
                errors="coerce",
            )
            < N_MIN_ANIOS
        ]
        .copy()
    )

    print()
    print(
        f"Estaciones con menos de "
        f"{N_MIN_ANIOS} máximos anuales: "
        f"{len(cobertura_limitada)}"
    )

    if not cobertura_limitada.empty:

        print()
        print(
            cobertura_limitada[
                [
                    "station",
                    "nombre",
                    "n_years",
                    "note",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # REVISAR BOOTSTRAP
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIÓN BOOTSTRAP")
    print("=" * 72)

    boot_a = pd.to_numeric(
        resumen_estaciones[
            "bootA_naccepted"
        ],
        errors="coerce",
    )

    boot_b = pd.to_numeric(
        resumen_estaciones[
            "bootB_naccepted"
        ],
        errors="coerce",
    )

    print()
    print(
        "Bootstrap A — mínimo aceptado:",
        int(
            boot_a.min()
        )
        if boot_a.notna().any()
        else "N/D",
    )

    print(
        "Bootstrap A — máximo aceptado:",
        int(
            boot_a.max()
        )
        if boot_a.notna().any()
        else "N/D",
    )

    print(
        "Bootstrap B — mínimo aceptado:",
        int(
            boot_b.min()
        )
        if boot_b.notna().any()
        else "N/D",
    )

    print(
        "Bootstrap B — máximo aceptado:",
        int(
            boot_b.max()
        )
        if boot_b.notna().any()
        else "N/D",
    )

    sin_boot_a = resumen_estaciones[
        boot_a <= 0
    ]

    print()
    print(
        f"Estaciones sin réplicas válidas Bootstrap A: "
        f"{len(sin_boot_a)}"
    )

    # ========================================================
    # LOG
    # ========================================================

    print()
    print("=" * 72)
    print("LOG DEL PROCESO")
    print("=" * 72)

    if (
        log_df is not None
        and not log_df.empty
    ):

        if "status" in log_df.columns:

            print()
            print(
                log_df[
                    "status"
                ]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )

        errores_log = log_df[
            log_df.get(
                "status",
                pd.Series(
                    index=log_df.index,
                    dtype=object,
                )
            ).isin(
                [
                    "error",
                    "incompatible",
                ]
            )
        ]

        if not errores_log.empty:

            print()
            print("Entradas no OK:")
            print(
                errores_log.to_string(
                    index=False
                )
            )

    # ========================================================
    # ARCHIVOS GENERADOS
    # ========================================================

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
    # VALIDACIONES FINALES
    # ========================================================

    print()
    print("=" * 72)
    print("VALIDACIONES FINALES")
    print("=" * 72)

    if total_estaciones != 37:
        raise ValueError(
            "Se esperaban 37 estaciones CONAGUA con "
            f"resultados GEV y se obtuvieron {total_estaciones}."
        )

    filas_esperadas = (
        total_estaciones
        * len(
            NIVELES_RETORNO
        )
    )

    if total_filas != filas_esperadas:
        raise ValueError(
            f"Se esperaban {filas_esperadas} filas en "
            f"la tabla maestra y se obtuvieron {total_filas}."
        )

    if errores_periodos:
        raise ValueError(
            "Existen estaciones con periodos de retorno "
            "incompletos."
        )

    if not parametros_invalidos.empty:
        raise ValueError(
            "Existen estaciones con parámetros GEV inválidos."
        )

    if not niveles_invalidos.empty:
        raise ValueError(
            "Existen niveles de retorno inválidos."
        )

    print()
    print("Estaciones: OK")
    print("Periodos de retorno: OK")
    print("Parámetros GEV: OK")
    print("Niveles de retorno: OK")
    print("Metadatos CONAGUA: OK")
    print("Tabla maestra: OK")

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()