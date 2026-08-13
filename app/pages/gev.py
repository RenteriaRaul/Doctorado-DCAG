from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.batch_return_levels import (
    ejecutar_proceso_batch_conagua,
)
from scripts.conagua_reader import (
    leer_lote_conagua,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARPETA_SALIDA = (
    PROJECT_ROOT
    / "results"
    / "streamlit"
    / "gev"
)

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "gev_batch_result" not in st.session_state:
    st.session_state.gev_batch_result = None


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def interpretar_calidad_bootstrap(
    aceptadas,
    solicitadas,
):
    """
    Clasifica de forma descriptiva la proporción de
    réplicas Bootstrap A aceptadas.
    """

    if (
        pd.isna(aceptadas)
        or solicitadas <= 0
    ):
        return (
            "No disponible",
            None,
        )

    proporcion = (
        float(aceptadas)
        / float(solicitadas)
    )

    if proporcion >= 0.80:

        return (
            "Alta",
            proporcion,
        )

    if proporcion >= 0.50:

        return (
            "Moderada",
            proporcion,
        )

    return (
        "Limitada",
        proporcion,
    )


def construir_etiqueta_estacion(
    station,
    nombre,
):
    """
    Construye etiqueta legible para selectores.
    """

    if pd.notna(nombre):

        return (
            f"{station} — {nombre}"
        )

    return str(
        station
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "Análisis GEV y periodos de retorno"
)

st.write(
    """
    Este módulo estima niveles de precipitación extrema mediante
    la distribución Generalizada de Valores Extremos (GEV).

    La plataforma trabaja directamente con los archivos Excel
    originales de estaciones CONAGUA. Para cada estación se
    extraen los máximos anuales de precipitación, se ajusta una
    distribución GEV y se estiman niveles asociados a distintos
    periodos de retorno.
    """
)

st.info(
    """
    **Flujo del análisis:** archivos originales CONAGUA →
    limpieza de datos → máximos anuales → ajuste GEV →
    niveles de retorno → intervalo de confianza mediante
    Bootstrap A robusto.
    """
)

st.caption(
    "El Bootstrap robusto se utiliza como método principal "
    "para representar la incertidumbre de los niveles de retorno."
)


# ============================================================
# 1. FUENTE DE DATOS
# ============================================================

st.subheader(
    "1. Fuente de datos CONAGUA"
)

dir_in = st.text_input(
    "Carpeta con archivos originales de estaciones CONAGUA",
    placeholder=(
        r"G:\...\estaciones_conagua_excel\Colima"
    ),
    help=(
        "La carpeta debe contener los archivos Excel originales "
        "descargados de CONAGUA."
    ),
    key="gev_source_folder",
)

patron = st.text_input(
    "Patrón de archivos",
    value="*.xlsx",
    help=(
        "Normalmente no es necesario modificar este valor."
    ),
    key="gev_excel_pattern",
)


# ============================================================
# VERIFICAR CARPETA
# ============================================================

carpeta = None
carpeta_valida = False

estaciones_lectura = []
metadata_df = pd.DataFrame()
log_lectura = pd.DataFrame()


if dir_in.strip():

    carpeta = Path(
        dir_in.strip()
    ).expanduser()

    if not carpeta.exists():

        st.error(
            "La carpeta indicada no existe."
        )

    elif not carpeta.is_dir():

        st.error(
            "La ruta indicada no corresponde a una carpeta."
        )

    else:

        try:

            with st.spinner(
                "Leyendo archivos CONAGUA..."
            ):

                (
                    estaciones_lectura,
                    metadata_df,
                    log_lectura,
                ) = leer_lote_conagua(
                    carpeta=carpeta,
                    patron=patron,
                )

            carpeta_valida = (
                len(estaciones_lectura)
                > 0
            )

            total_ok = int(
                (
                    log_lectura[
                        "status"
                    ] == "ok"
                ).sum()
            )

            total_no_ok = int(
                (
                    log_lectura[
                        "status"
                    ] != "ok"
                ).sum()
            )

            if total_ok > 0:

                st.success(
                    f"Se detectaron {total_ok} estaciones "
                    "CONAGUA compatibles."
                )

            if total_no_ok > 0:

                st.warning(
                    f"{total_no_ok} archivo(s) no corresponden "
                    "al formato esperado de una estación CONAGUA "
                    "y serán ignorados."
                )

        except Exception as error:

            st.error(
                "No fue posible leer la carpeta."
            )

            st.exception(
                error
            )


# ============================================================
# 2. ESTACIONES DETECTADAS
# ============================================================

st.subheader(
    "2. Estaciones detectadas"
)


if (
    carpeta_valida
    and not metadata_df.empty
):

    total_estaciones = len(
        metadata_df
    )

    estados_detectados = sorted(
        metadata_df[
            "estado"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    municipios_detectados = int(
        metadata_df[
            "municipio"
        ]
        .dropna()
        .nunique()
    )

    sin_coordenadas = int(
        metadata_df[
            [
                "latitud",
                "longitud",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    col_e1, col_e2, col_e3, col_e4 = (
        st.columns(4)
    )

    col_e1.metric(
        "Estaciones",
        total_estaciones,
    )

    col_e2.metric(
        "Estados",
        len(
            estados_detectados
        ),
    )

    col_e3.metric(
        "Municipios",
        municipios_detectados,
    )

    col_e4.metric(
        "Sin coordenadas",
        sin_coordenadas,
    )

    if estados_detectados:

        st.caption(
            "Estado(s) detectado(s): "
            + ", ".join(
                estados_detectados
            )
        )

    columnas_metadata = [
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

    with st.expander(
        "Ver estaciones y metadatos"
    ):

        st.dataframe(
            metadata_df[
                columnas_metadata
            ],
            use_container_width=True,
            hide_index=True,
        )

    archivos_no_compatibles = (
        log_lectura[
            log_lectura[
                "status"
            ] != "ok"
        ]
        if (
            log_lectura is not None
            and not log_lectura.empty
        )
        else pd.DataFrame()
    )

    if not archivos_no_compatibles.empty:

        with st.expander(
            "Archivos no compatibles"
        ):

            st.dataframe(
                archivos_no_compatibles,
                use_container_width=True,
                hide_index=True,
            )

else:

    st.info(
        "Indique una carpeta válida para detectar "
        "las estaciones."
    )


# ============================================================
# 3. PARÁMETROS DEL ANÁLISIS
# ============================================================

st.subheader(
    "3. Parámetros del análisis"
)

col_a, col_b = st.columns(
    2
)


# ------------------------------------------------------------
# COLUMNA IZQUIERDA
# ------------------------------------------------------------

with col_a:

    n_min_anios = st.number_input(
        "Mínimo recomendado de máximos anuales",
        min_value=2,
        max_value=100,
        value=10,
        step=1,
        help=(
            "Las estaciones con menos años no serán excluidas, "
            "pero se marcarán con una advertencia de alta "
            "incertidumbre."
        ),
        key="gev_min_years",
    )

    n_boot = st.number_input(
        "Réplicas Bootstrap A",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
        help=(
            "Número de remuestreos utilizados para construir "
            "el intervalo de confianza robusto."
        ),
        key="gev_n_boot",
    )

    alpha = st.number_input(
        "Nivel de significancia",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f",
        help=(
            "0.05 corresponde a un intervalo de confianza "
            "del 95%."
        ),
        key="gev_alpha",
    )


# ------------------------------------------------------------
# COLUMNA DERECHA
# ------------------------------------------------------------

with col_b:

    periodos_texto = st.text_input(
        "Periodos de retorno, separados por comas",
        value="2, 5, 10, 25, 50, 100",
        help=(
            "Los valores deben ser mayores que 1."
        ),
        key="gev_return_periods",
    )

    seed = st.number_input(
        "Semilla reproducible",
        min_value=0,
        value=42,
        step=1,
        help=(
            "Permite reproducir el proceso bootstrap."
        ),
        key="gev_seed",
    )

    st.text_input(
        "Método de incertidumbre",
        value="Bootstrap A robusto",
        disabled=True,
    )


# ============================================================
# VALIDAR PERIODOS DE RETORNO
# ============================================================

niveles_retorno = None
error_periodos = None


try:

    niveles_retorno = np.array(
        [
            float(
                valor.strip()
            )
            for valor
            in periodos_texto.split(",")
            if valor.strip()
        ],
        dtype=float,
    )

    if len(
        niveles_retorno
    ) == 0:

        raise ValueError(
            "Debe indicar al menos un periodo de retorno."
        )

    if not np.isfinite(
        niveles_retorno
    ).all():

        raise ValueError(
            "Los periodos de retorno deben ser valores "
            "numéricos finitos."
        )

    if np.any(
        niveles_retorno <= 1
    ):

        raise ValueError(
            "Todos los periodos de retorno deben ser "
            "mayores que 1."
        )

    niveles_retorno = np.unique(
        niveles_retorno
    )


except ValueError as error:

    error_periodos = str(
        error
    )

    st.error(
        f"Periodos de retorno no válidos: "
        f"{error_periodos}"
    )


# ============================================================
# RESUMEN DE CONFIGURACIÓN
# ============================================================

if niveles_retorno is not None:

    st.caption(
        "Periodos que serán analizados: "
        + ", ".join(
            f"{valor:g}"
            for valor in niveles_retorno
        )
        + " años."
    )


# ============================================================
# 4. EJECUCIÓN
# ============================================================

st.subheader(
    "4. Ejecutar análisis"
)


puede_ejecutar = (
    carpeta_valida
    and len(
        estaciones_lectura
    ) > 0
    and niveles_retorno is not None
    and error_periodos is None
)


if st.button(
    "🚀 Procesar estaciones GEV",
    type="primary",
    use_container_width=True,
    disabled=not puede_ejecutar,
):

    try:

        with st.spinner(
            f"Procesando {len(estaciones_lectura)} estaciones. "
            "El ajuste GEV y el bootstrap pueden tardar "
            "varios minutos..."
        ):

            (
                maestro,
                log_df,
                out_master,
                out_log,
                metadata_resultado,
            ) = ejecutar_proceso_batch_conagua(
                dir_in=str(
                    carpeta
                ),
                patron=patron,
                n_min_anios=int(
                    n_min_anios
                ),
                niveles_retorno=(
                    niveles_retorno
                ),
                n_boot=int(
                    n_boot
                ),
                alpha=float(
                    alpha
                ),
                seed=int(
                    seed
                ),

                # Bootstrap B se desactiva deliberadamente.
                usar_boot_parametrico=False,

                plot_max_t=float(
                    niveles_retorno.max()
                ),
            )

        st.session_state[
            "gev_batch_result"
        ] = {
            "maestro": maestro,
            "log_df": log_df,
            "out_master": out_master,
            "out_log": out_log,
            "metadata_df": metadata_resultado,
            "dir_in": str(
                carpeta
            ),
            "n_archivos": len(
                estaciones_lectura
            ),
            "n_boot": int(
                n_boot
            ),
            "alpha": float(
                alpha
            ),
            "niveles_retorno": (
                niveles_retorno.tolist()
            ),
            "n_min_anios": int(
                n_min_anios
            ),
        }

        if (
            maestro is not None
            and not maestro.empty
        ):

            st.success(
                "El procesamiento GEV terminó correctamente."
            )

        else:

            st.warning(
                "El proceso terminó, pero no se generó "
                "una tabla maestra. Revise el log."
            )

    except Exception as error:

        st.error(
            "No fue posible completar el análisis GEV."
        )

        st.exception(
            error
        )


# ============================================================
# 5. RESULTADOS
# ============================================================

resultado = st.session_state.get(
    "gev_batch_result"
)


if resultado is not None:

    maestro = resultado[
        "maestro"
    ]

    log_df = resultado[
        "log_df"
    ]

    out_master = resultado[
        "out_master"
    ]

    out_log = resultado[
        "out_log"
    ]

    n_boot_resultado = resultado[
        "n_boot"
    ]

    alpha_resultado = resultado[
        "alpha"
    ]

    n_min_resultado = resultado[
        "n_min_anios"
    ]


    st.divider()

    st.subheader(
        "5. Resultados"
    )


    # ========================================================
    # MÉTRICAS GENERALES
    # ========================================================

    total_estaciones = resultado[
        "n_archivos"
    ]

    estaciones_exitosas = (
        maestro[
            "station"
        ]
        .astype(str)
        .nunique()
        if (
            maestro is not None
            and not maestro.empty
        )
        else 0
    )

    if (
        log_df is not None
        and not log_df.empty
        and "status" in log_df.columns
    ):

        total_errores = int(
            (
                log_df[
                    "status"
                ] == "error"
            ).sum()
        )

        total_incompatibles = int(
            (
                log_df[
                    "status"
                ] == "incompatible"
            ).sum()
        )

    else:

        total_errores = 0
        total_incompatibles = 0


    col_m1, col_m2, col_m3, col_m4 = (
        st.columns(4)
    )

    col_m1.metric(
        "Estaciones detectadas",
        total_estaciones,
    )

    col_m2.metric(
        "Estaciones procesadas",
        estaciones_exitosas,
    )

    col_m3.metric(
        "Errores GEV",
        total_errores,
    )

    col_m4.metric(
        "Archivos incompatibles",
        total_incompatibles,
    )


    st.caption(
        f"Bootstrap A robusto: "
        f"{n_boot_resultado} réplicas · "
        f"IC {(1 - alpha_resultado) * 100:.0f}%."
    )


    # ========================================================
    # PESTAÑAS
    # ========================================================

    pestañas = st.tabs(
        [
            "Tabla maestra",
            "Resumen por estación",
            "Comparación",
            "Calidad estadística",
            "Log del proceso",
            "Descargas",
        ]
    )


    # ========================================================
    # TABLA MAESTRA
    # ========================================================

    with pestañas[0]:

        if (
            maestro is not None
            and not maestro.empty
        ):

            st.dataframe(
                maestro,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No existe una tabla maestra disponible."
            )


    # ========================================================
    # RESUMEN POR ESTACIÓN
    # ========================================================

    with pestañas[1]:

        if (
            maestro is not None
            and not maestro.empty
        ):

            estaciones_df = (
                maestro[
                    [
                        "station",
                        "nombre",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "station"
                )
            )

            opciones_estacion = {
                construir_etiqueta_estacion(
                    fila.station,
                    fila.nombre,
                ): fila.station
                for fila
                in estaciones_df.itertuples()
            }

            estacion_visible = st.selectbox(
                "Seleccionar estación",
                options=list(
                    opciones_estacion.keys()
                ),
                key="gev_station_result",
            )

            estacion = (
                opciones_estacion[
                    estacion_visible
                ]
            )

            datos_estacion = (
                maestro[
                    maestro[
                        "station"
                    ].astype(str)
                    == str(
                        estacion
                    )
                ]
                .copy()
                .sort_values(
                    "T_years"
                )
            )

            primera_fila = (
                datos_estacion.iloc[0]
            )


            # =================================================
            # IDENTIFICACIÓN
            # =================================================

            nombre = primera_fila.get(
                "nombre",
                "",
            )

            municipio = primera_fila.get(
                "municipio",
                "",
            )

            estado = primera_fila.get(
                "estado",
                "",
            )

            latitud = pd.to_numeric(
                primera_fila.get(
                    "latitud",
                    np.nan,
                ),
                errors="coerce",
            )

            longitud = pd.to_numeric(
                primera_fila.get(
                    "longitud",
                    np.nan,
                ),
                errors="coerce",
            )

            st.subheader(
                f"{estacion} — {nombre}"
            )

            descripcion = (
                f"{municipio}, {estado}"
            )

            if (
                np.isfinite(
                    latitud
                )
                and np.isfinite(
                    longitud
                )
            ):

                descripcion += (
                    f" · Lat. {latitud:.5f}"
                    f" · Lon. {longitud:.5f}"
                )

            st.caption(
                descripcion
            )


            # =================================================
            # PARÁMETROS GEV
            # =================================================

            shape = pd.to_numeric(
                primera_fila.get(
                    "gev_shape",
                    np.nan,
                ),
                errors="coerce",
            )

            loc = pd.to_numeric(
                primera_fila.get(
                    "gev_loc",
                    np.nan,
                ),
                errors="coerce",
            )

            scale = pd.to_numeric(
                primera_fila.get(
                    "gev_scale",
                    np.nan,
                ),
                errors="coerce",
            )

            n_years = pd.to_numeric(
                primera_fila.get(
                    "n_years",
                    np.nan,
                ),
                errors="coerce",
            )

            slope = pd.to_numeric(
                primera_fila.get(
                    "trend_slope_mm_per_year",
                    np.nan,
                ),
                errors="coerce",
            )

            boot_a = pd.to_numeric(
                primera_fila.get(
                    "bootA_naccepted",
                    np.nan,
                ),
                errors="coerce",
            )


            fila_metricas_1 = st.columns(
                4
            )

            fila_metricas_1[0].metric(
                "Parámetro de forma",
                (
                    f"{shape:.4f}"
                    if np.isfinite(
                        shape
                    )
                    else "N/D"
                ),
            )

            fila_metricas_1[1].metric(
                "Localización",
                (
                    f"{loc:.2f} mm"
                    if np.isfinite(
                        loc
                    )
                    else "N/D"
                ),
            )

            fila_metricas_1[2].metric(
                "Escala",
                (
                    f"{scale:.2f} mm"
                    if np.isfinite(
                        scale
                    )
                    else "N/D"
                ),
            )

            fila_metricas_1[3].metric(
                "Máximos anuales",
                (
                    f"{int(n_years)}"
                    if np.isfinite(
                        n_years
                    )
                    else "N/D"
                ),
            )


            fila_metricas_2 = st.columns(
                3
            )

            fila_metricas_2[0].metric(
                "Pendiente lineal",
                (
                    f"{slope:.3f} mm/año"
                    if np.isfinite(
                        slope
                    )
                    else "N/D"
                ),
            )

            fila_metricas_2[1].metric(
                "Bootstrap aceptado",
                (
                    f"{int(boot_a)} / "
                    f"{n_boot_resultado}"
                    if np.isfinite(
                        boot_a
                    )
                    else "N/D"
                ),
            )

            calidad_bootstrap, proporcion_boot = (
                interpretar_calidad_bootstrap(
                    aceptadas=boot_a,
                    solicitadas=n_boot_resultado,
                )
            )

            fila_metricas_2[2].metric(
                "Calidad Bootstrap",
                calidad_bootstrap,
            )


            # =================================================
            # ADVERTENCIAS
            # =================================================

            note = str(
                primera_fila.get(
                    "note",
                    "",
                )
            ).strip()

            if (
                note
                and note.lower()
                != "nan"
            ):

                st.warning(
                    note
                )


            if (
                proporcion_boot is not None
                and proporcion_boot < 0.50
            ):

                st.warning(
                    "El intervalo de confianza de esta estación "
                    "se obtuvo con menos del 50% de las réplicas "
                    "bootstrap solicitadas. Los límites deben "
                    "interpretarse con especial cautela."
                )

            elif (
                proporcion_boot is not None
                and proporcion_boot < 0.80
            ):

                st.info(
                    "El Bootstrap A descartó una proporción "
                    "relevante de réplicas inestables. "
                    "El intervalo mostrado corresponde únicamente "
                    "a los ajustes aceptados por los filtros robustos."
                )


            # =================================================
            # CURVA GEV
            # =================================================

            columnas_grafica = {
                "T_years",
                "level_mm",
                "CI_low95_bootA",
                "CI_high95_bootA",
            }

            if columnas_grafica.issubset(
                datos_estacion.columns
            ):

                grafica = datos_estacion[
                    [
                        "T_years",
                        "level_mm",
                        "CI_low95_bootA",
                        "CI_high95_bootA",
                    ]
                ].copy()

                for columna in grafica.columns:

                    grafica[
                        columna
                    ] = pd.to_numeric(
                        grafica[
                            columna
                        ],
                        errors="coerce",
                    )

                grafica = grafica.dropna(
                    subset=[
                        "T_years",
                        "level_mm",
                    ]
                )

                if not grafica.empty:

                    t = grafica[
                        "T_years"
                    ].to_numpy(
                        dtype=float
                    )

                    nivel = grafica[
                        "level_mm"
                    ].to_numpy(
                        dtype=float
                    )

                    low_a = grafica[
                        "CI_low95_bootA"
                    ].to_numpy(
                        dtype=float
                    )

                    high_a = grafica[
                        "CI_high95_bootA"
                    ].to_numpy(
                        dtype=float
                    )


                    fig, ax = plt.subplots(
                        figsize=(
                            10,
                            5.5,
                        )
                    )


                    # -----------------------------------------
                    # BANDA DE CONFIANZA
                    # -----------------------------------------

                    mascara_ic = (
                        np.isfinite(
                            low_a
                        )
                        & np.isfinite(
                            high_a
                        )
                    )

                    if mascara_ic.any():

                        ax.fill_between(
                            t[
                                mascara_ic
                            ],
                            low_a[
                                mascara_ic
                            ],
                            high_a[
                                mascara_ic
                            ],
                            alpha=0.25,
                            label=(
                                "IC 95% Bootstrap A robusto"
                            ),
                        )


                    # -----------------------------------------
                    # NIVEL PUNTUAL
                    # -----------------------------------------

                    ax.plot(
                        t,
                        nivel,
                        marker="o",
                        linewidth=2,
                        label=(
                            "Nivel de retorno GEV"
                        ),
                    )

                    ax.set_xscale(
                        "log"
                    )

                    ax.set_xticks(
                        t
                    )

                    ax.set_xticklabels(
                        [
                            f"{valor:g}"
                            for valor in t
                        ]
                    )

                    ax.set_xlabel(
                        "Periodo de retorno (años)"
                    )

                    ax.set_ylabel(
                        "Precipitación (mm)"
                    )

                    ax.set_title(
                        "Curva de niveles de retorno GEV — "
                        f"{estacion} — {nombre}"
                    )

                    ax.grid(
                        True,
                        which="both",
                        linestyle="--",
                        alpha=0.45,
                    )

                    ax.legend()

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(
                        fig
                    )


            # =================================================
            # TABLA DE NIVELES
            # =================================================

            st.subheader(
                "Niveles de retorno e intervalo robusto"
            )

            columnas_mostrar = [
                columna
                for columna in [
                    "T_years",
                    "level_mm",
                    "CI_low95_bootA",
                    "CI_high95_bootA",
                ]
                if columna
                in datos_estacion.columns
            ]

            tabla_visual = (
                datos_estacion[
                    columnas_mostrar
                ]
                .copy()
            )

            tabla_visual = (
                tabla_visual.rename(
                    columns={
                        "T_years": (
                            "Periodo de retorno (años)"
                        ),
                        "level_mm": (
                            "Nivel estimado (mm)"
                        ),
                        "CI_low95_bootA": (
                            "IC 95% inferior (mm)"
                        ),
                        "CI_high95_bootA": (
                            "IC 95% superior (mm)"
                        ),
                    }
                )
            )

            for columna in tabla_visual.columns:

                if columna != (
                    "Periodo de retorno (años)"
                ):

                    tabla_visual[
                        columna
                    ] = (
                        pd.to_numeric(
                            tabla_visual[
                                columna
                            ],
                            errors="coerce",
                        )
                        .round(
                            2
                        )
                    )

            st.dataframe(
                tabla_visual,
                use_container_width=True,
                hide_index=True,
            )


            # =================================================
            # DESCARGA INDIVIDUAL
            # =================================================

            csv_estacion = (
                datos_estacion
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                label=(
                    f"📥 Descargar resultados de "
                    f"{estacion} — {nombre}"
                ),
                data=csv_estacion,
                file_name=(
                    f"{estacion}_resultados_GEV.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        else:

            st.info(
                "No existen resultados por estación."
            )


    # ========================================================
    # COMPARACIÓN
    # ========================================================

    with pestañas[2]:

        if (
            maestro is not None
            and not maestro.empty
        ):

            periodos_disponibles = sorted(
                pd.to_numeric(
                    maestro[
                        "T_years"
                    ],
                    errors="coerce",
                )
                .dropna()
                .unique()
                .tolist()
            )

            periodo_comparacion = st.selectbox(
                "Periodo de retorno para comparar",
                options=periodos_disponibles,
                index=(
                    periodos_disponibles.index(
                        100.0
                    )
                    if 100.0
                    in periodos_disponibles
                    else len(
                        periodos_disponibles
                    )
                    - 1
                ),
                format_func=lambda valor: (
                    f"{valor:g} años"
                ),
                key="gev_comparison_period",
            )

            comparacion = (
                maestro[
                    pd.to_numeric(
                        maestro[
                            "T_years"
                        ],
                        errors="coerce",
                    )
                    == float(
                        periodo_comparacion
                    )
                ]
                .copy()
            )

            comparacion[
                "level_mm"
            ] = pd.to_numeric(
                comparacion[
                    "level_mm"
                ],
                errors="coerce",
            )

            comparacion = (
                comparacion
                .dropna(
                    subset=[
                        "level_mm"
                    ]
                )
                .sort_values(
                    "level_mm",
                    ascending=False,
                )
            )

            comparacion[
                "etiqueta"
            ] = (
                comparacion[
                    "station"
                ].astype(str)
                + " — "
                + comparacion[
                    "nombre"
                ].astype(str)
            )


            max_estaciones = len(
                comparacion
            )

            if max_estaciones > 1:

                numero_estaciones = st.slider(
                    "Número de estaciones a mostrar",
                    min_value=min(
                        5,
                        max_estaciones,
                    ),
                    max_value=max_estaciones,
                    value=min(
                        15,
                        max_estaciones,
                    ),
                    step=1,
                    key="gev_comparison_n",
                )

            else:

                numero_estaciones = 1


            top_estaciones = comparacion.head(
                numero_estaciones
            )


            fig, ax = plt.subplots(
                figsize=(
                    11,
                    6,
                )
            )

            ax.barh(
                top_estaciones[
                    "etiqueta"
                ],
                top_estaciones[
                    "level_mm"
                ],
            )

            ax.invert_yaxis()

            ax.set_xlabel(
                "Nivel de retorno (mm)"
            )

            ax.set_ylabel(
                "Estación"
            )

            ax.set_title(
                "Comparación de niveles de retorno — "
                f"T = {periodo_comparacion:g} años"
            )

            ax.grid(
                axis="x",
                linestyle="--",
                alpha=0.35,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(
                fig
            )


            columnas_comparacion = [
                columna
                for columna in [
                    "station",
                    "nombre",
                    "municipio",
                    "T_years",
                    "level_mm",
                    "CI_low95_bootA",
                    "CI_high95_bootA",
                    "n_years",
                    "bootA_naccepted",
                ]
                if columna
                in comparacion.columns
            ]

            st.dataframe(
                comparacion[
                    columnas_comparacion
                ],
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # CALIDAD ESTADÍSTICA
    # ========================================================

    with pestañas[3]:

        if (
            maestro is not None
            and not maestro.empty
        ):

            resumen_calidad = (
                maestro[
                    [
                        "station",
                        "nombre",
                        "municipio",
                        "n_years",
                        "gev_shape",
                        "gev_loc",
                        "gev_scale",
                        "trend_slope_mm_per_year",
                        "bootA_naccepted",
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
                .copy()
            )

            resumen_calidad[
                "bootA_porcentaje_aceptado"
            ] = (
                pd.to_numeric(
                    resumen_calidad[
                        "bootA_naccepted"
                    ],
                    errors="coerce",
                )
                / float(
                    n_boot_resultado
                )
                * 100
            ).round(
                1
            )

            st.dataframe(
                resumen_calidad,
                use_container_width=True,
                hide_index=True,
            )


            estaciones_pocos_anios = (
                resumen_calidad[
                    pd.to_numeric(
                        resumen_calidad[
                            "n_years"
                        ],
                        errors="coerce",
                    )
                    < n_min_resultado
                ]
            )

            boot_limitado = (
                resumen_calidad[
                    pd.to_numeric(
                        resumen_calidad[
                            "bootA_naccepted"
                        ],
                        errors="coerce",
                    )
                    < (
                        n_boot_resultado
                        * 0.50
                    )
                ]
            )


            col_q1, col_q2 = st.columns(
                2
            )

            col_q1.metric(
                "Cobertura temporal limitada",
                len(
                    estaciones_pocos_anios
                ),
            )

            col_q2.metric(
                "Bootstrap < 50% aceptado",
                len(
                    boot_limitado
                ),
            )


            if not estaciones_pocos_anios.empty:

                st.warning(
                    "Existen estaciones con menos máximos "
                    "anuales que el mínimo recomendado."
                )

                st.dataframe(
                    estaciones_pocos_anios[
                        [
                            "station",
                            "nombre",
                            "n_years",
                            "note",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


            if not boot_limitado.empty:

                st.warning(
                    "Las siguientes estaciones tienen menos "
                    "del 50% de réplicas Bootstrap A aceptadas. "
                    "Sus intervalos de confianza deben "
                    "interpretarse con especial cautela."
                )

                st.dataframe(
                    boot_limitado[
                        [
                            "station",
                            "nombre",
                            "bootA_naccepted",
                            "bootA_porcentaje_aceptado",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


    # ========================================================
    # LOG
    # ========================================================

    with pestañas[4]:

        if (
            log_df is not None
            and not log_df.empty
        ):

            st.dataframe(
                log_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No existe un log disponible."
            )


    # ========================================================
    # DESCARGAS
    # ========================================================

    with pestañas[5]:

        if (
            maestro is not None
            and not maestro.empty
        ):

            # Eliminamos las columnas Bootstrap B
            # del producto descargable principal.
            columnas_excluir = [
                "CI_low95_bootB",
                "CI_high95_bootB",
                "bootB_naccepted",
            ]

            maestro_descarga = (
                maestro.drop(
                    columns=[
                        columna
                        for columna
                        in columnas_excluir
                        if columna
                        in maestro.columns
                    ]
                )
                .copy()
            )

            csv_master = (
                maestro_descarga
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                "📥 Descargar tabla maestra GEV",
                data=csv_master,
                file_name=(
                    "MASTER_GEV_CONAGUA_"
                    "BootstrapA_robusto.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


        if (
            log_df is not None
            and not log_df.empty
        ):

            csv_log = (
                log_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                "📥 Descargar log del proceso",
                data=csv_log,
                file_name=(
                    "log_GEV_CONAGUA.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


        if out_master:

            st.caption(
                "Tabla maestra generada por el motor:"
            )

            st.code(
                out_master,
                language=None,
            )


        if out_log:

            st.caption(
                "Log generado por el motor:"
            )

            st.code(
                out_log,
                language=None,
            )