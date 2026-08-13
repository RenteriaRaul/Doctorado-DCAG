from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.conagua_reader import (
    leer_lote_conagua,
)
from scripts.exceedance import (
    procesar_excedencia_batch_conagua,
)
from scripts.raster_export import (
    exportar_desde_puntos_a_geotiff,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARPETA_SALIDA = (
    PROJECT_ROOT
    / "results"
    / "streamlit"
    / "excedencias"
)

CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True,
)

CRS = "EPSG:4326"
RESOLUCION = 300


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "exceedance_batch_result" not in st.session_state:
    st.session_state.exceedance_batch_result = None

if "exceedance_raster_path" not in st.session_state:
    st.session_state.exceedance_raster_path = None

if "exceedance_raster_metadata" not in st.session_state:
    st.session_state.exceedance_raster_metadata = None

if "exceedance_spatial_data" not in st.session_state:
    st.session_state.exceedance_spatial_data = None


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def construir_nombre_umbral(
    threshold,
):
    """
    Convierte el umbral en una cadena segura para nombres
    de archivo.

    50.0 -> 50
    75.5 -> 75_5
    """

    valor = float(
        threshold
    )

    if valor.is_integer():
        return str(
            int(valor)
        )

    return str(
        valor
    ).replace(
        ".",
        "_",
    )


def calcular_frecuencia_empirica(
    total_dias,
    dias_excedencia,
):
    """
    Calcula el número medio de días válidos por cada evento
    de excedencia.
    """

    if (
        pd.notna(total_dias)
        and pd.notna(dias_excedencia)
        and dias_excedencia > 0
    ):
        return (
            total_dias
            / dias_excedencia
        )

    return np.nan


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "Probabilidad de excedencia de precipitación"
)

st.write(
    """
    Este módulo calcula la probabilidad empírica de que la
    precipitación diaria de cada estación CONAGUA sea igual o
    superior a un umbral definido por el usuario.

    La plataforma trabaja directamente con los archivos Excel
    originales de estaciones CONAGUA. Los metadatos geográficos
    y las series climáticas se extraen automáticamente.
    """
)

st.info(
    """
    **Cálculo aplicado:** número de días con precipitación mayor
    o igual al umbral dividido entre el total de días válidos de
    cada estación.
    """
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
    key="conagua_source_folder",
)

patron = st.text_input(
    "Patrón de archivos",
    value="*.xlsx",
    help=(
        "Normalmente no es necesario modificar este valor."
    ),
    key="conagua_excel_pattern",
)


# ============================================================
# VALIDAR CARPETA
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

        carpeta_valida = True

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

            st.success(
                f"Se detectaron {total_ok} estaciones CONAGUA "
                "compatibles."
            )

            if total_no_ok > 0:

                st.warning(
                    f"{total_no_ok} archivo(s) de la carpeta no "
                    "corresponden al formato esperado de una estación "
                    "CONAGUA y serán ignorados en el análisis."
                )

        except Exception as error:

            st.error(
                "No fue posible leer la carpeta de estaciones."
            )

            st.exception(
                error
            )

            carpeta_valida = False


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

    municipios_detectados = (
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

    col_1, col_2, col_3, col_4 = st.columns(
        4
    )

    col_1.metric(
        "Estaciones",
        total_estaciones,
    )

    col_2.metric(
        "Estados detectados",
        len(
            estados_detectados
        ),
    )

    col_3.metric(
        "Municipios",
        municipios_detectados,
    )

    col_4.metric(
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

    if (
        log_lectura is not None
        and not log_lectura.empty
    ):

        archivos_no_compatibles = (
            log_lectura[
                log_lectura[
                    "status"
                ] != "ok"
            ]
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
        "Indique una carpeta válida para detectar las estaciones."
    )


# ============================================================
# 3. CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

st.subheader(
    "3. Configuración del análisis"
)

col_config_1, col_config_2 = st.columns(
    2
)

with col_config_1:

    threshold = st.number_input(
        "Umbral de precipitación",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=5.0,
        format="%.2f",
        help=(
            "Se contará como excedencia cada día con precipitación "
            "igual o superior a este valor."
        ),
        key="exceedance_threshold",
    )

with col_config_2:

    consolidar_duplicados = st.checkbox(
        "Consolidar fechas duplicadas",
        value=True,
        help=(
            "Cuando existe más de un registro para la misma fecha, "
            "se conserva la precipitación máxima diaria."
        ),
        key="exceedance_duplicates",
    )

st.caption(
    f"Se calculará la probabilidad de precipitación diaria "
    f"≥ {threshold:.2f} mm."
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
    ) >= 1
)

if st.button(
    "🚀 Procesar excedencias",
    type="primary",
    use_container_width=True,
    disabled=not puede_ejecutar,
):

    try:

        with st.spinner(
            "Calculando excedencias y preparando la superficie espacial..."
        ):

            # =================================================
            # EXCEDENCIAS
            # =================================================

            (
                resultados,
                log_df,
                out_master,
                out_log,
            ) = procesar_excedencia_batch_conagua(
                carpeta_estaciones=carpeta,
                patron=patron,
                threshold=float(
                    threshold
                ),
                consolidar_duplicados=(
                    consolidar_duplicados
                ),
                exportar=False,
            )

            if (
                resultados is None
                or resultados.empty
            ):
                raise ValueError(
                    "El análisis no produjo resultados."
                )

            # =================================================
            # GUARDAR TABLA MAESTRA
            # =================================================

            nombre_umbral = (
                construir_nombre_umbral(
                    threshold
                )
            )

            csv_master = (
                CARPETA_SALIDA
                / (
                    "MASTER_excedencia_CONAGUA_"
                    f"{nombre_umbral}mm.csv"
                )
            )

            csv_log = (
                CARPETA_SALIDA
                / (
                    "log_excedencia_CONAGUA_"
                    f"{nombre_umbral}mm.csv"
                )
            )

            resultados.to_csv(
                csv_master,
                index=False,
                encoding="utf-8-sig",
            )

            log_df.to_csv(
                csv_log,
                index=False,
                encoding="utf-8-sig",
            )

            # =================================================
            # PREPARAR SUPERFICIE ESPACIAL
            # =================================================

            datos_espaciales = (
                resultados
                .dropna(
                    subset=[
                        "longitud",
                        "latitud",
                        "prob_excedencia",
                    ]
                )
                .copy()
            )

            if len(
                datos_espaciales
            ) < 3:
                raise ValueError(
                    "Se requieren al menos tres estaciones "
                    "con coordenadas válidas para generar "
                    "la superficie espacial."
                )

            tif_path = (
                CARPETA_SALIDA
                / (
                    "excedencia_"
                    f"{nombre_umbral}mm_"
                    "interpolada.tif"
                )
            )

            # =================================================
            # GEOTIFF OBSERVACIONAL
            # =================================================

            resultado_tif = (
                exportar_desde_puntos_a_geotiff(
                    df=datos_espaciales,
                    out_tif=str(
                        tif_path
                    ),
                    col_lon="longitud",
                    col_lat="latitud",
                    col_val="prob_excedencia",
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

            # =================================================
            # SESSION STATE
            # =================================================

            st.session_state[
                "exceedance_batch_result"
            ] = {
                "resultados": resultados,
                "log_df": log_df,
                "out_master": str(
                    csv_master
                ),
                "out_log": str(
                    csv_log
                ),
                "n_archivos": len(
                    metadata_df
                ),
                "threshold": float(
                    threshold
                ),
                "dir_in": str(
                    carpeta
                ),
            }

            st.session_state[
                "exceedance_raster_path"
            ] = str(
                resultado_tif[
                    "out_tif"
                ]
            )

            st.session_state[
                "exceedance_spatial_data"
            ] = datos_espaciales

            st.session_state[
                "exceedance_raster_metadata"
            ] = {
                "threshold": float(
                    threshold
                ),
                "method": "linear",
                "nx": RESOLUCION,
                "ny": RESOLUCION,
                "margin": 0.0,
                "fill_nearest": False,
                "crs": CRS,
                "stations_valid": int(
                    resultado_tif[
                        "calidad"
                    ][
                        "total_estaciones_validas"
                    ]
                ),
                "source": "CONAGUA",
                "source_folder": str(
                    carpeta
                ),
            }

        st.success(
            "El análisis de excedencias y la superficie espacial "
            "se generaron correctamente."
        )

    except Exception as error:

        st.error(
            "No fue posible completar el análisis."
        )

        st.exception(
            error
        )


# ============================================================
# 5. RESULTADOS
# ============================================================

resultado_sesion = (
    st.session_state.exceedance_batch_result
)

if resultado_sesion is not None:

    resultados = resultado_sesion[
        "resultados"
    ]

    log_df = resultado_sesion[
        "log_df"
    ]

    out_master = resultado_sesion[
        "out_master"
    ]

    out_log = resultado_sesion[
        "out_log"
    ]

    total_archivos = resultado_sesion[
        "n_archivos"
    ]

    threshold_resultado = resultado_sesion[
        "threshold"
    ]

    st.divider()

    st.subheader(
        "5. Resultados"
    )

    # --------------------------------------------------------
    # MÉTRICAS GENERALES
    # --------------------------------------------------------

    if (
        log_df is not None
        and not log_df.empty
    ):

        total_errores = int(
            (
                log_df[
                    "status"
                ] == "error"
            ).sum()
        )

    else:

        total_errores = 0

    estaciones_exitosas = (
        resultados[
            "station"
        ].nunique()
        if (
            resultados is not None
            and not resultados.empty
        )
        else 0
    )

    prob_media = pd.to_numeric(
        resultados[
            "prob_excedencia"
        ],
        errors="coerce",
    ).mean()

    metrica_1, metrica_2, metrica_3, metrica_4 = (
        st.columns(
            4
        )
    )

    metrica_1.metric(
        "Estaciones detectadas",
        total_archivos,
    )

    metrica_2.metric(
        "Estaciones procesadas",
        estaciones_exitosas,
    )

    metrica_3.metric(
        "Errores",
        total_errores,
    )

    metrica_4.metric(
        "Excedencia promedio",
        (
            f"{prob_media * 100:.3f}%"
            if np.isfinite(
                prob_media
            )
            else "N/D"
        ),
    )

    st.caption(
        f"Umbral analizado: precipitación diaria "
        f"≥ {threshold_resultado:.2f} mm."
    )

    # ========================================================
    # PESTAÑAS
    # ========================================================

    pestañas = st.tabs(
        [
            "Tabla maestra",
            "Resumen por estación",
            "Comparación de estaciones",
            "Calidad y log",
            "Superficie espacial",
            "Descargas",
        ]
    )

    # ========================================================
    # TABLA MAESTRA
    # ========================================================

    with pestañas[0]:

        st.dataframe(
            resultados,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # RESUMEN POR ESTACIÓN
    # ========================================================

    with pestañas[1]:

        estaciones = (
            resultados[
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
            (
                f"{fila.station} — "
                f"{fila.nombre}"
            ): fila.station
            for fila
            in estaciones.itertuples()
        }

        estacion_visible = st.selectbox(
            "Seleccionar estación",
            options=list(
                opciones_estacion.keys()
            ),
            key="exceedance_station_result",
        )

        estacion = (
            opciones_estacion[
                estacion_visible
            ]
        )

        datos_estacion = (
            resultados[
                resultados[
                    "station"
                ].astype(str)
                == str(
                    estacion
                )
            ]
            .copy()
        )

        fila = datos_estacion.iloc[
            0
        ]

        total_dias = pd.to_numeric(
            fila[
                "total_dias"
            ],
            errors="coerce",
        )

        dias_excedencia = pd.to_numeric(
            fila[
                "dias_excedencia"
            ],
            errors="coerce",
        )

        prob_excedencia = pd.to_numeric(
            fila[
                "prob_excedencia"
            ],
            errors="coerce",
        )

        n_anios = pd.to_numeric(
            fila[
                "n_anios"
            ],
            errors="coerce",
        )

        frecuencia_dias = (
            calcular_frecuencia_empirica(
                total_dias=total_dias,
                dias_excedencia=(
                    dias_excedencia
                ),
            )
        )

        st.subheader(
            f"{fila['nombre']} — "
            f"Estación {fila['station']}"
        )

        st.caption(
            f"{fila['municipio']}, "
            f"{fila['estado']} · "
            f"Lat. {fila['latitud']:.5f} · "
            f"Lon. {fila['longitud']:.5f}"
        )

        metricas_1 = st.columns(
            4
        )

        metricas_1[0].metric(
            "Días válidos",
            f"{int(total_dias):,}",
        )

        metricas_1[1].metric(
            "Días con excedencia",
            f"{int(dias_excedencia):,}",
        )

        metricas_1[2].metric(
            "Probabilidad",
            f"{prob_excedencia * 100:.4f}%",
        )

        metricas_1[3].metric(
            "Años disponibles",
            (
                int(
                    n_anios
                )
                if np.isfinite(
                    n_anios
                )
                else "N/D"
            ),
        )

        metricas_2 = st.columns(
            3
        )

        metricas_2[0].metric(
            "Frecuencia empírica",
            (
                f"1 cada "
                f"{frecuencia_dias:,.0f} días"
                if np.isfinite(
                    frecuencia_dias
                )
                else "Sin eventos"
            ),
        )

        metricas_2[1].metric(
            "Inicio",
            pd.to_datetime(
                fila[
                    "fecha_inicio"
                ]
            ).strftime(
                "%Y-%m-%d"
            ),
        )

        metricas_2[2].metric(
            "Fin",
            pd.to_datetime(
                fila[
                    "fecha_fin"
                ]
            ).strftime(
                "%Y-%m-%d"
            ),
        )

        # ----------------------------------------------------
        # GRÁFICA
        # ----------------------------------------------------

        dias_sin_excedencia = (
            total_dias
            - dias_excedencia
        )

        fig, ax = plt.subplots(
            figsize=(
                8,
                4.5,
            )
        )

        categorias = [
            "Sin excedencia",
            "Con excedencia",
        ]

        valores = [
            dias_sin_excedencia,
            dias_excedencia,
        ]

        barras = ax.bar(
            categorias,
            valores,
        )

        ax.set_ylabel(
            "Número de días"
        )

        ax.set_title(
            f"Frecuencia de excedencia — "
            f"{fila['nombre']}"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35,
        )

        for barra, valor in zip(
            barras,
            valores,
        ):

            ax.text(
                barra.get_x()
                + barra.get_width()
                / 2,
                barra.get_height(),
                f"{int(valor):,}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(
            fig
        )

        st.dataframe(
            datos_estacion,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    with pestañas[2]:

        comparacion = resultados[
            [
                "station",
                "nombre",
                "municipio",
                "prob_excedencia",
                "dias_excedencia",
                "total_dias",
            ]
        ].copy()

        comparacion[
            "prob_excedencia"
        ] = pd.to_numeric(
            comparacion[
                "prob_excedencia"
            ],
            errors="coerce",
        )

        comparacion = (
            comparacion
            .dropna(
                subset=[
                    "prob_excedencia"
                ]
            )
            .sort_values(
                "prob_excedencia",
                ascending=False,
            )
        )

        max_estaciones = len(
            comparacion
        )

        if max_estaciones > 1:

            numero_estaciones = st.slider(
                "Número de estaciones a mostrar",
                min_value=5,
                max_value=max_estaciones,
                value=min(
                    15,
                    max_estaciones,
                ),
                step=1,
                key="exceedance_top_n",
            )

        else:

            numero_estaciones = 1

        top_estaciones = (
            comparacion.head(
                numero_estaciones
            )
            .copy()
        )

        top_estaciones[
            "etiqueta"
        ] = (
            top_estaciones[
                "station"
            ].astype(str)
            + " — "
            + top_estaciones[
                "nombre"
            ].astype(str)
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
            (
                top_estaciones[
                    "prob_excedencia"
                ]
                * 100
            ),
        )

        ax.invert_yaxis()

        ax.set_xlabel(
            "Probabilidad de excedencia (%)"
        )

        ax.set_ylabel(
            "Estación"
        )

        ax.set_title(
            "Estaciones con mayor excedencia "
            f"para ≥ {threshold_resultado:g} mm"
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

        comparacion[
            "prob_excedencia_pct"
        ] = (
            comparacion[
                "prob_excedencia"
            ]
            * 100
        ).round(
            4
        )

        st.dataframe(
            comparacion,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # CALIDAD Y LOG
    # ========================================================

    with pestañas[3]:

        columnas_calidad = [
            columna
            for columna in [
                "station",
                "nombre",
                "total_registros_originales",
                "registros_precipitacion_nulos",
                "registros_fecha_nulos",
                "registros_negativos",
                "duplicados_fecha_detectados",
                "total_dias_validos",
                "fecha_inicio",
                "fecha_fin",
                "n_anios",
            ]
            if columna in resultados.columns
        ]

        st.markdown(
            "#### Control de calidad por estación"
        )

        st.dataframe(
            resultados[
                columnas_calidad
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Log del procesamiento"
        )

        st.dataframe(
            log_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # SUPERFICIE ESPACIAL
    # ========================================================

    with pestañas[4]:

        raster_path = st.session_state.get(
            "exceedance_raster_path"
        )

        raster_metadata = (
            st.session_state.get(
                "exceedance_raster_metadata"
            )
        )

        if raster_path:

            st.success(
                "La superficie espacial está disponible "
                "para el módulo de mapas territoriales."
            )

            col_s1, col_s2, col_s3 = st.columns(
                3
            )

            col_s1.metric(
                "Resolución",
                (
                    f"{raster_metadata['nx']} × "
                    f"{raster_metadata['ny']}"
                ),
            )

            col_s2.metric(
                "Estaciones utilizadas",
                raster_metadata[
                    "stations_valid"
                ],
            )

            col_s3.metric(
                "Umbral",
                (
                    f"≥ "
                    f"{raster_metadata['threshold']:g} mm"
                ),
            )

            st.write(
                "**Método espacial**"
            )

            st.write(
                "Interpolación lineal mediante `griddata`, "
                "sin margen adicional y sin extrapolación "
                "por vecino más cercano."
            )

            st.code(
                raster_path,
                language=None,
            )

        else:

            st.info(
                "Todavía no existe una superficie espacial "
                "activa."
            )

    # ========================================================
    # DESCARGAS
    # ========================================================

    with pestañas[5]:

        csv_master_bytes = (
            resultados
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        csv_log_bytes = (
            log_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        col_d1, col_d2 = st.columns(
            2
        )

        with col_d1:

            st.download_button(
                "📥 Descargar tabla maestra",
                data=csv_master_bytes,
                file_name=(
                    "MASTER_excedencia_CONAGUA_"
                    f"{threshold_resultado:g}mm.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with col_d2:

            st.download_button(
                "📥 Descargar log",
                data=csv_log_bytes,
                file_name=(
                    "log_excedencia_CONAGUA_"
                    f"{threshold_resultado:g}mm.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        raster_path = st.session_state.get(
            "exceedance_raster_path"
        )

        if (
            raster_path
            and Path(
                raster_path
            ).exists()
        ):

            with open(
                raster_path,
                "rb",
            ) as archivo_tif:

                tif_bytes = (
                    archivo_tif.read()
                )

            st.download_button(
                "🌎 Descargar GeoTIFF interpolado",
                data=tif_bytes,
                file_name=Path(
                    raster_path
                ).name,
                mime="image/tiff",
                use_container_width=True,
            )

        st.caption(
            "Archivos locales de trabajo:"
        )

        st.code(
            out_master,
            language=None,
        )

        st.code(
            out_log,
            language=None,
        )