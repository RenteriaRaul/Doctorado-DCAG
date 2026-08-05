import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.exceedance import procesar_excedencia_batch_csv


# ============================================================
# ENCABEZADO
# ============================================================

st.title("Probabilidad de excedencia de precipitación")

st.write(
    """
    Este módulo calcula la probabilidad empírica de que la precipitación
    diaria de cada estación sea igual o superior a un umbral definido
    por el usuario.

    El análisis puede aplicarse a estaciones CONAGUA de cualquier estado,
    siempre que los archivos estén previamente descargados y preparados
    en formato CSV.
    """
)

st.info(
    """
    **Cálculo aplicado:** número de días con precipitación mayor o igual
    al umbral dividido entre el total de días válidos de la estación.
    """
)


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "exceedance_batch_result" not in st.session_state:
    st.session_state.exceedance_batch_result = None


# ============================================================
# 1. FUENTE DE DATOS
# ============================================================

st.subheader("1. Fuente de datos")

dir_in = st.text_input(
    "Ruta de la carpeta con los archivos de estaciones",
    placeholder=(
        r"C:\Usuarios\Usuario\Documentos\Estaciones_CONAGUA"
    ),
    help=(
        "La carpeta debe contener los archivos CSV de las estaciones "
        "que serán procesadas."
    ),
    key="exceedance_dir_input",
)

patron = st.text_input(
    "Patrón de búsqueda de archivos",
    value="dat*.csv",
    help=(
        "Ejemplo: dat*.csv localiza archivos como dat6001.csv, "
        "dat6002.csv y dat6003.csv."
    ),
    key="exceedance_pattern",
)


# ============================================================
# VERIFICACIÓN DE LA CARPETA
# ============================================================

archivos = []
carpeta_valida = False
carpeta = None

if dir_in.strip():
    carpeta = Path(
        dir_in.strip()
    ).expanduser()

    if carpeta.exists() and carpeta.is_dir():
        carpeta_valida = True

        archivos = sorted(
            glob.glob(
                os.path.join(
                    str(carpeta),
                    patron,
                )
            )
        )

        if archivos:
            st.success(
                f"Carpeta localizada. Se encontraron "
                f"{len(archivos)} archivos."
            )
        else:
            st.warning(
                "La carpeta existe, pero no se encontraron archivos "
                f"con el patrón `{patron}`."
            )

    else:
        st.error(
            "La ruta indicada no existe o no corresponde a una carpeta."
        )


# ============================================================
# 2. ESTRUCTURA DE LOS ARCHIVOS
# ============================================================

st.subheader("2. Estructura de los archivos")

columnas_disponibles = []
vista_previa = None

if archivos:
    archivo_muestra = archivos[0]

    try:
        vista_previa = pd.read_csv(
            archivo_muestra,
            nrows=10,
        )

        columnas_disponibles = list(
            vista_previa.columns
        )

        st.caption(
            "Archivo utilizado para verificar la estructura: "
            f"`{os.path.basename(archivo_muestra)}`"
        )

        with st.expander(
            "Ver muestra del archivo"
        ):
            st.dataframe(
                vista_previa,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.error(
            f"No fue posible leer el archivo de muestra: {error}"
        )


if columnas_disponibles:
    indice_fecha = (
        columnas_disponibles.index("date")
        if "date" in columnas_disponibles
        else 0
    )

    indice_pp = (
        columnas_disponibles.index("pp")
        if "pp" in columnas_disponibles
        else min(
            1,
            len(columnas_disponibles) - 1,
        )
    )

    col_fecha = st.selectbox(
        "Columna de fecha",
        options=columnas_disponibles,
        index=indice_fecha,
        key="exceedance_date_column",
    )

    col_precip = st.selectbox(
        "Columna de precipitación",
        options=columnas_disponibles,
        index=indice_pp,
        key="exceedance_precip_column",
    )

else:
    col_fecha = st.text_input(
        "Nombre de la columna de fecha",
        value="date",
        key="exceedance_date_text",
    )

    col_precip = st.text_input(
        "Nombre de la columna de precipitación",
        value="pp",
        key="exceedance_precip_text",
    )


# ============================================================
# 3. CONFIGURACIÓN DEL ANÁLISIS
# ============================================================

st.subheader("3. Configuración del análisis")

col_config_1, col_config_2 = st.columns(2)

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

st.subheader("4. Ejecutar análisis")

puede_ejecutar = (
    carpeta_valida
    and len(archivos) > 0
    and bool(col_fecha)
    and bool(col_precip)
    and threshold >= 0
)

if st.button(
    "🚀 Procesar excedencias",
    type="primary",
    use_container_width=True,
    disabled=not puede_ejecutar,
):
    try:
        with st.spinner(
            f"Procesando {len(archivos)} estaciones..."
        ):
            resultados, log_df, out_master, out_log = (
                procesar_excedencia_batch_csv(
                    carpeta_estaciones=str(carpeta),
                    patron=patron,
                    col_precip=col_precip,
                    threshold=float(threshold),
                    col_fecha=col_fecha,
                    consolidar_duplicados=(
                        consolidar_duplicados
                    ),
                    exportar=True,
                )
            )

        st.session_state.exceedance_batch_result = {
            "resultados": resultados,
            "log_df": log_df,
            "out_master": out_master,
            "out_log": out_log,
            "n_archivos": len(archivos),
            "threshold": float(threshold),
            "dir_in": str(carpeta),
        }

        if resultados is not None and not resultados.empty:
            st.success(
                "El análisis de excedencias terminó correctamente."
            )
        else:
            st.warning(
                "El proceso terminó, pero no se generaron resultados."
            )

    except Exception as error:
        st.exception(error)


# ============================================================
# 5. RESULTADOS
# ============================================================

resultado_sesion = (
    st.session_state.exceedance_batch_result
)

if resultado_sesion is not None:
    resultados = resultado_sesion["resultados"]
    log_df = resultado_sesion["log_df"]
    out_master = resultado_sesion["out_master"]
    out_log = resultado_sesion["out_log"]
    total_archivos = resultado_sesion["n_archivos"]
    threshold_resultado = resultado_sesion["threshold"]

    st.divider()

    st.subheader("5. Resultados")

    if log_df is not None and not log_df.empty:
        total_errores = int(
            (
                log_df["status"] == "error"
            ).sum()
        )
    else:
        total_errores = 0

    estaciones_exitosas = (
        resultados["station"].nunique()
        if (
            resultados is not None
            and not resultados.empty
            and "station" in resultados.columns
        )
        else 0
    )

    if (
        resultados is not None
        and not resultados.empty
        and "prob_excedencia" in resultados.columns
    ):
        prob_media = pd.to_numeric(
            resultados["prob_excedencia"],
            errors="coerce",
        ).mean()
    else:
        prob_media = np.nan

    metrica_1, metrica_2, metrica_3, metrica_4 = (
        st.columns(4)
    )

    metrica_1.metric(
        "Archivos localizados",
        total_archivos,
    )

    metrica_2.metric(
        "Estaciones procesadas",
        estaciones_exitosas,
    )

    metrica_3.metric(
        "Archivos con error",
        total_errores,
    )

    metrica_4.metric(
        "Excedencia promedio",
        (
            f"{prob_media * 100:.3f}%"
            if np.isfinite(prob_media)
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
            "Log del proceso",
            "Descargas",
        ]
    )

    # --------------------------------------------------------
    # TABLA MAESTRA
    # --------------------------------------------------------

    with pestañas[0]:
        if resultados is not None and not resultados.empty:
            st.dataframe(
                resultados,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No existe una tabla maestra disponible."
            )

    # --------------------------------------------------------
    # RESUMEN POR ESTACIÓN
    # --------------------------------------------------------

    with pestañas[1]:
        if resultados is not None and not resultados.empty:
            estaciones = sorted(
                resultados["station"]
                .dropna()
                .unique()
            )

            estacion = st.selectbox(
                "Seleccionar estación",
                estaciones,
                key="exceedance_station_result",
            )

            datos_estacion = (
                resultados[
                    resultados["station"] == estacion
                ]
                .copy()
            )

            fila = datos_estacion.iloc[0]

            total_dias = pd.to_numeric(
                fila.get(
                    "total_dias",
                    np.nan,
                ),
                errors="coerce",
            )

            dias_excedencia = pd.to_numeric(
                fila.get(
                    "dias_excedencia",
                    np.nan,
                ),
                errors="coerce",
            )

            prob_excedencia = pd.to_numeric(
                fila.get(
                    "prob_excedencia",
                    np.nan,
                ),
                errors="coerce",
            )

            n_anios = pd.to_numeric(
                fila.get(
                    "n_anios",
                    np.nan,
                ),
                errors="coerce",
            )

            fecha_inicio = fila.get(
                "fecha_inicio",
                pd.NaT,
            )

            fecha_fin = fila.get(
                "fecha_fin",
                pd.NaT,
            )

            registros_precipitacion_nulos = pd.to_numeric(
                fila.get(
                    "registros_precipitacion_nulos",
                    0,
                ),
                errors="coerce",
            )

            registros_fecha_nulos = pd.to_numeric(
                fila.get(
                    "registros_fecha_nulos",
                    0,
                ),
                errors="coerce",
            )

            registros_negativos = pd.to_numeric(
                fila.get(
                    "registros_negativos",
                    0,
                ),
                errors="coerce",
            )

            registros_invalidos = (
                np.nan_to_num(
                    registros_precipitacion_nulos,
                    nan=0.0,
                )
                + np.nan_to_num(
                    registros_fecha_nulos,
                    nan=0.0,
                )
                + np.nan_to_num(
                    registros_negativos,
                    nan=0.0,
                )
            )

            # =================================================
            # FRECUENCIA EMPÍRICA
            # =================================================

            if (
                np.isfinite(total_dias)
                and np.isfinite(dias_excedencia)
                and dias_excedencia > 0
            ):
                frecuencia_dias = (
                    total_dias / dias_excedencia
                )
            else:
                frecuencia_dias = np.nan

            st.subheader(
                f"Resultados de la estación {estacion}"
            )

            # =================================================
            # PRIMERA FILA DE MÉTRICAS
            # =================================================

            metricas_1 = st.columns(4)

            metricas_1[0].metric(
                "Días válidos",
                (
                    f"{int(total_dias):,}"
                    if np.isfinite(total_dias)
                    else "N/D"
                ),
            )

            metricas_1[1].metric(
                "Días con excedencia",
                (
                    f"{int(dias_excedencia):,}"
                    if np.isfinite(dias_excedencia)
                    else "N/D"
                ),
            )

            metricas_1[2].metric(
                "Probabilidad",
                (
                    f"{prob_excedencia * 100:.4f}%"
                    if np.isfinite(prob_excedencia)
                    else "N/D"
                ),
            )

            metricas_1[3].metric(
                "Años disponibles",
                (
                    f"{int(n_anios)}"
                    if np.isfinite(n_anios)
                    else "N/D"
                ),
            )

            # =================================================
            # SEGUNDA FILA DE MÉTRICAS
            # =================================================

            metricas_2 = st.columns(4)

            metricas_2[0].metric(
                "Frecuencia empírica",
                (
                    f"1 cada {frecuencia_dias:,.0f} días"
                    if np.isfinite(frecuencia_dias)
                    else "Sin eventos"
                ),
                help=(
                    "Representa el número promedio de días válidos "
                    "por cada día que alcanzó o superó el umbral. "
                    "Es una frecuencia descriptiva del registro histórico."
                ),
            )

            metricas_2[1].metric(
                "Inicio de registro",
                (
                    pd.to_datetime(
                        fecha_inicio
                    ).strftime("%Y-%m-%d")
                    if pd.notna(fecha_inicio)
                    else "N/D"
                ),
            )

            metricas_2[2].metric(
                "Fin de registro",
                (
                    pd.to_datetime(
                        fecha_fin
                    ).strftime("%Y-%m-%d")
                    if pd.notna(fecha_fin)
                    else "N/D"
                ),
            )

            metricas_2[3].metric(
                "Registros inválidos",
                (
                    f"{int(registros_invalidos)}"
                    if np.isfinite(registros_invalidos)
                    else "N/D"
                ),
            )

            # =================================================
            # INTERPRETACIÓN DE LA FRECUENCIA
            # =================================================

            if np.isfinite(frecuencia_dias):
                frecuencia_texto = (
                    "Esto equivale aproximadamente a un día con "
                    f"excedencia por cada {frecuencia_dias:,.0f} "
                    "días válidos observados."
                )
            else:
                frecuencia_texto = (
                    "No se registraron excedencias para el "
                    "umbral seleccionado."
                )

            if (
                np.isfinite(dias_excedencia)
                and np.isfinite(total_dias)
            ):
                st.info(
                    f"En la estación **{estacion}**, "
                    f"{int(dias_excedencia):,} de "
                    f"{int(total_dias):,} días válidos registraron "
                    "precipitación igual o superior a "
                    f"**{threshold_resultado:.2f} mm**. "
                    f"{frecuencia_texto}"
                )

            st.caption(
                "La frecuencia empírica es una medida descriptiva. "
                "No significa que el evento ocurra regularmente cada "
                "cierto número exacto de días."
            )

            # =================================================
            # GRÁFICA RESUMEN DE LA ESTACIÓN
            # =================================================

            if (
                np.isfinite(total_dias)
                and np.isfinite(dias_excedencia)
                and total_dias > 0
            ):
                dias_sin_excedencia = (
                    total_dias - dias_excedencia
                )

                fig, ax = plt.subplots(
                    figsize=(8, 4.5)
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
                    f"Frecuencia de excedencia — {estacion}"
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
                        + barra.get_width() / 2,
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

                plt.close(fig)

            # =================================================
            # TABLA DE LA ESTACIÓN
            # =================================================

            st.dataframe(
                datos_estacion,
                use_container_width=True,
                hide_index=True,
            )

            # =================================================
            # DESCARGA INDIVIDUAL
            # =================================================

            csv_estacion = (
                datos_estacion
                .to_csv(index=False)
                .encode("utf-8-sig")
            )

            st.download_button(
                label=(
                    f"📥 Descargar resultados de {estacion}"
                ),
                data=csv_estacion,
                file_name=(
                    f"{estacion}_excedencia_"
                    f"{threshold_resultado:g}mm.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info(
                "No existen resultados por estación."
            )

    # --------------------------------------------------------
    # COMPARACIÓN ENTRE ESTACIONES
    # --------------------------------------------------------

    with pestañas[2]:
        if resultados is not None and not resultados.empty:
            comparacion = resultados[
                [
                    "station",
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

            if not comparacion.empty:
                max_estaciones = min(
                    30,
                    len(comparacion),
                )

                min_estaciones = min(
                    5,
                    max_estaciones,
                )

                valor_inicial = min(
                    15,
                    max_estaciones,
                )

                if max_estaciones > 1:
                    numero_estaciones = st.slider(
                        "Número de estaciones a mostrar",
                        min_value=min_estaciones,
                        max_value=max_estaciones,
                        value=max(
                            min_estaciones,
                            valor_inicial,
                        ),
                        step=1,
                        key="exceedance_top_n",
                    )
                else:
                    numero_estaciones = 1

                top_estaciones = comparacion.head(
                    numero_estaciones
                )

                fig, ax = plt.subplots(
                    figsize=(11, 6)
                )

                ax.barh(
                    top_estaciones["station"],
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

                plt.close(fig)

                tabla_comparacion = (
                    comparacion.copy()
                )

                tabla_comparacion[
                    "prob_excedencia_pct"
                ] = (
                    tabla_comparacion[
                        "prob_excedencia"
                    ]
                    * 100
                ).round(4)

                tabla_comparacion[
                    "frecuencia_dias"
                ] = np.where(
                    tabla_comparacion[
                        "dias_excedencia"
                    ] > 0,
                    (
                        tabla_comparacion[
                            "total_dias"
                        ]
                        / tabla_comparacion[
                            "dias_excedencia"
                        ]
                    ),
                    np.nan,
                )

                tabla_comparacion[
                    "frecuencia_dias"
                ] = tabla_comparacion[
                    "frecuencia_dias"
                ].round(0)

                st.dataframe(
                    tabla_comparacion,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No hay probabilidades válidas para comparar."
                )

        else:
            st.info(
                "No existen resultados para comparar."
            )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    with pestañas[3]:
        if log_df is not None and not log_df.empty:
            st.dataframe(
                log_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No existe un log disponible."
            )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------

    with pestañas[4]:
        if resultados is not None and not resultados.empty:
            csv_master = (
                resultados
                .to_csv(index=False)
                .encode("utf-8-sig")
            )

            st.download_button(
                "📥 Descargar tabla maestra",
                data=csv_master,
                file_name=(
                    f"MASTER_excedencia_"
                    f"{threshold_resultado:g}mm.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        if log_df is not None and not log_df.empty:
            csv_log = (
                log_df
                .to_csv(index=False)
                .encode("utf-8-sig")
            )

            st.download_button(
                "📥 Descargar log del proceso",
                data=csv_log,
                file_name=(
                    f"log_excedencia_"
                    f"{threshold_resultado:g}mm.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        if out_master:
            st.caption(
                "Tabla maestra guardada localmente en:"
            )

            st.code(
                out_master,
                language=None,
            )

        if out_log:
            st.caption(
                "Log guardado localmente en:"
            )

            st.code(
                out_log,
                language=None,
            )