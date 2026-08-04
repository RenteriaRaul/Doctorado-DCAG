import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scripts.batch_return_levels import ejecutar_proceso_batch


# ============================================================
# ENCABEZADO
# ============================================================

st.title("📈 Análisis GEV y periodos de retorno")

st.write(
    """
    Este módulo procesa archivos históricos de estaciones meteorológicas
    previamente descargados y preparados por el usuario. Puede utilizarse
    con estaciones CONAGUA de cualquier entidad federativa, siempre que los
    archivos contengan una columna de fecha y otra de precipitación.
    """
)

st.info(
    """
    **Flujo del análisis:** limpieza de datos → máximos anuales →
    ajuste GEV → niveles de retorno → intervalos de confianza bootstrap.
    """
)


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "gev_batch_result" not in st.session_state:
    st.session_state.gev_batch_result = None


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
        "La plataforma buscará los archivos dentro de esta carpeta. "
        "La ruta debe existir en la computadora donde se ejecuta Streamlit."
    ),
)

patron = st.text_input(
    "Patrón de búsqueda de archivos",
    value="dat*.csv",
    help=(
        "Ejemplo: dat*.csv encuentra dat6001.csv, dat6002.csv, etc. "
        "También puede utilizar *.csv."
    ),
)


# ============================================================
# VERIFICACIÓN DE CARPETA Y ARCHIVOS
# ============================================================

archivos = []
carpeta_valida = False

if dir_in.strip():
    carpeta = Path(dir_in.strip()).expanduser()

    if carpeta.exists() and carpeta.is_dir():
        carpeta_valida = True

        archivos = sorted(
            glob.glob(
                os.path.join(str(carpeta), patron)
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
archivo_muestra = None
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
            f"Archivo utilizado para verificar la estructura: "
            f"`{os.path.basename(archivo_muestra)}`"
        )

        with st.expander("Ver muestra del archivo"):
            st.dataframe(
                vista_previa,
                use_container_width=True,
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
        else min(1, len(columnas_disponibles) - 1)
    )

    col_fecha = st.selectbox(
        "Columna de fecha",
        options=columnas_disponibles,
        index=indice_fecha,
    )

    col_pp = st.selectbox(
        "Columna de precipitación",
        options=columnas_disponibles,
        index=indice_pp,
    )

else:
    col_fecha = st.text_input(
        "Nombre de la columna de fecha",
        value="date",
    )

    col_pp = st.text_input(
        "Nombre de la columna de precipitación",
        value="pp",
    )


# ============================================================
# 3. PARÁMETROS ESTADÍSTICOS
# ============================================================

st.subheader("3. Parámetros del análisis")

col_a, col_b = st.columns(2)

with col_a:
    n_min_anios = st.number_input(
        "Mínimo recomendado de años",
        min_value=2,
        max_value=100,
        value=10,
        step=1,
    )

    n_boot = st.number_input(
        "Réplicas bootstrap",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )

    alpha = st.number_input(
        "Nivel de significancia",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f",
    )

with col_b:
    periodos_texto = st.text_input(
        "Periodos de retorno, separados por comas",
        value="2, 5, 10, 25, 50, 100",
    )

    seed = st.number_input(
        "Semilla reproducible",
        min_value=0,
        value=42,
        step=1,
    )

    usar_boot_parametrico = st.checkbox(
        "Calcular también bootstrap paramétrico",
        value=True,
    )


# ============================================================
# VALIDACIÓN DE PERIODOS
# ============================================================

niveles_retorno = None
error_periodos = None

try:
    niveles_retorno = np.array(
        [
            float(valor.strip())
            for valor in periodos_texto.split(",")
            if valor.strip()
        ],
        dtype=float,
    )

    if len(niveles_retorno) == 0:
        raise ValueError(
            "Debe indicar al menos un periodo de retorno."
        )

    if np.any(niveles_retorno <= 1):
        raise ValueError(
            "Todos los periodos de retorno deben ser mayores que 1."
        )

    niveles_retorno = np.unique(
        niveles_retorno
    )

except ValueError as error:
    error_periodos = str(error)
    st.error(
        f"Periodos de retorno no válidos: {error_periodos}"
    )


# ============================================================
# 4. EJECUCIÓN
# ============================================================

st.subheader("4. Ejecutar análisis")

puede_ejecutar = (
    carpeta_valida
    and len(archivos) > 0
    and niveles_retorno is not None
    and error_periodos is None
    and bool(col_fecha)
    and bool(col_pp)
)

if st.button(
    "🚀 Procesar estaciones",
    type="primary",
    use_container_width=True,
    disabled=not puede_ejecutar,
):
    try:
        with st.spinner(
            f"Procesando {len(archivos)} estaciones. "
            "Este proceso puede tardar varios minutos..."
        ):
            maestro, log_df, out_master, out_log = (
                ejecutar_proceso_batch(
                    dir_in=str(carpeta),
                    patron=patron,
                    col_fecha=col_fecha,
                    col_pp=col_pp,
                    n_min_anios=int(n_min_anios),
                    niveles_retorno=niveles_retorno,
                    n_boot=int(n_boot),
                    alpha=float(alpha),
                    seed=int(seed),
                    usar_boot_parametrico=usar_boot_parametrico,
                    plot_max_t=float(
                        niveles_retorno.max()
                    ),
                )
            )

        st.session_state.gev_batch_result = {
            "maestro": maestro,
            "log_df": log_df,
            "out_master": out_master,
            "out_log": out_log,
            "dir_in": str(carpeta),
            "n_archivos": len(archivos),
        }

        if maestro is not None:
            st.success(
                "El procesamiento GEV terminó correctamente."
            )
        else:
            st.warning(
                "El proceso terminó, pero no se generó una tabla maestra. "
                "Revise el log de procesamiento."
            )

    except Exception as error:
        st.exception(error)


# ============================================================
# 5. RESULTADOS
# ============================================================

resultado = st.session_state.gev_batch_result

if resultado is not None:
    maestro = resultado["maestro"]
    log_df = resultado["log_df"]
    out_master = resultado["out_master"]
    out_log = resultado["out_log"]

    st.divider()
    st.subheader("5. Resultados")

    total_archivos = resultado["n_archivos"]

    if log_df is not None and not log_df.empty:
        if "error" in log_df.columns:
            total_errores = int(
                log_df["error"].notna().sum()
            )
        else:
            total_errores = 0
    else:
        total_errores = 0

    estaciones_exitosas = (
        maestro["station"].nunique()
        if maestro is not None
        and not maestro.empty
        and "station" in maestro.columns
        else 0
    )

    metrica_1, metrica_2, metrica_3 = st.columns(3)

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

    pestañas = st.tabs(
        [
            "Tabla maestra",
            "Log del proceso",
            "Resumen por estación",
            "Descargas",
        ]
    )

    # --------------------------------------------------------
    # TABLA MAESTRA
    # --------------------------------------------------------

    with pestañas[0]:
        if maestro is not None and not maestro.empty:
            st.dataframe(
                maestro,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No existe una tabla maestra disponible."
            )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    with pestañas[1]:
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
    # RESUMEN POR ESTACIÓN
    # --------------------------------------------------------

    with pestañas[2]:

        if maestro is not None and not maestro.empty:

            estaciones = sorted(
                maestro["station"].dropna().unique()
            )

            estacion = st.selectbox(
                "Seleccionar estación",
                estaciones,
                key="gev_station_result",
            )

            datos_estacion = (
                maestro[
                    maestro["station"] == estacion
                ]
                .copy()
                .sort_values("T_years")
            )

            # =================================================
            # PARÁMETROS PRINCIPALES
            # =================================================

            primera_fila = datos_estacion.iloc[0]

            shape = primera_fila.get(
                "gev_shape",
                np.nan,
            )

            loc = primera_fila.get(
                "gev_loc",
                np.nan,
            )

            scale = primera_fila.get(
                "gev_scale",
                np.nan,
            )

            n_years = primera_fila.get(
                "n_years",
                np.nan,
            )

            slope = primera_fila.get(
                "trend_slope_mm_per_year",
                np.nan,
            )

            boot_a = primera_fila.get(
                "bootA_naccepted",
                np.nan,
            )

            boot_b = primera_fila.get(
                "bootB_naccepted",
                np.nan,
            )

            st.subheader(
                f"Resultados de la estación {estacion}"
            )

            fila_metricas_1 = st.columns(4)

            fila_metricas_1[0].metric(
                "Parámetro de forma",
                (
                    f"{shape:.4f}"
                    if np.isfinite(shape)
                    else "N/D"
                ),
            )

            fila_metricas_1[1].metric(
                "Localización",
                (
                    f"{loc:.2f} mm"
                    if np.isfinite(loc)
                    else "N/D"
                ),
            )

            fila_metricas_1[2].metric(
                "Escala",
                (
                    f"{scale:.2f} mm"
                    if np.isfinite(scale)
                    else "N/D"
                ),
            )

            fila_metricas_1[3].metric(
                "Años analizados",
                (
                    f"{int(n_years)}"
                    if np.isfinite(n_years)
                    else "N/D"
                ),
            )

            fila_metricas_2 = st.columns(3)

            fila_metricas_2[0].metric(
                "Pendiente lineal",
                (
                    f"{slope:.3f} mm/año"
                    if np.isfinite(slope)
                    else "N/D"
                ),
            )

            fila_metricas_2[1].metric(
                "Bootstrap robusto aceptado",
                (
                    f"{int(boot_a)}"
                    if np.isfinite(boot_a)
                    else "N/D"
                ),
            )

            fila_metricas_2[2].metric(
                "Bootstrap paramétrico aceptado",
                (
                    f"{int(boot_b)}"
                    if np.isfinite(boot_b)
                    else "N/D"
                ),
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

            if note and note.lower() != "nan":
                st.warning(note)

            columnas_boot_b = {
                "level_mm",
                "CI_low95_bootB",
                "CI_high95_bootB",
            }

            if columnas_boot_b.issubset(
                datos_estacion.columns
            ):

                nivel = pd.to_numeric(
                    datos_estacion["level_mm"],
                    errors="coerce",
                )

                high_b = pd.to_numeric(
                    datos_estacion["CI_high95_bootB"],
                    errors="coerce",
                )

                mascara_inestable = (
                    high_b.notna()
                    & nivel.notna()
                    & (high_b > nivel * 10)
                )

                if mascara_inestable.any():

                    periodos_inestables = (
                        datos_estacion.loc[
                            mascara_inestable,
                            "T_years",
                        ]
                        .astype(str)
                        .tolist()
                    )

                    st.warning(
                        "El bootstrap paramétrico presenta "
                        "límites superiores extremadamente altos "
                        "para los periodos de retorno: "
                        + ", ".join(periodos_inestables)
                        + " años. Para esta estación se recomienda "
                        "priorizar el intervalo robusto Bootstrap A."
                    )

            # =================================================
            # GRÁFICA GEV
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
                    grafica[columna] = pd.to_numeric(
                        grafica[columna],
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
                    ].to_numpy(dtype=float)

                    nivel = grafica[
                        "level_mm"
                    ].to_numpy(dtype=float)

                    low_a = grafica[
                        "CI_low95_bootA"
                    ].to_numpy(dtype=float)

                    high_a = grafica[
                        "CI_high95_bootA"
                    ].to_numpy(dtype=float)

                    fig, ax = plt.subplots(
                        figsize=(10, 5.5)
                    )

                    # Banda de confianza
                    mascara_ic = (
                        np.isfinite(low_a)
                        & np.isfinite(high_a)
                    )

                    if mascara_ic.any():

                        ax.fill_between(
                            t[mascara_ic],
                            low_a[mascara_ic],
                            high_a[mascara_ic],
                            alpha=0.25,
                            label=(
                                "IC 95% Bootstrap robusto"
                            ),
                        )

                    # Nivel puntual
                    ax.plot(
                        t,
                        nivel,
                        marker="o",
                        linewidth=2,
                        label="Nivel de retorno GEV",
                    )

                    ax.set_xscale("log")

                    ax.set_xticks(t)

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
                        f"Curva de niveles de retorno GEV — "
                        f"{estacion}"
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

                    plt.close(fig)

                else:
                    st.warning(
                        "No hay valores numéricos suficientes "
                        "para generar la curva GEV."
                    )

            else:
                st.warning(
                    "No se encontraron todas las columnas "
                    "necesarias para generar la curva GEV."
                )

            # =================================================
            # TABLA DE RESULTADOS
            # =================================================

            st.subheader(
                "Niveles de retorno e intervalos de confianza"
            )

            columnas_mostrar = [
                columna
                for columna in [
                    "T_years",
                    "level_mm",
                    "CI_low95_bootA",
                    "CI_high95_bootA",
                    "CI_low95_bootB",
                    "CI_high95_bootB",
                ]
                if columna in datos_estacion.columns
            ]

            tabla_visual = datos_estacion[
                columnas_mostrar
            ].copy()

            nombres_columnas = {
                "T_years": "Periodo de retorno (años)",
                "level_mm": "Nivel estimado (mm)",
                "CI_low95_bootA": "IC inferior Boot A",
                "CI_high95_bootA": "IC superior Boot A",
                "CI_low95_bootB": "IC inferior Boot B",
                "CI_high95_bootB": "IC superior Boot B",
            }

            tabla_visual = tabla_visual.rename(
                columns=nombres_columnas
            )

            columnas_numericas = [
                columna
                for columna in tabla_visual.columns
                if columna
                != "Periodo de retorno (años)"
            ]

            for columna in columnas_numericas:
                tabla_visual[columna] = (
                    pd.to_numeric(
                        tabla_visual[columna],
                        errors="coerce",
                    )
                    .round(2)
                )

            st.dataframe(
                tabla_visual,
                use_container_width=True,
                hide_index=True,
            )

            # =================================================
            # DESCARGAS INDIVIDUALES
            # =================================================

            st.subheader(
                "Descargas de la estación"
            )

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
                    f"{estacion}_resultados_GEV.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info(
                "No existen resultados por estación."
            )