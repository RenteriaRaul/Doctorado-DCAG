from pathlib import Path

import pandas as pd

import streamlit as st

from scripts.sustax.inventory import (

    build_sustax_inventory,

    build_group_summary,

)

from scripts.sustax.merge import (

    build_long_total_from_inventory,

    build_by_scenario_total,

)

from scripts.sustax.projection import (

    detect_projection_coverage,

    summarize_projection_coverage,

    evaluate_projection_period,

    build_climate_change_projection,

)

# ============================================================

# CONFIGURACIÓN

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpetas candidatas donde podrían encontrarse los CSV

# originales de Sustax dentro del proyecto.

CARPETAS_SUSTAX_CANDIDATAS = [

    PROJECT_ROOT / "SUSTAX",

    PROJECT_ROOT / "Sustax",

    PROJECT_ROOT / "sustax",

    PROJECT_ROOT / "data" / "SUSTAX",

    PROJECT_ROOT / "data" / "Sustax",

    PROJECT_ROOT / "data" / "sustax",

]

SUSTAX_RESULTS_DIR = PROJECT_ROOT / "results" / "streamlit" / "sustax"

SUSTAX_LONG_DIR = SUSTAX_RESULTS_DIR / "LONG"

SUSTAX_BY_SCENARIO_DIR = SUSTAX_RESULTS_DIR / "BY_SCENARIO"

SUSTAX_PROJECTION_DIR = SUSTAX_RESULTS_DIR / "PROJECTIONS"

# ============================================================

# HELPERS

# ============================================================

def detectar_carpeta_sustax() -> Path | None:

    """

    Busca automáticamente una carpeta del proyecto que contenga

    archivos CSV originales de Sustax.

    No modifica ni procesa los archivos.

    """

    for carpeta in CARPETAS_SUSTAX_CANDIDATAS:

        if not carpeta.exists() or not carpeta.is_dir():

            continue

        archivos = [

            fp

            for fp in carpeta.glob("*.csv")

            if fp.is_file()

        ]

        if archivos:

            return carpeta

    return None

def extraer_escenarios(inventory_df: pd.DataFrame) -> list[str]:

    """

    Recupera la lista única de escenarios encontrados en el

    inventario Sustax.

    """

    if inventory_df.empty:

        return []

    if "scenarios_detected" not in inventory_df.columns:

        return []

    escenarios = set()

    for valor in inventory_df["scenarios_detected"].dropna():

        for escenario in str(valor).split(","):

            escenario = escenario.strip()

            if escenario:

                escenarios.add(escenario)

    orden_preferido = [

        "ERA5",

        "SSP119",

        "SSP126",

        "SSP245",

        "SSP370",

        "SSP434",

        "SSP460",

        "SSP585",

    ]

    ordenados = [

        escenario

        for escenario in orden_preferido

        if escenario in escenarios

    ]

    otros = sorted(

        escenarios.difference(orden_preferido)

    )

    return ordenados + otros

def obtener_periodo_archivo(archivo: Path) -> tuple[pd.Timestamp, pd.Timestamp]:

    """

    Obtiene fecha inicial y final utilizando el lector científico

    existente de Sustax.

    Se importa aquí para mantener la lógica de lectura centralizada

    en scripts/sustax/loader.py.

    """

    from scripts.sustax.loader import load_sustax_file_safe

    df_vals, _ = load_sustax_file_safe(

        str(archivo),

        return_pandas_df=True,

        return_metadata=False,

    )

    if df_vals is None or df_vals.empty:

        return pd.NaT, pd.NaT

    fechas = pd.to_datetime(

        df_vals.index,

        errors="coerce",

    )

    fechas = fechas[~pd.isna(fechas)]

    if len(fechas) == 0:

        return pd.NaT, pd.NaT

    return fechas.min(), fechas.max()

def obtener_periodo_global(

    inventory_df: pd.DataFrame,

) -> tuple[pd.Timestamp, pd.Timestamp, list[dict]]:

    """

    Calcula el periodo temporal global de los archivos compatibles.

    También devuelve una lista de errores de lectura de fechas,

    sin interrumpir la interfaz.

    """

    if inventory_df.empty:

        return pd.NaT, pd.NaT, []

    fechas_inicio = []

    fechas_fin = []

    errores = []

    for _, row in inventory_df.iterrows():

        archivo = Path(str(row["path"]))

        try:

            inicio, fin = obtener_periodo_archivo(archivo)

            if pd.notna(inicio):

                fechas_inicio.append(inicio)

            if pd.notna(fin):

                fechas_fin.append(fin)

        except Exception as exc:

            errores.append(

                {

                    "archivo": archivo.name,

                    "error": str(exc),

                }

            )

    fecha_inicio = (

        min(fechas_inicio)

        if fechas_inicio

        else pd.NaT

    )

    fecha_fin = (

        max(fechas_fin)

        if fechas_fin

        else pd.NaT

    )

    return fecha_inicio, fecha_fin, errores

def contar_puntos_con_coordenadas(

    group_summary_df: pd.DataFrame,

) -> int:

    """

    Cuenta los puntos Sustax con latitud y longitud válidas.

    """

    if group_summary_df.empty:

        return 0

    if not {"lat", "lon"}.issubset(group_summary_df.columns):

        return 0

    coords = group_summary_df[["lat", "lon"]].copy()

    coords["lat"] = pd.to_numeric(

        coords["lat"],

        errors="coerce",

    )

    coords["lon"] = pd.to_numeric(

        coords["lon"],

        errors="coerce",

    )

    return int(

        coords.dropna(

            subset=["lat", "lon"]

        ).shape[0]

    )

def preparar_tabla_puntos(
    inventory_df: pd.DataFrame,
    group_summary_df: pd.DataFrame
) -> pd.DataFrame:
    if group_summary_df.empty:
        return pd.DataFrame()
    tabla = group_summary_df.copy()
    if "punto_id" not in tabla.columns:
        tabla["punto_id"] = tabla.apply(
            lambda r: f"{float(r['lat']):.4f}_{float(r['lon']):.4f}",
            axis=1
        )
    if "scenarios" in tabla.columns:
        tabla["escenarios"] = tabla["scenarios"]
    elif "escenarios" not in tabla.columns:
        tabla["escenarios"] = "No identificado"
    tabla = tabla.rename(columns={
        "punto_id": "Punto Sustax",
        "lat": "Latitud",
        "lon": "Longitud",
        "n_files": "Archivos",
        "files": "Archivos originales",
        "escenarios": "Escenarios"
    })
    columnas = [
        "Punto Sustax", "Latitud", "Longitud",
        "Archivos", "Escenarios", "Archivos originales"
    ]
    return tabla[[c for c in columnas if c in tabla.columns]]

st.title("Datos climáticos Sustax")

st.caption(

    "Inventario y verificación de archivos climáticos originales "

    "Sustax utilizados en el análisis doctoral."

)

st.info(

    "Esta sección permite identificar y preparar los archivos "

    "climáticos Sustax, revisar su cobertura temporal y analizar "

    "cambios proyectados en precipitación extrema bajo escenarios SSP."

)

# ============================================================

# 1. FUENTE DE DATOS

# ============================================================

st.subheader("1. Fuente de datos")

carpeta_automatica = detectar_carpeta_sustax()

opciones_fuente = [

    "Datos Sustax de la plataforma",

    "Carpeta externa",

]

fuente = st.radio(

    "Seleccione la fuente de los archivos:",

    options=opciones_fuente,

    horizontal=True,

)

# ------------------------------------------------------------

# DATOS DE LA PLATAFORMA

# ------------------------------------------------------------

if fuente == "Datos Sustax de la plataforma":

    if carpeta_automatica is None:

        st.warning(

            "No se detectó automáticamente una carpeta con archivos "

            "CSV Sustax dentro del proyecto."

        )

        st.caption(

            "Puede utilizar la opción 'Carpeta externa' para indicar "

            "manualmente dónde se encuentran los archivos."

        )

        carpeta = None

    else:

        carpeta = carpeta_automatica

        st.success(

            f"Carpeta Sustax detectada: {carpeta}"

        )

# ------------------------------------------------------------

# CARPETA EXTERNA

# ------------------------------------------------------------

else:

    ruta_externa = st.text_input(

        "Ruta de la carpeta con archivos originales Sustax:",

        value="",

        placeholder=r"C:\ruta\a\archivos\Sustax",

    )

    if ruta_externa.strip():

        carpeta = Path(ruta_externa.strip())

        if carpeta.exists() and carpeta.is_dir():

            st.success(

                f"Carpeta encontrada: {carpeta}"

            )

        else:

            st.error(

                "La ruta indicada no corresponde a una carpeta válida."

            )

            carpeta = None

    else:

        carpeta = None

# ============================================================

# DETENER SI NO HAY CARPETA

# ============================================================

if carpeta is None:

    st.stop()

# ============================================================

# 2. INVENTARIO

# ============================================================

st.subheader("2. Datos Sustax disponibles")

with st.spinner(

    "Leyendo archivos Sustax y construyendo inventario..."

):

    try:

        inventory_df, bad_df = build_sustax_inventory(

            str(carpeta)

        )

        group_summary_df = build_group_summary(

            inventory_df

        )

    except Exception as exc:

        st.error(

            "No fue posible construir el inventario Sustax."

        )

        st.exception(exc)

        st.stop()

# ============================================================

# SIN ARCHIVOS COMPATIBLES

# ============================================================

if inventory_df.empty:

    st.warning(

        "No se encontraron archivos originales Sustax compatibles "

        "en la carpeta seleccionada."

    )

    if bad_df is not None and not bad_df.empty:

        with st.expander(

            "Ver diagnóstico de archivos con error"

        ):

            st.dataframe(

                bad_df,

                use_container_width=True,

                hide_index=True,

            )

    st.stop()

# ============================================================

# 3. PERIODO Y ESCENARIOS

# ============================================================

escenarios = extraer_escenarios(

    inventory_df

)

with st.spinner(

    "Verificando cobertura temporal..."

):

    fecha_inicio, fecha_fin, errores_periodo = (

        obtener_periodo_global(

            inventory_df

        )

    )

# ============================================================

# 4. RESUMEN

# ============================================================

n_archivos = len(inventory_df)

n_puntos = len(group_summary_df)

n_coordenadas = contar_puntos_con_coordenadas(

    group_summary_df

)

n_escenarios = len(escenarios)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Archivos compatibles",

        n_archivos,

    )

with col2:

    st.metric(

        "Puntos climáticos",

        n_puntos,

    )

with col3:

    st.metric(

        "Con coordenadas",

        f"{n_coordenadas}/{n_puntos}",

    )

with col4:

    st.metric(

        "Escenarios detectados",

        n_escenarios,

    )

# ------------------------------------------------------------

# COBERTURA TEMPORAL

# ------------------------------------------------------------

if pd.notna(fecha_inicio) and pd.notna(fecha_fin):

    st.write(

        "**Cobertura temporal detectada:** "

        f"{fecha_inicio.strftime('%d/%m/%Y')} – "

        f"{fecha_fin.strftime('%d/%m/%Y')}"

    )

else:

    st.warning(

        "No fue posible determinar la cobertura temporal global."

    )

# ------------------------------------------------------------

# ESCENARIOS

# ------------------------------------------------------------

if escenarios:

    st.write(

        "**Escenarios disponibles:** "

        + " · ".join(escenarios)

    )

else:

    st.warning(

        "No se pudieron identificar escenarios climáticos."

    )

# ============================================================

# 5. TABLA DE PUNTOS

# ============================================================

st.subheader("3. Puntos climáticos disponibles")

tabla_puntos = preparar_tabla_puntos(

    inventory_df,

    group_summary_df,

)

st.dataframe(

    tabla_puntos,

    use_container_width=True,

    hide_index=True,

)

# ============================================================

# 6. COMPATIBILIDAD

# ============================================================

st.subheader("4. Compatibilidad de los datos")

if n_archivos > 0:

    st.success(

        f"{n_archivos} archivo(s) Sustax fueron leídos correctamente."

    )

if n_coordenadas == n_puntos:

    st.success(

        "Todos los puntos climáticos cuentan con coordenadas."

    )

else:

    st.warning(

        f"{n_puntos - n_coordenadas} punto(s) no cuentan con "

        "coordenadas completas."

    )

if "ERA5" in escenarios:

    st.success(

        "Se detectaron datos ERA5 como referencia climática histórica."

    )

else:

    st.warning(

        "No se detectaron datos ERA5."

    )

escenarios_ssp = [

    escenario

    for escenario in escenarios

    if escenario.startswith("SSP")

]

if escenarios_ssp:

    st.success(

        f"Se detectaron {len(escenarios_ssp)} escenario(s) SSP."

    )

else:

    st.warning(

        "No se detectaron escenarios SSP."

    )

# ============================================================

# 5. PREPARACIÓN PARA PROYECCIONES

# ============================================================

st.subheader("5. Preparación para proyecciones climáticas")

st.write(

    "Los archivos originales Sustax se organizan por punto climático y escenario "

    "antes de ejecutar las proyecciones."

)

st.caption(

    "Este proceso no modifica los archivos originales. Los productos derivados "

    "se almacenan en results/streamlit/sustax."

)

archivos_by_scenario = []

if SUSTAX_BY_SCENARIO_DIR.exists():

    archivos_by_scenario = [

        fp for fp in SUSTAX_BY_SCENARIO_DIR.glob("*.csv")

        if fp.is_file()

        and fp.name != "SUSTAX_TOTAL_BYSCEN_summary.csv"

        and "__" in fp.name

    ]

if not escenarios_ssp:

    st.warning("No existen escenarios SSP disponibles para preparar.")

elif archivos_by_scenario:

    st.success(

        f"Datos preparados: {len(archivos_by_scenario)} archivo(s) disponibles."

    )

    preparar_datos = st.button(

        "Regenerar datos preparados",

        type="secondary",

        use_container_width=True,

    )

else:

    st.info(

        "Todavía no existen datos preparados para ejecutar proyecciones climáticas."

    )

    preparar_datos = st.button(

        "Preparar datos para proyecciones",

        type="primary",

        use_container_width=True,

    )

if escenarios_ssp and preparar_datos:
    import shutil
    import tempfile
    import uuid

    with st.spinner("Consolidando y validando series Sustax..."):
        try:
            SUSTAX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="sustax_preparacion_",
                dir=str(SUSTAX_RESULTS_DIR.resolve())
            ) as tmp:
                temporal = Path(tmp)
                long_tmp = temporal / "LONG"
                by_tmp = temporal / "BY_SCENARIO"

                merged_summary_df, _ = build_long_total_from_inventory(
                    inventory_df=inventory_df,
                    out_dir=str(long_tmp)
                )
                byscen_summary_df, _ = build_by_scenario_total(
                    long_total_dir=str(long_tmp),
                    out_dir=str(by_tmp)
                )

                n_puntos = len(merged_summary_df)
                n_series = int(
                    byscen_summary_df["n_scenarios_generated"].sum()
                )
                archivos_long = list(long_tmp.glob("*Total_LONG_mm.csv"))
                archivos_byscen = list(by_tmp.glob("*__mm.csv"))

                if n_puntos != len(group_summary_df):
                    raise ValueError(
                        f"Puntos esperados: {len(group_summary_df)}; "
                        f"generados: {n_puntos}"
                    )
                if len(archivos_long) != n_puntos:
                    raise ValueError("Cantidad incorrecta de archivos LONG")
                if len(archivos_byscen) != n_series:
                    raise ValueError("Cantidad incorrecta de series BY_SCENARIO")
                if n_series == 0:
                    raise ValueError("No se generaron escenarios climáticos")

                # Verificar la cantidad de escenarios por coordenada.
                conteos = byscen_summary_df.set_index(
                    "sustax_total"
                )["n_scenarios_generated"].to_dict()
                for _, punto in group_summary_df.iterrows():
                    punto_id = (
                        f"{float(punto['lat']):.4f}_"
                        f"{float(punto['lon']):.4f}Total"
                    )
                    esperado = int(punto["n_scenarios"])
                    obtenido = int(conteos.get(punto_id, 0))
                    if obtenido != esperado:
                        raise ValueError(
                            f"Escenarios inconsistentes en {punto_id}: "
                            f"{obtenido} de {esperado}"
                        )

                # Publicar las dos carpetas con posibilidad de restauración.
                identificador = uuid.uuid4().hex
                destinos = [
                    (long_tmp, SUSTAX_LONG_DIR),
                    (by_tmp, SUSTAX_BY_SCENARIO_DIR)
                ]
                respaldos = []
                publicados = []
                try:
                    for nueva, destino in destinos:
                        respaldo = SUSTAX_RESULTS_DIR / (
                            f".respaldo_{destino.name}_{identificador}"
                        )
                        if destino.exists():
                            destino.rename(respaldo)
                            respaldos.append((respaldo, destino))
                        nueva.rename(destino)
                        publicados.append(destino)
                except Exception:
                    for destino in reversed(publicados):
                        if destino.exists():
                            shutil.rmtree(destino)
                    for respaldo, destino in reversed(respaldos):
                        if respaldo.exists():
                            respaldo.rename(destino)
                    raise
                else:
                    for respaldo, _ in respaldos:
                        shutil.rmtree(respaldo)

            # Invalidar resultados calculados con otra colección.
            for clave in list(st.session_state.keys()):
                if clave.startswith("sustax_projection"):
                    del st.session_state[clave]

            st.success(
                f"Consolidación correcta: {n_puntos} puntos "
                f"y {n_series} series climáticas."
            )
            st.rerun()
        except Exception as exc:
            st.error(
                "La nueva preparación no pudo completarse. "
                "Revise el diagnóstico antes de continuar."
            )
            st.exception(exc)

if archivos_by_scenario:

    st.subheader("6. Proyecciones climáticas")

    escenarios_proyeccion = [e for e in escenarios_ssp if e != "ERA5"]

    try:

        coverage_df = detect_projection_coverage(

            byscen_total_folder=str(SUSTAX_BY_SCENARIO_DIR),

            ssp_list=escenarios_proyeccion,

        )

        coverage_summary = summarize_projection_coverage(coverage_df)

    except Exception as exc:

        st.error("No fue posible determinar la cobertura de los escenarios SSP preparados.")

        st.exception(exc)

        coverage_summary = pd.DataFrame()

    if coverage_summary.empty:

        st.warning("No fue posible determinar la cobertura de los escenarios SSP preparados.")

    else:

        st.write("**Cobertura común por escenario**")

        tabla_cobertura = coverage_summary[

            ["scenario", "n_points", "common_start", "common_end", "common_years", "all_continuous"]

        ].rename(columns={

            "scenario": "Escenario",

            "n_points": "Puntos",

            "common_start": "Inicio",

            "common_end": "Fin",

            "common_years": "Años",

            "all_continuous": "Serie continua",

        })

        st.dataframe(tabla_cobertura, use_container_width=True, hide_index=True)

        escenario_sel = st.selectbox(

            "Escenario SSP:", options=coverage_summary["scenario"].tolist()

        )

        cobertura_sel = coverage_summary[

            coverage_summary["scenario"] == escenario_sel

        ].iloc[0]

        common_start = int(cobertura_sel["common_start"])

        common_end = int(cobertura_sel["common_end"])

        st.caption(

            f"Cobertura común disponible para {escenario_sel}: {common_start}–{common_end}."

        )

        st.markdown("#### Periodo de referencia")

        hist_start_default = max(1981, common_start)

        hist_end_default = min(2010, common_end)

        if hist_end_default < hist_start_default:

            hist_end_default = hist_start_default

        col_h1, col_h2 = st.columns(2)

        with col_h1:

            hist_start = st.number_input(

                "Año inicial de referencia",

                min_value=common_start,

                max_value=common_end,

                value=hist_start_default,

                step=1,

            )

        with col_h2:

            hist_end = st.number_input(

                "Año final de referencia",

                min_value=int(hist_start),

                max_value=common_end,

                value=max(int(hist_start), hist_end_default),

                step=1,

            )

        st.markdown("#### Horizonte futuro")

        future_start_default = min(max(2026, int(hist_end) + 1), common_end)

        future_end_default = min(future_start_default + 29, common_end)

        col_f1, col_f2 = st.columns(2)

        with col_f1:

            future_start = st.number_input(

                "Año inicial futuro",

                min_value=common_start,

                max_value=common_end,

                value=future_start_default,

                step=1,

            )

        with col_f2:

            future_end = st.number_input(

                "Año final futuro",

                min_value=int(future_start),

                max_value=common_end,

                value=max(int(future_start), future_end_default),

                step=1,

            )

        hist_eval = evaluate_projection_period(

            coverage_summary=coverage_summary,

            scenario=escenario_sel,

            period=(int(hist_start), int(hist_end)),

        )

        future_eval = evaluate_projection_period(

            coverage_summary=coverage_summary,

            scenario=escenario_sel,

            period=(int(future_start), int(future_end)),

        )

        hist_years = int(hist_end) - int(hist_start) + 1

        future_years = int(future_end) - int(future_start) + 1

        col_hist, col_future = st.columns(2)

        with col_hist:

            st.metric("Referencia", f"{hist_start}–{hist_end}", f"{hist_years} años")

        with col_future:

            st.metric("Horizonte futuro", f"{future_start}–{future_end}", f"{future_years} años")

        periodo_valido = True

        if hist_eval["status"] != "COMPLETE":

            st.error("El periodo histórico seleccionado no cuenta con cobertura completa.")

            periodo_valido = False

        if future_eval["status"] != "COMPLETE":

            st.error("El horizonte futuro seleccionado no cuenta con cobertura completa.")

            periodo_valido = False

        if int(future_start) <= int(hist_end):

            st.error("El horizonte futuro debe comenzar después del periodo de referencia.")

            periodo_valido = False

        if hist_years < 30:

            st.warning(

                "El periodo de referencia contiene menos de 30 años. Para el análisis "

                "climático definitivo se requiere una ventana de 30 años."

            )

            periodo_valido = False

        if future_years < 30:

            st.warning(

                "El horizonte futuro contiene menos de 30 años. Los datos actuales pueden "

                "revisarse, pero no se habilita el análisis climático definitivo hasta "

                "disponer de una ventana completa de 30 años."

            )

            periodo_valido = False

        threshold_mm = st.number_input(

            "Umbral de precipitación intensa (mm/día)",

            min_value=1.0,

            max_value=500.0,

            value=50.0,

            step=1.0,

            help=(

                "Se utiliza para calcular R50mm: número medio anual de días con "

                "precipitación igual o superior al umbral."

            ),

        )

        if not periodo_valido:

            st.info(

                "La configuración puede revisarse, pero la ejecución permanece deshabilitada "

                "mientras los periodos no cumplan la cobertura completa y la duración mínima."

            )

        ejecutar = st.button(

            "Ejecutar proyección climática",

            type="primary",

            use_container_width=True,

            disabled=not periodo_valido,

        )

        if ejecutar:

            with st.spinner("Calculando indicadores de cambio climático..."):

                try:

                    output_dir = (

                        SUSTAX_PROJECTION_DIR

                        / escenario_sel

                        / f"{hist_start}_{hist_end}__{future_start}_{future_end}"

                    )

                    output_dir.mkdir(parents=True, exist_ok=True)

                    projection_df, scenario_signal_df, point_signal_df = build_climate_change_projection(

                        byscen_total_folder=str(SUSTAX_BY_SCENARIO_DIR),

                        out_folder=str(output_dir),

                        ssp_list=[escenario_sel],

                        historical_period=(int(hist_start), int(hist_end)),

                        future_period=(int(future_start), int(future_end)),

                        threshold_mm=float(threshold_mm),

                        require_complete_period=True,

                    )

                    st.session_state["sustax_projection"] = projection_df

                    st.session_state["sustax_projection_scenario"] = escenario_sel

                    st.session_state["sustax_projection_scenario_signal"] = scenario_signal_df

                    st.session_state["sustax_projection_point_signal"] = point_signal_df

                    st.session_state["sustax_projection_output"] = str(output_dir)
                    st.session_state["sustax_projection_config"] = {
                        "scenario": escenario_sel,
                        "historical_period": (int(hist_start), int(hist_end)),
                        "future_period": (int(future_start), int(future_end)),
                        "threshold_mm": float(threshold_mm),
                    }

                    st.success("Proyección climática calculada correctamente.")

                except Exception as exc:

                    st.error("No fue posible calcular la proyección climática.")

                    st.exception(exc)

        if "sustax_projection" in st.session_state:
            projection_df = st.session_state["sustax_projection"]
            saved_config = st.session_state.get("sustax_projection_config")
            active_config = {
                "scenario": escenario_sel,
                "historical_period": (int(hist_start), int(hist_end)),
                "future_period": (int(future_start), int(future_end)),
                "threshold_mm": float(threshold_mm),
            }
            st.markdown("#### 7. Resultados de precipitación extrema")

            if saved_config is None:
                st.warning(
                    "Hay resultados de una ejecución anterior en la sesión, "
                    "pero no se dispone de su configuración temporal. "
                    "Ejecute nuevamente el análisis para interpretarlos."
                )
            elif saved_config != active_config:
                st.info(
                    "Los resultados guardados en la sesión corresponden a otra "
                    "configuración. Restablezca sus parámetros o ejecute la nueva proyección."
                )
            elif projection_df is None or projection_df.empty:
                st.warning("La ejecución no produjo resultados para mostrar.")
            else:
                st.caption(
                    f"{escenario_sel} · Referencia {hist_start}–{hist_end} · "
                    f"Futuro {future_start}–{future_end} · "
                    f"Umbral {threshold_mm:g} mm/día"
                )
                indicadores = [
                    ("P95", "p95_daily_mm", "mm"),
                    ("P99", "p99_daily_mm", "mm"),
                    ("Rx1day (promedio de máximos diarios anuales)", "rx1day_mm", "mm"),
                    ("R50mm (promedio anual de días con umbral)", "r50mm_days_per_year", "días/año"),
                ]
                columnas = set(projection_df.columns)

                def primera_columna(candidatas):
                    return next((c for c in candidatas if c in columnas), None)

                def columnas_indicador(clave):
                    return {
                        "historico": primera_columna([
                            f"hist_{clave}", f"historical_{clave}",
                            f"baseline_{clave}", f"reference_{clave}",
                            f"{clave}_hist", f"{clave}_historical",
                        ]),
                        "futuro": primera_columna([
                            f"future_{clave}", f"fut_{clave}",
                            f"{clave}_future",
                        ]),
                        "absoluto": primera_columna([
                            f"delta_{clave}", f"abs_change_{clave}",
                            f"change_{clave}",
                        ]),
                        "porcentaje": primera_columna([
                            f"pct_change_{clave}", f"percent_change_{clave}",
                        ]),
                    }

                puntos = sorted(projection_df["sustax_total"].dropna().astype(str).unique())
                punto_sel = st.selectbox(
                    "Punto climático para consultar indicadores:",
                    options=puntos,
                    key="sustax_result_point",
                )
                registro = projection_df.loc[
                    projection_df["sustax_total"].astype(str) == punto_sel
                ].iloc[0]
                filas = []
                incompletos = []
                for etiqueta, clave, unidad in indicadores:
                    cols = columnas_indicador(clave)
                    if not cols["historico"] or not cols["futuro"]:
                        incompletos.append(etiqueta)
                        continue
                    historico = pd.to_numeric(registro[cols["historico"]], errors="coerce")
                    futuro = pd.to_numeric(registro[cols["futuro"]], errors="coerce")
                    absoluto = (
                        pd.to_numeric(registro[cols["absoluto"]], errors="coerce")
                        if cols["absoluto"] else futuro - historico
                    )
                    porcentaje = (
                        pd.to_numeric(registro[cols["porcentaje"]], errors="coerce")
                        if cols["porcentaje"] else (
                            100 * (futuro - historico) / historico
                            if pd.notna(historico) and historico != 0 else float("nan")
                        )
                    )
                    filas.append({
                        "Indicador": etiqueta,
                        "Unidad": unidad,
                        "Histórico": historico,
                        "Futuro": futuro,
                        "Cambio absoluto": absoluto,
                        "Cambio (%)": porcentaje,
                    })
                if filas:
                    st.dataframe(
                        pd.DataFrame(filas).style.format({
                            "Histórico": "{:.2f}",
                            "Futuro": "{:.2f}",
                            "Cambio absoluto": "{:+.2f}",
                            "Cambio (%)": "{:+.2f}",
                        }, na_rep="N/D"),
                        use_container_width=True,
                        hide_index=True,
                    )
                if incompletos:
                    st.warning(
                        "No se encontraron las columnas históricas y futuras de: "
                        + ", ".join(incompletos)
                        + ". Revise los nombres producidos por projection.py."
                    )
                st.caption(
                    "R50mm muestra días por año y su cambio relativo; porcentajes "
                    "elevados pueden partir de frecuencias históricas pequeñas. "
                    "Estos indicadores representan cambios en precipitación extrema, "
                    "no estimaciones directas del riesgo de inundación."
                )
                with st.expander("Ver resultados completos por punto"):
                    st.dataframe(projection_df, use_container_width=True, hide_index=True)
# ============================================================

# 7. DIAGNÓSTICO

# ============================================================

if bad_df is not None and not bad_df.empty:

    with st.expander(

        f"Archivos con error ({len(bad_df)})"

    ):

        st.dataframe(

            bad_df,

            use_container_width=True,

            hide_index=True,

        )

if errores_periodo:

    with st.expander(

        f"Advertencias de cobertura temporal "

        f"({len(errores_periodo)})"

    ):

        st.dataframe(

            pd.DataFrame(errores_periodo),

            use_container_width=True,

            hide_index=True,

        )

# ============================================================

# 8. INVENTARIO TÉCNICO

# ============================================================

with st.expander(

    "Ver inventario técnico"

):

    columnas_inventario = [

        c

        for c in [

            "sustax_file",

            "base_name",

            "lat",

            "lon",

            "n_rows",

            "n_cols",

            "scenarios_detected",

            "path",

        ]

        if c in inventory_df.columns

    ]

    st.dataframe(

        inventory_df[columnas_inventario],

        use_container_width=True,

        hide_index=True,

    )

# ============================================================

# NOTA METODOLÓGICA

# ============================================================

st.caption(

    "Los archivos originales Sustax se conservan sin modificaciones. "

    "La lectura mostrada en esta sección utiliza los módulos científicos "

    "del repositorio Doctorado-DCAG. Los productos derivados se guardan "

    "separadamente dentro de results/streamlit/sustax."

)