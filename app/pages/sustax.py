
from pathlib import Path
import os
import shutil
import tempfile
import uuid

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

from scripts.sustax.loader import load_sustax_file_safe

from scripts.sustax.projection import (
    detect_projection_coverage,
    summarize_projection_coverage,
    build_climate_change_projection,
)

from scripts.sustax.coverage_filter import (
    evaluar_cobertura_ssp,
)


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARPETAS_SUSTAX_CANDIDATAS = [
    PROJECT_ROOT / "SUSTAX",
    PROJECT_ROOT / "Sustax",
    PROJECT_ROOT / "sustax",
    PROJECT_ROOT / "data" / "SUSTAX",
    PROJECT_ROOT / "data" / "Sustax",
    PROJECT_ROOT / "data" / "sustax",
]

SUSTAX_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "streamlit" / "sustax"
)

SUSTAX_LONG_DIR = SUSTAX_RESULTS_DIR / "LONG"

SUSTAX_BY_SCENARIO_DIR = (
    SUSTAX_RESULTS_DIR / "BY_SCENARIO"
)

SUSTAX_PROJECTION_DIR = (
    SUSTAX_RESULTS_DIR / "PROJECTIONS"
)

ORDEN_ESCENARIOS = [
    "ERA5",
    "SSP119",
    "SSP126",
    "SSP245",
    "SSP370",
    "SSP434",
    "SSP460",
    "SSP585",
]


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def detectar_carpeta_sustax():
    """Localiza una carpeta con archivos Sustax originales."""

    for carpeta in CARPETAS_SUSTAX_CANDIDATAS:
        if carpeta.is_dir() and any(carpeta.glob("*.csv")):
            return carpeta

    return None


def extraer_escenarios(inventory_df):
    """Recupera los escenarios presentes en el inventario."""

    if (
        inventory_df.empty
        or "scenarios_detected" not in inventory_df.columns
    ):
        return []

    encontrados = set()

    for valor in inventory_df["scenarios_detected"].dropna():
        for escenario in str(valor).split(","):
            escenario = escenario.strip()

            if escenario:
                encontrados.add(escenario)

    ordenados = [
        e for e in ORDEN_ESCENARIOS
        if e in encontrados
    ]

    otros = sorted(
        encontrados.difference(ORDEN_ESCENARIOS)
    )

    return ordenados + otros


def obtener_periodo_archivo(archivo):
    """Obtiene el periodo temporal con el lector Sustax."""

    df, _ = load_sustax_file_safe(
        str(archivo),
        return_pandas_df=True,
        return_metadata=False,
    )

    if df is None or df.empty:
        return pd.NaT, pd.NaT

    fechas = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    fechas = fechas[~pd.isna(fechas)]

    if len(fechas) == 0:
        return pd.NaT, pd.NaT

    return fechas.min(), fechas.max()


def obtener_periodo_global(inventory_df):
    """Obtiene fechas extremas y registra errores de lectura."""

    inicios = []
    finales = []
    errores = []

    for _, fila in inventory_df.iterrows():
        archivo = Path(str(fila["path"]))

        try:
            inicio, fin = obtener_periodo_archivo(archivo)

            if pd.notna(inicio):
                inicios.append(inicio)

            if pd.notna(fin):
                finales.append(fin)

        except Exception as exc:
            errores.append({
                "archivo": archivo.name,
                "error": str(exc),
            })

    return (
        min(inicios) if inicios else pd.NaT,
        max(finales) if finales else pd.NaT,
        errores,
    )


def contar_puntos_con_coordenadas(group_summary_df):

    if group_summary_df.empty:
        return 0

    if not {"lat", "lon"}.issubset(
        group_summary_df.columns
    ):
        return 0

    coords = group_summary_df[["lat", "lon"]].copy()

    for columna in ("lat", "lon"):
        coords[columna] = pd.to_numeric(
            coords[columna],
            errors="coerce",
        )

    return int(
        coords.dropna(subset=["lat", "lon"]).shape[0]
    )


def preparar_tabla_puntos(inventory_df, group_summary_df):

    if group_summary_df.empty:
        return pd.DataFrame()

    tabla = group_summary_df.copy()

    if "punto_id" not in tabla.columns:
        tabla["punto_id"] = tabla.apply(
            lambda r: (
                f"{float(r['lat']):.4f}_"
                f"{float(r['lon']):.4f}"
            ),
            axis=1,
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
        "escenarios": "Escenarios",
    })

    columnas = [
        "Punto Sustax",
        "Latitud",
        "Longitud",
        "Archivos",
        "Escenarios",
        "Archivos originales",
    ]

    return tabla[
        [c for c in columnas if c in tabla.columns]
    ]


@st.cache_data(show_spinner=False)
def obtener_limites_temporales_ssp(carpeta, escenario):
    """
    Obtiene el rango global de años disponibles.

    No restringe el horizonte a la intersección
    temporal de todos los puntos.
    """

    archivos = sorted(
        Path(carpeta).glob(
            f"*__{escenario}__mm.csv"
        )
    )

    inicios = []
    finales = []

    for archivo in archivos:

        fechas = pd.read_csv(
            archivo,
            usecols=["date"],
        )

        fechas = pd.to_datetime(
            fechas["date"],
            errors="coerce",
        ).dropna()

        if not fechas.empty:
            inicios.append(
                int(fechas.dt.year.min())
            )

            finales.append(
                int(fechas.dt.year.max())
            )

    if not inicios:
        raise ValueError(
            f"No se encontraron fechas para {escenario}"
        )

    return min(inicios), max(finales)


def archivos_preparados():

    if not SUSTAX_BY_SCENARIO_DIR.is_dir():
        return []

    return sorted(
        fp
        for fp in SUSTAX_BY_SCENARIO_DIR.glob("*.csv")
        if fp.is_file()
        and fp.name != "SUSTAX_TOTAL_BYSCEN_summary.csv"
        and "__" in fp.name
    )


# ============================================================
# 3. CONSOLIDACIÓN
# ============================================================

def publicar_preparacion(
    inventory_df,
    group_summary_df,
):
    """
    Consolida series en carpetas temporales y publica
    los productos con posibilidad de restauración.
    """

    SUSTAX_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="sustax_preparacion_",
        dir=str(SUSTAX_RESULTS_DIR.resolve()),
    ) as tmp:

        temporal = Path(tmp)

        long_tmp = temporal / "LONG"
        by_tmp = temporal / "BY_SCENARIO"

        merged_summary_df, _ = (
            build_long_total_from_inventory(
                inventory_df=inventory_df,
                out_dir=str(long_tmp),
            )
        )

        byscen_summary_df, _ = (
            build_by_scenario_total(
                long_total_dir=str(long_tmp),
                out_dir=str(by_tmp),
            )
        )

        n_puntos = len(merged_summary_df)

        n_series = int(
            byscen_summary_df[
                "n_scenarios_generated"
            ].sum()
        )

        archivos_long = list(
            long_tmp.glob("*Total_LONG_mm.csv")
        )

        archivos_byscen = list(
            by_tmp.glob("*__mm.csv")
        )

        if n_puntos != len(group_summary_df):
            raise ValueError(
                "La cantidad de puntos consolidados "
                "no coincide con el inventario."
            )

        if len(archivos_long) != n_puntos:
            raise ValueError(
                "Cantidad incorrecta de archivos LONG."
            )

        if len(archivos_byscen) != n_series:
            raise ValueError(
                "Cantidad incorrecta de series BY_SCENARIO."
            )

        if n_series == 0:
            raise ValueError(
                "No se generaron escenarios climáticos."
            )

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

        identificador = uuid.uuid4().hex

        destinos = [
            (long_tmp, SUSTAX_LONG_DIR),
            (by_tmp, SUSTAX_BY_SCENARIO_DIR),
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
                    respaldos.append(
                        (respaldo, destino)
                    )

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

    return n_puntos, n_series


# ============================================================
# 4. TABLA DE INDICADORES
# ============================================================

def tabla_indicadores(registro):
    """
    Construye la tabla visual usando directamente
    los valores producidos por projection.py.
    """

    indicadores = [
        ("P95", "p95_daily_mm", "mm"),
        ("P99", "p99_daily_mm", "mm"),
        (
            "Rx1day (promedio de máximos diarios anuales)",
            "rx1day_mm",
            "mm",
        ),
        (
            "R50mm (promedio anual de días con umbral)",
            "r50mm_days_per_year",
            "días/año",
        ),
    ]

    filas = []

    for etiqueta, clave, unidad in indicadores:

        nombres = {
            "hist": f"hist_{clave}",
            "future": f"future_{clave}",
            "delta": f"delta_{clave}",
            "pct": f"pct_change_{clave}",
        }

        if not all(
            nombre in registro.index
            for nombre in nombres.values()
        ):
            continue

        filas.append({
            "Indicador": etiqueta,
            "Unidad": unidad,
            "Histórico": pd.to_numeric(
                registro[nombres["hist"]],
                errors="coerce",
            ),
            "Futuro": pd.to_numeric(
                registro[nombres["future"]],
                errors="coerce",
            ),
            "Cambio absoluto": pd.to_numeric(
                registro[nombres["delta"]],
                errors="coerce",
            ),
            "Cambio (%)": pd.to_numeric(
                registro[nombres["pct"]],
                errors="coerce",
            ),
        })

    return pd.DataFrame(filas)


# ============================================================
# 5. EXPORTACIÓN VALIDADA
# ============================================================

def exportar_proyeccion_validada(
    projection_df: pd.DataFrame,
    diagnostico_cobertura: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Exporta únicamente resultados con cobertura diaria
    completa y guarda el diagnóstico de exclusiones.

    Los CSV técnicos de projection.py permanecen intactos.
    """

    output_dir = Path(output_dir)

    carpeta_validada = (
        output_dir / "VALIDATED"
    )

    if projection_df.empty:
        raise ValueError(
            "No hay resultados validados para exportar."
        )

    if diagnostico_cobertura.empty:
        raise ValueError(
            "No existe diagnóstico de cobertura."
        )

    columnas_requeridas = {
        "sustax_total",
        "scenario",
        "change_computed",
    }

    if not columnas_requeridas.issubset(
        projection_df.columns
    ):
        raise ValueError(
            "Faltan columnas obligatorias de proyección."
        )

    columnas_cobertura = {
        "punto",
        "escenario",
        "elegible",
    }

    if not columnas_cobertura.issubset(
        diagnostico_cobertura.columns
    ):
        raise ValueError(
            "Faltan columnas obligatorias de cobertura."
        )

    resultados = projection_df.copy()
    diagnostico = diagnostico_cobertura.copy()

    if not resultados["change_computed"].eq(True).all():
        raise ValueError(
            "Existen cambios no calculados en los resultados."
        )

    claves_resultados = set(
        zip(
            resultados["sustax_total"].astype(str),
            resultados["scenario"].astype(str),
        )
    )

    elegibles = diagnostico.loc[
        diagnostico["elegible"].eq(True)
    ].copy()

    claves_elegibles = set(
        zip(
            elegibles["punto"].astype(str),
            elegibles["escenario"].astype(str),
        )
    )

    if claves_resultados != claves_elegibles:
        raise ValueError(
            "Los resultados no coinciden con las series "
            "elegibles del diagnóstico de cobertura."
        )

    if resultados.duplicated(
        ["sustax_total", "scenario"]
    ).any():
        raise ValueError(
            "Existen puntos duplicados en la proyección."
        )

    excluidos = diagnostico.loc[
        diagnostico["elegible"].eq(False)
    ].copy()

    destinos = {
        "resultados": (
            carpeta_validada
            / "climate_change_projection_validated.csv"
        ),
        "cobertura": (
            carpeta_validada
            / "daily_coverage_diagnostic.csv"
        ),
        "excluidos": (
            carpeta_validada
            / "excluded_points.csv"
        ),
    }

    tablas = {
        "resultados": resultados,
        "cobertura": diagnostico,
        "excluidos": excluidos,
    }

    # Preparación de los tres CSV en un directorio
    # temporal, antes de publicar la carpeta completa.

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="sustax_export_",
        dir=str(output_dir.resolve()),
    ) as tmp:

        temporal = Path(tmp)

        nueva = temporal / "VALIDATED"
        nueva.mkdir()

        for nombre, destino in destinos.items():
            tablas[nombre].to_csv(
                nueva / destino.name,
                index=False,
                encoding="utf-8-sig",
            )

        respaldo = output_dir / (
            f".VALIDATED_backup_{uuid.uuid4().hex}"
        )

        habia_anterior = carpeta_validada.exists()

        if habia_anterior:
            carpeta_validada.rename(respaldo)

        try:
            nueva.rename(carpeta_validada)

        except Exception:
            if habia_anterior and respaldo.exists():
                respaldo.rename(carpeta_validada)
            raise

        else:
            if respaldo.exists():
                shutil.rmtree(respaldo)

    return {
        "carpeta": str(carpeta_validada),
        "resultados": str(destinos["resultados"]),
        "cobertura": str(destinos["cobertura"]),
        "excluidos": str(destinos["excluidos"]),
        "n_elegibles": len(resultados),
        "n_excluidos": len(excluidos),
    }


# ============================================================
# 6. ENCABEZADO
# ============================================================

st.title("Datos climáticos Sustax")

st.caption(
    "Inventario, consolidación y análisis de escenarios "
    "climáticos de precipitación Sustax."
)

st.info(
    "Esta sección permite verificar los archivos originales, "
    "revisar su cobertura temporal y analizar cambios "
    "proyectados en precipitación extrema bajo escenarios SSP."
)


# ============================================================
# 7. FUENTE DE DATOS
# ============================================================

st.subheader("1. Fuente de datos")

carpeta_automatica = detectar_carpeta_sustax()

fuente = st.radio(
    "Seleccione la fuente de los archivos:",
    options=[
        "Datos Sustax de la plataforma",
        "Carpeta externa",
    ],
    horizontal=True,
)

if fuente == "Datos Sustax de la plataforma":

    carpeta = carpeta_automatica

    if carpeta is None:
        st.warning(
            "No se detectó automáticamente una carpeta "
            "con archivos CSV Sustax."
        )

    else:
        st.success(
            f"Carpeta detectada: {carpeta}"
        )

else:

    ruta_externa = st.text_input(
        "Ruta de la carpeta con archivos originales:",
        value="",
        placeholder=r"C:\ruta\a\archivos\Sustax",
    )

    carpeta = (
        Path(ruta_externa.strip())
        if ruta_externa.strip()
        else None
    )

    if carpeta is not None:

        if carpeta.is_dir():
            st.success(
                f"Carpeta encontrada: {carpeta}"
            )

        else:
            st.error(
                "La carpeta indicada no existe."
            )
            carpeta = None

if carpeta is None:
    st.stop()


# ============================================================
# 8. INVENTARIO
# ============================================================

st.subheader("2. Datos Sustax disponibles")

with st.spinner(
    "Construyendo inventario Sustax..."
):
    try:

        inventory_df, bad_df = (
            build_sustax_inventory(
                str(carpeta)
            )
        )

        group_summary_df = (
            build_group_summary(
                inventory_df
            )
        )

    except Exception as exc:

        st.error(
            "No fue posible construir el inventario."
        )

        st.exception(exc)
        st.stop()

if inventory_df.empty:

    st.warning(
        "No se encontraron archivos Sustax compatibles."
    )

    if bad_df is not None and not bad_df.empty:
        st.dataframe(
            bad_df,
            width="stretch",
            hide_index=True,
        )

    st.stop()

escenarios = extraer_escenarios(
    inventory_df
)

escenarios_ssp = [
    e for e in escenarios
    if e.startswith("SSP")
]

with st.spinner(
    "Verificando cobertura temporal..."
):
    fecha_inicio, fecha_fin, errores_periodo = (
        obtener_periodo_global(
            inventory_df
        )
    )

n_archivos = len(inventory_df)
n_puntos = len(group_summary_df)

n_coordenadas = contar_puntos_con_coordenadas(
    group_summary_df
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Archivos compatibles",
    n_archivos,
)

col2.metric(
    "Puntos climáticos",
    n_puntos,
)

col3.metric(
    "Con coordenadas",
    f"{n_coordenadas}/{n_puntos}",
)

col4.metric(
    "Escenarios detectados",
    len(escenarios),
)

if pd.notna(fecha_inicio) and pd.notna(fecha_fin):

    st.write(
        "**Cobertura temporal detectada:** "
        f"{fecha_inicio:%d/%m/%Y} – "
        f"{fecha_fin:%d/%m/%Y}"
    )

st.write(
    "**Escenarios disponibles:** "
    + " · ".join(escenarios)
)


# ============================================================
# 9. PUNTOS CLIMÁTICOS
# ============================================================

st.subheader(
    "3. Puntos climáticos disponibles"
)

tabla_puntos = preparar_tabla_puntos(
    inventory_df,
    group_summary_df,
)

st.dataframe(
    tabla_puntos,
    width="stretch",
    hide_index=True,
)


# ============================================================
# 10. COMPATIBILIDAD
# ============================================================

st.subheader(
    "4. Compatibilidad de los datos"
)

st.success(
    f"{n_archivos} archivos Sustax compatibles."
)

if n_coordenadas == n_puntos:

    st.success(
        "Todos los puntos cuentan con coordenadas."
    )

else:

    st.warning(
        f"{n_puntos - n_coordenadas} puntos sin "
        "coordenadas completas."
    )

if "ERA5" in escenarios:

    st.success(
        "Se detectaron datos ERA5 como referencia "
        "climática histórica."
    )

else:
    st.warning(
        "No se detectaron datos ERA5."
    )

if escenarios_ssp:

    st.success(
        f"Se detectaron {len(escenarios_ssp)} "
        "escenarios SSP."
    )

else:
    st.warning(
        "No se detectaron escenarios SSP."
    )


# ============================================================
# 11. PREPARACIÓN DE DATOS
# ============================================================

st.subheader(
    "5. Preparación para proyecciones climáticas"
)

st.write(
    "Los archivos originales se organizan por punto "
    "climático y escenario antes del análisis."
)

st.caption(
    "Los archivos originales no se modifican. "
    "Los productos derivados se almacenan en "
    "results/streamlit/sustax."
)

archivos_by_scenario = archivos_preparados()

preparar_datos = False

if not escenarios_ssp:

    st.warning(
        "No existen escenarios SSP disponibles."
    )

elif archivos_by_scenario:

    st.success(
        f"Datos preparados: "
        f"{len(archivos_by_scenario)} archivos."
    )

    preparar_datos = st.button(
        "Regenerar datos preparados",
        type="secondary",
        width="stretch",
    )

else:

    st.info(
        "Todavía no existen datos preparados."
    )

    preparar_datos = st.button(
        "Preparar datos para proyecciones",
        type="primary",
        width="stretch",
    )

if escenarios_ssp and preparar_datos:

    with st.spinner(
        "Consolidando y validando series Sustax..."
    ):
        try:

            n_puntos_generados, n_series = (
                publicar_preparacion(
                    inventory_df,
                    group_summary_df,
                )
            )

            obtener_limites_temporales_ssp.clear()

            for clave in list(st.session_state.keys()):
                if clave.startswith(
                    "sustax_projection"
                ):
                    del st.session_state[clave]

            st.success(
                f"Consolidación correcta: "
                f"{n_puntos_generados} puntos "
                f"y {n_series} series climáticas."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                "No fue posible completar la preparación."
            )

            st.exception(exc)


# ============================================================
# 12. PROYECCIONES CLIMÁTICAS
# ============================================================

archivos_by_scenario = archivos_preparados()

if archivos_by_scenario and escenarios_ssp:

    st.subheader(
        "6. Proyecciones climáticas"
    )

    try:

        coverage_df = detect_projection_coverage(
            byscen_total_folder=str(
                SUSTAX_BY_SCENARIO_DIR
            ),
            ssp_list=escenarios_ssp,
        )

        coverage_summary = (
            summarize_projection_coverage(
                coverage_df
            )
        )

    except Exception as exc:

        st.error(
            "No fue posible determinar la cobertura SSP."
        )

        st.exception(exc)
        coverage_summary = pd.DataFrame()

    if coverage_summary.empty:

        st.warning(
            "No se encontró cobertura de escenarios SSP."
        )

    else:

        st.write(
            "**Cobertura común por escenario**"
        )

        columnas_cobertura = [
            "scenario",
            "n_points",
            "common_start",
            "common_end",
            "common_years",
            "all_continuous",
        ]

        tabla_cobertura = coverage_summary[
            columnas_cobertura
        ].rename(columns={
            "scenario": "Escenario",
            "n_points": "Puntos disponibles",
            "common_start": "Inicio común",
            "common_end": "Fin común",
            "common_years": "Años comunes",
            "all_continuous": "Continuidad reportada",
        })

        st.dataframe(
            tabla_cobertura,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "La cobertura común considera todos los "
            "puntos del escenario. Los periodos de "
            "proyección se validan individualmente "
            "mediante cobertura diaria."
        )

        escenario_sel = st.selectbox(
            "Escenario SSP:",
            options=coverage_summary[
                "scenario"
            ].tolist(),
        )

        try:

            limite_inicio, limite_fin = (
                obtener_limites_temporales_ssp(
                    str(SUSTAX_BY_SCENARIO_DIR),
                    escenario_sel,
                )
            )

        except Exception as exc:

            st.error(
                "No fue posible obtener los límites "
                "temporales del escenario."
            )

            st.exception(exc)
            st.stop()

        st.caption(
            f"Rango disponible en {escenario_sel}: "
            f"{limite_inicio}–{limite_fin}. "
            "No todos los puntos necesariamente "
            "cubren el rango completo."
        )

        # ====================================================
        # 12.1 PERIODO HISTÓRICO
        # ====================================================

        st.markdown(
            "#### Periodo de referencia"
        )

        hist_start_default = max(
            1981,
            limite_inicio,
        )

        hist_end_default = min(
            2010,
            limite_fin,
        )

        col_h1, col_h2 = st.columns(2)

        with col_h1:

            hist_start = st.number_input(
                "Año inicial de referencia",
                min_value=limite_inicio,
                max_value=limite_fin,
                value=hist_start_default,
                step=1,
                key=f"sustax_hist_start_{escenario_sel}",
            )

        with col_h2:

            hist_end = st.number_input(
                "Año final de referencia",
                min_value=int(hist_start),
                max_value=limite_fin,
                value=max(
                    int(hist_start),
                    hist_end_default,
                ),
                step=1,
                key=f"sustax_hist_end_{escenario_sel}",
            )

        # ====================================================
        # 12.2 PERIODO FUTURO
        # ====================================================

        st.markdown(
            "#### Horizonte futuro"
        )

        future_start_default = min(
            max(2026, int(hist_end) + 1),
            limite_fin,
        )

        future_end_default = min(
            future_start_default + 29,
            limite_fin,
        )

        col_f1, col_f2 = st.columns(2)

        with col_f1:

            future_start = st.number_input(
                "Año inicial futuro",
                min_value=limite_inicio,
                max_value=limite_fin,
                value=future_start_default,
                step=1,
                key=f"sustax_future_start_{escenario_sel}",
            )

        with col_f2:

            future_end = st.number_input(
                "Año final futuro",
                min_value=int(future_start),
                max_value=limite_fin,
                value=max(
                    int(future_start),
                    future_end_default,
                ),
                step=1,
                key=f"sustax_future_end_{escenario_sel}",
            )

        hist_years = (
            int(hist_end) - int(hist_start) + 1
        )

        future_years = (
            int(future_end) - int(future_start) + 1
        )

        col_hist, col_future = st.columns(2)

        col_hist.metric(
            "Referencia",
            f"{hist_start}–{hist_end}",
            f"{hist_years} años",
        )

        col_future.metric(
            "Horizonte futuro",
            f"{future_start}–{future_end}",
            f"{future_years} años",
        )

        # ====================================================
        # 12.3 VALIDACIÓN TEMPORAL
        # ====================================================

        periodo_valido = True

        if int(future_start) <= int(hist_end):

            st.error(
                "El horizonte futuro debe comenzar "
                "después del periodo de referencia."
            )

            periodo_valido = False

        if hist_years < 30:

            st.warning(
                "El periodo histórico debe contener "
                "al menos 30 años."
            )

            periodo_valido = False

        if future_years < 30:

            st.warning(
                "El periodo futuro debe contener "
                "al menos 30 años."
            )

            periodo_valido = False

        # ====================================================
        # 12.4 COBERTURA DIARIA
        # ====================================================

        diagnostico_cobertura = pd.DataFrame()
        puntos_elegibles = set()

        if periodo_valido:

            try:

                with st.spinner(
                    "Validando cobertura diaria..."
                ):

                    diagnostico_cobertura = (
                        evaluar_cobertura_ssp(
                            carpeta=SUSTAX_BY_SCENARIO_DIR,
                            periodo_historico=(
                                int(hist_start),
                                int(hist_end),
                            ),
                            periodo_futuro=(
                                int(future_start),
                                int(future_end),
                            ),
                            escenarios=[escenario_sel],
                        )
                    )

                if diagnostico_cobertura.empty:

                    st.error(
                        "No se encontraron series para "
                        "el escenario seleccionado."
                    )

                    periodo_valido = False

                else:

                    puntos_elegibles = set(
                        diagnostico_cobertura.loc[
                            diagnostico_cobertura[
                                "elegible"
                            ].eq(True),
                            "punto",
                        ].astype(str)
                    )

                    total_puntos = len(
                        diagnostico_cobertura
                    )

                    n_elegibles = len(
                        puntos_elegibles
                    )

                    n_excluidos = (
                        total_puntos - n_elegibles
                    )

                    col_a, col_b, col_c = (
                        st.columns(3)
                    )

                    col_a.metric(
                        "Puntos disponibles",
                        total_puntos,
                    )

                    col_b.metric(
                        "Puntos elegibles",
                        n_elegibles,
                    )

                    col_c.metric(
                        "Puntos excluidos",
                        n_excluidos,
                    )

                    if n_elegibles == 0:

                        st.error(
                            "Ningún punto cumple ambos "
                            "periodos con cobertura diaria."
                        )

                        periodo_valido = False

                    else:

                        st.success(
                            f"{n_elegibles} puntos cumplen "
                            "la cobertura diaria requerida."
                        )

                    with st.expander(
                        "Ver diagnóstico de cobertura "
                        "por punto"
                    ):

                        st.dataframe(
                            diagnostico_cobertura,
                            width="stretch",
                            hide_index=True,
                        )

            except Exception as exc:

                periodo_valido = False

                st.error(
                    "No fue posible validar la "
                    "cobertura diaria."
                )

                st.exception(exc)

        # ====================================================
        # 12.5 UMBRAL
        # ====================================================

        threshold_mm = st.number_input(
            "Umbral de precipitación intensa (mm/día)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help=(
                "Umbral utilizado para calcular "
                "el número medio anual de días "
                "con precipitación intensa."
            ),
        )

        if not periodo_valido:

            st.info(
                "La ejecución permanecerá deshabilitada "
                "hasta contar con periodos válidos y "
                "al menos un punto elegible."
            )

        # ====================================================
        # 12.6 EJECUCIÓN
        # ====================================================

        ejecutar = st.button(
            "Ejecutar proyección climática",
            type="primary",
            width="stretch",
            disabled=not periodo_valido,
        )

        configuracion_actual = {
            "scenario": escenario_sel,
            "historical_period": (
                int(hist_start),
                int(hist_end),
            ),
            "future_period": (
                int(future_start),
                int(future_end),
            ),
            "threshold_mm": float(threshold_mm),
        }

        if ejecutar:

            with st.spinner(
                "Calculando indicadores climáticos..."
            ):

                try:

                    output_dir = (
                        SUSTAX_PROJECTION_DIR
                        / escenario_sel
                        / (
                            f"{hist_start}_{hist_end}"
                            f"__{future_start}_{future_end}"
                        )
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    (
                        projection_df,
                        scenario_signal_df,
                        point_signal_df,
                    ) = build_climate_change_projection(
                        byscen_total_folder=str(
                            SUSTAX_BY_SCENARIO_DIR
                        ),
                        out_folder=str(output_dir),
                        ssp_list=[escenario_sel],
                        historical_period=(
                            int(hist_start),
                            int(hist_end),
                        ),
                        future_period=(
                            int(future_start),
                            int(future_end),
                        ),
                        threshold_mm=float(threshold_mm),
                        require_complete_period=True,
                    )

                    # ----------------------------------------
                    # FILTRO ESTRICTO
                    # ----------------------------------------

                    if "change_computed" not in (
                        projection_df.columns
                    ):
                        raise ValueError(
                            "El motor no devolvió "
                            "la columna change_computed."
                        )

                    projection_df = projection_df.loc[
                        projection_df[
                            "change_computed"
                        ].eq(True)
                        & projection_df[
                            "sustax_total"
                        ].astype(str).isin(
                            puntos_elegibles
                        )
                    ].copy()

                    if projection_df.empty:
                        raise ValueError(
                            "No quedaron resultados "
                            "después del filtro de "
                            "cobertura diaria."
                        )

                    puntos_calculados = set(
                        projection_df[
                            "sustax_total"
                        ].astype(str)
                    )

                    if puntos_calculados != puntos_elegibles:
                        raise ValueError(
                            "Los puntos calculados no "
                            "coinciden con los puntos "
                            "validados por cobertura diaria."
                        )

                    # ----------------------------------------
                    # EXPORTACIÓN VALIDADA
                    # ----------------------------------------

                    reporte_exportacion = (
                        exportar_proyeccion_validada(
                            projection_df=projection_df,
                            diagnostico_cobertura=(
                                diagnostico_cobertura
                            ),
                            output_dir=output_dir,
                        )
                    )

                    # ----------------------------------------
                    # RESULTADOS DE SESIÓN
                    # ----------------------------------------

                    st.session_state[
                        "sustax_projection"
                    ] = projection_df

                    st.session_state[
                        "sustax_projection_scenario"
                    ] = escenario_sel

                    st.session_state[
                        "sustax_projection_scenario_signal"
                    ] = scenario_signal_df

                    st.session_state[
                        "sustax_projection_point_signal"
                    ] = point_signal_df

                    st.session_state[
                        "sustax_projection_output"
                    ] = str(output_dir)

                    st.session_state[
                        "sustax_projection_config"
                    ] = configuracion_actual

                    st.session_state[
                        "sustax_projection_export"
                    ] = reporte_exportacion

                    st.success(
                        "Proyección calculada correctamente "
                        f"para {len(projection_df)} puntos."
                    )

                    st.success(
                        "Exportación validada completada: "
                        f"{reporte_exportacion['n_elegibles']} "
                        "puntos elegibles y "
                        f"{reporte_exportacion['n_excluidos']} "
                        "puntos excluidos."
                    )

                    st.caption(
                        "Archivos validados: "
                        f"{reporte_exportacion['carpeta']}"
                    )

                except Exception as exc:

                    st.error(
                        "No fue posible calcular o exportar "
                        "la proyección climática."
                    )

                    st.exception(exc)

        # ====================================================
        # 12.7 RESULTADOS
        # ====================================================

        if "sustax_projection" in st.session_state:

            st.markdown(
                "#### 7. Resultados de precipitación extrema"
            )

            saved_config = st.session_state.get(
                "sustax_projection_config"
            )

            projection_df = st.session_state[
                "sustax_projection"
            ]

            if saved_config is None:

                st.warning(
                    "Existen resultados anteriores "
                    "sin configuración registrada. "
                    "Ejecute nuevamente el análisis."
                )

            elif saved_config != configuracion_actual:

                st.info(
                    "Los resultados guardados corresponden "
                    "a otra configuración. Ejecute "
                    "nuevamente la proyección."
                )

            elif projection_df.empty:

                st.warning(
                    "No existen resultados para mostrar."
                )

            else:

                st.caption(
                    f"{escenario_sel} · "
                    f"Referencia {hist_start}–{hist_end} · "
                    f"Futuro {future_start}–{future_end} · "
                    f"Umbral {threshold_mm:g} mm/día"
                )

                st.metric(
                    "Puntos con resultados válidos",
                    len(projection_df),
                )

                puntos = sorted(
                    projection_df[
                        "sustax_total"
                    ].dropna().astype(str).unique()
                )

                punto_sel = st.selectbox(
                    "Punto climático para consultar indicadores:",
                    options=puntos,
                    key="sustax_result_point",
                )

                registro = projection_df.loc[
                    projection_df[
                        "sustax_total"
                    ].astype(str).eq(punto_sel)
                ].iloc[0]

                tabla_resultados = (
                    tabla_indicadores(
                        registro
                    )
                )

                if tabla_resultados.empty:

                    st.warning(
                        "No se encontraron las columnas "
                        "de indicadores esperadas."
                    )

                else:

                    st.dataframe(
                        tabla_resultados.style.format({
                            "Histórico": "{:.2f}",
                            "Futuro": "{:.2f}",
                            "Cambio absoluto": "{:+.2f}",
                            "Cambio (%)": "{:+.2f}",
                        }, na_rep="N/D"),
                        width="stretch",
                        hide_index=True,
                    )

                st.caption(
                    "P95 y P99 representan percentiles "
                    "de precipitación diaria. Rx1day es "
                    "el promedio de máximos diarios anuales. "
                    "R50mm representa el promedio anual "
                    "de días que alcanzan el umbral definido."
                )

                st.caption(
                    "Los indicadores describen cambios "
                    "en precipitación extrema; no constituyen "
                    "estimaciones directas del riesgo "
                    "de inundación."
                )

                with st.expander(
                    "Ver resultados completos por punto"
                ):

                    st.dataframe(
                        projection_df,
                        width="stretch",
                        hide_index=True,
                    )

                # ----------------------------------------
                # ARCHIVOS DE EXPORTACIÓN
                # ----------------------------------------

                reporte = st.session_state.get(
                    "sustax_projection_export"
                )

                if reporte:

                    with st.expander(
                        "Ver archivos de exportación validada"
                    ):

                        st.write(
                            "**Carpeta de resultados:**"
                        )

                        st.code(
                            reporte["carpeta"],
                            language=None,
                        )

                        st.write(
                            f"Puntos elegibles: "
                            f"{reporte['n_elegibles']}"
                        )

                        st.write(
                            f"Puntos excluidos: "
                            f"{reporte['n_excluidos']}"
                        )

                        for nombre, ruta in [
                            (
                                "Resultados validados",
                                reporte["resultados"],
                            ),
                            (
                                "Diagnóstico de cobertura",
                                reporte["cobertura"],
                            ),
                            (
                                "Puntos excluidos",
                                reporte["excluidos"],
                            ),
                        ]:

                            archivo = Path(ruta)

                            if archivo.is_file():

                                st.download_button(
                                    label=f"Descargar {nombre}",
                                    data=archivo.read_bytes(),
                                    file_name=archivo.name,
                                    mime="text/csv",
                                    key=(
                                        "sustax_download_"
                                        + archivo.name
                                    ),
                                )


# ============================================================
# 13. DIAGNÓSTICOS
# ============================================================

if bad_df is not None and not bad_df.empty:

    with st.expander(
        f"Archivos con error ({len(bad_df)})"
    ):

        st.dataframe(
            bad_df,
            width="stretch",
            hide_index=True,
        )

if errores_periodo:

    with st.expander(
        "Advertencias de cobertura temporal "
        f"({len(errores_periodo)})"
    ):

        st.dataframe(
            pd.DataFrame(errores_periodo),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# 14. INVENTARIO TÉCNICO
# ============================================================

with st.expander(
    "Ver inventario técnico"
):

    columnas_inventario = [
        "sustax_file",
        "base_name",
        "lat",
        "lon",
        "n_rows",
        "n_cols",
        "scenarios_detected",
        "path",
    ]

    columnas_presentes = [
        c for c in columnas_inventario
        if c in inventory_df.columns
    ]

    st.dataframe(
        inventory_df[columnas_presentes],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 15. NOTA METODOLÓGICA
# ============================================================

st.caption(
    "Los archivos originales Sustax se conservan "
    "sin modificaciones. Las series consolidadas "
    "se expresan en milímetros y los resultados "
    "de cambio climático se calculan mediante "
    "los módulos científicos del repositorio "
    "Doctorado-DCAG. La elegibilidad de los puntos "
    "se verifica mediante cobertura diaria "
    "en ambos periodos seleccionados. Los resultados "
    "validados se almacenan separadamente de "
    "los productos técnicos del motor científico."
)
