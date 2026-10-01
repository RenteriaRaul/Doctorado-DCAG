from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.sustax.inventory import (
    build_sustax_inventory,
    build_group_summary,
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
    group_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una tabla amigable para Streamlit con un registro
    por punto Sustax.
    """
    if group_summary_df.empty:
        return pd.DataFrame()

    tabla = group_summary_df.copy()

    escenarios_por_punto = {}

    if not inventory_df.empty:
        for keys, grp in inventory_df.groupby(
            ["base_name", "lat", "lon"],
            dropna=False,
        ):
            escenarios = set()

            for valor in grp["scenarios_detected"].dropna():
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

            escenarios_por_punto[keys] = (
                ", ".join(ordenados + otros)
            )

    def buscar_escenarios(row):
        key = (
            row["base_name"],
            row["lat"],
            row["lon"],
        )

        return escenarios_por_punto.get(
            key,
            "No identificado",
        )

    tabla["escenarios"] = tabla.apply(
        buscar_escenarios,
        axis=1,
    )

    tabla = tabla.rename(
        columns={
            "base_name": "Punto Sustax",
            "lat": "Latitud",
            "lon": "Longitud",
            "n_files": "Archivos",
            "files": "Archivos originales",
            "escenarios": "Escenarios",
        }
    )

    columnas = [
        "Punto Sustax",
        "Latitud",
        "Longitud",
        "Archivos",
        "Escenarios",
        "Archivos originales",
    ]

    columnas = [
        c
        for c in columnas
        if c in tabla.columns
    ]

    return tabla[columnas]


# ============================================================
# ENCABEZADO
# ============================================================

st.title("Datos climáticos Sustax")

st.caption(
    "Inventario y verificación de archivos climáticos originales "
    "Sustax utilizados en el análisis doctoral."
)

st.info(
    "Esta etapa identifica los archivos disponibles, sus coordenadas "
    "y los escenarios climáticos contenidos en ellos. "
    "No se ejecutan todavía análisis de extremos, validación ERA5, "
    "proyecciones SSP ni modelos de peligro."
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
        "Se detectaron datos ERA5 para validación histórica."
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
    "del repositorio Doctorado-DCAG. La conversión, integración y análisis "
    "de escenarios se incorporarán en las siguientes etapas."
)