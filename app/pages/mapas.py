from pathlib import Path

import streamlit as st

from scripts.gis_manager import GISManager
from scripts.territorial_mapping import (
    generar_mapa_raster_territorial,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUTA_MARCO = Path(
    r"C:\Users\rente\Desktop\Marco_geoestadistico_2025"
)

RASTER_BASE = (
    PROJECT_ROOT
    / "results"
    / "test_mapas"
    / "excedencia_interpolada.tif"
)

CARPETA_SALIDA = (
    PROJECT_ROOT
    / "results"
    / "streamlit"
    / "mapas_territoriales"
)


# ============================================================
# INICIALIZAR GIS MANAGER
# ============================================================

@st.cache_resource
def cargar_gis_manager():
    """
    Inicializa GISManager una sola vez durante la sesión.
    """

    return GISManager(
        root=RUTA_MARCO
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_estados_disponibles(
    gis,
):
    """
    Devuelve las entidades con paquete INEGI disponible.
    """

    estados = gis.listar_estados()

    estados = estados[
        estados["disponible"]
    ].copy()

    return estados


def obtener_lista_municipios(
    gis,
    estado,
):
    """
    Obtiene municipios del estado seleccionado.
    """

    gdf = gis.obtener_municipios(
        estado=estado,
        territorio="principal",
    )

    return (
        gdf[
            [
                "CVE_MUN",
                "NOMGEO",
            ]
        ]
        .sort_values("NOMGEO")
        .reset_index(drop=True)
    )


def obtener_lista_localidades(
    gis,
    estado,
    municipio,
):
    """
    Obtiene localidades amanzanadas del municipio.
    """

    gdf = gis.obtener_localidades(
        estado=estado,
        municipio=municipio,
        territorio="principal",
    )

    return (
        gdf[
            [
                "CVE_LOC",
                "NOMGEO",
                "AMBITO",
            ]
        ]
        .sort_values("NOMGEO")
        .reset_index(drop=True)
    )


def obtener_lista_ageb_urbanas(
    gis,
    estado,
    municipio,
    localidad,
):
    """
    Obtiene AGEB urbanas disponibles.
    """

    gdf = gis.obtener_ageb_urbanas(
        estado=estado,
        municipio=municipio,
        localidad=localidad,
        territorio="principal",
    )

    return sorted(
        gdf["CVE_AGEB"]
        .astype(str)
        .unique()
        .tolist()
    )


def obtener_lista_ageb_rurales(
    gis,
    estado,
    municipio,
):
    """
    Obtiene AGEB rurales disponibles.
    """

    gdf = gis.obtener_ageb_rurales(
        estado=estado,
        municipio=municipio,
        territorio="principal",
    )

    return sorted(
        gdf["CVE_AGEB"]
        .astype(str)
        .unique()
        .tolist()
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "Mapas territoriales"
)

st.caption(
    "Exploración espacial de la probabilidad empírica "
    "de excedencia de precipitación."
)

st.info(
    "Este módulo representa actualmente el componente de peligro "
    "hidrometeorológico. La integración de exposición y vulnerabilidad "
    "para construir los mapas de riesgo se realizará posteriormente."
)


# ============================================================
# VALIDAR RECURSOS
# ============================================================

if not RUTA_MARCO.exists():

    st.error(
        "No se encontró la carpeta del Marco Geoestadístico INEGI."
    )

    st.code(
        str(RUTA_MARCO)
    )

    st.stop()


if not RASTER_BASE.exists():

    st.warning(
        "No se encontró todavía el GeoTIFF base de excedencia."
    )

    st.code(
        str(RASTER_BASE)
    )

    st.info(
        "Ejecute primero el análisis de excedencias "
        "y la generación del raster interpolado."
    )

    st.stop()


CARPETA_SALIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GIS MANAGER
# ============================================================

try:

    gis = cargar_gis_manager()

except Exception as error:

    st.error(
        "No fue posible iniciar el gestor territorial."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# SELECTORES TERRITORIALES
# ============================================================

st.subheader(
    "1. Selección territorial"
)

col_estado, col_nivel = st.columns(2)


# ------------------------------------------------------------
# ESTADO
# ------------------------------------------------------------

with col_estado:

    estados_df = obtener_estados_disponibles(
        gis
    )

    nombres_estados = (
        estados_df["estado"]
        .tolist()
    )

    indice_colima = (
        nombres_estados.index("Colima")
        if "Colima" in nombres_estados
        else 0
    )

    estado = st.selectbox(
        "Estado",
        options=nombres_estados,
        index=indice_colima,
    )


# ------------------------------------------------------------
# NIVEL
# ------------------------------------------------------------

with col_nivel:

    opciones_nivel = {
        "Estado": "estado",
        "Municipio": "municipio",
        "Localidad": "localidad",
        "AGEB urbana": "ageb_urbana",
        "AGEB rural": "ageb_rural",
    }

    nivel_visible = st.selectbox(
        "Nivel territorial",
        options=list(
            opciones_nivel.keys()
        ),
    )

    nivel = opciones_nivel[
        nivel_visible
    ]


# ============================================================
# SELECTORES DEPENDIENTES
# ============================================================

municipio = None
localidad = None
ageb = None


# ------------------------------------------------------------
# MUNICIPIO
# ------------------------------------------------------------

if nivel in {
    "municipio",
    "localidad",
    "ageb_urbana",
    "ageb_rural",
}:

    municipios_df = obtener_lista_municipios(
        gis=gis,
        estado=estado,
    )

    nombres_municipios = (
        municipios_df["NOMGEO"]
        .tolist()
    )

    indice_manzanillo = (
        nombres_municipios.index("Manzanillo")
        if "Manzanillo" in nombres_municipios
        else 0
    )

    municipio = st.selectbox(
        "Municipio",
        options=nombres_municipios,
        index=indice_manzanillo,
    )


# ------------------------------------------------------------
# LOCALIDAD
# ------------------------------------------------------------

if nivel in {
    "localidad",
    "ageb_urbana",
}:

    localidades_df = obtener_lista_localidades(
        gis=gis,
        estado=estado,
        municipio=municipio,
    )

    opciones_localidades = (
        localidades_df["NOMGEO"]
        .tolist()
    )

    indice_localidad = (
        opciones_localidades.index("Manzanillo")
        if "Manzanillo" in opciones_localidades
        else 0
    )

    localidad = st.selectbox(
        "Localidad",
        options=opciones_localidades,
        index=indice_localidad,
    )


# ------------------------------------------------------------
# AGEB URBANA
# ------------------------------------------------------------

if nivel == "ageb_urbana":

    ageb_urbanas = obtener_lista_ageb_urbanas(
        gis=gis,
        estado=estado,
        municipio=municipio,
        localidad=localidad,
    )

    if not ageb_urbanas:

        st.warning(
            "No existen AGEB urbanas para la selección actual."
        )

        st.stop()

    ageb = st.selectbox(
        "AGEB urbana",
        options=ageb_urbanas,
    )


# ------------------------------------------------------------
# AGEB RURAL
# ------------------------------------------------------------

if nivel == "ageb_rural":

    ageb_rurales = obtener_lista_ageb_rurales(
        gis=gis,
        estado=estado,
        municipio=municipio,
    )

    if not ageb_rurales:

        st.warning(
            "No existen AGEB rurales para la selección actual."
        )

        st.stop()

    ageb = st.selectbox(
        "AGEB rural",
        options=ageb_rurales,
    )


# ============================================================
# PARÁMETROS DEL PRODUCTO
# ============================================================

st.subheader(
    "2. Producto cartográfico"
)

variable = st.selectbox(
    "Variable",
    options=[
        "Probabilidad de excedencia ≥ 50 mm",
    ],
    disabled=True,
)

suavizado_visual = st.checkbox(
    "Suavizado cartográfico para visualización",
    value=False,
    help=(
        "El suavizado afecta únicamente la representación visual "
        "del PNG. No modifica el GeoTIFF ni los valores científicos."
    ),
)


# ============================================================
# GENERAR MAPA
# ============================================================

generar = st.button(
    "Generar mapa territorial",
    type="primary",
    use_container_width=True,
)


if generar:

    with st.spinner(
        "Generando producto territorial..."
    ):

        try:

            # ------------------------------------------------
            # IDENTIFICADOR DE SALIDA
            # ------------------------------------------------

            partes_nombre = [
                nivel,
                estado,
            ]

            if municipio:
                partes_nombre.append(
                    municipio
                )

            if localidad:
                partes_nombre.append(
                    localidad
                )

            if ageb:
                partes_nombre.append(
                    str(ageb)
                )

            nombre_base = "_".join(
                str(parte)
                .strip()
                .replace(" ", "_")
                for parte in partes_nombre
            )

            output_tif = (
                CARPETA_SALIDA
                / f"{nombre_base}.tif"
            )

            output_png = (
                CARPETA_SALIDA
                / f"{nombre_base}.png"
            )

            # ------------------------------------------------
            # MOTOR TERRITORIAL
            # ------------------------------------------------

            resultado = generar_mapa_raster_territorial(
                gis=gis,
                input_tif=RASTER_BASE,
                output_tif=output_tif,
                output_png=output_png,
                nivel=nivel,
                estado=estado,
                municipio=municipio,
                localidad=localidad,
                ageb=ageb,
                territorio="principal",
                colorbar_label=(
                    "Probabilidad de excedencia (%)"
                ),
                cmap="YlOrRd",
                convertir_a_porcentaje=True,
                mostrar_limite=True,
                suavizar_visual=suavizado_visual,
                sigma_suavizado=1.2,
                interpolation_display="bilinear",
                figsize=(11, 9),
            )

            st.session_state[
                "resultado_mapa_territorial"
            ] = resultado

        except Exception as error:

            st.error(
                "No fue posible generar el mapa territorial."
            )

            st.exception(
                error
            )


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

resultado = st.session_state.get(
    "resultado_mapa_territorial"
)


if resultado is not None:

    st.divider()

    st.subheader(
        "3. Resultado"
    )

    st.markdown(
        f"### {resultado['nombre_territorio']}"
    )

    # --------------------------------------------------------
    # MAPA + INDICADORES
    # --------------------------------------------------------

    col_mapa, col_info = st.columns(
        [3, 1]
    )

    with col_mapa:

        st.image(
            resultado["output_png"],
            use_container_width=True,
        )

    with col_info:

        metadata = resultado[
            "mapa"
        ]

        st.metric(
            "Valor mínimo",
            f"{metadata['valor_minimo']:.3f} %",
        )

        st.metric(
            "Valor máximo",
            f"{metadata['valor_maximo']:.3f} %",
        )

        st.metric(
            "Valor promedio",
            f"{metadata['valor_promedio']:.3f} %",
        )

        st.caption(
            f"Nivel: {resultado['nivel']}"
        )

        st.caption(
            f"CRS: {metadata['crs']}"
        )

        st.caption(
            "Fuente territorial: "
            "Marco Geoestadístico INEGI."
        )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------

    st.subheader(
        "4. Descargas"
    )

    col_png, col_tif = st.columns(2)

    with open(
        resultado["output_png"],
        "rb",
    ) as archivo_png:

        png_bytes = archivo_png.read()

    with open(
        resultado["output_tif"],
        "rb",
    ) as archivo_tif:

        tif_bytes = archivo_tif.read()

    with col_png:

        st.download_button(
            label="Descargar PNG",
            data=png_bytes,
            file_name=Path(
                resultado["output_png"]
            ).name,
            mime="image/png",
            use_container_width=True,
        )

    with col_tif:

        st.download_button(
            label="Descargar GeoTIFF",
            data=tif_bytes,
            file_name=Path(
                resultado["output_tif"]
            ).name,
            mime="image/tiff",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # DETALLE TÉCNICO
    # --------------------------------------------------------

    with st.expander(
        "Información técnica"
    ):

        st.write(
            "**Territorio**"
        )

        st.json(
            {
                clave: (
                    float(valor)
                    if hasattr(
                        valor,
                        "item",
                    )
                    else valor
                )
                for clave, valor
                in resultado[
                    "territorio"
                ].items()
            }
        )

        st.write(
            "**Raster**"
        )

        st.write(
            {
                "dimensiones": resultado[
                    "raster"
                ][
                    "dimensiones_recortadas"
                ],
                "bounds": resultado[
                    "raster"
                ][
                    "bounds_recortados"
                ],
                "nodata": resultado[
                    "raster"
                ][
                    "nodata"
                ],
            }
        )