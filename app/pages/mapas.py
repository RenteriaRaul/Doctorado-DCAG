from pathlib import Path

import streamlit as st

from scripts.gis_manager import GISManager
from scripts.raster_export import (
    exportar_desde_puntos_a_geotiff,
)
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

    return (
        estados[
            estados["disponible"]
        ]
        .copy()
        .reset_index(drop=True)
    )


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


def construir_nombre_salida(
    nivel,
    estado,
    threshold,
    cobertura,
    municipio=None,
    localidad=None,
    ageb=None,
):
    """
    Construye un nombre reproducible para los productos
    cartográficos.
    """

    threshold_texto = (
        str(
            int(threshold)
        )
        if float(threshold).is_integer()
        else str(
            threshold
        ).replace(
            ".",
            "_",
        )
    )

    cobertura_texto = (
        "lineal_nearest"
        if cobertura
        else "lineal"
    )

    partes = [
        f"excedencia_{threshold_texto}mm",
        cobertura_texto,
        nivel,
        estado,
    ]

    if municipio:
        partes.append(
            municipio
        )

    if localidad:
        partes.append(
            localidad
        )

    if ageb:
        partes.append(
            str(ageb)
        )

    return "_".join(
        str(parte)
        .strip()
        .replace(" ", "_")
        for parte in partes
    )


def construir_titulo_territorio(
    nivel,
    estado,
    municipio=None,
    localidad=None,
    ageb=None,
):
    """
    Construye el nombre territorial que aparecerá en el mapa.
    """

    if nivel == "estado":

        return str(
            estado
        )

    if nivel == "municipio":

        return (
            f"{municipio}, "
            f"{estado}"
        )

    if nivel == "localidad":

        return (
            f"Localidad de {localidad}, "
            f"{municipio}, "
            f"{estado}"
        )

    if nivel == "ageb_urbana":

        return (
            f"AGEB urbana {ageb}, "
            f"{localidad}, "
            f"{municipio}, "
            f"{estado}"
        )

    if nivel == "ageb_rural":

        return (
            f"AGEB rural {ageb}, "
            f"{municipio}, "
            f"{estado}"
        )

    return str(
        estado
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "Análisis espacial de excedencias"
)

st.caption(
    "Representación territorial de la probabilidad empírica "
    "de excedencia de precipitación."
)

st.info(
    "Este módulo representa el componente de peligro "
    "hidrometeorológico. La superficie espacial utilizada "
    "se genera previamente en el módulo de Excedencias "
    "a partir de archivos originales de estaciones CONAGUA."
)


# ============================================================
# OBTENER SUPERFICIE ACTIVA
# ============================================================

raster_activo = st.session_state.get(
    "exceedance_raster_path"
)

metadata_raster = st.session_state.get(
    "exceedance_raster_metadata"
)

datos_espaciales = st.session_state.get(
    "exceedance_spatial_data"
)


# ============================================================
# VALIDAR SUPERFICIE
# ============================================================

if not raster_activo:

    st.warning(
        "No existe una superficie de excedencia activa."
    )

    st.info(
        "Primero ingrese al módulo **Excedencias**, "
        "procese las estaciones CONAGUA y genere "
        "la superficie espacial."
    )

    st.stop()


RASTER_BASE = Path(
    raster_activo
)

if not RASTER_BASE.exists():

    st.error(
        "La superficie registrada en la sesión "
        "ya no se encuentra disponible."
    )

    st.code(
        str(RASTER_BASE)
    )

    st.info(
        "Vuelva a ejecutar el análisis en el módulo "
        "de Excedencias."
    )

    st.stop()


# ============================================================
# METADATOS DEL RASTER ACTIVO
# ============================================================

if metadata_raster is None:
    metadata_raster = {}


UMBRAL_MM = float(
    metadata_raster.get(
        "threshold",
        50.0,
    )
)

METODO = metadata_raster.get(
    "method",
    "linear",
)

RESOLUCION_X = metadata_raster.get(
    "nx",
    300,
)

RESOLUCION_Y = metadata_raster.get(
    "ny",
    300,
)

ESTACIONES_VALIDAS = metadata_raster.get(
    "stations_valid",
    "N/D",
)

FUENTE = metadata_raster.get(
    "source",
    "CONAGUA",
)

CRS_RASTER = metadata_raster.get(
    "crs",
    "EPSG:4326",
)


# ============================================================
# RESUMEN DEL ANÁLISIS ACTIVO
# ============================================================

st.success(
    f"Superficie activa: excedencia de precipitación "
    f"≥ {UMBRAL_MM:g} mm."
)

col_activo_1, col_activo_2, col_activo_3, col_activo_4 = (
    st.columns(4)
)

col_activo_1.metric(
    "Umbral",
    f"≥ {UMBRAL_MM:g} mm",
)

col_activo_2.metric(
    "Estaciones",
    ESTACIONES_VALIDAS,
)

col_activo_3.metric(
    "Resolución",
    f"{RESOLUCION_X} × {RESOLUCION_Y}",
)

col_activo_4.metric(
    "Método base",
    str(METODO).capitalize(),
)

st.caption(
    f"Fuente climática: {FUENTE} · "
    f"CRS: {CRS_RASTER}"
)


# ============================================================
# VALIDAR RECURSOS TERRITORIALES
# ============================================================

if not RUTA_MARCO.exists():

    st.error(
        "No se encontró la carpeta del Marco "
        "Geoestadístico INEGI."
    )

    st.code(
        str(RUTA_MARCO)
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
# 1. SELECCIÓN TERRITORIAL
# ============================================================

st.subheader(
    "1. Selección territorial"
)

col_estado, col_nivel = st.columns(
    2
)


# ------------------------------------------------------------
# ESTADO
# ------------------------------------------------------------

with col_estado:

    estados_df = obtener_estados_disponibles(
        gis
    )

    nombres_estados = (
        estados_df[
            "estado"
        ]
        .tolist()
    )

    indice_colima = (
        nombres_estados.index(
            "Colima"
        )
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
        municipios_df[
            "NOMGEO"
        ]
        .tolist()
    )

    indice_manzanillo = (
        nombres_municipios.index(
            "Manzanillo"
        )
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
        localidades_df[
            "NOMGEO"
        ]
        .tolist()
    )

    indice_localidad = (
        opciones_localidades.index(
            "Manzanillo"
        )
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
            "No existen AGEB urbanas para la "
            "selección actual."
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
            "No existen AGEB rurales para la "
            "selección actual."
        )

        st.stop()

    ageb = st.selectbox(
        "AGEB rural",
        options=ageb_rurales,
    )


# ============================================================
# 2. PRODUCTO CARTOGRÁFICO
# ============================================================

st.subheader(
    "2. Producto cartográfico"
)

st.text_input(
    "Variable",
    value=(
        "Probabilidad de excedencia "
        f"≥ {UMBRAL_MM:g} mm"
    ),
    disabled=True,
)


# ------------------------------------------------------------
# COBERTURA ESPACIAL
# ------------------------------------------------------------

completar_vecino = st.checkbox(
    "Completar áreas sin interpolación mediante vecino más cercano",
    value=False,
    help=(
        "Mantiene la interpolación lineal en el área "
        "soportada por las estaciones y utiliza vecino "
        "más cercano únicamente para completar zonas "
        "sin valor."
    ),
)

if completar_vecino:

    st.warning(
        "Las zonas sin soporte de interpolación lineal "
        "serán completadas mediante vecino más cercano. "
        "Esto permite cubrir completamente el territorio, "
        "pero constituye una extrapolación espacial fuera "
        "del dominio respaldado directamente por las estaciones."
    )

else:

    st.caption(
        "Se conservarán como NoData las zonas fuera del "
        "dominio espacial cubierto por la interpolación lineal."
    )


# ------------------------------------------------------------
# SUAVIZADO VISUAL
# ------------------------------------------------------------

suavizado_visual = st.checkbox(
    "Suavizado visual",
    value=False,
    help=(
        "Afecta únicamente la representación cartográfica "
        "del PNG. No modifica el GeoTIFF ni los valores "
        "científicos."
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

            # =================================================
            # RASTER QUE SE UTILIZARÁ
            # =================================================

            RASTER_PARA_MAPA = (
                RASTER_BASE
            )

            METODO_COBERTURA = (
                "Interpolación lineal"
            )


            # =================================================
            # COMPLETAR MEDIANTE VECINO MÁS CERCANO
            # =================================================

            if completar_vecino:

                if (
                    datos_espaciales is None
                    or datos_espaciales.empty
                ):

                    raise ValueError(
                        "No están disponibles los datos "
                        "espaciales de las estaciones para "
                        "generar la superficie completa."
                    )

                threshold_texto = (
                    str(
                        int(
                            UMBRAL_MM
                        )
                    )
                    if float(
                        UMBRAL_MM
                    ).is_integer()
                    else str(
                        UMBRAL_MM
                    ).replace(
                        ".",
                        "_",
                    )
                )

                raster_vecino = (
                    CARPETA_SALIDA
                    / (
                        "superficie_excedencia_"
                        f"{threshold_texto}mm_"
                        "lineal_nearest.tif"
                    )
                )

                exportar_desde_puntos_a_geotiff(
                    df=datos_espaciales,
                    out_tif=str(
                        raster_vecino
                    ),
                    col_lon="longitud",
                    col_lat="latitud",
                    col_val="prob_excedencia",
                    margin=0.0,
                    nx=int(
                        RESOLUCION_X
                    ),
                    ny=int(
                        RESOLUCION_Y
                    ),
                    method="linear",
                    fill_nearest=True,
                    nodata=-9999.0,
                    crs=CRS_RASTER,
                    dtype="float32",
                    eliminar_duplicados=True,
                )

                RASTER_PARA_MAPA = (
                    raster_vecino
                )

                METODO_COBERTURA = (
                    "Interpolación lineal + "
                    "relleno por vecino más cercano"
                )


            # =================================================
            # NOMBRE DE ARCHIVOS
            # =================================================

            nombre_base = construir_nombre_salida(
                nivel=nivel,
                estado=estado,
                threshold=UMBRAL_MM,
                cobertura=(
                    completar_vecino
                ),
                municipio=municipio,
                localidad=localidad,
                ageb=ageb,
            )

            output_tif = (
                CARPETA_SALIDA
                / f"{nombre_base}.tif"
            )

            output_png = (
                CARPETA_SALIDA
                / f"{nombre_base}.png"
            )


            # =================================================
            # TÍTULO TERRITORIAL
            # =================================================

            titulo_territorio = (
                construir_titulo_territorio(
                    nivel=nivel,
                    estado=estado,
                    municipio=municipio,
                    localidad=localidad,
                    ageb=ageb,
                )
            )

            titulo_mapa = (
                "Probabilidad empírica de excedencia "
                f"≥ {UMBRAL_MM:g} mm — "
                f"{titulo_territorio}"
            )


            # =================================================
            # MOTOR TERRITORIAL
            # =================================================

            resultado = (
                generar_mapa_raster_territorial(
                    gis=gis,
                    input_tif=RASTER_PARA_MAPA,
                    output_tif=output_tif,
                    output_png=output_png,
                    nivel=nivel,
                    estado=estado,
                    municipio=municipio,
                    localidad=localidad,
                    ageb=ageb,
                    territorio="principal",
                    title=titulo_mapa,
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
            )


            # =================================================
            # METADATOS ADICIONALES
            # =================================================

            resultado[
                "threshold_mm"
            ] = UMBRAL_MM

            resultado[
                "raster_fuente"
            ] = str(
                RASTER_PARA_MAPA
            )

            resultado[
                "metodo_cobertura"
            ] = METODO_COBERTURA

            resultado[
                "completado_vecino"
            ] = completar_vecino


            # =================================================
            # SESSION STATE
            # =================================================

            st.session_state[
                "resultado_mapa_territorial"
            ] = resultado

            st.session_state[
                "suavizado_mapa_territorial"
            ] = suavizado_visual

            st.session_state[
                "raster_usado_mapa_territorial"
            ] = str(
                RASTER_PARA_MAPA
            )

            st.session_state[
                "raster_base_mapa_territorial"
            ] = str(
                RASTER_BASE
            )

            st.session_state[
                "metodo_cobertura_mapa"
            ] = METODO_COBERTURA

            st.session_state[
                "completar_vecino_mapa"
            ] = completar_vecino


        except Exception as error:

            st.error(
                "No fue posible generar el mapa territorial."
            )

            st.exception(
                error
            )


# ============================================================
# 3. RESULTADO
# ============================================================

resultado = st.session_state.get(
    "resultado_mapa_territorial"
)

raster_base_resultado = st.session_state.get(
    "raster_base_mapa_territorial"
)


# ------------------------------------------------------------
# EVITAR MOSTRAR UN RESULTADO DE OTRO ANÁLISIS
# ------------------------------------------------------------

if (
    resultado is not None
    and raster_base_resultado is not None
    and raster_base_resultado
    != str(
        RASTER_BASE
    )
):

    resultado = None

    st.session_state[
        "resultado_mapa_territorial"
    ] = None


if resultado is not None:

    st.divider()

    st.subheader(
        "3. Resultado"
    )

    st.markdown(
        f"### {resultado['nombre_territorio']}"
    )

    st.caption(
        f"Probabilidad de excedencia de precipitación "
        f"≥ {UMBRAL_MM:g} mm"
    )


    # ========================================================
    # METADATOS DEL RESULTADO
    # ========================================================

    metadata = resultado[
        "mapa"
    ]

    dimensiones = resultado[
        "raster"
    ][
        "dimensiones_recortadas"
    ]

    ancho = int(
        dimensiones[0]
    )

    alto = int(
        dimensiones[1]
    )

    total_celdas_extension = (
        ancho
        * alto
    )

    metodo_cobertura_resultado = (
        resultado.get(
            "metodo_cobertura",
            "Interpolación lineal",
        )
    )


    # ========================================================
    # INDICADORES
    # ========================================================

    col_min, col_prom, col_max = (
        st.columns(3)
    )

    with col_min:

        st.metric(
            "Valor mínimo",
            f"{metadata['valor_minimo']:.3f} %",
        )

    with col_prom:

        st.metric(
            "Valor promedio",
            f"{metadata['valor_promedio']:.3f} %",
        )

    with col_max:

        st.metric(
            "Valor máximo",
            f"{metadata['valor_maximo']:.3f} %",
        )


    # ========================================================
    # INFORMACIÓN DE COBERTURA
    # ========================================================

    if resultado.get(
        "completado_vecino",
        False,
    ):

        st.info(
            "La superficie mostrada utiliza interpolación "
            "lineal dentro del dominio de las estaciones y "
            "vecino más cercano para completar las áreas "
            "sin cobertura."
        )

    else:

        st.info(
            "La superficie mostrada conserva únicamente "
            "los valores respaldados por la interpolación "
            "lineal. Las zonas sin soporte permanecen "
            "como NoData."
        )


    # ========================================================
    # ADVERTENCIA DE RESOLUCIÓN
    # ========================================================

    if total_celdas_extension <= 25:

        st.warning(
            "Resolución espacial limitada para el "
            "territorio seleccionado. "
            f"El raster recortado abarca aproximadamente "
            f"{total_celdas_extension} celdas en su extensión "
            "rectangular. El resultado permite caracterizar "
            "el valor territorial, pero no debe interpretarse "
            "como evidencia de variabilidad espacial detallada."
        )

    elif total_celdas_extension <= 100:

        st.info(
            "El territorio seleccionado presenta una "
            "resolución raster relativamente limitada. "
            "Se recomienda interpretar con cautela los "
            "patrones espaciales internos."
        )


    # ========================================================
    # MAPA
    # ========================================================

    st.image(
        resultado[
            "output_png"
        ],
        use_container_width=True,
    )


    # ========================================================
    # RESUMEN DEL PRODUCTO
    # ========================================================

    col_info_1, col_info_2, col_info_3 = (
        st.columns(3)
    )

    with col_info_1:

        st.caption(
            f"Nivel territorial: "
            f"{resultado['nivel']}"
        )

    with col_info_2:

        st.caption(
            f"CRS: "
            f"{metadata['crs']}"
        )

    with col_info_3:

        st.caption(
            f"Umbral: "
            f"≥ {UMBRAL_MM:g} mm"
        )

    st.caption(
        "Fuente climática: CONAGUA · "
        "Fuente territorial: Marco Geoestadístico INEGI."
    )


    # ========================================================
    # 4. DESCARGAS
    # ========================================================

    st.subheader(
        "4. Descargas"
    )

    col_png, col_tif = st.columns(
        2
    )

    with open(
        resultado[
            "output_png"
        ],
        "rb",
    ) as archivo_png:

        png_bytes = (
            archivo_png.read()
        )

    with open(
        resultado[
            "output_tif"
        ],
        "rb",
    ) as archivo_tif:

        tif_bytes = (
            archivo_tif.read()
        )


    with col_png:

        st.download_button(
            label="Descargar PNG",
            data=png_bytes,
            file_name=Path(
                resultado[
                    "output_png"
                ]
            ).name,
            mime="image/png",
            use_container_width=True,
        )


    with col_tif:

        st.download_button(
            label="Descargar GeoTIFF territorial",
            data=tif_bytes,
            file_name=Path(
                resultado[
                    "output_tif"
                ]
            ).name,
            mime="image/tiff",
            use_container_width=True,
        )


    # ========================================================
    # INFORMACIÓN TÉCNICA
    # ========================================================

    with st.expander(
        "Información técnica"
    ):

        st.write(
            "**Análisis de origen**"
        )

        st.write(
            {
                "umbral_mm": UMBRAL_MM,
                "fuente": FUENTE,
                "metodo_base": METODO,
                "metodo_cobertura": (
                    metodo_cobertura_resultado
                ),
                "resolucion": (
                    f"{RESOLUCION_X} × "
                    f"{RESOLUCION_Y}"
                ),
                "estaciones_utilizadas": (
                    ESTACIONES_VALIDAS
                ),
                "raster_base": str(
                    RASTER_BASE
                ),
                "raster_utilizado": resultado.get(
                    "raster_fuente"
                ),
                "relleno_vecino_mas_cercano": (
                    resultado.get(
                        "completado_vecino",
                        False,
                    )
                ),
            }
        )

        st.write(
            "**Territorio**"
        )

        st.json(
            {
                clave: (
                    float(
                        valor
                    )
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
            "**Raster territorial**"
        )

        st.write(
            {
                "dimensiones": dimensiones,
                "celdas_extension_rectangular": (
                    total_celdas_extension
                ),
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
                "umbral_precipitacion_mm": (
                    UMBRAL_MM
                ),
                "suavizado_visual": (
                    st.session_state.get(
                        "suavizado_mapa_territorial",
                        False,
                    )
                ),
            }
        )