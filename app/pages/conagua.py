from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# MOTORES CONAGUA
# ============================================================

from CONAGUA.descargar_estaciones import (
    obtener_catalogo_conagua,
    filtrar_estaciones,
    procesar_estacion,
    limpiar_nombre,
)

from scripts.conagua_reader import (
    leer_estacion_conagua,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_DESCARGAS = (
    ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
)


# ============================================================
# CACHE DEL CATÁLOGO CONAGUA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def cargar_catalogo_conagua():
    """
    Consulta el catálogo oficial de estaciones climatológicas
    del SMN-CONAGUA.
    """

    return obtener_catalogo_conagua()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def construir_nombre_archivo(estacion):
    """
    Genera el nombre esperado del archivo Excel.
    """

    clave = str(
        estacion["clave"]
    ).strip()

    nombre = limpiar_nombre(
        estacion["nombre"]
    )

    return f"{clave}_{nombre}.xlsx"


def archivo_estacion_existe(
    estacion,
    carpeta_estado
):
    """
    Determina si una estación ya existe localmente.
    """

    archivo = (
        carpeta_estado
        / construir_nombre_archivo(estacion)
    )

    return archivo.exists()


def guardar_diagnosticos(
    resultados,
    errores,
    carpeta_estado
):
    """
    Guarda los archivos diagnósticos asociados a una descarga.
    """

    if resultados:

        df_resultados = pd.DataFrame(
            resultados
        )

        archivo_diagnostico = (
            carpeta_estado
            / "diagnostico_descarga.csv"
        )

        df_resultados.to_csv(
            archivo_diagnostico,
            index=False,
            encoding="utf-8-sig"
        )

    if errores:

        df_errores = pd.DataFrame(
            errores
        )

        archivo_errores = (
            carpeta_estado
            / "errores_descarga.csv"
        )

        df_errores.to_csv(
            archivo_errores,
            index=False,
            encoding="utf-8-sig"
        )


# ============================================================
# FUNCIONES PARA INTEGRACIÓN CON CONAGUA_READER
# ============================================================

def obtener_metadatos_estacion(estacion):
    """
    Recupera el diccionario de metadatos devuelto por
    conagua_reader.py.
    """

    if not isinstance(estacion, dict):
        return {}

    if "metadata" in estacion:
        metadata = estacion["metadata"]

    elif "metadatos" in estacion:
        metadata = estacion["metadatos"]

    else:
        metadata = {}

    if metadata is None:
        metadata = {}

    return metadata


def obtener_datos_estacion(estacion):
    """
    Recupera el DataFrame climático sin evaluar
    DataFrames como booleanos.
    """

    if not isinstance(estacion, dict):
        return None

    if "data" in estacion:
        return estacion["data"]

    if "datos" in estacion:
        return estacion["datos"]

    if "clima" in estacion:
        return estacion["clima"]

    return None


def obtener_metadata(
    metadata,
    posibles_claves
):
    """
    Busca un valor dentro del diccionario de metadatos
    considerando distintas formas de nombrar una variable.
    """

    if not isinstance(metadata, dict):
        return None

    for clave in posibles_claves:

        if clave in metadata:

            valor = metadata[clave]

            if pd.notna(valor):
                return valor

    return None


def generar_firma_carpeta(carpeta):
    """
    Genera una firma basada en nombre, tamaño y fecha de
    modificación de los XLSX.

    Esta firma permite invalidar la caché cuando se descarga
    o actualiza una estación.
    """

    if not carpeta.exists():
        return tuple()

    archivos = sorted(
        archivo
        for archivo in carpeta.glob("*.xlsx")
        if not archivo.name.startswith("~$")
    )

    firma = []

    for archivo in archivos:

        stat = archivo.stat()

        firma.append(
            (
                archivo.name,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )

    return tuple(firma)


@st.cache_data(
    show_spinner=False
)
def analizar_archivos_disponibles(
    carpeta_texto,
    firma
):
    """
    Analiza todos los XLSX disponibles utilizando
    scripts/conagua_reader.py.

    La firma se utiliza únicamente para invalidar la caché
    cuando los archivos cambian.
    """

    carpeta = Path(
        carpeta_texto
    )

    archivos = sorted(
        archivo
        for archivo in carpeta.glob("*.xlsx")
        if not archivo.name.startswith("~$")
    )

    compatibles = []
    errores = []

    for archivo in archivos:

        try:

            estacion = leer_estacion_conagua(
                archivo
            )

            if estacion is None:

                raise ValueError(
                    "El lector devolvió None."
                )

            if not isinstance(
                estacion,
                dict
            ):

                raise TypeError(
                    "El lector no devolvió un diccionario."
                )

            metadata = obtener_metadatos_estacion(
                estacion
            )

            datos = obtener_datos_estacion(
                estacion
            )

            if datos is None:

                raise ValueError(
                    "No se encontró el DataFrame climático."
                )

            if not isinstance(
                datos,
                pd.DataFrame
            ):

                raise TypeError(
                    "Los datos climáticos no son un DataFrame."
                )

            if datos.empty:

                raise ValueError(
                    "El DataFrame climático está vacío."
                )

            columnas_requeridas = [
                "date",
                "pp",
            ]

            faltantes = [
                columna
                for columna in columnas_requeridas
                if columna not in datos.columns
            ]

            if faltantes:

                raise ValueError(
                    f"Faltan columnas: {faltantes}"
                )

            datos = datos.copy()

            datos["date"] = pd.to_datetime(
                datos["date"],
                errors="coerce"
            )

            datos["pp"] = pd.to_numeric(
                datos["pp"],
                errors="coerce"
            )

            fechas_validas = int(
                datos["date"]
                .notna()
                .sum()
            )

            precip_validas = int(
                datos["pp"]
                .notna()
                .sum()
            )

            if fechas_validas == 0:

                raise ValueError(
                    "No existen fechas válidas."
                )

            if precip_validas == 0:

                raise ValueError(
                    "No existe precipitación válida."
                )

            clave = obtener_metadata(
                metadata,
                [
                    "clave",
                    "CLAVE",
                    "estacion",
                    "ESTACION",
                ]
            )

            if clave is None:

                clave = (
                    archivo
                    .stem
                    .split("_")[0]
                )

            nombre = obtener_metadata(
                metadata,
                [
                    "nombre",
                    "NOMBRE",
                ]
            )

            if nombre is None:

                nombre = archivo.stem

            estado_meta = obtener_metadata(
                metadata,
                [
                    "estado",
                    "ESTADO",
                ]
            )

            municipio = obtener_metadata(
                metadata,
                [
                    "municipio",
                    "MUNICIPIO",
                ]
            )

            situacion = obtener_metadata(
                metadata,
                [
                    "situacion",
                    "SITUACION",
                ]
            )

            latitud = obtener_metadata(
                metadata,
                [
                    "latitud",
                    "LATITUD",
                    "latitude",
                    "LATITUDE",
                ]
            )

            longitud = obtener_metadata(
                metadata,
                [
                    "longitud",
                    "LONGITUD",
                    "longitude",
                    "LONGITUDE",
                ]
            )

            fecha_inicio = (
                datos["date"]
                .min()
            )

            fecha_fin = (
                datos["date"]
                .max()
            )

            precip_max = (
                datos["pp"]
                .max()
            )

            faltantes_pp = int(
                datos["pp"]
                .isna()
                .sum()
            )

            eventos_50 = int(
                (
                    datos["pp"] >= 50
                ).sum()
            )

            duplicados = int(
                datos["date"]
                .duplicated()
                .sum()
            )

            compatibles.append(
                {
                    "archivo":
                        archivo.name,

                    "clave":
                        str(clave),

                    "nombre":
                        nombre,

                    "estado":
                        estado_meta,

                    "municipio":
                        municipio,

                    "situacion":
                        situacion,

                    "registros":
                        len(datos),

                    "fecha_inicio":
                        fecha_inicio,

                    "fecha_fin":
                        fecha_fin,

                    "precip_max":
                        precip_max,

                    "eventos_50mm":
                        eventos_50,

                    "faltantes_pp":
                        faltantes_pp,

                    "duplicados":
                        duplicados,

                    "latitud":
                        latitud,

                    "longitud":
                        longitud,
                }
            )

        except Exception as error:

            errores.append(
                {
                    "archivo":
                        archivo.name,

                    "tipo_error":
                        type(error).__name__,

                    "error":
                        str(error),
                }
            )

    return compatibles, errores


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "📡 Datos climatológicos CONAGUA"
)

st.write(
    """
    Consulta, descarga y prepara registros de estaciones
    climatológicas del Servicio Meteorológico Nacional
    (SMN-CONAGUA) para su utilización en los módulos
    científicos de la plataforma.
    """
)


# ============================================================
# 1. CATÁLOGO OFICIAL
# ============================================================

st.subheader(
    "1. Catálogo de estaciones"
)

try:

    with st.spinner(
        "Consultando catálogo oficial de CONAGUA..."
    ):

        catalogo = cargar_catalogo_conagua()

except Exception as error:

    st.error(
        "No fue posible consultar el catálogo oficial "
        "de estaciones de CONAGUA."
    )

    st.exception(
        error
    )

    st.stop()


if catalogo.empty:

    st.warning(
        "El catálogo oficial no contiene estaciones."
    )

    st.stop()


# ============================================================
# 2. SELECCIÓN TERRITORIAL
# ============================================================

st.subheader(
    "2. Selección territorial"
)

estados = (
    catalogo["estado"]
    .dropna()
    .astype(str)
    .str.strip()
    .sort_values()
    .unique()
    .tolist()
)

col_estado, col_situacion = st.columns(2)

with col_estado:

    indice_colima = (
        estados.index("COLIMA")
        if "COLIMA" in estados
        else 0
    )

    estado = st.selectbox(
        "Estado",
        options=estados,
        index=indice_colima
    )

with col_situacion:

    situacion = st.selectbox(
        "Situación de la estación",
        options=[
            "OPERANDO",
            "TODAS",
            "SUSPENDIDA",
        ],
        index=0
    )


situacion_filtro = (
    None
    if situacion == "TODAS"
    else situacion
)


estaciones = filtrar_estaciones(
    catalogo,
    estado=estado,
    situacion=situacion_filtro
).copy()


if estaciones.empty:

    st.warning(
        "No se encontraron estaciones con los filtros seleccionados."
    )

    st.stop()


estaciones["clave"] = (
    estaciones["clave"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CARPETA DEL ESTADO
# ============================================================

carpeta_estado = (
    CARPETA_DESCARGAS
    / limpiar_nombre(estado)
)

carpeta_estado.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. MÉTRICAS DEL CATÁLOGO
# ============================================================

st.subheader(
    "3. Estaciones encontradas"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Estaciones",
    len(estaciones)
)


if "situacion" in estaciones.columns:

    estaciones_operando = (
        estaciones["situacion"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("OPERANDO")
        .sum()
    )

else:

    estaciones_operando = 0


m2.metric(
    "Operando",
    int(estaciones_operando)
)

m3.metric(
    "Estado",
    estado
)


archivos_existentes = sum(
    archivo_estacion_existe(
        estacion,
        carpeta_estado
    )
    for _, estacion in estaciones.iterrows()
)


m4.metric(
    "Ya descargadas",
    archivos_existentes
)


# ============================================================
# 4. SELECCIÓN DE ESTACIONES
# ============================================================

st.subheader(
    "4. Selección de estaciones"
)

columnas_deseadas = [
    "clave",
    "nombre",
    "municipio",
    "situacion",
    "LATITUD",
    "LONGITUD",
]

columnas_disponibles = [
    columna
    for columna in columnas_deseadas
    if columna in estaciones.columns
]

tabla = estaciones[
    columnas_disponibles
].copy()

tabla.insert(
    0,
    "Seleccionar",
    False
)

tabla["Descargada"] = [
    archivo_estacion_existe(
        estacion,
        carpeta_estado
    )
    for _, estacion in estaciones.iterrows()
]


tabla = tabla.rename(
    columns={
        "clave":
            "Clave",

        "nombre":
            "Estación",

        "municipio":
            "Municipio",

        "situacion":
            "Situación",

        "LATITUD":
            "Latitud",

        "LONGITUD":
            "Longitud",
    }
)


# ============================================================
# CONTROL SELECCIONAR TODAS
# ============================================================

col_todas, col_ninguna = st.columns(2)

with col_todas:

    if st.button(
        "☑️ Seleccionar todas",
        use_container_width=True
    ):

        st.session_state[
            "conagua_seleccionar_todas"
        ] = True

with col_ninguna:

    if st.button(
        "⬜ Limpiar selección",
        use_container_width=True
    ):

        st.session_state[
            "conagua_seleccionar_todas"
        ] = False


if st.session_state.get(
    "conagua_seleccionar_todas",
    False
):

    tabla[
        "Seleccionar"
    ] = True


# ============================================================
# TABLA INTERACTIVA
# ============================================================

tabla_editada = st.data_editor(
    tabla,
    use_container_width=True,
    hide_index=True,
    disabled=[
        columna
        for columna in tabla.columns
        if columna != "Seleccionar"
    ],
    column_config={

        "Seleccionar":
            st.column_config.CheckboxColumn(
                "Seleccionar",
                help=(
                    "Marque las estaciones que desea descargar."
                ),
                default=False
            ),

        "Descargada":
            st.column_config.CheckboxColumn(
                "Ya descargada",
                disabled=True
            ),
    }
)


filas_seleccionadas = tabla_editada[
    tabla_editada["Seleccionar"]
].copy()

cantidad_seleccionada = len(
    filas_seleccionadas
)


st.caption(
    f"Estaciones seleccionadas: "
    f"{cantidad_seleccionada}"
)


# ============================================================
# 5. CONFIGURACIÓN DE DESCARGA
# ============================================================

st.subheader(
    "5. Configuración de descarga"
)

politica_existentes = st.radio(
    "Si una estación ya fue descargada:",
    options=[
        "Omitir archivo existente",
        "Actualizar y reemplazar",
    ],
    index=0,
    horizontal=True
)

st.caption(
    f"Carpeta de destino: `{carpeta_estado}`"
)


# ============================================================
# 6. DESCARGAR
# ============================================================

st.subheader(
    "6. Descargar estaciones"
)

boton_descargar = st.button(
    "⬇️ Descargar estaciones seleccionadas",
    type="primary",
    use_container_width=True,
    disabled=(
        cantidad_seleccionada == 0
    )
)


if boton_descargar:

    claves_seleccionadas = (
        filas_seleccionadas[
            "Clave"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    estaciones_descarga = estaciones[
        estaciones[
            "clave"
        ].isin(
            claves_seleccionadas
        )
    ].copy()

    total = len(
        estaciones_descarga
    )

    resultados = []
    errores = []
    omitidas = []

    barra = st.progress(
        0
    )

    texto_estado = st.empty()

    for numero, (_, estacion) in enumerate(
        estaciones_descarga.iterrows(),
        start=1
    ):

        clave = str(
            estacion["clave"]
        ).strip()

        nombre = str(
            estacion["nombre"]
        ).strip()

        texto_estado.write(
            f"Procesando {numero}/{total}: "
            f"{clave} — {nombre}"
        )

        archivo_existe = archivo_estacion_existe(
            estacion,
            carpeta_estado
        )

        if (
            archivo_existe
            and politica_existentes
            == "Omitir archivo existente"
        ):

            omitidas.append(
                {
                    "clave":
                        clave,

                    "nombre":
                        nombre,

                    "motivo":
                        "Archivo existente"
                }
            )

            barra.progress(
                numero / total
            )

            continue

        try:

            resultado = procesar_estacion(
                estacion=estacion,
                carpeta_salida=carpeta_estado
            )

            resultado[
                "estatus_descarga"
            ] = "OK"

            resultado[
                "error"
            ] = ""

            resultados.append(
                resultado
            )

        except Exception as error:

            errores.append(
                {
                    "clave":
                        clave,

                    "nombre":
                        nombre,

                    "estatus_descarga":
                        "ERROR",

                    "error":
                        str(error)
                }
            )

        barra.progress(
            numero / total
        )


    texto_estado.empty()


    guardar_diagnosticos(
        resultados=resultados,
        errores=errores,
        carpeta_estado=carpeta_estado
    )


    st.success(
        "Proceso de descarga finalizado."
    )


    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Solicitadas",
        total
    )

    r2.metric(
        "Descargadas",
        len(resultados)
    )

    r3.metric(
        "Omitidas",
        len(omitidas)
    )

    r4.metric(
        "Errores",
        len(errores)
    )


    if resultados:

        st.subheader(
            "Descargas correctas"
        )

        df_resultados = pd.DataFrame(
            resultados
        )

        columnas_resultados = [
            columna
            for columna in [
                "clave",
                "nombre",
                "filas_validas",
                "fecha_inicio",
                "fecha_fin",
                "precip_max",
                "eventos_50mm",
                "faltantes_precip",
            ]
            if columna in df_resultados.columns
        ]

        tabla_resultados = df_resultados[
            columnas_resultados
        ].copy()

        tabla_resultados = tabla_resultados.rename(
            columns={
                "clave":
                    "Clave",

                "nombre":
                    "Estación",

                "filas_validas":
                    "Registros válidos",

                "fecha_inicio":
                    "Fecha inicial",

                "fecha_fin":
                    "Fecha final",

                "precip_max":
                    "Precipitación máxima (mm)",

                "eventos_50mm":
                    "Eventos ≥ 50 mm",

                "faltantes_precip":
                    "Faltantes de precipitación",
            }
        )

        st.dataframe(
            tabla_resultados,
            use_container_width=True,
            hide_index=True
        )


    if omitidas:

        with st.expander(
            f"Estaciones omitidas ({len(omitidas)})"
        ):

            st.dataframe(
                pd.DataFrame(
                    omitidas
                ),
                use_container_width=True,
                hide_index=True
            )


    if errores:

        st.warning(
            "Algunas estaciones no pudieron descargarse."
        )

        with st.expander(
            f"Errores de descarga ({len(errores)})"
        ):

            st.dataframe(
                pd.DataFrame(
                    errores
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 7. DATOS DISPONIBLES PARA ANÁLISIS
# ============================================================

st.divider()

st.subheader(
    "7. Datos disponibles para análisis"
)

st.write(
    """
    Los archivos descargados se verifican automáticamente mediante
    el lector científico de la plataforma antes de considerarlos
    disponibles para los módulos de análisis.
    """
)


firma = generar_firma_carpeta(
    carpeta_estado
)


if not firma:

    st.info(
        "Todavía no existen estaciones descargadas "
        "para este estado."
    )

else:

    with st.spinner(
        "Verificando archivos disponibles..."
    ):

        compatibles, errores_lectura = (
            analizar_archivos_disponibles(
                str(carpeta_estado),
                firma
            )
        )


    df_compatibles = pd.DataFrame(
        compatibles
    )


    total_archivos = len(
        firma
    )

    total_compatibles = len(
        compatibles
    )

    total_errores = len(
        errores_lectura
    )


    if not df_compatibles.empty:

        coordenadas_completas = int(
            df_compatibles[
                [
                    "latitud",
                    "longitud"
                ]
            ]
            .notna()
            .all(axis=1)
            .sum()
        )

    else:

        coordenadas_completas = 0


    # ========================================================
    # MÉTRICAS
    # ========================================================

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Archivos descargados",
        total_archivos
    )

    d2.metric(
        "Compatibles",
        total_compatibles
    )

    d3.metric(
        "Con coordenadas",
        coordenadas_completas
    )

    d4.metric(
        "Errores",
        total_errores
    )


    # ========================================================
    # COBERTURA TEMPORAL GLOBAL
    # ========================================================

    if not df_compatibles.empty:

        fecha_global_inicio = (
            df_compatibles[
                "fecha_inicio"
            ].min()
        )

        fecha_global_fin = (
            df_compatibles[
                "fecha_fin"
            ].max()
        )


        st.info(
            "📅 **Cobertura global del conjunto:** "
            f"{fecha_global_inicio:%d/%m/%Y} "
            "→ "
            f"{fecha_global_fin:%d/%m/%Y}\n\n"
            "Este intervalo representa la unión temporal de "
            "las estaciones disponibles; no implica que todas "
            "las estaciones posean registros continuos durante "
            "todo el periodo."
        )


    # ========================================================
    # ESTADO GENERAL
    # ========================================================

    if (
        total_archivos > 0
        and total_errores == 0
    ):

        st.success(
            "Todos los archivos descargados son compatibles "
            "con el lector científico de la plataforma."
        )

    elif total_compatibles > 0:

        st.warning(
            "Existen archivos compatibles y otros que requieren "
            "revisión."
        )

    else:

        st.error(
            "Ningún archivo disponible pudo ser interpretado "
            "correctamente."
        )


    # ========================================================
    # TABLA DE ESTACIONES DISPONIBLES
    # ========================================================

    if not df_compatibles.empty:

        with st.expander(
            "Ver estaciones disponibles",
            expanded=False
        ):

            tabla_disponibles = (
                df_compatibles.copy()
            )

            columnas_tabla = [
                "clave",
                "nombre",
                "registros",
                "fecha_inicio",
                "fecha_fin",
                "precip_max",
                "eventos_50mm",
                "faltantes_pp",
                "duplicados",
                "latitud",
                "longitud",
            ]

            columnas_tabla = [
                columna
                for columna in columnas_tabla
                if columna
                in tabla_disponibles.columns
            ]

            tabla_disponibles = (
                tabla_disponibles[
                    columnas_tabla
                ]
            )

            tabla_disponibles = (
                tabla_disponibles.rename(
                    columns={
                        "clave":
                            "Clave",

                        "nombre":
                            "Estación",

                        "registros":
                            "Registros",

                        "fecha_inicio":
                            "Fecha inicial",

                        "fecha_fin":
                            "Fecha final",

                        "precip_max":
                            "Máx. precipitación (mm)",

                        "eventos_50mm":
                            "Eventos ≥ 50 mm",

                        "faltantes_pp":
                            "Faltantes PP",

                        "duplicados":
                            "Fechas duplicadas",

                        "latitud":
                            "Latitud",

                        "longitud":
                            "Longitud",
                    }
                )
            )

            st.dataframe(
                tabla_disponibles,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # ERRORES DEL LECTOR
    # ========================================================

    if errores_lectura:

        with st.expander(
            f"Archivos con problemas ({total_errores})"
        ):

            st.dataframe(
                pd.DataFrame(
                    errores_lectura
                ),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # DISPONIBILIDAD PARA MÓDULOS CIENTÍFICOS
    # ========================================================

    if total_compatibles > 0:

        st.markdown(
            "#### Disponibilidad para módulos científicos"
        )

        st.write(
            """
            La compatibilidad estructural indica que estas estaciones
            pueden ser entregadas a los motores científicos de la
            plataforma. Cada módulo aplicará posteriormente sus propios
            criterios metodológicos de suficiencia y control de calidad.
            """
        )

        mod1, mod2, mod3, mod4 = st.columns(4)

        mod1.success(
            "📈 GEV\n\nDatos disponibles"
        )

        mod2.success(
            "🌧️ Excedencias\n\nDatos disponibles"
        )

        mod3.success(
            "🗺️ Mapas\n\nDatos disponibles"
        )

        mod4.success(
            "📊 Tendencias\n\nDatos disponibles"
        )


# ============================================================
# INFORMACIÓN METODOLÓGICA
# ============================================================

with st.expander(
    "¿Cómo se utilizan estos datos?"
):

    st.write(
        """
        Los archivos descargados contienen los registros
        climatológicos diarios disponibles para cada estación.

        La compatibilidad mostrada en esta página corresponde
        únicamente a una validación estructural de los archivos:
        presencia de fechas, precipitación y metadatos necesarios
        para que puedan ser interpretados por la plataforma.

        Esto no significa automáticamente que una estación sea
        estadísticamente adecuada para cualquier análisis.

        Por ejemplo, el módulo GEV determinará posteriormente si
        existe un número suficiente de máximos anuales y aplicará
        sus propios controles de calidad estadística.

        De forma equivalente, los módulos de excedencias, tendencias
        y análisis espacial conservarán sus respectivos criterios
        científicos.
        """
    )