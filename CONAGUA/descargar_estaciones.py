"""
Módulo para descarga y procesamiento de estaciones climatológicas
del SMN-CONAGUA.

Funciones principales:
- Obtener catálogo oficial de estaciones.
- Filtrar estaciones por estado.
- Descargar registros diarios.
- Procesar archivos TXT.
- Guardar resultados en Excel.
"""

from pathlib import Path
from io import StringIO
import re
import unicodedata

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

URL_GEOJSON = (
    "https://smn.conagua.gob.mx/"
    "tools/GUI/estaciones-climatologicas/"
    "data/estaciones_climatologicas.geojson"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36"
    )
}

COLUMNAS_CLIMA = [
    "FECHA",
    "PRECIP",
    "EVAP",
    "TMAX",
    "TMIN",
]


# ============================================================
# 1. NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(texto):
    """
    Convierte un texto a mayúsculas y elimina acentos.
    """

    texto = str(texto).strip().upper()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


# ============================================================
# 2. LIMPIAR NOMBRE PARA ARCHIVOS
# ============================================================

def limpiar_nombre(texto):
    """
    Genera nombres seguros para carpetas y archivos.
    """

    texto = normalizar_texto(texto)

    texto = re.sub(
        r'[<>:"/\\|?*]',
        "",
        texto
    )

    texto = re.sub(
        r"\s+",
        "_",
        texto
    )

    return texto


# ============================================================
# 3. OBTENER CATÁLOGO OFICIAL
# ============================================================

def obtener_catalogo_conagua():

    print("Consultando catálogo oficial de CONAGUA...")

    response = requests.get(
        URL_GEOJSON,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    datos = response.json()

    features = datos.get(
        "features",
        []
    )

    estaciones = []

    for feature in features:

        propiedades = feature.get(
            "properties",
            {}
        ).copy()

        coordenadas = (
            feature
            .get("geometry", {})
            .get("coordinates", [])
        )

        if len(coordenadas) >= 2:

            propiedades["LONGITUD"] = (
                coordenadas[0]
            )

            propiedades["LATITUD"] = (
                coordenadas[1]
            )

        else:

            propiedades["LONGITUD"] = None
            propiedades["LATITUD"] = None

        estaciones.append(
            propiedades
        )

    df = pd.DataFrame(
        estaciones
    )

    print(
        f"Estaciones encontradas: {len(df):,}"
    )

    return df


# ============================================================
# 4. FILTRAR POR ESTADO
# ============================================================

def filtrar_estaciones(
    df,
    estado,
    situacion=None
):

    estado_busqueda = normalizar_texto(
        estado
    )

    mascara = (
        df["estado"]
        .fillna("")
        .apply(normalizar_texto)
        == estado_busqueda
    )

    resultado = df[
        mascara
    ].copy()

    # Filtro opcional por situación
    if situacion is not None:

        situacion_busqueda = normalizar_texto(
            situacion
        )

        resultado = resultado[
            resultado["situacion"]
            .fillna("")
            .apply(normalizar_texto)
            == situacion_busqueda
        ].copy()

    return resultado


# ============================================================
# 5. DESCARGAR TXT DIARIO
# ============================================================

def descargar_txt_diario(url):

    if pd.isna(url) or not str(url).strip():

        raise ValueError(
            "La estación no tiene URL de datos diarios."
        )

    response = requests.get(
        str(url).strip(),
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    texto = response.text

    if "<html" in texto.lower():

        raise ValueError(
            "La URL devolvió HTML en lugar "
            "de datos climatológicos."
        )

    return texto


# ============================================================
# 6. PROCESAR TXT DIARIO
# ============================================================

def procesar_txt_diario(texto):

    lineas = texto.splitlines()

    inicio_datos = None
    metadatos = {}

    # --------------------------------------------------------
    # Localizar encabezado
    # --------------------------------------------------------

    for i, linea in enumerate(lineas):

        linea_limpia = linea.strip()

        if "FECHA" in linea_limpia.upper():

            inicio_datos = i
            break

        if ":" in linea_limpia:

            clave_info, valor = (
                linea_limpia.split(
                    ":",
                    1
                )
            )

            metadatos[
                clave_info.strip()
            ] = valor.strip()

    if inicio_datos is None:

        raise ValueError(
            "No se encontró el encabezado FECHA."
        )

    # --------------------------------------------------------
    # Extraer registros
    # --------------------------------------------------------

    datos_clima = [
        linea.strip()
        for linea in lineas[
            inicio_datos + 2:
        ]
        if linea.strip()
    ]

    if not datos_clima:

        raise ValueError(
            "El archivo no contiene registros climáticos."
        )

    # --------------------------------------------------------
    # Crear DataFrame
    # --------------------------------------------------------

    df = pd.read_csv(
        StringIO(
            "\n".join(datos_clima)
        ),
        sep=r"\s+",
        names=COLUMNAS_CLIMA,
        engine="python"
    )

    filas_originales = len(df)

    # --------------------------------------------------------
    # Fecha
    # IMPORTANTE:
    # CONAGUA utiliza YYYY-MM-DD
    # --------------------------------------------------------

    df["FECHA"] = pd.to_datetime(
        df["FECHA"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    fechas_invalidas = (
        df["FECHA"]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    for columna in [
        "PRECIP",
        "EVAP",
        "TMAX",
        "TMIN"
    ]:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Eliminar únicamente fechas inválidas
    # --------------------------------------------------------

    df = df.dropna(
        subset=["FECHA"]
    ).copy()

    # --------------------------------------------------------
    # Ordenar
    # --------------------------------------------------------

    df = df.sort_values(
        "FECHA"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Duplicados
    # --------------------------------------------------------

    duplicados = (
        df.duplicated(
            subset=["FECHA"]
        )
        .sum()
    )

    # NO eliminamos duplicados silenciosamente.
    # Primero queremos detectarlos.

    diagnostico = {
        "filas_originales":
            filas_originales,

        "filas_validas":
            len(df),

        "fechas_invalidas":
            int(fechas_invalidas),

        "fechas_duplicadas":
            int(duplicados),

        "fecha_inicio":
            df["FECHA"].min(),

        "fecha_fin":
            df["FECHA"].max(),

        "precip_max":
            df["PRECIP"].max(),

        "eventos_50mm":
            int(
                (df["PRECIP"] >= 50)
                .sum()
            ),

        "faltantes_precip":
            int(
                df["PRECIP"]
                .isna()
                .sum()
            )
    }

    return (
        df,
        metadatos,
        diagnostico
    )


# ============================================================
# 7. GUARDAR ESTACIÓN EN EXCEL
# ============================================================

def guardar_estacion_excel(
    df,
    metadatos,
    estacion,
    carpeta_salida
):

    clave = str(
        estacion["clave"]
    ).strip()

    nombre = limpiar_nombre(
        estacion["nombre"]
    )

    carpeta_salida = Path(
        carpeta_salida
    )

    carpeta_salida.mkdir(
        parents=True,
        exist_ok=True
    )

    archivo = (
        carpeta_salida
        / f"{clave}_{nombre}.xlsx"
    )

    # --------------------------------------------------------
    # Información
    # --------------------------------------------------------

    df_info = pd.DataFrame(
        list(
            metadatos.items()
        ),
        columns=[
            "Clave",
            "Valor"
        ]
    )

    # --------------------------------------------------------
    # Exportar
    # --------------------------------------------------------

    with pd.ExcelWriter(
        archivo,
        engine="xlsxwriter"
    ) as writer:

        df_info.to_excel(
            writer,
            sheet_name="Información",
            index=False
        )

        df.to_excel(
            writer,
            sheet_name="Datos Clima",
            index=False
        )

    return archivo


# ============================================================
# 8. PROCESAR UNA ESTACIÓN COMPLETA
# ============================================================

def procesar_estacion(
    estacion,
    carpeta_salida
):

    clave = str(
        estacion["clave"]
    ).strip()

    nombre = str(
        estacion["nombre"]
    ).strip()

    print(
        f"\nProcesando "
        f"{clave} - {nombre}"
    )

    texto = descargar_txt_diario(
        estacion["diarios"]
    )

    (
        df,
        metadatos,
        diagnostico
    ) = procesar_txt_diario(
        texto
    )

    archivo = guardar_estacion_excel(
        df=df,
        metadatos=metadatos,
        estacion=estacion,
        carpeta_salida=carpeta_salida
    )

    return {
        "clave":
            clave,

        "nombre":
            nombre,

        "archivo":
            str(archivo),

        **diagnostico
    }

def descargar_estado(
    estado,
    carpeta_base="CONAGUA/datos",
    situacion=None
):
    """
    Descarga y procesa todas las estaciones de un estado.

    Parámetros
    ----------
    estado : str
        Nombre del estado, por ejemplo "COLIMA".

    carpeta_base : str
        Carpeta donde se guardarán los archivos.

    situacion : str o None
        Puede ser:
        - None → todas
        - "OPERANDO"
        - "SUSPENDIDA"
    """

    print("=" * 70)
    print(f"DESCARGA DE ESTACIONES - {estado}")
    print("=" * 70)

    # Obtener catálogo completo
    catalogo = obtener_catalogo_conagua()

    # Filtrar estado
    estaciones = filtrar_estaciones(
        catalogo,
        estado=estado,
        situacion=situacion
    )

    print(
        f"\nEstaciones encontradas: "
        f"{len(estaciones)}"
    )

    if estaciones.empty:
        print("No se encontraron estaciones.")
        return None, None

    # Carpeta por estado
    carpeta_estado = (
        Path(carpeta_base)
        / limpiar_nombre(estado)
    )

    carpeta_estado.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados = []
    errores = []

    total = len(estaciones)

    # Recorrer estaciones
    for numero, (_, estacion) in enumerate(
        estaciones.iterrows(),
        start=1
    ):

        clave = str(
            estacion["clave"]
        ).strip()

        nombre = str(
            estacion["nombre"]
        ).strip()

        print()
        print("-" * 70)
        print(
            f"[{numero}/{total}] "
            f"{clave} - {nombre}"
        )

        try:

            resultado = procesar_estacion(
                estacion,
                carpeta_estado
            )

            resultados.append(
                resultado
            )

            print(
                f"  OK | "
                f"{resultado['filas_validas']:,} registros"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            errores.append({
                "clave": clave,
                "nombre": nombre,
                "error": str(e)
            })

    # ========================================================
    # Guardar diagnóstico
    # ========================================================

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

    else:

        df_resultados = pd.DataFrame()

    # ========================================================
    # Guardar errores
    # ========================================================

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

    else:

        df_errores = pd.DataFrame()

    # ========================================================
    # Resumen
    # ========================================================

    print()
    print("=" * 70)
    print("DESCARGA TERMINADA")
    print("=" * 70)

    print(
        "Estado:",
        estado
    )

    print(
        "Estaciones encontradas:",
        total
    )

    print(
        "Descargas correctas:",
        len(resultados)
    )

    print(
        "Estaciones con error:",
        len(errores)
    )

    print(
        "Carpeta:",
        carpeta_estado
    )

    return (
        df_resultados,
        df_errores
    )
    
def descargar_estaciones_estado(
    estado,
    carpeta_salida="CONAGUA/estaciones_descargadas"
):
    """
    Descarga todas las estaciones climatológicas disponibles
    de un estado desde el catálogo oficial del SMN-CONAGUA.

    Parámetros
    ----------
    estado : str
        Nombre del estado. Ejemplo:
        "COLIMA", "JALISCO", "MICHOACAN", "NAYARIT"

    carpeta_salida : str
        Carpeta base donde se guardarán los archivos Excel.

    Retorna
    -------
    DataFrame
        Resumen de estaciones procesadas.
    """

    print("=" * 70)
    print(f"DESCARGA AUTOMÁTICA DE ESTACIONES - {estado}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Obtener catálogo oficial
    # --------------------------------------------------------

    catalogo = obtener_catalogo_conagua()

    # --------------------------------------------------------
    # 2. Filtrar por estado
    # --------------------------------------------------------

    estaciones = filtrar_estaciones(
        catalogo,
        estado=estado
    )

    if estaciones.empty:
        print(
            f"No se encontraron estaciones para {estado}."
        )
        return pd.DataFrame()

    print(
        f"\nEstaciones encontradas en {estado}: "
        f"{len(estaciones)}"
    )

    # --------------------------------------------------------
    # 3. Crear carpeta específica del estado
    # --------------------------------------------------------

    nombre_estado = limpiar_nombre(
        estado
    )

    carpeta_estado = (
        Path(carpeta_salida)
        / nombre_estado
    )

    carpeta_estado.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nLos archivos se guardarán en:"
    )

    print(
        carpeta_estado
    )

    # --------------------------------------------------------
    # 4. Inicializar resultados
    # --------------------------------------------------------

    resultados = []

    total = len(estaciones)

    # --------------------------------------------------------
    # 5. Recorrer estaciones
    # --------------------------------------------------------

    for numero, (_, estacion) in enumerate(
        estaciones.iterrows(),
        start=1
    ):

        clave = str(
            estacion["clave"]
        ).strip()

        nombre = str(
            estacion["nombre"]
        ).strip()

        print()
        print("-" * 70)

        print(
            f"[{numero}/{total}] "
            f"{clave} - {nombre}"
        )

        try:

            resultado = procesar_estacion(
                estacion,
                carpeta_estado
            )

            resultado["estado"] = estado
            resultado["estatus_descarga"] = "OK"
            resultado["error"] = ""

            resultados.append(
                resultado
            )

            print(
                f"  Descarga correcta"
            )

            print(
                f"  Registros: "
                f"{resultado['filas_validas']:,}"
            )

            print(
                f"  Periodo: "
                f"{resultado['fecha_inicio'].date()} "
                f"a "
                f"{resultado['fecha_fin'].date()}"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            resultados.append({

                "clave":
                    clave,

                "nombre":
                    nombre,

                "estado":
                    estado,

                "archivo":
                    None,

                "filas_originales":
                    None,

                "filas_validas":
                    None,

                "fechas_invalidas":
                    None,

                "fechas_duplicadas":
                    None,

                "fecha_inicio":
                    None,

                "fecha_fin":
                    None,

                "precip_max":
                    None,

                "eventos_50mm":
                    None,

                "faltantes_precip":
                    None,

                "estatus_descarga":
                    "ERROR",

                "error":
                    str(e)
            })

    # --------------------------------------------------------
    # 6. Crear resumen
    # --------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # 7. Guardar diagnóstico general
    # --------------------------------------------------------

    archivo_resumen = (
        carpeta_estado
        / "resumen_descarga.csv"
    )

    df_resultados.to_csv(
        archivo_resumen,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 8. Mostrar resumen final
    # --------------------------------------------------------

    correctas = (
        df_resultados[
            "estatus_descarga"
        ]
        == "OK"
    ).sum()

    errores = (
        df_resultados[
            "estatus_descarga"
        ]
        == "ERROR"
    ).sum()

    print()
    print("=" * 70)
    print("DESCARGA FINALIZADA")
    print("=" * 70)

    print(
        f"Estado: {estado}"
    )

    print(
        f"Estaciones encontradas: {total}"
    )

    print(
        f"Descargas correctas: {correctas}"
    )

    print(
        f"Descargas con error: {errores}"
    )

    print(
        "\nCarpeta de salida:"
    )

    print(
        carpeta_estado
    )

    print(
        "\nResumen:"
    )

    print(
        archivo_resumen
    )

    return df_resultados

if __name__ == "__main__":
    
    descargar_estaciones_estado(
        estado="JALISCO"
    )