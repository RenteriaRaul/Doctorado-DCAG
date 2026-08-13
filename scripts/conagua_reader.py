from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar_texto(
    texto,
):
    """
    Normaliza texto para comparaciones internas.

    - elimina espacios externos;
    - convierte a mayúsculas;
    - elimina acentos.

    No modifica los valores originales que se entregan
    posteriormente al usuario.
    """

    if pd.isna(texto):
        return ""

    texto = str(
        texto
    ).strip()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(
            caracter
        )
    )

    return texto.upper()


# ============================================================
# EXTRAER VALOR NUMÉRICO
# ============================================================

def extraer_numero(
    valor,
):
    """
    Extrae el primer número presente en un valor.

    Ejemplos
    --------
    "18.93833333 °"   -> 18.93833333
    "-103.9463889 °"  -> -103.9463889
    "37 msnm"         -> 37.0
    """

    if pd.isna(valor):
        return np.nan

    if isinstance(
        valor,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return float(
            valor
        )

    texto = str(
        valor
    ).strip()

    coincidencia = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        texto,
    )

    if coincidencia is None:
        return np.nan

    try:
        return float(
            coincidencia.group()
        )

    except ValueError:
        return np.nan


# ============================================================
# NORMALIZAR CLAVE DE ESTACIÓN
# ============================================================

def normalizar_clave_estacion(
    valor,
):
    """
    Normaliza la clave de estación como texto.

    Ejemplos
    --------
    6001       -> "6001"
    6001.0     -> "6001"
    "6001"     -> "6001"
    "dat6001"  -> "6001"
    """

    if pd.isna(valor):
        return None

    texto = str(
        valor
    ).strip()

    coincidencia = re.search(
        r"(\d+)",
        texto,
    )

    if coincidencia is None:
        return texto

    return coincidencia.group(
        1
    )


# ============================================================
# LOCALIZAR HOJA DE INFORMACIÓN
# ============================================================

def localizar_hoja_informacion(
    excel_file,
):
    """
    Localiza la hoja que contiene metadatos de la estación.

    No depende exclusivamente del nombre de la hoja.
    Busca campos característicos como ESTACIÓN, LATITUD
    y LONGITUD.
    """

    for nombre_hoja in excel_file.sheet_names:

        muestra = pd.read_excel(
            excel_file,
            sheet_name=nombre_hoja,
            header=None,
            nrows=30,
        )

        valores = {
            normalizar_texto(valor)
            for valor in muestra.values.flatten()
            if pd.notna(valor)
        }

        campos = {
            "ESTACION",
            "LATITUD",
            "LONGITUD",
        }

        if campos.issubset(
            valores
        ):
            return nombre_hoja

    raise ValueError(
        "No fue posible localizar una hoja con los "
        "metadatos de la estación."
    )


# ============================================================
# LOCALIZAR HOJA DE DATOS CLIMÁTICOS
# ============================================================

def localizar_hoja_datos_clima(
    excel_file,
):
    """
    Localiza la hoja que contiene la serie climática.

    Busca una fila con FECHA y PRECIP.
    """

    for nombre_hoja in excel_file.sheet_names:

        muestra = pd.read_excel(
            excel_file,
            sheet_name=nombre_hoja,
            header=None,
            nrows=20,
        )

        for _, fila in muestra.iterrows():

            valores = {
                normalizar_texto(valor)
                for valor in fila.tolist()
                if pd.notna(valor)
            }

            if {
                "FECHA",
                "PRECIP",
            }.issubset(
                valores
            ):
                return nombre_hoja

    raise ValueError(
        "No fue posible localizar la hoja con "
        "los datos climáticos."
    )


# ============================================================
# EXTRAER METADATOS
# ============================================================

def leer_metadatos_conagua(
    archivo,
    hoja=None,
):
    """
    Extrae los metadatos de una estación CONAGUA.

    Retorna un diccionario estandarizado.
    """

    archivo = Path(
        archivo
    )

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {archivo}"
        )

    excel_file = pd.ExcelFile(
        archivo
    )

    if hoja is None:
        hoja = localizar_hoja_informacion(
            excel_file
        )

    df = pd.read_excel(
        excel_file,
        sheet_name=hoja,
        header=None,
    )

    if df.shape[1] < 2:
        raise ValueError(
            "La hoja de información no contiene "
            "al menos dos columnas."
        )

    metadata_raw = {}

    for _, fila in df.iterrows():

        clave = fila.iloc[0]
        valor = fila.iloc[1]

        clave_norm = normalizar_texto(
            clave
        )

        if clave_norm:
            metadata_raw[
                clave_norm
            ] = valor

    clave_estacion = (
        metadata_raw.get(
            "ESTACION"
        )
    )

    metadata = {
        "station": normalizar_clave_estacion(
            clave_estacion
        ),
        "nombre": metadata_raw.get(
            "NOMBRE"
        ),
        "estado": metadata_raw.get(
            "ESTADO"
        ),
        "municipio": metadata_raw.get(
            "MUNICIPIO"
        ),
        "situacion": metadata_raw.get(
            "SITUACION"
        ),
        "cve_omm": metadata_raw.get(
            "CVE-OMM"
        ),
        "latitud": extraer_numero(
            metadata_raw.get(
                "LATITUD"
            )
        ),
        "longitud": extraer_numero(
            metadata_raw.get(
                "LONGITUD"
            )
        ),
        "altitud_msnm": extraer_numero(
            metadata_raw.get(
                "ALTITUD"
            )
        ),
        "emision": metadata_raw.get(
            "EMISION"
        ),
        "archivo": str(
            archivo
        ),
        "hoja_informacion": hoja,
    }

    if metadata[
        "station"
    ] is None:
        raise ValueError(
            "No se encontró la clave de estación."
        )

    if not np.isfinite(
        metadata[
            "latitud"
        ]
    ):
        raise ValueError(
            "No se encontró una latitud válida."
        )

    if not np.isfinite(
        metadata[
            "longitud"
        ]
    ):
        raise ValueError(
            "No se encontró una longitud válida."
        )

    return metadata


# ============================================================
# DETECTAR FILA DE ENCABEZADOS CLIMÁTICOS
# ============================================================

def detectar_fila_encabezado_clima(
    excel_file,
    hoja,
):
    """
    Localiza la fila que contiene FECHA y PRECIP.
    """

    muestra = pd.read_excel(
        excel_file,
        sheet_name=hoja,
        header=None,
        nrows=50,
    )

    for indice, fila in muestra.iterrows():

        valores = [
            normalizar_texto(valor)
            for valor in fila.tolist()
        ]

        if (
            "FECHA" in valores
            and "PRECIP" in valores
        ):
            return int(
                indice
            )

    raise ValueError(
        "No se encontró la fila de encabezados "
        "de los datos climáticos."
    )


# ============================================================
# LEER DATOS CLIMÁTICOS
# ============================================================

def leer_datos_clima_conagua(
    archivo,
    hoja=None,
):
    """
    Lee la serie climática de una estación CONAGUA.

    Devuelve columnas estandarizadas:

    date
    pp
    evap
    tmax
    tmin
    """

    archivo = Path(
        archivo
    )

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {archivo}"
        )

    excel_file = pd.ExcelFile(
        archivo
    )

    if hoja is None:
        hoja = localizar_hoja_datos_clima(
            excel_file
        )

    fila_header = detectar_fila_encabezado_clima(
        excel_file=excel_file,
        hoja=hoja,
    )

    df = pd.read_excel(
        excel_file,
        sheet_name=hoja,
        header=fila_header,
    )

    # --------------------------------------------------------
    # NORMALIZAR NOMBRES DE COLUMNAS
    # --------------------------------------------------------

    mapa_columnas = {}

    for columna in df.columns:

        columna_norm = normalizar_texto(
            columna
        )

        if columna_norm == "FECHA":
            mapa_columnas[
                columna
            ] = "date"

        elif columna_norm in {
            "PRECIP",
            "PRECIPITACION",
        }:
            mapa_columnas[
                columna
            ] = "pp"

        elif columna_norm in {
            "EVAP",
            "EVAPORACION",
        }:
            mapa_columnas[
                columna
            ] = "evap"

        elif columna_norm in {
            "TMAX",
            "TEMP MAX",
            "TEMPERATURA MAXIMA",
        }:
            mapa_columnas[
                columna
            ] = "tmax"

        elif columna_norm in {
            "TMIN",
            "TEMP MIN",
            "TEMPERATURA MINIMA",
        }:
            mapa_columnas[
                columna
            ] = "tmin"

    df = df.rename(
        columns=mapa_columnas
    )

    if "date" not in df.columns:
        raise ValueError(
            "No se encontró la columna de fecha."
        )

    if "pp" not in df.columns:
        raise ValueError(
            "No se encontró la columna de precipitación."
        )

    # --------------------------------------------------------
    # CONVERSIÓN DE DATOS
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    columnas_numericas = [
        columna
        for columna in [
            "pp",
            "evap",
            "tmax",
            "tmin",
        ]
        if columna in df.columns
    ]

    for columna in columnas_numericas:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    # Se eliminan únicamente filas sin fecha.
    # Los NaN climáticos permanecen para control de calidad.
    df = df.dropna(
        subset=[
            "date"
        ]
    ).copy()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    columnas_salida = [
        columna
        for columna in [
            "date",
            "pp",
            "evap",
            "tmax",
            "tmin",
        ]
        if columna in df.columns
    ]

    return df[
        columnas_salida
    ].copy()


# ============================================================
# LEER UNA ESTACIÓN COMPLETA
# ============================================================

def leer_estacion_conagua(
    archivo,
):
    """
    Lee un archivo original de estación CONAGUA.

    Retorna
    -------
    dict
        {
            "metadata": dict,
            "data": DataFrame,
            "calidad": dict,
        }
    """

    archivo = Path(
        archivo
    )

    metadata = leer_metadatos_conagua(
        archivo
    )

    data = leer_datos_clima_conagua(
        archivo
    )

    calidad = {
        "station": metadata[
            "station"
        ],
        "archivo": str(
            archivo
        ),
        "total_registros": len(
            data
        ),
        "fecha_inicio": (
            data["date"].min()
            if not data.empty
            else pd.NaT
        ),
        "fecha_fin": (
            data["date"].max()
            if not data.empty
            else pd.NaT
        ),
        "precipitacion_valida": int(
            data["pp"]
            .notna()
            .sum()
        )
        if "pp" in data.columns
        else 0,
        "precipitacion_nula": int(
            data["pp"]
            .isna()
            .sum()
        )
        if "pp" in data.columns
        else 0,
        "latitud": metadata[
            "latitud"
        ],
        "longitud": metadata[
            "longitud"
        ],
    }

    return {
        "metadata": metadata,
        "data": data,
        "calidad": calidad,
    }


# ============================================================
# LISTAR ARCHIVOS CONAGUA
# ============================================================

def listar_archivos_conagua(
    carpeta,
    patron="*.xlsx",
):
    """
    Lista archivos Excel potencialmente compatibles
    con el formato original CONAGUA.
    """

    carpeta = Path(
        carpeta
    ).expanduser()

    if not carpeta.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta: {carpeta}"
        )

    if not carpeta.is_dir():
        raise NotADirectoryError(
            f"La ruta no corresponde a una carpeta: {carpeta}"
        )

    archivos = sorted(
        carpeta.glob(
            patron
        )
    )

    # Ignorar archivos temporales creados por Excel
    # cuando un libro se encuentra abierto.
    archivos_validos = [
        archivo
        for archivo in archivos
        if not archivo.name.startswith("~$")
    ]

    return archivos_validos


# ============================================================
# LEER LOTE DE ESTACIONES
# ============================================================

def leer_lote_conagua(
    carpeta,
    patron="*.xlsx",
):
    """
    Lee un conjunto de estaciones originales CONAGUA.

    Los errores de un archivo no detienen todo el lote.

    Retorna
    -------
    estaciones : list[dict]
        Una entrada por estación válida.

    metadata_df : pd.DataFrame
        Tabla maestra de metadatos.

    log_df : pd.DataFrame
        Resultado de lectura por archivo.
    """

    archivos = listar_archivos_conagua(
        carpeta=carpeta,
        patron=patron,
    )

    if not archivos:
        raise FileNotFoundError(
            "No se encontraron archivos compatibles "
            f"con el patrón '{patron}'."
        )

    estaciones = []
    metadata_rows = []
    log_rows = []

    for archivo in archivos:

        try:

            estacion = leer_estacion_conagua(
                archivo
            )

            estaciones.append(
                estacion
            )

            metadata_rows.append(
                estacion[
                    "metadata"
                ]
            )

            log_rows.append(
                {
                    "archivo": str(
                        archivo
                    ),
                    "station": estacion[
                        "metadata"
                    ][
                        "station"
                    ],
                    "status": "ok",
                    "mensaje": "",
                    "registros": len(
                        estacion[
                            "data"
                        ]
                    ),
                }
            )

        except Exception as error:

            log_rows.append(
                {
                    "archivo": str(
                        archivo
                    ),
                    "station": None,
                    "status": "error",
                    "mensaje": str(
                        error
                    ),
                    "registros": 0,
                }
            )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    log_df = pd.DataFrame(
        log_rows
    )

    return (
        estaciones,
        metadata_df,
        log_df,
    )