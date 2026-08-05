import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# UTILIDADES INTERNAS
# ============================================================

def _nombre_columna_excedencia(threshold):
    """
    Genera un nombre descriptivo para la columna de excedencia.

    Ejemplos
    --------
    50      -> EXCEDENCIA_50MM
    75.5    -> EXCEDENCIA_75_5MM
    """
    threshold_float = float(threshold)

    if threshold_float.is_integer():
        threshold_texto = str(int(threshold_float))
    else:
        threshold_texto = str(threshold_float).replace(".", "_")

    return f"EXCEDENCIA_{threshold_texto}MM"


def _validar_threshold(threshold):
    """
    Valida que el umbral sea numérico, finito y no negativo.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "El umbral de precipitación debe ser numérico."
        ) from error

    if not np.isfinite(threshold):
        raise ValueError(
            "El umbral de precipitación debe ser un valor finito."
        )

    if threshold < 0:
        raise ValueError(
            "El umbral de precipitación no puede ser negativo."
        )

    return threshold


# ============================================================
# LECTURA Y LIMPIEZA DE CSV
# ============================================================

def cargar_y_limpiar_precipitacion_csv(
    path_csv,
    col_precip="pp",
    col_fecha="date",
    consolidar_duplicados=True,
):
    """
    Carga y limpia los datos diarios de precipitación de una estación CSV.

    El procedimiento realiza:

    - lectura del archivo;
    - validación de columnas;
    - conversión de precipitación a numérico;
    - conversión opcional de fecha;
    - eliminación de valores nulos;
    - eliminación de precipitaciones negativas;
    - consolidación opcional de fechas duplicadas usando el máximo diario.

    Parámetros
    ----------
    path_csv : str
        Ruta al archivo CSV.

    col_precip : str
        Nombre de la columna de precipitación.

    col_fecha : str o None
        Nombre de la columna de fecha. Si es None, no se procesa fecha.

    consolidar_duplicados : bool
        Si True y existe columna de fecha, agrupa duplicados por fecha
        utilizando la precipitación máxima del día.

    Retorna
    -------
    df_clean : pd.DataFrame
        Datos válidos para el análisis.

    calidad : dict
        Resumen del control de calidad aplicado.
    """
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo: {path_csv}"
        )

    df = pd.read_csv(path_csv)

    if col_precip not in df.columns:
        raise ValueError(
            f"El archivo {path_csv} no contiene la columna "
            f"de precipitación '{col_precip}'."
        )

    if col_fecha is not None and col_fecha not in df.columns:
        raise ValueError(
            f"El archivo {path_csv} no contiene la columna "
            f"de fecha '{col_fecha}'."
        )

    total_registros_originales = len(df)

    df[col_precip] = pd.to_numeric(
        df[col_precip],
        errors="coerce",
    )

    registros_precipitacion_nulos = int(
        df[col_precip].isna().sum()
    )

    registros_negativos = int(
        (df[col_precip] < 0).sum()
    )

    if col_fecha is not None:
        df[col_fecha] = pd.to_datetime(
            df[col_fecha],
            errors="coerce",
        )

        registros_fecha_nulos = int(
            df[col_fecha].isna().sum()
        )

        df_clean = df.dropna(
            subset=[col_fecha, col_precip]
        ).copy()
    else:
        registros_fecha_nulos = 0

        df_clean = df.dropna(
            subset=[col_precip]
        ).copy()

    df_clean = df_clean[
        df_clean[col_precip] >= 0
    ].copy()

    duplicados_fecha = 0

    if col_fecha is not None and consolidar_duplicados:
        duplicados_fecha = int(
            df_clean.duplicated(
                subset=[col_fecha],
                keep=False,
            ).sum()
        )

        df_clean = (
            df_clean
            .sort_values(col_fecha)
            .groupby(
                col_fecha,
                as_index=False,
            )[col_precip]
            .max()
        )

    if col_fecha is not None and not df_clean.empty:
        fecha_inicio = df_clean[col_fecha].min()
        fecha_fin = df_clean[col_fecha].max()
        n_anios = int(
            df_clean[col_fecha].dt.year.nunique()
        )
    else:
        fecha_inicio = pd.NaT
        fecha_fin = pd.NaT
        n_anios = np.nan

    calidad = {
        "total_registros_originales": (
            total_registros_originales
        ),
        "registros_precipitacion_nulos": (
            registros_precipitacion_nulos
        ),
        "registros_fecha_nulos": registros_fecha_nulos,
        "registros_negativos": registros_negativos,
        "duplicados_fecha_detectados": duplicados_fecha,
        "total_dias_validos": len(df_clean),
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "n_anios": n_anios,
    }

    return df_clean, calidad


# ============================================================
# EXCEDENCIA PARA UNA ESTACIÓN CSV
# ============================================================

def calcular_excedencia_estacion(
    path_csv,
    col_precip="pp",
    threshold=50.0,
    col_fecha="date",
    consolidar_duplicados=True,
):
    """
    Calcula la probabilidad empírica de excedencia para una estación CSV.

    La excedencia se define como:

        días con precipitación >= umbral
        --------------------------------
                total de días válidos

    Parámetros
    ----------
    path_csv : str
        Ruta al archivo CSV de la estación.

    col_precip : str
        Nombre de la columna de precipitación.

    threshold : float
        Umbral de precipitación en milímetros.

    col_fecha : str o None
        Nombre de la columna de fecha.

    consolidar_duplicados : bool
        Si True, consolida fechas duplicadas utilizando el máximo diario.

    Retorna
    -------
    dict
        Métricas de excedencia y calidad de datos.
    """
    threshold = _validar_threshold(threshold)

    df_validos, calidad = cargar_y_limpiar_precipitacion_csv(
        path_csv=path_csv,
        col_precip=col_precip,
        col_fecha=col_fecha,
        consolidar_duplicados=consolidar_duplicados,
    )

    total_dias = len(df_validos)

    dias_excedencia = int(
        (
            df_validos[col_precip] >= threshold
        ).sum()
    )

    if total_dias > 0:
        prob_excedencia = (
            dias_excedencia / total_dias
        )
    else:
        prob_excedencia = np.nan

    station_name = os.path.splitext(
        os.path.basename(path_csv)
    )[0]

    nombre_columna = _nombre_columna_excedencia(
        threshold
    )

    resultado = {
        "station": station_name,
        "station_file": station_name,
        "threshold_mm": threshold,
        "total_dias": total_dias,
        "dias_excedencia": dias_excedencia,
        "prob_excedencia": prob_excedencia,
        nombre_columna: prob_excedencia,
    }

    resultado.update(calidad)

    return resultado


# ============================================================
# PROCESAMIENTO BATCH DE CSV
# ============================================================

def procesar_excedencia_batch_csv(
    carpeta_estaciones,
    patron="dat*.csv",
    col_precip="pp",
    threshold=50.0,
    col_fecha="date",
    consolidar_duplicados=True,
    exportar=True,
    nombre_salida_dir="_salidas_excedencias",
):
    """
    Procesa una carpeta completa de archivos CSV de estaciones.

    Parámetros
    ----------
    carpeta_estaciones : str
        Carpeta que contiene los archivos CSV.

    patron : str
        Patrón de búsqueda, por ejemplo 'dat*.csv' o '*.csv'.

    col_precip : str
        Nombre de la columna de precipitación.

    threshold : float
        Umbral de excedencia en milímetros.

    col_fecha : str o None
        Nombre de la columna de fecha.

    consolidar_duplicados : bool
        Si True, consolida registros duplicados por fecha.

    exportar : bool
        Si True, guarda tabla maestra y log en archivos CSV.

    nombre_salida_dir : str
        Nombre de la carpeta de resultados.

    Retorna
    -------
    df_resultados : pd.DataFrame
        Tabla maestra de excedencias.

    log_df : pd.DataFrame
        Log del procesamiento.

    out_resultados : str o None
        Ruta del CSV maestro generado.

    out_log : str o None
        Ruta del log generado.
    """
    threshold = _validar_threshold(threshold)

    if not os.path.isdir(carpeta_estaciones):
        raise NotADirectoryError(
            "La ruta indicada no corresponde a una carpeta válida: "
            f"{carpeta_estaciones}"
        )

    archivos = sorted(
        glob.glob(
            os.path.join(
                carpeta_estaciones,
                patron,
            )
        )
    )

    resultados = []
    log = []

    for path_csv in archivos:
        station_name = os.path.splitext(
            os.path.basename(path_csv)
        )[0]

        try:
            resultado = calcular_excedencia_estacion(
                path_csv=path_csv,
                col_precip=col_precip,
                threshold=threshold,
                col_fecha=col_fecha,
                consolidar_duplicados=(
                    consolidar_duplicados
                ),
            )

            resultados.append(resultado)

            log.append({
                "station": station_name,
                "archivo": path_csv,
                "status": "ok",
                "message": "",
                "total_dias": resultado[
                    "total_dias"
                ],
                "dias_excedencia": resultado[
                    "dias_excedencia"
                ],
                "prob_excedencia": resultado[
                    "prob_excedencia"
                ],
            })

        except Exception as error:
            log.append({
                "station": station_name,
                "archivo": path_csv,
                "status": "error",
                "message": str(error),
                "total_dias": np.nan,
                "dias_excedencia": np.nan,
                "prob_excedencia": np.nan,
            })

    df_resultados = pd.DataFrame(resultados)
    log_df = pd.DataFrame(log)

    out_resultados = None
    out_log = None

    if exportar:
        dir_out = os.path.join(
            carpeta_estaciones,
            nombre_salida_dir,
        )

        os.makedirs(
            dir_out,
            exist_ok=True,
        )

        fecha_tag = datetime.now().strftime(
            "%Y%m%d_%H%M"
        )

        threshold_texto = (
            str(int(threshold))
            if float(threshold).is_integer()
            else str(threshold).replace(".", "_")
        )

        out_resultados = os.path.join(
            dir_out,
            (
                "MASTER_excedencia_"
                f"{threshold_texto}mm_"
                f"{fecha_tag}.csv"
            ),
        )

        out_log = os.path.join(
            dir_out,
            (
                "log_excedencia_"
                f"{threshold_texto}mm_"
                f"{fecha_tag}.csv"
            ),
        )

        df_resultados.to_csv(
            out_resultados,
            index=False,
        )

        log_df.to_csv(
            out_log,
            index=False,
        )

    return (
        df_resultados,
        log_df,
        out_resultados,
        out_log,
    )


# ============================================================
# EXCEDENCIA PARA UNA ESTACIÓN EXCEL
# ============================================================

def calcular_excedencia_estacion_excel(
    path_excel,
    sheet_name="Datos Clima",
    col_precip="PRECIP",
    threshold=50.0,
):
    """
    Calcula la probabilidad de excedencia para una estación Excel.
    """
    threshold = _validar_threshold(threshold)

    df = pd.read_excel(
        path_excel,
        sheet_name=sheet_name,
    )

    if col_precip not in df.columns:
        raise ValueError(
            f"El archivo {path_excel} no contiene la columna "
            f"'{col_precip}'."
        )

    df[col_precip] = pd.to_numeric(
        df[col_precip],
        errors="coerce",
    )

    df_validos = df.dropna(
        subset=[col_precip]
    ).copy()

    df_validos = df_validos[
        df_validos[col_precip] >= 0
    ].copy()

    total_dias = len(df_validos)

    dias_excedencia = int(
        (
            df_validos[col_precip] >= threshold
        ).sum()
    )

    if total_dias > 0:
        prob_excedencia = (
            dias_excedencia / total_dias
        )
    else:
        prob_excedencia = np.nan

    station_name = os.path.splitext(
        os.path.basename(path_excel)
    )[0]

    nombre_columna = _nombre_columna_excedencia(
        threshold
    )

    return {
        "station": station_name,
        "station_file": station_name,
        "threshold_mm": threshold,
        "total_dias": total_dias,
        "dias_excedencia": dias_excedencia,
        "prob_excedencia": prob_excedencia,
        nombre_columna: prob_excedencia,
    }


# ============================================================
# BATCH DE EXCEL CON COORDENADAS
# ============================================================

def procesar_excedencia_batch_excel(
    carpeta_estaciones,
    archivo_coordenadas,
    threshold=50.0,
    sheet_name="Datos Clima",
    col_precip="PRECIP",
    col_clave="CLAVE",
    col_nombre="NOMBRE",
    export_csv_path=None,
):
    """
    Procesa excedencia para múltiples estaciones en Excel
    y une los resultados con un archivo de coordenadas.

    Esta función conserva la convención original de nombres:

        CLAVE_NOMBRE_ESTACION.xlsx
    """
    threshold = _validar_threshold(threshold)

    df_coords = pd.read_excel(
        archivo_coordenadas
    )

    if col_clave not in df_coords.columns:
        raise ValueError(
            f"El archivo de coordenadas no contiene "
            f"la columna '{col_clave}'."
        )

    if col_nombre not in df_coords.columns:
        raise ValueError(
            f"El archivo de coordenadas no contiene "
            f"la columna '{col_nombre}'."
        )

    resultados = []
    log = []

    for _, fila in df_coords.iterrows():
        clave = fila[col_clave]
        nombre_estacion = fila[col_nombre]

        nombre_archivo = (
            f"{clave}_"
            f"{str(nombre_estacion).replace(' ', '_').upper()}"
            ".xlsx"
        )

        ruta = os.path.join(
            carpeta_estaciones,
            nombre_archivo,
        )

        if not os.path.exists(ruta):
            log.append({
                col_clave: clave,
                "archivo": nombre_archivo,
                "status": "missing",
                "message": (
                    f"Archivo no encontrado: {ruta}"
                ),
            })
            continue

        try:
            resultado = (
                calcular_excedencia_estacion_excel(
                    path_excel=ruta,
                    sheet_name=sheet_name,
                    col_precip=col_precip,
                    threshold=threshold,
                )
            )

            resultados.append({
                col_clave: clave,
                "threshold_mm": threshold,
                "prob_excedencia": resultado[
                    "prob_excedencia"
                ],
                "total_dias": resultado[
                    "total_dias"
                ],
                "dias_excedencia": resultado[
                    "dias_excedencia"
                ],
            })

            log.append({
                col_clave: clave,
                "archivo": nombre_archivo,
                "status": "ok",
                "message": "",
            })

        except Exception as error:
            log.append({
                col_clave: clave,
                "archivo": nombre_archivo,
                "status": "error",
                "message": str(error),
            })

    df_resultados = pd.DataFrame(
        resultados
    )

    if df_resultados.empty:
        df_final = df_coords.copy()
        df_final["threshold_mm"] = threshold
        df_final["prob_excedencia"] = np.nan
        df_final["total_dias"] = np.nan
        df_final["dias_excedencia"] = np.nan
    else:
        df_final = pd.merge(
            df_coords,
            df_resultados,
            on=col_clave,
            how="left",
        )

    log_df = pd.DataFrame(log)

    if export_csv_path is not None:
        df_final.to_csv(
            export_csv_path,
            index=False,
        )

    return df_final, log_df


# ============================================================
# UNIÓN CON COORDENADAS
# ============================================================

def unir_excedencia_coordenadas(
    df_coords,
    df_excedencia,
    col_clave="CLAVE",
):
    """
    Une una tabla de coordenadas con una tabla de excedencia.
    """
    if col_clave not in df_coords.columns:
        raise ValueError(
            f"df_coords no contiene la columna "
            f"'{col_clave}'."
        )

    if col_clave not in df_excedencia.columns:
        raise ValueError(
            f"df_excedencia no contiene la columna "
            f"'{col_clave}'."
        )

    return pd.merge(
        df_coords,
        df_excedencia,
        on=col_clave,
        how="left",
    )


# ============================================================
# PREPARACIÓN PARA INTERPOLACIÓN
# ============================================================

def preparar_excedencia_para_interpolacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="prob_excedencia",
    convertir_a_porcentaje=False,
):
    """
    Prepara datos de excedencia y coordenadas para interpolación espacial.
    """
    columnas_requeridas = [
        col_lon,
        col_lat,
        col_val,
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}"
        )

    df_clean = df.dropna(
        subset=columnas_requeridas
    ).copy()

    df_clean[col_lon] = pd.to_numeric(
        df_clean[col_lon],
        errors="coerce",
    )

    df_clean[col_lat] = pd.to_numeric(
        df_clean[col_lat],
        errors="coerce",
    )

    df_clean[col_val] = pd.to_numeric(
        df_clean[col_val],
        errors="coerce",
    )

    df_clean = df_clean.dropna(
        subset=columnas_requeridas
    ).copy()

    if convertir_a_porcentaje:
        df_clean[col_val] = (
            df_clean[col_val] * 100.0
        )

    points = df_clean[
        [col_lon, col_lat]
    ].to_numpy()

    values = df_clean[
        col_val
    ].to_numpy()

    return df_clean, points, values