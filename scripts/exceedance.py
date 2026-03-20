import os
import pandas as pd
import numpy as np


def calcular_excedencia_estacion(
    path_csv,
    col_precip="pp",
    threshold=50.0,
):
    """
    Calcula la probabilidad de excedencia para una estación a partir de un CSV.

    La excedencia se define como la proporción de días con precipitación
    mayor o igual al umbral especificado.

    Parámetros
    ----------
    path_csv : str
        Ruta al archivo CSV de la estación.
    col_precip : str
        Nombre de la columna de precipitación.
    threshold : float
        Umbral de precipitación para definir excedencia.

    Retorna
    -------
    dict
        Diccionario con métricas de la estación.
    """
    df = pd.read_csv(path_csv)

    if col_precip not in df.columns:
        raise ValueError(f"El archivo {path_csv} no contiene la columna '{col_precip}'.")

    df[col_precip] = pd.to_numeric(df[col_precip], errors="coerce")
    df_validos = df.dropna(subset=[col_precip])

    total_dias = len(df_validos)
    dias_excedencia = int((df_validos[col_precip] >= threshold).sum())

    excedencia = np.nan
    if total_dias > 0:
        excedencia = dias_excedencia / total_dias

    station_name = os.path.splitext(os.path.basename(path_csv))[0]

    return {
        "station_file": station_name,
        "total_dias": total_dias,
        "dias_excedencia": dias_excedencia,
        "EXCEDENCIA_50MM": excedencia,
    }


def calcular_excedencia_estacion_excel(
    path_excel,
    sheet_name="Datos Clima",
    col_precip="PRECIP",
    threshold=50.0,
):
    """
    Calcula la probabilidad de excedencia para una estación a partir de un Excel.

    Parámetros
    ----------
    path_excel : str
        Ruta al archivo Excel de la estación.
    sheet_name : str
        Nombre de la hoja donde están los datos climáticos.
    col_precip : str
        Nombre de la columna de precipitación.
    threshold : float
        Umbral de precipitación.

    Retorna
    -------
    dict
        Diccionario con métricas de excedencia.
    """
    df = pd.read_excel(path_excel, sheet_name=sheet_name)

    if col_precip not in df.columns:
        raise ValueError(f"El archivo {path_excel} no contiene la columna '{col_precip}'.")

    df[col_precip] = pd.to_numeric(df[col_precip], errors="coerce")
    total_dias = int(df[col_precip].count())
    dias_excedencia = int((df[col_precip] >= threshold).sum())

    excedencia = np.nan
    if total_dias > 0:
        excedencia = dias_excedencia / total_dias

    station_name = os.path.splitext(os.path.basename(path_excel))[0]

    return {
        "station_file": station_name,
        "total_dias": total_dias,
        "dias_excedencia": dias_excedencia,
        "EXCEDENCIA_50MM": excedencia,
    }


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
    Procesa excedencia para múltiples estaciones en archivos Excel y une
    los resultados con el archivo de coordenadas.

    Parámetros
    ----------
    carpeta_estaciones : str
        Carpeta donde están los archivos Excel de las estaciones.
    archivo_coordenadas : str
        Ruta al archivo Excel con coordenadas de estaciones.
    threshold : float
        Umbral de precipitación.
    sheet_name : str
        Hoja del Excel con datos climáticos.
    col_precip : str
        Nombre de columna de precipitación.
    col_clave : str
        Nombre de la columna clave de estación.
    col_nombre : str
        Nombre de la columna nombre de estación.
    export_csv_path : str o None
        Si se proporciona, exporta el resultado final a CSV.

    Retorna
    -------
    df_final : pd.DataFrame
        Tabla final unida con coordenadas y excedencia.
    log_df : pd.DataFrame
        Log del procesamiento, incluyendo errores.
    """
    df_coords = pd.read_excel(archivo_coordenadas)

    if col_clave not in df_coords.columns:
        raise ValueError(f"El archivo de coordenadas no contiene la columna '{col_clave}'.")
    if col_nombre not in df_coords.columns:
        raise ValueError(f"El archivo de coordenadas no contiene la columna '{col_nombre}'.")

    resultados = []
    log = []

    for clave in df_coords[col_clave]:
        nombre_estacion = df_coords.loc[df_coords[col_clave] == clave, col_nombre].values[0]
        nombre_archivo = f"{clave}_{str(nombre_estacion).replace(' ', '_').upper()}.xlsx"
        ruta = os.path.join(carpeta_estaciones, nombre_archivo)

        if not os.path.exists(ruta):
            log.append({
                "CLAVE": clave,
                "archivo": nombre_archivo,
                "status": "missing",
                "message": f"Archivo no encontrado: {ruta}",
            })
            continue

        try:
            res = calcular_excedencia_estacion_excel(
                path_excel=ruta,
                sheet_name=sheet_name,
                col_precip=col_precip,
                threshold=threshold,
            )

            resultados.append({
                "CLAVE": clave,
                "EXCEDENCIA_50MM": res["EXCEDENCIA_50MM"],
                "total_dias": res["total_dias"],
                "dias_excedencia": res["dias_excedencia"],
            })

            log.append({
                "CLAVE": clave,
                "archivo": nombre_archivo,
                "status": "ok",
                "message": "",
            })

        except Exception as e:
            log.append({
                "CLAVE": clave,
                "archivo": nombre_archivo,
                "status": "error",
                "message": str(e),
            })

    df_resultados = pd.DataFrame(resultados)
    df_final = pd.merge(df_coords, df_resultados, on="CLAVE", how="left")
    log_df = pd.DataFrame(log)

    if export_csv_path is not None:
        df_final.to_csv(export_csv_path, index=False)

    return df_final, log_df


def unir_excedencia_coordenadas(
    df_coords,
    df_excedencia,
    col_clave="CLAVE",
):
    """
    Une una tabla de coordenadas con una tabla de excedencia por estación.

    Parámetros
    ----------
    df_coords : pd.DataFrame
        Tabla de coordenadas.
    df_excedencia : pd.DataFrame
        Tabla de excedencia.
    col_clave : str
        Columna llave.

    Retorna
    -------
    pd.DataFrame
        DataFrame unido.
    """
    if col_clave not in df_coords.columns:
        raise ValueError(f"df_coords no contiene la columna '{col_clave}'.")
    if col_clave not in df_excedencia.columns:
        raise ValueError(f"df_excedencia no contiene la columna '{col_clave}'.")

    return pd.merge(df_coords, df_excedencia, on=col_clave, how="left")


def preparar_excedencia_para_interpolacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="EXCEDENCIA_50MM",
    convertir_a_porcentaje=False,
):
    """
    Prepara un DataFrame para interpolación espacial.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas y valores.
    col_lon : str
        Columna de longitud.
    col_lat : str
        Columna de latitud.
    col_val : str
        Columna del valor a interpolar.
    convertir_a_porcentaje : bool
        Si True, multiplica el valor por 100.

    Retorna
    -------
    df_clean : pd.DataFrame
        DataFrame limpio.
    points : np.ndarray
        Coordenadas para interpolación.
    values : np.ndarray
        Valores a interpolar.
    """
    cols_req = [col_lon, col_lat, col_val]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    df_clean = df.dropna(subset=cols_req).copy()

    df_clean[col_lon] = pd.to_numeric(df_clean[col_lon], errors="coerce")
    df_clean[col_lat] = pd.to_numeric(df_clean[col_lat], errors="coerce")
    df_clean[col_val] = pd.to_numeric(df_clean[col_val], errors="coerce")

    df_clean = df_clean.dropna(subset=cols_req).copy()

    if convertir_a_porcentaje:
        df_clean[col_val] = df_clean[col_val] * 100.0

    points = df_clean[[col_lon, col_lat]].values
    values = df_clean[col_val].values

    return df_clean, points, values
