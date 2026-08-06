import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


# ============================================================
# LIMPIEZA Y VALIDACIÓN DE DATOS ESPACIALES
# ============================================================

def preparar_datos_espaciales(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="prob_excedencia",
    crs="EPSG:4326",
    eliminar_duplicados=True,
):
    """
    Prepara y valida coordenadas y valores para interpolación.

    Las estaciones con coordenadas duplicadas no se combinan ni
    se promedian. Se conserva únicamente la primera aparición.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla con coordenadas y variable a interpolar.

    col_lon : str
        Nombre de la columna de longitud o coordenada X.

    col_lat : str
        Nombre de la columna de latitud o coordenada Y.

    col_val : str
        Nombre de la variable numérica a interpolar.

    crs : str
        Sistema de referencia espacial.

    eliminar_duplicados : bool
        Si True, conserva únicamente el primer registro cuando
        existen coordenadas repetidas.

    Retorna
    -------
    data : pd.DataFrame
        Tabla limpia y validada.

    calidad : dict
        Resumen del proceso de validación.
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

    data = df.copy()

    total_original = len(data)

    for columna in columnas_requeridas:
        data[columna] = pd.to_numeric(
            data[columna],
            errors="coerce",
        )

    registros_no_numericos = int(
        data[columnas_requeridas]
        .isna()
        .any(axis=1)
        .sum()
    )

    data = data.dropna(
        subset=columnas_requeridas
    ).copy()

    registros_fuera_rango = 0

    # Validación geográfica solamente para coordenadas
    # en longitud/latitud.
    if str(crs).upper() == "EPSG:4326":
        mascara_valida = (
            data[col_lon].between(-180, 180)
            & data[col_lat].between(-90, 90)
        )

        registros_fuera_rango = int(
            (~mascara_valida).sum()
        )

        data = data[
            mascara_valida
        ].copy()

    duplicados_detectados = int(
        data.duplicated(
            subset=[col_lon, col_lat],
            keep=False,
        ).sum()
    )

    duplicados_eliminados = 0

    if eliminar_duplicados:
        total_antes = len(data)

        # Se conserva la primera estación.
        # No se calcula media, suma, máximo ni mediana.
        data = data.drop_duplicates(
            subset=[col_lon, col_lat],
            keep="first",
        ).copy()

        duplicados_eliminados = (
            total_antes - len(data)
        )

    if data.empty:
        raise ValueError(
            "No quedaron estaciones válidas después de limpiar "
            "las coordenadas y la variable seleccionada."
        )

    calidad = {
        "total_registros_originales": total_original,
        "registros_no_numericos": registros_no_numericos,
        "registros_fuera_rango": registros_fuera_rango,
        "duplicados_detectados": duplicados_detectados,
        "duplicados_eliminados": duplicados_eliminados,
        "total_estaciones_validas": len(data),
    }

    return data, calidad


# ============================================================
# CREACIÓN DE MALLA
# ============================================================

def crear_malla_exportacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    margin=0.25,
    nx=400,
    ny=400,
):
    """
    Crea una malla regular para exportar un raster.

    El margen se expresa en las unidades del CRS utilizado.
    Para EPSG:4326, el margen corresponde a grados.
    """
    if nx < 2 or ny < 2:
        raise ValueError(
            "nx y ny deben ser mayores o iguales a 2."
        )

    if margin < 0:
        raise ValueError(
            "El margen espacial no puede ser negativo."
        )

    if col_lon not in df.columns or col_lat not in df.columns:
        raise ValueError(
            "No se encontraron las columnas de coordenadas "
            "solicitadas."
        )

    lon_min = float(
        df[col_lon].min()
    ) - float(margin)

    lon_max = float(
        df[col_lon].max()
    ) + float(margin)

    lat_min = float(
        df[col_lat].min()
    ) - float(margin)

    lat_max = float(
        df[col_lat].max()
    ) + float(margin)

    if np.isclose(lon_min, lon_max):
        raise ValueError(
            "Todas las estaciones tienen la misma coordenada X."
        )

    if np.isclose(lat_min, lat_max):
        raise ValueError(
            "Todas las estaciones tienen la misma coordenada Y."
        )

    gx = np.linspace(
        lon_min,
        lon_max,
        int(nx),
    )

    gy = np.linspace(
        lat_min,
        lat_max,
        int(ny),
    )

    grid_x, grid_y = np.meshgrid(
        gx,
        gy,
        indexing="xy",
    )

    extent = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "nx": int(nx),
        "ny": int(ny),
        "margin": float(margin),
    }

    return (
        gx,
        gy,
        grid_x,
        grid_y,
        extent,
    )


# ============================================================
# INTERPOLACIÓN
# ============================================================

def interpolar_para_raster(
    points,
    values,
    grid_x,
    grid_y,
    method="linear",
    fill_nearest=True,
):
    """
    Interpola una superficie sobre la malla de exportación.

    Métodos permitidos:
    - linear
    - cubic
    - nearest
    """
    metodos_validos = {
        "linear",
        "cubic",
        "nearest",
    }

    if method not in metodos_validos:
        raise ValueError(
            f"Método no válido: {method}. "
            f"Use uno de: {sorted(metodos_validos)}"
        )

    points = np.asarray(
        points,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "points debe tener dimensiones (n, 2)."
        )

    if values.ndim != 1:
        raise ValueError(
            "values debe ser un vector unidimensional."
        )

    if len(points) != len(values):
        raise ValueError(
            "El número de coordenadas no coincide con "
            "el número de valores."
        )

    if len(points) == 0:
        raise ValueError(
            "No existen puntos para interpolar."
        )

    if method in {"linear", "cubic"} and len(points) < 3:
        raise ValueError(
            "La interpolación lineal o cúbica requiere "
            "al menos tres estaciones válidas."
        )

    try:
        arr = griddata(
            points,
            values,
            (grid_x, grid_y),
            method=method,
        )

    except Exception as error:
        raise ValueError(
            "No fue posible realizar la interpolación. "
            "Compruebe que las estaciones no sean colineales "
            "y que exista una distribución espacial suficiente. "
            f"Detalle: {error}"
        ) from error

    if arr is None:
        raise ValueError(
            "La interpolación no produjo una superficie."
        )

    if fill_nearest and method != "nearest":
        arr_near = griddata(
            points,
            values,
            (grid_x, grid_y),
            method="nearest",
        )

        arr = np.where(
            np.isnan(arr),
            arr_near,
            arr,
        )

    if not np.any(
        np.isfinite(arr)
    ):
        raise ValueError(
            "La superficie interpolada no contiene valores finitos."
        )

    return arr


# ============================================================
# TRANSFORMACIÓN ESPACIAL
# ============================================================

def construir_transform_desde_centros(
    gx,
    gy,
):
    """
    Construye la transformación Affine de rasterio a partir
    de coordenadas que representan centros de celda.
    """
    gx = np.asarray(
        gx,
        dtype=float,
    )

    gy = np.asarray(
        gy,
        dtype=float,
    )

    if len(gx) < 2 or len(gy) < 2:
        raise ValueError(
            "gx y gy deben tener al menos dos elementos."
        )

    dx = float(
        np.abs(gx[1] - gx[0])
    )

    dy = float(
        np.abs(gy[1] - gy[0])
    )

    if dx <= 0 or dy <= 0:
        raise ValueError(
            "El tamaño de celda debe ser mayor que cero."
        )

    west_corner = (
        float(gx.min()) - dx / 2.0
    )

    north_corner = (
        float(gy.max()) + dy / 2.0
    )

    transform = from_origin(
        west_corner,
        north_corner,
        dx,
        dy,
    )

    return transform, dx, dy


# ============================================================
# PREPARACIÓN DEL ARRAY
# ============================================================

def preparar_array_raster(
    arr,
    nodata=-9999.0,
    dtype="float32",
):
    """
    Convierte la superficie en un array compatible con rasterio.
    """
    arr = np.asarray(
        arr,
        dtype=dtype,
    )

    arr_out = np.where(
        np.isfinite(arr),
        arr,
        nodata,
    )

    arr_out = np.ascontiguousarray(
        arr_out
    )

    return arr_out


# ============================================================
# EXPORTACIÓN GEOTIFF
# ============================================================

def exportar_geotiff_interpolado(
    out_tif,
    arr,
    gx,
    gy,
    crs="EPSG:4326",
    nodata=-9999.0,
    dtype="float32",
    overwrite=True,
):
    """
    Exporta una superficie interpolada a GeoTIFF north-up.

    La malla se genera de sur a norte, mientras que un raster
    north-up debe iniciar en la fila más septentrional. Por eso
    el array se invierte verticalmente antes de escribirlo.
    """
    if not out_tif:
        raise ValueError(
            "Debe indicar una ruta de salida para el GeoTIFF."
        )

    out_tif = os.path.abspath(
        out_tif
    )

    out_dir = os.path.dirname(
        out_tif
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    transform, dx, dy = (
        construir_transform_desde_centros(
            gx,
            gy,
        )
    )

    arr = np.asarray(
        arr
    )

    if arr.ndim != 2:
        raise ValueError(
            "La superficie del raster debe ser bidimensional."
        )

    if arr.shape != (
        len(gy),
        len(gx),
    ):
        raise ValueError(
            "Las dimensiones del array no coinciden con gx y gy. "
            f"Array: {arr.shape}; esperado: "
            f"({len(gy)}, {len(gx)})."
        )

    # Corrección de orientación:
    # primera fila del GeoTIFF = norte.
    arr_north_up = np.flipud(
        arr
    )

    arr_out = preparar_array_raster(
        arr_north_up,
        nodata=nodata,
        dtype=dtype,
    )

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata,
        "width": int(
            arr_out.shape[1]
        ),
        "height": int(
            arr_out.shape[0]
        ),
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    if overwrite:
        for extra in (
            "",
            ".aux.xml",
        ):
            try:
                os.remove(
                    out_tif + extra
                )
            except FileNotFoundError:
                pass

    elif os.path.exists(out_tif):
        raise FileExistsError(
            f"El archivo ya existe: {out_tif}"
        )

    with rasterio.open(
        out_tif,
        "w",
        **profile,
    ) as dst:
        dst.write(
            arr_out,
            1,
        )

        dst.update_tags(
            interpolation="spatial_interpolation",
            orientation="north_up",
            cell_size_x=dx,
            cell_size_y=dy,
        )

    return out_tif, profile


# ============================================================
# FLUJO COMPLETO
# ============================================================

def exportar_desde_puntos_a_geotiff(
    df,
    out_tif,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="prob_excedencia",
    margin=0.25,
    nx=400,
    ny=400,
    method="linear",
    fill_nearest=True,
    nodata=-9999.0,
    crs="EPSG:4326",
    dtype="float32",
    eliminar_duplicados=True,
):
    """
    Flujo completo de exportación:

    1. valida y limpia las estaciones;
    2. conserva una estación cuando hay coordenadas duplicadas;
    3. crea la malla;
    4. interpola la superficie;
    5. exporta el GeoTIFF north-up.
    """
    data, calidad = preparar_datos_espaciales(
        df=df,
        col_lon=col_lon,
        col_lat=col_lat,
        col_val=col_val,
        crs=crs,
        eliminar_duplicados=eliminar_duplicados,
    )

    if (
        method in {"linear", "cubic"}
        and len(data) < 3
    ):
        raise ValueError(
            "No existen suficientes estaciones válidas para "
            f"la interpolación {method}."
        )

    points = data[
        [col_lon, col_lat]
    ].to_numpy(
        dtype=float
    )

    values = data[
        col_val
    ].to_numpy(
        dtype=float
    )

    (
        gx,
        gy,
        grid_x,
        grid_y,
        extent,
    ) = crear_malla_exportacion(
        data,
        col_lon=col_lon,
        col_lat=col_lat,
        margin=margin,
        nx=nx,
        ny=ny,
    )

    arr = interpolar_para_raster(
        points=points,
        values=values,
        grid_x=grid_x,
        grid_y=grid_y,
        method=method,
        fill_nearest=fill_nearest,
    )

    out_tif, profile = (
        exportar_geotiff_interpolado(
            out_tif=out_tif,
            arr=arr,
            gx=gx,
            gy=gy,
            crs=crs,
            nodata=nodata,
            dtype=dtype,
            overwrite=True,
        )
    )

    return {
        "out_tif": out_tif,
        "profile": profile,
        "gx": gx,
        "gy": gy,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "arr": arr,
        "extent": extent,
        "data": data,
        "calidad": calidad,
    }