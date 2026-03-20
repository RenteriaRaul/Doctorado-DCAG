import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


def crear_malla_exportacion(
    df,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    margin=0.25,
    nx=400,
    ny=400,
):
    """
    Crea una malla de exportación regular para raster.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas.
    col_lon : str
        Columna de longitud.
    col_lat : str
        Columna de latitud.
    margin : float
        Margen espacial adicional.
    nx : int
        Número de columnas del raster.
    ny : int
        Número de filas del raster.

    Retorna
    -------
    gx, gy : np.ndarray
        Vectores 1D de coordenadas.
    grid_x, grid_y : np.ndarray
        Malla 2D.
    extent : dict
        Extensión usada.
    """
    lon_min = float(df[col_lon].min()) - margin
    lon_max = float(df[col_lon].max()) + margin
    lat_min = float(df[col_lat].min()) - margin
    lat_max = float(df[col_lat].max()) + margin

    gx = np.linspace(lon_min, lon_max, nx)
    gy = np.linspace(lat_min, lat_max, ny)

    grid_x, grid_y = np.meshgrid(gx, gy, indexing="xy")

    extent = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "nx": nx,
        "ny": ny,
    }

    return gx, gy, grid_x, grid_y, extent


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

    Parámetros
    ----------
    points : np.ndarray
        Coordenadas de estaciones.
    values : np.ndarray
        Valores observados.
    grid_x, grid_y : np.ndarray
        Malla de exportación.
    method : str
        Método principal de interpolación.
    fill_nearest : bool
        Si True, rellena NaN con nearest.

    Retorna
    -------
    arr : np.ndarray
        Superficie interpolada 2D.
    """
    arr = griddata(points, values, (grid_x, grid_y), method=method)

    if fill_nearest:
        arr_near = griddata(points, values, (grid_x, grid_y), method="nearest")
        arr = np.where(np.isnan(arr), arr_near, arr)

    return arr


def construir_transform_desde_centros(gx, gy):
    """
    Construye el affine transform de rasterio a partir de centros de celda.

    Parámetros
    ----------
    gx, gy : np.ndarray
        Coordenadas 1D de centros de celda.

    Retorna
    -------
    transform : affine.Affine
        Transformación espacial para rasterio.
    dx, dy : float
        Tamaño de celda en X y Y.
    """
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError("gx y gy deben tener al menos 2 elementos.")

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])

    west_corner = float(gx.min()) - dx / 2.0
    north_corner = float(gy.max()) + dy / 2.0

    transform = from_origin(west_corner, north_corner, dx, dy)
    return transform, dx, dy


def preparar_array_raster(arr, nodata=-9999.0, dtype="float32"):
    """
    Prepara una matriz para escritura raster.

    Parámetros
    ----------
    arr : np.ndarray
        Superficie interpolada.
    nodata : float
        Valor nodata.
    dtype : str
        Tipo de dato final.

    Retorna
    -------
    arr_out : np.ndarray
        Array listo para escritura.
    """
    arr = np.asarray(arr, dtype=dtype)
    arr_out = np.where(np.isnan(arr), nodata, arr)
    arr_out = np.ascontiguousarray(arr_out)
    return arr_out


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
    Exporta una superficie interpolada a GeoTIFF.

    Parámetros
    ----------
    out_tif : str
        Ruta de salida del GeoTIFF.
    arr : np.ndarray
        Superficie 2D.
    gx, gy : np.ndarray
        Centros de celda en X e Y.
    crs : str
        Sistema de referencia.
    nodata : float
        Valor nodata.
    dtype : str
        Tipo de dato del raster.
    overwrite : bool
        Si True, elimina archivos previos con el mismo nombre.

    Retorna
    -------
    out_tif : str
        Ruta final escrita.
    profile : dict
        Perfil raster usado.
    """
    transform, _, _ = construir_transform_desde_centros(gx, gy)
    arr_out = preparar_array_raster(arr, nodata=nodata, dtype=dtype)

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata,
        "width": int(arr_out.shape[1]),
        "height": int(arr_out.shape[0]),
        "count": 1,
        "crs": crs,
        "transform": transform,
    }

    if overwrite:
        for extra in ("", ".aux.xml"):
            try:
                os.remove(out_tif + extra)
            except FileNotFoundError:
                pass

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr_out, 1)

    return out_tif, profile


def exportar_desde_puntos_a_geotiff(
    df,
    out_tif,
    col_lon="LONGITUD",
    col_lat="LATITUD",
    col_val="EXCEDENCIA_50MM",
    margin=0.25,
    nx=400,
    ny=400,
    method="linear",
    fill_nearest=True,
    nodata=-9999.0,
    crs="EPSG:4326",
    dtype="float32",
):
    """
    Flujo completo:
    - crea malla de exportación
    - interpola desde puntos
    - exporta a GeoTIFF

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas espaciales y valor.
    out_tif : str
        Archivo GeoTIFF de salida.
    col_lon, col_lat, col_val : str
        Columnas de longitud, latitud y valor.
    margin : float
        Margen espacial.
    nx, ny : int
        Tamaño de la malla.
    method : str
        Método principal de interpolación.
    fill_nearest : bool
        Si True, rellena NaN con nearest.
    nodata : float
        Valor nodata.
    crs : str
        Sistema de referencia.
    dtype : str
        Tipo de dato de salida.

    Retorna
    -------
    resultados : dict
        Diccionario con rutas y objetos clave.
    """
    cols_req = [col_lon, col_lat, col_val]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    data = df.dropna(subset=cols_req).copy()
    data[col_lon] = data[col_lon].astype(float)
    data[col_lat] = data[col_lat].astype(float)
    data[col_val] = data[col_val].astype(float)

    points = data[[col_lon, col_lat]].values
    values = data[col_val].values

    gx, gy, grid_x, grid_y, extent = crear_malla_exportacion(
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

    out_tif, profile = exportar_geotiff_interpolado(
        out_tif=out_tif,
        arr=arr,
        gx=gx,
        gy=gy,
        crs=crs,
        nodata=nodata,
        dtype=dtype,
        overwrite=True,
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
    }
