import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask


# ============================================================
# LECTURA DEL LÍMITE TERRITORIAL
# ============================================================

def cargar_limite_territorial(
    path_boundary,
    crs_destino=None,
    layer=None,
):
    """
    Lee y valida un límite territorial.

    Admite cualquier formato vectorial compatible con GeoPandas,
    por ejemplo:

    - KML
    - GeoJSON
    - Shapefile
    - GeoPackage

    Parámetros
    ----------
    path_boundary : str o Path
        Ruta del archivo territorial.

    crs_destino : str o CRS o None
        CRS al que se reproyectará el límite. Si es None,
        se conserva el CRS original.

    layer : str o None
        Nombre de capa cuando el archivo contiene varias capas.

    Retorna
    -------
    limite : geopandas.GeoDataFrame
        Límite territorial limpio y validado.

    metadata : dict
        Información del archivo territorial.
    """
    path_boundary = Path(path_boundary)

    if not path_boundary.exists():
        raise FileNotFoundError(
            f"No se encontró el límite territorial: {path_boundary}"
        )

    try:
        if layer is None:
            limite = gpd.read_file(
                path_boundary
            )
        else:
            limite = gpd.read_file(
                path_boundary,
                layer=layer,
            )

    except Exception as error:
        raise ValueError(
            "No fue posible leer el límite territorial. "
            "Compruebe que el formato sea compatible y que todos "
            f"los archivos asociados estén disponibles. Detalle: {error}"
        ) from error

    if limite.empty:
        raise ValueError(
            "El archivo territorial no contiene geometrías."
        )

    total_original = len(limite)

    # Eliminar geometrías vacías o nulas
    limite = limite[
        limite.geometry.notna()
    ].copy()

    limite = limite[
        ~limite.geometry.is_empty
    ].copy()

    if limite.empty:
        raise ValueError(
            "El archivo territorial no contiene geometrías válidas."
        )

    # Intentar corregir geometrías inválidas
    geometries_invalidas = int(
        (~limite.geometry.is_valid).sum()
    )

    if geometries_invalidas > 0:
        limite["geometry"] = (
            limite.geometry.buffer(0)
        )

        limite = limite[
            limite.geometry.notna()
            & ~limite.geometry.is_empty
            & limite.geometry.is_valid
        ].copy()

    if limite.empty:
        raise ValueError(
            "No fue posible obtener una geometría territorial válida."
        )

    if limite.crs is None:
        raise ValueError(
            "El archivo territorial no tiene un sistema de "
            "referencia espacial definido."
        )

    crs_original = str(
        limite.crs
    )

    if crs_destino is not None:
        limite = limite.to_crs(
            crs_destino
        )

    # Unificar todas las geometrías en una sola entidad
    limite_disuelto = limite.dissolve()

    limite_disuelto = limite_disuelto[
        limite_disuelto.geometry.notna()
        & ~limite_disuelto.geometry.is_empty
    ].copy()

    if limite_disuelto.empty:
        raise ValueError(
            "No fue posible construir un límite territorial unificado."
        )

    metadata = {
        "archivo": str(path_boundary),
        "formato": path_boundary.suffix.lower(),
        "geometrias_originales": total_original,
        "geometrias_invalidas_detectadas": geometries_invalidas,
        "geometrias_finales": len(limite_disuelto),
        "crs_original": crs_original,
        "crs_final": str(limite_disuelto.crs),
        "bounds": tuple(
            limite_disuelto.total_bounds
        ),
    }

    return limite_disuelto, metadata


# ============================================================
# PREPARAR GEOMETRÍAS PARA RASTERIO
# ============================================================

def geometries_para_rasterio(
    gdf_limite,
    crs_destino,
):
    """
    Reproyecta el límite al CRS del raster y devuelve
    las geometrías compatibles con rasterio.

    Parámetros
    ----------
    gdf_limite : geopandas.GeoDataFrame
        Límite territorial.

    crs_destino : CRS
        CRS del raster.

    Retorna
    -------
    limite_reproyectado : geopandas.GeoDataFrame
    geometries : list
    """
    if not isinstance(
        gdf_limite,
        gpd.GeoDataFrame,
    ):
        raise TypeError(
            "gdf_limite debe ser un GeoDataFrame."
        )

    if gdf_limite.empty:
        raise ValueError(
            "El límite territorial está vacío."
        )

    if gdf_limite.crs is None:
        raise ValueError(
            "El límite territorial no tiene CRS."
        )

    if crs_destino is None:
        raise ValueError(
            "El raster no tiene un CRS definido."
        )

    limite_reproyectado = (
        gdf_limite.to_crs(
            crs_destino
        )
    )

    geometries = [
        geometria.__geo_interface__
        for geometria
        in limite_reproyectado.geometry
        if geometria is not None
        and not geometria.is_empty
    ]

    if not geometries:
        raise ValueError(
            "No existen geometrías válidas para recortar el raster."
        )

    return (
        limite_reproyectado,
        geometries,
    )


# ============================================================
# RECORTE DEL GEOTIFF
# ============================================================

def recortar_geotiff_con_limite(
    input_tif,
    output_tif,
    gdf_limite,
    crop=True,
    all_touched=False,
    nodata=None,
    overwrite=True,
    compress="deflate",
):
    """
    Recorta un GeoTIFF mediante un límite territorial.

    El raster original no se modifica. El resultado se guarda
    como un archivo nuevo.

    Parámetros
    ----------
    input_tif : str o Path
        GeoTIFF original.

    output_tif : str o Path
        Ruta del GeoTIFF recortado.

    gdf_limite : geopandas.GeoDataFrame
        Polígono territorial.

    crop : bool
        Si True, reduce el raster a la extensión del polígono.

    all_touched : bool
        Si True, incluye cualquier celda tocada por el polígono.

    nodata : float o None
        Valor nodata de salida. Si es None, se conserva el original.

    overwrite : bool
        Permite sobrescribir el archivo de salida.

    compress : str
        Método de compresión GeoTIFF.

    Retorna
    -------
    resultado : dict
        Información del GeoTIFF recortado.
    """
    input_tif = Path(
        input_tif
    )

    output_tif = Path(
        output_tif
    )

    if not input_tif.exists():
        raise FileNotFoundError(
            f"No se encontró el GeoTIFF original: {input_tif}"
        )

    if output_tif.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo de salida ya existe: {output_tif}"
        )

    output_tif.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        input_tif
    ) as src:

        limite_reproyectado, geometries = (
            geometries_para_rasterio(
                gdf_limite=gdf_limite,
                crs_destino=src.crs,
            )
        )

        raster_nodata = (
            nodata
            if nodata is not None
            else src.nodata
        )

        if raster_nodata is None:
            raster_nodata = -9999.0

        try:
            array_recortado, transform_recortado = mask(
                dataset=src,
                shapes=geometries,
                crop=crop,
                nodata=raster_nodata,
                filled=True,
                all_touched=all_touched,
            )

        except ValueError as error:
            raise ValueError(
                "El límite territorial no se superpone con el raster. "
                "Compruebe los sistemas de coordenadas y la ubicación "
                f"de ambos archivos. Detalle: {error}"
            ) from error

        profile = src.profile.copy()

        profile.update({
            "height": int(
                array_recortado.shape[1]
            ),
            "width": int(
                array_recortado.shape[2]
            ),
            "transform": transform_recortado,
            "nodata": raster_nodata,
            "compress": compress,
        })

        crs_raster = str(
            src.crs
        )

        bounds_originales = tuple(
            src.bounds
        )

        dimensiones_originales = (
            src.width,
            src.height,
        )

    if overwrite:
        for extra in (
            "",
            ".aux.xml",
        ):
            try:
                os.remove(
                    str(output_tif) + extra
                )
            except FileNotFoundError:
                pass

    with rasterio.open(
        output_tif,
        "w",
        **profile,
    ) as dst:
        dst.write(
            array_recortado
        )

        dst.update_tags(
            clipped="true",
            boundary_source="user_boundary",
        )

    with rasterio.open(
        output_tif
    ) as src_out:
        bounds_recortados = tuple(
            src_out.bounds
        )

        dimensiones_recortadas = (
            src_out.width,
            src_out.height,
        )

        transform_salida = src_out.transform

    resultado = {
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
        "crs": crs_raster,
        "nodata": raster_nodata,
        "bounds_originales": bounds_originales,
        "bounds_recortados": bounds_recortados,
        "dimensiones_originales": dimensiones_originales,
        "dimensiones_recortadas": dimensiones_recortadas,
        "transform": transform_salida,
        "profile": profile,
        "limite_reproyectado": limite_reproyectado,
    }

    return resultado


# ============================================================
# VISUALIZACIÓN DEL GEOTIFF RECORTADO
# ============================================================

def plot_geotiff_recortado(
    path_tif,
    gdf_limite=None,
    title="Superficie interpolada recortada",
    colorbar_label="Valor",
    cmap="YlOrRd",
    convertir_a_porcentaje=False,
    mostrar_limite=True,
    figsize=(11, 9),
):
    """
    Genera una figura a partir de un GeoTIFF recortado.

    Parámetros
    ----------
    path_tif : str o Path
        GeoTIFF recortado.

    gdf_limite : geopandas.GeoDataFrame o None
        Límite territorial para dibujar el contorno.

    convertir_a_porcentaje : bool
        Si True, multiplica los valores válidos por 100.

    Retorna
    -------
    fig, ax, metadata
    """
    path_tif = Path(
        path_tif
    )

    if not path_tif.exists():
        raise FileNotFoundError(
            f"No se encontró el GeoTIFF: {path_tif}"
        )

    with rasterio.open(
        path_tif
    ) as src:
        arr = src.read(
            1
        ).astype(float)

        nodata = src.nodata

        if nodata is not None:
            arr[
                np.isclose(
                    arr,
                    nodata,
                )
            ] = np.nan

        if convertir_a_porcentaje:
            arr = arr * 100.0

        bounds = src.bounds
        crs = src.crs

        extent = [
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ]

        width = src.width
        height = src.height

    valores_validos = arr[
        np.isfinite(arr)
    ]

    if valores_validos.size == 0:
        raise ValueError(
            "El GeoTIFF recortado no contiene valores válidos."
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    image = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
    )

    if (
        mostrar_limite
        and gdf_limite is not None
    ):
        limite = gdf_limite.to_crs(
            crs
        )

        limite.boundary.plot(
            ax=ax,
            color="black",
            linewidth=1.3,
            zorder=3,
        )

    cbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.82,
    )

    cbar.set_label(
        colorbar_label
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel(
        "Longitud / Coordenada X"
    )

    ax.set_ylabel(
        "Latitud / Coordenada Y"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    metadata = {
        "path": str(path_tif),
        "crs": str(crs),
        "width": width,
        "height": height,
        "bounds": tuple(bounds),
        "valor_minimo": float(
            np.nanmin(arr)
        ),
        "valor_maximo": float(
            np.nanmax(arr)
        ),
        "valor_promedio": float(
            np.nanmean(arr)
        ),
        "convertido_a_porcentaje": (
            convertir_a_porcentaje
        ),
    }

    return (
        fig,
        ax,
        metadata,
    )