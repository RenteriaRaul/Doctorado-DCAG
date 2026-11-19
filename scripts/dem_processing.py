"""Recorte y visualizacion municipal de un MDE con Rasterio.

Este modulo es independiente de Streamlit. Lee solo la ventana del municipio
mediante un WarpedVRT y conserva las elevaciones en sus unidades originales.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from scripts.boundary import geometries_para_rasterio
from scripts.gis_manager import GISManager


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = (
    REPO_ROOT
    / "datos_territoriales"
    / "INEGI"
    / "CEM_4_0"
    / "originales"
    / "conjunto_de_datos"
    / "06_Colima_r15m_v4.tif"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "territorial" / "dem" / "villa_de_alvarez"
DEFAULT_TARGET_CRS = "EPSG:6368"  # Mexico ITRF2008 / UTM 13N


def generar_mapa_elevacion_municipal(
    dem_path=DEFAULT_DEM,
    marco_geoestadistico_dir=None,
    output_dir=DEFAULT_OUTPUT,
    estado="Colima",
    municipio="Villa de Álvarez",
    crs_destino=DEFAULT_TARGET_CRS,
    resolucion_m=15.0,
    nodata_salida=-32768,
    unidades_elevacion="m",
):
    """Genera GeoTIFF municipal, PNG de elevacion y JSON de metadatos.

    `marco_geoestadistico_dir` debe apuntar a la carpeta que contiene los ZIP
    estatales del Marco Geoestadistico, tal como espera GISManager.
    Los archivos de salida preexistentes nunca se sobrescriben.
    """
    dem_path = Path(dem_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if marco_geoestadistico_dir is None:
        raise ValueError("Indica la carpeta del Marco Geoestadístico INEGI.")
    marco_geoestadistico_dir = Path(marco_geoestadistico_dir).expanduser().resolve()

    tif_path = output_dir / "elevacion_villa_de_alvarez_utm13n.tif"
    png_path = output_dir / "elevacion_villa_de_alvarez.png"
    metadata_path = output_dir / "metadatos_elevacion.json"
    salidas = (tif_path, png_path, metadata_path)

    if not dem_path.is_file():
        raise FileNotFoundError(f"No se encontró el MDE: {dem_path}")
    existentes = [str(p) for p in salidas if p.exists()]
    if existentes:
        raise FileExistsError(
            "Ya existen productos de salida; no se sobrescriben: "
            + ", ".join(existentes)
        )

    manager = GISManager(marco_geoestadistico_dir)
    limite = manager.obtener_municipio(
        estado=estado,
        municipio=municipio,
        crs_destino=crs_destino,
        territorio="completo",
    )
    limite, geometrias = geometries_para_rasterio(limite, crs_destino)

    with rasterio.open(dem_path) as src:
        if src.count != 1:
            raise ValueError(f"Se esperaba un MDE de una banda; se encontraron {src.count}.")
        if src.crs is None:
            raise ValueError("El MDE no contiene CRS; no se puede reproyectar con seguridad.")
        if not np.isfinite(resolucion_m) or resolucion_m <= 0:
            raise ValueError("resolucion_m debe ser un valor positivo en metros.")

        nodata_entrada = src.nodata
        if nodata_entrada is None:
            raise ValueError(
                "El MDE no declara NoData. Especifica y valida ese valor antes del proceso."
            )

        # Alinear la extensión municipal a una cuadrícula métrica con tamaño
        # exacto de píxel. Definir resolución solamente en WarpedVRT puede
        # redondear sus dimensiones y producir una resolución efectiva distinta.
        minx, miny, maxx, maxy = limite.total_bounds
        left = math.floor(minx / resolucion_m) * resolucion_m
        bottom = math.floor(miny / resolucion_m) * resolucion_m
        right = math.ceil(maxx / resolucion_m) * resolucion_m
        top = math.ceil(maxy / resolucion_m) * resolucion_m
        vrt_width = int(round((right - left) / resolucion_m))
        vrt_height = int(round((top - bottom) / resolucion_m))
        vrt_transform = from_origin(left, top, resolucion_m, resolucion_m)

        # WarpedVRT reproyecta por bloques segun la ventana solicitada por mask;
        # no materializa el MDE estatal completo en memoria.
        with WarpedVRT(
            src,
            crs=crs_destino,
            transform=vrt_transform,
            width=vrt_width,
            height=vrt_height,
            resampling=Resampling.nearest,
            src_nodata=nodata_entrada,
            nodata=nodata_salida,
        ) as vrt:
            datos, transform = mask(
                vrt,
                geometrias,
                crop=True,
                all_touched=False,
                filled=True,
                nodata=nodata_salida,
            )
            perfil = vrt.profile.copy()
            perfil.update(
                driver="GTiff",
                height=datos.shape[1],
                width=datos.shape[2],
                transform=transform,
                crs=crs_destino,
                nodata=nodata_salida,
                compress="deflate",
            )
            ventana_transform = transform
            crs_origen = src.crs.to_string()
            resolucion_origen = tuple(float(v) for v in src.res)
            dimensiones_origen = (int(src.height), int(src.width))
            tipo_origen = src.dtypes[0]
            bounds_origen = tuple(float(v) for v in src.bounds)
            tags_origen = src.tags(1)

    validos = datos[0] != nodata_salida
    if not np.any(validos):
        raise ValueError("El recorte no contiene celdas válidas del MDE.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path, "w", **perfil) as dst:
        dst.write(datos)
        dst.set_band_description(1, "Elevación")
        dst.update_tags(
            1,
            units=unidades_elevacion,
            source=str(dem_path),
            resampling="nearest",
            boundary=f"{estado} / {municipio} (Marco Geoestadístico INEGI)",
        )

    # Registrar la resolución medida en el GeoTIFF final, que es la fuente
    # de verdad para los productos escritos.
    with rasterio.open(tif_path) as resultado:
        resolucion_real_m = [abs(float(v)) for v in resultado.res]

    # La imagen usa exclusivamente el recorte municipal ya procesado.
    masked = np.ma.masked_equal(datos[0], nodata_salida)
    west = ventana_transform.c
    north = ventana_transform.f
    east = west + ventana_transform.a * datos.shape[2]
    south = north + ventana_transform.e * datos.shape[1]
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = ax.imshow(
        masked,
        extent=(west, east, south, north),
        origin="upper",
        cmap="terrain",
        interpolation="nearest",
    )
    limite.boundary.plot(ax=ax, color="black", linewidth=0.8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(f"Elevación ({unidades_elevacion})")
    ax.set_title(f"Elevación del terreno — {municipio}, {estado}")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "procesado_utc": datetime.now(timezone.utc).isoformat(),
        "fuente": str(dem_path),
        "marco_geoestadistico": str(marco_geoestadistico_dir),
        "territorio": {"estado": estado, "municipio": municipio},
        "crs_origen": crs_origen,
        "crs_destino": str(crs_destino),
        "resolucion_origen_en_unidades_crs": resolucion_origen,
        "resolucion_destino_solicitada_m": [resolucion_m, resolucion_m],
        "resolucion_destino_m": resolucion_real_m,
        "dimensiones_origen_filas_columnas": dimensiones_origen,
        "bounds_origen_en_unidades_crs": bounds_origen,
        "tipo_dato_origen": tipo_origen,
        "banda": 1,
        "nodata_origen": float(nodata_entrada),
        "nodata_salida": float(nodata_salida),
        "unidades_elevacion": unidades_elevacion,
        "metodo": (
            "GISManager.obtener_municipio; WarpedVRT de Rasterio a CRS métrico; "
            "recorte con rasterio.mask.mask (crop=True, all_touched=False); "
            "remuestreo nearest para conservar valores de elevación; "
            "GeoTIFF Deflate y PNG con contorno municipal."
        ),
        "celdas_validas": int(validos.sum()),
        "elevacion_min": float(datos[0][validos].min()),
        "elevacion_max": float(datos[0][validos].max()),
        "salidas": {"geotiff": str(tif_path), "png": str(png_path)},
        "tags_banda_origen": tags_origen,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"geotiff": tif_path, "png": png_path, "metadatos": metadata_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--marco-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crs", default=DEFAULT_TARGET_CRS)
    parser.add_argument("--resolucion-m", type=float, default=15.0)
    args = parser.parse_args()
    salidas = generar_mapa_elevacion_municipal(
        dem_path=args.dem,
        marco_geoestadistico_dir=args.marco_dir,
        output_dir=args.output_dir,
        crs_destino=args.crs,
        resolucion_m=args.resolucion_m,
    )
    for tipo, ruta in salidas.items():
        print(f"{tipo}: {ruta}")


if __name__ == "__main__":
    main()
