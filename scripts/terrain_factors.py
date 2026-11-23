"""Cálculo municipal de pendiente por el método de Horn (1981).

El MDE original es la fuente de elevaciones. El GeoTIFF municipal validado se
usa solo como plantilla de cuadrícula para mantener la alineación espacial.
Este módulo es independiente de Streamlit.
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
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT

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
DEFAULT_GRID_REFERENCE = (
    REPO_ROOT
    / "results"
    / "territorial"
    / "dem"
    / "villa_de_alvarez_v2"
    / "elevacion_villa_de_alvarez_utm13n.tif"
)
DEFAULT_MARCO = Path(r"C:\Users\rente\Desktop\Marco_geoestadistico_2025")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "territorial" / "slope" / "villa_de_alvarez"
TARGET_EPSG = 6368
HALO_PIXELS = 2
NODATA_SLOPE = -9999.0


def calcular_pendiente_horn(elevacion, pixel_size_x=15.0, pixel_size_y=15.0, nodata=None):
    """Calcula pendientes Horn 3x3 en grados y porcentaje.

    Acepta un ndarray o masked array. Devuelve dos arrays float32 del mismo
    tamaño; las celdas de borde y toda ventana 3x3 con datos inválidos son NaN.
    ``pixel_size_x`` y ``pixel_size_y`` se expresan en las mismas unidades
    horizontales que las elevaciones.
    """
    z_in = np.ma.asarray(elevacion)
    z = np.asarray(z_in.astype(np.float64).filled(np.nan), dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("La elevación debe ser una matriz bidimensional.")
    if z.shape[0] < 3 or z.shape[1] < 3:
        raise ValueError("Horn requiere una superficie de al menos 3x3 celdas.")
    if not math.isfinite(pixel_size_x) or pixel_size_x <= 0:
        raise ValueError("pixel_size_x debe ser positivo y finito.")
    if not math.isfinite(pixel_size_y) or pixel_size_y <= 0:
        raise ValueError("pixel_size_y debe ser positivo y finito.")

    valid = ~np.ma.getmaskarray(z_in) & np.isfinite(z)
    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
            valid &= ~np.isnan(z)
        else:
            valid &= z != nodata

    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z5, z6 = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    window_valid = (
        valid[:-2, :-2]
        & valid[:-2, 1:-1]
        & valid[:-2, 2:]
        & valid[1:-1, :-2]
        & valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[2:, :-2]
        & valid[2:, 1:-1]
        & valid[2:, 2:]
    )

    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * pixel_size_x)
    dz_dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * pixel_size_y)
    gradient = np.hypot(dz_dx, dz_dy)
    degrees_inner = np.degrees(np.arctan(gradient))
    percent_inner = 100.0 * gradient

    degrees = np.full(z.shape, np.nan, dtype=np.float32)
    percent = np.full(z.shape, np.nan, dtype=np.float32)
    degrees_view = degrees[1:-1, 1:-1]
    percent_view = percent[1:-1, 1:-1]
    degrees_view[window_valid] = degrees_inner[window_valid].astype(np.float32)
    percent_view[window_valid] = percent_inner[window_valid].astype(np.float32)
    return degrees, percent


def _assert_north_up_metric_grid(dataset):
    if dataset.crs is None or dataset.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"La cuadrícula de referencia debe ser EPSG:{TARGET_EPSG}.")
    t = dataset.transform
    if not (math.isclose(t.b, 0.0) and math.isclose(t.d, 0.0) and t.a > 0 and t.e < 0):
        raise ValueError("Se requiere una cuadrícula norte-arriba sin rotación.")
    if not (math.isclose(t.a, 15.0, abs_tol=1e-9) and math.isclose(-t.e, 15.0, abs_tol=1e-9)):
        raise ValueError("La resolución de referencia no es exactamente 15 x 15 m.")


def _estadisticas(array, valid):
    values = np.asarray(array[valid], dtype=np.float64)
    total = int(valid.size)
    n_valid = int(values.size)
    if n_valid:
        stats = {
            "minimo": float(np.min(values)),
            "maximo": float(np.max(values)),
            "media": float(np.mean(values)),
            "mediana": float(np.median(values)),
            "desviacion_estandar": float(np.std(values, ddof=0)),
        }
    else:
        stats = {k: None for k in ("minimo", "maximo", "media", "mediana", "desviacion_estandar")}
    stats.update({"celdas_validas": n_valid, "celdas_invalidas": total - n_valid})
    return stats


def _guardar_mapa(array, valid, transform, limite, titulo, etiqueta, ruta, cmap):
    visual = np.ma.masked_where(~valid, array)
    west = transform.c
    north = transform.f
    east = west + transform.a * array.shape[1]
    south = north + transform.e * array.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    img = ax.imshow(
        visual,
        extent=(west, east, south, north),
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
    )
    limite.boundary.plot(ax=ax, color="black", linewidth=0.7)
    colorbar = fig.colorbar(img, ax=ax, shrink=0.84)
    colorbar.set_label(etiqueta)
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal")
    fig.savefig(ruta, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generar_pendientes_municipales(
    dem_path=DEFAULT_DEM,
    referencia_path=DEFAULT_GRID_REFERENCE,
    marco_geoestadistico_dir=DEFAULT_MARCO,
    output_dir=DEFAULT_OUTPUT,
    estado="Colima",
    municipio="Villa de Álvarez",
    nodata_salida=NODATA_SLOPE,
):
    """Genera bandas Horn en grados/% más mapas y metadatos JSON.

    El algoritmo reproyecta por ventana desde el DEM original. El GeoTIFF de
    referencia fija exactamente la cuadrícula final y solo se usa para esa
    alineación y para identificar celdas interiores válidas.
    """
    dem_path = Path(dem_path).expanduser().resolve()
    referencia_path = Path(referencia_path).expanduser().resolve()
    marco_geoestadistico_dir = Path(marco_geoestadistico_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    outputs = {
        "geotiff": output_dir / "pendientes_villa_de_alvarez.tif",
        "grados_png": output_dir / "pendiente_grados.png",
        "porcentaje_png": output_dir / "pendiente_porcentaje.png",
        "metadatos": output_dir / "metadatos_pendiente.json",
    }
    existentes = [str(p) for p in outputs.values() if p.exists()]
    if existentes:
        raise FileExistsError("No se sobrescriben productos existentes: " + ", ".join(existentes))
    if not dem_path.is_file():
        raise FileNotFoundError(f"No se encontró el MDE original: {dem_path}")
    if not referencia_path.is_file():
        raise FileNotFoundError(f"No se encontró la referencia de cuadrícula: {referencia_path}")

    manager = GISManager(marco_geoestadistico_dir)
    limite = manager.obtener_municipio(
        estado=estado,
        municipio=municipio,
        crs_destino=f"EPSG:{TARGET_EPSG}",
        territorio="completo",
    )
    if limite.empty or limite.crs is None:
        raise ValueError("GISManager no devolvió una geometría municipal con CRS.")
    limite = limite.to_crs(f"EPSG:{TARGET_EPSG}")
    geometrias = [g.__geo_interface__ for g in limite.geometry if g is not None and not g.is_empty]
    if not geometrias:
        raise ValueError("El límite municipal no contiene geometrías válidas.")

    with rasterio.open(referencia_path) as ref:
        _assert_north_up_metric_grid(ref)
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width, ref_height = ref.width, ref.height
        ref_profile = ref.profile.copy()
        ref_valid = ref.read_masks(1) > 0
        municipal_mask = geometry_mask(
            geometrias,
            out_shape=(ref_height, ref_width),
            transform=ref_transform,
            invert=True,
            all_touched=False,
        )
        ref_valid &= municipal_mask
        reference_valid_count = int(ref_valid.sum())
        if reference_valid_count == 0:
            raise ValueError("La referencia no contiene celdas válidas dentro del municipio.")

    # La referencia ya define la cuadrícula municipal. Extender su afín dos
    # filas/columnas en cada lado conserva exactamente la misma fase de píxel.
    halo_transform = ref_transform * Affine.translation(-HALO_PIXELS, -HALO_PIXELS)
    halo_width = ref_width + 2 * HALO_PIXELS
    halo_height = ref_height + 2 * HALO_PIXELS

    with rasterio.open(dem_path) as src:
        if src.count < 1 or src.crs is None:
            raise ValueError("El MDE original debe contener al menos una banda y tener CRS definido.")
        source_nodata = src.nodata
        if source_nodata is None:
            raise ValueError("El MDE original no declara NoData; no se inferirá como cero.")
        with WarpedVRT(
            src,
            crs=ref_crs,
            transform=halo_transform,
            width=halo_width,
            height=halo_height,
            resampling=Resampling.nearest,
            src_nodata=source_nodata,
            nodata=source_nodata,
            dtype="float32",
        ) as vrt:
            # Solo se materializa la ventana municipal más el halo de 30 m.
            elevations = vrt.read(1, masked=True).astype(np.float32)
        source_crs = src.crs.to_string()
        source_resolution = [float(v) for v in src.res]
        source_dimensions = [int(src.height), int(src.width)]
        source_dtype = src.dtypes[0]
        source_nodata_json = float(source_nodata)

    slope_degrees_halo, slope_percent_halo = calcular_pendiente_horn(
        elevations,
        pixel_size_x=15.0,
        pixel_size_y=15.0,
        nodata=source_nodata,
    )
    inner = (slice(HALO_PIXELS, HALO_PIXELS + ref_height), slice(HALO_PIXELS, HALO_PIXELS + ref_width))
    slope_degrees = slope_degrees_halo[inner]
    slope_percent = slope_percent_halo[inner]
    slope_valid = ref_valid & np.isfinite(slope_degrees) & np.isfinite(slope_percent)
    degrees_out = np.full((ref_height, ref_width), nodata_salida, dtype=np.float32)
    percent_out = np.full((ref_height, ref_width), nodata_salida, dtype=np.float32)
    degrees_out[slope_valid] = slope_degrees[slope_valid]
    percent_out[slope_valid] = slope_percent[slope_valid]

    valid_slope_count = int(slope_valid.sum())
    lost_valid_count = reference_valid_count - valid_slope_count
    total_municipal_cells = int(municipal_mask.sum())
    output_invalid_count = total_municipal_cells - valid_slope_count
    if valid_slope_count == 0:
        raise ValueError("No quedaron celdas válidas tras el cálculo Horn y la máscara NoData.")

    stats = {
        "grados": _estadisticas(degrees_out, slope_valid),
        "porcentaje": _estadisticas(percent_out, slope_valid),
    }
    metadata = {
        "procesado_utc": datetime.now(timezone.utc).isoformat(),
        "territorio": {"estado": estado, "municipio": municipio},
        "fuente_elevaciones": str(dem_path),
        "referencia_cuadricula": str(referencia_path),
        "marco_geoestadistico": str(marco_geoestadistico_dir),
        "crs_origen": source_crs,
        "crs_destino": ref_crs.to_string(),
        "resolucion_origen_en_unidades_crs": source_resolution,
        "resolucion_destino_m": [abs(float(ref_transform.a)), abs(float(ref_transform.e))],
        "dimensiones_fuente_filas_columnas": source_dimensions,
        "tipo_dato_fuente": source_dtype,
        "tipo_dato_salida": "float32",
        "transformacion_afin": list(ref_transform)[:6],
        "dimensiones_salida_filas_columnas": [ref_height, ref_width],
        "extension_salida": None,
        "nodata_fuente": source_nodata_json,
        "nodata_salida": float(nodata_salida),
        "algoritmo": "Horn (1981), ventana 3x3",
        "remuestreo_reproyeccion": "nearest",
        "margen_exterior_pixeles": HALO_PIXELS,
        "margen_exterior_m": HALO_PIXELS * 15.0,
        "tratamiento_nodata": (
            "NoData no se reemplaza por cero; una ventana 3x3 con cualquier celda inválida "
            "produce NoData en el píxel central."
        ),
        "celdas_validas_referencia_municipal": reference_valid_count,
        "celdas_validas_salida": valid_slope_count,
        "perdida_celdas_validas_por_nodata": lost_valid_count,
        "porcentaje_perdida_celdas_validas": (
            100.0 * lost_valid_count / reference_valid_count if reference_valid_count else 0.0
        ),
        "celdas_invalidas_salida_en_municipio": output_invalid_count,
        "estadisticas": stats,
        "salidas": {k: str(v) for k, v in outputs.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=2,
        dtype="float32",
        nodata=nodata_salida,
        compress="deflate",
    )
    with rasterio.open(outputs["geotiff"], "w", **profile) as dst:
        dst.write(degrees_out, 1)
        dst.write(percent_out, 2)
        dst.set_band_description(1, "Pendiente (grados)")
        dst.set_band_description(2, "Pendiente (porcentaje)")
        dst.update_tags(
            algorithm="Horn (1981), 3x3",
            source=str(dem_path),
            grid_reference=str(referencia_path),
            resampling="nearest",
            halo_m="30",
        )

    # Comprobación explícita de igualdad con el GeoTIFF de referencia.
    with rasterio.open(referencia_path) as ref, rasterio.open(outputs["geotiff"]) as out:
        checks = {
            "CRS": out.crs == ref.crs,
            "resolución": out.res == ref.res,
            "dimensiones": (out.width, out.height) == (ref.width, ref.height),
            "transformación afín": out.transform == ref.transform,
            "extensión": out.bounds == ref.bounds,
        }
        if not all(checks.values()):
            raise RuntimeError("La cuadrícula de salida no coincide exactamente: " + str(checks))
        metadata["verificacion_cuadricula"] = checks
        metadata["extension_salida"] = [float(v) for v in out.bounds]

    _guardar_mapa(
        degrees_out,
        slope_valid,
        ref_transform,
        limite,
        f"Pendiente del terreno — {municipio}, {estado}",
        "Pendiente (grados)",
        outputs["grados_png"],
        "viridis",
    )
    _guardar_mapa(
        percent_out,
        slope_valid,
        ref_transform,
        limite,
        f"Pendiente del terreno — {municipio}, {estado}",
        "Pendiente (%)",
        outputs["porcentaje_png"],
        "plasma",
    )
    outputs["metadatos"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--referencia", type=Path, default=DEFAULT_GRID_REFERENCE)
    parser.add_argument("--marco-dir", type=Path, default=DEFAULT_MARCO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for tipo, ruta in generar_pendientes_municipales(
        dem_path=args.dem,
        referencia_path=args.referencia,
        marco_geoestadistico_dir=args.marco_dir,
        output_dir=args.output_dir,
    ).items():
        print(f"{tipo}: {ruta}")


if __name__ == "__main__":
    main()
