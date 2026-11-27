"""Municipalización y cartografía sin interpolación de factores físicos existentes.

Este módulo solo extrae ventanas enteras de rásteres ya calculados, aplica el
límite municipal como máscara y escribe productos en una cuadrícula de referencia.
No recalcula procesos hidrológicos ni cambia los valores válidos de las fuentes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

from scripts.terrain_factors import _guardar_mapa


TARGET_EPSG = 6368
REFERENCE_REL = Path("results/territorial/dem/villa_de_alvarez_v2/elevacion_villa_de_alvarez_utm13n.tif")
BOUNDARY_REL = Path("datos_territoriales/red_hidrografica_villa_de_alvarez.gpkg")
BOUNDARY_LAYER = "limite_municipal"
OUTPUT_REL = Path("resultados/territorial/factores_fisicos/villa_de_alvarez")

FACTOR_SPECS = (
    {
        "factor": "elevacion",
        "source_rel": REFERENCE_REL,
        "band": 1,
        "unit": "m",
        "unit_basis": "etiqueta units=m en metadatos de banda",
        "description": "Elevación (m s. n. m.)",
        "cmap": "terrain",
        "filename": "elevacion_villa_de_alvarez",
        "method": "lectura en cuadrícula municipal existente y máscara municipal",
    },
    {
        "factor": "pendiente",
        "source_rel": Path("results/territorial/slope/villa_de_alvarez/pendientes_villa_de_alvarez.tif"),
        "band": 1,
        "unit": "grados",
        "unit_basis": "descripción de banda 1 del módulo terrain_factors.py",
        "description": "Pendiente (grados)",
        "cmap": "viridis",
        "filename": "pendiente_grados_villa_de_alvarez",
        "method": "lectura exacta de banda 1 (grados) y máscara municipal",
    },
    {
        "factor": "distancia_cauces",
        "source_rel": Path("results/territorial/distance_to_streams/villa_de_alvarez/distancia_cauces_villa_de_alvarez.tif"),
        "band": 1,
        "unit": "m",
        "unit_basis": "métrica documentada por distance_to_streams.py; CRS EPSG:6368",
        "description": "Distancia al cauce más cercano (m)",
        "cmap": "Blues",
        "filename": "distancia_cauces_villa_de_alvarez",
        "method": "lectura en cuadrícula municipal existente y máscara municipal",
    },
    {
        "factor": "acumulacion_flujo",
        "source_rel": Path("datos_territoriales/INEGI/CEM_4_0/procesados/Etapa_2_Hidrologia/94_Acumulacion_Flujo_Celdas.tif"),
        "band": 1,
        "unit": "celdas contribuyentes",
        "unit_basis": "d8_flow_accumulation(out_type='cells') en hydrological_processing.py",
        "description": "Acumulación de flujo (celdas contribuyentes)",
        "cmap": "magma",
        "filename": "acumulacion_flujo_villa_de_alvarez",
        "method": "recorte por ventana entera alineada; sin remuestreo; máscara municipal",
    },
    {
        "factor": "hand",
        "source_rel": Path("datos_territoriales/INEGI/CEM_4_0/procesados/Etapa_2_Hidrologia/97_Altura_HAND.tif"),
        "band": 1,
        "unit": "m",
        "unit_basis": "altura vertical relativa; MDE fuente en m; validación del módulo reporta HAND en m; el ráster no declara unidad",
        "description": "Altura sobre drenaje más cercano, HAND (m)",
        "cmap": "cividis",
        "filename": "hand_villa_de_alvarez",
        "method": "recorte por ventana entera alineada; sin remuestreo; máscara municipal",
    },
    {
        "factor": "twi",
        "source_rel": Path("datos_territoriales/INEGI/CEM_4_0/procesados/Etapa_2_Hidrologia/96_Indice_TWI.tif"),
        "band": 1,
        "unit": "índice (sin unidad física declarada)",
        "unit_basis": "índice calculado por calculate_twi en hydrological_processing.py; banda sin etiqueta de unidad",
        "description": "Índice topográfico de humedad (TWI)",
        "cmap": "YlGnBu",
        "filename": "twi_villa_de_alvarez",
        "method": "recorte por ventana entera alineada; sin remuestreo; máscara municipal",
    },
)


def _valid_nodata(values: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(values)
    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and math.isnan(float(nodata)):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata
    return valid


def aligned_window(source, reference) -> Window:
    """Return the exact integer source window matching the reference grid.

    Raises instead of interpolating if CRS, pixel size, orientation, phase, or
    source coverage is incompatible.
    """
    if source.crs is None or reference.crs is None or source.crs != reference.crs:
        raise ValueError("Fuente y referencia deben compartir CRS definido.")
    if source.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"Se requiere EPSG:{TARGET_EPSG}.")
    st, rt = source.transform, reference.transform
    if any(not math.isclose(v, 0.0, abs_tol=1e-12) for v in (st.b, st.d, rt.b, rt.d)):
        raise ValueError("Solo se admiten cuadrículas norte-arriba sin rotación.")
    if st.a <= 0 or st.e >= 0 or rt.a <= 0 or rt.e >= 0:
        raise ValueError("La cuadrícula debe tener orientación norte-arriba.")
    if not (math.isclose(st.a, rt.a, abs_tol=1e-9) and math.isclose(st.e, rt.e, abs_tol=1e-9)):
        raise ValueError("La resolución difiere; se rechaza el remuestreo.")
    col_float = (rt.c - st.c) / st.a
    row_float = (rt.f - st.f) / st.e
    col_off, row_off = round(col_float), round(row_float)
    if not (math.isclose(col_float, col_off, abs_tol=1e-8) and math.isclose(row_float, row_off, abs_tol=1e-8)):
        raise ValueError("El origen no coincide con fase de píxel entera; se rechaza interpolar.")
    window = Window(int(col_off), int(row_off), reference.width, reference.height)
    if col_off < 0 or row_off < 0 or col_off + reference.width > source.width or row_off + reference.height > source.height:
        raise ValueError("La fuente no cubre toda la cuadrícula de referencia.")
    if source.window_transform(window) != reference.transform:
        raise ValueError("La transformación de la ventana no coincide exactamente con la referencia.")
    return window


def prepare_masked_array(source, reference, geometry, band=1):
    """Read an exactly aligned source window and mask outside geometry to NoData."""
    if band < 1 or band > source.count:
        raise ValueError(f"La banda {band} no existe en la fuente.")
    if source.nodata is None:
        raise ValueError("La fuente no declara NoData; no se inferirá ni sustituirá por cero.")
    window = aligned_window(source, reference)
    values = source.read(band, window=window)
    source_valid = source.read_masks(band, window=window) > 0
    source_valid &= _valid_nodata(values, source.nodata)
    inside = geometry_mask(
        [geometry.__geo_interface__],
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        invert=True,
        all_touched=False,
    )
    valid = source_valid & inside
    result = values.copy()
    result[~valid] = source.nodata
    return result, valid, inside, window


def _write_factor(source_path, output_path, reference_path, geometry, spec, source_sha256):
    if output_path.exists():
        raise FileExistsError(f"No se sobrescribe producto existente: {output_path}")
    with rasterio.open(reference_path) as reference, rasterio.open(source_path) as source:
        if reference.width != 2674 or reference.height != 1552 or reference.res != (15.0, 15.0):
            raise ValueError("La referencia no es la cuadrícula 2674 × 1552 a 15 m.")
        if reference.crs is None or reference.crs.to_epsg() != TARGET_EPSG:
            raise ValueError(f"La referencia debe ser EPSG:{TARGET_EPSG}.")
        if source.nodata is None:
            raise ValueError(f"La fuente no declara NoData: {source_path}")
        values, valid, inside, window = prepare_masked_array(
            source, reference, geometry, band=spec["band"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "width": reference.width,
            "height": reference.height,
            "count": 1,
            "dtype": source.dtypes[spec["band"] - 1],
            "crs": reference.crs,
            "transform": reference.transform,
            "nodata": source.nodata,
            "compress": "deflate",
        }
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(values, 1)
            destination.set_band_description(1, spec["description"])
            if spec["unit"] in ("m", "grados"):
                destination.set_band_unit(1, spec["unit"])
            destination.update_tags(
                factor=spec["factor"],
                source_path=str(source_path),
                source_inventory_sha256=source_sha256,
                alignment_method=spec["method"],
                resampling="none; exact integer window extraction",
                municipal_mask="pixel center, all_touched=False",
                source_band=str(spec["band"]),
                unit_basis=spec["unit_basis"],
            )
    return {
        "valid": valid,
        "inside": inside,
        "window": window,
        "source_nodata": source.nodata,
        "source_dtype": source.dtypes[spec["band"] - 1],
    }


def process_factors(data_root: Path, output_dir: Path, boundary_path: Path | None = None):
    """Create six single-band municipal GeoTIFFs, PNGs, and one CSV manifest."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    boundary_path = Path(boundary_path) if boundary_path else data_root / BOUNDARY_REL
    manifest_path = output_dir / "manifiesto_factores_fisicos.csv"
    expected_paths = [manifest_path]
    for spec in FACTOR_SPECS:
        expected_paths.extend(
            [output_dir / f"{spec['filename']}.tif", output_dir / f"{spec['filename']}.png"]
        )
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError("No se sobrescriben productos existentes: " + ", ".join(map(str, existing)))

    reference_path = data_root / REFERENCE_REL
    for spec in FACTOR_SPECS:
        path = data_root / spec["source_rel"]
        if not path.is_file():
            raise FileNotFoundError(f"Falta la fuente local: {path}")
    if not reference_path.is_file() or not boundary_path.is_file():
        raise FileNotFoundError("Falta la cuadrícula de referencia o el GeoPackage municipal local.")

    boundary = gpd.read_file(boundary_path, layer=BOUNDARY_LAYER)
    if boundary.empty or boundary.crs is None or boundary.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"La capa {BOUNDARY_LAYER} debe tener geometrías y CRS EPSG:{TARGET_EPSG}.")
    if len(boundary) != 1 or not boundary.geometry.iloc[0].is_valid or boundary.geometry.iloc[0].is_empty:
        raise ValueError("Se requiere un único polígono municipal válido.")
    geometry = boundary.geometry.iloc[0]

    # Source hashes are carried from the previously generated inventory; they
    # are not recalculated here.
    inventory_path = data_root / "inventario/inventario_factores_20261125T213730Z_4b965a4b.csv"
    inventory = {}
    if inventory_path.is_file():
        with inventory_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                inventory[row.get("ruta", "").casefold()] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in FACTOR_SPECS:
        source_path = data_root / spec["source_rel"]
        output_tif = output_dir / f"{spec['filename']}.tif"
        output_png = output_dir / f"{spec['filename']}.png"
        source_key = ("G:\\My Drive\\Doctorado-DCAG\\" + str(spec["source_rel"]).replace("/", "\\")).casefold()
        inventory_row = inventory.get(source_key, {})
        source_sha = inventory_row.get("sha256", "")
        result = _write_factor(source_path, output_tif, reference_path, geometry, spec, source_sha)

        # Reuse the repository's existing cartographic renderer. It uses the
        # projected raster bounds, nearest display, and no visual smoothing.
        with rasterio.open(output_tif) as product:
            array = product.read(1)
            valid = product.read_masks(1) > 0
            valid &= _valid_nodata(array, product.nodata)
            _guardar_mapa(
                array,
                valid,
                product.transform,
                boundary,
                f"{spec['description']} — Villa de Álvarez, Colima",
                spec["unit"],
                output_png,
                spec["cmap"],
            )
            bounds = tuple(float(v) for v in product.bounds)
            width, height = product.width, product.height
            resolution = tuple(float(v) for v in product.res)
            nodata = product.nodata

        municipality_cells = int(result["inside"].sum())
        valid_cells = int(result["valid"].sum())
        manifest.append(
            {
                "factor": spec["factor"],
                "source_path_local": str(source_path),
                "source_sha256_inventario_no_recalculado": source_sha,
                "source_band": spec["band"],
                "source_band_description": spec["description"],
                "unit": spec["unit"],
                "unit_basis": spec["unit_basis"],
                "source_nodata": nodata,
                "output_nodata": nodata,
                "crs": f"EPSG:{TARGET_EPSG}",
                "resolution_x_m": resolution[0],
                "resolution_y_m": resolution[1],
                "width": width,
                "height": height,
                "bounds_epsg6368": ";".join(map(str, bounds)),
                "municipal_cells": municipality_cells,
                "valid_municipal_cells": valid_cells,
                "valid_coverage_pct": round(100 * valid_cells / municipality_cells, 6) if municipality_cells else 0,
                "alignment_method": spec["method"],
                "resampling": "none; exact integer pixel window",
                "output_geotiff": str(output_tif),
                "output_png": str(output_png),
            }
        )

    fieldnames = list(manifest[0])
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--boundary", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_root / OUTPUT_REL
    manifest, rows = process_factors(args.data_root, output_dir, args.boundary)
    print(f"manifiesto: {manifest}")
    for row in rows:
        print(f"{row['factor']}: {row['output_geotiff']} | {row['output_png']}")


if __name__ == "__main__":
    main()
