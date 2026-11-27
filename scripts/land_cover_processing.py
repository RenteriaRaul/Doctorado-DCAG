"""Municipal categorical rasterization of INEGI Serie VII land cover."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from shapely.ops import unary_union

from scripts.edaphology_processing import (
    EXPECTED_HEIGHT, EXPECTED_RESOLUTION, EXPECTED_WIDTH, EPSG, NODATA,
    _surface_statistics, _typed_value, _value_display, _write_map,
    assert_new_output_dir, find_overlaps, prepare_polygon_records,
    rasterize_categories, stable_category_mapping, validate_reference_grid,
)
from scripts.edaphology_audit import coverage_pct, geometry_metrics


def build_description_lookup(records):
    """Map each non-null CLAVE to its single original DESCRIPCIO.

    A source code with multiple descriptions is ambiguous and must be resolved
    against the source dictionary before rasterization; nothing is guessed.
    """
    descriptions = {}
    for record in records:
        code = record["properties"].get("CLAVE")
        description = record["properties"].get("DESCRIPCIO")
        if code is None or pd.isna(code) or (isinstance(code, str) and not code.strip()):
            continue
        key = _typed_value(code)
        value_key = ("null", None) if description is None or pd.isna(description) else _typed_value(description)
        if key in descriptions and descriptions[key][0] != value_key:
            raise ValueError(f"CLAVE {code!r} tiene más de una DESCRIPCIO en el dominio municipal.")
        descriptions[key] = (value_key, description)
    return {key: pair[1] for key, pair in descriptions.items()}


def build_category_catalog(records):
    """Create reproducible raster IDs and retain source code/description."""
    values = [record["properties"].get("CLAVE") for record in records]
    mapping = stable_category_mapping(values)
    descriptions = build_description_lookup(records)
    source_values = {_typed_value(value): value for value in values if value is not None and not pd.isna(value) and not (isinstance(value, str) and not value.strip())}
    counts = Counter(_typed_value(value) for value in values if value is not None and not pd.isna(value) and not (isinstance(value, str) and not value.strip()))
    catalog = []
    for key, raster_code in sorted(mapping.items(), key=lambda item: item[1]):
        value = source_values[key]
        description = descriptions.get(key)
        catalog.append({
            "raster_code": int(raster_code),
            "CLAVE": str(value),
            "CLAVE_typed_json": _value_display(value),
            "DESCRIPCIO": "" if description is None or pd.isna(description) else str(description),
            "description_is_null": bool(description is None or pd.isna(description)),
            "polygon_count": int(counts[key]),
            "nodata_code": NODATA,
        })
    return mapping, catalog


def rasterize_land_cover(records, reference, inside_mask):
    """Rasterize original CLAVE categories; pixel center, no interpolation."""
    values = [record["properties"].get("CLAVE") for record in records]
    mapping = stable_category_mapping(values)
    array, mapping = rasterize_categories(records, "CLAVE", reference, inside_mask, mapping)
    return array, mapping


def _land_cover_statistics(records, array, inside_mask, municipality_area_m2, catalog):
    base = _surface_statistics(records, "CLAVE", array, inside_mask, municipality_area_m2)
    for row in base:
        match = next(item for item in catalog if item["raster_code"] == row["raster_code"])
        row["CLAVE"] = match["CLAVE"]
        row["DESCRIPCIO"] = match["DESCRIPCIO"]
        row["description_is_null"] = match["description_is_null"]
        row["source_value"] = match["CLAVE"]
        row["inegi_category_name"] = match["DESCRIPCIO"]
    return base


def process_land_cover(
    source_path: Path,
    metadata_path: Path,
    dictionary_path: Path,
    boundary_path: Path,
    grid_path: Path,
    output_dir: Path,
):
    source_path, metadata_path, dictionary_path, boundary_path, grid_path = map(
        Path, (source_path, metadata_path, dictionary_path, boundary_path, grid_path)
    )
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"No se sobrescriben resultados: {out}")
    if not source_path.is_file() or not metadata_path.is_file() or not dictionary_path.is_file():
        raise FileNotFoundError("Falta el shapefile, metadato o diccionario Serie VII indicado.")

    import pyogrio
    source_info = pyogrio.read_info(source_path)
    source_crs = CRS.from_user_input(source_info["crs"])
    source_fields = set(source_info["fields"])
    missing = {"CLAVE", "DESCRIPCIO"} - source_fields
    if missing:
        raise KeyError(f"Faltan campos fuente obligatorios: {sorted(missing)}")

    boundary = gpd.read_file(boundary_path, layer="limite_municipal")
    if len(boundary) != 1 or boundary.crs is None or boundary.crs.to_epsg() != EPSG:
        raise ValueError("Se requiere el límite municipal único en EPSG:6368.")
    municipality = boundary.geometry.iloc[0]
    if municipality.is_empty or not municipality.is_valid:
        raise ValueError("El límite municipal no es válido.")

    with rasterio.open(grid_path) as reference:
        transform = validate_reference_grid(reference)
        reference_profile = reference.profile.copy()
        reference_mask = reference.read_masks(1) > 0
        if int(reference_mask.sum()) != 1_277_051:
            raise ValueError("La máscara del ráster físico de referencia no tiene 1,277,051 celdas.")
    inside_mask = geometry_mask(
        [municipality.__geo_interface__],
        out_shape=(EXPECTED_HEIGHT, EXPECTED_WIDTH),
        transform=transform,
        invert=True,
        all_touched=False,
    )
    if not np.array_equal(inside_mask, reference_mask):
        raise ValueError("La máscara municipal no coincide con el ráster físico de referencia.")

    municipality_source = gpd.GeoSeries([municipality], crs=boundary.crs).to_crs(source_crs).iloc[0]
    # Only read the source-space municipal bounding box, then retain exact
    # intersections through the existing edaphology selection/repair helper.
    candidates = gpd.read_file(source_path, bbox=municipality_source.bounds, columns=["CLAVE", "DESCRIPCIO"])
    records, repairs = prepare_polygon_records(candidates, municipality_source, ["CLAVE", "DESCRIPCIO"])
    if not records:
        raise ValueError("No hay polígonos Serie VII que intersecten el municipio.")
    selected = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs).to_crs(boundary.crs)
    records = [
        {"feature_id": str(row.feature_id), "geometry": row.geometry, "properties": row.properties, "repaired": row.repaired}
        for row in selected.itertuples()
    ]
    records = [{**row, "geometry": row["geometry"].intersection(municipality)} for row in records]
    records = [row for row in records if not row["geometry"].is_empty]

    overlaps = find_overlaps(records, ["CLAVE", "DESCRIPCIO"])
    conflicts = [row for row in overlaps if row["contradictory_values"]]
    if conflicts:
        raise ValueError(f"Hay {len(conflicts)} solapamientos con CLAVE/DESCRIPCIO contradictorios; rasterización detenida.")

    mapping, catalog = build_category_catalog(records)
    array, mapping = rasterize_land_cover(records, {"transform": transform}, inside_mask)
    municipal_area = float(municipality.area)
    stats = _land_cover_statistics(records, array, inside_mask, municipal_area, catalog)
    vector_coverage = coverage_pct([row["geometry"] for row in records], municipality)
    geometry_report = geometry_metrics([row["geometry"] for row in records])
    valid_cells = int(np.count_nonzero(inside_mask & (array != NODATA)))
    if not np.all(array[~inside_mask] == NODATA):
        raise AssertionError("Se encontraron datos fuera de la máscara municipal.")

    assert_new_output_dir(out)
    tif_path = out / "uso_suelo_vegetacion_villa_de_alvarez.tif"
    png_path = out / "uso_suelo_vegetacion_villa_de_alvarez.png"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=EXPECTED_WIDTH,
        height=EXPECTED_HEIGHT,
        count=1,
        dtype="uint16",
        crs=reference_profile["crs"],
        transform=transform,
        nodata=NODATA,
        compress="deflate",
    ) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, "Uso de suelo y vegetación Serie VII INEGI")
        dst.update_tags(
            source=str(source_path),
            source_layer=source_info["layer_name"],
            category_code_field="CLAVE",
            category_description_field="DESCRIPCIO",
            rasterization="categorical; pixel center; all_touched=False; no interpolation",
            nodata_code=str(NODATA),
            category_table="catalogo_categorias_uso_suelo.csv",
            source_scale="1:250,000",
        )

    for row in stats:
        row["inegi_category_name"] = row["DESCRIPCIO"]
    _write_map(array, inside_mask, transform, boundary, stats, "Uso de suelo y vegetación — Villa de Álvarez", png_path)
    pd.DataFrame(catalog).to_csv(out / "catalogo_categorias_uso_suelo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stats).to_csv(out / "estadisticas_superficie_uso_suelo.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "municipality": "Villa de Álvarez, Colima",
        "source": {
            "path": str(source_path), "layer": source_info["layer_name"], "format": source_info["driver"],
            "geometry": source_info["geometry_type"], "crs_original_wkt": source_info["crs"],
            "national_feature_count_metadata": 179099, "fields_used": ["CLAVE", "DESCRIPCIO"],
            "series": "INEGI Uso de Suelo y Vegetación Serie VII", "scale": "1:250,000",
            "metadata_path": str(metadata_path), "dictionary_path": str(dictionary_path),
            "metadata_modified": "2021-11-19", "imagery_base_year": 2018,
        },
        "grid": {
            "reference_path": str(grid_path), "crs": "EPSG:6368", "resolution_m": list(EXPECTED_RESOLUTION),
            "width": EXPECTED_WIDTH, "height": EXPECTED_HEIGHT, "transform": list(transform),
            "municipal_mask_cells": int(inside_mask.sum()), "valid_category_cells": valid_cells,
            "coverage_pct_cells": 100 * valid_cells / int(inside_mask.sum()), "mask_matches_reference": True,
        },
        "processing": {
            "spatial_selection": "source-CRS municipal bbox followed by exact municipal intersection; no national processing",
            "selected_polygon_count": len(records), "categories": len(catalog),
            "invalid_geometries_repaired_in_memory": repairs, "geometry_audit_selected": geometry_report,
            "overlap_pairs_by_field": overlaps, "contradictory_overlaps": len(conflicts),
            "overlap_policy": "contradictory positive-area overlaps stop processing; identical categories are value-invariant; pixel-center burn ordered by feature id",
            "nodata_code": NODATA, "inside_mask_nodata_cells": int(inside_mask.sum()) - valid_cells,
            "resampling": "none", "uncovered_areas_filled": False,
            "vector_coverage": vector_coverage,
            "surface_method": "vector union area and raster pixel count × 225 m² reported separately",
        },
        "products": {
            "geotiff": str(tif_path), "map_png": str(png_path),
            "category_catalog_csv": str(out / "catalogo_categorias_uso_suelo.csv"),
            "surface_statistics_csv": str(out / "estadisticas_superficie_uso_suelo.csv"),
            "report_md": str(out / "informe_procesamiento_uso_suelo.md"),
        },
    }
    (out / "manifiesto_uso_suelo.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    coverage_rows = "\n".join(
        f"| {row['CLAVE']} | {row['DESCRIPCIO']} | {row['polygon_count']} | {row['vector_area_km2']:.6f} | {row['vector_pct_municipality']:.4f}% | {row['raster_cells']} | {row['raster_area_km2']:.6f} | {row['raster_pct_municipal_cells']:.4f}% |"
        for row in stats
    )
    invalid_note = "No se repararon geometrías inválidas." if not repairs else "Reparaciones en memoria (áreas antes/después y deltas) registradas en manifiesto; los originales permanecen intactos."
    overlap_note = "No se encontraron solapamientos positivos de área." if not overlaps else f"Se detectaron {len(overlaps)} filas de solapamiento; conflictos contradictorios: {len(conflicts)}."
    report = f'''# Procesamiento de Uso de Suelo y Vegetación — Villa de Álvarez

## Fuente y método

INEGI, Uso de Suelo y Vegetación Serie VII, Conjunto Nacional; escala temática original **1:250,000**. La documentación local señala imágenes con año base 2018 y actualización temporal 2017–2021. Fuente: `{source_path}`; metadato: `{metadata_path}`; diccionario: `{dictionary_path}`.

Se usaron los campos originales `CLAVE` y `DESCRIPCIO`. Se leyeron únicamente entidades dentro del bbox municipal en el CRS original, seguido de intersección exacta con Villa de Álvarez. Se seleccionaron {len(records)} polígonos y {len(catalog)} categorías. No se procesó el conjunto nacional completo.

La cuadrícula se reutilizó de `{grid_path}`: EPSG:6368, 15 m, {EXPECTED_WIDTH} × {EXPECTED_HEIGHT}, transformación afín idéntica y máscara municipal de {int(inside_mask.sum()):,} celdas. La rasterización usó identificadores enteros reproducibles, centro de píxel y ninguna interpolación. El NoData interno se preservó; no se rellenaron huecos.

**La rasterización a 15 m no incrementa la precisión cartográfica de la fuente a escala 1:250,000.**

## Categorías y superficies

| CLAVE | DESCRIPCIO | Polígonos | Vector km² | Vector % municipal | Celdas | Ráster km² | % celdas municipales |
|---|---|---:|---:|---:|---:|---:|---:|
{coverage_rows}

Superficie vectorial por categoría calculada con unión geométrica dentro del límite municipal. Superficie ráster calculada como celdas × 225 m²; ambas cifras se mantienen separadas.

## Calidad y limitaciones

- Geometría: {geometry_report['features']} polígonos seleccionados; inválidos tras el tratamiento: {geometry_report['invalid']}; vacíos/nulos: {geometry_report['null_or_empty']}.
- Tratamiento de inválidos: {invalid_note}
- Solapamientos: {overlap_note}
- Cobertura ráster municipal: {valid_cells:,}/{int(inside_mask.sum()):,} celdas ({100*valid_cells/int(inside_mask.sum()):.4f}%). El NoData interior se conserva y las zonas sin información no se completan.
- La clase categórica describe el uso/cobertura de la fuente; no se tradujo a infiltración, escurrimiento o susceptibilidad.
- La resolución de 15 m es la cuadrícula de salida, no la precisión temática original.
'''
    (out / "informe_procesamiento_uso_suelo.md").write_text(report, encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = process_land_cover(args.source, args.metadata, args.dictionary, args.boundary, args.grid, args.output_dir)
    print(f"Productos: {args.output_dir}")
    print(f"Polígonos: {manifest['processing']['selected_polygon_count']}; categorías: {manifest['processing']['categories']}")


if __name__ == "__main__":
    main()
