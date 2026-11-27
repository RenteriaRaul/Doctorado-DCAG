"""Process INEGI soil classes onto the existing Villa de Álvarez grid.

Only municipal-intersecting polygons are read and rasterized. Source vectors,
the existing physical-factor grid, and the complementary profile points remain
unchanged. No point interpolation or susceptibility scoring is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from rasterio.features import geometry_mask, rasterize
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from scripts.edaphology_audit import coverage_pct, geometry_metrics


EPSG = 6368
NODATA = 65535
EXPECTED_WIDTH = 2674
EXPECTED_HEIGHT = 1552
EXPECTED_RESOLUTION = (15.0, 15.0)
PIXEL_AREA_M2 = 225.0
REGIONAL_SUBBASINS = {"RH15Aa", "RH15Ab", "RH16Ba", "RH16Bb", "RH16Bc"}
PROFILE_RADIUS_M = 15000.0

VARIABLES = {
    "suelo_dominante": {"field": "Grupo1", "title": "Suelo dominante", "definition": "Código del grupo de suelo dominante (al menos 60% de la unidad).", "domain": "Grupos de suelo INEGI/WRB; códigos conservados."},
    "suelo_secundario": {"field": "Grupo2", "title": "Suelo secundario", "definition": "Código del grupo secundario (estimado entre 20% y 40% de la unidad).", "domain": "Grupos de suelo INEGI/WRB; N/N/A conservados como categorías especiales."},
    "textura_superficial": {"field": "Clase_tex", "title": "Textura superficial", "definition": "Textura de los primeros 30 cm del suelo dominante.", "domain": "1 gruesa; 2 media; 3 fina; otros códigos/sentinelas preservados."},
    "limitante_fisica": {"field": "Lmte_sup", "title": "Limitante física superficial", "definition": "Fragmentos de roca/mineral estimados en más de 30% del área.", "domain": "P pedregosa; G gravosa; N/N/A preservados."},
    "fase_fisica": {"field": "Fase_fis_u", "title": "Fase física", "definition": "Material duro, roca continua o cementación en algún punto entre 0 y 100 cm.", "domain": "Códigos INEGI de fase física; N/N/A preservados."},
    "fase_quimica": {"field": "Fase_qui_u", "title": "Fase química", "definition": "Salinidad, sodicidad o saturación de sodio más magnesio.", "domain": "Códigos INEGI de fase química; N conservado."},
}

ATTRIBUTE_CROSSWALK = [
    ("Grupo1", "Suelo dominante", "area", "Código del grupo que ocupa al menos 60% de la unidad", "rasterizado"),
    ("Grupo2", "Suelo secundario", "area", "Grupo estimado entre 20% y 40% de la unidad", "rasterizado"),
    ("Grupo3", "Suelo terciario", "area", "Grupo estimado hasta 20% de la unidad", "catálogo; sin GeoTIFF solicitado"),
    ("Califp_g1", "Calificador primario dominante", "area", "Cualidad dominante del grupo principal", "catálogo; sin GeoTIFF solicitado"),
    ("Califs_g1", "Calificador secundario dominante", "area", "Segunda cualidad del grupo principal", "catálogo; sin GeoTIFF solicitado"),
    ("Califp_g2", "Calificador primario secundario", "area", "Cualidad del grupo secundario", "catálogo; sin GeoTIFF solicitado"),
    ("Califs_g2", "Calificador secundario secundario", "area", "Segunda cualidad del grupo secundario", "catálogo; sin GeoTIFF solicitado"),
    ("Califp_g3", "Calificador primario terciario", "area", "Cualidad del grupo terciario", "catálogo; sin GeoTIFF solicitado"),
    ("Califs_g3", "Calificador secundario terciario", "area", "Segunda cualidad del grupo terciario", "catálogo; sin GeoTIFF solicitado"),
    ("Clase_tex", "Textura del suelo", "area", "Código 1 gruesa, 2 media, 3 fina; suelo dominante, 30 cm superficiales", "rasterizado"),
    ("Lmte_sup", "Limitante física superficial", "area", "Código P pedregosa o G gravosa; N/N/A con significado normativo", "rasterizado"),
    ("Fase_fis_u", "Fase física de unidad", "area", "Fase física del suelo según códigos INEGI", "rasterizado"),
    ("Fase_qui_u", "Fase química de unidad", "area", "Fase química del suelo según códigos INEGI", "rasterizado"),
    ("Infiltra", "Infiltración", "pto", "Velocidad cualitativa dentro del perfil; no es una tasa", "complementario; no extrapolar"),
    ("Escurr_sup", "Escurrimiento superficial", "pto", "Velocidad cualitativa sobre la superficie", "complementario; no extrapolar"),
    ("DA", "Densidad aparente", "pto", "Masa seca por volumen; valor analítico por horizonte", "complementario; validar unidades"),
    ("R", "Arena", "pto", "Porcentaje de partículas de 0.2 a 2 mm", "complementario; validar horizonte"),
    ("L", "Limo", "pto", "Porcentaje de partículas >0.002 y ≤0.2 mm", "complementario; validar horizonte"),
    ("A", "Arcilla", "pto", "Porcentaje de partículas ≤0.002 mm", "complementario; validar horizonte"),
    ("Clase_text", "Clase textural de laboratorio", "pto", "Clase textural de la tierra fina por horizonte", "complementario; validar horizonte"),
]

GROUP_NAMES = {"AC":"Acrisol","AL":"Alisol","AN":"Andosol","AT":"Anthrosol","AR":"Arenosol","CL":"Calcisol","CM":"Cambisol","CH":"Chernozem","CR":"Cryosol","DU":"Durisol","FR":"Ferralsol","FL":"Fluvisol","GL":"Gleysol","GY":"Gypsisol","HS":"Histosol","KS":"Kastanozem","LP":"Leptosol","LX":"Lixisol","LV":"Luvisol","NT":"Nitisol","PH":"Phaeozem","PL":"Planosol","PT":"Plinthosol","PZ":"Podzol","RG":"Regosol","RT":"Retisol","SC":"Solonchak","SN":"Solonetz","ST":"Stagnosol","TC":"Technosol","UM":"Umbrisol","VR":"Vertisol"}
PHYSICAL_PHASE_NAMES = {"LI":"Lítico","LEP":"Epiléptico","LEN":"Endoléptico","PDP":"Epipetrodúrico","PDN":"Endopetrodúrico","PCP":"Epipetrocálcico","PCN":"Endopetrocálcico","PGP":"Epipetrogípsico","PGN":"Endopetrogípsico"}
CHEMICAL_PHASE_NAMES = {"QZ":"Protosálica","SZ":"Sálica","JZ":"Hipersálica","QS":"Protosódica","SO":"Sódica"}


def _is_null(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _typed_value(value):
    """Stable typed key so numeric 1 and text '1' stay distinct."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    return ("str", str(value))


def category_kind(value) -> str:
    if _is_null(value) or (isinstance(value, str) and not value.strip()):
        return "nulo_real"
    text = str(value).strip().upper()
    if text in {"N", "NINGUNO"}:
        return "ausencia_indicada"
    if text in {"N/A", "NA", "NO APLICABLE"}:
        return "no_aplicable"
    if text in {"N/D", "ND", "NO DISPONIBLE", "DESCONOCIDO"}:
        return "no_disponible_o_desconocido"
    return "categoria_original"


def inegi_category_name(field, value) -> str:
    """Return a label only when the official dictionary gives a verified code."""
    if _is_null(value) or (isinstance(value, str) and not value.strip()):
        return "Nulo real"
    kind=category_kind(value)
    if kind=="ausencia_indicada": return "Ninguno"
    if kind=="no_aplicable": return "No aplicable"
    if kind=="no_disponible_o_desconocido": return "No disponible/desconocido"
    raw=str(value).strip()
    upper=raw.upper()
    if field in {"Grupo1","Grupo2","Grupo3"}:
        return GROUP_NAMES.get(upper, "Código de grupo no listado")
    if field=="Clase_tex":
        return {"1":"Gruesa","2":"Media","3":"Fina","N/D":"No disponible","N/A":"No aplicable","N":"Ninguno"}.get(upper,"Código de textura no listado")
    if field=="Lmte_sup":
        return {"P":"Pedregosa","G":"Gravosa","N":"Ninguno","N/A":"No aplicable","N/D":"No disponible"}.get(upper,"Código de limitante no listado")
    if field=="Fase_fis_u":
        return {**PHYSICAL_PHASE_NAMES,"N":"Ninguno","N/A":"No aplicable"}.get(upper,"Código de fase física no listado")
    if field=="Fase_qui_u":
        return {**CHEMICAL_PHASE_NAMES,"N":"Ninguno"}.get(upper,"Código de fase química no listado")
    if category_kind(value)=="ausencia_indicada": return "Ninguno"
    if category_kind(value)=="no_aplicable": return "No aplicable"
    if category_kind(value)=="no_disponible_o_desconocido": return "No disponible/desconocido"
    return "Etiqueta no verificada; código fuente preservado"


def stable_category_mapping(values):
    keys = {_typed_value(value) for value in values if not _is_null(value) and not (isinstance(value, str) and not value.strip())}
    ordered = sorted(keys, key=lambda item: (item[0], str(item[1])))
    if len(ordered) >= NODATA:
        raise ValueError("Demasiadas categorías para el tipo UInt16 reservado.")
    return {key: index + 1 for index, key in enumerate(ordered)}


def _polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return geom
    parts = []
    for part in getattr(geom, "geoms", ()):
        if part.geom_type == "Polygon":
            parts.append(part)
        elif part.geom_type == "MultiPolygon":
            parts.extend(part.geoms)
    return unary_union(parts) if parts else None


def prepare_polygon_records(frame, municipality, fields):
    """Repair only invalid selected features in memory and log area changes."""
    records = []
    repairs = []
    for row_index, row in frame.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        feature_id = row.get("OBJECTID", row.get("FID", row_index))
        before_area = float(geom.area)
        was_invalid = not geom.is_valid
        if was_invalid:
            geom = _polygonal(make_valid(geom))
            after_area = float(geom.area) if geom is not None else 0.0
            if geom is not None and geom.intersects(municipality):
                repairs.append({"feature_id": str(feature_id), "area_before_m2_source_crs": before_area, "area_after_m2_source_crs": after_area, "area_delta_m2_source_crs": after_area - before_area, "method": "shapely.make_valid en memoria; se conservaron solo componentes poligonales"})
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        if not geom.intersects(municipality):
            continue
        clipped = _polygonal(geom.intersection(municipality))
        if clipped is None or clipped.is_empty:
            continue
        item = {field: row.get(field) for field in fields}
        records.append({"feature_id": str(feature_id), "geometry": clipped, "properties": item, "repaired": was_invalid})
    return records, repairs


def find_overlaps(records, fields, area_tolerance_m2=1e-6):
    """List all positive-area polygon intersections; flag value conflicts."""
    found = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if not left["geometry"].intersects(right["geometry"]):
                continue
            intersection = left["geometry"].intersection(right["geometry"])
            overlap_area = float(intersection.area)
            if overlap_area <= area_tolerance_m2:
                continue
            for field in fields:
                a, b = left["properties"].get(field), right["properties"].get(field)
                conflict = _typed_value(a) != _typed_value(b)
                found.append({"feature_id_a": left["feature_id"], "feature_id_b": right["feature_id"], "field": field, "value_a": "<NULL>" if _is_null(a) else str(a), "value_b": "<NULL>" if _is_null(b) else str(b), "overlap_area_m2": overlap_area, "contradictory_values": conflict, "policy": "sin conflicto: mismo código, resultado invariante; conflicto: detener antes de rasterizar"})
    return found


def rasterize_categories(records, field, reference, inside_mask, mapping=None, nodata=NODATA):
    """Burn polygon categories by pixel center; never interpolates categories."""
    values = [r["properties"].get(field) for r in records]
    mapping = mapping or stable_category_mapping(values)
    burn = []
    for record in sorted(records, key=lambda x: x["feature_id"]):
        value = record["properties"].get(field)
        if _is_null(value) or (isinstance(value, str) and not value.strip()):
            continue
        if record["geometry"].geom_type not in {"Polygon", "MultiPolygon"}:
            raise TypeError("La rasterización categórica requiere polígonos; no extrapola puntos.")
        code = mapping[_typed_value(value)]
        burn.append((record["geometry"], code))
    transform = reference.transform if hasattr(reference, "transform") else reference["transform"]
    array = rasterize(burn, out_shape=inside_mask.shape, transform=transform, fill=nodata, dtype="uint16", all_touched=False, merge_alg=rasterio.enums.MergeAlg.replace)
    array[~inside_mask] = nodata
    return array, mapping


def validate_reference_grid(dataset, width=EXPECTED_WIDTH, height=EXPECTED_HEIGHT):
    if dataset.crs is None or dataset.crs.to_epsg() != EPSG:
        raise ValueError("La cuadrícula debe estar en EPSG:6368.")
    if (dataset.width, dataset.height) != (width, height):
        raise ValueError(f"Dimensiones inesperadas: {dataset.width} × {dataset.height}.")
    if dataset.res != EXPECTED_RESOLUTION:
        raise ValueError(f"Resolución inesperada: {dataset.res}.")
    if not math.isclose(dataset.transform.b, 0.0) or not math.isclose(dataset.transform.d, 0.0):
        raise ValueError("La cuadrícula debe estar sin rotación.")
    return dataset.transform


def assert_new_output_dir(path: Path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"No se sobrescribe una carpeta de ejecución existente: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _value_display(value):
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _category_catalog(field, mapping, values):
    by_key = {_typed_value(v): v for v in values if not _is_null(v) and not (isinstance(v, str) and not v.strip())}
    result = []
    for key, code in sorted(mapping.items(), key=lambda pair: pair[1]):
        value = by_key[key]
        result.append({"raster_code": code, "source_field": field, "source_value": str(value), "inegi_category_name": inegi_category_name(field,value), "source_value_typed_json": _value_display(value), "category_kind": category_kind(value), "nodata_code": NODATA})
    return result


def _surface_statistics(records, field, array, inside_mask, municipality_area_m2):
    counts = Counter(int(v) for v in array[inside_mask] if int(v) != NODATA)
    mapping = stable_category_mapping([r["properties"].get(field) for r in records])
    union_by_code = defaultdict(list)
    feature_counts = Counter()
    for record in records:
        value = record["properties"].get(field)
        if _is_null(value) or (isinstance(value, str) and not value.strip()):
            continue
        code = mapping[_typed_value(value)]
        union_by_code[code].append(record["geometry"])
        feature_counts[code] += 1
    rows = []
    total_cells = int(inside_mask.sum())
    for row in _category_catalog(field, mapping, [r["properties"].get(field) for r in records]):
        code = row["raster_code"]
        vector_area = float(unary_union(union_by_code[code]).area) if union_by_code[code] else 0.0
        cells = int(counts.get(code, 0))
        rows.append({"variable": field, **row, "polygon_count": int(feature_counts.get(code, 0)), "vector_area_km2": vector_area / 1_000_000, "vector_pct_municipality": vector_area / municipality_area_m2 * 100 if municipality_area_m2 else 0, "raster_cells": cells, "raster_area_km2": cells * PIXEL_AREA_M2 / 1_000_000, "raster_pct_municipal_cells": cells / total_cells * 100 if total_cells else 0})
    return rows


def _write_map(array, inside_mask, transform, boundary, category_rows, title, output_path):
    codes = sorted(int(v) for v in np.unique(array[inside_mask]) if int(v) != NODATA)
    cmap = plt.get_cmap("tab20", max(1, len(codes)))
    colors = {code: cmap(i) for i, code in enumerate(codes)}
    visual = np.ma.masked_where((array == NODATA) | ~inside_mask, array)
    west, north = transform.c, transform.f
    east = west + transform.a * array.shape[1]
    south = north + transform.e * array.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.imshow(visual, extent=(west, east, south, north), origin="upper", interpolation="nearest", cmap=matplotlib.colors.ListedColormap([colors[c] for c in codes]), vmin=min(codes) - .5 if codes else 0, vmax=max(codes) + .5 if codes else 1)
    boundary.boundary.plot(ax=ax, color="black", linewidth=.8)
    label_by_code = {int(x["raster_code"]): f"{x['source_value']} — {x['inegi_category_name']}" for x in category_rows}
    handles = [Patch(facecolor=colors[c], label=label_by_code.get(c, str(c))) for c in codes]
    if np.any(inside_mask & (array == NODATA)):
        handles.append(Patch(facecolor="white", edgecolor="black", label="NoData interior"))
    ax.legend(handles=handles, title="Código/categoría INEGI", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _profile_catalog(pto_path, source_region, municipality, boundary, output_path):
    pto_info = gpd.read_file(pto_path, rows=1)
    region_plus_near = unary_union([source_region, municipality.buffer(PROFILE_RADIUS_M)])
    source_mask = gpd.GeoSeries([region_plus_near], crs=boundary.crs).to_crs(pto_info.crs).iloc[0]
    profiles = gpd.read_file(pto_path, mask=source_mask)
    if profiles.empty:
        pd.DataFrame(columns=["Id_Sitio", "inside_municipality", "within_15km", "in_regional_support", "distance_to_boundary_m", "distance_to_centroid_m"]).to_csv(output_path, index=False, encoding="utf-8-sig")
        return 0
    projected = profiles.to_crs(boundary.crs)
    centroid = municipality.centroid
    rows = []
    for idx, row in projected.iterrows():
        point = row.geometry
        dist = float(point.distance(municipality.boundary))
        values = row.drop(labels=[projected.geometry.name]).to_dict()
        rows.append({**values, "x_epsg6368": float(point.x), "y_epsg6368": float(point.y), "inside_municipality": bool(municipality.covers(point)), "within_15km": bool(dist <= PROFILE_RADIUS_M), "in_regional_support": bool(source_region.covers(point)), "distance_to_boundary_m": dist, "distance_to_centroid_m": float(point.distance(centroid)), "source_layer": "pto", "source_path": str(pto_path)})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(rows)


def process_edaphology(area_path: Path, pto_path: Path, boundary_path: Path, grid_path: Path, audit_dir: Path, output_dir: Path):
    """Create six categorical GeoTIFFs and audit products in a new run folder."""
    area_path, pto_path, boundary_path, grid_path, audit_dir = map(Path, (area_path, pto_path, boundary_path, grid_path, audit_dir))
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"No se sobrescriben resultados: {out}")
    audit_coverage_path=audit_dir/"cobertura_edafologica.json"
    if not audit_coverage_path.is_file():
        raise FileNotFoundError(f"Falta el resultado de auditoría existente: {audit_coverage_path}")
    previous_audit=json.loads(audit_coverage_path.read_text(encoding="utf-8"))
    boundary = gpd.read_file(boundary_path, layer="limite_municipal")
    if len(boundary) != 1 or boundary.crs is None or boundary.crs.to_epsg() != EPSG or not boundary.geometry.iloc[0].is_valid:
        raise ValueError("El límite municipal debe ser un polígono válido en EPSG:6368.")
    municipality = boundary.geometry.iloc[0]
    with rasterio.open(grid_path) as reference:
        transform = validate_reference_grid(reference)
        reference_profile = reference.profile.copy()
        reference_mask = reference.read_masks(1) > 0
    inside_mask = geometry_mask([municipality.__geo_interface__], out_shape=(EXPECTED_HEIGHT, EXPECTED_WIDTH), transform=transform, invert=True, all_touched=False)
    if not np.array_equal(inside_mask, reference_mask):
        raise ValueError("La máscara geométrica municipal no coincide con la máscara del ráster físico de referencia.")

    physical_dir = Path(grid_path).parent
    companion_rasters = [p for p in physical_dir.glob("*.tif")]
    for path in companion_rasters:
        with rasterio.open(path) as dataset:
            validate_reference_grid(dataset)
            if dataset.transform != transform or not np.array_equal(dataset.read_masks(1) > 0, reference_mask):
                raise ValueError(f"El factor físico no comparte cuadrícula/máscara: {path}")

    area_info = gpd.read_file(area_path, rows=1)
    municipality_source = gpd.GeoSeries([municipality], crs=boundary.crs).to_crs(area_info.crs).iloc[0]
    bbox = municipality_source.bounds
    area_candidates = gpd.read_file(area_path, bbox=bbox)
    source_fields = list(area_candidates.columns)
    pto_fields = list(gpd.read_file(pto_path, rows=1).columns)
    required_fields = [spec["field"] for spec in VARIABLES.values()]
    missing = [field for field in required_fields if field not in source_fields]
    if missing:
        raise KeyError(f"Faltan campos fuente requeridos: {missing}")
    records, repairs = prepare_polygon_records(area_candidates, municipality_source, required_fields)
    if not records:
        raise ValueError("No se encontraron polígonos edafológicos municipales.")
    # Project only the already selected, municipal-intersecting features in memory.
    selected = gpd.GeoDataFrame(records, geometry="geometry", crs=area_info.crs).to_crs(boundary.crs)
    records = [{"feature_id": row.feature_id, "geometry": row.geometry, "properties": row.properties, "repaired": row.repaired} for row in selected.itertuples()]
    records = [{**r, "geometry": _polygonal(r["geometry"].intersection(municipality))} for r in records]
    records = [r for r in records if r["geometry"] is not None and not r["geometry"].is_empty]
    invalid_report = geometry_metrics([r["geometry"] for r in records])

    overlaps = find_overlaps(records, required_fields)
    conflicts = [row for row in overlaps if row["contradictory_values"]]
    assert_new_output_dir(out)
    overlap_path = out / "diagnostico_solapamientos.csv"
    overlap_columns = ["feature_id_a", "feature_id_b", "field", "value_a", "value_b", "overlap_area_m2", "contradictory_values", "policy"]
    if not overlaps:
        pd.DataFrame([{"feature_id_a":"", "feature_id_b":"", "field":"", "value_a":"", "value_b":"", "overlap_area_m2":0, "contradictory_values":False, "policy":"Sin solapamientos poligonales de área positiva dentro del municipio."}], columns=overlap_columns).to_csv(overlap_path,index=False,encoding="utf-8-sig")
    else:
        pd.DataFrame(overlaps, columns=overlap_columns).to_csv(overlap_path,index=False,encoding="utf-8-sig")
    if conflicts:
        raise ValueError(f"Se detectaron {len(conflicts)} solapamientos con categorías contradictorias; se detuvo antes de rasterizar. Revisar {overlap_path}")

    category_rows=[]
    stats_rows=[]
    correspondence=[]
    nodata_by_field={}
    for field, concept, layer, definition, status in ATTRIBUTE_CROSSWALK:
        available_fields=source_fields if layer=="area" else pto_fields
        present=field in available_fields
        correspondence.append({"source_layer":layer,"source_field":field,"inegi_concept":concept,"definition_or_domain":definition,"field_present":present,"processing_status":status,"dictionary":"INEGI Diccionario de Datos Edafológicos 1:250 000 v4 (2016)"})

    for variable, spec in VARIABLES.items():
        field=spec["field"]
        values=[r["properties"].get(field) for r in records]
        mapping=stable_category_mapping(values)
        array, mapping=rasterize_categories(records,field,reference_profile,inside_mask,mapping)
        catalog=_category_catalog(field,mapping,values)
        category_rows.extend([{"variable":variable,"variable_title":spec["title"],**row} for row in catalog])
        stats_rows.extend(_surface_statistics(records,field,array,inside_mask,float(municipality.area)))
        raster_path=out/f"{variable}_villa_de_alvarez.tif"
        png_path=out/f"{variable}_villa_de_alvarez.png"
        if raster_path.exists() or png_path.exists():
            raise FileExistsError(f"No se sobrescriben resultados: {raster_path} / {png_path}")
        profile={"driver":"GTiff","width":EXPECTED_WIDTH,"height":EXPECTED_HEIGHT,"count":1,"dtype":"uint16","crs":reference_profile["crs"],"transform":transform,"nodata":NODATA,"compress":"deflate"}
        with rasterio.open(raster_path,"w",**profile) as dst:
            dst.write(array,1)
            dst.set_band_description(1,spec["title"])
            dst.update_tags(source_path=str(area_path),source_layer="area",source_field=field,dictionary="INEGI Edafológico 1:250 000 v4 (2016)",rasterization="categorical; pixel center; all_touched=False; no interpolation",mask="existing municipal grid and EPSG:6368 boundary",nodata_semantics="actual null/blank attributes only; N/N-A/N-D tokens preserved as categories",category_table="catalogo_categorias_edafologicas.csv")
        _write_map(array,inside_mask,transform,boundary,catalog,spec["title"],png_path)
        nodata_by_field[field]=int(np.count_nonzero(inside_mask & (array==NODATA)))

    pd.DataFrame(category_rows).to_csv(out/"catalogo_categorias_edafologicas.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(stats_rows).to_csv(out/"estadisticas_superficie_edafologica.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(correspondence).to_csv(out/"correspondencia_atributos_inegi.csv",index=False,encoding="utf-8-sig")

    sub=gpd.read_file(boundary_path,layer="subcuencas_regionales")
    sub_selected=sub.loc[sub["CVE_SUBCUE"].isin(REGIONAL_SUBBASINS)]
    regional=unary_union(list(sub_selected.geometry)).buffer(2000)
    profile_count=_profile_catalog(pto_path,regional,municipality,boundary,out/"catalogo_perfiles_complementarios.csv")
    final_mask_reports={}
    for variable,spec in VARIABLES.items():
        path=out/f"{variable}_villa_de_alvarez.tif"
        with rasterio.open(path) as dst:
            arr=dst.read(1)
            final_mask_reports[variable]={"crs":dst.crs.to_string(),"width":dst.width,"height":dst.height,"resolution_m":list(dst.res),"transform":list(dst.transform),"nodata":dst.nodata,"municipal_mask_cells":int(inside_mask.sum()),"valid_category_cells":int(np.count_nonzero(inside_mask & (arr!=NODATA))),"outside_mask_values_are_nodata":bool(np.all(arr[~inside_mask]==NODATA)),"values_within_catalog":bool(set(np.unique(arr[inside_mask]).tolist())- {NODATA} <= set(row["raster_code"] for row in category_rows if row["source_field"]==spec["field"]))}
    vector_coverage=coverage_pct([r["geometry"] for r in records],municipality)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"project":"Doctorado-DCAG","municipality":"Villa de Álvarez, Colima","source":{"area_path":str(area_path),"area_layer":"area","pto_path":str(pto_path),"pto_layer":"pto","metadata":"Serie III, INEGI, escala 1:250 000, WRB 2014; Diccionario v4 (2016)"},"prior_audit_reused":{"coverage_json":str(audit_coverage_path),"profiles_inside_municipality":previous_audit.get("profiles_inside_municipality"),"profiles_in_regional_support":previous_audit.get("profiles_in_regional_support")},"grid_reference":{"path":str(grid_path),"crs":"EPSG:6368","resolution_m":list(EXPECTED_RESOLUTION),"width":EXPECTED_WIDTH,"height":EXPECTED_HEIGHT,"transform":list(transform),"municipal_mask_cells":int(inside_mask.sum()),"mask_matches_reference":True,"existing_physical_rasters_checked":[str(p) for p in companion_rasters]},"processing":{"spatial_selection":"municipal bounding box followed by exact intersection; selected features only","selected_polygon_count":len(records),"municipality_area_km2":float(municipality.area/1_000_000),"vector_coverage":vector_coverage,"invalid_geometries_selected_and_repaired":repairs,"post_repair_geometry_audit":invalid_report,"overlap_rows":len(overlaps),"contradictory_overlaps":len(conflicts),"overlap_policy":"positive-area contradictory overlaps block rasterization; identical category overlap is value-invariant; pixel-center burn; sort by feature id","nodata_by_source_field_inside_municipality":nodata_by_field,"categorical_interpolation":"none","point_interpolation":"none","profile_features_catalogued":profile_count,"raster_validation":final_mask_reports},"products":[{"variable":name,"field":spec["field"],"geotiff":str(out/f"{name}_villa_de_alvarez.tif"),"png":str(out/f"{name}_villa_de_alvarez.png")} for name,spec in VARIABLES.items()]}
    (out/"manifiesto_edafologia.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    invalid_details="\n".join(f"- {r['feature_id']}: área fuente antes={r['area_before_m2_source_crs']:.9f} m²; después={r['area_after_m2_source_crs']:.9f} m²; cambio={r['area_delta_m2_source_crs']:.9f} m²; método={r['method']}." for r in repairs) or "- No se detectaron geometrías inválidas entre los polígonos seleccionados para el municipio."
    total_categories=len(category_rows)
    report=f'''# Procesamiento de factores edafológicos — Villa de Álvarez

## Fuentes y cuadrícula

Fuente: {area_path} (`area`); perfiles complementarios: {pto_path} (`pto`). Escala 1:250 000, Serie III. Se seleccionaron solo entidades que intersectan el municipio. La cuadrícula se reutilizó de `{grid_path}`: EPSG:6368, 15 m, {EXPECTED_WIDTH} × {EXPECTED_HEIGHT}, transformación afín idéntica a los seis rásteres físicos existentes. Máscara municipal: {int(inside_mask.sum()):,} celdas. No se reconstruyeron factores ni se modificaron fuentes.

## Rásteres categóricos

Se generaron seis GeoTIFF y seis mapas PNG: suelo dominante (`Grupo1`), secundario (`Grupo2`), textura superficial (`Clase_tex`), limitante superficial (`Lmte_sup`), fase física (`Fase_fis_u`) y fase química (`Fase_qui_u`). Las categorías originales se codifican mediante identificadores enteros documentados en el catálogo; `N`, `N/A`, `N/D`, valores no disponibles y nulos reales no se fusionan. Nulos reales se representan como NoData. Rasterización por centro de píxel, sin interpolación.

Las superficies vectoriales (unión de polígonos recortados por categoría) y áreas rasterizadas (número de celdas × 225 m²) se reportan por separado. Los porcentajes se calculan respecto del área municipal y respecto de celdas interiores, respectivamente. Consulte `estadisticas_superficie_edafologica.csv`.

## Calidad geométrica y solapamientos

Geometrías inválidas seleccionadas y tratamiento:
{invalid_details}

Solapamientos positivos de área: {len(overlaps)} combinaciones campo/par. Contradictorios: {len(conflicts)}. Los contactos de borde no se consideran solapamiento. La política es detenerse si un solapamiento espacial tiene clases incompatibles; solapamientos con igual clase son invariantes. Rasterización con orden determinista por identificador de entidad.

## Perfiles

Se catalogaron {profile_count} perfiles dentro del soporte regional hidrológico o a ≤15 km del límite municipal. La auditoría previa documentó cero perfiles dentro del municipio; se mantienen como contexto regional y no se interpolan ni se asignan al territorio municipal.

## Correspondencia y límites de uso

Las correspondencias de campos y definiciones están en `correspondencia_atributos_inegi.csv`; claves de códigos y tipo semántico en `catalogo_categorias_edafologicas.csv`. Se conservan Grupo3 y calificadores en la correspondencia, pero no se generan sus rásteres porque no forman parte de los seis GeoTIFF nombrados para esta entrega. Infiltración y escurrimiento de `pto` son descripciones categóricas cualitativas y no se convierten a valores numéricos. Los campos de perfil por horizonte no se extrapolan.

La escala 1:250 000 no representa precisión espacial de 15 m. La celda de 15 m solo fija el soporte/alineación de salida. No se asignaron susceptibilidades ni ponderaciones.
'''
    (out/"informe_procesamiento_edafologico.md").write_text(report,encoding="utf-8")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area",type=Path,required=True); p.add_argument("--pto",type=Path,required=True)
    p.add_argument("--boundary",type=Path,required=True); p.add_argument("--grid",type=Path,required=True)
    p.add_argument("--audit-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    manifest=process_edaphology(args.area,args.pto,args.boundary,args.grid,args.audit_dir,args.output_dir)
    print(f"Productos edafológicos: {args.output_dir}")
    print(f"Rásteres: {len(manifest['products'])}; perfiles complementarios: {manifest['processing']['profile_features_catalogued']}")


if __name__=="__main__":
    main()
