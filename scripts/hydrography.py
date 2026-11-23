"""Procesamiento reproducible de la Red Hidrográfica INEGI para Villa de Álvarez.

Conserva las fuentes originales, almacena redes regionales sin recortar y crea
capas municipales derivadas. No calcula distancias, HAND, TWI ni susceptibilidad.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from scripts.gis_manager import GISManager


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "datos_territoriales" / "INEGI" / "Red_Hidrografica" / "originales"
DEFAULT_MARCO_DIR = Path(r"C:\Users\rente\Desktop\Marco_geoestadistico_2025")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez"
TARGET_CRS = "EPSG:6368"
EXPECTED_SUBBASINS = 15
SUFFIXES = ("_hl", "_ha", "_dr", "_subc", "_to")
LAYER_NAMES = {
    "_hl": "corrientes_regionales",
    "_ha": "cuerpos_agua_regionales",
    "_dr": "drenaje_regional",
    "_subc": "subcuencas_regionales",
    "_to": "toponimia_regional",
}
ENABLED_TOKENS = {"true", "enabled", "habilitado", "habilitada", "si", "yes"}
DISABLED_TOKENS = {"false", "disabled", "deshabilitado", "deshabilitada", "no"}


def _normalizar_texto(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def _codigo_numerico(value):
    """Normaliza enteros/cadenas numéricas sin asignarles semántica adicional."""
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value))):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def clasificar_hidrografia(tipo, entidad, fc, condicion):
    """Clasifica usando la combinación conjunta de dominios INEGI.

    Los cuatro atributos deben coincidir con una combinación documentada.
    Etiquetas parciales, valores desconocidos y combinaciones inconsistentes
    quedan en ``desconocida``.
    """
    tipo_code = _codigo_numerico(tipo)
    fc_code = _codigo_numerico(fc)
    entity_text = _normalizar_texto(entidad) if entidad is not None else ""
    condition_text = _normalizar_texto(condicion) if condicion is not None else ""

    if tipo_code == 101 and entity_text == "corriente de agua":
        if fc_code == 3271 and condition_text in {"intermitente", "corriente de agua intermitente"}:
            return "corriente_intermitente"
        if fc_code == 3272 and condition_text in {"perenne", "corriente de agua perenne"}:
            return "corriente_perenne"
        return "desconocida"

    if tipo_code == 102 and entity_text == "canal":
        channel_combinations = {
            (3180, "en operacion"),
            (3181, "en construccion"),
            (3182, "fuera de uso"),
        }
        return "canal" if (fc_code, condition_text) in channel_combinations else "desconocida"

    if tipo_code == 103 and entity_text == "linea central" and fc_code == 3273:
        # INEGI documenta línea central de cuerpo de agua; la fuente inspeccionada
        # denomina esta categoría FLUJO VIRTUAL. Se aceptan ambas etiquetas exactas.
        if condition_text in {"flujo virtual", "linea central de cuerpo de agua"}:
            return "linea_flujo_virtual"
    return "desconocida"


def clasificar_enabled(value):
    """Interpreta ENABLED conforme al dominio INEGI: 0 falso, 1 verdadero."""
    if isinstance(value, (bool, np.bool_)):
        return "habilitada" if bool(value) else "deshabilitada"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "desconocida"
    numeric = _codigo_numerico(value)
    if numeric == 0:
        return "deshabilitada"
    if numeric == 1:
        return "habilitada"
    text = _normalizar_texto(value)
    if text in ENABLED_TOKENS:
        return "habilitada"
    if text in DISABLED_TOKENS:
        return "deshabilitada"
    return "desconocida"


def _attribute_fields(columns):
    normalized = {_normalizar_texto(c).replace(" ", ""): c for c in columns}
    return {
        "tipo": normalized.get("tipo"),
        "entidad": normalized.get("entidad"),
        "fc": normalized.get("fc"),
        "condicion": normalized.get("condicion"),
        "enabled": normalized.get("enabled"),
        "lengthm": normalized.get("lengthm"),
    }


def _codigo_subcuenca(stem, suffix=None):
    match = re.search(r"RH\d{2}[A-Za-z]{2}", stem, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    if suffix and stem.casefold().endswith(suffix.casefold()):
        stem = stem[:-len(suffix)]
    return stem


def descubrir_archivos_hidrograficos(source_dir):
    """Descubre los shapefiles por sufijo, sin asumir un orden de carpetas."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de fuentes: {source_dir}")
    found = {suffix: [] for suffix in SUFFIXES}
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".shp"):
        stem_lower = path.stem.casefold()
        for suffix in SUFFIXES:
            if stem_lower.endswith(suffix):
                found[suffix].append(path)
                break
    missing = [s for s, paths in found.items() if not paths]
    if missing:
        raise FileNotFoundError(f"No se encontraron shapefiles para sufijos: {missing}")
    return found


def validar_inventario_hidrografico(discovered, expected_subbasins=EXPECTED_SUBBASINS):
    """Exige un shapefile por cada combinación subcuenca/sufijo esperada."""
    files_by_pair = defaultdict(list)
    all_ids = set()
    for suffix, paths in discovered.items():
        for path in paths:
            basin = _codigo_subcuenca(Path(path).stem, suffix)
            all_ids.add(basin)
            files_by_pair[(basin, suffix)].append(str(path))
    duplicates = {
        f"{basin}{suffix}": paths
        for (basin, suffix), paths in files_by_pair.items()
        if len(paths) > 1
    }
    missing = [
        f"{basin}{suffix}"
        for basin in sorted(all_ids)
        for suffix in SUFFIXES
        if (basin, suffix) not in files_by_pair
    ]
    report = {
        "archivos_detectados": sum(len(paths) for paths in discovered.values()),
        "archivos_esperados": expected_subbasins * len(SUFFIXES),
        "subcuencas_detectadas": len(all_ids),
        "subcuencas_esperadas": expected_subbasins,
        "archivos_por_subcuenca_y_sufijo": {
            f"{basin}{suffix}": paths for (basin, suffix), paths in sorted(files_by_pair.items())
        },
        "duplicados": duplicates,
        "faltantes": missing,
    }
    if (
        report["archivos_detectados"] != report["archivos_esperados"]
        or len(all_ids) != expected_subbasins
        or duplicates
        or missing
    ):
        raise ValueError("Inventario hidrográfico incompleto o duplicado: " + json.dumps(report, ensure_ascii=False))
    return report


def _repair_geometry(geometry):
    if geometry is None or geometry.is_empty or geometry.is_valid:
        return geometry, False
    try:
        from shapely.validation import make_valid
    except ImportError as exc:
        raise RuntimeError("Hay geometrías inválidas y Shapely no expone make_valid().") from exc
    repaired = make_valid(geometry)
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        raise ValueError("No fue posible reparar una geometría inválida.")
    return repaired, True


def _extract_dimension(geometry, dimension):
    if geometry is None or geometry.is_empty:
        return None
    accepted = {
        "line": (LineString, MultiLineString),
        "polygon": (Polygon, MultiPolygon),
        "point": (Point, MultiPoint),
    }[dimension]
    if isinstance(geometry, accepted):
        return geometry
    if isinstance(geometry, GeometryCollection):
        parts = []
        for part in geometry.geoms:
            extracted = _extract_dimension(part, dimension)
            if extracted is not None:
                if isinstance(extracted, (MultiLineString, MultiPolygon, MultiPoint)):
                    parts.extend(list(extracted.geoms))
                else:
                    parts.append(extracted)
        if not parts:
            return None
        if dimension == "line":
            return MultiLineString(parts)
        if dimension == "polygon":
            return MultiPolygon(parts)
        return MultiPoint(parts)
    return None


EXPECTED_DIMENSION = {"_hl": "line", "_ha": "polygon", "_dr": "point", "_subc": "polygon", "_to": "point"}


def _read_and_prepare(path, suffix, target_crs=TARGET_CRS):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"La capa no declara CRS; no se asignará uno por suposición: {path}")
    source_crs = gdf.crs.to_string()
    source_subbasin = _codigo_subcuenca(path.stem, suffix)
    gdf = gdf.copy()
    empty_count = int(gdf.geometry.is_empty.sum()) if not gdf.empty else 0
    null_count = int(gdf.geometry.isna().sum()) if not gdf.empty else 0
    invalid_count = int((~gdf.geometry.is_valid & ~gdf.geometry.isna() & ~gdf.geometry.is_empty).sum()) if not gdf.empty else 0
    if not gdf.empty:
        fixed = []
        repair_flags = []
        repaired_count = 0
        for geom in gdf.geometry:
            geom, repaired = _repair_geometry(geom)
            repaired_count += int(repaired)
            repair_flags.append(bool(repaired))
            fixed.append(geom)
        gdf.set_geometry(gpd.GeoSeries(fixed, index=gdf.index, crs=gdf.crs), inplace=True)
    else:
        repaired_count = 0
    gdf = gdf.to_crs(target_crs)
    records = gdf.copy()
    records["_dcag_source_file"] = str(path)
    records["_dcag_subbasin"] = source_subbasin
    records["_dcag_source_crs"] = source_crs
    records["_dcag_source_fid"] = [str(i) for i in records.index]
    records["_dcag_source_suffix"] = suffix
    records["_dcag_repaired"] = repair_flags if not gdf.empty else []

    expected_types = {
        "line": (LineString, MultiLineString),
        "polygon": (Polygon, MultiPolygon),
        "point": (Point, MultiPoint),
    }[EXPECTED_DIMENSION[suffix]]
    invalid_after = int((~records.geometry.is_valid & ~records.geometry.isna() & ~records.geometry.is_empty).sum()) if not records.empty else 0
    wrong_after = [
        (idx, geom.geom_type)
        for idx, geom in records.geometry.items()
        if geom is not None and not geom.is_empty and not isinstance(geom, expected_types)
    ]
    if invalid_after or wrong_after:
        raise ValueError(
            f"Validación posterior a reparación/reproyección falló en {path}: "
            f"inválidas={invalid_after}, dimensión_incorrecta={wrong_after[:5]}"
        )

    required = EXPECTED_DIMENSION[suffix]
    wrong_types = []
    for idx, geom in records.geometry.items():
        if geom is not None and not geom.is_empty and _extract_dimension(geom, required) is None:
            wrong_types.append((idx, geom.geom_type))
    if wrong_types:
        raise TypeError(f"Geometrías no compatibles con {suffix} en {path}: {wrong_types[:5]}")

    length_field = _attribute_fields(records.columns)["lengthm"]
    if suffix == "_hl":
        geometric_m = records.geometry.length.astype(float)
        records["_dcag_geom_full_m"] = geometric_m
        records["_dcag_geom_full_km"] = geometric_m / 1000.0
        if length_field:
            original_length = pd.to_numeric(records[length_field], errors="coerce")
            records["_dcag_LENGTHM_diff_m"] = geometric_m - original_length
            records["_dcag_LENGTHM_abs_diff_m"] = (geometric_m - original_length).abs()
        else:
            records["_dcag_LENGTHM_diff_m"] = np.nan
            records["_dcag_LENGTHM_abs_diff_m"] = np.nan
    return records, {"path": str(path), "sufijo": suffix, "subcuenca": source_subbasin,
                     "crs_origen": source_crs, "features": int(len(gdf)),
                     "geometrias_vacias": empty_count, "geometrias_nulas": null_count,
                     "geometrias_invalidas_entrada": invalid_count,
                     "geometrias_invalidas_despues_reparacion_y_reproyeccion": invalid_after,
                     "geometrias_dimension_incorrecta_despues_reproyeccion": len(wrong_after),
                     "metodo_reparacion": "shapely.validation.make_valid solo para inválidas; reproyección con GeoPandas.to_crs",
                     "geometrias_reparadas": repaired_count,
                     "campos_originales": [str(c) for c in gdf.columns if not str(c).startswith("_dcag_")],
                     "campos_clasificacion": _attribute_fields(gdf.columns)}


def _concat_layers(records):
    nonempty = [g for g in records if g is not None]
    if not nonempty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    return gpd.GeoDataFrame(pd.concat(nonempty, ignore_index=True, sort=False), geometry="geometry", crs=TARGET_CRS)


def identificar_subcuencas(subcuencas, municipio_geom):
    """Devuelve todas las subcuencas con intersección y métricas de cobertura."""
    if subcuencas.crs is None:
        raise ValueError("Las subcuencas deben tener CRS definido.")
    gdf = subcuencas.to_crs(TARGET_CRS).copy()
    municipio_area_km2 = float(municipio_geom.area) / 1_000_000.0
    intersected = []
    intersection_areas = []
    for geom in gdf.geometry:
        hit = geom is not None and not geom.is_empty and geom.intersects(municipio_geom)
        intersected.append(bool(hit))
        intersection_areas.append(float(geom.intersection(municipio_geom).area) / 1_000_000.0 if hit else 0.0)
    gdf["_dcag_intersects_municipio"] = intersected
    gdf["_dcag_intersection_km2"] = intersection_areas
    gdf["_dcag_intersection_pct_municipio"] = [100.0 * a / municipio_area_km2 if municipio_area_km2 else 0.0 for a in intersection_areas]
    selected = gdf.loc[gdf["_dcag_intersects_municipio"]].copy()
    sum_individual = float(sum(intersection_areas))
    intersection_union = gpd.GeoSeries(
        [geom.intersection(municipio_geom) for geom in gdf.geometry if geom is not None and geom.intersects(municipio_geom)],
        crs=TARGET_CRS,
    ).union_all()
    union_area_km2 = float(intersection_union.area) / 1_000_000.0
    coverage = {
        "area_municipal_km2": municipio_area_km2,
        "suma_areas_interseccion_individual_km2": sum_individual,
        "union_areas_interseccion_km2": union_area_km2,
        "solapamiento_subcuencas_km2": max(0.0, sum_individual - union_area_km2),
        "superficie_municipal_sin_cobertura_km2": max(0.0, municipio_area_km2 - union_area_km2),
        "superficie_sobrepasada_km2": max(0.0, union_area_km2 - municipio_area_km2),
        "cobertura_union_pct_municipal": 100.0 * union_area_km2 / municipio_area_km2 if municipio_area_km2 else 0.0,
    }
    return gdf, selected, coverage


def recortar_lineas_municipales(lineas, municipio_geom):
    """Recorta líneas y conserva todos los atributos de cada entidad fuente."""
    if lineas.crs is None:
        raise ValueError("Las corrientes deben tener CRS definido.")
    rows = []
    for _, row in lineas.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.intersects(municipio_geom):
            continue
        clipped = _extract_dimension(geom.intersection(municipio_geom), "line")
        if clipped is None or clipped.is_empty or clipped.length == 0:
            continue
        item = row.copy()
        item.geometry = clipped
        item["_dcag_length_municipal_m"] = float(clipped.length)
        item["_dcag_length_municipal_km"] = float(clipped.length) / 1000.0
        rows.append(item)
    if not rows:
        return gpd.GeoDataFrame(columns=list(lineas.columns) + ["_dcag_length_municipal_m", "_dcag_length_municipal_km"], geometry="geometry", crs=lineas.crs)
    return gpd.GeoDataFrame(rows, columns=list(rows[0].index), geometry="geometry", crs=lineas.crs).reset_index(drop=True)


def recortar_geometrias_municipales(gdf, municipio_geom, dimension):
    """Intersección para polígonos; selección espacial para puntos."""
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.intersects(municipio_geom):
            continue
        if dimension == "point":
            clipped = geom
        else:
            clipped = _extract_dimension(geom.intersection(municipio_geom), dimension)
        if clipped is None or clipped.is_empty:
            continue
        item = row.copy()
        item.geometry = clipped
        rows.append(item)
    if not rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry="geometry", crs=gdf.crs)
    return gpd.GeoDataFrame(rows, columns=list(rows[0].index), geometry="geometry", crs=gdf.crs).reset_index(drop=True)


def longitud_total_entidades_km(lineas):
    """Suma de longitudes geométricas por entidad (puede incluir coincidencias)."""
    if lineas.empty:
        return 0.0
    return float(lineas.geometry.length.sum()) / 1000.0


def longitud_geometrica_unica_km(lineas):
    """Longitud de unión geométrica; coincidencias exactas/parciales cuentan una vez."""
    if lineas.empty:
        return 0.0
    union = lineas.geometry.union_all()
    return float(union.length) / 1000.0


def calcular_densidad_drenaje(longitud_km, area_km2):
    if not math.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError("El área municipal debe ser positiva y finita.")
    if not math.isfinite(longitud_km) or longitud_km < 0:
        raise ValueError("La longitud debe ser no negativa y finita.")
    return float(longitud_km) / float(area_km2)


def calcular_densidades_hidrograficas(lineas, area_km2):
    """Calcula densidades cartográfica y habilitada sobre corrientes naturales.

    La primera incluye perennes e intermitentes sin filtrar ENABLED. La segunda
    aplica ENABLED=1. Ambas usan longitudes únicas mediante GeoSeries.union_all.
    """
    if "_dcag_hydro_category" not in lineas.columns or "_dcag_enabled_status" not in lineas.columns:
        raise ValueError("La capa requiere _dcag_hydro_category y _dcag_enabled_status.")
    streams = lineas.loc[
        lineas["_dcag_hydro_category"].isin(("corriente_perenne", "corriente_intermitente"))
    ]
    cartographic_km = longitud_geometrica_unica_km(streams)
    enabled = streams.loc[streams["_dcag_enabled_status"] == "habilitada"]
    enabled_km = longitud_geometrica_unica_km(enabled)
    return {
        "longitud_cartografica_km": cartographic_km,
        "densidad_cartografica_km_km2": calcular_densidad_drenaje(cartographic_km, area_km2),
        "longitud_habilitada_km": enabled_km,
        "densidad_habilitada_km_km2": calcular_densidad_drenaje(enabled_km, area_km2),
    }


def _classify_line_records(lineas, field_info):
    gdf = lineas.copy()
    columns = ("tipo", "entidad", "fc", "condicion")
    values = {
        key: (gdf[field_info[key]] if field_info.get(key) else [None] * len(gdf))
        for key in columns
    }
    gdf["_dcag_hydro_category"] = [
        clasificar_hidrografia(tipo, entidad, fc, condicion)
        for tipo, entidad, fc, condicion in zip(
            values["tipo"], values["entidad"], values["fc"], values["condicion"]
        )
    ]
    category = gdf["_dcag_hydro_category"]
    gdf["_dcag_flow_condition"] = category.map(
        {
            "corriente_perenne": "perenne",
            "corriente_intermitente": "intermitente",
        }
    ).fillna("desconocida")
    gdf["_dcag_nature"] = np.where(
        category.isin(("corriente_perenne", "corriente_intermitente")), "natural",
        np.where(category == "desconocida", "desconocida", "otro"),
    )
    enabled_field = field_info.get("enabled")
    gdf["_dcag_enabled_status"] = [
        clasificar_enabled(v) for v in (gdf[enabled_field] if enabled_field else [None] * len(gdf))
    ]
    gdf["_dcag_classification_inconsistent"] = category == "desconocida"
    return gdf


def _geometry_union_length(gdf):
    return longitud_geometrica_unica_km(gdf)


def _line_metrics(lineas_municipales, municipality_area_km2):
    gdf = lineas_municipales
    category_labels = (
        "corriente_perenne", "corriente_intermitente", "canal",
        "linea_flujo_virtual", "desconocida",
    )
    by_category = {}
    for label in category_labels:
        group = gdf.loc[gdf["_dcag_hydro_category"] == label]
        by_category[label] = {
            "longitud_total_por_entidad_km": longitud_total_entidades_km(group),
            "longitud_geometrica_unica_km": _geometry_union_length(group),
            "entidades": int(len(group)),
        }
    enabled_counts = Counter(gdf["_dcag_enabled_status"].astype(str))
    enabled_lengths = {
        status: longitud_total_entidades_km(gdf.loc[gdf["_dcag_enabled_status"] == status])
        for status in ("habilitada", "deshabilitada", "desconocida")
    }
    unique_all = _geometry_union_length(gdf)
    density_metrics = calcular_densidades_hidrograficas(gdf, municipality_area_km2)
    unique_cartographic = density_metrics["longitud_cartografica_km"]
    unique_enabled = density_metrics["longitud_habilitada_km"]
    raw_total = longitud_total_entidades_km(gdf)
    length_field = _attribute_fields(gdf.columns)["lengthm"]
    length_differences = pd.to_numeric(gdf.get("_dcag_LENGTHM_diff_m", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "longitudes_por_categoria_hidrografica": by_category,
        "longitudes_por_estado_enabled_km": enabled_lengths,
        "conteos_por_estado_enabled": dict(enabled_counts),
        "longitud_total_por_entidad_km": raw_total,
        "longitud_geometrica_unica_km": unique_all,
        "longitud_redundante_por_coincidencias_km": max(0.0, raw_total - unique_all),
        "longitud_geometrica_unica_corrientes_perennes_intermitentes_km": unique_cartographic,
        "densidad_cartografica_principal_km_km2": density_metrics["densidad_cartografica_km_km2"],
        "longitud_geometrica_unica_corrientes_perennes_intermitentes_enabled_1_km": unique_enabled,
        "densidad_red_habilitada_km_km2": density_metrics["densidad_habilitada_km_km2"],
        "definiciones_densidad": {
            "cartografica_principal": (
                "union_all de corrientes de agua perennes e intermitentes municipales, "
                "independientemente de ENABLED, dividido entre la superficie municipal."
            ),
            "red_habilitada": (
                "union_all de corrientes de agua perennes e intermitentes municipales con "
                "ENABLED=1, dividido entre la superficie municipal."
            ),
            "exclusiones_ambos_indicadores": (
                "Canales, líneas centrales de flujo virtual y combinaciones desconocidas o "
                "inconsistentes se excluyen; sus longitudes permanecen en las capas y se reportan aparte."
            ),
        },
        "lengthm": {
            "campo_original_preservado": length_field,
            "entidades_con_diferencia_calculable": int(len(length_differences)),
            "diferencia_firmada_geometria_menos_LENGTHM_media_m": float(length_differences.mean()) if len(length_differences) else None,
            "diferencia_absoluta_maxima_m": float(length_differences.abs().max()) if len(length_differences) else None,
            "nota": "Comparación con la geometría regional completa reproyectada; LENGTHM original no se altera.",
        },
    }


def _raw_value_counts(gdf, field):
    if field is None or field not in gdf.columns:
        return {}
    return {str(k): int(v) for k, v in gdf[field].fillna("<NULL>").astype(str).value_counts(dropna=False).items()}


def resumir_combinaciones_inconsistentes(lineas, field_info):
    """Cuenta y conserva combinaciones TIPO/ENTIDAD/FC/CONDICION no reconocidas."""
    if "_dcag_classification_inconsistent" not in lineas.columns:
        raise ValueError("La capa debe estar clasificada antes de resumir combinaciones.")
    unknown = lineas.loc[lineas["_dcag_classification_inconsistent"]]
    keys = ("tipo", "entidad", "fc", "condicion")
    fields = [field_info.get(key) for key in keys]
    if not all(fields):
        return {"segmentos": int(len(unknown)), "combinaciones": [], "atributos_disponibles": fields}
    counts = Counter(
        tuple("<NULL>" if pd.isna(row[field]) else str(row[field]) for field in fields)
        for _, row in unknown.iterrows()
    )
    combinations = [
        {"tipo": values[0], "entidad": values[1], "fc": values[2], "condicion": values[3], "segmentos": int(count)}
        for values, count in counts.most_common()
    ]
    return {"segmentos": int(len(unknown)), "combinaciones": combinations, "atributos_disponibles": fields}


def _scale_bar(ax, bounds):
    minx, miny, maxx, maxy = bounds
    width, height = maxx - minx, maxy - miny
    raw_m = width * 0.22
    step = 10 ** math.floor(math.log10(max(raw_m, 1)))
    length_m = max(step, round(raw_m / step) * step)
    x0, y0 = minx + width * 0.06, miny + height * 0.06
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=3, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + height * 0.015, f"{length_m / 1000:g} km", ha="center", va="bottom", fontsize=8)


def _map_footer(ax, bounds):
    _scale_bar(ax, bounds)
    ax.text(0.01, 0.01, "CRS: EPSG:6368 | Fuente: INEGI, Red Hidrográfica 1:50 000", transform=ax.transAxes, fontsize=8, va="bottom")


def _save_hydro_map(lineas, cuerpos, limite, path, title):
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    bounds = limite.total_bounds
    if cuerpos is not None and not cuerpos.empty:
        cuerpos.plot(ax=ax, color="#9ecae1", edgecolor="#3182bd", linewidth=0.4, label="Cuerpos de agua")
    palette = {
        "corriente_perenne": "#08519c",
        "corriente_intermitente": "#2ca25f",
        "canal": "#e6550d",
        "linea_flujo_virtual": "#756bb1",
        "desconocida": "#969696",
    }
    if lineas is not None and not lineas.empty:
        for category, group in lineas.groupby("_dcag_hydro_category", dropna=False):
            group.plot(ax=ax, color=palette.get(category, palette["desconocida"]), linewidth=0.8)
    limite.boundary.plot(ax=ax, color="black", linewidth=1.0)
    labels = {
        "corriente_perenne": "Corriente de agua perenne",
        "corriente_intermitente": "Corriente de agua intermitente",
        "canal": "Canal",
        "linea_flujo_virtual": "Línea central de flujo virtual",
        "desconocida": "Combinación desconocida/inconsistente",
    }
    handles = [Line2D([0], [0], color=palette[key], lw=1.7, label=label) for key, label in labels.items()]
    handles.append(Line2D([0], [0], color="black", lw=1.5, label="Límite municipal"))
    if cuerpos is not None and not cuerpos.empty:
        handles.append(Line2D([0], [0], color="#3182bd", lw=5, alpha=0.5, label="Cuerpos de agua"))
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.set_title(title)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    _map_footer(ax, bounds)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_subbasin_map(selected, limite, path):
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    if selected.empty:
        bounds = limite.total_bounds
    else:
        selected_bounds = selected.total_bounds
        municipal_bounds = limite.total_bounds
        bounds = np.array([
            min(selected_bounds[0], municipal_bounds[0]),
            min(selected_bounds[1], municipal_bounds[1]),
            max(selected_bounds[2], municipal_bounds[2]),
            max(selected_bounds[3], municipal_bounds[3]),
        ])
    handles = []
    if not selected.empty:
        basin_ids = sorted(selected["_dcag_subbasin"].astype(str).unique())
        cmap = plt.get_cmap("tab20")
        basin_colors = {basin: cmap(i % 20) for i, basin in enumerate(basin_ids)}
        for basin in basin_ids:
            group = selected.loc[selected["_dcag_subbasin"].astype(str) == basin]
            group.plot(ax=ax, color=basin_colors[basin], edgecolor="#555555", linewidth=0.6, alpha=0.55)
            union_geom = group.geometry.union_all()
            rp = union_geom.representative_point()
            ax.annotate(basin, (rp.x, rp.y), fontsize=8, ha="center")
            handles.append(Patch(facecolor=basin_colors[basin], edgecolor="#555555", alpha=0.65, label=basin))
    limite.boundary.plot(ax=ax, color="black", linewidth=1.2)
    handles.append(Line2D([0], [0], color="black", lw=1.5, label="Límite municipal"))
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.set_title("Subcuencas hidrográficas que intersectan Villa de Álvarez")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    _map_footer(ax, bounds)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_geopackage(gpkg_path, layers):
    written = []
    for name, gdf in layers.items():
        if gdf is None or gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"La capa {name} no tiene CRS al escribir GeoPackage.")
        gdf.to_file(gpkg_path, layer=name, driver="GPKG", mode="w" if not written else "a", index=False)
        written.append(name)
    return written


def generar_productos_hidrografia(
    source_dir=DEFAULT_SOURCE_DIR,
    marco_geoestadistico_dir=DEFAULT_MARCO_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    estado="Colima",
    municipio="Villa de Álvarez",
    expected_subbasins=EXPECTED_SUBBASINS,
):
    """Procesa fuentes regionales y crea GPKG, mapas PNG y JSON sin sobrescritura."""
    source_dir = Path(source_dir).expanduser().resolve()
    marco_geoestadistico_dir = Path(marco_geoestadistico_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    outputs = {
        "geopackage": output_dir / "red_hidrografica_villa_de_alvarez.gpkg",
        "mapa_red": output_dir / "red_hidrografica_municipal.png",
        "mapa_subcuencas": output_dir / "subcuencas_intersectan_municipio.png",
        "metadatos": output_dir / "metadatos_hidrografia.json",
    }
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError("No se sobrescriben resultados existentes: " + ", ".join(existing))

    discovered = descubrir_archivos_hidrograficos(source_dir)
    inventory_report = validar_inventario_hidrografico(discovered, expected_subbasins)
    manager = GISManager(marco_geoestadistico_dir)
    municipal_boundary = manager.obtener_municipio(
        estado=estado, municipio=municipio, crs_destino=TARGET_CRS, territorio="completo"
    )
    if municipal_boundary.empty or municipal_boundary.crs is None:
        raise ValueError("GISManager no devolvió el límite municipal.")
    municipal_boundary = municipal_boundary.to_crs(TARGET_CRS).copy()
    municipality_geom = municipal_boundary.geometry.union_all()
    if municipality_geom.is_empty or not municipality_geom.is_valid:
        raise ValueError("La unión del límite municipal está vacía o es inválida.")
    municipality_area_km2 = float(municipality_geom.area) / 1_000_000.0
    municipal_boundary["_dcag_area_km2"] = municipality_area_km2

    per_suffix = defaultdict(list)
    file_metadata = []
    for suffix, paths in discovered.items():
        for path in paths:
            layer, meta = _read_and_prepare(path, suffix)
            per_suffix[suffix].append(layer)
            file_metadata.append(meta)
    regional = {suffix: _concat_layers(per_suffix[suffix]) for suffix in SUFFIXES}
    if regional["_subc"].empty:
        raise ValueError("No hay polígonos de subcuencas en los archivos _subc.")
    subbasin_ids = set(regional["_subc"]["_dcag_subbasin"].astype(str))
    if len(subbasin_ids) != expected_subbasins:
        raise ValueError(
            f"Se esperaban {expected_subbasins} subcuencas y se detectaron {len(subbasin_ids)}: "
            f"{sorted(subbasin_ids)}"
        )

    subbasins_all, subbasins_selected, coverage = identificar_subcuencas(regional["_subc"], municipality_geom)
    coverage["subcuencas_detectadas"] = len(subbasin_ids)
    coverage["subcuencas_intersectadas"] = int(len(subbasins_selected["_dcag_subbasin"].unique()))
    coverage["intersecciones_por_subcuenca"] = {
        str(row["_dcag_subbasin"]): {
            "area_interseccion_km2": float(row["_dcag_intersection_km2"]),
            "porcentaje_area_municipal": float(row["_dcag_intersection_pct_municipio"]),
        }
        for _, row in subbasins_selected.groupby("_dcag_subbasin", sort=True).agg(
            {"_dcag_intersection_km2": "sum", "_dcag_intersection_pct_municipio": "sum"}
        ).reset_index().iterrows()
    }

    field_info = _attribute_fields(regional["_hl"].columns)
    all_lines = _classify_line_records(regional["_hl"], field_info)
    regional["_hl"] = all_lines
    municipal_lines = recortar_lineas_municipales(all_lines, municipality_geom)
    municipal_lines = _classify_line_records(municipal_lines, field_info)
    municipal_water = recortar_geometrias_municipales(regional["_ha"], municipality_geom, "polygon")
    municipal_drainage = recortar_geometrias_municipales(regional["_dr"], municipality_geom, "point")
    municipal_toponyms = recortar_geometrias_municipales(regional["_to"], municipality_geom, "point")

    # Calcular longitud de cada línea municipal sin alterar LENGTHM original.
    if not municipal_lines.empty:
        municipal_lines["_dcag_length_municipal_m"] = municipal_lines.geometry.length.astype(float)
        municipal_lines["_dcag_length_municipal_km"] = municipal_lines["_dcag_length_municipal_m"] / 1000.0
    metrics = _line_metrics(municipal_lines, municipality_area_km2)

    layers = {
        LAYER_NAMES["_hl"]: regional["_hl"],
        "corrientes_municipales": municipal_lines,
        LAYER_NAMES["_ha"]: regional["_ha"],
        "cuerpos_agua_municipales": municipal_water,
        LAYER_NAMES["_dr"]: regional["_dr"],
        "drenaje_municipal": municipal_drainage,
        LAYER_NAMES["_subc"]: subbasins_all,
        "subcuencas_intersectan_municipio": subbasins_selected,
        LAYER_NAMES["_to"]: regional["_to"],
        "toponimia_municipal": municipal_toponyms,
        "limite_municipal": municipal_boundary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written_layers = _write_geopackage(outputs["geopackage"], layers)
    _save_hydro_map(municipal_lines, municipal_water, municipal_boundary, outputs["mapa_red"],
                    f"Red hidrográfica de {municipio}, {estado}")
    _save_subbasin_map(subbasins_selected, municipal_boundary, outputs["mapa_subcuencas"])

    condition_values = _raw_value_counts(all_lines, field_info["condicion"])
    type_values = _raw_value_counts(all_lines, field_info["tipo"])
    entity_values = _raw_value_counts(all_lines, field_info["entidad"])
    fc_values = _raw_value_counts(all_lines, field_info["fc"])
    enabled_values = _raw_value_counts(all_lines, field_info["enabled"])
    inconsistency_report = resumir_combinaciones_inconsistentes(all_lines, field_info)
    metadata = {
        "procesado_utc": datetime.now(timezone.utc).isoformat(),
        "territorio": {"estado": estado, "municipio": municipio},
        "fuente": str(source_dir),
        "marco_geoestadistico": str(marco_geoestadistico_dir),
        "crs_destino": TARGET_CRS,
        "escala_red_hidrografica": "1:50 000",
        "inventario_archivos": {suffix: [str(p) for p in paths] for suffix, paths in discovered.items()},
        "validacion_inventario": inventory_report,
        "metadatos_archivos": file_metadata,
        "validacion": {
            "shapefiles_detectados": inventory_report["archivos_detectados"],
            "archivos_esperados": inventory_report["archivos_esperados"],
            "inventario_completo_sin_duplicados": True,
            "subcuencas_esperadas": expected_subbasins,
            "subcuencas_detectadas": len(subbasin_ids),
            "crs_origenes": dict(Counter(m["crs_origen"] for m in file_metadata)),
            "campos_de_clasificacion": field_info,
            "valores_originales_TIPO": type_values,
            "valores_originales_ENTIDAD": entity_values,
            "valores_originales_FC": fc_values,
            "valores_originales_condicion": condition_values,
            "valores_originales_ENABLED": enabled_values,
            "enabled_policy": (
                "ENABLED=0 significa deshabilitado para redes geométricas y ENABLED=1 habilitado; se reconocen "
                "valores numéricos, booleanos y representaciones equivalentes. Todos los segmentos permanecen en "
                "capas; ENABLED=0 no representa ausencia física y se incluye en densidad cartográfica, pero no en densidad habilitada."
            ),
            "unknown_codes_policy": "Solo combinaciones conjuntas TIPO/ENTIDAD/FC/CONDICION documentadas se clasifican; códigos y combinaciones restantes son desconocidos.",
            "documentacion_dominios_inegi": [
                "https://antares.inegi.org.mx/analisis/red_hidro/PDF/Doc.pdf",
                "https://antares.inegi.org.mx/analisis/red_hidro/PDF/Taller_Red_Hidro_50K_SIATL_oct.pdf",
            ],
            "clasificacion_hidrografica": {
                "conteos_por_categoria": {str(k): int(v) for k, v in all_lines["_dcag_hydro_category"].value_counts(dropna=False).items()},
                "combinaciones_desconocidas_o_inconsistentes": inconsistency_report["segmentos"],
                "detalle_combinaciones_desconocidas_o_inconsistentes": inconsistency_report["combinaciones"],
                "combinaciones_documentadas": [
                    {"tipo": 101, "entidad": "CORRIENTE DE AGUA", "fc": 3271, "condicion": "INTERMITENTE", "categoria": "corriente_intermitente"},
                    {"tipo": 101, "entidad": "CORRIENTE DE AGUA", "fc": 3272, "condicion": "PERENNE", "categoria": "corriente_perenne"},
                    {"tipo": 102, "entidad": "CANAL", "fc": 3180, "condicion": "EN OPERACION", "categoria": "canal"},
                    {"tipo": 102, "entidad": "CANAL", "fc": 3181, "condicion": "EN CONSTRUCCION", "categoria": "canal"},
                    {"tipo": 102, "entidad": "CANAL", "fc": 3182, "condicion": "FUERA DE USO", "categoria": "canal"},
                    {"tipo": 103, "entidad": "LINEA CENTRAL", "fc": 3273, "condicion": "FLUJO VIRTUAL o LINEA CENTRAL DE CUERPO DE AGUA", "categoria": "linea_flujo_virtual"},
                ],
            },
            "geometry_repairs": int(sum(m["geometrias_reparadas"] for m in file_metadata)),
            "geometrias_invalidas_despues_reparacion_y_reproyeccion": int(sum(m["geometrias_invalidas_despues_reparacion_y_reproyeccion"] for m in file_metadata)),
            "geometrias_dimension_incorrecta_despues_reproyeccion": int(sum(m["geometrias_dimension_incorrecta_despues_reproyeccion"] for m in file_metadata)),
            "geometrias_nulas": int(sum(m["geometrias_nulas"] for m in file_metadata)),
            "geometrias_vacias": int(sum(m["geometrias_vacias"] for m in file_metadata)),
            "feature_duplicates_removed": 0,
        },
        "area_y_cobertura_subcuencas": coverage,
        "area_municipal_km2": municipality_area_km2,
        "longitudes_y_densidad": metrics,
        "notas": {
            "longitud_total_por_entidad": "Suma de geometrías de cada entidad lineal municipal; puede contar coincidencias más de una vez.",
            "longitud_geometrica_unica": "Longitud de union_all de segmentos; coincidencias exactas y parciales se contabilizan una sola vez en ese indicador, sin eliminar entidades del GeoPackage.",
            "densidad_cartografica": "Corrientes perennes e intermitentes únicas por union_all, incluye ENABLED=0; canales, líneas virtuales y desconocidas se excluyen.",
            "densidad_red_habilitada": "Corrientes perennes e intermitentes únicas por union_all con ENABLED=1; canales, líneas virtuales, ENABLED=0 y desconocidas se excluyen.",
            "lengthm": "LENGTHM original se conserva; diferencia firmada se calcula contra la longitud de la geometría regional reproyectada.",
        },
        "capas_geopackage": written_layers,
        "salidas": {k: str(v) for k, v in outputs.items()},
    }
    outputs["metadatos"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--marco-dir", type=Path, default=DEFAULT_MARCO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for key, path in generar_productos_hidrografia(
        source_dir=args.source_dir,
        marco_geoestadistico_dir=args.marco_dir,
        output_dir=args.output_dir,
    ).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
