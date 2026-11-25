"""Diagnóstico espacial de los productos hidrológicos WhiteboxTools 90–97.

El módulo solo lee los rasters y vectores de entrada. Produce reportes, tablas de
anomalías y mapas en una carpeta de diagnóstico separada; nunca recalcula ni
reescribe los productos hidrológicos. Las reducciones raster se usan solo para
visualización; las estadísticas y controles se calculan sobre todas las celdas
por ventanas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import Point, box
from shapely.strtree import STRtree


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados"
DEFAULT_REFERENCE_DEM = PROCESSED_DIR / "CEM40_Colima_Jalisco_Dominio_Hidrologico_Buffer2000m_UTM13N_15m.tif"
DEFAULT_HYDRO_DIR = REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez"
DEFAULT_HYDRO_GPKG = DEFAULT_HYDRO_DIR / "red_hidrografica_villa_de_alvarez.gpkg"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "Etapa_2_Hidrologia" / "diagnostico_98_1"

PRODUCTS = {
    "mde_relleno_sin_flats": "90_MDE_Relleno_Sin_Flats.tif",
    "profundidad_depresiones": "90_Profundidad_Depresiones.tif",
    "mde_acondicionado": "91_MDE_Acondicionado.tif",
    "pendiente_grados": "92_Pendiente_Grados.tif",
    "direccion_d8": "93_Direccion_Flujo_D8.tif",
    "acumulacion_celdas": "94_Acumulacion_Flujo_Celdas.tif",
    "drenaje_potencial": "95_Red_Drenaje_Potencial.tif",
    "twi": "96_Indice_TWI.tif",
    "hand": "97_Altura_HAND.tif",
}
PRODUCT_LABELS = {
    "mde_relleno_sin_flats": ("MDE rellenado (sin flats)", "Elevación (m)", "terrain"),
    "profundidad_depresiones": ("Profundidad de depresiones", "Profundidad (m)", "viridis"),
    "mde_acondicionado": ("MDE acondicionado", "Elevación (m)", "terrain"),
    "pendiente_grados": ("Pendiente WhiteboxTools", "Pendiente (grados)", "magma"),
    "direccion_d8": ("Dirección de flujo D8 (WhiteboxTools)", "Código D8", "d8"),
    "acumulacion_celdas": ("Acumulación D8", "log₁₀(celdas + 1), solo visualización", "plasma"),
    "drenaje_potencial": ("Red de drenaje potencial", "0 fondo / 1 corriente", "streams"),
    "twi": ("Índice topográfico de humedad (TWI)", "TWI (adimensional)", "YlGnBu"),
    "hand": ("Altura sobre drenaje conectado (HAND)", "Altura (m)", "cividis"),
}
REGIONAL_MAPS = (
    ("mde_original", "MDE regional original", "Elevación (m)", "terrain"),
    ("profundidad_depresiones", "Profundidad de depresiones", "Profundidad (m)", "viridis"),
    ("pendiente_grados", "Pendiente WhiteboxTools", "Pendiente (grados)", "magma"),
    ("direccion_d8", "Dirección de flujo D8 (WhiteboxTools)", "Código D8", "d8"),
    ("acumulacion_celdas", "Acumulación D8", "log₁₀(celdas + 1), solo visualización", "plasma"),
    ("drenaje_potencial", "Red de drenaje potencial", "0 fondo / 1 corriente", "streams"),
    ("twi", "Índice topográfico de humedad (TWI)", "TWI (adimensional)", "YlGnBu"),
    ("hand", "Altura sobre drenaje conectado (HAND)", "Altura (m)", "cividis"),
)
DEFAULT_SUBBASINS = ("RH15Aa", "RH15Ab", "RH16Ba", "RH16Bb", "RH16Bc")
D8_OFFSETS = {
    # WhiteboxTools native pointer scheme (esri_pntr=False): row, column.
    1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
    16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0),
}
D8_CODES = (0, *D8_OFFSETS)
D8_LABELS = {0: "0 sin flujo", 1: "1 NE", 2: "2 E", 4: "4 SE", 8: "8 S",
             16: "16 SO", 32: "32 O", 64: "64 NO", 128: "128 N"}
DEPTH_INTERVALS_M = ((0.0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 5),
                     (5, 10), (10, 20), (20, 50), (50, 100), (100, math.inf))


class ValidationError(RuntimeError):
    """Error de validación o de insumos del diagnóstico."""


@dataclass
class ValidationConfig:
    reference_dem: Path = DEFAULT_REFERENCE_DEM
    products_dir: Path = PROCESSED_DIR / "Etapa_2_Hidrologia"
    hydrography_gpkg: Path | None = DEFAULT_HYDRO_GPKG
    output_dir: Path = DEFAULT_OUTPUT_DIR
    streams_layer: str = "corrientes_regionales"
    basins_layer: str = "subcuencas_regionales"
    municipal_layer: str = "limite_municipal"
    drainage_layer: str = "drenaje_regional"
    streams_category_field: str = "_dcag_hydro_category"
    basin_id_field: str = "_dcag_subbasin"
    subbasin_ids: tuple[str, ...] = DEFAULT_SUBBASINS
    coastline_path: Path | None = None
    coastline_layer: str | None = None
    expected_crs: str = "EPSG:6368"
    expected_resolution_m: float = 15.0
    expected_width: int = 7702
    expected_height: int = 12262
    expected_valid_cells: int = 62504976
    block_rows: int = 256
    histogram_bins: int = 8192
    display_max_dimension: int = 1200
    stream_tolerance_m: float = 15.0
    hand_zero_tolerance_m: float = 0.001
    slope_floor_degrees: float = 0.1
    expected_results: dict = field(default_factory=lambda: {
        "profundidad_maxima_m": 98.450439453125,
        "d8_cero": 2107,
        "acumulacion_maxima_celdas": 57801368,
        "drenaje_celdas": 1271111,
        "twi_minimo": 1.16632,
        "twi_maximo": 26.93134,
        "hand_validas": 62302138,
    })
    overwrite: bool = False
    generate_maps: bool = True

    def __post_init__(self):
        self.reference_dem = Path(self.reference_dem).expanduser().resolve()
        self.products_dir = Path(self.products_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.hydrography_gpkg is not None:
            self.hydrography_gpkg = Path(self.hydrography_gpkg).expanduser().resolve()
        if self.coastline_path is not None:
            self.coastline_path = Path(self.coastline_path).expanduser().resolve()
        if self.block_rows <= 0 or self.histogram_bins < 2 or self.display_max_dimension < 100:
            raise ValueError("block_rows, histogram_bins o display_max_dimension inválidos.")
        if self.stream_tolerance_m < 0 or not 0 < self.slope_floor_degrees <= 90:
            raise ValueError("Tolerancia de cauces o piso de pendiente inválido.")


@dataclass
class StreamingStats:
    """Resumen exacto por bloques y cuantiles aproximados con histograma."""

    valid_count: int = 0
    invalid_count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    mean: float = 0.0
    m2: float = 0.0
    histogram: np.ndarray | None = None
    edges: np.ndarray | None = None

    def add(self, values, nodata_count=0):
        source = np.asarray(values)
        flat = source.ravel()
        finite = flat[np.isfinite(flat)].astype(np.float64, copy=False)
        self.invalid_count += int(nodata_count + flat.size - finite.size)
        if not finite.size:
            return
        n = int(finite.size)
        block_mean = float(finite.mean(dtype=np.float64))
        block_m2 = float(np.square(finite - block_mean).sum(dtype=np.float64))
        if not self.valid_count:
            self.mean, self.m2 = block_mean, block_m2
        else:
            total = self.valid_count + n
            delta = block_mean - self.mean
            self.m2 += block_m2 + delta * delta * self.valid_count * n / total
            self.mean += delta * n / total
        self.valid_count += n
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

    def summary(self):
        if not self.valid_count:
            return {"celdas_validas": 0, "celdas_invalidas": self.invalid_count}
        return {
            "celdas_validas": self.valid_count,
            "celdas_invalidas": self.invalid_count,
            "minimo": self.minimum,
            "maximo": self.maximum,
            "media": self.mean,
            "desviacion_estandar_poblacional": math.sqrt(max(self.m2, 0.0) / self.valid_count),
        }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _window_rows(dataset, block_rows):
    for row in range(0, dataset.height, block_rows):
        yield Window(0, row, dataset.width, min(block_rows, dataset.height - row))


def _load_geometries(config):
    """Carga límites y capas vectoriales opcionales; geometrías al CRS del MDE."""
    if config.hydrography_gpkg is None or not config.hydrography_gpkg.is_file():
        return {"basins": [], "municipality": [], "streams": None, "drainage": None,
                "warnings": ["GeoPackage hidrográfico no disponible; se omiten análisis por zona y red INEGI."]}
    with rasterio.open(config.reference_dem) as src:
        target_crs = src.crs
    data = {"basins": [], "municipality": [], "streams": None, "drainage": None, "warnings": []}
    gdf = gpd.read_file(config.hydrography_gpkg, layer=config.basins_layer)
    if gdf.crs is None:
        raise ValidationError(f"La capa {config.basins_layer!r} no declara CRS.")
    gdf = gdf.to_crs(target_crs)
    if config.basin_id_field not in gdf.columns:
        raise ValidationError(f"Falta el atributo de subcuenca {config.basin_id_field!r}.")
    gdf = gdf.loc[gdf[config.basin_id_field].astype(str).isin(config.subbasin_ids)].copy()
    present = set(gdf[config.basin_id_field].astype(str))
    missing = sorted(set(config.subbasin_ids) - present)
    if missing:
        data["warnings"].append(f"Subcuencas solicitadas ausentes: {missing}")
    data["basins"] = [(str(row[config.basin_id_field]), row.geometry) for _, row in gdf.iterrows()
                       if row.geometry is not None and not row.geometry.is_empty]
    muni = gpd.read_file(config.hydrography_gpkg, layer=config.municipal_layer)
    if muni.crs is None:
        raise ValidationError(f"La capa {config.municipal_layer!r} no declara CRS.")
    muni = muni.to_crs(target_crs)
    data["municipality"] = [geom for geom in muni.geometry if geom is not None and not geom.is_empty]
    try:
        streams = gpd.read_file(config.hydrography_gpkg, layer=config.streams_layer)
        if streams.crs is None:
            raise ValidationError(f"La capa {config.streams_layer!r} no declara CRS.")
        streams = streams.to_crs(target_crs)
        if config.streams_category_field not in streams.columns:
            raise ValidationError(f"Falta {config.streams_category_field!r} en corrientes oficiales.")
        streams = streams.loc[streams[config.streams_category_field].isin(
            ("corriente_perenne", "corriente_intermitente"))].copy()
        # ENABLED se conserva deliberadamente sin utilizarlo como filtro.
        data["streams"] = streams
    except Exception as exc:
        data["warnings"].append(f"No se pudo leer red de corrientes oficiales: {exc}")
    try:
        drainage = gpd.read_file(config.hydrography_gpkg, layer=config.drainage_layer)
        if drainage.crs is None:
            raise ValidationError(f"La capa {config.drainage_layer!r} no declara CRS.")
        data["drainage"] = drainage.to_crs(target_crs)
    except Exception as exc:
        data["warnings"].append(f"No se pudo leer capa de puntos de drenaje INEGI: {exc}")
    return data


def _grid_metadata(ds):
    image_tags = ds.tags(ns="IMAGE_STRUCTURE")
    return {
        "crs": ds.crs.to_string() if ds.crs else None,
        "resolucion": [float(v) for v in ds.res],
        "dimensiones_columnas_filas": [ds.width, ds.height],
        "bandas": ds.count,
        "tipo_dato": list(ds.dtypes),
        "nodata": ds.nodata,
        "transformacion_afin": [float(v) for v in ds.transform][:6],
        "extension": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
        "compresion": ds.profile.get("compress"),
        "predictor": image_tags.get("PREDICTOR"),
        "tamaño_bytes": Path(ds.name).stat().st_size,
        "modificado_utc": datetime.fromtimestamp(Path(ds.name).stat().st_mtime, timezone.utc).isoformat(),
    }


def compare_grid(reference, product, config):
    checks = {
        "crs": product.crs == reference.crs,
        "resolucion": product.res == reference.res,
        "dimensiones": (product.width, product.height) == (reference.width, reference.height),
        "transformacion_afin": product.transform == reference.transform,
        "extension": product.bounds == reference.bounds,
        "una_banda": product.count == 1,
    }
    if config.expected_crs:
        checks["crs_esperado"] = reference.crs.to_string().upper() == config.expected_crs.upper()
    if config.expected_resolution_m is not None:
        checks["resolucion_esperada"] = all(
            math.isclose(abs(v), config.expected_resolution_m, rel_tol=0, abs_tol=1e-6)
            for v in (reference.transform.a, reference.transform.e)
        )
    if config.expected_width is not None:
        checks["ancho_esperado"] = reference.width == config.expected_width
    if config.expected_height is not None:
        checks["alto_esperado"] = reference.height == config.expected_height
    return checks


def _stats_and_histogram(path, block_rows, bins):
    stats = StreamingStats()
    with rasterio.open(path) as src:
        for window in _window_rows(src, block_rows):
            mask = src.read_masks(1, window=window) > 0
            data = src.read(1, window=window)
            stats.add(data[mask], int(mask.size - mask.sum()))
        summary = stats.summary()
        if not stats.valid_count:
            summary.update({"percentiles_aproximados": {}, "metodo_percentiles": "sin celdas válidas"})
            return summary
        if stats.minimum == stats.maximum:
            summary.update({"percentiles_aproximados": {str(p): stats.minimum for p in (1, 5, 25, 50, 75, 95, 99)},
                            "metodo_percentiles": "constante; exacto", "histogram_bins": bins,
                            "ancho_bin": 0.0})
            return summary
        edges = np.linspace(stats.minimum, stats.maximum, bins + 1, dtype=np.float64)
        histogram = np.zeros(bins, dtype=np.int64)
        for window in _window_rows(src, block_rows):
            mask = src.read_masks(1, window=window) > 0
            values = src.read(1, window=window)[mask]
            histogram += np.histogram(values, bins=edges)[0]
    cumulative = np.cumsum(histogram, dtype=np.int64)
    percentiles = {}
    for p in (1, 5, 25, 50, 75, 95, 99):
        rank = (p / 100.0) * max(stats.valid_count - 1, 0)
        index = min(int(np.searchsorted(cumulative, math.floor(rank) + 1, side="left")), bins - 1)
        before = int(cumulative[index - 1]) if index else 0
        count_bin = int(histogram[index])
        fraction = 0.5 if count_bin == 0 else min(max((rank - before) / count_bin, 0.0), 1.0)
        percentiles[str(p)] = float(edges[index] + fraction * (edges[index + 1] - edges[index]))
    summary.update({
        "percentiles_aproximados": percentiles,
        "metodo_percentiles": "histograma uniforme de dos pasadas; aproximado",
        "histogram_bins": bins,
        "ancho_bin": float((stats.maximum - stats.minimum) / bins),
        "nota_percentiles": "El ancho de bin es resolución nominal, no cota del error respecto al cuantil de muestra.",
    })
    return summary


def _rasterized_zone(geometries, out_shape, transform):
    if not geometries:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(((geom, 1) for geom in geometries), out_shape=out_shape,
                     transform=transform, fill=0, dtype="uint8", all_touched=False).astype(bool)


def coverage_by_zone(config, vectors, grid_results):
    """Cuenta máscara de referencia y productos por región, municipio y subcuenca."""
    zones = {"Villa de Álvarez": vectors["municipality"]}
    zones.update({name: [geom] for name, geom in vectors["basins"]})
    if not any(zones.values()):
        return {"disponible": False, "motivo": "No hay geometrías territoriales cargadas."}
    counts = {name: {key: 0 for key in ("celdas_zona", "mde_validas", *[f"{k}_validas" for k in PRODUCTS],
                                                 *[f"{k}_perdidas" for k in PRODUCTS],
                                                 *[f"{k}_ganadas" for k in PRODUCTS])}
              for name in zones}
    with rasterio.open(config.reference_dem) as ref:
        product_handles = {}
        for key, result in grid_results.items():
            if result["comparable"]:
                product_handles[key] = rasterio.open(config.products_dir / PRODUCTS[key])
        try:
            for window in _window_rows(ref, config.block_rows):
                shape = (int(window.height), int(window.width))
                transform = window_transform(window, ref.transform)
                ref_valid = ref.read_masks(1, window=window) > 0
                mask_by_product = {key: ds.read_masks(1, window=window) > 0
                                   for key, ds in product_handles.items()}
                zone_masks = {name: _rasterized_zone(geoms, shape, transform) for name, geoms in zones.items()}
                for name, zone_mask in zone_masks.items():
                    counts[name]["celdas_zona"] += int(zone_mask.sum())
                    counts[name]["mde_validas"] += int(np.count_nonzero(zone_mask & ref_valid))
                    for key, product_valid in mask_by_product.items():
                        counts[name][f"{key}_validas"] += int(np.count_nonzero(zone_mask & product_valid))
                        counts[name][f"{key}_perdidas"] += int(np.count_nonzero(zone_mask & ref_valid & ~product_valid))
                        counts[name][f"{key}_ganadas"] += int(np.count_nonzero(zone_mask & ~ref_valid & product_valid))
        finally:
            for ds in product_handles.values():
                ds.close()
    for name, values in counts.items():
        with rasterio.open(config.reference_dem) as ref:
            pixel_area = abs(float(ref.transform.a * ref.transform.e - ref.transform.b * ref.transform.d))
        values["superficie_zona_km2"] = values["celdas_zona"] * pixel_area / 1_000_000
        for key in PRODUCTS:
            if f"{key}_validas" in values:
                values[f"{key}_cobertura_pct_del_MDE"] = (
                    100.0 * values[f"{key}_validas"] / values["mde_validas"] if values["mde_validas"] else None
                )
    return {"disponible": True, "metodo": "conteo por rasterización all_touched=False en cuadrícula de referencia",
            "zonas": counts}


def _point_zone(point, basin_geoms):
    for basin, geometry in basin_geoms:
        if geometry.covers(point):
            return basin
    return "fuera_de_subcuencas_seleccionadas"


def _read_pixel_value(ds, row, col):
    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return None, False
    window = Window(col, row, 1, 1)
    valid = bool(ds.read_masks(1, window=window)[0, 0] > 0)
    return (ds.read(1, window=window)[0, 0].item(), valid)


def _near_invalid(mask_halo, local_row, local_col):
    row0, row1 = max(local_row - 1, 0), min(local_row + 2, mask_halo.shape[0])
    col0, col1 = max(local_col - 1, 0), min(local_col + 2, mask_halo.shape[1])
    return not bool(mask_halo[row0:row1, col0:col1].all())


def _d8_neighbor(row, col, code):
    """Regresa (fila, columna) destino bajo la convención Whitebox nativa."""
    try:
        dr, dc = D8_OFFSETS[int(code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Código D8 no direccional: {code!r}") from exc
    return row + dr, col + dc


def d8_diagnostics(config, vectors):
    """Localiza ceros D8, verifica gradiente/topología y transiciones entre cuencas."""
    d8_path = config.products_dir / PRODUCTS["direccion_d8"]
    acc_path = config.products_dir / PRODUCTS["acumulacion_celdas"]
    conditioned_path = config.products_dir / PRODUCTS["mde_acondicionado"]
    anomalies = []
    counts = {"zero_total": 0, "zero_borde_o_cerca_NoData": 0, "zero_interior": 0,
              "fuera_raster": 0, "direccion_hacia_NoData": 0, "direcciones_hacia_arriba": 0,
              "direcciones_en_plano": 0, "acumulacion_no_monotona": 0,
              "comparaciones_acumulacion_sobre_2pow24": 0,
              "acumulacion_igual_adyacente_sobre_2pow24": 0,
              "acumulacion_no_monotona_sobre_2pow24": 0,
              "transiciones_subcuenca": 0}
    transitions = {}
    zero_by_subbasin = {}
    zero_by_class = {"borde_o_cerca_NoData": 0, "interior": 0}
    drainage_points = []
    if vectors.get("drainage") is not None:
        drainage_points = [g for g in vectors["drainage"].geometry
                           if isinstance(g, Point) and not g.is_empty]
    drainage_tree = STRtree(drainage_points) if drainage_points else None
    drainage_distance_stats = {"crossings_checked": 0, "min_distance_m": None,
                                "max_distance_m": None, "mean_distance_m": None}
    drainage_distance_sum = 0.0
    with rasterio.open(config.reference_dem) as ref, rasterio.open(d8_path) as d8, \
            rasterio.open(acc_path) as acc, rasterio.open(conditioned_path) as conditioned:
        transform = ref.transform
        height, width = ref.height, ref.width
        basin_codes = {name: i + 1 for i, (name, _) in enumerate(vectors["basins"])}
        code_names = {value: name for name, value in basin_codes.items()}
        for window in _window_rows(ref, config.block_rows):
            start, stop = int(window.row_off), int(window.row_off + window.height)
            halo_start, halo_stop = max(0, start - 1), min(height, stop + 1)
            halo_window = Window(0, halo_start, width, halo_stop - halo_start)
            core_slice = slice(start - halo_start, stop - halo_start)
            d8_halo = d8.read(1, window=halo_window)
            d8_mask_halo = d8.read_masks(1, window=halo_window) > 0
            acc_halo = acc.read(1, window=halo_window)
            acc_mask_halo = acc.read_masks(1, window=halo_window) > 0
            z_halo = conditioned.read(1, window=halo_window)
            z_mask_halo = conditioned.read_masks(1, window=halo_window) > 0
            ref_mask_halo = ref.read_masks(1, window=halo_window) > 0
            d8_core = d8_halo[core_slice]
            d8_valid_core = d8_mask_halo[core_slice]
            zeros = np.argwhere(d8_valid_core & (d8_core == 0))
            for rr, cc in zeros:
                gr, gc = start + int(rr), int(cc)
                lr, lc = gr - halo_start, gc
                near_boundary = (gr == 0 or gc == 0 or gr == height - 1 or gc == width - 1
                                 or _near_invalid(ref_mask_halo, lr, lc)
                                 or _near_invalid(d8_mask_halo, lr, lc))
                x, y = transform * (gc + 0.5, gr + 0.5)
                point = Point(x, y)
                record = {"row": gr, "col": gc, "x": float(x), "y": float(y),
                          "clasificacion": "borde_o_cerca_NoData" if near_boundary else "interior",
                          "subcuenca": _point_zone(point, vectors["basins"])}
                anomalies.append(record)
                zero_by_subbasin[record["subcuenca"]] = zero_by_subbasin.get(record["subcuenca"], 0) + 1
                zero_by_class[record["clasificacion"]] += 1
            counts["zero_total"] += int(len(zeros))
            counts["zero_borde_o_cerca_NoData"] += sum(a["clasificacion"] == "borde_o_cerca_NoData" for a in anomalies[-len(zeros):]) if len(zeros) else 0
            counts["zero_interior"] += sum(a["clasificacion"] == "interior" for a in anomalies[-len(zeros):]) if len(zeros) else 0

            if vectors["basins"]:
                shapes = [(geom, basin_codes[name]) for name, geom in vectors["basins"]]
                basin_grid = rasterize(shapes, out_shape=d8_halo.shape,
                                       transform=window_transform(halo_window, transform),
                                       fill=0, dtype="int16", all_touched=False)
            else:
                basin_grid = np.zeros(d8_halo.shape, dtype=np.int16)

            # Controles para cada flujo D8 dirigido; se usan celdas vecinas reales.
            for code in D8_OFFSETS:
                dr, dc = _d8_neighbor(0, 0, code)
                rr, cc = np.where(d8_valid_core & (d8_core == code))
                if not len(rr):
                    continue
                gr, gc = rr + start, cc
                tr, tc = gr + dr, gc + dc
                inside = (tr >= 0) & (tr < height) & (tc >= 0) & (tc < width)
                counts["fuera_raster"] += int((~inside).sum())
                if not inside.any():
                    continue
                src_lr, src_lc = gr[inside] - halo_start, gc[inside]
                dst_lr, dst_lc = tr[inside] - halo_start, tc[inside]
                src_z = z_halo[src_lr, src_lc].astype(np.float64)
                dst_z = z_halo[dst_lr, dst_lc].astype(np.float64)
                src_z_valid = z_mask_halo[src_lr, src_lc]
                dst_z_valid = z_mask_halo[dst_lr, dst_lc]
                counts["direccion_hacia_NoData"] += int(np.count_nonzero(src_z_valid & ~dst_z_valid))
                valid_z = src_z_valid & dst_z_valid
                dz = dst_z[valid_z] - src_z[valid_z]
                counts["direcciones_hacia_arriba"] += int(np.count_nonzero(dz > 1e-4))
                counts["direcciones_en_plano"] += int(np.count_nonzero(np.abs(dz) <= 1e-4))
                src_acc = acc_halo[src_lr, src_lc].astype(np.float64)
                dst_acc = acc_halo[dst_lr, dst_lc].astype(np.float64)
                av = acc_mask_halo[src_lr, src_lc] & acc_mask_halo[dst_lr, dst_lc]
                decreases = av & (dst_acc < src_acc)
                high_precision_range = av & ((src_acc >= 16777216) | (dst_acc >= 16777216))
                counts["acumulacion_no_monotona"] += int(np.count_nonzero(decreases))
                counts["comparaciones_acumulacion_sobre_2pow24"] += int(np.count_nonzero(high_precision_range))
                counts["acumulacion_igual_adyacente_sobre_2pow24"] += int(
                    np.count_nonzero(high_precision_range & (dst_acc == src_acc)))
                counts["acumulacion_no_monotona_sobre_2pow24"] += int(
                    np.count_nonzero(high_precision_range & decreases))
                if vectors["basins"]:
                    src_b = basin_grid[src_lr, src_lc]
                    dst_b = basin_grid[dst_lr, dst_lc]
                    crossing = (src_b > 0) & (dst_b > 0) & (src_b != dst_b)
                    for left, right in zip(src_b[crossing], dst_b[crossing]):
                        src_name, dst_name = code_names[int(left)], code_names[int(right)]
                        pair = f"{src_name}->{dst_name}"
                        transitions[pair] = transitions.get(pair, 0) + 1
                        counts["transiciones_subcuenca"] += 1
                    if crossing.any() and drainage_tree is not None:
                        crossing_rows = gr[inside][crossing]
                        crossing_cols = gc[inside][crossing]
                        crossing_points = [Point(*(transform * (int(col) + 0.5, int(row) + 0.5)))
                                           for row, col in zip(crossing_rows, crossing_cols)]
                        _, distances = drainage_tree.query_nearest(
                            crossing_points, all_matches=False, return_distance=True
                        )
                        distances = np.asarray(distances, dtype=np.float64)
                        if distances.size:
                            drainage_distance_stats["crossings_checked"] += int(distances.size)
                            min_old = drainage_distance_stats["min_distance_m"]
                            max_old = drainage_distance_stats["max_distance_m"]
                            drainage_distance_stats["min_distance_m"] = float(distances.min()) if min_old is None else min(min_old, float(distances.min()))
                            drainage_distance_stats["max_distance_m"] = float(distances.max()) if max_old is None else max(max_old, float(distances.max()))
                            drainage_distance_sum += float(distances.sum(dtype=np.float64))
    # Clasificar conexiones RH15/RH16 por región hidrológica; son alertas diagnósticas, no errores automáticos.
    rh_transitions = {pair: value for pair, value in transitions.items()
                      if pair.split("->")[0][:4] != pair.split("->")[1][:4]}
    if drainage_distance_stats["crossings_checked"]:
        drainage_distance_stats["mean_distance_m"] = (
            drainage_distance_sum / drainage_distance_stats["crossings_checked"]
        )
    return {
        "conteos": counts,
        "ceros_por_subcuenca": zero_by_subbasin,
        "ceros_por_clasificacion": zero_by_class,
        "transiciones_entre_subcuencas": transitions,
        "transiciones_entre_RH15_y_RH16_candidatas_a_revision": rh_transitions,
        "proximidad_transiciones_a_puntos_drenaje_INEGI": drainage_distance_stats if drainage_tree else {
            "disponible": False, "motivo": "No hay capa regional de puntos de drenaje válida."
        },
        "celdas_D8_cero": anomalies,
        "metodo_cero": "Código 0 se clasifica como borde si toca borde raster o vecindad NoData; en otro caso, interior.",
        "metodo_topologia": "Comprueba dirección hacia elevación no mayor (tolerancia 0.1 mm) y acumulación descendente no monotónica; no corrige datos.",
        "nota_acumulacion": (
            "La topología exige que el destino no tenga menor acumulación que el origen; igualdad se admite. "
            "Se reportan aparte aristas con valores >= 2^24 (16,777,216), donde Float32 ya no representa "
            "todos los enteros consecutivos; la igualdad local en ese rango puede deberse a cuantización y "
            "no se interpreta por sí sola como error."
        ),
        "umbral_precision_float32_celdas": 16777216,
    }


def _maxima_and_depression(config, vectors):
    dem_path = config.reference_dem
    depth_path = config.products_dir / PRODUCTS["profundidad_depresiones"]
    acc_path = config.products_dir / PRODUCTS["acumulacion_celdas"]
    maximum_depth = -math.inf
    depth_cells = []
    accumulation_max = -math.inf
    accumulation_cells = []
    depth_hist = {
        (f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">={lo:g}"): 0
        for lo, hi in DEPTH_INTERVALS_M
    }
    interval_depth_sum = {key: 0.0 for key in depth_hist}
    interval_area = {key: 0.0 for key in depth_hist}
    depth_positive = 0
    with rasterio.open(dem_path) as ref, rasterio.open(depth_path) as depth, rasterio.open(acc_path) as accum:
        pixel_area = abs(float(ref.transform.a * ref.transform.e - ref.transform.b * ref.transform.d))
        for window in _window_rows(ref, config.block_rows):
            row_start, col_start = int(window.row_off), int(window.col_off)
            ref_mask = ref.read_masks(1, window=window) > 0
            dmask = depth.read_masks(1, window=window) > 0
            values = depth.read(1, window=window).astype(np.float64)
            valid = ref_mask & dmask & np.isfinite(values)
            if valid.any():
                block_max = float(values[valid].max())
                if block_max > maximum_depth:
                    maximum_depth, depth_cells = block_max, []
                if block_max == maximum_depth:
                    for rr, cc in np.argwhere(valid & (values == maximum_depth)):
                        row, col = row_start + int(rr), col_start + int(cc)
                        x, y = ref.transform * (col + 0.5, row + 0.5)
                        depth_cells.append({"row": row, "col": col, "x": float(x), "y": float(y),
                                            "subcuenca": _point_zone(Point(x, y), vectors["basins"])})
                positive = values[valid & (values > 0)]
                depth_positive += int(positive.size)
                if positive.size:
                    for lo, hi in DEPTH_INTERVALS_M:
                        key = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">={lo:g}"
                        group = positive[(positive >= lo) & (positive < hi)]
                        interval_depth_sum[key] += float(group.sum(dtype=np.float64))
                        interval_area[key] += group.size * pixel_area
                        interval_depth_sum.setdefault(key, 0.0)
                        # Store counts in same exact block pass.
                        depth_hist[key] += int(group.size)
            amask = accum.read_masks(1, window=window) > 0
            aval = accum.read(1, window=window)
            if np.any(amask):
                block_max = float(aval[amask].max())
                if block_max > accumulation_max:
                    accumulation_max, accumulation_cells = block_max, []
                if block_max == accumulation_max:
                    for rr, cc in np.argwhere(amask & (aval == accumulation_max)):
                        row, col = row_start + int(rr), col_start + int(cc)
                        x, y = ref.transform * (col + 0.5, row + 0.5)
                        accumulation_cells.append({"row": row, "col": col, "x": float(x), "y": float(y),
                                                   "elevacion_acondicionada_m": None,
                                                   "subcuenca": _point_zone(Point(x, y), vectors["basins"])})
    # Agregar elevación en la(s) celda(s) máximas sin retener todo el raster.
    with rasterio.open(config.products_dir / PRODUCTS["mde_acondicionado"]) as conditioned:
        for record in accumulation_cells:
            value, valid = _read_pixel_value(conditioned, record["row"], record["col"])
            record["elevacion_acondicionada_m"] = float(value) if valid else None
    with rasterio.open(config.reference_dem) as ref:
        pixel_area = abs(float(ref.transform.a * ref.transform.e - ref.transform.b * ref.transform.d))
    return {
        "profundidad": {
            "maximo_m": None if not math.isfinite(maximum_depth) else maximum_depth,
            "celdas_maximas": depth_cells,
            "celdas_con_profundidad_positiva": depth_positive,
            "superficie_positiva_km2": depth_positive * pixel_area / 1_000_000,
            "intervalos_m": {
                key: {"celdas": depth_hist[key], "superficie_km2": interval_area[key] / 1_000_000,
                      "volumen_geometrico_m3": interval_depth_sum[key] * pixel_area}
                for key in depth_hist
            },
        },
        "acumulacion": {
            "maximo_celdas": None if not math.isfinite(accumulation_max) else accumulation_max,
            "area_contribuyente_km2": accumulation_max * pixel_area / 1_000_000 if math.isfinite(accumulation_max) else None,
            "porcentaje_celdas_validas_MDE": None,
            "celdas_maximas": accumulation_cells,
            "precision_float32": {
                "dtype": "float32",
                "limite_enteros_exactos_2pow24": 16777216,
                "maximo_supera_limite": accumulation_max > 16777216,
                "nota": "Sobre 2^24, Float32 no representa todos los enteros consecutivos exactamente.",
            },
        },
    }


def _coast_distances(config, maxima):
    if config.coastline_path is None or not config.coastline_path.is_file():
        return {"disponible": False, "motivo": "No se proporcionó una capa de costa; no se infiere costa desde cuerpos de agua interiores."}
    coast = gpd.read_file(config.coastline_path, layer=config.coastline_layer) if config.coastline_layer else gpd.read_file(config.coastline_path)
    if coast.crs is None:
        raise ValidationError("La capa de costa no declara CRS.")
    with rasterio.open(config.reference_dem) as ref:
        coast = coast.to_crs(ref.crs)
    geometries = [g for g in coast.geometry if g is not None and not g.is_empty]
    if not geometries:
        return {"disponible": False, "motivo": "La capa de costa no contiene geometrías válidas."}
    tree = STRtree(geometries)
    records = []
    for cell in maxima["acumulacion"]["celdas_maximas"]:
        p = Point(cell["x"], cell["y"])
        indices, distances = tree.query_nearest(p, return_distance=True, all_matches=False)
        distance = float(distances[0]) if len(distances) else None
        records.append({"row": cell["row"], "col": cell["col"], "distancia_costa_m": distance})
    return {"disponible": True, "capa": str(config.coastline_path), "capa_vectorial": config.coastline_layer,
            "distancias_celdas_maximas": records, "criterio": "distancia planar al segmento de costa más próximo en CRS métrico."}


def _twi_diagnostics(config):
    twi_path = config.products_dir / PRODUCTS["twi"]
    slope_path = config.products_dir / PRODUCTS["pendiente_grados"]
    accum_path = config.products_dir / PRODUCTS["acumulacion_celdas"]
    stats = {"coincidencias": 0, "celdas_validas_comparables": 0, "celdas_piso_pendiente": 0,
             "celdas_con_slope_menor_piso": 0, "error_absoluto_maximo_formula": 0.0,
             "errores_absolutos_suma": 0.0, "twi_menor_cero": 0}
    with rasterio.open(twi_path) as twi, rasterio.open(slope_path) as slope, rasterio.open(accum_path) as accum:
        cell_width = abs(float(twi.transform.a))
        if not math.isclose(cell_width, abs(float(twi.transform.e)), abs_tol=1e-7):
            return {"formula_verificable": False, "motivo": "La aproximación configurada requiere celda cuadrada."}
        floor_tan = math.tan(math.radians(config.slope_floor_degrees))
        for window in _window_rows(twi, config.block_rows):
            tm = twi.read_masks(1, window=window) > 0
            sm = slope.read_masks(1, window=window) > 0
            am = accum.read_masks(1, window=window) > 0
            tv = twi.read(1, window=window).astype(np.float64)
            sv = slope.read(1, window=window).astype(np.float64)
            av = accum.read(1, window=window).astype(np.float64)
            valid = tm & sm & am & (av > 0) & np.isfinite(tv) & np.isfinite(sv) & np.isfinite(av)
            stats["celdas_validas_comparables"] += int(valid.sum())
            stats["celdas_con_slope_menor_piso"] += int(np.count_nonzero(valid & (sv < config.slope_floor_degrees)))
            stats["celdas_piso_pendiente"] += int(np.count_nonzero(valid & np.isclose(sv, config.slope_floor_degrees, atol=1e-5)))
            stats["twi_menor_cero"] += int(np.count_nonzero(valid & (tv < 0)))
            if valid.any():
                denominator = np.maximum(np.tan(np.deg2rad(sv[valid])), floor_tan)
                expected = np.log(av[valid] * cell_width / denominator)
                errors = np.abs(tv[valid] - expected)
                stats["errores_absolutos_suma"] += float(errors.sum(dtype=np.float64))
                stats["error_absoluto_maximo_formula"] = max(stats["error_absoluto_maximo_formula"], float(errors.max()))
                stats["coincidencias"] += int(np.count_nonzero(errors <= 1e-4))
    stats["formula_verificable"] = True
    stats["formula"] = "ln((acumulacion_celdas × ancho_celda_m) / tan(max(pendiente_grados, piso_grados)))"
    stats["pendiente_piso_grados"] = config.slope_floor_degrees
    stats["error_absoluto_medio"] = (
        stats["errores_absolutos_suma"] / stats["celdas_validas_comparables"]
        if stats["celdas_validas_comparables"] else None
    )
    return stats


def _twi_extremes(config, vectors):
    return _extreme_locations(config.products_dir / PRODUCTS["twi"], config.reference_dem,
                               vectors, config.block_rows)


def _hand_diagnostics(config):
    hand_path = config.products_dir / PRODUCTS["hand"]
    streams_path = config.products_dir / PRODUCTS["drenaje_potencial"]
    result = {"celdas_drenaje": 0, "celdas_drenaje_validas_HAND": 0,
              "celdas_drenaje_noData_HAND": 0, "celdas_drenaje_HAND_no_cero": 0,
              "max_abs_HAND_en_drenaje_m": 0.0, "diferencias_mascara": 0,
              "diferencias_por_subcuenca": {}}
    with rasterio.open(config.reference_dem) as ref, rasterio.open(hand_path) as hand, rasterio.open(streams_path) as streams:
        for window in _window_rows(ref, config.block_rows):
            ref_mask = ref.read_masks(1, window=window) > 0
            hm = hand.read_masks(1, window=window) > 0
            sm = streams.read_masks(1, window=window) > 0
            h = hand.read(1, window=window).astype(np.float64)
            s = streams.read(1, window=window)
            missing = ref_mask & ~hm
            result["diferencias_mascara"] += int(missing.sum())
            stream = sm & (s == 1)
            result["celdas_drenaje"] += int(stream.sum())
            result["celdas_drenaje_validas_HAND"] += int(np.count_nonzero(stream & hm))
            result["celdas_drenaje_noData_HAND"] += int(np.count_nonzero(stream & ~hm))
            vals = np.abs(h[stream & hm])
            if vals.size:
                result["max_abs_HAND_en_drenaje_m"] = max(result["max_abs_HAND_en_drenaje_m"], float(vals.max()))
                result["celdas_drenaje_HAND_no_cero"] += int(np.count_nonzero(vals > config.hand_zero_tolerance_m))
    result["tolerancia_cero_m"] = config.hand_zero_tolerance_m
    result["nota"] = "NoData no se reemplaza por cero; las diferencias se conservan como máscara distinta."
    return result


def compare_stream_networks(config, vectors):
    """Coincidencia cartográfica exacta y con tolerancia de un píxel configurable."""
    official = vectors.get("streams")
    if official is None or official.empty:
        return {"disponible": False, "motivo": "Red oficial perenne/intermitente no disponible."}
    geometries = [(g, i) for i, g in enumerate(official.geometry) if g is not None and not g.is_empty]
    sindex = official.sindex
    stats = {"celdas_derivadas": 0, "pixeles_oficiales": 0, "coincidencia_exacta": 0,
             "derivadas_a_distancia_tolerada": 0, "oficiales_a_distancia_tolerada": 0}
    with rasterio.open(config.products_dir / PRODUCTS["drenaje_potencial"]) as derived:
        pixel_size = min(abs(derived.transform.a), abs(derived.transform.e))
        radius = int(math.ceil(config.stream_tolerance_m / pixel_size)) if pixel_size else 0
        for window in _window_rows(derived, config.block_rows):
            start, stop = int(window.row_off), int(window.row_off + window.height)
            ext_start, ext_stop = max(0, start - radius), min(derived.height, stop + radius)
            ext_window = Window(0, ext_start, derived.width, ext_stop - ext_start)
            core_rows = slice(start - ext_start, stop - ext_start)
            h, w = int(ext_window.height), int(ext_window.width)
            trans = window_transform(ext_window, derived.transform)
            bounds = rasterio.windows.bounds(ext_window, derived.transform)
            search_geom = box(*bounds).buffer(config.stream_tolerance_m)
            candidate_idx = np.asarray(sindex.query(search_geom, predicate="intersects"), dtype=int)
            candidates = [official.geometry.iloc[i] for i in candidate_idx]
            official_mask = rasterize(((g, 1) for g in candidates), out_shape=(h, w), transform=trans,
                                      fill=0, all_touched=True, dtype="uint8").astype(bool) if candidates else np.zeros((h, w), bool)
            derived_valid = derived.read_masks(1, window=ext_window) > 0
            stream_extended = derived_valid & (derived.read(1, window=ext_window) == 1)
            stream_mask = stream_extended[core_rows]
            official_core = official_mask[core_rows]
            # Conteo oficial solo del centro del bloque para evitar doble conteo del halo.
            official_count = int(official_core.sum())
            derived_count = int(stream_mask.sum())
            stats["celdas_derivadas"] += derived_count
            stats["pixeles_oficiales"] += official_count
            stats["coincidencia_exacta"] += int(np.count_nonzero(stream_mask & official_core))
            # Tolerancia cartográfica: vecindad cuadrada de ceil(tolerancia/pixel) celdas.
            official_near = official_mask
            derived_near = stream_extended
            for _ in range(radius):
                official_near = _dilate8(official_near)
                derived_near = _dilate8(derived_near)
            stats["derivadas_a_distancia_tolerada"] += int(np.count_nonzero(stream_mask & official_near[core_rows]))
            stats["oficiales_a_distancia_tolerada"] += int(np.count_nonzero(official_core & derived_near[core_rows]))
    d, o = stats["celdas_derivadas"], stats["pixeles_oficiales"]
    exact = stats["coincidencia_exacta"]
    matched_d = stats["derivadas_a_distancia_tolerada"]
    matched_o = stats["oficiales_a_distancia_tolerada"]
    return {
        "disponible": True,
        "categorias_oficiales": ["corriente_perenne", "corriente_intermitente"],
        "filtro_ENABLED": "No aplicado; ENABLED=0 y 1 se conservan.",
        "tolerancia_m": config.stream_tolerance_m,
        "radio_aproximado_celdas": radius,
        **stats,
        "precision_exacta": exact / d if d else None,
        "recall_exacta": exact / o if o else None,
        "jaccard_exacta": exact / (d + o - exact) if d + o - exact else None,
        "precision_tolerada": matched_d / d if d else None,
        "recall_tolerado": matched_o / o if o else None,
        "nota": "Coincidencia a píxel exacto y a vecindad 8-direccional. Métricas cartográficas, no evaluación de amenaza.",
    }


def _dilate8(mask):
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    out = np.zeros(mask.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
    return out


def _extreme_locations(path, reference_path, vectors, block_rows=256):
    minimum, maximum = math.inf, -math.inf
    min_cells, max_cells = [], []
    with rasterio.open(path) as src, rasterio.open(reference_path) as ref:
        for window in _window_rows(src, block_rows):
            valid = src.read_masks(1, window=window) > 0
            data = src.read(1, window=window)
            values = data[valid]
            if not values.size:
                continue
            lo, hi = float(values.min()), float(values.max())
            if lo < minimum:
                minimum, min_cells = lo, []
            if hi > maximum:
                maximum, max_cells = hi, []
            row0, col0 = int(window.row_off), int(window.col_off)
            if lo == minimum:
                for rr, cc in np.argwhere(valid & (data == minimum)):
                    row, col = row0 + int(rr), col0 + int(cc)
                    x, y = ref.transform * (col + 0.5, row + 0.5)
                    min_cells.append({"row": row, "col": col, "x": float(x), "y": float(y),
                                      "subcuenca": _point_zone(Point(x, y), vectors["basins"])})
            if hi == maximum:
                for rr, cc in np.argwhere(valid & (data == maximum)):
                    row, col = row0 + int(rr), col0 + int(cc)
                    x, y = ref.transform * (col + 0.5, row + 0.5)
                    max_cells.append({"row": row, "col": col, "x": float(x), "y": float(y),
                                      "subcuenca": _point_zone(Point(x, y), vectors["basins"])})
    return {"minimo": minimum if math.isfinite(minimum) else None,
            "maximo": maximum if math.isfinite(maximum) else None,
            "celdas_minimas": min_cells, "celdas_maximas": max_cells}


def _conditioning_checks(config):
    output = {}
    with rasterio.open(config.reference_dem) as original:
        pixel_area = abs(float(original.transform.a * original.transform.e - original.transform.b * original.transform.d))
        for label, filename in (("relleno_sin_flats", PRODUCTS["mde_relleno_sin_flats"]),
                                ("acondicionado", PRODUCTS["mde_acondicionado"])):
            path = config.products_dir / filename
            decrements = 0
            increases = 0
            max_increase = 0.0
            min_difference = math.inf
            decrease_min = 0.0
            with rasterio.open(path) as product:
                for window in _window_rows(original, config.block_rows):
                    valid = ((original.read_masks(1, window=window) > 0)
                             & (product.read_masks(1, window=window) > 0))
                    a = original.read(1, window=window).astype(np.float64)
                    b = product.read(1, window=window).astype(np.float64)
                    delta = b[valid] - a[valid]
                    if delta.size:
                        decrements += int(np.count_nonzero(delta < -1e-4))
                        increases += int(np.count_nonzero(delta > 1e-4))
                        max_increase = max(max_increase, float(delta.max()))
                        min_difference = min(min_difference, float(delta.min()))
                        decrease_min = min(decrease_min, float(delta.min()))
            output[label] = {
                "celdas_con_disminucion_mayor_tolerancia": decrements,
                "celdas_con_aumento": increases,
                "aumento_maximo_m": max_increase,
                "diferencia_minima_m": min_difference if math.isfinite(min_difference) else None,
                "disminucion_maxima_m": abs(decrease_min) if decrease_min < -1e-4 else 0.0,
                "area_celda_m2": pixel_area,
                "control": "No se modificaron valores; disminuciones se reportan, no se corrigen.",
            }
    return output


def _read_preview(path, max_dim, transform_ref=None):
    with rasterio.open(path) as src:
        scale = max(src.width / max_dim, src.height / max_dim, 1.0)
        out_w = max(1, int(round(src.width / scale)))
        out_h = max(1, int(round(src.height / scale)))
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
        mask = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
        bounds = src.bounds
        nodata = src.nodata
        crs = src.crs
    data[~mask] = np.nan
    return data, bounds, crs


def _map_subbasins(ax, vectors, crs):
    if not vectors["basins"]:
        return
    gdf = gpd.GeoDataFrame({"basin": [n for n, _ in vectors["basins"]]},
                           geometry=[g for _, g in vectors["basins"]], crs=crs)
    gdf.boundary.plot(ax=ax, color="#333333", linewidth=0.5, alpha=0.75)


def _scale_bar(ax, bounds):
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if width <= 0 or height <= 0:
        return
    raw = width * 0.15
    power = 10 ** math.floor(math.log10(max(raw, 1)))
    length = max(power, round(raw / power) * power)
    x, y = bounds[0] + width * 0.04, bounds[1] + height * 0.05
    ax.plot([x, x + length], [y, y], color="black", lw=3, solid_capstyle="butt")
    ax.text(x + length / 2, y + height * 0.012, f"{length / 1000:g} km", ha="center", fontsize=8)


def generate_maps(config, vectors, output_files, maxima, d8_report, hand_report):
    """Genera panel regional y panel de anomalías con raster reducido solo para dibujo."""
    panel_path = output_files["panel_regional"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    for ax, (key, title, unit, cmap_name) in zip(axes.flat, REGIONAL_MAPS):
        path = config.reference_dem if key == "mde_original" else config.products_dir / PRODUCTS[key]
        data, bounds, crs = _read_preview(path, config.display_max_dimension)
        if key == "acumulacion_celdas":
            data = np.log10(np.maximum(data, 0) + 1.0)  # transformación exclusiva para visualización
        if key == "direccion_d8":
            # La escala categoriza los códigos numéricos nativos sin convertirlos a esquema ESRI.
            cmap = ListedColormap(["#111111", "#e41a1c", "#ff7f00", "#ffff33", "#4daf4a",
                                   "#377eb8", "#984ea3", "#a65628", "#f781bf"])
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 3, 6, 12, 24, 48, 96, 128.5], cmap.N)
            image = ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                              origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        elif key == "drenaje_potencial":
            cmap = ListedColormap(["#f7f7f7", "#2166ac"])
            image = ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                              origin="upper", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        else:
            image = ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                              origin="upper", cmap=cmap_name, interpolation="nearest")
        _map_subbasins(ax, vectors, crs)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        ax.set_aspect("equal")
        colorbar = fig.colorbar(image, ax=ax, shrink=0.74, label=unit)
        if key == "direccion_d8":
            colorbar.set_ticks(D8_CODES)
            colorbar.set_ticklabels([D8_LABELS[code] for code in D8_CODES])
        _scale_bar(ax, (bounds.left, bounds.bottom, bounds.right, bounds.top))
    fig.suptitle("Diagnóstico hidrológico regional — Colima y Jalisco", fontsize=15)
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    anomaly_path = output_files["panel_anomalias"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    anomaly_layers = [
        ("profundidad_depresiones", "Profundidad de depresiones (m)"),
        ("direccion_d8", "Celdas D8 sin dirección (código 0)"),
        ("acumulacion_celdas", "Celdas de acumulación máxima"),
        ("hand", "HAND NoData dentro de la máscara del MDE"),
    ]
    for ax, (key, title) in zip(axes.flat, anomaly_layers):
        path = config.products_dir / PRODUCTS[key]
        data, bounds, crs = _read_preview(path, config.display_max_dimension)
        if key == "hand":
            with rasterio.open(config.reference_dem) as ref, rasterio.open(path) as hand:
                out_w, out_h = data.shape[1], data.shape[0]
                ref_valid = ref.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
                hand_valid = hand.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
                data = np.where(ref_valid & ~hand_valid, 1.0, np.nan).astype(np.float32)
        image = ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                          origin="upper", cmap="magma" if key != "hand" else "cividis", interpolation="nearest")
        _map_subbasins(ax, vectors, crs)
        if key == "direccion_d8" and d8_report["celdas_D8_cero"]:
            zeros = d8_report["celdas_D8_cero"]
            step = max(1, len(zeros) // 5000)
            ax.scatter([x["x"] for x in zeros[::step]], [x["y"] for x in zeros[::step]],
                       s=3, c="cyan", alpha=0.65, label="D8 = 0")
            ax.legend(loc="lower right", fontsize=7)
        if key == "acumulacion_celdas":
            points = maxima["acumulacion"]["celdas_maximas"]
            if points:
                ax.scatter([x["x"] for x in points], [x["y"] for x in points],
                           s=35, marker="x", c="red", label="Máximo")
                ax.legend(loc="lower right", fontsize=7)
        if key == "hand":
            ax.text(0.02, 0.02, f"NoData vs MDE: {hand_report['diferencias_mascara']:,} celdas",
                    transform=ax.transAxes, color="white", fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 3})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.76)
        _scale_bar(ax, (bounds.left, bounds.bottom, bounds.right, bounds.top))
    fig.suptitle("Anomalías espaciales para revisión; sin corrección automática", fontsize=14)
    fig.savefig(anomaly_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


class HydrologicalValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.started = time.perf_counter()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.paths = {key: config.products_dir / filename for key, filename in PRODUCTS.items()}
        self.output_files = {
            "json": config.output_dir / f"diagnostico_hidrologico_{self.run_id}.json",
            "csv": config.output_dir / f"estadisticas_hidrologicas_{self.run_id}.csv",
            "d8_zero_csv": config.output_dir / f"celdas_D8_cero_{self.run_id}.csv",
            "acc_max_csv": config.output_dir / f"celdas_acumulacion_maxima_{self.run_id}.csv",
            "panel_regional": config.output_dir / f"panel_hidrologico_regional_{self.run_id}.png",
            "panel_anomalias": config.output_dir / f"panel_anomalias_hidrologicas_{self.run_id}.png",
        }

    def _validate_paths(self):
        required = {"referencia": self.config.reference_dem, **self.paths}
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Faltan rasters requeridos: " + "; ".join(missing))
        existing = [str(path) for path in self.output_files.values() if path.exists()]
        if existing and not self.config.overwrite:
            raise FileExistsError("No se sobrescriben salidas diagnósticas: " + "; ".join(existing))

    def validate_grids(self):
        results = {}
        with rasterio.open(self.config.reference_dem) as reference:
            ref_meta = _grid_metadata(reference)
            reference_valid_cells = 0
            for window in _window_rows(reference, self.config.block_rows):
                reference_valid_cells += int(np.count_nonzero(reference.read_masks(1, window=window)))
            with rasterio.open(self.paths["mde_relleno_sin_flats"]) as fill:
                fill_meta = _grid_metadata(fill)
            for key, path in self.paths.items():
                with rasterio.open(path) as product:
                    checks = compare_grid(reference, product, self.config)
                    comparable = all(checks[name] for name in (
                        "crs", "resolucion", "dimensiones", "transformacion_afin", "extension", "una_banda"
                    ))
                    mask_diff = {"perdidas": 0, "ganancias": 0, "diferencias": 0,
                                 "validas_referencia": 0, "validas_producto": 0}
                    if comparable:
                        for window in _window_rows(reference, self.config.block_rows):
                            refmask = reference.read_masks(1, window=window) > 0
                            prodmask = product.read_masks(1, window=window) > 0
                            mask_diff["perdidas"] += int(np.count_nonzero(refmask & ~prodmask))
                            mask_diff["ganancias"] += int(np.count_nonzero(~refmask & prodmask))
                            mask_diff["diferencias"] += int(np.count_nonzero(refmask != prodmask))
                            mask_diff["validas_referencia"] += int(refmask.sum())
                            mask_diff["validas_producto"] += int(prodmask.sum())
                    results[key] = {
                        "ruta": str(path), "metadatos": _grid_metadata(product),
                        "controles_cuadricula": checks,
                        "cuadricula_coincide": comparable,
                        "mascara_vs_referencia": mask_diff if comparable else None,
                    }
        expected_valid_pass = ref_meta["dimensiones_columnas_filas"] == [self.config.expected_width, self.config.expected_height]
        results["referencia"] = {
            "ruta": str(self.config.reference_dem), "metadatos": ref_meta,
            "controles_dominio": {
                "crs": (ref_meta["crs"] or "").upper() == self.config.expected_crs.upper(),
                "dimensiones": expected_valid_pass,
                "resolucion": all(math.isclose(abs(x), self.config.expected_resolution_m, abs_tol=1e-6)
                                   for x in ref_meta["resolucion"]),
                "celdas_validas": (reference_valid_cells == self.config.expected_valid_cells
                                   if self.config.expected_valid_cells is not None else None),
                "celdas_validas_observadas": reference_valid_cells,
            },
        }
        return results

    def run(self):
        self._validate_paths()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        vectors = _load_geometries(self.config)
        grid = self.validate_grids()
        comparable = {key: {"comparable": bool(item["cuadricula_coincide"])}
                      for key, item in grid.items() if key in PRODUCTS}
        coverage = coverage_by_zone(self.config, vectors, comparable)
        statistics = {key: _stats_and_histogram(path, self.config.block_rows, self.config.histogram_bins)
                      for key, path in self.paths.items()}
        maxima = _maxima_and_depression(self.config, vectors)
        valid_reference = 0
        with rasterio.open(self.config.reference_dem) as ref:
            for window in _window_rows(ref, self.config.block_rows):
                valid_reference += int(np.count_nonzero(ref.read_masks(1, window=window)))
        if valid_reference:
            maxima["acumulacion"]["porcentaje_celdas_validas_MDE"] = (
                100.0 * maxima["acumulacion"]["maximo_celdas"] / valid_reference
            )
        d8 = d8_diagnostics(self.config, vectors)
        d8["celdas_D8_cero"] = d8["celdas_D8_cero"]
        twi = _twi_diagnostics(self.config)
        twi["extremos_espaciales"] = _twi_extremes(self.config, vectors)
        hand = _hand_diagnostics(self.config)
        if coverage.get("disponible"):
            hand["cobertura_por_zona"] = {
                zone: {"hand_validas": values.get("hand_validas"),
                       "hand_perdidas": values.get("hand_perdidas"),
                       "mde_validas": values.get("mde_validas")}
                for zone, values in coverage["zonas"].items()
            }
        streams = compare_stream_networks(self.config, vectors)
        coast = _coast_distances(self.config, maxima["acumulacion"])
        conditioning = _conditioning_checks(self.config)
        observed_results = {
            "profundidad_maxima_m": maxima["profundidad"]["maximo_m"],
            "d8_cero": d8["conteos"]["zero_total"],
            "acumulacion_maxima_celdas": maxima["acumulacion"]["maximo_celdas"],
            "drenaje_celdas": hand["celdas_drenaje"],
            "twi_minimo": statistics["twi"]["minimo"],
            "twi_maximo": statistics["twi"]["maximo"],
            "hand_validas": statistics["hand"]["celdas_validas"],
        }
        reported_comparison = {}
        for metric, expected in self.config.expected_results.items():
            observed = observed_results.get(metric)
            tolerance = 0 if metric in ("d8_cero", "drenaje_celdas", "hand_validas") else 1e-5
            reported_comparison[metric] = {
                "reportado": expected, "observado": observed, "tolerancia_comparacion": tolerance,
                "coincide_dentro_tolerancia": (
                    math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=tolerance)
                    if observed is not None and expected is not None else None
                ),
                "nota": "Comparación informativa; no modifica ni valida/corrige el producto por sí sola.",
            }
        if self.config.generate_maps:
            generate_maps(self.config, vectors, self.output_files, maxima, d8, hand)
        _write_csv(self.output_files["csv"], ["producto", "metrica", "valor", "unidad", "metodo"],
                   self._statistics_rows(statistics, coverage) +
                   self._diagnostic_rows(maxima, d8, twi, hand, streams, conditioning))
        _write_csv(self.output_files["d8_zero_csv"],
                   ["row", "col", "x", "y", "clasificacion", "subcuenca"], d8["celdas_D8_cero"])
        _write_csv(self.output_files["acc_max_csv"],
                   ["row", "col", "x", "y", "elevacion_acondicionada_m", "subcuenca"],
                   maxima["acumulacion"]["celdas_maximas"])
        report = {
            "run_id": self.run_id,
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "caso_piloto": "Villa de Álvarez, Colima",
            "dominio_procesamiento": "Regional Colima-Jalisco; 5 subcuencas más buffer hidrológico declarado de 2 km.",
            "configuracion": _jsonable(asdict(self.config)),
            "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                          "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                          "geopandas": gpd.__version__, "shapely": shapely.__version__,
                          "matplotlib": matplotlib.__version__},
            "fuentes": {"mde_referencia": str(self.config.reference_dem),
                        "productos": {key: str(path) for key, path in self.paths.items()},
                        "geopackage_hidrografico": str(self.config.hydrography_gpkg) if self.config.hydrography_gpkg else None,
                        "capas": {"subcuencas": self.config.basins_layer, "municipio": self.config.municipal_layer,
                                  "corrientes_oficiales": self.config.streams_layer, "drenaje": self.config.drainage_layer}},
            "validacion_cuadricula_y_mascaras": grid,
            "cobertura_espacial": coverage,
            "estadisticas_raster": statistics,
            "comparacion_con_resultados_reportados": reported_comparison,
            "comparacion_MDE_original_vs_acondicionamientos": conditioning,
            "depresiones_y_maximos": maxima,
            "diagnostico_D8_y_acumulacion": {k: v for k, v in d8.items() if k != "celdas_D8_cero"},
            "TWI": twi,
            "HAND": hand,
            "comparacion_red_INEGI_vs_derivada": streams,
            "proximidad_costa_maximos": coast,
            "geometrias_cargadas": {
                "subcuencas": [n for n, _ in vectors["basins"]],
                "municipio_geometrias": len(vectors["municipality"]),
                "corrientes_oficiales_naturales": int(len(vectors["streams"])) if vectors["streams"] is not None else None,
                "alcance_red_regional": (
                    "La red natural regional se filtra solo por categoría hidrográfica; no se recorta a las "
                    "subcuencas seleccionadas ni al municipio, para conservar corrientes del buffer y evitar "
                    "efectos de borde."
                ),
            },
            "advertencias": vectors["warnings"],
            "metodologia": {
                "estadisticas": "Cálculo sobre todas las celdas por ventanas; no usa la reducción de visualización.",
                "percentiles": "Histograma uniforme de dos pasadas; aproximados y documentados por ancho nominal de bin.",
                "volumen_depresiones": "Suma de profundidad × área de celda; volumen geométrico, no capacidad real de almacenamiento.",
                "acumulacion_visual": "Se aplica log10(celdas+1) únicamente para el panel; el GeoTIFF y estadísticas usan conteos sin transformar.",
                "red_oficial": "Solo corriente_perenne e corriente_intermitente; no se filtra ENABLED.",
                "seleccion_subcuencas": (
                    "DEFAULT_SUBBASINS selecciona polígonos para cobertura, geolocalización y análisis por "
                    "cuenca. No restringe la capa regional de corrientes; se conservan entidades exteriores "
                    "que puedan representar aportaciones del buffer."
                ),
                "sin_correccion": "El diagnóstico no modifica ni corrige productos fuente.",
                "tolerancia_coincidencia_cauces": "Vecindad de 8 celdas, distancia de comparación configurable; no es umbral hidrológico.",
            },
            "salidas": {key: str(path) for key, path in self.output_files.items()},
            "tiempo_total_segundos": time.perf_counter() - self.started,
        }
        self.output_files["json"].write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _statistics_rows(statistics, coverage):
        rows = []
        for product, summary in statistics.items():
            for metric, value in summary.items():
                if isinstance(value, (int, float)) or value is None:
                    rows.append({"producto": product, "metrica": metric, "valor": value,
                                 "unidad": "conteo" if "celdas" in metric else "valor del raster",
                                 "metodo": "Streaming por bloques"})
            for percentile, value in summary.get("percentiles_aproximados", {}).items():
                rows.append({"producto": product, "metrica": f"percentil_{percentile}", "valor": value,
                             "unidad": "valor del raster", "metodo": summary.get("metodo_percentiles")})
        if coverage.get("disponible"):
            for zone, metrics in coverage["zonas"].items():
                for metric, value in metrics.items():
                    rows.append({"producto": f"cobertura:{zone}", "metrica": metric, "valor": value,
                                 "unidad": "conteo o porcentaje", "metodo": coverage["metodo"]})
        return rows

    @staticmethod
    def _diagnostic_rows(maxima, d8, twi, hand, streams, conditioning):
        rows = []
        groups = {"depresiones_y_maximos": maxima, "diagnostico_D8": {k: v for k, v in d8.items()
                                                                         if k != "celdas_D8_cero"},
                  "TWI": twi, "HAND": hand, "red_INEGI_vs_derivada": streams,
                  "acondicionamiento": conditioning}

        def visit(prefix, value):
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                rows.append({"producto": prefix.split(".", 1)[0],
                             "metrica": prefix.partition(".")[2], "valor": value,
                             "unidad": "conteo o valor reportado",
                             "metodo": "Diagnóstico sintético/espacial según JSON"})

        for group, value in groups.items():
            visit(group, value)
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dem", type=Path, default=DEFAULT_REFERENCE_DEM)
    parser.add_argument("--products-dir", type=Path, default=PROCESSED_DIR / "Etapa_2_Hidrologia")
    parser.add_argument("--hydrography-gpkg", type=Path, default=DEFAULT_HYDRO_GPKG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coastline", type=Path)
    parser.add_argument("--coastline-layer")
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--histogram-bins", type=int, default=8192)
    parser.add_argument("--stream-tolerance-m", type=float, default=15.0)
    parser.add_argument("--slope-floor-degrees", type=float, default=0.1)
    parser.add_argument("--no-maps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = ValidationConfig(
        reference_dem=args.reference_dem, products_dir=args.products_dir,
        hydrography_gpkg=args.hydrography_gpkg, output_dir=args.output_dir,
        coastline_path=args.coastline, coastline_layer=args.coastline_layer,
        block_rows=args.block_rows, histogram_bins=args.histogram_bins,
        stream_tolerance_m=args.stream_tolerance_m, slope_floor_degrees=args.slope_floor_degrees,
        generate_maps=not args.no_maps, overwrite=args.overwrite,
    )
    report = HydrologicalValidator(config).run()
    print("Diagnóstico completado.")
    print(report["salidas"])


if __name__ == "__main__":
    main()
