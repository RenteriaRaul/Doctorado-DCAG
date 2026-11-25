"""Diagnóstico global de conectividad D8 para los productos hidrológicos DCAG.

Lee los productos existentes y crea reportes/mapas en una carpeta independiente.
No recalcula ni modifica los GeoTIFF fuente. La conectividad se resuelve como un
grafo dirigido global; los bloques solo limitan memoria de entrada/salida.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import Point, box

try:
    from scripts.hydrological_validation import (
        DEFAULT_REFERENCE_DEM,
        DEFAULT_SUBBASINS,
        D8_OFFSETS,
        PRODUCTS,
    )
except ImportError:
    from hydrological_validation import (
        DEFAULT_REFERENCE_DEM,
        DEFAULT_SUBBASINS,
        D8_OFFSETS,
        PRODUCTS,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados"
DEFAULT_PRODUCTS_DIR = PROCESSED_DIR / "Etapa_2_Hidrologia"
DEFAULT_HYDRO_GPKG = REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez" / "red_hidrografica_villa_de_alvarez.gpkg"
DEFAULT_OUTPUT_DIR = DEFAULT_PRODUCTS_DIR / "diagnostico_98_2"
UINT32_MAX = np.iinfo(np.uint32).max
FLOAT32_EXACT_INTEGER_LIMIT = 2**24
REGION_NAMES = {"RH15": ("RH15Aa", "RH15Ab"), "RH16": ("RH16Ba", "RH16Bb", "RH16Bc")}


class ConnectivityError(RuntimeError):
    """Insumo o estado de conectividad no válido."""


@dataclass
class ConnectivityConfig:
    reference_dem: Path = DEFAULT_REFERENCE_DEM
    products_dir: Path = DEFAULT_PRODUCTS_DIR
    hydrography_gpkg: Path = DEFAULT_HYDRO_GPKG
    output_dir: Path = DEFAULT_OUTPUT_DIR
    subbasin_ids: tuple[str, ...] = DEFAULT_SUBBASINS
    basin_id_field: str = "_dcag_subbasin"
    streams_category_field: str = "_dcag_hydro_category"
    basins_layer: str = "subcuencas_regionales"
    municipal_layer: str = "limite_municipal"
    streams_layer: str = "corrientes_regionales"
    drainage_layer: str = "drenaje_regional"
    target_cell: tuple[int, int] | None = (9570, 114)
    depression_xy: tuple[float, float] = (565612.5, 2206747.5)
    depression_map_radius_m: float = 3000.0
    block_rows: int = 256
    traversal_batch_size: int = 250_000
    display_max_dimension: int = 1800
    overwrite: bool = False

    def __post_init__(self):
        for name in ("reference_dem", "products_dir", "hydrography_gpkg", "output_dir"):
            setattr(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.block_rows < 1 or self.traversal_batch_size < 1:
            raise ValueError("block_rows y traversal_batch_size deben ser positivos.")
        if self.display_max_dimension < 100 or self.depression_map_radius_m <= 0:
            raise ValueError("Parámetros cartográficos inválidos.")
        if len(set(self.subbasin_ids)) != len(self.subbasin_ids):
            raise ValueError("subbasin_ids contiene identificadores duplicados.")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_default(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _windows(ds, rows):
    for row in range(0, ds.height, rows):
        yield Window(0, row, ds.width, min(rows, ds.height - row))


def _open_vectors(config, target_crs):
    if not config.hydrography_gpkg.is_file():
        raise FileNotFoundError(f"No existe el GeoPackage: {config.hydrography_gpkg}")
    basins = gpd.read_file(config.hydrography_gpkg, layer=config.basins_layer)
    municipal = gpd.read_file(config.hydrography_gpkg, layer=config.municipal_layer)
    if basins.crs is None or municipal.crs is None:
        raise ConnectivityError("Las capas de subcuencas y municipio deben declarar CRS.")
    if config.basin_id_field not in basins.columns:
        raise ConnectivityError(f"Falta el campo {config.basin_id_field!r} en subcuencas.")
    basins = basins.to_crs(target_crs)
    municipal = municipal.to_crs(target_crs)
    basins = basins.loc[basins[config.basin_id_field].astype(str).isin(config.subbasin_ids)].copy()
    present = set(basins[config.basin_id_field].astype(str))
    missing = sorted(set(config.subbasin_ids) - present)
    if missing:
        raise ConnectivityError(f"Faltan subcuencas configuradas: {missing}")
    basin_geoms = [(str(row[config.basin_id_field]), row.geometry) for _, row in basins.iterrows()
                   if row.geometry is not None and not row.geometry.is_empty]
    municipality_geoms = [g for g in municipal.geometry if g is not None and not g.is_empty]
    for name, geometry in basin_geoms:
        if not geometry.is_valid:
            raise ConnectivityError(f"Geometría inválida en {name}.")
    return basins, municipal, basin_geoms, municipality_geoms


def _check_input_grids(config):
    required = {"MDE de referencia": config.reference_dem}
    required.update({name: config.products_dir / file for name, file in PRODUCTS.items()})
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan insumos: " + "; ".join(missing))
    with rasterio.open(config.reference_dem) as ref:
        if ref.crs is None or not ref.crs.is_projected:
            raise ConnectivityError("El MDE debe usar un CRS proyectado.")
        unit = (ref.crs.linear_units or "").lower()
        if unit not in {"metre", "meter", "metres", "meters"}:
            raise ConnectivityError(f"El CRS del MDE no declara unidades métricas: {unit!r}")
        reference = (ref.crs, ref.transform, ref.width, ref.height, ref.bounds)
        metadata = {
            "crs": ref.crs.to_string(), "resolucion": list(map(float, ref.res)),
            "dimensiones_columnas_filas": [ref.width, ref.height],
            "transformacion_afin": list(map(float, ref.transform))[:6],
            "extension": [float(v) for v in ref.bounds], "nodata": ref.nodata,
            "tipo_dato": {"MDE_referencia": list(ref.dtypes)},
        }
    masks = {}
    with rasterio.open(config.reference_dem) as reference_ds:
        ref_valid = sum(int(np.count_nonzero(reference_ds.read_masks(1, window=win)))
                        for win in _windows(reference_ds, config.block_rows))
        masks["MDE_referencia"] = {
            "validas": ref_valid,
            "nodata_o_invalidas": reference_ds.width * reference_ds.height - ref_valid,
            "perdidas_vs_MDE": 0, "ganadas_vs_MDE": 0,
        }
    for label, path in required.items():
        if label == "MDE de referencia":
            continue
        with rasterio.open(path) as ds:
            candidate = (ds.crs, ds.transform, ds.width, ds.height, ds.bounds)
            if candidate != reference or ds.count != 1:
                raise ConnectivityError(f"La cuadrícula de {label} no coincide exactamente con el MDE.")
            valid_count = lost = gained = 0
            with rasterio.open(config.reference_dem) as ref:
                for win in _windows(ref, config.block_rows):
                    rm = ref.read_masks(1, window=win) > 0
                    pm = ds.read_masks(1, window=win) > 0
                    valid_count += int(pm.sum())
                    lost += int(np.count_nonzero(rm & ~pm))
                    gained += int(np.count_nonzero(~rm & pm))
            masks[label] = {"validas": valid_count,
                            "nodata_o_invalidas": ds.width * ds.height - valid_count,
                            "perdidas_vs_MDE": lost, "ganadas_vs_MDE": gained,
                            "nodata_declarado": ds.nodata}
            metadata["tipo_dato"][label] = list(ds.dtypes)
    metadata["mascaras_por_raster"] = masks
    return metadata


def _rasterize_basins(config, basin_geoms, shape, transform):
    code_by_name = {name: idx + 1 for idx, name in enumerate(config.subbasin_ids)}
    shapes = [(geom, code_by_name[name]) for name, geom in basin_geoms]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                     dtype="uint8", all_touched=False)


def _fill_direction_memmap(config, path, shape):
    height, width = shape
    direction = np.memmap(path, mode="w+", dtype=np.uint8, shape=(height * width,))
    direction[:] = 255  # 255 representa NoData en esta estructura temporal compacta.
    with rasterio.open(config.products_dir / PRODUCTS["direccion_d8"]) as src:
        for win in _windows(src, config.block_rows):
            data = src.read(1, window=win)
            valid = src.read_masks(1, window=win) > 0
            bad = valid & ~np.isin(data, (0, *D8_OFFSETS.keys()))
            if bad.any():
                values, counts = np.unique(data[bad], return_counts=True)
                raise ConnectivityError(f"Códigos D8 no reconocidos: {dict(zip(values.tolist(), counts.tolist()))}")
            row0, row1 = int(win.row_off), int(win.row_off + win.height)
            direction[row0 * width:row1 * width] = np.where(valid, data, 255).astype(np.uint8).ravel()
    direction.flush()
    return direction


def _fill_basin_memmap(config, path, shape, transform, basin_geoms):
    height, width = shape
    basin = np.memmap(path, mode="w+", dtype=np.uint8, shape=(height * width,))
    basin[:] = 0
    for row0 in range(0, height, config.block_rows):
        win = Window(0, row0, width, min(config.block_rows, height - row0))
        basin_mask = _rasterize_basins(config, basin_geoms, (int(win.height), width),
                                       window_transform(win, transform))
        start = row0 * width
        basin[start:start + basin_mask.size] = basin_mask.ravel()
    basin.flush()
    return basin


def _outlet_seeds(direction, height, width, batch_size):
    """Yield (flat index, kind) for all graph sinks and edges to invalid/outside."""
    total = height * width
    offsets = tuple(D8_OFFSETS.items())
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = np.arange(start, stop, dtype=np.uint32)
        dirs = np.asarray(direction[start:stop])
        valid_local = np.flatnonzero(dirs != 255)
        if not valid_local.size:
            continue
        zero = valid_local[dirs[valid_local] == 0]
        if zero.size:
            yield flat[zero], np.full(zero.size, 0, dtype=np.uint8)  # D8=0
        rows = flat[valid_local].astype(np.int64) // width
        cols = flat[valid_local].astype(np.int64) % width
        vals = dirs[valid_local]
        edge_parts, nodata_parts = [], []
        for code, (dr, dc) in offsets:
            sel = vals == code
            if not sel.any():
                continue
            idx = flat[valid_local[sel]]
            rr, cc = rows[sel] + dr, cols[sel] + dc
            outside = (rr < 0) | (rr >= height) | (cc < 0) | (cc >= width)
            if outside.any():
                edge_parts.append(idx[outside])
            inside_ids = idx[~outside]
            inside_rr, inside_cc = rr[~outside], cc[~outside]
            if inside_ids.size:
                target = np.asarray(direction[inside_rr * width + inside_cc])
                missing = target == 255
                if missing.any():
                    nodata_parts.append(inside_ids[missing])
        if edge_parts:
            edge = np.concatenate(edge_parts)
            yield edge, np.full(edge.size, 1, dtype=np.uint8)  # direction exits raster
        if nodata_parts:
            nodata = np.concatenate(nodata_parts)
            yield nodata, np.full(nodata.size, 2, dtype=np.uint8)  # enters D8 NoData


def _assign_all_outlets(direction, labels, queue, height, width, batch_size):
    """Etiqueta cada celda por su salida terminal con BFS global inversa."""
    sink_kind_names = {0: "D8=0", 1: "fuera_del_raster", 2: "hacia_NoData"}
    sink_records = []
    queue_write = 0
    outlet_id = 0
    for seed_indices, kinds in _outlet_seeds(direction, height, width, batch_size):
        if not seed_indices.size:
            continue
        order = np.argsort(seed_indices, kind="stable")
        seed_indices, kinds = seed_indices[order], kinds[order]
        # Cada celda tiene como máximo un destino y por ello solo puede ser una salida.
        keep = np.ones(seed_indices.size, dtype=bool)
        keep[1:] = seed_indices[1:] != seed_indices[:-1]
        seed_indices, kinds = seed_indices[keep], kinds[keep]
        next_ids = np.arange(outlet_id + 1, outlet_id + 1 + seed_indices.size, dtype=np.uint32)
        labels[seed_indices] = next_ids
        queue[queue_write:queue_write + seed_indices.size] = seed_indices.astype(np.uint32)
        queue_write += seed_indices.size
        rows, cols = seed_indices // width, seed_indices % width
        sink_records.extend({"outlet_id": int(i), "cell_index": int(cell), "row": int(row),
                             "col": int(col), "tipo": sink_kind_names[int(kind)]}
                            for i, cell, row, col, kind in zip(next_ids, seed_indices, rows, cols, kinds))
        outlet_id += seed_indices.size
    if outlet_id == 0:
        return sink_records, 0
    read = 0
    while read < queue_write:
        stop = min(read + batch_size, queue_write)
        targets = np.asarray(queue[read:stop], dtype=np.int64)
        target_rows, target_cols = targets // width, targets % width
        target_labels = np.asarray(labels[targets], dtype=np.uint32)
        for code, (dr, dc) in D8_OFFSETS.items():
            src_rows, src_cols = target_rows - dr, target_cols - dc
            inside = ((src_rows >= 0) & (src_rows < height) &
                      (src_cols >= 0) & (src_cols < width))
            if not inside.any():
                continue
            source_indices = src_rows[inside] * width + src_cols[inside]
            eligible = (np.asarray(direction[source_indices]) == code) & (np.asarray(labels[source_indices]) == 0)
            if eligible.any():
                sources = source_indices[eligible]
                labels[sources] = target_labels[inside][eligible]
                queue[queue_write:queue_write + sources.size] = sources.astype(np.uint32)
                queue_write += sources.size
        read = stop
    labels.flush()
    queue.flush()
    return sink_records, int(queue_write)


def _trace_upstream(direction, target_index, height, width, batch_size):
    """Reconstruye todos los predecesores de una celda; devuelve máscara memmap y cola."""
    n = height * width
    mask = np.memmap(target_index["mask_path"], mode="w+", dtype=np.uint8, shape=(n,))
    mask[:] = 0
    queue = np.memmap(target_index["queue_path"], mode="w+", dtype=np.uint32, shape=(n,))
    if direction[target_index["index"]] == 255:
        raise ConnectivityError("La celda objetivo no es válida en el raster D8.")
    mask[target_index["index"]] = 1
    queue[0] = target_index["index"]
    read, write = 0, 1
    while read < write:
        stop = min(read + batch_size, write)
        targets = np.asarray(queue[read:stop], dtype=np.int64)
        target_rows, target_cols = targets // width, targets % width
        for code, (dr, dc) in D8_OFFSETS.items():
            sr, sc = target_rows - dr, target_cols - dc
            inside = (sr >= 0) & (sr < height) & (sc >= 0) & (sc < width)
            if not inside.any():
                continue
            source_indices = sr[inside] * width + sc[inside]
            new = ((np.asarray(direction[source_indices]) == code) &
                   (np.asarray(mask[source_indices]) == 0))
            if new.any():
                sources = source_indices[new]
                mask[sources] = 1
                queue[write:write + sources.size] = sources.astype(np.uint32)
                write += sources.size
        read = stop
    mask.flush()
    queue.flush()
    return mask, queue, int(write)


def _cycle_diagnostics(direction, labels, height, width, batch_size, scratch_dir):
    """En residuales sin salida, encuentra exactamente nodos cíclicos y ciclos."""
    total = height * width
    state = np.memmap(scratch_dir / "cycle_state.bin", mode="w+", dtype=np.uint8, shape=(total,))
    token = np.memmap(scratch_dir / "cycle_token.bin", mode="w+", dtype=np.uint32, shape=(total,))
    step_no = np.memmap(scratch_dir / "cycle_step.bin", mode="w+", dtype=np.uint32, shape=(total,))
    path = np.memmap(scratch_dir / "cycle_path.bin", mode="w+", dtype=np.uint32, shape=(total,))
    state[:] = 0
    token[:] = 0
    cycle_count = cycle_cells = residual_count = 0
    next_token = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        residual = np.flatnonzero((np.asarray(direction[start:stop]) != 255) &
                                  (np.asarray(labels[start:stop]) == 0)) + start
        residual_count += int(residual.size)
        candidates = residual[np.asarray(state[residual]) == 0]
        for seed in candidates:
            if state[seed] != 0:
                continue
            next_token += 1
            if next_token > UINT32_MAX:
                raise ConnectivityError("Demasiados componentes para el identificador temporal de ciclos.")
            current = int(seed)
            path_len = 0
            while 0 <= current < total and direction[current] != 255 and labels[current] == 0 and state[current] == 0:
                code = int(direction[current])
                if code == 0:
                    break
                row, col = divmod(current, width)
                dr, dc = D8_OFFSETS[code]
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    break
                token[current] = next_token
                step_no[current] = path_len
                path[path_len] = current
                path_len += 1
                state[current] = 1
                current = nr * width + nc
            if current >= 0 and current < total and token[current] == next_token:
                cycle_count += 1
                cycle_cells += path_len - int(step_no[current])
            if path_len:
                path_cells = np.asarray(path[:path_len], dtype=np.uint32)
                state[path_cells] = 2
    for array in (state, token, step_no, path):
        array.flush()
    return {"celdas_sin_salida_terminal": residual_count,
            "numero_ciclos_dirigidos": cycle_count,
            "celdas_pertenecientes_a_ciclos": cycle_cells,
            "nota": "El residual incluye celdas de árboles que desembocan en ciclos, además de las celdas cíclicas."}


def _aggregate_outlets(direction, labels, basin_ids, sinks, shape, batch_size, scratch_path, pixel_area_m2):
    height, width = shape
    n_outlets = len(sinks)
    counts = np.memmap(scratch_path, mode="w+", dtype=np.uint32, shape=(n_outlets + 1, 6))
    counts[:] = 0
    total = height * width
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        dirs = np.asarray(direction[start:stop])
        outlet = np.asarray(labels[start:stop])
        basin = np.asarray(basin_ids[start:stop])
        valid = (dirs != 255) & (outlet > 0)
        if not valid.any():
            continue
        pair = outlet[valid].astype(np.uint64) * 6 + basin[valid].astype(np.uint64)
        unique, occurrences = np.unique(pair, return_counts=True)
        ids, basin_codes = unique // 6, unique % 6
        np.add.at(counts, (ids.astype(np.intp), basin_codes.astype(np.intp)), occurrences.astype(np.uint32))
    counts.flush()
    records = []
    for sink in sinks:
        values = np.asarray(counts[sink["outlet_id"]], dtype=np.uint64)
        counts_by_basin = {str(i): int(values[i]) for i in range(1, 6)}
        rh15 = int(values[1] + values[2])
        rh16 = int(values[3] + values[4] + values[5])
        exterior = int(values[0])
        records.append({**sink, "celdas_contribuyentes": int(values.sum()),
                        "area_contribuyente_km2": float(values.sum() * pixel_area_m2 / 1_000_000),
                        "celdas_RH15": rh15, "celdas_RH16": rh16,
                        "celdas_exteriores_a_subcuencas": exterior,
                        "contribucion_ambas_RH": bool(rh15 and rh16),
                        "celdas_por_codigo_subcuenca": counts_by_basin})
    del counts
    return records


def _sample_maxima(path, block_rows):
    maximum = -math.inf
    cells = []
    with rasterio.open(path) as ds:
        for window in _windows(ds, block_rows):
            mask = ds.read_masks(1, window=window) > 0
            values = ds.read(1, window=window)
            valid = values[mask & np.isfinite(values)]
            if not valid.size:
                continue
            block_max = float(valid.max())
            if block_max > maximum:
                maximum, cells = block_max, []
            if block_max == maximum:
                for row, col in np.argwhere(mask & (values == maximum)):
                    cells.append((int(row + window.row_off), int(col + window.col_off)))
    return maximum, cells


def _mask_counts(mask, basin_ids, shape, batch_size):
    n = shape[0] * shape[1]
    counts = np.zeros(6, dtype=np.uint64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        cells = np.asarray(mask[start:stop]) > 0
        if cells.any():
            counts += np.bincount(np.asarray(basin_ids[start:stop])[cells], minlength=6).astype(np.uint64)
    return counts


def _float32_accumulation_diagnostic(stored_value, reconstructed_count):
    stored = np.float32(stored_value)
    ulp = float(abs(np.spacing(stored)))
    difference = float(stored) - int(reconstructed_count)
    return {
        "valor_almacenado_float32": float(stored),
        "conteo_reconstruido": int(reconstructed_count),
        "diferencia_almacenado_menos_reconstruido": difference,
        "ulp_float32_en_valor_almacenado": ulp,
        "tipo_dato": "float32",
        "umbral_enteros_exactos": FLOAT32_EXACT_INTEGER_LIMIT,
        "supera_umbral_enteros_exactos": float(stored) > FLOAT32_EXACT_INTEGER_LIMIT,
        "diferencia_menor_o_igual_a_media_ulp": abs(difference) <= ulp / 2,
        "nota": "ULP describe el espaciado local representable; no acota el error acumulado de redondeos de toda la red.",
    }


def _get_attributes(basins):
    result = {}
    for _, row in basins.iterrows():
        name = str(row.get("_dcag_subbasin", ""))
        result[name] = {}
        for wanted in ("DRENAJE", "DESCARGA"):
            col = next((c for c in row.index if str(c).upper() == wanted), None)
            result[name][wanted] = None if col is None else row[col]
    return result


def _transition_table(config, vectors, shape, transform, scratch_dir, batch_size):
    height, width = shape
    path = scratch_dir / "d8_codes.bin"
    direction = np.memmap(path, mode="r", dtype=np.uint8, shape=(height * width,))
    basin_path = scratch_dir / "basin_ids.bin"
    basin_map = np.memmap(basin_path, mode="r", dtype=np.uint8, shape=(height * width,))
    code_by_name = {idx + 1: name for idx, name in enumerate(config.subbasin_ids)}
    basin_attributes = _get_attributes(vectors[0])
    transitions = []
    rasters = {key: rasterio.open(config.products_dir / PRODUCTS[key])
               for key in ("mde_acondicionado", "profundidad_depresiones", "acumulacion_celdas")}
    original = rasterio.open(config.reference_dem)
    try:
        for row0 in range(0, height, config.block_rows):
            core_height = min(config.block_rows, height - row0)
            halo_start, halo_end = max(0, row0 - 1), min(height, row0 + core_height + 1)
            halo_height = halo_end - halo_start
            halo_window = Window(0, halo_start, width, halo_height)
            core_slice = slice(row0 - halo_start, row0 - halo_start + core_height)
            code_halo = np.asarray(direction[halo_start * width:halo_end * width]).reshape(halo_height, width)
            basin_halo = np.asarray(basin_map[halo_start * width:halo_end * width]).reshape(halo_height, width)
            code_block = code_halo[core_slice]
            for code, (dr, dc) in D8_OFFSETS.items():
                rr, cc = np.where(code_block == code)
                if not len(rr):
                    continue
                src_global_rows = rr + row0
                dst_global_rows, dst_cols = src_global_rows + dr, cc + dc
                inside = ((dst_global_rows >= 0) & (dst_global_rows < height) &
                          (dst_cols >= 0) & (dst_cols < width))
                rr, cc = rr[inside], cc[inside]
                src_global_rows, dst_global_rows, dst_cols = (src_global_rows[inside],
                                                              dst_global_rows[inside], dst_cols[inside])
                src_local, dst_local = src_global_rows - halo_start, dst_global_rows - halo_start
                source_basin = basin_halo[src_local, cc]
                dest_basin = basin_halo[dst_local, dst_cols]
                crossing = (source_basin > 0) & (dest_basin > 0) & (source_basin != dest_basin)
                if not crossing.any():
                    continue
                src_r, src_c = src_global_rows[crossing], cc[crossing]
                dst_r, dst_c = dst_global_rows[crossing], dst_cols[crossing]
                orig = original.read(1, window=halo_window)
                orig_mask = original.read_masks(1, window=halo_window) > 0
                cond = rasters["mde_acondicionado"].read(1, window=halo_window)
                cond_mask = rasters["mde_acondicionado"].read_masks(1, window=halo_window) > 0
                depth = rasters["profundidad_depresiones"].read(1, window=halo_window)
                depth_mask = rasters["profundidad_depresiones"].read_masks(1, window=halo_window) > 0
                accum = rasters["acumulacion_celdas"].read(1, window=halo_window)
                accum_mask = rasters["acumulacion_celdas"].read_masks(1, window=halo_window) > 0
                for sr, sc, trr, tcc, sb, db in zip(src_r, src_c, dst_r, dst_c,
                                                    source_basin[crossing], dest_basin[crossing]):
                    sr_local, tr_local = int(sr) - halo_start, int(trr) - halo_start
                    values = {}
                    for key, data, data_mask in (("elevacion_original_m", orig, orig_mask),
                                                 ("elevacion_acondicionada_m", cond, cond_mask),
                                                 ("profundidad_relleno_m", depth, depth_mask)):
                        values[key] = float(data[sr_local, int(sc)]) if data_mask[sr_local, int(sc)] else None
                    transitions.append({
                        "fila_origen": int(sr), "columna_origen": int(sc),
                        "x_origen": float((transform * (int(sc) + 0.5, int(sr) + 0.5))[0]),
                        "y_origen": float((transform * (int(sc) + 0.5, int(sr) + 0.5))[1]),
                        "fila_destino": int(trr), "columna_destino": int(tcc),
                        "x_destino": float((transform * (int(tcc) + 0.5, int(trr) + 0.5))[0]),
                        "y_destino": float((transform * (int(tcc) + 0.5, int(trr) + 0.5))[1]),
                        "subcuenca_origen": code_by_name[int(sb)], "subcuenca_destino": code_by_name[int(db)],
                        "DRENAJE_origen": basin_attributes.get(code_by_name[int(sb)], {}).get("DRENAJE"),
                        "DESCARGA_origen": basin_attributes.get(code_by_name[int(sb)], {}).get("DESCARGA"),
                        "DRENAJE_destino": basin_attributes.get(code_by_name[int(db)], {}).get("DRENAJE"),
                        "DESCARGA_destino": basin_attributes.get(code_by_name[int(db)], {}).get("DESCARGA"),
                        "region_origen": code_by_name[int(sb)][:4], "region_destino": code_by_name[int(db)][:4],
                        "cruce_RH15_RH16": code_by_name[int(sb)][:4] != code_by_name[int(db)][:4],
                        "tipo_transicion": "directa_entre_subcuencas", **values,
                        "direccion_D8": int(code),
                        "acumulacion_origen_celdas": (float(accum[sr_local, int(sc)])
                                                       if accum_mask[sr_local, int(sc)] else None),
                        "acumulacion_destino_celdas": (float(accum[tr_local, int(tcc)])
                                                        if accum_mask[tr_local, int(tcc)] else None),
                    })
    finally:
        for ds in rasters.values():
            ds.close()
        original.close()
    by_value = sorted((r["acumulacion_origen_celdas"] or 0.0 for r in transitions))
    for record in transitions:
        value = record["acumulacion_origen_celdas"] or 0.0
        record["percentil_empirico_acumulacion_cruces"] = (
            100.0 * np.searchsorted(by_value, value, side="right") / len(by_value) if by_value else None
        )
    return transitions


def _depression_component(depth, valid, height, width, row, col):
    """Componente 8-conexa de profundidad positiva que contiene la celda máxima."""
    if not valid[row, col] or depth[row, col] <= 0:
        return np.zeros(depth.shape, dtype=bool)
    mask = valid & np.isfinite(depth) & (depth > 0)
    component = np.zeros(mask.shape, dtype=bool)
    stack = [(row, col)]
    component[row, col] = True
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if ((dr or dc) and 0 <= nr < height and 0 <= nc < width
                        and mask[nr, nc] and not component[nr, nc]):
                    component[nr, nc] = True
                    stack.append((nr, nc))
    return component


def _save_overview_map(config, path, ref_path, catchment, basin_geoms, municipality_geoms,
                       outlet_records, basin_names, target_cell, transitions, crs, transform):
    with rasterio.open(ref_path) as src:
        scale = max(src.width / config.display_max_dimension, src.height / config.display_max_dimension, 1)
        out_w, out_h = max(1, round(src.width / scale)), max(1, round(src.height / scale))
        dem = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest).astype(float)
        mask = src.read_masks(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest) > 0
        bounds = src.bounds
    dem[~mask] = np.nan
    row_idx = np.minimum((np.arange(out_h) * src.height / out_h).astype(int), src.height - 1)
    col_idx = np.minimum((np.arange(out_w) * src.width / out_w).astype(int), src.width - 1)
    catch = np.asarray(catchment.reshape(src.height, src.width)[np.ix_(row_idx, col_idx)]) > 0
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax.imshow(dem, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin="upper", cmap="terrain")
    rgba = np.zeros((*catch.shape, 4), dtype=np.float32)
    rgba[catch] = (0.9, 0.2, 0.1, 0.45)
    ax.imshow(rgba, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin="upper")
    basin_colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    basin_handles = []
    for idx, (name, geom) in enumerate(basin_geoms):
        color = basin_colors[idx % len(basin_colors)]
        gpd.GeoSeries([geom], crs=crs).boundary.plot(ax=ax, color=color, linewidth=1.0)
        basin_handles.append(Line2D([0], [0], color=color, label=name))
    for geom in municipality_geoms:
        gpd.GeoSeries([geom], crs=crs).boundary.plot(ax=ax, color="#12a4d9", linewidth=1.2)
    if outlet_records:
        xs, ys, sizes = [], [], []
        for out in outlet_records:
            x, y = transform * (out["col"] + 0.5, out["row"] + 0.5)
            xs.append(x); ys.append(y)
            sizes.append(5 + 12 * math.log1p(out["celdas_contribuyentes"]))
        ax.scatter(xs, ys, s=sizes, c="#2812a4", alpha=0.6, linewidths=0, label="Salidas")
    if transitions:
        selected = [r for r in transitions if r["cruce_RH15_RH16"]]
        ax.scatter([r["x_origen"] for r in selected], [r["y_origen"] for r in selected],
                   s=7, marker="x", c="#ff00cc", alpha=0.75, label="Cruce RH15–RH16")
    target_x, target_y = transform * (target_cell[1] + 0.5, target_cell[0] + 0.5)
    ax.scatter([target_x], [target_y], marker="*", s=130, c="#ffff00", edgecolors="black", label="Máxima acumulación")
    legend = basin_handles + [Line2D([0], [0], color="#12a4d9", label="Villa de Álvarez"),
                              Patch(facecolor=(0.9, .2, .1, .45), label="Área contribuyente reconstruida"),
                              Line2D([], [], linestyle="none", label="Exterior a las cinco subcuencas")]
    legend += ax.get_legend_handles_labels()[0]
    ax.legend(handles=legend, loc="lower left", fontsize=7, ncol=2)
    ax.set_title("Área contribuyente a la acumulación D8 máxima\nDominio regional; recuadro municipal del caso piloto")
    ax.set_xlabel("Este (m), " + crs.to_string())
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_territorial_boundaries(ax, basin_geoms, municipality_geoms, crs):
    palette = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    handles = []
    for idx, (name, geom) in enumerate(basin_geoms):
        color = palette[idx % len(palette)]
        gpd.GeoSeries([geom], crs=crs).boundary.plot(ax=ax, color=color, linewidth=1.0)
        handles.append(Line2D([0], [0], color=color, label=name))
    for geom in municipality_geoms:
        gpd.GeoSeries([geom], crs=crs).boundary.plot(ax=ax, color="#00a6d6", linewidth=1.3)
    handles.append(Line2D([0], [0], color="#00a6d6", label="Villa de Álvarez"))
    handles.append(Line2D([], [], linestyle="none", label="Exterior a las cinco subcuencas"))
    return handles


def _plot_outlets(path, ref_bounds, basin_geoms, municipality_geoms, outlets, crs, transform):
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    boundary_handles = _draw_territorial_boundaries(ax, basin_geoms, municipality_geoms, crs)
    rh15 = [o for o in outlets if o["celdas_RH15"]]
    rh16 = [o for o in outlets if o["celdas_RH16"]]
    mixed = [o for o in outlets if o["contribucion_ambas_RH"]]
    for group, color, label in ((rh15, "#1b9e77", "Recibe RH15"), (rh16, "#d95f02", "Recibe RH16"),
                                (mixed, "#984ea3", "Recibe ambas regiones")):
        if group:
            coords = [transform * (o["col"] + .5, o["row"] + .5) for o in group]
            sizes = [8 + 18 * math.log1p(o["celdas_contribuyentes"]) for o in group]
            ax.scatter([p[0] for p in coords], [p[1] for p in coords], s=sizes,
                       color=color, alpha=.72, label=label, edgecolors="white", linewidths=.3)
    ax.set_xlim(ref_bounds.left, ref_bounds.right); ax.set_ylim(ref_bounds.bottom, ref_bounds.top)
    ax.set_title("Salidas hidrológicas y aportación regional por terminal D8")
    ax.set_xlabel(f"Este (m), {crs.to_string()}"); ax.set_ylabel("Norte (m)")
    handles, _ = ax.get_legend_handles_labels()
    ax.set_aspect("equal"); ax.legend(handles=boundary_handles + handles, loc="best", fontsize=7, ncol=2)
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_transitions(path, basin_geoms, municipality_geoms, transitions, crs):
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    boundary_handles = _draw_territorial_boundaries(ax, basin_geoms, municipality_geoms, crs)
    selected = [r for r in transitions if r["cruce_RH15_RH16"]]
    if selected:
        values = np.log10(np.asarray([r["acumulacion_origen_celdas"] or 0 for r in selected]) + 1)
        points = ax.scatter([r["x_origen"] for r in selected], [r["y_origen"] for r in selected],
                            c=values, cmap="magma", s=12, marker="x", label="Cruce RH15–RH16")
        fig.colorbar(points, ax=ax, label="log₁₀(acumulación origen + 1), visualización")
    ax.set_title(f"Transiciones directas D8 entre RH15 y RH16 (n={len(selected):,})")
    ax.set_xlabel(f"Este (m), {crs.to_string()}"); ax.set_ylabel("Norte (m)")
    handles, _ = ax.get_legend_handles_labels()
    ax.set_aspect("equal"); ax.legend(handles=boundary_handles + handles, loc="best", fontsize=7, ncol=2)
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_maximum_zoom(path, config, target, ref, products, basin_geoms, municipality_geoms, crs):
    row, col = target
    radius = max(1000, config.depression_map_radius_m)
    r0, r1 = max(0, row - int(radius / abs(ref.transform.e))), min(ref.height, row + int(radius / abs(ref.transform.e)) + 1)
    c0, c1 = max(0, col - int(radius / abs(ref.transform.a))), min(ref.width, col + int(radius / abs(ref.transform.a)) + 1)
    window = Window(c0, r0, c1 - c0, r1 - r0)
    z = ref.read(1, window=window, masked=True)
    accum = products["acumulacion_celdas"].read(1, window=window, masked=True)
    transform = window_transform(window, ref.transform)
    bounds = rasterio.windows.bounds(window, ref.transform)
    left, bottom, right, top = bounds
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    panels = ((axes[0], z, "MDE original (m)", "terrain"),
              (axes[1], np.ma.log10(np.ma.maximum(accum, 0) + 1), "log₁₀(acumulación + 1), solo visual", "plasma"))
    x, y = ref.transform * (col + .5, row + .5)
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, extent=(left, right, bottom, top), origin="upper", cmap=cmap)
        ax.scatter([x], [y], c="red", marker="*", s=100, edgecolors="white")
        boundary_handles = _draw_territorial_boundaries(ax, basin_geoms, municipality_geoms, crs)
        ax.set_xlim(left, right); ax.set_ylim(bottom, top)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("Este (m)"); ax.set_ylabel("Norte (m)")
        if ax is axes[0]:
            ax.legend(handles=boundary_handles, loc="best", fontsize=7, ncol=2)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f"Entorno de acumulación máxima — celda ({row}, {col}), X={x:.1f}, Y={y:.1f}")
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_depression(path, config, ref, products, basin_geoms, municipality_geoms, crs):
    x, y = config.depression_xy
    row, col = ref.index(x, y)
    radius = config.depression_map_radius_m
    r0, r1 = max(0, row - int(radius / abs(ref.transform.e))), min(ref.height, row + int(radius / abs(ref.transform.e)) + 1)
    c0, c1 = max(0, col - int(radius / abs(ref.transform.a))), min(ref.width, col + int(radius / abs(ref.transform.a)) + 1)
    window = Window(c0, r0, c1 - c0, r1 - r0)
    bounds = rasterio.windows.bounds(window, ref.transform)
    left, bottom, right, top = bounds
    transform = window_transform(window, ref.transform)
    original = ref.read(1, window=window, masked=True)
    conditioned = products["mde_acondicionado"].read(1, window=window, masked=True)
    depth = products["profundidad_depresiones"].read(1, window=window, masked=True)
    valid = ~(np.ma.getmaskarray(depth) | np.ma.getmaskarray(original))
    depth_data = np.asarray(depth.filled(0), dtype=np.float32)
    component = _depression_component(depth_data, valid, depth_data.shape[0], depth_data.shape[1], row-r0, col-c0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    panels = [(axes[0, 0], original, "MDE original (m)", "terrain"),
              (axes[0, 1], conditioned, "MDE acondicionado (m)", "terrain"),
              (axes[1, 0], depth, "Profundidad de relleno (m)", "viridis")]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, extent=(left, right, bottom, top), origin="upper", cmap=cmap)
        if component.any():
            ax.contour(component.astype(np.uint8), levels=[.5], colors=["cyan"], linewidths=1.2,
                       extent=(left, right, bottom, top), origin="upper")
        ax.scatter([x], [y], c="red", marker="*", s=90, edgecolors="white")
        boundary_handles = _draw_territorial_boundaries(ax, basin_geoms, municipality_geoms, crs)
        ax.set_xlim(left, right); ax.set_ylim(bottom, top)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("Este (m)"); ax.set_ylabel("Norte (m)")
        if ax is axes[0, 0]:
            ax.legend(handles=boundary_handles, loc="best", fontsize=7, ncol=2)
        fig.colorbar(im, ax=ax, shrink=.75)
    # Perfiles de fila y columna que atraviesan la celda de profundidad máxima.
    axes[1, 1].plot(np.arange(depth_data.shape[1]) * abs(ref.transform.a) + left,
                    original[row-r0, :], label="Original")
    axes[1, 1].plot(np.arange(depth_data.shape[1]) * abs(ref.transform.a) + left,
                    conditioned[row-r0, :], label="Acondicionado")
    axes[1, 1].set_title("Perfil E–O por la depresión máxima"); axes[1, 1].set_xlabel("Este (m)"); axes[1, 1].set_ylabel("Elevación (m)")
    axes[1, 1].legend()
    fig.suptitle(f"Depresión máxima — ({row}, {col}), X={x:.1f}, Y={y:.1f}; área conectada 8-neighbor={component.sum():,} celdas")
    fig.savefig(path, dpi=180); plt.close(fig)
    profile_path = path.with_name(path.stem + "_perfiles.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(np.arange(depth_data.shape[1]) * abs(ref.transform.a) + left, original[row-r0, :], label="Original")
    axes[0].plot(np.arange(depth_data.shape[1]) * abs(ref.transform.a) + left, conditioned[row-r0, :], label="Acondicionado")
    axes[0].set_title("Perfil E–O"); axes[0].set_xlabel("Este (m)"); axes[0].set_ylabel("Elevación (m)"); axes[0].legend()
    axes[1].plot(np.arange(depth_data.shape[0]) * abs(ref.transform.e) * -1 + top, original[:, col-c0], label="Original")
    axes[1].plot(np.arange(depth_data.shape[0]) * abs(ref.transform.e) * -1 + top, conditioned[:, col-c0], label="Acondicionado")
    axes[1].set_title("Perfil N–S"); axes[1].set_xlabel("Norte (m)"); axes[1].set_ylabel("Elevación (m)"); axes[1].legend()
    fig.savefig(profile_path, dpi=180); plt.close(fig)
    return {"fila": row, "columna": col, "x": x, "y": y,
            "profundidad_en_celda_configurada_m": float(depth_data[row-r0, col-c0]),
            "celdas_componente_8_conexa": int(component.sum()),
            "superficie_componente_km2": float(component.sum() * abs(ref.transform.a * ref.transform.e) / 1e6),
            "mapa": str(path), "mapa_perfiles": str(profile_path),
            "metodo_delimitacion": "componente 8-conexa del raster profundidad>0 dentro de la ventana cartográfica; no es delimitación hidrodinámica."}


class HydrologicalConnectivity:
    def __init__(self, config: ConnectivityConfig):
        self.config = config
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        stem = self.run_id
        self.outputs = {
            "json": config.output_dir / f"conectividad_hidrologica_{stem}.json",
            "aportaciones_csv": config.output_dir / f"aportaciones_area_maxima_{stem}.csv",
            "salidas_csv": config.output_dir / f"salidas_D8_{stem}.csv",
            "transiciones_csv": config.output_dir / f"transiciones_RH15_RH16_{stem}.csv",
            "mapa_area_maxima": config.output_dir / f"area_contribuyente_maxima_{stem}.png",
            "mapa_salidas": config.output_dir / f"salidas_hidrologicas_{stem}.png",
            "mapa_transiciones": config.output_dir / f"transiciones_RH15_RH16_{stem}.png",
            "mapa_maximo_zoom": config.output_dir / f"entorno_acumulacion_maxima_{stem}.png",
            "mapa_depresion": config.output_dir / f"depresion_maxima_{stem}.png",
        }

    def run(self):
        started = time.perf_counter()
        grid = _check_input_grids(self.config)
        if self.config.target_cell is None:
            maximum, max_cells = _sample_maxima(self.config.products_dir / PRODUCTS["acumulacion_celdas"], self.config.block_rows)
            if not max_cells:
                raise ConnectivityError("No hay celdas válidas en acumulación.")
            target = max_cells[0]
        else:
            target = tuple(map(int, self.config.target_cell))
            maximum, max_cells = _sample_maxima(self.config.products_dir / PRODUCTS["acumulacion_celdas"], self.config.block_rows)
            if target not in max_cells:
                raise ConnectivityError(f"La celda configurada {target} no es máxima; máximo observado={maximum}, celdas={max_cells[:10]}")
        if any(path.exists() for path in self.outputs.values()) and not self.config.overwrite:
            raise FileExistsError("Alguna salida diagnóstica ya existe; no se sobrescribe.")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        basins, municipal, basin_geoms, municipality_geoms = _open_vectors(self.config, rasterio.crs.CRS.from_string(grid["crs"]))
        attributes = _get_attributes(basins)
        height, width = grid["dimensiones_columnas_filas"][1], grid["dimensiones_columnas_filas"][0]
        shape = (height, width)
        n = height * width
        affine = rasterio.Affine(*grid["transformacion_afin"])
        pixel_area = abs(affine.a * affine.e - affine.b * affine.d)
        if n > UINT32_MAX:
            raise ConnectivityError("El dominio excede la capacidad de índices uint32; aumente la estrategia de almacenamiento.")
        required_scratch_bytes = n * (1 + 1 + 4 + 4 + 1 + 4 + 4 + 4 + 1 + 4)
        scratch_root = self.config.output_dir
        with tempfile.TemporaryDirectory(prefix=f".tmp_connectivity_{self.run_id}_", dir=scratch_root,
                                         ignore_cleanup_errors=True) as temp_name:
            scratch = Path(temp_name)
            direction = _fill_direction_memmap(self.config, scratch / "d8_codes.bin", shape)
            basin_map = _fill_basin_memmap(self.config, scratch / "basin_ids.bin", shape, affine, basin_geoms)
            labels = np.memmap(scratch / "outlet_labels.bin", mode="w+", dtype=np.uint32, shape=(n,))
            labels[:] = 0
            queue = np.memmap(scratch / "global_queue.bin", mode="w+", dtype=np.uint32, shape=(n,))
            sinks, routed = _assign_all_outlets(direction, labels, queue, height, width,
                                                self.config.traversal_batch_size)
            outlet_counts = _aggregate_outlets(direction, labels, basin_map, sinks, shape,
                                                self.config.traversal_batch_size, scratch / "outlet_counts.bin", pixel_area)
            for outlet in outlet_counts:
                outlet["x"], outlet["y"] = affine * (outlet["col"] + .5, outlet["row"] + .5)
                outlet["terminal_en_exterior_a_subcuencas"] = bool(basin_map[outlet["cell_index"]] == 0)
                outlet["celdas_por_subcuenca"] = {
                    name: outlet["celdas_por_codigo_subcuenca"].get(str(code), 0)
                    for code, name in enumerate(self.config.subbasin_ids, start=1)
                }
            cycle_info = _cycle_diagnostics(direction, labels, height, width,
                                            self.config.traversal_batch_size, scratch)
            max_index = target[0] * width + target[1]
            catchment, catch_queue, catchment_count = _trace_upstream(
                direction, {"index": max_index, "mask_path": scratch / "max_catchment.bin",
                            "queue_path": scratch / "max_queue.bin"}, height, width,
                self.config.traversal_batch_size)
            catchment_by_zone = _mask_counts(catchment, basin_map, shape, self.config.traversal_batch_size)
            catchment_area = catchment_count * pixel_area / 1e6
            accumulator = rasterio.open(self.config.products_dir / PRODUCTS["acumulacion_celdas"])
            try:
                stored_max = float(accumulator.read(1, window=Window(target[1], target[0], 1, 1))[0, 0])
            finally:
                accumulator.close()
            float32_report = _float32_accumulation_diagnostic(stored_max, catchment_count)
            discrepancy = float32_report["diferencia_almacenado_menos_reconstruido"]
            basin_names_by_code = {idx + 1: name for idx, name in enumerate(self.config.subbasin_ids)}
            contribution_rows = []
            for code, name in basin_names_by_code.items():
                cells = int(catchment_by_zone[code])
                contribution_rows.append({"subcuenca": name, "celdas_area_maxima": cells,
                                          "superficie_km2": cells * pixel_area / 1e6,
                                          "DRENAJE": attributes.get(name, {}).get("DRENAJE"),
                                          "DESCARGA": attributes.get(name, {}).get("DESCARGA")})
            contribution_rows.append({"subcuenca": "territorio_exterior_a_las_cinco",
                                      "celdas_area_maxima": int(catchment_by_zone[0]),
                                      "superficie_km2": catchment_by_zone[0] * pixel_area / 1e6,
                                      "DRENAJE": None, "DESCARGA": None})
            transitions = _transition_table(self.config, (basins, municipal), shape,
                                            affine, scratch,
                                            self.config.traversal_batch_size)
            cross = [item for item in transitions if item["cruce_RH15_RH16"]]
            mixed_outlets = [item for item in outlet_counts if item["contribucion_ambas_RH"]]
            indirect_outlets = [item for item in mixed_outlets if item["terminal_en_exterior_a_subcuencas"]]
            with rasterio.open(self.config.reference_dem) as ref:
                target_x, target_y = ref.transform * (target[1] + .5, target[0] + .5)
                target_elev = ref.read(1, window=Window(target[1], target[0], 1, 1))[0, 0].item()
                target_crs = ref.crs
                bounds = ref.bounds
                transform = ref.transform
                pixel_area = abs(float(ref.transform.a * ref.transform.e - ref.transform.b * ref.transform.d))
            conditioned_path = self.config.products_dir / PRODUCTS["mde_acondicionado"]
            with rasterio.open(conditioned_path) as conditioned:
                target_conditioned_valid = bool(conditioned.read_masks(1, window=Window(target[1], target[0], 1, 1))[0, 0])
                target_conditioned = conditioned.read(1, window=Window(target[1], target[0], 1, 1))[0, 0]
            target_basin_code = int(basin_map[max_index])
            target_basin_name = (self.config.subbasin_ids[target_basin_code - 1]
                                 if target_basin_code else "territorio_exterior_a_las_cinco")
            max_cell_report = {"fila": target[0], "columna": target[1], "x": target_x, "y": target_y,
                               "subcuenca": target_basin_name,
                               "elevacion_original_m": target_elev,
                               "elevacion_acondicionada_m": float(target_conditioned) if target_conditioned_valid else None,
                               "direccion_D8": int(direction[max_index]),
                               "salida_terminal_id": int(labels[max_index]),
                               "acumulacion_almacenada_float32": stored_max,
                               "celdas_reconstruidas_inversamente": catchment_count,
                               "diferencia_almacenado_menos_reconstruido": discrepancy,
                               "diferencia_relativa": discrepancy / catchment_count if catchment_count else None,
                               "area_reconstruida_km2": catchment_area,
                               "area_segun_acumulacion_km2": stored_max * pixel_area / 1e6,
                               "precision_float32": float32_report}
            outlet_records = outlet_counts
            # Reporta métricas de red regional y mantiene salidas completas separadas.
            terminal_summary = {
                "salidas_terminales": len(sinks),
                "por_tipo": {kind: sum(x["tipo"] == kind for x in sinks)
                             for kind in ("D8=0", "fuera_del_raster", "hacia_NoData")},
                "celdas_enroutadas_a_salidas": routed,
                "conservacion_celdas_validas": {"D8_validas": int(grid["mascaras_por_raster"]["direccion_d8"]["validas"]),
                                               "asignadas_a_salida": routed,
                                               "sin_salida_terminal": cycle_info["celdas_sin_salida_terminal"],
                                               "suma": routed + cycle_info["celdas_sin_salida_terminal"]},
                "por_salida": outlet_records,
                "salidas_con_aportaciones_de_ambas_RH": len(indirect_outlets),
                "conexiones_indirectas_via_dominio_exterior": [x["outlet_id"] for x in indirect_outlets],
            }
            depth_max, depth_cells = _sample_maxima(self.config.products_dir / PRODUCTS["profundidad_depresiones"],
                                                    self.config.block_rows)
            # Los mapas utilizan únicamente una versión muestreada para dibujo.
            _save_overview_map(self.config, self.outputs["mapa_area_maxima"], self.config.reference_dem,
                               catchment, basin_geoms, municipality_geoms, outlet_records, basin_names_by_code,
                               target, transitions, target_crs, transform)
            _plot_outlets(self.outputs["mapa_salidas"], bounds, basin_geoms, municipality_geoms,
                          outlet_records, target_crs, transform)
            _plot_transitions(self.outputs["mapa_transiciones"], basin_geoms, municipality_geoms,
                              transitions, target_crs)
            with rasterio.open(self.config.reference_dem) as ref, rasterio.open(
                self.config.products_dir / PRODUCTS["acumulacion_celdas"]) as acc:
                _plot_maximum_zoom(self.outputs["mapa_maximo_zoom"], self.config, target, ref,
                                   {"acumulacion_celdas": acc}, basin_geoms, municipality_geoms, target_crs)
            with rasterio.open(self.config.reference_dem) as ref, rasterio.open(
                self.config.products_dir / PRODUCTS["mde_acondicionado"]) as cond, rasterio.open(
                self.config.products_dir / PRODUCTS["profundidad_depresiones"]) as dep:
                depression_report = _plot_depression(self.outputs["mapa_depresion"], self.config, ref,
                                                     {"mde_acondicionado": cond,
                                                      "profundidad_depresiones": dep},
                                                     basin_geoms, municipality_geoms, target_crs)
            dep_point = Point(self.config.depression_xy)
            depression_report["subcuenca"] = next(
                (name for name, geom in basin_geoms if geom.covers(dep_point)),
                "territorio_exterior_a_las_cinco",
            )
            depression_report["coincide_con_celda_de_profundidad_maxima"] = any(
                (cell_row, cell_col) == (depression_report["fila"], depression_report["columna"])
                for cell_row, cell_col in depth_cells
            )
            # Escribe las salidas tabulares sin conservar una segunda copia de los rasters.
            self._write_csv(self.outputs["aportaciones_csv"],
                            ["subcuenca", "celdas_area_maxima", "superficie_km2", "DRENAJE", "DESCARGA"],
                            contribution_rows)
            self._write_csv(self.outputs["salidas_csv"],
                            ["outlet_id", "row", "col", "x", "y", "tipo", "terminal_en_exterior_a_subcuencas",
                             "celdas_contribuyentes", "area_contribuyente_km2",
                             "celdas_RH15", "celdas_RH16", "celdas_exteriores_a_subcuencas",
                             "contribucion_ambas_RH", "celdas_por_subcuenca", "celdas_por_codigo_subcuenca"], outlet_records)
            self._write_csv(self.outputs["transiciones_csv"], list(transitions[0].keys()) if transitions else
                            ["fila_origen", "columna_origen", "fila_destino", "columna_destino"], transitions)
            report = {
                "run_id": self.run_id, "generado_utc": datetime.now(timezone.utc).isoformat(),
                "caso_piloto": "Villa de Álvarez, Colima",
                "dominio_hidrologico": "Regional Colima–Jalisco; cinco subcuencas, con aportaciones externas preservadas.",
                "configuracion": _json_default(asdict(self.config)), "cuadricula": grid,
                "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                              "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                              "geopandas": gpd.__version__},
                "memoria_y_estrategia": {
                    "metodo": "Grafo D8 global; lectura en ventanas y propagación inversa FIFO. No se tratan ventanas como cuencas independientes.",
                    "almacenamiento_temporal": "memmap uint8 para D8/máscara territorial, uint32 para etiquetas y colas; temporales eliminados al finalizar.",
                    "memoria_residente_estimada": (
                        "Orden de cientos de MiB de arreglos temporales por lote, más caché de Rasterio/GDAL y "
                        "GeoPandas. Es una estimación de diseño, no una cota; el sistema operativo puede cachear páginas memmap."
                    ),
                    "espacio_temporal_estimado_bytes": required_scratch_bytes + (len(sinks) + 1) * 6 * 4,
                    "batch_size": self.config.traversal_batch_size,
                    "umbral_indices_uint32": int(UINT32_MAX),
                    "cuadricula_bytes_estimados_por_celda_temporal": 28,
                },
                "mascaras_por_raster": grid["mascaras_por_raster"],
                "area_contribuyente_maxima": max_cell_report,
                "aportacion_por_subcuenca_y_exterior": contribution_rows,
                "salidas_hidrologicas": terminal_summary,
                "ciclos_y_trayectorias_sin_salida": cycle_info,
                "transiciones_D8_entre_subcuencas": {
                    "total": len(transitions), "RH15_a_RH16": sum(x["cruce_RH15_RH16"] for x in transitions),
                    "directas": "Cada arista entre celdas de subcuencas distintas se registra una sola vez por celda origen.",
                    "principales_por_acumulacion": sorted(cross, key=lambda x: x["acumulacion_origen_celdas"] or 0, reverse=True)[:100],
                    "salidas_comunes_de_RH15_y_RH16": [x["outlet_id"] for x in indirect_outlets],
                    "salidas_comunes_RH15_RH16_independientemente_de_ubicacion": [x["outlet_id"] for x in mixed_outlets],
                    "atributos_INEGI_DRENAJE_DESCARGA": attributes,
                },
                "depresion_maxima": {"profundidad_maxima_m": depth_max, "celdas_maximas": depth_cells,
                                     **depression_report},
                "criterios": {
                    "D8": "WhiteboxTools nativo esri_pntr=False; offsets por código verificados contra WhiteboxTools 2.4.0.",
                    "salidas": "D8=0, dirección fuera del raster y dirección hacia NoData se identifican separadamente.",
                    "area_maxima": "Reconstrucción inversa desde la celda objetivo por aristas D8; no se usa el valor de acumulación como máscara de pertenencia.",
                    "cuencas": "DEFAULT_SUBBASINS filtra los polígonos para codificar territorio; todos los segmentos D8 exteriores permanecen en el grafo.",
                    "depresion_conectada": depression_report["metodo_delimitacion"],
                    "NoData": "D8 usa su máscara propia para construir el grafo; las máscaras de elevación, profundidad y acumulación se leen de cada raster al muestrear atributos.",
                },
                "salidas": {key: str(path) for key, path in self.outputs.items()},
                "tiempo_segundos": time.perf_counter() - started,
            }
            self.outputs["json"].write_text(json.dumps(_json_default(report), ensure_ascii=False, indent=2), encoding="utf-8")
            # Cierra memmaps antes de que TemporaryDirectory retire los temporales de Windows.
            for obj in (direction, basin_map, labels, queue, catchment, catch_queue):
                try:
                    obj.flush()
                    obj._mmap.close()
                except Exception:
                    pass
            return report

    @staticmethod
    def _write_csv(path, fields, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(_json_default(value), ensure_ascii=False)
                                 if isinstance(value, (dict, list, tuple)) else _json_default(value)
                                 for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dem", type=Path, default=DEFAULT_REFERENCE_DEM)
    parser.add_argument("--products-dir", type=Path, default=DEFAULT_PRODUCTS_DIR)
    parser.add_argument("--hydrography-gpkg", type=Path, default=DEFAULT_HYDRO_GPKG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subbasins", nargs="+", default=list(DEFAULT_SUBBASINS))
    parser.add_argument("--target-row", type=int, default=9570)
    parser.add_argument("--target-col", type=int, default=114)
    parser.add_argument("--auto-target", action="store_true",
                        help="Localiza la máxima acumulación desde el raster para otro dominio.")
    parser.add_argument("--depression-x", type=float, default=565612.5)
    parser.add_argument("--depression-y", type=float, default=2206747.5)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = ConnectivityConfig(
        reference_dem=args.reference_dem, products_dir=args.products_dir,
        hydrography_gpkg=args.hydrography_gpkg, output_dir=args.output_dir,
        subbasin_ids=tuple(args.subbasins),
        target_cell=None if args.auto_target else (args.target_row, args.target_col),
        depression_xy=(args.depression_x, args.depression_y),
        block_rows=args.block_rows, traversal_batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    report = HydrologicalConnectivity(config).run()
    print(json.dumps(report["salidas"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
