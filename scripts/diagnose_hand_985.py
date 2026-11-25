"""Diagnóstico de cobertura y conectividad de celdas HAND NoData (paso 98.5).

El módulo solo lee los productos 90–97. Las matrices globales de D8 y etiquetas
se mantienen en memmap temporal; los bloques limitan E/S, nunca separan la red.
No rellena HAND ni cambia umbrales o productos fuente.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from scripts.hydrological_validation import DEFAULT_REFERENCE_DEM, DEFAULT_SUBBASINS, D8_OFFSETS, PRODUCTS
except ImportError:
    from hydrological_validation import DEFAULT_REFERENCE_DEM, DEFAULT_SUBBASINS, D8_OFFSETS, PRODUCTS


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados"
DEFAULT_PRODUCTS_DIR = PROCESSED_DIR / "Etapa_2_Hidrologia"
DEFAULT_GPKG = REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez" / "red_hidrografica_villa_de_alvarez.gpkg"
DEFAULT_OUTPUT_DIR = DEFAULT_PRODUCTS_DIR / "diagnostico_98_5"
THRESHOLDS = (100, 250, 500, 750, 1000, 1500, 2000)
NODATA_CODE = np.uint8(255)
CONNECTIVITY = {1: "alcanza_drenaje", 2: "termina_D8_0", 3: "llega_a_NoData",
                4: "sale_del_dominio", 5: "ciclo", 6: "sin_conexion_identificable"}


class HandDiagnosticError(RuntimeError):
    """Insumos incompatibles o anomalía que impide continuar el diagnóstico."""


@dataclass
class HandDiagnosticConfig:
    reference_dem: Path = DEFAULT_REFERENCE_DEM
    products_dir: Path = DEFAULT_PRODUCTS_DIR
    hydrography_gpkg: Path = DEFAULT_GPKG
    output_dir: Path = DEFAULT_OUTPUT_DIR
    block_rows: int = 256
    basin_ids: tuple[str, ...] = DEFAULT_SUBBASINS
    overwrite: bool = False
    streams_layer: str = "corrientes_regionales"
    basins_layer: str = "subcuencas_regionales"
    basin_field: str = "_dcag_subbasin"
    hydro_category_field: str = "_dcag_hydro_category"

    def __post_init__(self):
        for key in ("reference_dem", "products_dir", "hydrography_gpkg", "output_dir"):
            setattr(self, key, Path(getattr(self, key)).expanduser().resolve())
        if self.block_rows < 1:
            raise ValueError("block_rows debe ser positivo")
        if len(set(self.basin_ids)) != len(self.basin_ids):
            raise ValueError("Hay identificadores de subcuenca duplicados")


def accumulation_distribution(values, stream_values=None, thresholds=THRESHOLDS):
    """Distribución de acumulación para el conjunto de celdas HAND ausentes."""
    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("La acumulación de las celdas faltantes debe ser finita y no negativa")
    streams = np.zeros(vals.shape, dtype=bool) if stream_values is None else np.asarray(stream_values) == 1
    if streams.shape != vals.shape:
        raise ValueError("Máscara de drenaje incompatible")
    bins = (0, *thresholds, math.inf)
    histogram = []
    n = vals.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (vals >= lo) & (vals < hi)
        histogram.append({"limite_inferior": lo, "limite_superior_exclusivo": None if math.isinf(hi) else hi,
                          "celdas": int(mask.sum()), "porcentaje_faltantes": 100 * float(mask.sum()) / n if n else 0.0,
                          "celdas_drenaje_1": int(np.count_nonzero(mask & streams))})
    coverage = [{"umbral_celdas": int(t), "celdas_sobre_umbral": int(np.count_nonzero(vals >= t)),
                 "porcentaje_faltantes_sobre_umbral": 100 * float(np.count_nonzero(vals >= t)) / n if n else 0.0,
                 "ya_marcadas_como_drenaje": int(np.count_nonzero((vals >= t) & streams)),
                 "potencialmente_cubiertas_sin_modificar_red": int(np.count_nonzero((vals >= t) & ~streams))}
                for t in thresholds]
    return {"intervalos": histogram, "umbrales": coverage}


def classify_d8_paths(d8, drainage, valid_mask):
    """Clasifica el destino final de cada celda en un grafo D8 completo.

    El algoritmo recorre la función de sucesor con memoización global. Sirve para
    matrices pequeñas y como referencia sintética; la ejecución regional escribe
    sus estados auxiliares en memmap y conserva el mismo grafo global.
    """
    direction = np.asarray(d8)
    streams = np.asarray(drainage, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if direction.shape != streams.shape or direction.shape != valid.shape or direction.ndim != 2:
        raise ValueError("D8, drenaje y máscara deben ser matrices 2D de igual forma")
    h, w = direction.shape
    bad = valid & ~np.isin(direction, (0, *D8_OFFSETS))
    if bad.any():
        raise ValueError("Código D8 no reconocido")
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[~valid] = CONNECTIVITY_CODE["llega_a_NoData"]
    labels[valid & streams] = CONNECTIVITY_CODE["alcanza_drenaje"]
    for start_r in range(h):
        for start_c in range(w):
            if not valid[start_r, start_c] or labels[start_r, start_c]:
                continue
            path, locations = [], {}
            r, c = start_r, start_c
            outcome = None
            while True:
                if not (0 <= r < h and 0 <= c < w):
                    outcome = CONNECTIVITY_CODE["sale_del_dominio"]; break
                if not valid[r, c]:
                    outcome = CONNECTIVITY_CODE["llega_a_NoData"]; break
                if labels[r, c]:
                    outcome = int(labels[r, c]); break
                idx = r * w + c
                if idx in locations:
                    outcome = CONNECTIVITY_CODE["ciclo"]
                    for rr, cc in path[locations[idx]:]:
                        labels[rr, cc] = outcome
                    break
                locations[idx] = len(path); path.append((r, c))
                code = int(direction[r, c])
                if code == 0:
                    outcome = CONNECTIVITY_CODE["termina_D8_0"]; break
                dr, dc = D8_OFFSETS[code]
                r, c = r + dr, c + dc
            for rr, cc in reversed(path):
                if labels[rr, cc] == 0:
                    labels[rr, cc] = outcome
    return labels


CONNECTIVITY_CODE = {name: code for code, name in CONNECTIVITY.items()}


def classify_d8_memmap_global(direction, drainage, valid, labels, queue, height, width, batch_size=250_000):
    """Clasifica un grafo D8 global con etiquetas/cola disk-backed.

    El BFS inverso propaga destinos conocidos a todos sus predecesores. Las
    celdas que quedan sin etiqueta forman ciclos o árboles que desembocan en
    ciclos; para el diagnóstico HAND se reportan juntas como ciclo/no salida.
    """
    total = height * width
    labels[:] = 0
    queue_write = 0
    # Semillas terminales en pasadas secuenciales; 255 es NoData temporal.
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        d = np.asarray(direction[start:stop]); v = np.asarray(valid[start:stop], dtype=bool)
        s = np.asarray(drainage[start:stop], dtype=bool)
        local = np.flatnonzero(v & s)
        if local.size:
            ids = local.astype(np.int64) + start
            labels[ids] = CONNECTIVITY_CODE["alcanza_drenaje"]
            queue[queue_write:queue_write+len(ids)] = ids.astype(np.uint32); queue_write += len(ids)
        d8_nodata = np.flatnonzero(v & (d == NODATA_CODE) & ~s)
        if d8_nodata.size:
            ids = d8_nodata.astype(np.int64) + start
            labels[ids] = CONNECTIVITY_CODE["llega_a_NoData"]
            queue[queue_write:queue_write+len(ids)] = ids.astype(np.uint32); queue_write += len(ids)
        zero = np.flatnonzero(v & (d == 0) & ~s)
        if zero.size:
            ids = zero.astype(np.int64) + start
            labels[ids] = CONNECTIVITY_CODE["termina_D8_0"]
            queue[queue_write:queue_write+len(ids)] = ids.astype(np.uint32); queue_write += len(ids)
        nonzero = np.flatnonzero(v & (d != 0) & ~s)
        if nonzero.size:
            ids = nonzero.astype(np.int64) + start
            rows, cols = ids // width, ids % width
            codes = d[nonzero]
            exits = np.zeros(ids.size, dtype=bool); to_nodata = np.zeros(ids.size, dtype=bool)
            for code, (dr, dc) in D8_OFFSETS.items():
                sel = codes == code
                if not sel.any(): continue
                rr, cc = rows[sel] + dr, cols[sel] + dc
                outside = (rr < 0) | (rr >= height) | (cc < 0) | (cc >= width)
                positions = np.flatnonzero(sel)
                exits[positions[outside]] = True
                inside = ~outside
                if inside.any():
                    dest = rr[inside] * width + cc[inside]
                    to_nodata[positions[inside]] = ~np.asarray(valid[dest], dtype=bool)
            for mask, category in ((exits, "sale_del_dominio"), (to_nodata, "llega_a_NoData")):
                ids_seed = ids[mask]
                if ids_seed.size:
                    labels[ids_seed] = CONNECTIVITY_CODE[category]
                    queue[queue_write:queue_write+len(ids_seed)] = ids_seed.astype(np.uint32); queue_write += len(ids_seed)
    # Flood fill topológico inverso; cada celda se etiqueta una sola vez.
    read = 0
    while read < queue_write:
        stop = min(queue_write, read + batch_size)
        targets = np.asarray(queue[read:stop], dtype=np.int64)
        tr, tc = targets // width, targets % width
        target_labels = np.asarray(labels[targets], dtype=np.uint8)
        for code, (dr, dc) in D8_OFFSETS.items():
            sr, sc = tr - dr, tc - dc
            inside = (sr >= 0) & (sr < height) & (sc >= 0) & (sc < width)
            if not inside.any(): continue
            source_ids = sr[inside] * width + sc[inside]
            eligible = (np.asarray(direction[source_ids]) == code) & (np.asarray(labels[source_ids]) == 0)
            if eligible.any():
                ids = source_ids[eligible]
                labels[ids] = target_labels[inside][eligible]
                queue[queue_write:queue_write+len(ids)] = ids.astype(np.uint32)
                queue_write += len(ids)
        read = stop
    # En un grafo funcional finito, residuales válidos implican ciclo o flujo a ciclo.
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        v = np.asarray(valid[start:stop], dtype=bool)
        lab = np.asarray(labels[start:stop])
        residual = v & (lab == 0)
        if residual.any():
            labels[start:stop][residual] = CONNECTIVITY_CODE["ciclo"]
        nodata = ~v
        if nodata.any():
            labels[start:stop][nodata] = CONNECTIVITY_CODE["llega_a_NoData"]
    labels.flush(); queue.flush()
    return int(queue_write)


def missing_distance_summary(distances):
    arr = np.asarray(distances, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"celdas": 0, "minimo_m": None, "maximo_m": None, "media_m": None,
                "mediana_m": None, "percentiles_m": {str(q): None for q in (5, 25, 75, 95, 99)}}
    return {"celdas": int(arr.size), "minimo_m": float(arr.min()), "maximo_m": float(arr.max()),
            "media_m": float(arr.mean()), "mediana_m": float(np.median(arr)),
            "percentiles_m": {str(q): float(np.percentile(arr, q)) for q in (5, 25, 75, 95, 99)}}


def _sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _product_paths(config):
    return {key: config.products_dir / filename for key, filename in PRODUCTS.items()}


def _read_generation_provenance(products):
    """Lee sidecars 95/97 para recuperar umbral y versión Whitebox usados."""
    result = {}
    for step, key in (("95", "drenaje_potencial"), ("97", "hand")):
        product = products[key]
        sidecar = product.with_name(product.name + ".provenance.json")
        if not sidecar.is_file():
            result[step] = {"sidecar": str(sidecar), "disponible": False}
            continue
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result[step] = {"sidecar": str(sidecar), "disponible": False, "error": str(exc)}
            continue
        params = record.get("parameters", {})
        result[step] = {"sidecar": str(sidecar), "disponible": True,
                        "whitebox_version": record.get("whitebox_version"),
                        "step": params.get("step"), "stream_threshold_cells": params.get("stream_threshold_cells"),
                        "inputs": record.get("inputs"), "output_sha256": record.get("output_sha256")}
    return result


def _verify_grids(config, paths):
    import rasterio
    required = {"referencia": config.reference_dem, **paths}
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Faltan GeoTIFF requeridos: " + ", ".join(missing))
    with rasterio.open(config.reference_dem) as ref:
        if ref.crs is None or not ref.crs.is_projected:
            raise HandDiagnosticError("El MDE de referencia debe tener CRS proyectado")
        sig = (ref.crs, ref.transform, ref.width, ref.height, ref.bounds)
        if (ref.crs.linear_units or "").lower() not in {"metre", "meter", "metres", "meters"}:
            raise HandDiagnosticError("El CRS debe estar expresado en metros")
    for name, path in required.items():
        with rasterio.open(path) as ds:
            if (ds.crs, ds.transform, ds.width, ds.height, ds.bounds) != sig or ds.count != 1:
                raise HandDiagnosticError(f"La cuadrícula de {name} no coincide exactamente con MDE")
    return sig


def _read_vectors(config, crs):
    import geopandas as gpd
    if not config.hydrography_gpkg.is_file():
        raise FileNotFoundError(config.hydrography_gpkg)
    lines = gpd.read_file(config.hydrography_gpkg, layer=config.streams_layer)
    basins = gpd.read_file(config.hydrography_gpkg, layer=config.basins_layer)
    for frame in (lines, basins):
        if frame.crs is None:
            raise HandDiagnosticError("Las capas vectoriales requieren CRS declarado")
    if config.basin_field not in basins or config.hydro_category_field not in lines:
        raise HandDiagnosticError("Faltan campos de categoría hidrográfica o subcuenca")
    lines = lines.to_crs(crs)
    basins = basins.to_crs(crs)
    natural = lines[lines[config.hydro_category_field].astype(str).isin(
        {"corriente_perenne", "corriente_intermitente"})].copy()
    selected = basins[basins[config.basin_field].astype(str).isin(config.basin_ids)].copy()
    if set(selected[config.basin_field].astype(str)) != set(config.basin_ids):
        raise HandDiagnosticError("No están presentes todas las subcuencas configuradas")
    return natural, selected


def _stream_distance(points_xy, geometries):
    """Distancia exacta punto-línea usando STRtree; vector geométrico oficial.

    Mantiene compatibilidad con Shapely 1.x/2.x. La geometría usada se reporta
    como red lineal INEGI; no es una distancia de conectividad D8.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        raise HandDiagnosticError("No hay geometrías lineales para calcular distancia")
    tree = STRtree(geoms)
    result = np.empty(len(points_xy), dtype=np.float64)
    for i, (x, y) in enumerate(points_xy):
        p = Point(float(x), float(y))
        nearest = tree.nearest(p)
        if isinstance(nearest, (int, np.integer)):
            geom = geoms[int(nearest)]
        else:
            geom = nearest
        result[i] = p.distance(geom)
    return result


def _potential_network_segments(stream_mask, d8, height, width, transform, batch_cells=1_000_000):
    """Construye segmentos centro-a-centro de celdas conectadas en el raster 95.

    Cada segmento representa una arista D8 cuyos extremos pertenecen a la red
    potencial rasterizada. Así la distancia geométrica corresponde al producto
    95 y no se sustituye por distancia a vértices ni por hidrografía distinta.
    """
    from shapely.geometry import LineString
    total = height * width
    lines = []
    for start in range(0, total, batch_cells):
        stop = min(total, start + batch_cells)
        indices = np.flatnonzero(np.asarray(stream_mask[start:stop], dtype=bool)).astype(np.int64) + start
        if not indices.size:
            continue
        codes = np.asarray(d8[indices], dtype=np.uint8)
        rows, cols = indices // width, indices % width
        for code, (dr, dc) in D8_OFFSETS.items():
            selected = codes == code
            if not selected.any():
                continue
            rr, cc = rows[selected], cols[selected]
            nr, nc = rr + dr, cc + dc
            inside = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            rr, cc, nr, nc = rr[inside], cc[inside], nr[inside], nc[inside]
            connected = np.asarray(stream_mask[nr * width + nc], dtype=bool)
            for r0, c0, r1, c1 in zip(rr[connected], cc[connected], nr[connected], nc[connected]):
                x0, y0 = transform * (int(c0) + 0.5, int(r0) + 0.5)
                x1, y1 = transform * (int(c1) + 0.5, int(r1) + 0.5)
                lines.append(LineString(((x0, y0), (x1, y1))))
    if not lines:
        raise HandDiagnosticError("La red potencial no contiene aristas D8 conectadas para medir distancia")
    return lines


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(config: HandDiagnosticConfig):
    """Ejecuta el diagnóstico únicamente cuando se invoca explícitamente."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import Window
    from rasterio.plot import plotting_extent
    from rasterio.features import rasterize
    from shapely.geometry import Point

    products = _product_paths(config)
    _verify_grids(config, products)
    output_files = [config.output_dir / name for name in (
        "diagnostico_hand_985.json", "celdas_hand_faltantes.csv", "estadisticas_por_subcuenca.csv",
        "distribucion_acumulacion.csv", "clasificacion_conectividad.csv", "cobertura_hand.png", "anomalias_hand.png")]
    if not config.overwrite and any(p.exists() for p in output_files):
        raise FileExistsError("Ya existen productos de salida; elija otro directorio o autorice overwrite")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    paths = {"dem": config.reference_dem, "hand": products["hand"], "accumulation": products["acumulacion_celdas"],
             "streams": products["drenaje_potencial"], "d8": products["direccion_d8"], "twi": products["twi"]}
    with rasterio.open(config.reference_dem) as ref:
        h, w, transform, crs = ref.height, ref.width, ref.transform, ref.crs
        n = h * w
        natural_lines, basins = _read_vectors(config, crs)
        basin_geoms = [(str(row[config.basin_field]), row.geometry) for _, row in basins.iterrows()]
        basin_raster = rasterize([(geom, i + 1) for i, (_, geom) in enumerate(basin_geoms)],
                                 out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
        # La capa de cuencas vectorizada supera el tamaño de la matriz auxiliar; liberable antes de la traza.
        scratch = Path(tempfile.mkdtemp(prefix="dcag_hand985_", dir=config.output_dir))
        names = {k: scratch / f"{k}.bin" for k in ("valid", "missing", "direction", "stream", "labels", "queue")}
        valid = np.memmap(names["valid"], mode="w+", dtype=np.uint8, shape=(n,)); valid[:] = 0
        missing = np.memmap(names["missing"], mode="w+", dtype=np.uint8, shape=(n,)); missing[:] = 0
        direction = np.memmap(names["direction"], mode="w+", dtype=np.uint8, shape=(n,)); direction[:] = NODATA_CODE
        stream = np.memmap(names["stream"], mode="w+", dtype=np.uint8, shape=(n,)); stream[:] = 0
        labels = np.memmap(names["labels"], mode="w+", dtype=np.uint8, shape=(n,)); labels[:] = 0
        queue = np.memmap(names["queue"], mode="w+", dtype=np.uint32, shape=(n,))
        missing_count = ref_valid_count = hand_valid_count = 0
        acc_missing, drain_missing, by_basin = [], [], {b: 0 for b in config.basin_ids}
        missing_rows = []
        hand_ds = rasterio.open(products["hand"]); acc_ds = rasterio.open(products["acumulacion_celdas"])
        stream_ds = rasterio.open(products["drenaje_potencial"]); d8_ds = rasterio.open(products["direccion_d8"])
        try:
            for row0 in range(0, h, config.block_rows):
                win = Window(0, row0, w, min(config.block_rows, h-row0)); sl = slice(row0*w, (row0+int(win.height))*w)
                rv = ref.read_masks(1, window=win) > 0; hm = hand_ds.read_masks(1, window=win) > 0
                sv = stream_ds.read_masks(1, window=win) > 0; dv = d8_ds.read_masks(1, window=win) > 0
                valid[sl] = rv.ravel(); missing_block = rv & ~hm; missing[sl] = missing_block.ravel()
                valid_count = int(rv.sum()); ref_valid_count += valid_count; hand_valid_count += int((rv & hm).sum())
                ds = d8_ds.read(1, window=win); dr = stream_ds.read(1, window=win)
                invalid_codes = dv & ~np.isin(ds, (0, *D8_OFFSETS))
                if invalid_codes.any(): raise HandDiagnosticError("D8 contiene códigos no reconocidos")
                direction[sl] = np.where(dv, ds, NODATA_CODE).astype(np.uint8).ravel()
                stream[sl] = ((sv & (dr == 1)) & rv).astype(np.uint8).ravel()
                if missing_block.any():
                    accumulation_valid = acc_ds.read_masks(1, window=win) > 0
                    if np.any(missing_block & ~accumulation_valid):
                        raise HandDiagnosticError("Acumulación NoData en celdas donde HAND está ausente; no puede cuantificarse")
                    acc_block = acc_ds.read(1, window=win)
                    accum = acc_block[missing_block].astype(np.float64)
                    drains = dr[missing_block]
                    acc_missing.extend(accum.tolist()); drain_missing.extend(drains.tolist())
                    missing_count += int(missing_block.sum())

                    # Mantener la ventana territorial en dos dimensiones.
                    basin_block = basin_raster[
                        row0:row0 + int(win.height), :
                    ]

                    # Extraer únicamente las subcuencas correspondientes
                    # a las celdas HAND ausentes.
                    basin_missing = basin_block[missing_block]

                    # Conteo de celdas faltantes por subcuenca.
                    for i, basin_id in enumerate(config.basin_ids, 1):
                        by_basin[basin_id] += int(
                            np.count_nonzero(basin_missing == i)
                        )

                    rows, cols = np.nonzero(missing_block)

                    for rr, cc in zip(rows.tolist(), cols.tolist()):
                        r, c = row0 + rr, cc

                        x, y = rasterio.transform.xy(
                            transform, r, c, offset="center"
                        )

                        missing_rows.append((
                            r,
                            c,
                            x,
                            y,
                            float(acc_block[rr, cc]),
                            int(dr[rr, cc]),
                            int(basin_block[rr, cc])
                        ))

        finally:
            hand_ds.close(); acc_ds.close(); stream_ds.close(); d8_ds.close()
        valid.flush(); missing.flush(); direction.flush(); stream.flush()
        # Etiquetado del grafo completo: las fronteras de bloque no son salidas.
        valid_count_d8 = int(np.count_nonzero(valid))
        classify_d8_memmap_global(direction, stream, valid, labels, queue, h, w)
        # Distancia geométrica por lotes a líneas hidrográficas perennes/intermitentes INEGI.
        # Se separa explícitamente de la ruta D8 y no altera la red ráster potencial.
        potential_lines = _potential_network_segments(stream, direction, h, w, transform)
        distances = []
        for start in range(0, len(missing_rows), max(1, config.block_rows*w)):
            batch = missing_rows[start:start+max(1, config.block_rows*w)]
            d = _stream_distance([(x,y) for _,_,x,y,_,_,_ in batch], potential_lines)
            distances.extend(d.tolist())
        label_counts = {name: 0 for name in CONNECTIVITY.values()}
        basin_rows = []
        for i, basin_id in enumerate(config.basin_ids, 1):
            basin_missing = [row for row in missing_rows if row[6] == i]
            basin_rows.append({"subcuenca": basin_id, "celdas_mde_validas": int(np.count_nonzero(basin_raster == i)),
                               "celdas_HAND_faltantes": len(basin_missing),
                               "porcentaje_faltantes_sobre_celdas_mde": 100*len(basin_missing)/max(1,int(np.count_nonzero(basin_raster == i)))})
        for _,_,_,_,_,_,_ in missing_rows:
            pass
        for row in missing_rows:
            code = int(labels[row[0]*w+row[1]])
            name = next((k for k,v in CONNECTIVITY_CODE.items() if v == code), "sin_conexion_identificable")
            label_counts[name] += 1
        distribution = accumulation_distribution(acc_missing, drain_missing)
        dist_summary = missing_distance_summary(distances)
        _write_csv(output_files[2], ("subcuenca","celdas_mde_validas","celdas_HAND_faltantes","porcentaje_faltantes_sobre_celdas_mde"), basin_rows)
        _write_csv(output_files[3], ("tipo","limite_inferior","limite_superior_exclusivo","celdas","porcentaje_faltantes","celdas_drenaje_1","umbral_celdas","celdas_sobre_umbral","porcentaje_faltantes_sobre_umbral","potencialmente_cubiertas_sin_modificar_red"),
                   [{"tipo":"intervalo",**{k:v for k,v in row.items() if k!="celdas_drenaje_1"},"celdas_drenaje_1":row["celdas_drenaje_1"]} for row in distribution["intervalos"]] +
                   [{"tipo":"umbral", "umbral_celdas":r["umbral_celdas"],"celdas_sobre_umbral":r["celdas_sobre_umbral"],"porcentaje_faltantes_sobre_umbral":r["porcentaje_faltantes_sobre_umbral"],"potencialmente_cubiertas_sin_modificar_red":r["potencialmente_cubiertas_sin_modificar_red"]} for r in distribution["umbrales"]])
        _write_csv(output_files[4], ("clasificacion","celdas","porcentaje_faltantes"),
                   [{"clasificacion":k,"celdas":v,"porcentaje_faltantes":100*v/max(1,missing_count)} for k,v in label_counts.items()])
        # La lista de anomalías se emite en streaming desde las filas de faltantes.
        with output_files[1].open("w",newline="",encoding="utf-8-sig") as fh:
            writer=csv.writer(fh); writer.writerow(("fila","columna","x_centro","y_centro","acumulacion_celdas","drenaje_potencial","subcuenca","conectividad_D8","distancia_red_hidrografica_m"))
            for row,dist in zip(missing_rows,distances):
                r,c,x,y,a,s,b=row; code=int(labels[r*w+c]); cls=next((k for k,v in CONNECTIVITY_CODE.items() if v==code),"sin_conexion_identificable")
                basin_name=config.basin_ids[b-1] if b else "exterior"
                writer.writerow((r,c,x,y,a,s,basin_name,cls,dist))
        # Vistas reducidas solo para cartografía; no intervienen en análisis.
        max_dim=1800; scale=max(1,math.ceil(max(h,w)/max_dim)); cov=np.zeros((math.ceil(h/scale),math.ceil(w/scale)),dtype=np.uint8)
        for rr in range(cov.shape[0]):
            r0=rr*scale;r1=min(h,r0+scale)
            for cc in range(cov.shape[1]):
                c0=cc*scale;c1=min(w,c0+scale)
                v=np.asarray(valid.reshape(h,w)[r0:r1,c0:c1]); m=np.asarray(missing.reshape(h,w)[r0:r1,c0:c1])
                cov[rr,cc]=2 if m.any() else (1 if v.any() else 0)
        extent = (ref.bounds.left, ref.bounds.right, ref.bounds.bottom, ref.bounds.top)
        for out, title in ((output_files[5],"Cobertura del producto HAND"),(output_files[6],"Celdas HAND NoData dentro del MDE válido")):
            fig,ax=plt.subplots(figsize=(11,8)); im=ax.imshow(cov if out==output_files[5] else (cov==2),origin="upper",extent=extent,cmap="viridis" if out==output_files[5] else "Reds",interpolation="nearest",alpha=.9)
            basins.boundary.plot(ax=ax,color="black",linewidth=.65)
            if len(natural_lines): natural_lines.plot(ax=ax,color="#2878b5",linewidth=.25,alpha=.6)
            ax.set_title(title); ax.set_xlabel(f"Este ({crs.to_string()})"); ax.set_ylabel("Norte (m)"); fig.colorbar(im,ax=ax,label="0 NoData MDE / 1 HAND válido / 2 HAND faltante" if out==output_files[5] else "HAND faltante")
            fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)
    elapsed=time.perf_counter()-started
    inputs={k:{"ruta":str(p),"sha256":_sha256(p)} for k,p in paths.items()}
    inputs["geopackage_hidrografico"]={"ruta":str(config.hydrography_gpkg),"sha256":_sha256(config.hydrography_gpkg)}
    try:
        import rasterio, geopandas, shapely
        versions={"python":platform.python_version(),"numpy":np.__version__,"rasterio":rasterio.__version__,"geopandas":geopandas.__version__,"shapely":shapely.__version__}
    except Exception: versions={"python":platform.python_version(),"numpy":np.__version__}
    report={"paso":"98.5","fecha_utc":datetime.now(timezone.utc).isoformat(),"insumos":inputs,
            "parametros":asdict(config),"versiones":versions,"celdas_referencia_validas":ref_valid_count,
            "celdas_HAND_validas_en_referencia":hand_valid_count,"celdas_HAND_faltantes":missing_count,
            "distribucion_acumulacion":distribution,"conectividad_D8_faltantes":label_counts,
            "distancia_euclidiana_red_potencial_m":dist_summary,
            "nota_distancia":"Distancia euclidiana exacta a segmentos centro-a-centro D8 de celdas contiguas del raster 95 de drenaje potencial; la capa lineal INEGI se dibuja como referencia cartográfica, no se usa como sustituto de esa red ni de conectividad D8.",
            "subcuencas":basin_rows,"metodo_HAND":"WhiteboxTools elevation_above_stream; hydrological_processing.py paso 97 usa MDE acondicionado paso 91 y drenaje potencial paso 95, sin parámetros explícitos adicionales; HAND NoData no se rellena.",
            "procedencia_generacion_pasos_95_97":_read_generation_provenance(products),
            "metodo_conectividad":"Grafo D8 global, convención nativa WhiteboxTools esri_pntr=False; NoData D8, salidas del raster, D8=0 y ciclos se distinguen.",
            "estrategia_memoria":{"arreglos_globales_en_memmap":True,"arreglos_compactos":{"mascara_validacion":"uint8","mascara_HAND_faltante":"uint8","D8":"uint8","drenaje":"uint8","etiquetas":"uint8","cola_BFS":"uint32"},"estimacion_temporal_disco_bytes":int(n*(1+1+1+1+1+4)),"estimacion_RAM_fija_rasterizacion_subcuencas_bytes":int(n),"RAM_adicional":"Dependiente de block_rows × ancho, número de celdas HAND ausentes y cantidad de aristas D8 de drenaje potencial almacenadas en STRtree; no se materializan simultáneamente todos los GeoTIFF. Conviene revisar espacio temporal (aprox. 850 MB para 94,441,924 celdas) y RAM disponible antes de ejecutar.","bloques_solo_controlan_E_S_y_no_particionan_grafo":True},
            "duracion_segundos":elapsed,"productos_generados":[str(p) for p in output_files]}
    output_files[0].write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    # Eliminar temporales solo tras terminar análisis; los archivos de fuente permanecen intactos.
    for arr in (valid,missing,direction,stream,labels,queue):
        arr.flush(); arr._mmap.close()
    for p in scratch.iterdir(): p.unlink()
    scratch.rmdir()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem",type=Path,default=DEFAULT_REFERENCE_DEM)
    parser.add_argument("--products-dir",type=Path,default=DEFAULT_PRODUCTS_DIR)
    parser.add_argument("--gpkg",type=Path,default=DEFAULT_GPKG)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-rows",type=int,default=256)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    result=run_diagnostic(HandDiagnosticConfig(args.dem,args.products_dir,args.gpkg,args.output_dir,args.block_rows,overwrite=args.overwrite))
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__ == "__main__":
    main()
