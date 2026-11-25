"""Pipeline reproducible para pasos hidrológicos raster 90–97 con WhiteboxTools.

Villa de Álvarez es el caso piloto y permanece como etiqueta/ruta predeterminada.
El MDE de soporte regional no sustituye la cuadrícula municipal de integración.
Este módulo no genera clases de susceptibilidad ni modifica insumos.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
try:
    from whitebox import WhiteboxTools
except ImportError:  # Permite ejecutar pruebas puras sin el runtime GIS instalado.
    WhiteboxTools = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DEM = (
    REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados"
    / "CEM40_Colima_Jalisco_Dominio_Hidrologico_Buffer2000m_UTM13N_15m_WBT.tif"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados"
    / "Etapa_2_Hidrologia"
)
DEFAULT_STREAM_THRESHOLD = 1000
DEFAULT_PROCESSING_THREADS = 2
DEFAULT_BLOCK_ROWS = 256
DEFAULT_SLOPE_FLOOR_DEGREES = 0.1
SUPPORTED_D8_CODES = (0, 1, 2, 4, 8, 16, 32, 64, 128)

PRODUCT_FILES = {
    "90": ("90_MDE_Relleno_Sin_Flats.tif", "90_Profundidad_Depresiones.tif"),
    "91": ("91_MDE_Acondicionado.tif",),
    "92": ("92_Pendiente_Grados.tif",),
    "93": ("93_Direccion_Flujo_D8.tif",),
    "94": ("94_Acumulacion_Flujo_Celdas.tif",),
    "95": ("95_Red_Drenaje_Potencial.tif",),
    "96": ("96_Indice_TWI.tif",),
    "97": ("97_Altura_HAND.tif",),
}
ALL_STEPS = tuple(PRODUCT_FILES)


class ProcessingError(RuntimeError):
    """Error controlado del procesamiento hidrológico."""


@dataclass
class RasterStats:
    count_valid: int = 0
    count_invalid: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    mean: float = 0.0
    m2: float = 0.0
    value_counts: dict[str, int] = field(default_factory=dict)

    def add(self, values, invalid_count=0, categorical=False):
        data = np.asarray(values)
        finite = data[np.isfinite(data)].astype(np.float64, copy=False).ravel()
        self.count_invalid += int(invalid_count + data.size - finite.size)
        if not finite.size:
            return
        n = int(finite.size)
        block_mean = float(finite.mean(dtype=np.float64))
        block_m2 = float(np.square(finite - block_mean).sum(dtype=np.float64))
        if not self.count_valid:
            self.mean, self.m2 = block_mean, block_m2
        else:
            total = self.count_valid + n
            delta = block_mean - self.mean
            self.m2 += block_m2 + delta * delta * self.count_valid * n / total
            self.mean += delta * n / total
        self.count_valid += n
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        if categorical:
            values, counts = np.unique(finite, return_counts=True)
            for value, count in zip(values, counts):
                key = str(int(value)) if float(value).is_integer() else str(float(value))
                self.value_counts[key] = self.value_counts.get(key, 0) + int(count)

    def to_dict(self):
        if self.count_valid == 0:
            return {"celdas_validas": 0, "celdas_invalidas": self.count_invalid}
        return {
            "celdas_validas": self.count_valid,
            "celdas_invalidas": self.count_invalid,
            "minimo": self.minimum,
            "maximo": self.maximum,
            "media": self.mean,
            "desviacion_estandar_poblacional": math.sqrt(max(self.m2, 0.0) / self.count_valid),
            **({"conteos_valor": self.value_counts} if self.value_counts else {}),
        }


@dataclass
class RasterMeta:
    crs: str | None
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    nodata: float | int | None
    resolution: tuple[float, float]
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]
    compression: str | None
    predictor: str | None

    @classmethod
    def from_dataset(cls, ds):
        image_tags = ds.tags(ns="IMAGE_STRUCTURE")
        return cls(
            crs=ds.crs.to_string() if ds.crs else None,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtypes=tuple(ds.dtypes),
            nodata=ds.nodata,
            resolution=tuple(float(v) for v in ds.res),
            transform=tuple(float(v) for v in ds.transform)[:6],
            bounds=(float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)),
            compression=ds.profile.get("compress"),
            predictor=image_tags.get("PREDICTOR"),
        )


@dataclass
class HydrologyConfig:
    input_dem: Path = DEFAULT_INPUT_DEM
    output_dir: Path = DEFAULT_OUTPUT_DIR
    stream_threshold_cells: int = DEFAULT_STREAM_THRESHOLD
    processing_threads: int = DEFAULT_PROCESSING_THREADS
    slope_floor_degrees: float = DEFAULT_SLOPE_FLOOR_DEGREES
    block_rows: int = DEFAULT_BLOCK_ROWS
    overwrite: bool = False
    resume: bool = False
    expected_whitebox_version: str = "2.4.0"
    allow_version_mismatch: bool = True
    memory_factor: float = 3.0
    disk_factor: float = 2.0

    def __post_init__(self):
        self.input_dem = Path(self.input_dem).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.stream_threshold_cells <= 0:
            raise ValueError("stream_threshold_cells debe ser positivo.")
        if self.processing_threads <= 0:
            raise ValueError("processing_threads debe ser positivo.")
        if not 0 < self.slope_floor_degrees <= 90:
            raise ValueError("slope_floor_degrees debe estar en (0, 90].")
        if self.block_rows <= 0:
            raise ValueError("block_rows debe ser positivo.")


def get_available_memory_bytes():
    """RAM disponible, usando API estándar de Windows o sysconf en POSIX."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("No se pudo consultar la memoria RAM disponible de Windows.")
        return int(status.ullAvailPhys)
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * pages)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Huella de archivo por bloques para verificar procedencia e integridad."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(message):
    print(message, flush=True)


def _pixel_area(transform):
    return abs(float(transform.a * transform.e - transform.b * transform.d))


def _stats_raster(path, block_rows=DEFAULT_BLOCK_ROWS, categorical=False):
    stats = RasterStats()
    with rasterio.open(path) as ds:
        for start in range(0, ds.height, block_rows):
            window = Window(0, start, ds.width, min(block_rows, ds.height - start))
            mask = ds.read_masks(1, window=window) > 0
            data = ds.read(1, window=window)
            stats.add(data[mask], invalid_count=int(mask.size - mask.sum()), categorical=categorical)
    return stats.to_dict()


def _grid_compare(reference_path, product_path, block_rows):
    with rasterio.open(reference_path) as ref, rasterio.open(product_path) as out:
        checks = {
            "crs": ref.crs == out.crs,
            "dimensiones": (ref.width, ref.height) == (out.width, out.height),
            "resolucion": ref.res == out.res,
            "transformacion": ref.transform == out.transform,
            "extension": ref.bounds == out.bounds,
            "una_banda": out.count == 1,
        }
        base_valid = output_valid = mask_differences = 0
        outside_reference = 0
        for start in range(0, ref.height, block_rows):
            window = Window(0, start, ref.width, min(block_rows, ref.height - start))
            ref_mask = ref.read_masks(1, window=window) > 0
            out_mask = out.read_masks(1, window=window) > 0
            base_valid += int(ref_mask.sum())
            output_valid += int(out_mask.sum())
            mask_differences += int(np.count_nonzero(ref_mask != out_mask))
            outside_reference += int(np.count_nonzero(out_mask & ~ref_mask))
        checks["sin_validos_fuera_del_MDE"] = outside_reference == 0
        return {
            "controles": checks,
            "controles_aprobados": sum(bool(v) for v in checks.values()),
            "controles_totales": len(checks),
            "celdas_validas_referencia": base_valid,
            "celdas_validas_producto": output_valid,
            "celdas_con_diferencia_mascara": mask_differences,
            "celdas_producto_fuera_de_mascara_referencia": outside_reference,
        }


def _validate_raster(path, reference_path, block_rows, mask_policy="subset", require_uncompressed=True):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ProcessingError(f"Producto raster ausente o vacío: {path}")
    with rasterio.open(path) as ds:
        meta = RasterMeta.from_dataset(ds)
        if ds.count != 1:
            raise ProcessingError(f"Se esperaba una banda en {path}.")
        if require_uncompressed and ds.profile.get("compress") not in (None, "none"):
            raise ProcessingError(f"WhiteboxTools produjo compresión inesperada: {ds.profile.get('compress')}")
    comparison = _grid_compare(reference_path, path, block_rows)
    failed = [name for name, passed in comparison["controles"].items() if not passed]
    if failed:
        raise ProcessingError(f"Metadatos/cuadrícula no coinciden para {path}: {failed}")
    if mask_policy == "exact" and comparison["celdas_con_diferencia_mascara"]:
        raise ProcessingError(f"La máscara de {path} debía coincidir exactamente con el MDE.")
    if mask_policy == "subset" and comparison["celdas_producto_fuera_de_mascara_referencia"]:
        raise ProcessingError(f"La máscara de {path} extiende fuera de la cobertura del MDE.")
    return {"metadatos": asdict(meta), "comparacion_mde": comparison}


def profundidad_depresiones(filled_path, original_path, output_path, block_rows=DEFAULT_BLOCK_ROWS):
    """Calcula relleno-original por ventanas; celdas fuera de máscara siguen NoData."""
    with rasterio.open(original_path) as original, rasterio.open(filled_path) as filled:
        if (original.crs, original.transform, original.width, original.height) != (
            filled.crs, filled.transform, filled.width, filled.height
        ):
            raise ProcessingError("El MDE rellenado no coincide con el MDE original.")
        profile = original.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", compress=None, predictor=None)
        nodata = original.nodata if original.nodata is not None else -9999.0
        profile.update(nodata=nodata)
        with rasterio.open(output_path, "w", **profile) as out:
            for start in range(0, original.height, block_rows):
                window = Window(0, start, original.width, min(block_rows, original.height - start))
                om = original.read_masks(1, window=window) > 0
                fm = filled.read_masks(1, window=window) > 0
                valid = om & fm
                z0 = original.read(1, window=window).astype(np.float32)
                zf = filled.read(1, window=window).astype(np.float32)
                depth = np.full(z0.shape, nodata, dtype=np.float32)
                difference = zf[valid] - z0[valid]
                # Small negative floating noise is clipped to zero; real decreases fail.
                if difference.size and float(np.min(difference)) < -1e-4:
                    raise ProcessingError("FillDepressions redujo elevación en la salida sin flats.")
                depth[valid] = np.maximum(difference, 0.0)
                out.write(depth, 1, window=window)


def metricas_profundidad_depresiones(path, cell_area_m2, block_rows=DEFAULT_BLOCK_ROWS):
    """Resume depresión positiva y volumen geométrico estimado por bloques."""
    modified = 0
    depth_sum = 0.0
    maximum = 0.0
    with rasterio.open(path) as ds:
        for start in range(0, ds.height, block_rows):
            window = Window(0, start, ds.width, min(block_rows, ds.height - start))
            mask = ds.read_masks(1, window=window) > 0
            values = ds.read(1, window=window)[mask].astype(np.float64)
            positive = values[values > 0]
            modified += int(positive.size)
            if positive.size:
                depth_sum += float(positive.sum(dtype=np.float64))
                maximum = max(maximum, float(positive.max()))
    return {
        "celdas_modificadas": modified,
        "superficie_celdas_modificadas_m2": modified * cell_area_m2,
        "superficie_celdas_modificadas_km2": modified * cell_area_m2 / 1_000_000,
        "profundidad_maxima_m": maximum,
        "volumen_geometrico_estimado_m3": depth_sum * cell_area_m2,
        "interpretacion_volumen": "Estimación geométrica de relleno; no equivale a capacidad real de almacenamiento de agua.",
    }


def calculate_twi(accum_path, slope_path, output_path, slope_floor_degrees=DEFAULT_SLOPE_FLOOR_DEGREES,
                  block_rows=DEFAULT_BLOCK_ROWS):
    """TWI D8 por ventanas: ln((n_celdas * ancho_celda) / tan(beta))."""
    with rasterio.open(accum_path) as accum, rasterio.open(slope_path) as slope:
        if (accum.crs, accum.transform, accum.width, accum.height) != (
            slope.crs, slope.transform, slope.width, slope.height
        ):
            raise ProcessingError("Acumulación y pendiente deben estar en la misma cuadrícula.")
        x_size, y_size = abs(accum.transform.a), abs(accum.transform.e)
        if not math.isclose(x_size, y_size, rel_tol=0, abs_tol=1e-9):
            raise ProcessingError("La aproximación D8 configurada requiere píxeles cuadrados.")
        profile = accum.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", compress=None, predictor=None, nodata=-9999.0)
        with rasterio.open(output_path, "w", **profile) as out:
            for start in range(0, accum.height, block_rows):
                window = Window(0, start, accum.width, min(block_rows, accum.height - start))
                am = accum.read_masks(1, window=window) > 0
                sm = slope.read_masks(1, window=window) > 0
                av = accum.read(1, window=window).astype(np.float64)
                sv = slope.read(1, window=window).astype(np.float64)
                valid = am & sm & np.isfinite(av) & np.isfinite(sv) & (av > 0)
                result = np.full(av.shape, -9999.0, dtype=np.float32)
                beta = np.deg2rad(sv[valid])
                tangent = np.tan(beta)
                tangent = np.maximum(tangent, math.tan(math.radians(slope_floor_degrees)))
                specific_area = av[valid] * x_size
                twi = np.log(specific_area / tangent)
                if not np.isfinite(twi).all():
                    raise ProcessingError("TWI generó valores no finitos en celdas válidas.")
                result[valid] = twi.astype(np.float32)
                out.write(result, 1, window=window)


def _make_runner(working_dir, processing_threads):
    if WhiteboxTools is None:
        raise ImportError("Se requiere el paquete WhiteboxTools (whitebox) disponible en el entorno local.")
    wbt = WhiteboxTools()
    if not Path(wbt.exe_path, wbt.exe_name).is_file():
        raise FileNotFoundError(f"No existe el ejecutable WhiteboxTools: {Path(wbt.exe_path, wbt.exe_name)}")
    wbt.set_working_dir(str(working_dir))
    statuses = {
        "set_max_procs": wbt.set_max_procs(processing_threads),
        "set_compress_rasters_false": wbt.set_compress_rasters(False),
    }
    if any(status != 0 for status in statuses.values()):
        raise ProcessingError(f"No se pudo configurar WhiteboxTools: {statuses}")
    return wbt, statuses


def _tool_version(wbt):
    try:
        raw = str(wbt.version()).strip()
        match = re.search(r"(?:WhiteboxTools\s+)?v?(\d+\.\d+\.\d+)", raw, flags=re.IGNORECASE)
        if not match:
            raise ProcessingError(f"No se pudo identificar la versión en: {raw!r}")
        return match.group(1)
    except Exception as exc:
        raise ProcessingError(f"No se pudo consultar la versión de WhiteboxTools: {exc}") from exc


def _verify_fill_flats_parameter(wbt):
    """Comprueba parámetro en el wrapper y en el help del ejecutable instalado."""
    import inspect

    if "fix_flats" not in inspect.signature(wbt.fill_depressions).parameters:
        raise ProcessingError("El wrapper no expone fix_flats.")
    help_text = wbt.tool_help("FillDepressions")
    if "--fix_flats" not in str(help_text):
        raise ProcessingError("La versión de WhiteboxTools no documenta --fix_flats.")
    return {
        "wrapper_expone_fix_flats": True,
        "ejecutable_documenta_flag": True,
        "sin_flats_comando": "fix_flats=False; el wrapper omite --fix_flats, por lo que queda desactivado.",
        "con_flats_comando": "fix_flats=True; el wrapper agrega --fix_flats.",
    }


class HydrologicalProcessor:
    """Orquesta productos raster, validaciones, reanudación y reportes."""

    def __init__(self, config: HydrologyConfig, whitebox=None):
        self.config = config
        self.whitebox = whitebox
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.records = []
        self.input_fingerprint = None
        self.reference_meta = None
        self.cell_area = None
        self.tool_version = None
        self._file_fingerprints = {}

    def preflight(self):
        if not self.config.input_dem.is_file():
            raise FileNotFoundError(f"No existe el MDE de entrada WBT: {self.config.input_dem}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(self.config.input_dem) as src:
            if src.crs is None or not src.crs.is_projected:
                raise ProcessingError("El MDE regional debe declarar un CRS proyectado.")
            if src.count != 1 or src.dtypes[0] != "float32":
                raise ProcessingError("El MDE WhiteboxTools de entrada debe ser una banda Float32.")
            if src.nodata is None:
                raise ProcessingError("El MDE requiere NoData explícito.")
            self.reference_meta = RasterMeta.from_dataset(src)
            self.cell_area = _pixel_area(src.transform)
            if not math.isclose(self.cell_area, 225.0, rel_tol=0, abs_tol=1e-5):
                _emit(f"AVISO: área de celda {self.cell_area:g} m²; no coincide con 15×15 m.")
            image_tags = src.tags(ns="IMAGE_STRUCTURE")
            predictor = image_tags.get("PREDICTOR")
            if str(predictor) == "3":
                raise ProcessingError("El MDE inicial usa PREDICTOR=3; utilice la copia _WBT.tif validada.")
        estimate = (self.reference_meta.width * self.reference_meta.height * 4
                    * self.config.memory_factor * self.config.processing_threads)
        available = get_available_memory_bytes()
        disk = shutil.disk_usage(self.config.output_dir)
        per_product_bytes = self.reference_meta.width * self.reference_meta.height * 4
        disk_required = int(per_product_bytes * self.config.disk_factor)
        if available < estimate:
            raise ProcessingError(
                f"RAM disponible insuficiente: {available} bytes; estimación preliminar {estimate} bytes."
            )
        if disk.free < disk_required:
            raise ProcessingError(
                f"Espacio insuficiente: {disk.free} bytes libres; al menos {disk_required} bytes requeridos."
            )
        self.input_fingerprint = self.fingerprint(self.config.input_dem)
        return {
            "mde": asdict(self.reference_meta),
            "huella_sha256": self.input_fingerprint["sha256"],
            "ram_disponible_bytes": available,
            "ram_estimada_por_herramienta_bytes": estimate,
            "espacio_libre_bytes": disk.free,
            "espacio_estimado_temporal_minimo_bytes": disk_required,
        }

    def fingerprint(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self._file_fingerprints:
            stat = path.stat()
            self._file_fingerprints[key] = {
                "ruta": key, "tamano_bytes": stat.st_size,
                "modificado_ns": stat.st_mtime_ns, "sha256": sha256_file(path),
            }
        return self._file_fingerprints[key]

    def product_path(self, step):
        return self.config.output_dir / PRODUCT_FILES[step][0]

    def sidecar_path(self, product):
        return product.with_name(product.name + ".provenance.json")

    def _dependencies(self, step):
        files = {
            "90": [self.config.input_dem],
            "91": [self.config.input_dem],
            "92": [self.config.input_dem],
            "93": [self.product_path("91")],
            "94": [self.product_path("93")],
            "95": [self.product_path("94")],
            "96": [self.product_path("94"), self.product_path("92")],
            "97": [self.product_path("91"), self.product_path("95")],
        }[step]
        for path in files:
            if not path.is_file():
                raise ProcessingError(f"Falta dependencia de {step}: {path}")
            if path != self.config.input_dem:
                sidecar = self.sidecar_path(path)
                if not sidecar.is_file():
                    raise ProcessingError(f"Dependencia sin procedencia verificable: {path}")
                record = json.loads(sidecar.read_text(encoding="utf-8"))
                if record.get("output_sha256") != sha256_file(path):
                    raise ProcessingError(f"Dependencia alterada o incompleta: {path}")
                if record.get("parameters", {}).get("input_dem_sha256") != self.input_fingerprint["sha256"]:
                    raise ProcessingError(f"Dependencia fue procesada desde otro MDE: {path}")
                if record.get("whitebox_version") != self.tool_version:
                    raise ProcessingError(f"Dependencia usa otra versión de WhiteboxTools: {path}")
                for ancestor in record.get("inputs", []):
                    ancestor_path = Path(ancestor["ruta"])
                    if not ancestor_path.is_file() or sha256_file(ancestor_path) != ancestor.get("sha256"):
                        raise ProcessingError(f"Procedencia aguas arriba alterada o ausente: {ancestor_path}")
                _validate_raster(path, self.config.input_dem, self.config.block_rows, mask_policy="subset")
        return files

    def _params(self, step):
        return {
            "step": step,
            "input_dem_sha256": self.input_fingerprint["sha256"],
            "whitebox_version": self.tool_version,
            "threads": self.config.processing_threads,
            "compression": False,
            "stream_threshold_cells": self.config.stream_threshold_cells if step in ("95", "97") else None,
            "slope_floor_degrees": self.config.slope_floor_degrees if step == "96" else None,
        }

    def _reuse_if_valid(self, step, destination, dependencies):
        if not destination.exists():
            return False
        if not self.config.resume:
            if not self.config.overwrite:
                raise FileExistsError(f"El producto ya existe: {destination}; use resume o overwrite explícito.")
            return False
        sidecar = self.sidecar_path(destination)
        if not sidecar.is_file():
            raise ProcessingError(f"Producto existente sin metadatos de procedencia; no se omite: {destination}")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        if provenance.get("parameters") != self._params(step):
            raise ProcessingError(f"Producto existente tiene parámetros incompatibles: {destination}")
        if provenance.get("inputs") != [self.fingerprint(path) for path in dependencies]:
            raise ProcessingError(f"Producto existente tiene insumos diferentes: {destination}")
        if provenance.get("output_sha256") != sha256_file(destination):
            raise ProcessingError(f"Producto existente está incompleto o fue modificado: {destination}")
        _validate_raster(destination, self.config.input_dem, self.config.block_rows, mask_policy="subset")
        self.records.append({"paso": step, "estado": "reutilizado_validado", "archivo": str(destination)})
        return True

    def _reuse_step_if_valid(self, step, destinations, dependencies):
        exists = [path.exists() for path in destinations]
        if not any(exists):
            return False
        if not all(exists):
            if self.config.resume:
                raise ProcessingError(f"Paso {step} tiene salidas parciales: {destinations}")
            if not self.config.overwrite:
                raise FileExistsError(f"Paso {step} tiene productos existentes: {destinations}")
            return False
        dependency_sets = [dependencies] * len(destinations)
        if step == "90" and len(destinations) == 2:
            dependency_sets[1] = [*dependencies, destinations[0]]
        for destination, input_paths in zip(destinations, dependency_sets):
            if not self._reuse_if_valid(step, destination, input_paths):
                return False
        return True

    def _check_resources(self):
        meta = self.reference_meta
        ram_est = meta.width * meta.height * 4 * self.config.memory_factor * self.config.processing_threads
        available = get_available_memory_bytes()
        disk = shutil.disk_usage(self.config.output_dir)
        temp_est = int(meta.width * meta.height * 4 * self.config.disk_factor)
        if available < ram_est or disk.free < temp_est:
            raise ProcessingError(
                f"Recursos insuficientes antes del paso: RAM {available}/{ram_est}; disco {disk.free}/{temp_est} bytes."
            )

    def _run_wbt_step(self, step, temp_outputs):
        inp = lambda key: str(self.product_path(key))
        if step == "90":
            status = self.whitebox.fill_depressions(
                str(self.config.input_dem), str(temp_outputs[0]), fix_flats=False, callback=_emit
            )
        elif step == "91":
            status = self.whitebox.fill_depressions(
                str(self.config.input_dem), str(temp_outputs[0]), fix_flats=True, callback=_emit
            )
        elif step == "92":
            status = self.whitebox.slope(str(self.config.input_dem), str(temp_outputs[0]), units="degrees", callback=_emit)
        elif step == "93":
            status = self.whitebox.d8_pointer(inp("91"), str(temp_outputs[0]), esri_pntr=False, callback=_emit)
        elif step == "94":
            status = self.whitebox.d8_flow_accumulation(
                inp("93"), str(temp_outputs[0]), out_type="cells", log=False,
                clip=False, pntr=True, esri_pntr=False, callback=_emit
            )
        elif step == "95":
            status = self.whitebox.extract_streams(
                inp("94"), str(temp_outputs[0]), threshold=self.config.stream_threshold_cells,
                zero_background=True, callback=_emit
            )
        elif step == "97":
            status = self.whitebox.elevation_above_stream(
                inp("91"), inp("95"), str(temp_outputs[0]), callback=_emit
            )
        else:
            raise ValueError(f"El paso {step} no es una llamada directa de WhiteboxTools.")
        if status not in (0, None):
            raise ProcessingError(f"WhiteboxTools devolvió estado {status} en paso {step}.")

    def _validate_step_values(self, step, temp_path):
        comparison = _validate_raster(temp_path, self.config.input_dem, self.config.block_rows, mask_policy="subset")
        with rasterio.open(temp_path) as ds:
            for start in range(0, ds.height, self.config.block_rows):
                window = Window(0, start, ds.width, min(self.config.block_rows, ds.height - start))
                valid = ds.read_masks(1, window=window) > 0
                values = ds.read(1, window=window)[valid]
                if not np.isfinite(values).all():
                    raise ProcessingError(f"{PRODUCT_FILES[step][0]} tiene valores no finitos dentro de su máscara.")
        if step == "90":
            with rasterio.open(temp_path) as ds:
                for start in range(0, ds.height, self.config.block_rows):
                    window = Window(0, start, ds.width, min(self.config.block_rows, ds.height - start))
                    valid = ds.read_masks(1, window=window) > 0
                    values = ds.read(1, window=window)[valid]
                    if not np.isfinite(values).all():
                        raise ProcessingError("El relleno de depresiones contiene valores no finitos.")
        elif step == "91":
            with rasterio.open(self.config.input_dem) as original, rasterio.open(temp_path) as conditioned:
                decreases = 0
                minimum_difference = math.inf
                for start in range(0, original.height, self.config.block_rows):
                    window = Window(0, start, original.width, min(self.config.block_rows, original.height - start))
                    valid = (original.read_masks(1, window=window) > 0) & (conditioned.read_masks(1, window=window) > 0)
                    diff = conditioned.read(1, window=window)[valid].astype(np.float64) - original.read(1, window=window)[valid]
                    if diff.size:
                        decreases += int(np.count_nonzero(diff < -1e-4))
                        minimum_difference = min(minimum_difference, float(diff.min()))
                comparison["disminuciones_menores_a_tolerancia"] = decreases
                comparison["diferencia_minima_m"] = minimum_difference if math.isfinite(minimum_difference) else None
                if decreases:
                    raise ProcessingError(f"El MDE acondicionado disminuyó elevación en {decreases} celdas.")
        elif step == "92":
            with rasterio.open(temp_path) as ds:
                for start in range(0, ds.height, self.config.block_rows):
                    window = Window(0, start, ds.width, min(self.config.block_rows, ds.height - start))
                    mask = ds.read_masks(1, window=window) > 0
                    values = ds.read(1, window=window)[mask]
                    if values.size and ((values < -1e-5).any() or (values > 90.0001).any()):
                        raise ProcessingError("Pendiente fuera del intervalo 0–90 grados.")
        elif step == "93":
            stats = _stats_raster(temp_path, self.config.block_rows, categorical=True)
            unexpected = set(stats.get("conteos_valor", {})) - {str(v) for v in SUPPORTED_D8_CODES}
            comparison["conteos_codigos_D8"] = stats.get("conteos_valor", {})
            comparison["codigos_no_reconocidos"] = sorted(unexpected)
            comparison["sin_direccion_borde"] = self._count_zero_d8_border(temp_path, edge=True)
            comparison["sin_direccion_interior"] = self._count_zero_d8_border(temp_path, edge=False)
            if unexpected:
                raise ProcessingError(f"Códigos D8 no reconocidos: {unexpected}")
        elif step == "94":
            with rasterio.open(temp_path) as ds:
                comparison["tipo_dato_acumulacion"] = ds.dtypes[0]
                comparison["area_celda_m2"] = self.cell_area
                comparison["factor_area_m2_por_celda"] = self.cell_area
                comparison["advertencia_precision"] = (
                    "Float32 representa exactamente enteros solo hasta 2^24; revisar el dtype y rango del raster."
                )
            stats = _stats_raster(temp_path, self.config.block_rows)
            if stats.get("minimo", 0) < 0:
                raise ProcessingError("La acumulación D8 contiene valores negativos.")
            comparison["area_contribuyente"] = {
                "minimo_m2": stats.get("minimo", 0) * self.cell_area,
                "maximo_m2": stats.get("maximo", 0) * self.cell_area,
                "media_m2": stats.get("media", 0) * self.cell_area,
                "nota": "Área equivalente = celdas acumuladas × área real de celda; D8 discreto.",
            }
        elif step == "95":
            stats = _stats_raster(temp_path, self.config.block_rows, categorical=True)
            counts = stats.get("conteos_valor", {})
            values = set(counts)
            if not values.issubset({"0", "1"}):
                raise ProcessingError(f"Red potencial debe contener solo 0 y 1; valores: {values}")
            comparison["conteos_fondo_y_corriente"] = counts
            comparison["umbral_celdas"] = self.config.stream_threshold_cells
            comparison["area_equivalente_umbral_km2"] = (
                self.config.stream_threshold_cells * self.cell_area / 1_000_000
            )
        elif step == "97":
            stats = _stats_raster(temp_path, self.config.block_rows)
            comparison["celdas_sin_resultado_respecto_a_MDE"] = (
                comparison["comparacion_mde"]["celdas_validas_referencia"]
                - comparison["comparacion_mde"]["celdas_validas_producto"]
            )
            comparison["nota_sin_conexion"] = "No se rellenan celdas que WhiteboxTools deje sin resultado."
            stream_path = self.product_path("95")
            stream_count = 0
            stream_cells_missing_hand = 0
            stream_hand_not_zero = 0
            max_stream_hand = 0.0
            with rasterio.open(stream_path) as streams, rasterio.open(temp_path) as hand:
                for start in range(0, hand.height, self.config.block_rows):
                    window = Window(0, start, hand.width, min(self.config.block_rows, hand.height - start))
                    smask = streams.read_masks(1, window=window) > 0
                    hmask = hand.read_masks(1, window=window) > 0
                    sdata = streams.read(1, window=window)
                    hdata = hand.read(1, window=window)
                    stream_cells = smask & (sdata == 1)
                    stream_count += int(stream_cells.sum())
                    stream_cells_missing_hand += int(np.count_nonzero(stream_cells & ~hmask))
                    valid_stream_hand = stream_cells & hmask
                    if valid_stream_hand.any():
                        absolute = np.abs(hdata[valid_stream_hand].astype(np.float64))
                        max_stream_hand = max(max_stream_hand, float(absolute.max()))
                        stream_hand_not_zero += int(np.count_nonzero(absolute > 1e-3))
            comparison["celdas_red_potencial"] = stream_count
            comparison["celdas_red_sin_HAND_valido"] = stream_cells_missing_hand
            comparison["celdas_red_HAND_mayor_a_1e-3_m"] = stream_hand_not_zero
            comparison["maximo_abs_HAND_en_red_m"] = max_stream_hand
            if stream_count == 0:
                raise ProcessingError("HAND no puede validarse: la red potencial no tiene celdas de corriente.")
            if stream_cells_missing_hand or stream_hand_not_zero:
                raise ProcessingError(
                    "HAND no es cercano a cero en todas las celdas de red o dejó celdas de red sin salida."
                )
            if stats.get("celdas_validas", 0):
                with rasterio.open(temp_path) as ds:
                    if ds.nodata is not None and ds.nodata == 0:
                        _emit("AVISO: HAND usa 0 como NoData; inspeccionar si los ceros de canal quedan enmascarados.")
        return comparison

    @staticmethod
    def _document_mask_policy(step):
        policies = {
            "90": "MDE rellenado: máscara dentro de la fuente; profundidad: intersección válida de fuente y relleno.",
            "91": "MDE acondicionado: diferencias de máscara se contabilizan; cobertura fuera de fuente no permitida.",
            "92": "Pendiente: el cálculo local puede reducir válidos en bordes o vecindades NoData; se cuentan diferencias.",
            "93": "D8: las celdas sin dirección son código 0 válido; NoData conserva la cobertura de terreno.",
            "94": "Acumulación: diferencias se registran y no se admite expansión fuera de la máscara del MDE.",
            "95": "Red potencial: 0 es fondo válido, 1 es cauce; NoData permanece separado del fondo.",
            "96": "TWI: intersección de máscaras de acumulación y pendiente, con acumulación positiva.",
            "97": "HAND: las celdas sin conexión pueden quedar NoData; no se rellenan artificialmente.",
        }
        return policies[step]

    @staticmethod
    def _method_description(step):
        return {
            "90": "WhiteboxTools FillDepressions, fix_flats=False; profundidad = MDE rellenado - MDE original.",
            "91": "WhiteboxTools FillDepressions, fix_flats=True; MDE de acondicionamiento hidrológico.",
            "92": "WhiteboxTools Slope en grados sobre el MDE original; separado del método Horn de terrain_factors.py.",
            "93": "WhiteboxTools D8Pointer, convención nativa WhiteboxTools (ESRI desactivada).",
            "94": "WhiteboxTools D8FlowAccumulation a partir del puntero; celdas, sin log ni clip.",
            "95": "WhiteboxTools ExtractStreams con umbral de celdas configurable.",
            "96": "TWI = ln((celdas acumuladas × ancho de celda) / tan(pendiente)); D8 por bloques.",
            "97": "WhiteboxTools ElevationAboveStream sobre MDE acondicionado y drenaje potencial.",
        }[step]

    def _count_zero_d8_border(self, path, edge=True):
        """Cuenta códigos 0 de D8 en borde raster/adyacentes a NoData o interiores."""
        total = 0
        with rasterio.open(path) as ds:
            for start in range(0, ds.height, self.config.block_rows):
                stop = min(start + self.config.block_rows, ds.height)
                halo_start, halo_stop = max(0, start - 1), min(ds.height, stop + 1)
                halo = Window(0, halo_start, ds.width, halo_stop - halo_start)
                mask = ds.read_masks(1, window=halo) > 0
                inner = mask.copy()
                inner[0, :] = False
                inner[-1, :] = False
                inner[:, 0] = False
                inner[:, -1] = False
                inner[1:, :] &= mask[:-1, :]
                inner[:-1, :] &= mask[1:, :]
                inner[:, 1:] &= mask[:, :-1]
                inner[:, :-1] &= mask[:, 1:]
                # Diagonales
                inner[1:, 1:] &= mask[:-1, :-1]
                inner[1:, :-1] &= mask[:-1, 1:]
                inner[:-1, 1:] &= mask[1:, :-1]
                inner[:-1, :-1] &= mask[1:, 1:]
                offset = start - halo_start
                local_mask = mask[offset:offset + (stop - start)]
                local_inner = inner[offset:offset + (stop - start)]
                data = ds.read(1, window=Window(0, start, ds.width, stop - start))
                zero = local_mask & (data == 0)
                edge_cells = local_mask & ~local_inner
                total += int(np.count_nonzero(zero & (edge_cells if edge else ~edge_cells)))
        return total

    def _compute_stats_and_sidecar(self, step, output_paths, dependencies_by_output, validation, elapsed):
        output_paths = [Path(p) for p in output_paths]
        product_records = []
        for output, dependencies in zip(output_paths, dependencies_by_output):
            stats = _stats_raster(output, self.config.block_rows, categorical=step in ("93", "95"))
            record = {
                "step": step,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "inputs": [self.fingerprint(path) for path in dependencies],
                "parameters": self._params(step),
                "validation": validation,
                "statistics": stats,
                "metodologia": self._method_description(step),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "whitebox_version": self.tool_version,
            }
            sidecar = self.sidecar_path(output)
            temp_sidecar = sidecar.with_name(sidecar.name + f".{uuid.uuid4().hex}.tmp")
            temp_sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if sidecar.exists() and not self.config.overwrite:
                temp_sidecar.unlink(missing_ok=True)
                raise FileExistsError(f"No se sobrescribe metadato existente: {sidecar}")
            os.replace(temp_sidecar, sidecar)
            product_records.append(record)
        self.records.append({
            "paso": step, "estado": "completado", "archivos": [str(p) for p in output_paths],
            "estadisticas": [r["statistics"] for r in product_records],
            "segundos": elapsed,
        })
        return product_records

    def _write_reports(self, status, error=None):
        report = {
            "run_id": self.run_id,
            "estado": status,
            "error": error,
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "input_dem": str(self.config.input_dem),
            "configuracion": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.config).items()},
            "whitebox_version": self.tool_version,
            "whitebox_python_package_version": importlib_metadata.version("whitebox"),
            "python": platform.python_version(),
            "rasterio": rasterio.__version__,
            "numpy": np.__version__,
            "preflight": self.preflight_record if hasattr(self, "preflight_record") else None,
            "resultados": self.records,
            "configuracion_whitebox": getattr(self, "tool_configuration", None),
            "advertencia_version": getattr(self, "version_warning", None),
        }
        json_path = self.config.output_dir / f"reporte_hidrologico_{self.run_id}.json"
        csv_path = self.config.output_dir / f"reporte_hidrologico_{self.run_id}.csv"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "run_id", "paso", "estado", "archivo", "segundos", "celdas_validas",
                "celdas_invalidas", "minimo", "maximo", "media", "desviacion_estandar",
                "whitebox_version", "parametros", "error",
            ])
            writer.writeheader()
            if self.records:
                for record in self.records:
                    files = record.get("archivos", [record.get("archivo", "")]) or [""]
                    stat_list = record.get("estadisticas", [])
                    for index, file in enumerate(files):
                        stat = stat_list[index] if index < len(stat_list) else {}
                        step = record.get("paso")
                        writer.writerow({
                            "run_id": self.run_id, "paso": step, "estado": record.get("estado"),
                            "archivo": file, "segundos": record.get("segundos"),
                            "celdas_validas": stat.get("celdas_validas"),
                            "celdas_invalidas": stat.get("celdas_invalidas"),
                            "minimo": stat.get("minimo"), "maximo": stat.get("maximo"),
                            "media": stat.get("media"),
                            "desviacion_estandar": stat.get("desviacion_estandar_poblacional"),
                            "whitebox_version": self.tool_version,
                            "parametros": json.dumps(self._params(step), ensure_ascii=False)
                            if step in PRODUCT_FILES else "",
                            "error": record.get("error") or error or "",
                        })
            else:
                writer.writerow({"run_id": self.run_id, "estado": status, "error": error or ""})
        return json_path, csv_path

    def run_step(self, step):
        if step not in PRODUCT_FILES:
            raise ValueError(f"Paso no reconocido: {step}; opciones {list(PRODUCT_FILES)}")
        self._check_resources()
        dependencies = self._dependencies(step)
        outputs = [self.config.output_dir / name for name in PRODUCT_FILES[step]]
        if self._reuse_step_if_valid(step, outputs, dependencies):
            return outputs
        existing = [p for p in [*outputs, *(self.sidecar_path(p) for p in outputs)] if p.exists()]
        if existing and not self.config.overwrite:
            raise FileExistsError(f"No se sobrescriben productos/metadatos existentes: {existing}")
        step_started = time.perf_counter()
        temp_outputs = [p.with_name(p.stem + f".{self.run_id}.tmp" + p.suffix) for p in outputs]
        try:
            if step == "96":
                calculate_twi(self.product_path("94"), self.product_path("92"), temp_outputs[0],
                              self.config.slope_floor_degrees, self.config.block_rows)
            elif step == "90":
                self._run_wbt_step(step, [temp_outputs[0]])
                self._validate_step_values(step, temp_outputs[0])
                depth_temp = temp_outputs[1]
                profundidad_depresiones(temp_outputs[0], self.config.input_dem, depth_temp, self.config.block_rows)
                depth_validation = _validate_raster(
                    depth_temp, self.config.input_dem, self.config.block_rows, mask_policy="exact"
                )
            else:
                self._run_wbt_step(step, temp_outputs)
            validation = self._validate_step_values(step, temp_outputs[0])
            validation["politica_mascara"] = self._document_mask_policy(step)
            if step == "90":
                validation["profundidad_depresiones"] = depth_validation
                validation["profundidad_depresiones"]["politica_mascara"] = self._document_mask_policy(step)
                validation["profundidad_depresiones"]["metricas_depresiones"] = metricas_profundidad_depresiones(
                    temp_outputs[1], self.cell_area, self.config.block_rows
                )
            for temp, final in zip(temp_outputs, outputs):
                if final.exists() and not self.config.overwrite:
                    raise FileExistsError(final)
                os.replace(temp, final)
            dependency_sets = [dependencies] * len(outputs)
            if step == "90":
                dependency_sets[1] = [*dependencies, outputs[0]]
            self._compute_stats_and_sidecar(
                step, outputs, dependency_sets, validation, time.perf_counter() - step_started
            )
            return outputs
        except Exception:
            for temp in temp_outputs:
                temp.unlink(missing_ok=True)
            raise

    def run(self, steps=None):
        selected_steps = list(steps or ALL_STEPS)
        unknown = set(selected_steps) - set(PRODUCT_FILES)
        if unknown:
            raise ValueError(f"Pasos desconocidos: {sorted(unknown)}")
        start = time.perf_counter()
        status, error = "completado", None
        try:
            self.preflight_record = self.preflight()
            self.whitebox, self.wbt_setup = _make_runner(self.config.output_dir, self.config.processing_threads)
            self.tool_version = _tool_version(self.whitebox)
            fill_validation = _verify_fill_flats_parameter(self.whitebox)
            self.version_warning = None
            if self.config.expected_whitebox_version and self.tool_version != self.config.expected_whitebox_version:
                self.version_warning = (
                    f"Instalada {self.tool_version}; referencia configurada {self.config.expected_whitebox_version}."
                )
                if not self.config.allow_version_mismatch:
                    raise ProcessingError(self.version_warning)
                _emit("AVISO DE VERSIÓN: " + self.version_warning)
            self.tool_configuration = {**self.wbt_setup, "fill_flats": fill_validation}
            for step in selected_steps:
                _emit(f"[{step}] Iniciando: {PRODUCT_FILES[step]}")
                self.run_step(step)
            self.records.append({"paso": "run", "estado": "completado", "segundos": time.perf_counter() - start})
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            self.records.append({"paso": "run", "estado": "error", "error": error})
            raise
        finally:
            self.records.append({"paso": "run_total", "estado": status, "segundos": time.perf_counter() - start})
            self.report_paths = self._write_reports(status, error)
        return self.report_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_INPUT_DEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step", choices=[*ALL_STEPS, "all"], default="all")
    parser.add_argument("--threshold-cells", type=int, default=DEFAULT_STREAM_THRESHOLD)
    parser.add_argument("--threads", type=int, default=DEFAULT_PROCESSING_THREADS)
    parser.add_argument("--slope-floor-degrees", type=float, default=DEFAULT_SLOPE_FLOOR_DEGREES)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--overwrite", action="store_true", help="Autoriza reemplazar salidas existentes.")
    parser.add_argument("--resume", action="store_true", help="Reutiliza solo productos con huella/procedencia válidas.")
    parser.add_argument("--require-whitebox-2-4", action="store_true", help="Falla si WhiteboxTools no es 2.4.0.")
    args = parser.parse_args(argv)
    config = HydrologyConfig(
        input_dem=args.dem,
        output_dir=args.output_dir,
        stream_threshold_cells=args.threshold_cells,
        processing_threads=args.threads,
        slope_floor_degrees=args.slope_floor_degrees,
        block_rows=args.block_rows,
        overwrite=args.overwrite,
        resume=args.resume,
        allow_version_mismatch=not args.require_whitebox_2_4,
    )
    processor = HydrologicalProcessor(config)
    reports = processor.run(None if args.step == "all" else [args.step])
    print(f"Reporte JSON: {reports[0]}")
    print(f"Reporte CSV: {reports[1]}")


if __name__ == "__main__":
    main()
