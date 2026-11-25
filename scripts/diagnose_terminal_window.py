#!/usr/bin/env python
"""Diagnóstico local por ventana de los nueve productos hidrológicos 90–97.

Lee únicamente una ventana pequeña de cada GeoTIFF. No modifica los archivos fuente.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

PRODUCTS = {
    "90_MDE_Relleno_Sin_Flats": "90_MDE_Relleno_Sin_Flats.tif",
    "90_Profundidad_Depresiones": "90_Profundidad_Depresiones.tif",
    "91_MDE_Acondicionado": "91_MDE_Acondicionado.tif",
    "92_Pendiente_Grados": "92_Pendiente_Grados.tif",
    "93_Direccion_Flujo_D8": "93_Direccion_Flujo_D8.tif",
    "94_Acumulacion_Flujo_Celdas": "94_Acumulacion_Flujo_Celdas.tif",
    "95_Red_Drenaje_Potencial": "95_Red_Drenaje_Potencial.tif",
    "96_Indice_TWI": "96_Indice_TWI.tif",
    "97_Altura_HAND": "97_Altura_HAND.tif",
}


class DiagnosticError(RuntimeError):
    """Error de configuración, alineación o lectura."""


@dataclass
class Config:
    products_dir: Path
    output_dir: Path
    row: int = 9570
    col: int = 114
    radius: int = 5
    overwrite: bool = False


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value),
        encoding="utf-8",
    )


def _metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        return {
            "archivo": str(path),
            "width": ds.width,
            "height": ds.height,
            "crs": ds.crs.to_string() if ds.crs else None,
            "transform": list(ds.transform)[:6],
            "bounds": list(ds.bounds),
            "resolution": list(ds.res),
            "dtype": ds.dtypes[0],
            "nodata": ds.nodata,
            "count": ds.count,
            "driver": ds.driver,
            "compression": ds.compression.value if ds.compression else None,
        }


def _compare_grids(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(metadata)
    reference_name = names[0]
    reference = metadata[reference_name]
    comparisons = {}
    aligned_all = True

    for name in names:
        current = metadata[name]
        same_dimensions = (
            current["width"] == reference["width"]
            and current["height"] == reference["height"]
        )
        same_crs = current["crs"] == reference["crs"]
        same_transform = np.allclose(
            current["transform"], reference["transform"], rtol=0.0, atol=1e-9
        )
        aligned = same_dimensions and same_crs and same_transform
        comparisons[name] = {
            "same_dimensions_as_reference": same_dimensions,
            "same_crs_as_reference": same_crs,
            "same_transform_as_reference": bool(same_transform),
            "aligned": bool(aligned),
        }
        aligned_all = aligned_all and aligned

    return {
        "reference_product": reference_name,
        "all_products_aligned": bool(aligned_all),
        "comparisons": comparisons,
    }


def run_diagnostic(config: Config) -> tuple[Path, Path]:
    if config.row < 0 or config.col < 0:
        raise DiagnosticError("row y col deben ser enteros no negativos.")
    if config.radius < 0:
        raise DiagnosticError("radius debe ser un entero no negativo.")

    missing = [
        filename for filename in PRODUCTS.values()
        if not (config.products_dir / filename).is_file()
    ]
    if missing:
        raise DiagnosticError(
            "Faltan GeoTIFF requeridos en products_dir: " + ", ".join(missing)
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / "ventana_terminal_1829.csv"
    json_path = config.output_dir / "resumen_cuadricula_1829.json"
    if not config.overwrite and (csv_path.exists() or json_path.exists()):
        raise DiagnosticError(
            "Ya existen archivos de salida. Usa --overwrite solo si autorizas reemplazarlos."
        )

    metadata = {
        name: _metadata(config.products_dir / filename)
        for name, filename in PRODUCTS.items()
    }
    grid_check = _compare_grids(metadata)
    summary: dict[str, Any] = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "objetivo": {"row": config.row, "col": config.col},
        "radio_celdas": config.radius,
        "ventana_solicitada": 2 * config.radius + 1,
        "productos": metadata,
        "validacion_cuadricula": grid_check,
        "lectura_realizada": False,
        "observaciones": [],
    }

    reference = metadata[next(iter(PRODUCTS))]
    if not grid_check["all_products_aligned"]:
        summary["observaciones"].append(
            "No se extrajo la ventana: los productos no comparten exactamente "
            "dimensiones, CRS y transformación. Los mismos índices no representarían "
            "necesariamente la misma ubicación."
        )
        _write_json(json_path, summary)
        raise DiagnosticError(
            f"Los GeoTIFF no están alineados. Revisa el reporte: {json_path}"
        )

    height, width = reference["height"], reference["width"]
    if config.row >= height or config.col >= width:
        raise DiagnosticError(
            f"La celda ({config.row}, {config.col}) queda fuera del raster "
            f"({height} filas, {width} columnas)."
        )

    row0 = max(0, config.row - config.radius)
    col0 = max(0, config.col - config.radius)
    row1 = min(height, config.row + config.radius + 1)
    col1 = min(width, config.col + config.radius + 1)
    window = Window(col0, row0, col1 - col0, row1 - row0)

    arrays: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    for name, filename in PRODUCTS.items():
        with rasterio.open(config.products_dir / filename) as ds:
            values = ds.read(1, window=window, masked=False)
            valid = ds.read_masks(1, window=window) > 0
            if np.issubdtype(values.dtype, np.floating):
                valid &= np.isfinite(values)
            arrays[name] = values
            valid_masks[name] = valid

    with rasterio.open(config.products_dir / next(iter(PRODUCTS.values()))) as ref_ds:
        ref_transform = ref_ds.transform
    rows = []
    for local_row in range(int(window.height)):
        for local_col in range(int(window.width)):
            global_row = row0 + local_row
            global_col = col0 + local_col
            x, y = ref_transform * (global_col + 0.5, global_row + 0.5)
            record: dict[str, Any] = {
                "row": global_row,
                "col": global_col,
                "x": x,
                "y": y,
                "es_celda_objetivo": (
                    global_row == config.row and global_col == config.col
                ),
            }
            for name in PRODUCTS:
                is_valid = bool(valid_masks[name][local_row, local_col])
                value = arrays[name][local_row, local_col]
                record[f"{name}_valid"] = is_valid
                record[name] = _json_value(value) if is_valid else ""
            rows.append(record)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mask_comparison = {}
    reference_mask = valid_masks[next(iter(PRODUCTS))]
    for name, mask in valid_masks.items():
        mask_comparison[name] = {
            "valid_cells_in_window": int(mask.sum()),
            "invalid_cells_in_window": int(mask.size - mask.sum()),
            "different_from_reference_mask": int(np.count_nonzero(mask != reference_mask)),
        }

    center = {}
    center_local_row = config.row - row0
    center_local_col = config.col - col0
    for name in PRODUCTS:
        is_valid = bool(valid_masks[name][center_local_row, center_local_col])
        value = arrays[name][center_local_row, center_local_col]
        center[name] = {
            "valid": is_valid,
            "value": _json_value(value) if is_valid else None,
        }

    summary.update({
        "lectura_realizada": True,
        "ventana_leida": {
            "row_start": row0,
            "row_stop_exclusive": row1,
            "col_start": col0,
            "col_stop_exclusive": col1,
            "rows": int(window.height),
            "cols": int(window.width),
            "cells": int(window.height * window.width),
        },
        "coordenada_centro": {
            "x": rows[(config.row - row0) * int(window.width) + (config.col - col0)]["x"],
            "y": rows[(config.row - row0) * int(window.width) + (config.col - col0)]["y"],
        },
        "valores_celda_objetivo": center,
        "mascaras_en_ventana": mask_comparison,
        "csv": str(csv_path),
        "json": str(json_path),
        "observaciones": [
            "Los valores se extrajeron por ventana; no se cargaron los GeoTIFF completos.",
            "Los productos fuente se abrieron únicamente en modo lectura y no fueron modificados.",
            "Las diferencias de máscara se reportan; no se consideran automáticamente errores.",
        ],
    })
    _write_json(json_path, summary)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae una ventana pequeña de los nueve GeoTIFF hidrológicos 90–97, "
            "comprueba la alineación y reporta diferencias de máscara. No modifica fuentes."
        )
    )
    parser.add_argument(
        "--products-dir", type=Path,
        default=Path(r"G:\My Drive\Doctorado-DCAG\datos_territoriales\INEGI"
                     r"\CEM_4_0\procesados\Etapa_2_Hidrologia"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Por defecto: products-dir/diagnostico_98_3",
    )
    parser.add_argument("--row", type=int, default=9570)
    parser.add_argument("--col", type=int, default=114)
    parser.add_argument("--radius", type=int, default=5,
                        help="Radio en celdas; 5 produce una ventana 11x11.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.products_dir / "diagnostico_98_3"
    config = Config(
        products_dir=args.products_dir,
        output_dir=output_dir,
        row=args.row,
        col=args.col,
        radius=args.radius,
        overwrite=args.overwrite,
    )
    try:
        csv_path, json_path = run_diagnostic(config)
    except (DiagnosticError, OSError, rasterio.errors.RasterioError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
