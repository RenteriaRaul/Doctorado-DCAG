"""Pruebas sintéticas del diagnóstico local por ventana."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

try:
    from scripts.diagnose_terminal_window import (
        Config, DiagnosticError, PRODUCTS, run_diagnostic
    )
except ImportError:
    from diagnose_terminal_window import (
        Config, DiagnosticError, PRODUCTS, run_diagnostic
    )


class DiagnoseTerminalWindowTests(unittest.TestCase):
    def _make_products(self, root: Path, *, misalign: str | None = None) -> Path:
        products_dir = root / "products"
        products_dir.mkdir()
        height, width = 24, 30
        transform = from_origin(500000, 2200000, 15, 15)
        for index, filename in enumerate(PRODUCTS.values()):
            this_transform = transform
            if misalign == filename:
                this_transform = from_origin(500015, 2200000, 15, 15)
            data = np.full((height, width), index + 1, dtype=np.float32)
            nodata = -9999.0 if index == 7 else -32768.0
            data[0, 0] = nodata
            if index == 7:
                data[12, 15] = nodata  # Diferencia deliberada de máscara.
            with rasterio.open(
                products_dir / filename,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:32613",
                transform=this_transform,
                nodata=nodata,
            ) as dst:
                dst.write(data, 1)
        return products_dir

    def test_extracts_small_window_and_reports_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            products = self._make_products(root)
            csv_path, json_path = run_diagnostic(
                Config(products, root / "out", row=12, col=15, radius=2)
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(report["validacion_cuadricula"]["all_products_aligned"])
            self.assertEqual(report["ventana_leida"]["rows"], 5)
            self.assertEqual(report["ventana_leida"]["cols"], 5)
            self.assertTrue(report["lectura_realizada"])
            self.assertFalse(
                report["valores_celda_objetivo"]["96_Indice_TWI"]["valid"]
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 25)
            self.assertEqual(sum(r["es_celda_objetivo"] == "True" for r in rows), 1)
            self.assertEqual(
                report["mascaras_en_ventana"]["96_Indice_TWI"]
                ["different_from_reference_mask"],
                1,
            )

    def test_refuses_to_extract_when_grids_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            products = self._make_products(root, misalign="93_Direccion_Flujo_D8.tif")
            with self.assertRaises(DiagnosticError):
                run_diagnostic(Config(products, root / "out", row=12, col=15, radius=1))
            report_path = root / "out" / "resumen_cuadricula_1829.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["validacion_cuadricula"]["all_products_aligned"])
            self.assertFalse(report["lectura_realizada"])

    def test_refuses_overwrite_without_permission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            products = self._make_products(root)
            config = Config(products, root / "out", row=12, col=15, radius=1)
            run_diagnostic(config)
            with self.assertRaises(DiagnosticError):
                run_diagnostic(config)

    def test_rejects_out_of_bounds_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            products = self._make_products(root)
            with self.assertRaises(DiagnosticError):
                run_diagnostic(Config(products, root / "out", row=999, col=1, radius=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
