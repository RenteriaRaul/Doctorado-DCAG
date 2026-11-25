"""Pruebas sintéticas de etapas hidrológicas y controles de integración."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

try:
    from scripts.hydrological_processing import (
        ALL_STEPS,
        PRODUCT_FILES,
        HydrologicalProcessor,
        HydrologyConfig,
        ProcessingError,
        RasterStats,
        calculate_twi,
        profundidad_depresiones,
    )
except ModuleNotFoundError:
    from hydrological_processing import (
        ALL_STEPS,
        PRODUCT_FILES,
        HydrologicalProcessor,
        HydrologyConfig,
        ProcessingError,
        RasterStats,
        calculate_twi,
        profundidad_depresiones,
    )


class TestHydrologicalProcessing(unittest.TestCase):
    def _synthetic_dem(self, path, width=15, height=15):
        rows, cols = np.indices((height, width))
        # Superficie inclinada con descenso hacia SE; se añaden depresión y zona plana.
        dem = (100.0 + (height - rows) * 0.7 + (width - cols) * 0.5).astype(np.float32)
        dem[height // 2, width // 2] -= 8.0
        dem[2:5, 2:5] = 115.0
        dem[0, 0] = -32768.0
        profile = {
            "driver": "GTiff", "width": width, "height": height, "count": 1,
            "dtype": "float32", "crs": "EPSG:32613",
            "transform": from_origin(500000, 2200000, 15, 15), "nodata": -32768.0,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(dem, 1)
        return dem

    def test_estadisticas_incrementales_sinteticas(self):
        stats = RasterStats()
        stats.add(np.array([1, 2, 3], dtype=np.float32), invalid_count=1)
        stats.add(np.array([4, 5], dtype=np.float32), invalid_count=2)
        result = stats.to_dict()
        self.assertEqual(result["celdas_validas"], 5)
        self.assertEqual(result["celdas_invalidas"], 3)
        self.assertAlmostEqual(result["media"], 3.0)
        self.assertAlmostEqual(result["desviacion_estandar_poblacional"], np.std([1, 2, 3, 4, 5]))

    def test_profundidad_depresiones_sintetica_y_nodata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, filled, depth = root / "dem.tif", root / "filled.tif", root / "depth.tif"
            dem = self._synthetic_dem(source)
            changed = dem.copy()
            changed[7, 7] += 8
            with rasterio.open(source) as src:
                profile = src.profile.copy()
            with rasterio.open(filled, "w", **profile) as dst:
                dst.write(changed, 1)
            profundidad_depresiones(filled, source, depth, block_rows=4)
            with rasterio.open(depth) as ds:
                out = ds.read(1)
                self.assertEqual(float(out[7, 7]), 8.0)
                self.assertEqual(float(out[0, 0]), -32768.0)
                self.assertEqual(ds.transform, profile["transform"])

    def test_twi_sintetico_por_bloques_y_piso_de_pendiente(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accum, slope, out = root / "acc.tif", root / "slope.tif", root / "twi.tif"
            profile = {
                "driver": "GTiff", "width": 4, "height": 3, "count": 1,
                "dtype": "float32", "crs": "EPSG:32613",
                "transform": from_origin(0, 45, 15, 15), "nodata": -9999.0,
            }
            with rasterio.open(accum, "w", **profile) as ds:
                ds.write(np.array([[1, 2, 4, 8], [1, 3, 5, 9], [1, 2, 3, -9999]], dtype=np.float32), 1)
            with rasterio.open(slope, "w", **profile) as ds:
                ds.write(np.array([[0.0, 1, 5, 10], [0.05, 2, 6, 15], [0.1, 3, 8, -9999]], dtype=np.float32), 1)
            calculate_twi(accum, slope, out, slope_floor_degrees=0.1, block_rows=1)
            with rasterio.open(out) as ds:
                values = ds.read(1, masked=True)
                self.assertTrue(np.isfinite(values.compressed()).all())
                self.assertEqual(float(ds.read(1)[2, 3]), -9999.0)
                self.assertEqual(ds.transform, profile["transform"])

    def test_whitebox_synthetic_pipeline_90_a_97(self):
        """Ejecuta herramientas WBT exclusivamente sobre un DEM diminuto temporal."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "synthetic_wbt_dem.tif"
            out = root / "Etapa_2_Hidrologia"
            self._synthetic_dem(source)
            config = HydrologyConfig(
                input_dem=source,
                output_dir=out,
                stream_threshold_cells=1,
                processing_threads=2,
                block_rows=4,
                allow_version_mismatch=True,
                memory_factor=1,
            )
            processor = HydrologicalProcessor(config)
            reports = processor.run(ALL_STEPS)
            self.assertTrue(reports[0].is_file())
            self.assertTrue(reports[1].is_file())
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["estado"], "completado")
            self.assertEqual(report.get("whitebox_version"), "2.4.0")
            self.assertTrue(report.get("whitebox_python_package_version"))
            with rasterio.open(out / PRODUCT_FILES["90"][0]) as no_flats, rasterio.open(out / PRODUCT_FILES["91"][0]) as with_flats:
                no_flats_values = no_flats.read(1, masked=True)
                with_flats_values = with_flats.read(1, masked=True)
                comparable = ~(no_flats_values.mask | with_flats_values.mask)
                # Superficie sintética plana debe evidenciar la aplicación del fix_flats.
                self.assertTrue(np.any(np.abs(no_flats_values.data[comparable] - with_flats_values.data[comparable]) > 1e-6))
            for filenames in PRODUCT_FILES.values():
                for filename in filenames:
                    product = out / filename
                    self.assertTrue(product.is_file(), filename)
                    self.assertTrue(Path(str(product) + ".provenance.json").is_file())
                    with rasterio.open(product) as ds:
                        self.assertEqual(ds.width, 15)
                        self.assertEqual(ds.height, 15)
                        self.assertEqual(ds.crs.to_epsg(), 32613)
            expected_slope = np.degrees(np.arctan(np.hypot(0.5 / 15.0, 0.7 / 15.0)))
            with rasterio.open(out / PRODUCT_FILES["92"][0]) as ds:
                slope = ds.read(1)
                self.assertTrue(0 <= slope[1, 10] <= 90)
                self.assertAlmostEqual(float(slope[1, 10]), expected_slope, delta=0.6)
            with rasterio.open(out / PRODUCT_FILES["93"][0]) as ds:
                d8 = ds.read(1, masked=True).compressed()
                self.assertTrue(set(np.unique(d8)).issubset({0, 1, 2, 4, 8, 16, 32, 64, 128}))
            with rasterio.open(out / PRODUCT_FILES["94"][0]) as ds:
                accumulation = ds.read(1, masked=True).compressed()
                self.assertTrue((accumulation >= 0).all())
                self.assertGreater(float(accumulation.max()), 1.0)
            with rasterio.open(out / PRODUCT_FILES["95"][0]) as ds:
                valid = ds.read(1, masked=True).compressed()
                self.assertTrue(set(np.unique(valid)).issubset({0.0, 1.0}))
            with rasterio.open(out / PRODUCT_FILES["97"][0]) as hand, rasterio.open(out / PRODUCT_FILES["95"][0]) as streams:
                smask = streams.read_masks(1) > 0
                hand_mask = hand.read_masks(1) > 0
                stream_pixels = smask & (streams.read(1) == 1)
                self.assertTrue(stream_pixels.any())
                self.assertTrue(np.all(hand_mask[stream_pixels]))
                np.testing.assert_allclose(hand.read(1)[stream_pixels], 0.0, atol=1e-3)
            step90 = next(record for record in report["resultados"] if record.get("paso") == "90")
            metrics = step90["estadisticas"][1]
            self.assertGreater(metrics["celdas_validas"], 0)
            self.assertIn("minimo", metrics)

    def test_resume_requiere_y_valida_procedencia(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "dem.tif", root / "out"
            self._synthetic_dem(source)
            config = HydrologyConfig(input_dem=source, output_dir=out, stream_threshold_cells=1, block_rows=4)
            first = HydrologicalProcessor(config)
            first.run(["90"])
            resume_config = HydrologyConfig(
                input_dem=source, output_dir=out, stream_threshold_cells=1, block_rows=4, resume=True
            )
            resumed = HydrologicalProcessor(resume_config)
            resumed.run(["90"])
            self.assertTrue(any(r.get("estado") == "reutilizado_validado" for r in resumed.records))
            # Corromper una copia sintética hace que la comprobación de SHA rechace su reutilización.
            product = out / PRODUCT_FILES["90"][0]
            with product.open("ab") as stream:
                stream.write(b"corruption")
            resumed_again = HydrologicalProcessor(resume_config)
            with self.assertRaises(ProcessingError):
                resumed_again.run(["90"])

    def test_no_sobrescribe_productos_sin_autorizacion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "dem.tif", root / "out"
            self._synthetic_dem(source)
            config = HydrologyConfig(input_dem=source, output_dir=out, block_rows=4)
            processor = HydrologicalProcessor(config)
            processor.preflight_record = processor.preflight()
            from whitebox import WhiteboxTools
            processor.whitebox = WhiteboxTools()
            processor.tool_version = str(processor.whitebox.version()).strip()
            product = out / PRODUCT_FILES["90"][0]
            product.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                processor.run_step("90")

    def test_detecta_producto_incompleto_al_reanudar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "dem.tif", root / "out"
            self._synthetic_dem(source)
            out.mkdir()
            (out / PRODUCT_FILES["90"][0]).write_bytes(b"partial")
            config = HydrologyConfig(input_dem=source, output_dir=out, block_rows=4, resume=True)
            processor = HydrologicalProcessor(config)
            processor.preflight_record = processor.preflight()
            processor.tool_version = "2.4.0"
            with self.assertRaises(ProcessingError):
                processor.run_step("90")

    def test_reporta_error_de_whitebox(self):
        class FailedWhitebox:
            def slope(self, *args, **kwargs):
                return 1

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "dem.tif"
            self._synthetic_dem(source)
            processor = HydrologicalProcessor(HydrologyConfig(input_dem=source, output_dir=root / "out"))
            processor.whitebox = FailedWhitebox()
            with self.assertRaises(ProcessingError):
                processor._run_wbt_step("92", [root / "slope.tmp.tif"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
