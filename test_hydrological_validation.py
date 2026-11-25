"""Pruebas sintéticas del diagnóstico hidrológico 98.1; no leen datos reales."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    from scripts.hydrological_validation import PRODUCTS, HydrologicalValidator, ValidationConfig
except ImportError:  # Permite ejecutar este archivo junto al módulo dentro de outputs/.
    from hydrological_validation import PRODUCTS, HydrologicalValidator, ValidationConfig

try:
    from whitebox.whitebox_tools import WhiteboxTools
except ImportError:  # WhiteboxTools solo se usa como contraste opcional con un raster sintético.
    WhiteboxTools = None


class HydrologicalValidationSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.products = self.root / "products"
        self.products.mkdir()
        self.gpkg = self.root / "hydro.gpkg"
        self.output = self.root / "diagnosis"
        self.width, self.height = 12, 10
        self.transform = from_origin(500000, 2200000, 15, 15)
        self.crs = "EPSG:6368"
        self.mask = np.ones((self.height, self.width), dtype=bool)
        self.mask[0, 0] = False
        y, x = np.indices(self.mask.shape)
        self.dem = (120.0 - x * 0.2 + y * 0.05).astype("float32")
        self.dem[~self.mask] = -32768
        self.reference = self.root / "reference.tif"
        self._write(self.reference, self.dem, "float32", -32768)

        arrays = {}
        arrays["mde_relleno_sin_flats"] = self.dem.copy()
        arrays["mde_acondicionado"] = self.dem.copy()
        arrays["mde_acondicionado"][4, 4] += 0.5
        arrays["profundidad_depresiones"] = np.zeros(self.mask.shape, dtype="float32")
        arrays["profundidad_depresiones"][4, 4] = 4.0
        arrays["pendiente_grados"] = np.full(self.mask.shape, 0.1, dtype="float32")
        arrays["direccion_d8"] = np.ones(self.mask.shape, dtype="uint8")
        arrays["direccion_d8"][5, :] = 2  # Whitebox-native east, matching the synthetic accumulation ramp.
        arrays["direccion_d8"][:, -1] = 0
        arrays["direccion_d8"][5, 5] = 0  # interior D8=0
        arrays["acumulacion_celdas"] = np.tile(np.arange(1, 13), (10, 1)).astype("float32")
        arrays["acumulacion_celdas"][:, -1] = 20
        arrays["acumulacion_celdas"][5] = (16777216 + np.arange(12)).astype("float32")
        arrays["acumulacion_celdas"][5, 9] = arrays["acumulacion_celdas"][5, 8] - 2
        arrays["drenaje_potencial"] = np.zeros(self.mask.shape, dtype="uint8")
        arrays["drenaje_potencial"][5, 1:10] = 1
        arrays["twi"] = np.log(
            arrays["acumulacion_celdas"] * 15.0 / np.tan(np.deg2rad(np.maximum(arrays["pendiente_grados"], 0.1)))
        ).astype("float32")
        arrays["hand"] = np.full(self.mask.shape, 5.0, dtype="float32")
        arrays["hand"][5, 1:10] = 0.0

        self.product_paths = {}
        for key, filename in PRODUCTS.items():
            path = self.products / filename
            dtype = str(arrays[key].dtype)
            nodata = -9999 if dtype.startswith("float") else 255
            out = arrays[key].copy()
            out[~self.mask] = nodata
            if key == "twi":
                out[2, 2] = nodata  # NoData propio de TWI, independiente del MDE.
            if key == "hand":
                out[1, 1] = nodata
            self._write(path, out, dtype, nodata)
            self.product_paths[key] = path
        self._write_vectors()

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, path, array, dtype, nodata):
        with rasterio.open(
            path, "w", driver="GTiff", height=self.height, width=self.width,
            count=1, dtype=dtype, crs=self.crs, transform=self.transform,
            nodata=nodata,
        ) as dst:
            dst.write(array.astype(dtype), 1)

    def _write_layer(self, gdf, layer):
        mode = "w" if not self.gpkg.exists() else "a"
        gdf.to_file(self.gpkg, layer=layer, driver="GPKG", mode=mode, index=False)

    def _write_vectors(self):
        xmin, ymax = 500000, 2200000
        xmax, ymin = xmin + 12 * 15, ymax - 10 * 15
        basin_geoms = [box(xmin, ymin, xmin + 2 * 15, ymax),
                       box(xmin + 2 * 15, ymin, xmin + 4 * 15, ymax),
                       box(xmin + 4 * 15, ymin, xmin + 6 * 15, ymax),
                       box(xmin + 6 * 15, ymin, xmin + 8 * 15, ymax),
                       box(xmin + 8 * 15, ymin, xmax, ymax),
                       box(xmax + 500, ymin, xmax + 1000, ymax)]
        basins = gpd.GeoDataFrame(
            {"_dcag_subbasin": ["RH15Aa", "RH15Ab", "RH16Ba", "RH16Bb", "RH16Bc", "RH99ZZ"]},
            geometry=basin_geoms, crs=self.crs
        )
        self._write_layer(basins, "subcuencas_regionales")
        municipal = gpd.GeoDataFrame({"name": ["Villa de Álvarez"]},
                                     geometry=[box(xmin, ymin, xmax, ymax)], crs=self.crs)
        self._write_layer(municipal, "limite_municipal")
        streams = gpd.GeoDataFrame(
            {"_dcag_hydro_category": ["corriente_perenne", "corriente_intermitente", "canal",
                                      "corriente_intermitente"],
             "ENABLED": [0, 1, 1, 0]},
            geometry=[LineString([(xmin, ymax - 5 * 15), (xmax, ymax - 5 * 15)]),
                      LineString([(xmin + 15, ymax - 6 * 15), (xmax - 15, ymax - 6 * 15)]),
                      LineString([(xmin, ymin + 15), (xmax, ymin + 15)]),
                      LineString([(xmax + 100, ymin), (xmax + 100, ymax)])], crs=self.crs,
        )
        self._write_layer(streams, "corrientes_regionales")
        drainage = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(xmin + 90, ymin + 75)], crs=self.crs)
        self._write_layer(drainage, "drenaje_regional")

    def _config(self, **kwargs):
        params = dict(
            reference_dem=self.reference,
            products_dir=self.products,
            hydrography_gpkg=self.gpkg,
            output_dir=self.output,
            expected_crs=self.crs,
            expected_resolution_m=15.0,
            expected_width=self.width,
            expected_height=self.height,
            expected_valid_cells=int(self.mask.sum()),
            block_rows=3,
            histogram_bins=64,
            generate_maps=False,
        )
        params.update(kwargs)
        return ValidationConfig(**params)

    def test_integrated_diagnostic_preserves_grid_and_reports_mask_differences(self):
        report = HydrologicalValidator(self._config()).run()
        self.assertTrue(all(report["validacion_cuadricula_y_mascaras"][key]["cuadricula_coincide"]
                            for key in PRODUCTS))
        self.assertEqual(report["validacion_cuadricula_y_mascaras"]["twi"]["metadatos"]["nodata"], -9999)
        self.assertEqual(report["validacion_cuadricula_y_mascaras"]["twi"]["mascara_vs_referencia"]["perdidas"], 1)
        self.assertEqual(report["HAND"]["diferencias_mascara"], 1)
        self.assertEqual(report["HAND"]["cobertura_por_zona"]["Villa de Álvarez"]["hand_perdidas"], 1)
        self.assertTrue(Path(report["salidas"]["json"]).is_file())
        self.assertEqual(report["estadisticas_raster"]["twi"]["celdas_validas"], int(self.mask.sum()) - 1)
        parsed = json.loads(Path(report["salidas"]["json"]).read_text(encoding="utf-8"))
        self.assertIn("TWI", parsed)
        self.assertIn("comparacion_MDE_original_vs_acondicionamientos", parsed)

    def test_d8_zero_cells_are_located_and_classified(self):
        report = HydrologicalValidator(self._config()).run()
        d8 = report["diagnostico_D8_y_acumulacion"]
        self.assertGreaterEqual(d8["conteos"]["zero_total"], self.height + 1)
        self.assertEqual(d8["ceros_por_clasificacion"]["interior"], 1)
        self.assertIn("RH16Ba", d8["ceros_por_subcuenca"])
        self.assertGreater(d8["conteos"]["comparaciones_acumulacion_sobre_2pow24"], 0)
        self.assertGreater(d8["conteos"]["acumulacion_igual_adyacente_sobre_2pow24"], 0)
        self.assertEqual(d8["conteos"]["acumulacion_no_monotona_sobre_2pow24"], 1)

    def test_all_eight_whitebox_native_offsets_individually(self):
        expected = {
            1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
            16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0),
        }
        from_module = __import__(HydrologicalValidator.__module__, fromlist=["_d8_neighbor"])
        for code, offset in expected.items():
            with self.subTest(code=code):
                self.assertEqual(from_module.D8_OFFSETS[code], offset)
                self.assertEqual(from_module._d8_neighbor(10, 10, code),
                                 (10 + offset[0], 10 + offset[1]))

    def test_d8_visual_categories_retain_native_codes(self):
        codes = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 3, 6, 12, 24, 48, 96, 128.5], 9)
        self.assertEqual([int(norm(value)) for value in codes], list(range(9)))

    @unittest.skipIf(WhiteboxTools is None, "No está instalado el binding WhiteboxTools.")
    def test_d8_directions_against_whiteboxtools_240_synthetic(self):
        from_module = __import__(HydrologicalValidator.__module__, fromlist=["D8_OFFSETS"])
        wbt = WhiteboxTools()
        executable = Path(wbt.exe_path) / wbt.exe_name
        version_result = subprocess.run([str(executable), "--version"], cwd=wbt.exe_path,
                                        capture_output=True, text=True, check=True, timeout=30)
        version = version_result.stdout + version_result.stderr
        if "WhiteboxTools v2.4.0" not in version:
            self.skipTest(f"Se requiere WhiteboxTools 2.4.0; disponible: {version.splitlines()[0]}")
        expected = from_module.D8_OFFSETS
        for code, (dr, dc) in expected.items():
            with self.subTest(code=code):
                dem_path = self.root / f"d8_input_{code}.tif"
                output_path = self.root / f"d8_output_{code}.tif"
                synthetic = np.full((5, 5), 30.0, dtype="float32")
                synthetic[2, 2] = 20.0
                synthetic[2 + dr, 2 + dc] = 10.0
                with rasterio.open(
                    dem_path, "w", driver="GTiff", height=5, width=5, count=1,
                    dtype="float32", crs=self.crs, transform=from_origin(500000, 2200000, 15, 15),
                    nodata=-9999,
                ) as source:
                    source.write(synthetic, 1)
                result = subprocess.run(
                    [str(executable), "--run=D8Pointer", f"--dem='{dem_path}'",
                     f"--output='{output_path}'", "-v=false"],
                    cwd=wbt.exe_path, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with rasterio.open(output_path) as generated:
                    self.assertEqual(int(generated.read(1)[2, 2]), code)

    def test_twi_formula_and_extreme_locations_are_checked(self):
        report = HydrologicalValidator(self._config()).run()
        twi = report["TWI"]
        self.assertTrue(twi["formula_verificable"])
        self.assertLess(twi["error_absoluto_maximo_formula"], 1e-4)
        self.assertIsNotNone(twi["extremos_espaciales"]["minimo"])
        self.assertTrue(twi["extremos_espaciales"]["celdas_maximas"])

    def test_stream_comparison_uses_only_natural_categories_and_keeps_enabled_zero(self):
        report = HydrologicalValidator(self._config()).run()
        comparison = report["comparacion_red_INEGI_vs_derivada"]
        vectors = report["geometrias_cargadas"]
        self.assertEqual(vectors["corrientes_oficiales_naturales"], 3)
        from_module = __import__(HydrologicalValidator.__module__, fromlist=["_load_geometries"])
        loaded = from_module._load_geometries(self._config())
        self.assertEqual({name for name, _ in loaded["basins"]},
                         {"RH15Aa", "RH15Ab", "RH16Ba", "RH16Bb", "RH16Bc"})
        self.assertTrue(any(geom.bounds[0] > 500180 for geom in loaded["streams"].geometry))
        self.assertEqual(comparison["filtro_ENABLED"], "No aplicado; ENABLED=0 y 1 se conservan.")
        self.assertGreater(comparison["pixeles_oficiales"], 0)

    def test_depression_area_uses_reference_pixel_area(self):
        report = HydrologicalValidator(self._config()).run()
        interval = report["depresiones_y_maximos"]["profundidad"]["intervalos_m"]["2-5"]
        self.assertEqual(interval["celdas"], 1)
        self.assertAlmostEqual(interval["superficie_km2"], 225 / 1_000_000)

    def test_output_files_are_unique_and_never_overwrite(self):
        first = HydrologicalValidator(self._config())
        second = HydrologicalValidator(self._config())
        self.assertNotEqual(first.output_files["json"], second.output_files["json"])
        self.assertFalse(first.config.overwrite)
        first.config.output_dir.mkdir(parents=True, exist_ok=True)
        first.output_files["json"].write_text("sentinel", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            first._validate_paths()
        self.assertEqual(first.output_files["json"].read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
