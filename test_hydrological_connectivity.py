"""Pruebas de conectividad global sobre un dominio D8 diminuto y sintético."""

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

try:
    from scripts.hydrological_connectivity import (
        ConnectivityConfig, HydrologicalConnectivity, _float32_accumulation_diagnostic,
    )
    from scripts.hydrological_validation import DEFAULT_SUBBASINS, PRODUCTS
except ImportError:
    from hydrological_connectivity import ConnectivityConfig, HydrologicalConnectivity, _float32_accumulation_diagnostic
    from hydrological_validation import DEFAULT_SUBBASINS, PRODUCTS


class HydrologicalConnectivitySyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.products = cls.root / "products"
        cls.products.mkdir()
        cls.reference = cls.root / "reference.tif"
        cls.gpkg = cls.root / "hydro.gpkg"
        cls.output = cls.root / "output"
        cls.height, cls.width = 5, 6
        cls.crs = "EPSG:6368"
        cls.transform = from_origin(500000, 2200000, 10, 10)
        cls.valid = np.ones((cls.height, cls.width), dtype=bool)
        cls.valid[4, 5] = False
        rows, cols = np.indices(cls.valid.shape)
        dem = (100 - cols * .2 + rows * .1).astype("float32")
        dem[~cls.valid] = -32768
        cls._write(cls.reference, dem, "float32", -32768)

        arrays = {
            "mde_relleno_sin_flats": dem.copy(),
            "profundidad_depresiones": np.zeros(cls.valid.shape, dtype="float32"),
            "mde_acondicionado": dem.copy(),
            "pendiente_grados": np.full(cls.valid.shape, 2.0, dtype="float32"),
            "direccion_d8": np.zeros(cls.valid.shape, dtype="uint8"),
            "acumulacion_celdas": np.ones(cls.valid.shape, dtype="float32"),
            "drenaje_potencial": np.zeros(cls.valid.shape, dtype="uint8"),
            "twi": np.full(cls.valid.shape, 3.0, dtype="float32"),
            "hand": np.full(cls.valid.shape, 1.0, dtype="float32"),
        }
        # RH15Aa -> RH15Ab -> RH16Ba -> RH16Bb -> RH16Bc -> exterior buffer outlet.
        arrays["direccion_d8"][2, :5] = 2  # Whitebox native east.
        arrays["direccion_d8"][2, 5] = 0
        arrays["acumulacion_celdas"][2, :] = np.arange(1, 7, dtype="float32")
        # Salida independiente dentro de RH15 y dos celdas que forman ciclo.
        arrays["direccion_d8"][3, 0] = 2
        arrays["direccion_d8"][3, 1] = 0
        arrays["direccion_d8"][0, 0] = 2
        arrays["direccion_d8"][0, 1] = 32
        # Una trayectoria termina al alcanzar NoData.
        arrays["direccion_d8"][3, 4] = 2
        arrays["direccion_d8"][3, 5] = 8
        arrays["acumulacion_celdas"][3, 0:2] = [1, 2]
        arrays["profundidad_depresiones"][1, 4] = 98.450439453125
        arrays["profundidad_depresiones"][1, 3] = 12.0
        arrays["mde_acondicionado"][1, 3:5] += arrays["profundidad_depresiones"][1, 3:5]

        for key, filename in PRODUCTS.items():
            arr = arrays[key].copy()
            dtype = str(arr.dtype)
            nodata = 255 if dtype == "uint8" else -9999
            arr[~cls.valid] = nodata
            if key == "twi":
                arr[1, 1] = nodata  # NoData propio de TWI.
            cls._write(cls.products / filename, arr, dtype, nodata)

        # Polígonos codificados por columnas: col 5 es el buffer fuera de cuencas.
        ymin = 2200000 - cls.height * 10
        basins = gpd.GeoDataFrame(
            {"_dcag_subbasin": list(DEFAULT_SUBBASINS),
             "DRENAJE": ["d1", "d2", "d3", "d4", "d5"],
             "DESCARGA": ["q1", "q2", "q3", "q4", "q5"]},
            geometry=[box(500000 + i * 10, ymin, 500000 + (i + 1) * 10, 2200000)
                      for i in range(5)], crs=cls.crs,
        )
        basins.to_file(cls.gpkg, layer="subcuencas_regionales", driver="GPKG", index=False)
        municipality = gpd.GeoDataFrame({"name": ["Villa de Álvarez"]},
                                        geometry=[box(500000, ymin, 500030, 2200000)], crs=cls.crs)
        municipality.to_file(cls.gpkg, layer="limite_municipal", driver="GPKG", mode="a", index=False)
        cls.config = ConnectivityConfig(
            reference_dem=cls.reference, products_dir=cls.products,
            hydrography_gpkg=cls.gpkg, output_dir=cls.output,
            target_cell=(2, 5), depression_xy=(500045, 2199985),
            depression_map_radius_m=60, block_rows=2, traversal_batch_size=7,
            display_max_dimension=500,
        )
        cls.report = HydrologicalConnectivity(cls.config).run()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def _write(cls, path, array, dtype, nodata):
        with rasterio.open(path, "w", driver="GTiff", height=cls.height, width=cls.width,
                           count=1, dtype=dtype, crs=cls.crs, transform=cls.transform,
                           nodata=nodata) as dst:
            dst.write(array.astype(dtype), 1)

    def test_reconstructs_maximum_area_by_inverse_d8_edges(self):
        maximum = self.report["area_contribuyente_maxima"]
        self.assertEqual(maximum["celdas_reconstruidas_inversamente"], 6)
        self.assertEqual(maximum["acumulacion_almacenada_float32"], 6.0)
        self.assertAlmostEqual(maximum["area_reconstruida_km2"], 0.0006)

    def test_contributions_are_separated_by_five_subbasins_and_exterior(self):
        rows = {r["subcuenca"]: r["celdas_area_maxima"]
                for r in self.report["aportacion_por_subcuenca_y_exterior"]}
        for name in DEFAULT_SUBBASINS:
            self.assertEqual(rows[name], 1)
        self.assertEqual(rows["territorio_exterior_a_las_cinco"], 1)

    def test_independent_catchment_and_multiple_outlets_are_reported(self):
        exits = self.report["salidas_hidrologicas"]["salidas_terminales"]
        self.assertGreater(exits, 1)
        self.assertTrue(any(o["celdas_RH15"] and not o["celdas_RH16"]
                            for o in self.report["salidas_hidrologicas"]["por_salida"]))

    def test_rh15_rh16_crossing_and_common_buffer_outlet_are_detected(self):
        crossing = self.report["transiciones_D8_entre_subcuencas"]
        self.assertGreaterEqual(crossing["RH15_a_RH16"], 1)
        self.assertTrue(crossing["salidas_comunes_de_RH15_y_RH16"])
        self.assertTrue(all(o["terminal_en_exterior_a_subcuencas"]
                            for o in self.report["salidas_hidrologicas"]["por_salida"]
                            if o["outlet_id"] in crossing["salidas_comunes_de_RH15_y_RH16"]))
        self.assertTrue(all("DRENAJE" in row and "DESCARGA" in row
                            for row in crossing["atributos_INEGI_DRENAJE_DESCARGA"].values()))

    def test_cycle_and_no_data_sink_are_distinguished(self):
        cycle = self.report["ciclos_y_trayectorias_sin_salida"]
        self.assertEqual(cycle["numero_ciclos_dirigidos"], 1)
        self.assertEqual(cycle["celdas_pertenecientes_a_ciclos"], 2)
        types = self.report["salidas_hidrologicas"]["por_tipo"]
        self.assertGreater(types["hacia_NoData"], 0)

    def test_valid_cells_are_conserved_between_routed_and_residual(self):
        summary = self.report["salidas_hidrologicas"]["conservacion_celdas_validas"]
        self.assertEqual(summary["D8_validas"], summary["suma"])
        self.assertEqual(summary["asignadas_a_salida"] + summary["sin_salida_terminal"], summary["D8_validas"])

    def test_each_raster_mask_is_reported_and_twi_nodata_is_independent(self):
        masks = self.report["mascaras_por_raster"]
        self.assertEqual(masks["twi"]["nodata_declarado"], -9999)
        self.assertEqual(masks["twi"]["perdidas_vs_MDE"], 1)
        self.assertEqual(masks["direccion_d8"]["nodata_o_invalidas"], 1)

    def test_float32_integer_precision_threshold_is_explicit(self):
        diagnostic = _float32_accumulation_diagnostic(16777220, 16777219)
        self.assertEqual(diagnostic["umbral_enteros_exactos"], 16777216)
        self.assertTrue(diagnostic["supera_umbral_enteros_exactos"])
        self.assertEqual(diagnostic["valor_almacenado_float32"], 16777220.0)
        self.assertEqual(diagnostic["ulp_float32_en_valor_almacenado"], 2.0)
        self.assertTrue(diagnostic["diferencia_menor_o_igual_a_media_ulp"])

    def test_depression_maximum_and_maps_are_generated(self):
        depression = self.report["depresion_maxima"]
        self.assertAlmostEqual(depression["profundidad_maxima_m"], 98.450439453125)
        self.assertEqual(depression["subcuenca"], "RH16Bc")
        self.assertTrue(depression["coincide_con_celda_de_profundidad_maxima"])
        self.assertEqual(depression["celdas_componente_8_conexa"], 2)
        for path in self.report["salidas"].values():
            self.assertTrue(Path(path).is_file())
        self.assertTrue(Path(depression["mapa_perfiles"]).is_file())


if __name__ == "__main__":
    unittest.main()
