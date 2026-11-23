"""Pruebas sintéticas unitarias para distance_to_streams.py."""

import unittest

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point

from scripts.distance_to_streams import (
    NODATA_OUTPUT,
    calcular_bloque_distancias,
    calcular_malla_por_bloques,
    construir_arbol_corrientes,
    distancias_exactas_a_corrientes,
    filtrar_corrientes_naturales,
)


class TestDistanceToStreams(unittest.TestCase):
    def test_distancia_conocida_a_linea_horizontal(self):
        tree, _ = construir_arbol_corrientes([LineString([(0, 0), (10, 0)])])
        result = distancias_exactas_a_corrientes([Point(5, 3)], tree)
        self.assertAlmostEqual(float(result[0]), 3.0)

    def test_distancia_conocida_a_linea_vertical(self):
        tree, _ = construir_arbol_corrientes([LineString([(2, 0), (2, 10)])])
        result = distancias_exactas_a_corrientes([Point(8, 6)], tree)
        self.assertAlmostEqual(float(result[0]), 6.0)

    def test_distancia_conocida_a_linea_diagonal(self):
        tree, _ = construir_arbol_corrientes([LineString([(0, 0), (10, 10)])])
        result = distancias_exactas_a_corrientes([Point(5, 0)], tree)
        self.assertAlmostEqual(float(result[0]), np.sqrt(12.5), places=7)

    def test_distancia_usa_interior_del_segmento_no_solo_extremos(self):
        tree, _ = construir_arbol_corrientes([LineString([(0, 0), (10, 0)])])
        result = distancias_exactas_a_corrientes([Point(5, 2)], tree)
        self.assertAlmostEqual(float(result[0]), 2.0)
        self.assertNotAlmostEqual(float(result[0]), np.sqrt(29.0))

    def test_selecciona_la_corriente_mas_cercana(self):
        tree, _ = construir_arbol_corrientes(
            [LineString([(0, 0), (10, 0)]), LineString([(0, 10), (10, 10)])]
        )
        result = distancias_exactas_a_corrientes([Point(1, 7)], tree)
        self.assertAlmostEqual(float(result[0]), 3.0)

    def test_corriente_exterior_al_municipio_influye_en_distancia(self):
        # La línea queda completamente fuera del polígono municipal de prueba.
        municipal_bounds = (0.0, 0.0, 10.0, 10.0)
        regional_stream = LineString([(-5, 5), (-1, 5)])
        self.assertFalse(regional_stream.intersects(__import__("shapely.geometry").geometry.box(*municipal_bounds)))
        tree, _ = construir_arbol_corrientes([regional_stream])
        self.assertAlmostEqual(float(distancias_exactas_a_corrientes([Point(0, 5)], tree)[0]), 1.0)

    def test_mascara_nodata_se_conserva_en_bloque(self):
        transform = Affine.translation(0, 20) * Affine.scale(10, -10)
        mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
        tree, _ = construir_arbol_corrientes([LineString([(0, 5), (30, 5)])])
        result = calcular_bloque_distancias(transform, 0, mask, tree)
        np.testing.assert_array_equal(result[~mask], NODATA_OUTPUT)
        self.assertTrue(np.isfinite(result[mask]).all())
        self.assertTrue((result[mask] >= 0).all())

    def test_filtro_excluye_canales_virtuales_desconocidos_e_incluye_enabled_cero(self):
        regional = gpd.GeoDataFrame(
            {
                "_dcag_hydro_category": [
                    "corriente_perenne", "corriente_intermitente", "canal",
                    "linea_flujo_virtual", "desconocida",
                ],
                "_dcag_enabled_status": ["deshabilitada", "habilitada", "habilitada", "habilitada", "habilitada"],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 1), (1, 1)]),
                LineString([(0, 2), (1, 2)]),
                LineString([(0, 3), (1, 3)]),
                LineString([(0, 4), (1, 4)]),
            ],
            crs="EPSG:6368",
        )
        selected, report = filtrar_corrientes_naturales(regional)
        self.assertEqual(selected["_dcag_hydro_category"].tolist(), ["corriente_perenne", "corriente_intermitente"])
        self.assertEqual(selected["_dcag_enabled_status"].iloc[0], "deshabilitada")
        self.assertFalse(report["filtro_ENABLED_aplicado"])
        self.assertEqual(report["segmentos_incluidos"], 2)
        self.assertEqual(report["segmentos_excluidos"], 3)

    def test_equivalencia_entre_bloques_y_calculo_directo(self):
        transform = Affine.translation(0, 40) * Affine.scale(10, -10)
        valid = np.array(
            [[True, True, False, True], [True, True, True, True], [False, True, True, True]],
            dtype=bool,
        )
        lines = [LineString([(0, 10), (40, 30)]), LineString([(20, -5), (20, 50)])]
        tree, _ = construir_arbol_corrientes(lines)
        by_one_row = calcular_malla_por_bloques(transform, valid, tree, block_rows=1)
        by_two_rows = calcular_malla_por_bloques(transform, valid, tree, block_rows=2)

        direct = np.full(valid.shape, NODATA_OUTPUT, dtype=np.float32)
        for row, col in zip(*np.nonzero(valid)):
            x, y = transform * (col + 0.5, row + 0.5)
            direct[row, col] = distancias_exactas_a_corrientes([Point(x, y)], tree)[0]
        np.testing.assert_array_equal(by_one_row, by_two_rows)
        np.testing.assert_allclose(by_one_row[valid], direct[valid], rtol=0, atol=0)
        np.testing.assert_array_equal(by_one_row[~valid], NODATA_OUTPUT)


if __name__ == "__main__":
    unittest.main()
