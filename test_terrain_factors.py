"""Pruebas sintéticas para la función Horn de terrain_factors."""

import unittest

import numpy as np

from scripts.terrain_factors import calcular_pendiente_horn


class TestPendienteHorn(unittest.TestCase):
    def setUp(self):
        self.dx = 15.0
        self.dy = 15.0
        self.x = (np.arange(9) - 4) * self.dx
        self.y = (np.arange(9) - 4) * self.dy
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def test_superficie_plana_es_cero(self):
        z = np.full((9, 9), 412.5, dtype=np.float32)
        grados, porcentaje = calcular_pendiente_horn(z, self.dx, self.dy)
        np.testing.assert_allclose(grados[1:-1, 1:-1], 0.0, atol=1e-7)
        np.testing.assert_allclose(porcentaje[1:-1, 1:-1], 0.0, atol=1e-7)
        self.assertTrue(np.isnan(grados[0, :]).all())

    def test_plano_con_pendiente_en_x_conocida(self):
        a = 0.2
        z = 700.0 + a * self.xx
        grados, porcentaje = calcular_pendiente_horn(z, self.dx, self.dy)
        expected_degrees = np.degrees(np.arctan(abs(a)))
        np.testing.assert_allclose(grados[1:-1, 1:-1], expected_degrees, atol=1e-5)
        np.testing.assert_allclose(porcentaje[1:-1, 1:-1], 100 * abs(a), atol=1e-5)

    def test_plano_diagonal_combina_derivadas(self):
        a, b = 0.12, -0.16
        z = 100.0 + a * self.xx + b * self.yy
        grados, porcentaje = calcular_pendiente_horn(z, self.dx, self.dy)
        gradient = np.hypot(a, b)
        np.testing.assert_allclose(grados[1:-1, 1:-1], np.degrees(np.arctan(gradient)), atol=1e-5)
        np.testing.assert_allclose(porcentaje[1:-1, 1:-1], 100 * gradient, atol=1e-5)

    def test_nodata_no_se_interpreta_como_cero(self):
        a = 0.1
        z = (500.0 + a * self.xx).astype(np.float32)
        z[4, 4] = -32768.0
        grados, porcentaje = calcular_pendiente_horn(z, self.dx, self.dy, nodata=-32768.0)
        # El centro y los ocho centros cuya vecindad 3x3 lo contiene son inválidos.
        self.assertTrue(np.isnan(grados[3:6, 3:6]).all())
        self.assertTrue(np.isnan(porcentaje[3:6, 3:6]).all())
        # Fuera de la vecindad NoData, la pendiente recupera el plano esperado.
        expected = np.degrees(np.arctan(abs(a)))
        np.testing.assert_allclose(grados[1:3, 1:3], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
