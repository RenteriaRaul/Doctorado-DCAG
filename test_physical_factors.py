"""Pruebas aisladas de encaje por ventana entera y máscara municipal."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from scripts.physical_factors import aligned_window, prepare_masked_array


class TestPhysicalFactorAlignment(unittest.TestCase):
    def _dataset(self, transform, array, nodata=-9999.0):
        memory = MemoryFile()
        dataset = memory.open(
            driver="GTiff",
            width=array.shape[1],
            height=array.shape[0],
            count=1,
            dtype=array.dtype,
            crs="EPSG:6368",
            transform=transform,
            nodata=nodata,
        )
        dataset.write(array, 1)
        return memory, dataset

    def test_extrae_ventana_entera_sin_interpolar_y_aplica_nodata(self):
        source_array = np.arange(100, dtype=np.int16).reshape(10, 10)
        source_array[3, 3] = -9999
        source_transform = Affine(15, 0, 0, 0, -15, 150)
        reference_transform = Affine(15, 0, 30, 0, -15, 120)
        reference_array = np.ones((3, 3), dtype=np.int16)
        src_mem, src = self._dataset(source_transform, source_array)
        ref_mem, ref = self._dataset(reference_transform, reference_array)
        self.addCleanup(src.close)
        self.addCleanup(src_mem.close)
        self.addCleanup(ref.close)
        self.addCleanup(ref_mem.close)

        window = aligned_window(src, ref)
        self.assertEqual((window.col_off, window.row_off, window.width, window.height), (2, 2, 3, 3))
        result, valid, inside, _ = prepare_masked_array(src, ref, box(30, 60, 60, 120))

        self.assertEqual(result[0, 0], source_array[2, 2])
        self.assertEqual(result[1, 1], -9999)
        self.assertFalse(valid[1, 1])
        self.assertEqual(result[0, 2], -9999)
        self.assertFalse(inside[0, 2])
        np.testing.assert_array_equal(result[~valid], np.full((int((~valid).sum()),), -9999, dtype=np.int16))

    def test_rechaza_fase_de_pixel_no_entera(self):
        arr = np.ones((5, 5), dtype=np.int16)
        src_mem, src = self._dataset(Affine(15, 0, 0, 0, -15, 75), arr)
        ref_mem, ref = self._dataset(Affine(15, 0, 7.5, 0, -15, 60), arr[:2, :2])
        self.addCleanup(src.close)
        self.addCleanup(src_mem.close)
        self.addCleanup(ref.close)
        self.addCleanup(ref_mem.close)
        with self.assertRaisesRegex(ValueError, "fase de píxel entera"):
            aligned_window(src, ref)

    def test_rechaza_fuente_que_no_cubre_referencia(self):
        arr = np.ones((2, 2), dtype=np.int16)
        src_mem, src = self._dataset(Affine(15, 0, 0, 0, -15, 30), arr)
        ref_mem, ref = self._dataset(Affine(15, 0, 15, 0, -15, 15), arr)
        self.addCleanup(src.close)
        self.addCleanup(src_mem.close)
        self.addCleanup(ref.close)
        self.addCleanup(ref_mem.close)
        with self.assertRaisesRegex(ValueError, "no cubre"):
            aligned_window(src, ref)

    def test_rechaza_nodata_no_declarado(self):
        arr = np.ones((3, 3), dtype=np.int16)
        src_mem, src = self._dataset(Affine(15, 0, 0, 0, -15, 45), arr, nodata=None)
        ref_mem, ref = self._dataset(Affine(15, 0, 0, 0, -15, 45), arr, nodata=None)
        self.addCleanup(src.close)
        self.addCleanup(src_mem.close)
        self.addCleanup(ref.close)
        self.addCleanup(ref_mem.close)
        with self.assertRaisesRegex(ValueError, "no declara NoData"):
            prepare_masked_array(src, ref, box(0, 0, 45, 45))


if __name__ == "__main__":
    unittest.main()
