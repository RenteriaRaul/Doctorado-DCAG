"""Pruebas sintéticas del diagnóstico HAND 98.5; no acceden a productos reales."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from scripts.diagnose_hand_985 import (
    CONNECTIVITY_CODE,
    accumulation_distribution,
    classify_d8_paths,
    missing_distance_summary,
    _stream_distance,
    _potential_network_segments,
    classify_d8_memmap_global,
)


class HandDiagnosticTests(unittest.TestCase):
    def test_flow_reaches_stream(self):
        d8 = np.array([[2, 2, 0]], dtype=np.uint8)  # E, E, terminal stream
        labels = classify_d8_paths(d8, np.array([[0, 0, 1]]), np.ones_like(d8, bool))
        self.assertTrue(np.all(labels == CONNECTIVITY_CODE["alcanza_drenaje"]))

    def test_independent_d8_terminal(self):
        d8 = np.array([[0, 0]], dtype=np.uint8)
        labels = classify_d8_paths(d8, np.zeros_like(d8), np.ones_like(d8, bool))
        self.assertTrue(np.all(labels == CONNECTIVITY_CODE["termina_D8_0"]))

    def test_nodata_and_domain_exit(self):
        d8 = np.array([[2, 2, 255], [32, 0, 0]], dtype=np.uint8)
        valid = d8 != 255
        labels = classify_d8_paths(d8, np.zeros_like(d8), valid)
        self.assertEqual(labels[0, 1], CONNECTIVITY_CODE["llega_a_NoData"])
        self.assertEqual(labels[1, 0], CONNECTIVITY_CODE["sale_del_dominio"])

    def test_cycle_and_feeder(self):
        # Este -> Oeste: ciclo de dos celdas; la primera fila desemboca en él.
        d8 = np.array([[2, 32], [2, 32]], dtype=np.uint8)
        labels = classify_d8_paths(d8, np.zeros_like(d8), np.ones_like(d8, bool))
        self.assertTrue(np.all(labels == CONNECTIVITY_CODE["ciclo"]))

    def test_accumulation_threshold_bins_and_potential_coverage(self):
        values = np.array([1, 100, 249, 500, 992], dtype=float)
        streams = np.array([0, 1, 0, 0, 0])
        report = accumulation_distribution(values, streams, thresholds=(100, 250, 500, 1000))
        self.assertEqual(sum(row["celdas"] for row in report["intervalos"]), 5)
        at_100 = next(row for row in report["umbrales"] if row["umbral_celdas"] == 100)
        self.assertEqual(at_100["celdas_sobre_umbral"], 4)
        self.assertEqual(at_100["potencialmente_cubiertas_sin_modificar_red"], 3)

    def test_exact_distance_to_line_interior(self):
        distances = _stream_distance([(5, 3)], [LineString([(0, 0), (10, 0)])])
        self.assertEqual(distances[0], 3.0)

    def test_distance_summary_percentiles(self):
        result = missing_distance_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["celdas"], 5)
        self.assertEqual(result["mediana_m"], 3.0)
        self.assertEqual(result["percentiles_m"]["5"], 1.2)

    def test_global_memmap_and_chunk_sizes_match_reference(self):
        d8 = np.array([[2, 2, 0], [2, 2, 0]], dtype=np.uint8)
        drainage = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.uint8)
        valid = np.ones_like(d8, dtype=np.uint8)
        expected = classify_d8_paths(d8, drainage, valid.astype(bool))
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for batch in (1, 2, 5):
                labels = np.memmap(Path(tmp) / f"labels{batch}.bin", mode="w+", dtype=np.uint8, shape=(d8.size,))
                queue = np.memmap(Path(tmp) / f"queue{batch}.bin", mode="w+", dtype=np.uint32, shape=(d8.size,))
                classify_d8_memmap_global(d8.ravel(), drainage.ravel(), valid.ravel(), labels, queue, 2, 3, batch)
                results.append(np.asarray(labels).reshape(d8.shape).copy())
                labels.flush(); queue.flush()
                labels._mmap.close(); queue._mmap.close()
            self.assertTrue(np.array_equal(results[0], expected))
            self.assertTrue(all(np.array_equal(results[0], r) for r in results[1:]))

    def test_potential_raster_becomes_connected_linear_segments(self):
        from affine import Affine
        stream = np.array([[1, 1, 1]], dtype=np.uint8)
        d8 = np.array([[2, 2, 0]], dtype=np.uint8)
        segments = _potential_network_segments(stream.ravel(), d8.ravel(), 1, 3, Affine(15, 0, 0, 0, -15, 30))
        self.assertEqual(len(segments), 2)
        # El punto medio del primer segmento está sobre la geometría derivada.
        self.assertEqual(_stream_distance([(7.5, 22.5)], segments)[0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
