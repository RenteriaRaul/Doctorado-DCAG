"""Pruebas sintéticas para la clasificación y métricas de hydrography.py."""

import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

from scripts.hydrography import (
    calcular_densidad_drenaje,
    calcular_densidades_hidrograficas,
    clasificar_enabled,
    clasificar_hidrografia,
    identificar_subcuencas,
    longitud_geometrica_unica_km,
    longitud_total_entidades_km,
    recortar_lineas_municipales,
    resumir_combinaciones_inconsistentes,
    validar_inventario_hidrografico,
)


class TestHydrographySynthetic(unittest.TestCase):
    def setUp(self):
        self.municipio = box(0, 0, 1000, 1000)
        self.crs = "EPSG:6368"

    def test_identifica_subcuencas_por_interseccion_y_cobertura(self):
        subcuencas = gpd.GeoDataFrame(
            {"codigo": ["A", "B", "C"]},
            geometry=[box(-100, 0, 500, 1000), box(500, 0, 1000, 1000), box(1200, 0, 1400, 1000)],
            crs=self.crs,
        )
        all_rows, selected, coverage = identificar_subcuencas(subcuencas, self.municipio)
        self.assertEqual(len(all_rows), 3)
        self.assertEqual(set(selected["codigo"]), {"A", "B"})
        self.assertAlmostEqual(coverage["union_areas_interseccion_km2"], 1.0)
        self.assertAlmostEqual(coverage["superficie_municipal_sin_cobertura_km2"], 0.0)
        self.assertAlmostEqual(coverage["solapamiento_subcuencas_km2"], 0.0)

    def test_registra_solapamiento_de_subcuencas(self):
        subcuencas = gpd.GeoDataFrame(
            {"codigo": ["A", "B"]},
            geometry=[box(0, 0, 700, 1000), box(500, 0, 1000, 1000)],
            crs=self.crs,
        )
        _, selected, coverage = identificar_subcuencas(subcuencas, self.municipio)
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(coverage["suma_areas_interseccion_individual_km2"], 1.2)
        self.assertAlmostEqual(coverage["union_areas_interseccion_km2"], 1.0)
        self.assertAlmostEqual(coverage["solapamiento_subcuencas_km2"], 0.2)

    def test_recorte_lineal_preserva_atributos(self):
        lines = gpd.GeoDataFrame(
            {"NOMBRE": ["cauce-a"], "ENABLED": [1], "LENGTHM": [1400.0]},
            geometry=[LineString([(-200, 500), (1200, 500)])],
            crs=self.crs,
        )
        clipped = recortar_lineas_municipales(lines, self.municipio)
        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped.iloc[0]["NOMBRE"], "cauce-a")
        self.assertEqual(int(clipped.iloc[0]["ENABLED"]), 1)
        self.assertEqual(float(clipped.iloc[0]["LENGTHM"]), 1400.0)
        self.assertAlmostEqual(clipped.geometry.iloc[0].length, 1000.0)
        self.assertAlmostEqual(clipped.iloc[0]["_dcag_length_municipal_km"], 1.0)

    def test_longitudes_km_y_densidad_basica(self):
        lines = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1500, 0)]), LineString([(0, 500), (500, 500)])],
            crs=self.crs,
        )
        total = longitud_total_entidades_km(lines)
        self.assertAlmostEqual(total, 2.0)
        self.assertAlmostEqual(calcular_densidad_drenaje(total, 4.0), 0.5)

    def test_clasifica_combinaciones_reales_inegi_y_rechaza_inconsistencias(self):
        self.assertEqual(
            clasificar_hidrografia(101, "CORRIENTE DE AGUA", 3271, "INTERMITENTE"),
            "corriente_intermitente",
        )
        self.assertEqual(
            clasificar_hidrografia("101", "Corriente de agua", "3272.0", "Perenne"),
            "corriente_perenne",
        )
        self.assertEqual(clasificar_hidrografia(102, "CANAL", 3180, "EN OPERACIÓN"), "canal")
        self.assertEqual(clasificar_hidrografia(102, "CANAL", 3181, "EN CONSTRUCCIÓN"), "canal")
        self.assertEqual(clasificar_hidrografia(102, "CANAL", 3182, "FUERA DE USO"), "canal")
        self.assertEqual(
            clasificar_hidrografia(103, "LINEA CENTRAL", 3273, "FLUJO VIRTUAL"),
            "linea_flujo_virtual",
        )
        self.assertEqual(
            clasificar_hidrografia(101, "CORRIENTE DE AGUA", 3272, "INTERMITENTE"),
            "desconocida",
        )
        self.assertEqual(
            clasificar_hidrografia(999, "CORRIENTE DE AGUA", 3271, "INTERMITENTE"),
            "desconocida",
        )
        self.assertEqual(
            clasificar_hidrografia(101, "CORRIENTE DE AGUA", 3271, "INTERMITENTE PERENNE"),
            "desconocida",
        )

    def test_combinaciones_desconocidas_se_reportan_por_valor(self):
        fields = {"tipo": "TIPO", "entidad": "ENTIDAD", "fc": "FC", "condicion": "CONDICION", "enabled": "ENABLED"}
        gdf = gpd.GeoDataFrame(
            {
                "TIPO": [101, 101, 999],
                "ENTIDAD": ["CORRIENTE DE AGUA"] * 3,
                "FC": [3272, 3272, 9999],
                "CONDICION": ["INTERMITENTE", "CONDICION NUEVA", "VALOR NUEVO"],
                "ENABLED": [1, 1, 0],
            },
            geometry=[LineString([(0, 0), (1, 0)])] * 3,
            crs=self.crs,
        )
        from scripts.hydrography import _classify_line_records

        classified = _classify_line_records(gdf, fields)
        summary = resumir_combinaciones_inconsistentes(classified, fields)
        self.assertEqual(summary["segmentos"], 3)
        self.assertEqual(len(summary["combinaciones"]), 3)
        self.assertEqual(classified["_dcag_hydro_category"].tolist(), ["desconocida"] * 3)

    def test_enabled_reconoce_dominios_numericos_y_equivalentes(self):
        for value in (0, 0.0, "0", "0.0", False, "false", "disabled"):
            self.assertEqual(clasificar_enabled(value), "deshabilitada")
        for value in (1, 1.0, "1", "1.0", True, "true", "enabled"):
            self.assertEqual(clasificar_enabled(value), "habilitada")
        self.assertEqual(clasificar_enabled(7), "desconocida")

    def test_densidades_excluyen_canales_virtuales_y_desconocidos(self):
        gdf = gpd.GeoDataFrame(
            {
                "_dcag_hydro_category": [
                    "corriente_perenne", "corriente_intermitente", "canal",
                    "linea_flujo_virtual", "desconocida",
                ],
                "_dcag_enabled_status": ["deshabilitada", "habilitada", "habilitada", "habilitada", "habilitada"],
            },
            geometry=[
                LineString([(0, 0), (1000, 0)]),
                LineString([(1000, 0), (2000, 0)]),
                LineString([(2000, 0), (3000, 0)]),
                LineString([(3000, 0), (4000, 0)]),
                LineString([(4000, 0), (5000, 0)]),
            ],
            crs=self.crs,
        )
        metrics = calcular_densidades_hidrograficas(gdf, area_km2=2.0)
        self.assertAlmostEqual(metrics["longitud_cartografica_km"], 2.0)
        self.assertAlmostEqual(metrics["densidad_cartografica_km_km2"], 1.0)
        self.assertAlmostEqual(metrics["longitud_habilitada_km"], 1.0)
        self.assertAlmostEqual(metrics["densidad_habilitada_km_km2"], 0.5)

    def test_union_all_no_sobrecuenta_duplicados_o_solapes_parciales(self):
        lines = gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (10000, 0)]),
                LineString([(0, 0), (10000, 0)]),
                LineString([(5000, 0), (15000, 0)]),
            ],
            crs=self.crs,
        )
        self.assertAlmostEqual(longitud_total_entidades_km(lines), 30.0)
        self.assertAlmostEqual(longitud_geometrica_unica_km(lines), 15.0)
        clipped = recortar_lineas_municipales(lines, box(-1, -1, 16000, 1))
        self.assertEqual(len(clipped), 3, "Las entidades de origen deben permanecer en la capa derivada.")
        self.assertAlmostEqual(longitud_total_entidades_km(clipped), 30.0)
        self.assertAlmostEqual(longitud_geometrica_unica_km(clipped), 15.0)

    def test_inventario_verifica_75_archivos_y_detecta_faltantes_duplicados(self):
        subbasins = [f"RH{i:02d}Aa" for i in range(1, 16)]
        discovered = {
            suffix: [Path(f"{basin}{suffix}.shp") for basin in subbasins]
            for suffix in ("_hl", "_ha", "_dr", "_subc", "_to")
        }
        report = validar_inventario_hidrografico(discovered)
        self.assertEqual(report["archivos_detectados"], 75)
        self.assertEqual(report["subcuencas_detectadas"], 15)

        missing = {key: list(paths) for key, paths in discovered.items()}
        missing["_to"].pop()
        with self.assertRaises(ValueError):
            validar_inventario_hidrografico(missing)

        duplicate = {key: list(paths) for key, paths in discovered.items()}
        duplicate["_hl"].append(Path("otra_copia/RH01Aa_hl.shp"))
        with self.assertRaises(ValueError):
            validar_inventario_hidrografico(duplicate)


if __name__ == "__main__":
    unittest.main()
