"""Synthetic tests for the edaphology categorical processing module."""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Point, Polygon, box

from scripts.edaphology_processing import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    NODATA,
    _surface_statistics,
    assert_new_output_dir,
    category_kind,
    category_kind_for_field,
    find_overlaps,
    inegi_category_name,
    prepare_polygon_records,
    rasterize_categories,
    stable_category_mapping,
    validate_conditional_na,
    validate_reference_grid,
)


class TestEdaphologyProcessing(unittest.TestCase):
    def test_exact_reference_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"grid.tif"
            with rasterio.open(path,"w",driver="GTiff",width=EXPECTED_WIDTH,height=EXPECTED_HEIGHT,count=1,dtype="uint8",crs="EPSG:6368",transform=Affine(15,0,600000,0,-15,2200000),nodata=0) as dst:
                dst.write(np.ones((EXPECTED_HEIGHT,EXPECTED_WIDTH),dtype="uint8"),1)
            with rasterio.open(path) as ds:
                self.assertEqual(validate_reference_grid(ds),ds.transform)

    def test_original_categories_and_special_values_are_retained(self):
        vals=["N","N/A","N/D",None,1]
        mapping=stable_category_mapping(vals)
        self.assertEqual(len(mapping),4)
        self.assertEqual(category_kind("N"),"ausencia_indicada")
        self.assertEqual(category_kind("N/A"),"no_aplicable")
        self.assertEqual(category_kind("N/D"),"no_disponible_o_desconocido")
        self.assertEqual(category_kind(None),"nulo_real")
        self.assertIn(("int",1),mapping)
        self.assertEqual(inegi_category_name("Grupo1","LP"),"Leptosol")
        self.assertEqual(inegi_category_name("Clase_tex",2),"Media")
        self.assertEqual(inegi_category_name("Grupo2","N"),"Ninguno")
        self.assertEqual(inegi_category_name("Fase_qui_u","N/A"),"Código fuente fuera del dominio documentado; valor preservado")

    def test_official_field_specific_domains_for_texture_and_phases(self):
        for code,label in [("1","Gruesa"),("2","Media"),("3","Fina")]:
            self.assertEqual(inegi_category_name("Clase_tex",code),label)
            self.assertEqual(category_kind_for_field("Clase_tex",code),"categoria_original")
        self.assertEqual(category_kind_for_field("Clase_tex","N/D"),"no_disponible_o_desconocido")
        self.assertEqual(category_kind_for_field("Clase_tex","N/A"),"fuera_del_dominio_documentado")
        self.assertEqual(category_kind_for_field("Fase_qui_u","N/D"),"no_disponible_o_desconocido")
        self.assertEqual(category_kind_for_field("Fase_qui_u","N/A"),"fuera_del_dominio_documentado")
        self.assertEqual(category_kind_for_field("Lmte_sup","N/A"),"no_aplicable")
        self.assertEqual(category_kind_for_field("Fase_fis_u","N/A"),"no_aplicable")
        self.assertEqual(inegi_category_name("Clase_tex","N/A"),"Código fuente fuera del dominio documentado; valor preservado")

    def test_conditional_na_for_surface_limit_and_physical_phase_requires_technosol(self):
        valid=[{"feature_id":"t1","properties":{"Grupo1":"TC","Lmte_sup":"N/A","Fase_fis_u":"N/A"}}]
        invalid=[{"feature_id":"x1","properties":{"Grupo1":"LP","Lmte_sup":"N/A","Fase_fis_u":"N/A"}}]
        self.assertEqual(validate_conditional_na(valid),[])
        self.assertEqual(len(validate_conditional_na(invalid)),2)

    def test_rasterization_preserves_codes_and_nodata(self):
        transform=Affine(1,0,0,0,-1,2)
        ref={"transform":transform}
        mask=np.array([[True,True,True],[True,True,False]])
        records=[{"feature_id":"1","geometry":box(0,1,1,2),"properties":{"soil":"N"}},
                 {"feature_id":"2","geometry":box(1,1,2,2),"properties":{"soil":"A"}},
                 {"feature_id":"3","geometry":box(2,1,3,2),"properties":{"soil":"N/A"}},
                 {"feature_id":"4","geometry":box(0,0,1,1),"properties":{"soil":None}},
                 {"feature_id":"5","geometry":box(1,0,2,1),"properties":{"soil":"B"}},
                 {"feature_id":"6","geometry":box(2,0,3,1),"properties":{"soil":"C"}}]
        mapping=stable_category_mapping(["N","A","N/A",None,"B","C"])
        array,_=rasterize_categories(records,"soil",ref,mask,mapping)
        self.assertEqual(array[0,0],mapping[("str","N")])
        self.assertEqual(array[0,1],mapping[("str","A")])
        self.assertEqual(array[0,2],mapping[("str","N/A")])
        self.assertEqual(array[1,0],NODATA)  # null inside the municipal mask
        self.assertEqual(array[1,2],NODATA)  # outside the municipal mask

    def test_invalid_geometry_is_detected_and_repaired_in_memory_with_area_log(self):
        bowtie=Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])
        frame=gpd.GeoDataFrame({"soil":["A"]},geometry=[bowtie],crs="EPSG:6368")
        records,repairs=prepare_polygon_records(frame,box(-1,-1,3,3),["soil"])
        self.assertEqual(len(repairs),1)
        self.assertIn("area_before_m2_source_crs",repairs[0])
        self.assertTrue(records[0]["geometry"].is_valid)

    def test_contradictory_polygon_overlaps_are_detected(self):
        records=[{"feature_id":"1","geometry":box(0,0,2,2),"properties":{"soil":"A"}},
                 {"feature_id":"2","geometry":box(1,0,3,2),"properties":{"soil":"B"}}]
        overlaps=find_overlaps(records,["soil"])
        self.assertEqual(len(overlaps),1)
        self.assertTrue(overlaps[0]["contradictory_values"])
        self.assertEqual(overlaps[0]["overlap_area_m2"],2.0)

    def test_surface_statistics_separate_vector_and_raster_area(self):
        records=[{"feature_id":"1","geometry":box(0,0,1,1),"properties":{"soil":"A"}}]
        mask=np.ones((1,1),dtype=bool)
        array=np.array([[1]],dtype=np.uint16)
        stats=_surface_statistics(records,"soil",array,mask,225.0)
        self.assertEqual(stats[0]["polygon_count"],1)
        self.assertEqual(stats[0]["vector_area_km2"],0.000001)
        self.assertEqual(stats[0]["raster_area_km2"],0.000225)

    def test_rasterization_is_deterministic(self):
        recs=[{"feature_id":"b","geometry":box(1,0,2,2),"properties":{"soil":"B"}},
              {"feature_id":"a","geometry":box(0,0,1,2),"properties":{"soil":"A"}}]
        mask=np.ones((2,2),dtype=bool); ref={"transform":Affine(1,0,0,0,-1,2)}
        x,_=rasterize_categories(recs,"soil",ref,mask)
        y,_=rasterize_categories(list(reversed(recs)),"soil",ref,mask)
        np.testing.assert_array_equal(x,y)

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"run"
            assert_new_output_dir(path)
            with self.assertRaises(FileExistsError):
                assert_new_output_dir(path)

    def test_point_geometry_is_not_interpolated_or_rasterized(self):
        with self.assertRaisesRegex(TypeError,"requiere polígonos"):
            rasterize_categories([{"feature_id":"1","geometry":Point(0.5,0.5),"properties":{"soil":"A"}}],"soil",{"transform":Affine(1,0,0,0,-1,1)},np.ones((1,1),dtype=bool))

    def test_synthetic_vector_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"area.gpkg"
            gpd.GeoDataFrame({"Grupo1":["A","B"]},geometry=[box(0,0,1,1),box(1,0,2,1)],crs=CRS.from_epsg(6368)).to_file(path,layer="area",driver="GPKG")
            frame=gpd.read_file(path,layer="area")
            self.assertEqual(len(frame),2)
            self.assertEqual(set(frame.Grupo1),{"A","B"})


if __name__=="__main__":
    unittest.main()
