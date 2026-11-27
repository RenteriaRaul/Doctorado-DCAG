"""Focused tests for Serie VII code/description handling and rasterization."""
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from scripts.land_cover_processing import (
    _land_cover_statistics,
    build_category_catalog,
    build_description_lookup,
    rasterize_land_cover,
)
from scripts.edaphology_processing import NODATA, assert_new_output_dir


def record(fid, geom, clave, descripcion):
    return {"feature_id": str(fid), "geometry": geom, "properties": {"CLAVE": clave, "DESCRIPCIO": descripcion}, "repaired": False}


class TestLandCoverProcessing(unittest.TestCase):
    def test_catalog_preserves_original_codes_descriptions_and_stable_ids(self):
        rows=[record(1,box(0,0,1,1),"Agricultura de riego","Agricultura"),record(2,box(1,0,2,1),"Bosque","Bosque de pino")]
        mapping,catalog=build_category_catalog(rows)
        again,_=build_category_catalog(list(reversed(rows)))
        self.assertEqual(mapping,again)
        self.assertEqual({r["CLAVE"]:r["DESCRIPCIO"] for r in catalog},{"Agricultura de riego":"Agricultura","Bosque":"Bosque de pino"})
        self.assertEqual([r["raster_code"] for r in catalog],sorted(r["raster_code"] for r in catalog))

    def test_conflicting_description_for_same_code_is_rejected(self):
        rows=[record(1,box(0,0,1,1),"A","Bosque"),record(2,box(1,0,2,1),"A","Selva")]
        with self.assertRaisesRegex(ValueError,"más de una DESCRIPCIO"):
            build_description_lookup(rows)

    def test_rasterization_keeps_categories_and_nodata_without_interpolation(self):
        rows=[record(1,box(0,1,1,2),"A","Bosque"),record(2,box(1,1,2,2),"B","Agricultura")]
        mask=np.array([[True,True],[True,False]])
        arr,mapping=rasterize_land_cover(rows,{"transform":Affine(1,0,0,0,-1,2)},mask)
        self.assertEqual(arr[0,0],mapping[("str","A")])
        self.assertEqual(arr[0,1],mapping[("str","B")])
        self.assertEqual(arr[1,1],NODATA)

    def test_vector_and_raster_surface_statistics_are_separate(self):
        rows=[record(1,box(0,0,1,1),"A","Bosque")]
        mask=np.ones((1,1),dtype=bool)
        arr=np.array([[1]],dtype=np.uint16)
        _,catalog=build_category_catalog(rows)
        stats=_land_cover_statistics(rows,arr,mask,225.0,catalog)
        self.assertEqual(stats[0]["CLAVE"],"A")
        self.assertEqual(stats[0]["DESCRIPCIO"],"Bosque")
        self.assertEqual(stats[0]["inegi_category_name"],"Bosque")
        self.assertAlmostEqual(stats[0]["vector_area_km2"],0.000001)
        self.assertAlmostEqual(stats[0]["raster_area_km2"],0.000225)

    def test_new_run_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"run"
            assert_new_output_dir(path)
            with self.assertRaises(FileExistsError):
                assert_new_output_dir(path)


if __name__=="__main__":
    unittest.main()
