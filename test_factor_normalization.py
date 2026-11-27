import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from scripts.factor_normalization import (
    CAT_NODATA, CELLS, LAND_GROUPS, land_cover_mapping, linear_decreasing,
    log_increasing, raster_stats, reclassify_categories, texture_mappings,
    write_float_raster,
)


class TestFactorNormalization(unittest.TestCase):
    def test_linear_bounds_monotonicity_and_clipping(self):
        x = np.array([-1., 0., 2., 5., 6.]); valid = np.ones(5, bool)
        y = linear_decreasing(x, valid, 0, 5)
        np.testing.assert_allclose(y, [1, 1, .6, 0, 0])
        self.assertTrue(np.all(np.diff(y) <= 0))

    def test_log_bounds_monotonicity(self):
        x = np.array([1., 2., 10., 100.]); y = log_increasing(x, np.ones(4,bool), 1, 100)
        self.assertAlmostEqual(float(y[0]), 0); self.assertAlmostEqual(float(y[-1]), 1)
        self.assertTrue(np.all(np.diff(y) >= 0))

    def test_invalid_bounds_and_negative_log_data(self):
        for a,b in ((-1,2),(1,1),(2,1),(float("nan"),3)):
            with self.assertRaises(ValueError): linear_decreasing(np.array([1]),np.array([True]),a,b)
        with self.assertRaises(ValueError): log_increasing(np.array([-1]),np.array([True]),0,2)

    def test_nodata_preservation_and_unknown_codes(self):
        src=np.array([[1,2,CAT_NODATA]])
        out=reclassify_categories(src,{1:.25,2:.75})
        self.assertTrue(np.isnan(out[0,2])); self.assertEqual(out[0,0],.25)
        with self.assertRaises(ValueError): reclassify_categories(np.array([9]),{1:.5})

    def test_variable_specific_texture_catalog_and_uncertainty_configs(self):
        df=pd.DataFrame([
            {"variable":"textura_superficial","raster_code":1,"source_value":"1","category_kind":"categoria_documentada"},
            {"variable":"textura_superficial","raster_code":2,"source_value":"2","category_kind":"categoria_documentada"},
            {"variable":"textura_superficial","raster_code":3,"source_value":"3","category_kind":"categoria_documentada"},
            {"variable":"textura_superficial","raster_code":4,"source_value":"N/A","category_kind":"fuera_del_dominio_documentado"},
            {"variable":"fase_fisica","raster_code":1,"source_value":"other","category_kind":"categoria_documentada"},
        ])
        m,u=texture_mappings(df[df.variable=="textura_superficial"])
        self.assertEqual(m,{1:.25,2:.5,3:.75}); self.assertEqual(u,{4})
        base=np.array([1,2,3,4]); variants=[]
        for fill in (.25,.5,.75):
            variants.append(reclassify_categories(base,{**m,4:fill}))
        np.testing.assert_array_equal(np.array(variants)[:,:3],np.array([[.25,.5,.75]]*3))
        self.assertEqual([v[-1] for v in variants],[.25,.5,.75])

    def test_case_sensitive_landcover_groups(self):
        cats=["VSa/BQ","VSA/SBC"]
        frame=pd.DataFrame([{"raster_code":i+1,"CLAVE":c,"DESCRIPCIO":"x"} for i,c in enumerate(cats)])
        # Test the case-sensitive assignment directly; full-catalog count is checked by integration.
        self.assertEqual(LAND_GROUPS["VSa/BQ"],("vegetacion_secundaria_arbustiva",.4))
        self.assertEqual(LAND_GROUPS["VSA/SBC"],("vegetacion_secundaria_arborea",.3))
        mapping, rows=land_cover_mapping(pd.DataFrame([{"raster_code":i+1,"CLAVE":c,"DESCRIPCIO":"desc"} for i,c in enumerate(LAND_GROUPS)]))
        self.assertEqual(len(mapping),21); self.assertEqual(len(rows),21)

    def test_stats_saturation_and_intermediate(self):
        s=raster_stats(np.array([0.,0.,.5,1.],dtype=np.float32))
        self.assertEqual(s["ceros"],2); self.assertEqual(s["unos"],1)
        self.assertAlmostEqual(s["intermedios_pct"],25)

    def test_grid_mask_and_no_overwrite(self):
        # A synthetic 1×1 grid exercises IO mechanics without municipality-scale allocations.
        with tempfile.TemporaryDirectory() as td:
            ref_path=Path(td)/"ref.tif"; out=Path(td)/"out.tif"
            profile={"driver":"GTiff","width":1,"height":1,"count":1,"dtype":"float32","crs":"EPSG:6368","transform":from_origin(0,15,15,15),"nodata":-9999}
            with rasterio.open(ref_path,"w",**profile) as ds: ds.write(np.array([[1]],dtype=np.float32),1)
            with rasterio.open(ref_path) as ref:
                write_float_raster(out,np.array([[.5]],dtype=np.float32),ref,np.array([[True]]),expected_cells=1)
                with rasterio.open(out) as result:
                    self.assertEqual(result.transform,ref.transform)
                    self.assertEqual(result.crs,ref.crs)
                    self.assertEqual(result.nodata,-9999)
                    self.assertEqual(result.read(1)[0,0],.5)
                with self.assertRaises(FileExistsError): write_float_raster(out,np.array([[.5]],dtype=np.float32),ref,np.array([[True]]))


if __name__ == "__main__": unittest.main()
