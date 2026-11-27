import unittest
import numpy as np
from pathlib import Path
import tempfile
from scripts.factor_redundancy import (
    descriptive_statistics,correlations,vif_tolerance,categorical_association,
    eta_squared,systematic_indices,create_run_dir,log1p_valid,vif_plot_matrix,read_stack,refresh_selection_table
)

class TestFactorRedundancy(unittest.TestCase):
    def test_perfect_and_inverse_correlations(self):
        x=np.arange(1,101,dtype=float); arrays={"x":x,"same":x.copy(),"inverse":-x}
        masks={n:np.ones(x.shape,bool) for n in arrays}
        p,s,_=correlations(arrays,masks,list(arrays))
        self.assertAlmostEqual(p.loc["x","same"],1.0)
        self.assertAlmostEqual(p.loc["x","inverse"],-1.0)
        self.assertAlmostEqual(s.loc["x","inverse"],-1.0)

    def test_constant_and_perfect_multicollinearity(self):
        x=np.arange(1,30,dtype=float); arrays={"x":x,"same":x.copy(),"constant":np.ones(x.shape)}
        v=vif_tolerance(arrays,list(arrays),np.ones(x.shape,bool)).set_index("factor")
        self.assertEqual(v.loc["constant","estado"],"predictor_constante")
        self.assertEqual(v.loc["x","estado"],"colinealidad_perfecta")
        self.assertTrue(np.isinf(v.loc["same","vif"]))

    def test_categorical_independence_association_constant_and_nodata(self):
        a=np.repeat([1,1,2,2],50); b=np.tile(np.repeat([1,2],50),2)
        independent,_=categorical_association(a,b)
        associated,_=categorical_association(a,a)
        self.assertLess(independent["cramers_v"],.1)
        self.assertGreater(associated["cramers_v"],.99)
        c,_=categorical_association(np.ones(12,int),np.arange(12)%3)
        self.assertEqual(c["estado"],"variable_constante")
        missing,_=categorical_association(np.array([1,2,65535]),np.array([1,2,1]))
        self.assertEqual(missing["n"],2)

    def test_eta_squared_mixed_and_descriptive_nodata(self):
        cats=np.repeat([1,2],20); x=np.r_[np.ones(20),np.full(20,5.)]
        self.assertAlmostEqual(eta_squared(x,cats)["eta_squared"],1)
        d=descriptive_statistics(np.array([0,0,1,2,np.nan]))
        self.assertEqual(d["n_validos"],4); self.assertEqual(d["ceros"],2)

    def test_systematic_sampling_repeatable_and_expected_steps(self):
        mask=np.ones((120,130),bool); mask[2:10,3:11]=False
        r1,c1,s1=systematic_indices(mask,250); r2,c2,s2=systematic_indices(mask,250)
        self.assertEqual(s1,17); self.assertEqual(s1,s2)
        np.testing.assert_array_equal(r1,r2); np.testing.assert_array_equal(c1,c2)

    def test_log1p_preserves_nodata_and_only_transforms_valid_cells(self):
        x=np.array([-32768.,0.,3.]); mask=np.array([False,True,True])
        y=log1p_valid(x,mask)
        self.assertEqual(y[0],-32768.)
        self.assertEqual(y[1],0.)
        self.assertAlmostEqual(y[2],np.log(4.))

    def test_run_directory_protects_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            create_run_dir(td,"run_test")
            with self.assertRaises(FileExistsError): create_run_dir(td,"run_test")

    def test_vif_plot_handles_omitted_predictors_in_scenarios(self):
        import pandas as pd
        tables={"full":pd.DataFrame([{"factor":"a","vif":2.0,"estado":"ok"}]),
                "reduced":pd.DataFrame([{"factor":"b","vif":np.inf,"estado":"colinealidad_perfecta"}])}
        z=vif_plot_matrix(tables,["a","b","c"])
        self.assertEqual(z[0,0],2.0); self.assertTrue(np.isnan(z[0,1])); self.assertEqual(z[1,1],30.0); self.assertTrue(np.isnan(z[1,2]))

    def test_grid_parameters_allow_other_municipality_dimensions(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"small.tif"
            with rasterio.open(p,"w",driver="GTiff",width=3,height=2,count=1,dtype="uint16",crs="EPSG:6368",transform=from_origin(10,20,15,15),nodata=65535) as ds:
                ds.write(np.ones((2,3),dtype=np.uint16),1)
            _,_,domain,common,grid,_=read_stack({"x":p},{},expected_shape=(2,3),expected_mask_cells=6)
            self.assertEqual(int(domain.sum()),6); self.assertTrue(np.array_equal(domain,common)); self.assertEqual(grid["dimensions"],[3,2])

    def test_selection_table_lists_non_diagonal_associations(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            pd.DataFrame([[1,.8],[.8,1]],index=["x","y"],columns=["x","y"]).to_csv(p/"correlacion_pearson.csv")
            pd.DataFrame([[1,.7],[.7,1]],index=["x","y"],columns=["x","y"]).to_csv(p/"correlacion_spearman.csv")
            pd.DataFrame([{"factor_a":"c","factor_b":"c","cramers_v":1,"nmi":1},{"factor_a":"c","factor_b":"d","cramers_v":.5,"nmi":.4}],columns=["factor_a","factor_b","cramers_v","nmi"]).to_csv(p/"resumen_asociacion_categorica.csv",index=False)
            pd.DataFrame([[.2,.3],[.1,.4]],index=["x","y"],columns=["c","d"]).to_csv(p/"asociacion_eta_cuadrado.csv")
            pd.DataFrame([{"factor":"x","vif":2,"tolerancia":.5,"estado":"ok","escenario":"A_originales"},{"factor":"y","vif":3,"tolerancia":.33,"estado":"ok","escenario":"A_originales"}]).to_csv(p/"vif_tolerancia.csv",index=False)
            self.assertEqual(refresh_selection_table(p),4)
            result=pd.read_csv(p/"matriz_seleccion_factores.csv")
            self.assertIn("rP=0.800",result.loc[result.factor=="x","asociaciones_principales"].iloc[0])
            self.assertIn("c: V=0.500",result.loc[result.factor=="d","asociaciones_principales"].iloc[0])

if __name__=="__main__": unittest.main()
