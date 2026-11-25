import tempfile, unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

class TestHand984Inputs(unittest.TestCase):
    def test_fixture_grid_and_mask_logic(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.tif'; arr=np.arange(25,dtype='float32').reshape(5,5); arr[2,2]=-9999
            with rasterio.open(p,'w',driver='GTiff',width=5,height=5,count=1,dtype='float32',crs='EPSG:6368',transform=from_origin(0,75,15,15),nodata=-9999) as d: d.write(arr,1)
            with rasterio.open(p) as d:
                a=d.read(1); mask=(d.read_masks(1)>0)&(a!=d.nodata)
                self.assertEqual(int(mask.sum()),24)
                self.assertFalse(mask[2,2])
                self.assertEqual(d.res,(15.0,15.0))
    def test_expected_product_names(self):
        from scripts.diagnose_hand_984 import PRODUCTS
        self.assertEqual(len(PRODUCTS),6)
        self.assertIn('hand',PRODUCTS)
        self.assertIn('d8',PRODUCTS)
if __name__=='__main__': unittest.main()
