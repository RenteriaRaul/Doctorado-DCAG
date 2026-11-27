import unittest
import tempfile
from pathlib import Path
import geopandas as gpd
import pyogrio
from shapely.geometry import box, Point
from scripts.edaphology_audit import attribute_stats, coverage_pct, geometry_metrics

class EdaphologyAuditTest(unittest.TestCase):
    def test_attribute_stats_counts_null_unique_and_numeric(self):
        s,c=attribute_stats([1,2,None,2])
        self.assertEqual(s['records'],4); self.assertEqual(s['nulls'],1); self.assertEqual(s['unique'],2)
        self.assertEqual(c['2'],2); self.assertEqual(s['min'],1.0); self.assertEqual(s['max'],2.0)
    def test_coverage_union_not_sum(self):
        result=coverage_pct([box(0,0,6,10),box(4,0,10,10)],box(0,0,10,10))
        self.assertAlmostEqual(result['coverage_pct'],100.0)
    def test_geometry_quality(self):
        result=geometry_metrics([Point(0,0),None])
        self.assertEqual(result['features'],2); self.assertEqual(result['null_or_empty'],1); self.assertEqual(result['geometry_types'],{'Point':1})

    def test_synthetic_vector_layer_read_statistics_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.gpkg'
            frame=gpd.GeoDataFrame({'class':['A','B'],'value':[2.0,4.0]},geometry=[box(0,0,5,10),box(5,0,10,10)],crs='EPSG:6368')
            frame.to_file(path,layer='area',driver='GPKG')
            info=pyogrio.read_info(path,layer='area')
            read=pyogrio.read_dataframe(path,layer='area')
            self.assertEqual(info['features'],2)
            stats,_=attribute_stats(read['value'].tolist())
            self.assertEqual(stats['records'],2); self.assertEqual(stats['mean'],3.0)
            self.assertAlmostEqual(coverage_pct(read.geometry.tolist(),box(0,0,10,10))['coverage_pct'],100.0)

if __name__=='__main__': unittest.main()

