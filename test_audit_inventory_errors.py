"""Pruebas sintéticas para direccionamiento de lectores y clasificación de auditoría."""
import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

try:
    from scripts.audit_inventory_errors import (
        AuditConfig, AuditError, _classify, _original_fields, _route_and_inspect, _sha256, audit_record,
        extract_error_records, run_audit,
    )
except ModuleNotFoundError:
    from audit_inventory_errors import (
        AuditConfig, AuditError, _classify, _original_fields, _route_and_inspect, _sha256, audit_record,
        extract_error_records, run_audit,
    )


class AuditInventoryErrorTests(unittest.TestCase):
    def test_extract_exact_errors(self):
        rows=[{"ruta":"x.json","error":"parse"},{"ruta":"b.tif","error":"driver"}]
        self.assertEqual(extract_error_records({"errores_lectura":rows}),rows)
        with self.assertRaises(AuditError): extract_error_records({"archivos":[]})
        self.assertEqual(_original_fields({"archivo":{"ruta":"nested.tif"},"mensaje":"x"})[0],"nested.tif")

    def test_json_and_provenance_sidecar_use_json_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ("report.json","dem.tif.provenance.json"):
                path=root/name;path.write_text('{"schema":1,"inputs":[]}',encoding="utf-8")
                result=_route_and_inspect(path)
                self.assertEqual(result["lector_asignado"],"json")
                self.assertTrue(result["lectura_correcta"])
                self.assertIn(result["categoria_formato"],{"json_metadatos_o_reporte","sidecar_procedencia"})

    def test_invalid_json_is_not_spatially_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.json";path.write_text("not-json")
            result=_route_and_inspect(path)
            self.assertEqual(result["lector_asignado"],"json")
            self.assertFalse(result["lectura_correcta"])
            self.assertEqual(result["categoria_formato"],"json_invalido")

    def test_csv_uses_table_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"table.csv";path.write_text("x,y\n1,2\n")
            result=_route_and_inspect(path)
            self.assertEqual(result["lector_asignado"],"tabular")
            self.assertEqual(result["metadatos_archivo"]["columnas"],["x","y"])

    def test_geotiff_uses_rasterio_and_reads_only_metadata_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"grid.tif"
            with rasterio.open(path,"w",driver="GTiff",width=3,height=2,count=1,dtype="float32",crs="EPSG:6368",transform=from_origin(0,30,15,15),nodata=-9999) as dst:
                dst.write(np.ones((2,3),dtype=np.float32),1)
            result=_route_and_inspect(path)
            self.assertEqual(result["lector_asignado"],"rasterio")
            self.assertTrue(result["lectura_correcta"])
            self.assertEqual(result["metadatos_archivo"]["width"],3)

    def test_valid_and_incomplete_shapefiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);valid=root/"valid.shp"
            frame=gpd.GeoDataFrame({"code":[1]},geometry=[LineString([(0,0),(1,1)])],crs="EPSG:6368")
            frame.to_file(valid,driver="ESRI Shapefile")
            result=_route_and_inspect(valid)
            self.assertEqual(result["lector_asignado"],"geopandas_shapefile")
            self.assertTrue(result["lectura_correcta"])
            incomplete=root/"incomplete.shp";incomplete.write_bytes(valid.read_bytes())
            result=_route_and_inspect(incomplete)
            self.assertEqual(result["categoria_formato"],"shapefile_incompleto")
            self.assertIn("incomplete.shx",result["componentes_faltantes"])

    def test_geopackage_uses_geopandas_and_lists_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.gpkg"
            gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,1,1)],crs="EPSG:6368").to_file(path,layer="boundary",driver="GPKG")
            result=_route_and_inspect(path)
            self.assertEqual(result["lector_asignado"],"geopandas_vector")
            self.assertTrue(result["lectura_correcta"])
            self.assertEqual(result["metadatos_archivo"]["capas"],["boundary"])

    def test_missing_unknown_and_permission_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=audit_record({"ruta":"missing.tif","error":"open failed"},Path(tmp)/"report.json",Path(tmp))
            self.assertFalse(result["existe"])
            self.assertEqual(result["categoria_error"],"E_error_de_acceso")
            unknown=_route_and_inspect(Path(tmp)/"a.zzz")
            self.assertEqual(unknown["categoria_formato"],"formato_desconocido")
        cls=_classify({}, {"extension":".tif","lectura_correcta":False,"categoria_formato":"acceso_denegado"},"",access_error="PermissionError: denied")
        self.assertEqual(cls["clasificacion"],"E_error_de_acceso")

    def test_import_error_is_implementation_issue(self):
        result=_classify({}, {"extension":".gpkg","lectura_correcta":True,"error_lector":None},"ModuleNotFoundError: No module named 'scripts'")
        self.assertEqual(result["clasificacion"],"F_error_de_implementacion")
        self.assertTrue(result["recuperable"])

    def test_report_audit_only_requested_records_and_protects_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/"metadata.json";src.write_text('{"x":1}')
            inventory=root/"inventory.json"
            digest=_sha256(src)
            inventory.write_text(json.dumps({"errores_lectura":[{"ruta":str(src),"error":"geospatial reader failed"}],"archivos":[{"ruta":str(src),"sha256":digest}]}))
            cfg=AuditConfig(inventory,root/"out",root,expected_error_count=1)
            with self.assertRaises(AuditError): run_audit(AuditConfig(inventory,root/"out",root,expected_error_count=2))
            result=run_audit(cfg)
            self.assertEqual(result["errores_auditados"],1)
            self.assertEqual(result["resultados"][0]["categoria_error"],"A_falso_positivo_por_formato")
            self.assertEqual(result["resultados"][0]["hash_esperado"],digest)
            self.assertTrue(Path(result["rutas_productos"]["json"]).is_file())


if __name__=="__main__":
    unittest.main(verbosity=2)
