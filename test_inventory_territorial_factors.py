"""Pruebas sintéticas del inventario; no acceden al inventario territorial real."""
import tempfile
import unittest
from types import ModuleType
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

try:
    from scripts.inventory_territorial_factors import (
        create_run_directory, evaluate_spatial_quality, file_sha256, identify_factor,
        inspect_raster, inspect_vector, inspect_json_document, select_reader, InventoryConfig, run_inventory,
        _load_boundary, _boundary_summary_markdown,
    )
except ModuleNotFoundError:
    from inventory_territorial_factors import (
        create_run_directory, evaluate_spatial_quality, file_sha256, identify_factor,
        inspect_raster, inspect_vector, inspect_json_document, select_reader, InventoryConfig, run_inventory,
        _load_boundary, _boundary_summary_markdown,
    )


class InventoryTests(unittest.TestCase):
    def _fake_manager_import(self, manager_class):
        import builtins
        original_import=builtins.__import__
        module=ModuleType("scripts.gis_manager");module.GISManager=manager_class
        def hooked(name,globals=None,locals=None,fromlist=(),level=0):
            if name=="scripts.gis_manager": return module
            return original_import(name,globals,locals,fromlist,level)
        return mock.patch("builtins.__import__",side_effect=hooked)

    def _backup_fixture(self, root, frame):
        gpkg=root/"backup.gpkg";gpkg.touch()
        marco=root/"marco";marco.mkdir(exist_ok=True)
        config=InventoryConfig(search_roots=(),output_dir=root/"out",reference_grid=root/"missing.tif",
                               regional_dem=root/"regional.tif",boundary_gpkg=gpkg,
                               marco_geoestadistico_dir=marco)
        return config,frame

    def _assert_markdown_boundary_safe(self, info):
        text="\n".join(_boundary_summary_markdown(info))
        self.assertIn("Límite municipal disponible:",text)
        self.assertIn("Ruta declarada:",text)
        self.assertIn("Estado de GISManager:",text)
        return text

    def test_boundary_from_gis_manager_has_consistent_metadata(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            config,_=self._backup_fixture(root,frame)
            class Manager:
                def __init__(self,path): pass
                def obtener_municipio(self,**kwargs): return frame
            with self._fake_manager_import(Manager):
                result,info=_load_boundary(config)
            self.assertIsNotNone(result);self.assertTrue(info["disponible"])
            self.assertEqual(info["ruta"],str(config.marco_geoestadistico_dir))
            self.assertEqual(info["fuente"],"GISManager.obtener_municipio")
            self.assertEqual(info["entidades"],1)
            self._assert_markdown_boundary_safe(info)

    def test_missing_gis_manager_uses_geopackage_backup(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);backup=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            config,_=self._backup_fixture(root,backup);config.marco_geoestadistico_dir=root/"not-mounted"
            with mock.patch("geopandas.read_file",return_value=backup) as read:
                result,info=_load_boundary(config)
            self.assertIsNotNone(result);self.assertEqual(read.call_count,1)
            self.assertEqual(info["fuente"],"GeoPackage procesado")
            self.assertTrue(info["error_GISManager"])
            self._assert_markdown_boundary_safe(info)

    def test_gis_manager_exception_is_retained_when_backup_is_used(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);backup=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            config,_=self._backup_fixture(root,backup)
            class Manager:
                def __init__(self,path): pass
                def obtener_municipio(self,**kwargs): raise RuntimeError("Marco read failed")
            with self._fake_manager_import(Manager),mock.patch("geopandas.read_file",return_value=backup):
                result,info=_load_boundary(config)
            self.assertIsNotNone(result);self.assertIn("Marco read failed",info["error_GISManager"])
            self._assert_markdown_boundary_safe(info)

    def test_empty_gis_manager_result_is_recorded_then_falls_back(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);backup=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            empty=gpd.GeoDataFrame({"id":[]},geometry=[],crs="EPSG:6368");config,_=self._backup_fixture(root,backup)
            class Manager:
                def __init__(self,path): pass
                def obtener_municipio(self,**kwargs): return empty
            with self._fake_manager_import(Manager),mock.patch("geopandas.read_file",return_value=backup):
                result,info=_load_boundary(config)
            self.assertIsNotNone(result);self.assertIn("capa vacía",info["error_GISManager"])
            self._assert_markdown_boundary_safe(info)

    def test_gis_manager_result_without_crs_is_recorded_then_falls_back(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);backup=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            no_crs=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,10,10)]);config,_=self._backup_fixture(root,backup)
            class Manager:
                def __init__(self,path): pass
                def obtener_municipio(self,**kwargs): return no_crs
            with self._fake_manager_import(Manager),mock.patch("geopandas.read_file",return_value=backup):
                result,info=_load_boundary(config)
            self.assertIsNotNone(result);self.assertIn("sin CRS",info["error_GISManager"])
            self._assert_markdown_boundary_safe(info)

    def test_no_boundary_source_returns_consistent_unavailable_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=InventoryConfig(search_roots=(),output_dir=root/"out",reference_grid=root/"missing.tif",
                regional_dem=root/"regional.tif",boundary_gpkg=root/"missing.gpkg",marco_geoestadistico_dir=root/"missing_marco")
            result,info=_load_boundary(config)
            self.assertIsNone(result);self.assertFalse(info["disponible"]);self.assertIsNone(info["ruta"])
            self.assertEqual(info["fuente"],None);self.assertTrue(info["error_GISManager"])
            self._assert_markdown_boundary_safe(info)

    def test_factor_names(self):
        self.assertEqual(identify_factor("97_Altura_HAND.tif"), "hand")
        self.assertEqual(identify_factor("92_Pendiente_Grados.tif"), "pendiente")
        self.assertEqual(identify_factor("suelo_serie_III.shp"), "suelos")
        self.assertEqual(identify_factor("archivo_inesperado.tif"), "sin_clasificar")

    def test_json_reader_selection_and_document_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"metadatos_dem.json"
            path.write_text('{"fuente":"INEGI","crs":"EPSG:6368"}',encoding="utf-8")
            route=select_reader(path)
            self.assertEqual((route["lector"],route["tipo"]),("json","documentacion_json"))
            doc=inspect_json_document(path,route)
            self.assertEqual(doc["claves"],["fuente","crs"])

    def test_provenance_sidecar_is_documentation_for_associated_raster(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"elevacion.tif.provenance.json"
            path.write_text('{"output_sha256":"abc","parameters":{}}',encoding="utf-8")
            route=select_reader(path)
            self.assertEqual(route["tipo"],"documentacion_procedencia")
            self.assertEqual(inspect_json_document(path,route)["producto_asociado"],"elevacion.tif")

    def test_json_geojson_structure_remains_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"features.json"
            path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"id":1},"geometry":{"type":"Point","coordinates":[-103.7,19.3]}}]}',encoding="utf-8")
            route=select_reader(path)
            self.assertEqual((route["lector"],route["tipo"]),("geopandas","vector"))

    def test_malformed_json_is_detected_as_json_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"broken.json";path.write_text('{broken')
            route=select_reader(path)
            self.assertEqual(route["lector"],"json")
            self.assertEqual(route["tipo"],"json_invalido")

    def test_reader_selection_regression_for_raster_and_vectors(self):
        self.assertEqual(select_reader(Path("grid.tif"))["lector"],"rasterio")
        self.assertEqual(select_reader(Path("rivers.gpkg"))["lector"],"geopandas")
        self.assertEqual(select_reader(Path("rivers.geojson"))["lector"],"geopandas")

    def test_documents_are_not_counted_as_vector_layers_in_synthetic_run(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data";root.mkdir()
            raster=root/"mde.tif"
            profile={"driver":"GTiff","height":2,"width":2,"count":1,"dtype":"float32","crs":"EPSG:6368","transform":from_origin(0,30,15,15),"nodata":-9999}
            with rasterio.open(raster,"w",**profile) as dst: dst.write(np.ones((2,2),dtype=np.float32),1)
            doc=root/"metadata.json";doc.write_text('{"title":"metadata"}',encoding="utf-8")
            sidecar=root/"mde.tif.provenance.json";sidecar.write_text('{"step":1}',encoding="utf-8")
            gpkg=root/"boundary.gpkg"
            gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,30,30)],crs="EPSG:6368").to_file(gpkg,layer="limite_municipal",driver="GPKG")
            out=Path(tmp)/"out"
            report=run_inventory(InventoryConfig(search_roots=(root,),output_dir=out,reference_grid=raster,regional_dem=root/"regional.tif",boundary_gpkg=gpkg))
            docs=[x for x in report["archivos"] if x.get("tipo_archivo")=="documentacion"]
            layers=report["capas_vectoriales"]
            self.assertEqual(len(docs),2)
            self.assertTrue(all(x["lector_asignado"]=="json" for x in docs))
            self.assertEqual(len(layers),1)

    def test_hash_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"a.bin";path.write_bytes(b"abc")
            self.assertEqual(file_sha256(path),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_raster_metadata_mask_nodata_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"elevacion.tif"
            data=np.array([[10,11,12,13],[14,-9999,16,17],[18,19,20,21],[22,23,24,25]],dtype="float32")
            profile={"driver":"GTiff","height":4,"width":4,"count":1,"dtype":"float32","crs":"EPSG:6368","transform":from_origin(0,60,15,15),"nodata":-9999}
            with rasterio.open(path,"w",**profile) as dst: dst.write(data,1)
            rec=inspect_raster(path,box(0,0,60,60),None,block_rows=2,target_crs="EPSG:6368")
            self.assertEqual(rec["crs_epsilon"],6368)
            self.assertEqual(rec["dimensiones_columnas_filas"],[4,4])
            self.assertEqual(rec["nodata"],-9999)
            self.assertEqual(rec["celdas_municipales_validas"],15)
            self.assertEqual(rec["cobertura_municipal_pct_aprox"],93.75)
            self.assertIsNone(rec["cuadricula_coincide_referencia"])

    def test_spatial_quality_flags_grid_and_resolution(self):
        rec={"ruta":"x","crs":"EPSG:4326","crs_proyectado_metrico":False,"resolucion":[0.0001,0.0001],"cuadricula_coincide_referencia":False,"cobertura_municipal_pct_aprox":70}
        quality=evaluate_spatial_quality(rec)
        self.assertEqual(quality["evaluacion"],"requiere_procesamiento_o_validacion")
        self.assertEqual(len(quality["observaciones"]),4)

    def test_run_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_run_directory(tmp,"run-1")
            with self.assertRaises(FileExistsError): create_run_directory(tmp,"run-1")

    def test_vector_layer_metadata_and_municipal_intersection(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"capas.gpkg"
            frame=gpd.GeoDataFrame({"escala":["1:50 000"],"anio":[2024]},geometry=[box(0,0,10,10)],crs="EPSG:6368")
            frame.to_file(path,layer="suelo",driver="GPKG")
            rec=inspect_vector(path,"suelo",frame,"EPSG:6368")
            self.assertEqual(rec["entidades"],1)
            self.assertEqual(rec["entidades_que_intersectan_municipio"],1)
            self.assertIn("escala",rec["atributos_de_fuente"])
            self.assertEqual(rec["crs_epsilon"],6368)

    def test_inventory_outputs_are_reproducible_and_not_overwritten(self):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data";root.mkdir()
            raster=root/"mde_municipal.tif"
            profile={"driver":"GTiff","height":4,"width":4,"count":1,"dtype":"float32","crs":"EPSG:6368","transform":from_origin(0,60,15,15),"nodata":-9999}
            with rasterio.open(raster,"w",**profile) as dst: dst.write(np.ones((4,4),dtype=np.float32),1)
            gpkg=root/"hidrografia.gpkg"
            municipal=gpd.GeoDataFrame({"id":[1]},geometry=[box(0,0,60,60)],crs="EPSG:6368")
            municipal.to_file(gpkg,layer="limite_municipal",driver="GPKG")
            sub=gpd.GeoDataFrame({"_dcag_subbasin":["RH16Ba"]},geometry=[box(0,0,60,60)],crs="EPSG:6368")
            sub.to_file(gpkg,layer="subcuencas_regionales",driver="GPKG",mode="a")
            out=Path(tmp)/"out"
            raster_hash_before=file_sha256(raster)
            cfg=InventoryConfig(search_roots=(root,),output_dir=out,reference_grid=raster,regional_dem=root/"regional_missing.tif",boundary_gpkg=gpkg)
            report=run_inventory(cfg)
            self.assertEqual(report["caso_piloto"],"Villa de Álvarez, Colima")
            self.assertTrue(Path(report["productos"]["json"]).is_file())
            self.assertTrue(Path(report["productos"]["map"]).is_file())
            self.assertTrue(any(x["factor"]=="elevacion" for x in report["factores"]))
            self.assertTrue(any(x["factor"]=="distancia_cauces" and x["estado"]=="no_disponible_en_rutas_inspeccionadas" for x in report["factores"]))
            # Una carpeta de corrida existente no se reutiliza, y las fuentes siguen intactas.
            self.assertEqual(file_sha256(raster),raster_hash_before)


if __name__=="__main__":
    unittest.main(verbosity=2)
