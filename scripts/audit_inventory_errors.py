"""Audita exclusivamente los errores de lectura ya registrados en el inventario.

No repite el escaneo general, no corrige ni escribe archivos fuente y solo lee los
registros de ``errores_lectura`` del JSON original. La ejecución crea reportes en
un subdirectorio único.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import stat
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "results" / "territorial" / "factor_inventory" / "villa_de_alvarez" / "20261125T213730Z_4b965a4b" / "inventario_factores_20261125T213730Z_4b965a4b.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "territorial" / "error_audit"
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img", ".gpkg", ".shp", ".geojson", ".gml", ".fgb"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}
SHAPE_COMPONENTS = (".shp", ".shx", ".dbf")


class AuditError(RuntimeError):
    """El informe no contiene la estructura esperada o no puede escribirse."""


@dataclass
class AuditConfig:
    inventory_json: Path = DEFAULT_INVENTORY
    output_dir: Path = DEFAULT_OUTPUT
    project_root: Path = REPO_ROOT
    expected_error_count: int = 38
    overwrite: bool = False

    def __post_init__(self):
        for key in ("inventory_json", "output_dir", "project_root"):
            setattr(self, key, Path(getattr(self, key)).expanduser().resolve())
        if self.expected_error_count < 0:
            raise ValueError("expected_error_count no puede ser negativo")


def extract_error_records(report):
    """Extrae literalmente los registros registrados; no descubre nuevos archivos."""
    candidates = ("errores_lectura", "read_errors", "errores")
    records = None
    for key in candidates:
        value = report.get(key) if isinstance(report, dict) else None
        if isinstance(value, list):
            records = value
            break
    if records is None:
        raise AuditError("El JSON no contiene una lista errores_lectura reconocible")
    if not all(isinstance(record, dict) for record in records):
        raise AuditError("errores_lectura debe contener objetos JSON")
    return records


def _original_fields(record):
    path_keys = ("ruta", "path", "archivo", "file", "filename", "source")
    message_keys = ("error", "mensaje", "message", "exception", "detalle")
    original_path = None
    nested = record.get("archivo")
    if isinstance(nested, dict):
        original_path = nested.get("ruta") or nested.get("path") or nested.get("filename")
    if original_path is None:
        original_path = next((record[k] for k in path_keys if record.get(k) not in (None, "") and not isinstance(record.get(k), dict)), None)
    original_message = next((record[k] for k in message_keys if record.get(k) not in (None, "")), None)
    return (str(original_path) if original_path is not None else "",
            str(original_message) if original_message is not None else "")


def _resolve_record_path(original_path, report_path, project_root):
    if not original_path:
        return None
    raw = Path(original_path).expanduser()
    if raw.is_absolute():
        return raw
    candidates = (project_root / raw, report_path.parent / raw, raw.resolve())
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash(record):
    for key in ("sha256", "hash", "hash_original", "sha256_original"):
        value = record.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
    for container in (record.get("metadatos"), record.get("metadata"), record.get("archivo")):
        if isinstance(container, dict):
            for key in ("sha256", "hash", "output_sha256"):
                value = container.get(key)
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                    return value.lower()
    return None


def _sidecar_of(path):
    name = path.name.lower()
    return name.endswith(".provenance.json") or name.endswith(".provenance.json")


def _route_and_inspect(path):
    """Inspecciona un único archivo con el lector propio de su formato."""
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    result = {"extension": suffix, "lector_asignado": None, "lectura_correcta": False,
              "metadatos_archivo": {}, "error_lector": None, "componentes_faltantes": []}
    if lower_name.endswith(".provenance.json") or suffix == ".json":
        result["lector_asignado"] = "json"
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            result["lectura_correcta"] = True
            result["categoria_formato"] = "sidecar_procedencia" if lower_name.endswith(".provenance.json") else "json_metadatos_o_reporte"
            result["metadatos_archivo"] = {"tipo_json": type(value).__name__, "claves_raiz": list(value)[:100] if isinstance(value, dict) else None}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            result["error_lector"] = f"{type(exc).__name__}: {exc}"
            result["categoria_formato"] = "json_invalido"
        return result
    if suffix in TABLE_EXTENSIONS:
        result["lector_asignado"] = "tabular"
        try:
            if suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream, delimiter=delimiter)
                    header = next(reader, [])
                result["metadatos_archivo"] = {"columnas": header[:100]}
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                result["metadatos_archivo"] = {"columnas": pq.read_schema(path).names}
            else:
                import openpyxl
                workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
                result["metadatos_archivo"] = {"hojas": workbook.sheetnames}
                workbook.close()
            result["lectura_correcta"] = True
            result["categoria_formato"] = "tabla_no_geoespacial"
        except Exception as exc:
            result["error_lector"] = f"{type(exc).__name__}: {exc}"
            result["categoria_formato"] = "tabla_ilegible_o_dependencia"
        return result
    if suffix in {".tif", ".tiff", ".vrt", ".img"}:
        result["lector_asignado"] = "rasterio"
        try:
            import rasterio
            from rasterio.windows import Window
            with rasterio.open(path) as dataset:
                dataset.read(1, window=Window(0, 0, min(1, dataset.width), min(1, dataset.height)))
                result["lectura_correcta"] = True
                result["categoria_formato"] = "raster_geoespacial_valido"
                result["metadatos_archivo"] = {"driver": dataset.driver, "crs": dataset.crs.to_string() if dataset.crs else None,
                    "width": dataset.width, "height": dataset.height, "count": dataset.count,
                    "dtype": list(dataset.dtypes), "nodata": dataset.nodata, "bounds": list(dataset.bounds),
                    "resolution": list(dataset.res)}
        except Exception as exc:
            result["error_lector"] = f"{type(exc).__name__}: {exc}"
            result["categoria_formato"] = "raster_controlador_o_archivo"
        return result
    if suffix == ".shp":
        result["lector_asignado"] = "geopandas_shapefile"
        missing = [path.with_suffix(ext).name for ext in SHAPE_COMPONENTS if not path.with_suffix(ext).is_file()]
        result["componentes_faltantes"] = missing
    elif suffix in {".gpkg", ".geojson", ".gml", ".fgb"}:
        result["lector_asignado"] = "geopandas_vector"
    else:
        result["lector_asignado"] = "ninguno_extension_no_soportada"
        result["categoria_formato"] = "formato_desconocido"
        return result
    try:
        import geopandas as gpd
        if suffix == ".gpkg":
            layers = gpd.list_layers(path)["name"].tolist()
            result["metadatos_archivo"]["capas"] = layers
            layer_results=[]
            for layer in layers:
                frame=gpd.read_file(path,layer=layer,rows=1)
                layer_results.append({"capa":layer,"crs":frame.crs.to_string() if frame.crs else None,
                                      "geometrias_muestra":len(frame),"tipos":sorted(set(frame.geometry.geom_type)) if "geometry" in frame else []})
            result["metadatos_archivo"]["muestras_capas"] = layer_results
        else:
            frame=gpd.read_file(path,rows=1)
            result["metadatos_archivo"].update({"crs":frame.crs.to_string() if frame.crs else None,
                                                "columnas":list(frame.columns),"tipos":sorted(set(frame.geometry.geom_type)) if "geometry" in frame else []})
        result["lectura_correcta"] = True
        result["categoria_formato"] = "vector_geoespacial_valido"
    except Exception as exc:
        result["error_lector"] = f"{type(exc).__name__}: {exc}"
        if result["componentes_faltantes"]:
            result["categoria_formato"] = "shapefile_incompleto"
        else:
            result["categoria_formato"] = "vector_controlador_estructura_o_archivo"
    return result


def _classify(record, route, original_message, access_error=None):
    message = (original_message + " " + (route.get("error_lector") or "") + " " + (access_error or "")).casefold()
    ext = route.get("extension", "")
    fmt = route.get("categoria_formato", "")
    if access_error or any(token in message for token in ("permissionerror", "access is denied", "permission denied", "archivo bloqueado")):
        category="E_error_de_acceso"; recoverable=False; advice="Revisar permisos, bloqueo del archivo y disponibilidad/sincronización de Google Drive; repetir solo este registro tras recuperar acceso."
    elif "modulenotfounderror" in message or "no module named 'scripts'" in message or "no module named \"scripts\"" in message:
        category="F_error_de_implementacion"; recoverable=route.get("lectura_correcta",False); advice="Ejecutar desde la raíz del repositorio como módulo (python -m scripts.inventory_territorial_factors) y comprobar que scripts/ sea paquete/importable. No es evidencia de corrupción del dato."
    elif fmt in {"sidecar_procedencia", "json_metadatos_o_reporte", "tabla_no_geoespacial"} and route.get("lectura_correcta"):
        category="A_falso_positivo_por_formato"; recoverable=True; advice="Conservar en el inventario documental y excluir del lector geoespacial; usar JSON/lector tabular según la extensión."
    elif fmt=="formato_desconocido":
        category="G_otro_formato_no_soportado"; recoverable=False; advice="Identificar el formato real mediante firma/metadata del productor; no renombrar ni convertir hasta documentar la fuente."
    elif fmt=="shapefile_incompleto":
        category="D_error_de_estructura"; recoverable=False; advice="Localizar los componentes .shp/.shx/.dbf faltantes desde la fuente; conservar el conjunto original y validar CRS (.prj)."
    elif fmt in {"json_invalido", "raster_controlador_o_archivo", "vector_controlador_estructura_o_archivo", "tabla_ilegible_o_dependencia"}:
        if any(t in message for t in ("driver", "unsupported", "not recognized", "no such driver")):
            category="B_error_de_controlador_o_formato"; recoverable=False; advice="Verificar driver/format support ya disponible y la identificación real del formato; no instalar ni convertir automáticamente."
        elif any(t in message for t in ("modulenotfounderror", "no module named", "importerror")):
            category="B_error_de_controlador_o_formato"; recoverable=False; advice="La lectura requiere una dependencia/driver ausente; documentar versión requerida antes de autorizar instalación."
        else:
            category="C_error_de_archivo"; recoverable=False; advice="Comparar tamaño/hash con la fuente y probar una copia de trabajo; no reparar ni sobrescribir el original."
    elif route.get("lectura_correcta"):
        category="A_falso_positivo_por_formato" if ext in {".json", ".csv", ".tsv", ".xlsx", ".xls", ".parquet"} else "G_lectura_actual_correcta"
        recoverable=True; advice="La lectura correspondiente al formato funciona ahora; conservar mensaje original y tratarlo como falso positivo o fallo transitorio, sujeto a revisar el flujo que generó el error."
    else:
        category="G_otro"; recoverable=False; advice="Revisar individualmente el mensaje original, el formato y el estado del archivo; no aplicar una reparación masiva."
    return {"clasificacion":category,"recuperable":recoverable,"recomendacion":advice}


def audit_record(record, inventory_path, project_root, expected_hash_override=None):
    original_path, original_message = _original_fields(record)
    resolved = _resolve_record_path(original_path, inventory_path, project_root)
    output={"ruta_original":original_path,"mensaje_original":original_message,"registro_original":record,
            "ruta_resuelta":str(resolved) if resolved else None,"existe":bool(resolved and resolved.exists()),
            "extension":resolved.suffix.lower() if resolved else None,"bytes":None,"fecha_modificacion":None,
            "hash_esperado":_expected_hash(record) or expected_hash_override,"hash_actual":None,"hash_coincide":None,
            "lector_asignado":None,"lectura_correcta":False,"metadatos_archivo":{},"componentes_faltantes":[],
            "error_lector":None,"categoria_formato":None,"categoria_error":None,"recuperable":False,"recomendacion":None}
    if resolved is None or not resolved.exists():
        output.update({"categoria_formato":"archivo_ausente","categoria_error":"E_error_de_acceso","recomendacion":"Verificar ruta absoluta, montaje/Google Drive y registro del inventario; no confundir ausencia con archivo corrupto."})
        return output
    try:
        stat_result=resolved.stat()
        output["bytes"]=stat_result.st_size;output["fecha_modificacion"]=datetime.fromtimestamp(stat_result.st_mtime,timezone.utc).isoformat()
        output["hash_actual"]=_sha256(resolved)
        expected=output["hash_esperado"]
        output["hash_coincide"]=(output["hash_actual"]==expected) if expected else None
        route=_route_and_inspect(resolved)
    except PermissionError as exc:
        route={"extension":resolved.suffix.lower(),"lector_asignado":None,"lectura_correcta":False,"metadatos_archivo":{},"error_lector":None,"componentes_faltantes":[],"categoria_formato":"acceso_denegado"}
        classification=_classify(record,route,original_message,access_error=f"{type(exc).__name__}: {exc}")
        output.update(route);output.update({"categoria_error":classification["clasificacion"],"recuperable":classification["recuperable"],"recomendacion":classification["recomendacion"],"error_lector":f"{type(exc).__name__}: {exc}"})
        return output
    classification=_classify(record,route,original_message)
    output.update(route)
    output.update({"categoria_error":classification["clasificacion"],"recuperable":classification["recuperable"],"recomendacion":classification["recomendacion"]})
    if output["hash_coincide"] is False:
        output["recomendacion"] += " El hash actual difiere del registrado: investigar cambio de versión antes de usarlo."
    return output


def _make_run_dir(base, run_id):
    path=Path(base)/run_id
    path.mkdir(parents=True,exist_ok=False)
    return path


def _write_csv(path, rows, fields):
    with Path(path).open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in rows:
            flat={k:(json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(dict,list)) else v) for k,v in row.items()}
            writer.writerow(flat)


def run_audit(config: AuditConfig):
    """Audita solo la lista de errores ya guardada; no explora carpetas fuente."""
    if not config.inventory_json.is_file(): raise FileNotFoundError(config.inventory_json)
    report=json.loads(config.inventory_json.read_text(encoding="utf-8-sig"))
    records=extract_error_records(report)
    if len(records)!=config.expected_error_count:
        raise AuditError(f"El reporte contiene {len(records)} registros de error; se esperaban {config.expected_error_count}. No se ejecuta la auditoría sin verificar el inventario de referencia.")
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    run_dir=_make_run_dir(config.output_dir,run_id)
    start=time.perf_counter()
    # El inventario suele mantener hashes de archivos válidos en su arreglo "archivos";
    # se recuperan por ruta si un error relacionado no repite el hash en su objeto.
    hash_by_path={}
    for item in report.get("archivos",[]) if isinstance(report,dict) else []:
        if not isinstance(item,dict): continue
        item_path,_=_original_fields(item)
        item_hash=_expected_hash(item)
        if item_path and item_hash:
            resolved_item=_resolve_record_path(item_path,config.inventory_json,config.project_root)
            if resolved_item: hash_by_path[str(resolved_item).casefold()]=item_hash
    audited=[]
    for row in records:
        original_path,_=_original_fields(row)
        resolved=_resolve_record_path(original_path,config.inventory_json,config.project_root)
        expected=hash_by_path.get(str(resolved).casefold()) if resolved else None
        audited.append(audit_record(row,config.inventory_json,config.project_root,expected_hash_override=expected))
    categories={}
    for row in audited: categories[row["categoria_error"]]=categories.get(row["categoria_error"],0)+1
    fields=sorted({key for row in audited for key in row if key!="registro_original"})
    paths={"json":run_dir/f"auditoria_errores_{run_id}.json","csv":run_dir/f"auditoria_errores_{run_id}.csv",
           "classification_csv":run_dir/f"clasificacion_errores_{run_id}.csv","recoverable_csv":run_dir/f"archivos_recuperables_{run_id}.csv",
           "excluded_csv":run_dir/f"archivos_excluidos_{run_id}.csv","markdown":run_dir/f"resumen_auditoria_{run_id}.md"}
    _write_csv(paths["csv"],audited,fields)
    _write_csv(paths["classification_csv"],audited,["ruta_original","mensaje_original","categoria_error","categoria_formato","lector_asignado","existe","lectura_correcta","recuperable","recomendacion"])
    _write_csv(paths["recoverable_csv"],[r for r in audited if r["recuperable"]],fields)
    _write_csv(paths["excluded_csv"],[r for r in audited if not r["recuperable"]],fields)
    counts={"registros":len(audited),"falsos_positivos_formato":sum(v for k,v in categories.items() if k=="A_falso_positivo_por_formato"),
            "errores_reales_o_pendientes":len(audited)-sum(1 for r in audited if r["categoria_error"].startswith("A_")),
            "recuperables":sum(1 for r in audited if r["recuperable"]),"requieren_intervencion":sum(1 for r in audited if not r["recuperable"]),
            "por_categoria":categories}
    report_out={"id_ejecucion":run_id,"fecha_utc":datetime.now(timezone.utc).isoformat(),"inventario_fuente":str(config.inventory_json),
        "sha256_inventario_fuente":_sha256(config.inventory_json),"errores_esperados":config.expected_error_count,
        "errores_auditados":len(audited),"conteos":counts,"resultados":audited,"versiones":{"python":platform.python_version()},
        "duracion_segundos":time.perf_counter()-start,"rutas_productos":{k:str(v) for k,v in paths.items()},
        "metodo":"Audita solamente entradas originales del arreglo errores_lectura; no recorre directorios ni recalcula inventario territorial."}
    paths["json"].write_text(json.dumps(report_out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    md=["# Auditoría de errores del inventario territorial","",f"ID: `{run_id}`",f"Fuente: `{config.inventory_json}`","",f"Registros auditados: **{len(audited)}**","", "## Conteos", "", "| Resultado | Cantidad |", "|---|---:|"]
    md.extend([f"| {key} | {value} |" for key,value in counts.items() if isinstance(value,int)])
    md += ["", "## Categorías", "", *[f"- `{key}`: {value}" for key,value in categories.items()], "", "## Método y límites", "", "Se preserva el mensaje original por registro. La extensión dirige el lector; la capacidad de abrir un archivo no determina por sí sola su validez temática. No se reparan, convierten, renombran ni sobrescriben fuentes.", ""]
    paths["markdown"].write_text("\n".join(md),encoding="utf-8")
    return report_out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",type=Path,default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--project-root",type=Path,default=REPO_ROOT)
    parser.add_argument("--expected-count",type=int,default=38)
    args=parser.parse_args(argv)
    result=run_audit(AuditConfig(args.inventory,args.output_dir,args.project_root,args.expected_count))
    print(json.dumps({"id_ejecucion":result["id_ejecucion"],"conteos":result["conteos"],"productos":result["rutas_productos"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
