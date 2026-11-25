"""Inventario reproducible y auditoría espacial de factores territoriales DCAG.

Solo inspecciona archivos. No reproyecta ni modifica fuentes, no calcula factores
y no genera mapas de amenaza, susceptibilidad o riesgo. La ejecución real requiere
invocación explícita de ``run_inventory`` o del CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRITORIAL_ROOT = REPO_ROOT / "datos_territoriales" / "INEGI"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "territorial" / "factor_inventory" / "villa_de_alvarez"
DEFAULT_REFERENCE_GRID = REPO_ROOT / "results" / "territorial" / "dem" / "villa_de_alvarez_v2" / "elevacion_villa_de_alvarez_utm13n.tif"
DEFAULT_REGIONAL_DEM = REPO_ROOT / "datos_territoriales" / "INEGI" / "CEM_4_0" / "procesados" / "CEM40_Colima_Jalisco_Dominio_Hidrologico_Buffer2000m_UTM13N_15m.tif"
DEFAULT_BOUNDARY_GPKG = REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez" / "red_hidrografica_villa_de_alvarez.gpkg"
DEFAULT_BOUNDARY_LAYER = "limite_municipal"
DEFAULT_MARCO_DIR = Path(r"C:\Users\rente\Desktop\Marco_geoestadistico_2025")
DEFAULT_CRS = "EPSG:6368"
RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img"}
VECTOR_EXTENSIONS = {".gpkg", ".shp", ".geojson", ".json", ".gml", ".fgb"}
TABLE_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}
GEOJSON_TYPES = {"Feature", "FeatureCollection", "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"}

FACTOR_PATTERNS = {
    "elevacion": ("mde", "dem", "elevacion", "elevation", "cem"),
    "pendiente": ("pendiente", "slope"),
    "hand": ("hand", "altura_sobre_drenaje"),
    "twi": ("twi", "topographic_wetness"),
    "acumulacion_flujo": ("acumulacion", "flow_accum", "accumulation"),
    "distancia_cauces": ("distancia_cauce", "distance_to_stream", "distancia_corriente"),
    "suelos": ("suelo", "edaf", "soil", "textura"),
    "uso_cobertura": ("uso_suelo", "uso_del_suelo", "cobertura", "landcover", "land_cover", "inegi_usv"),
    "precipitacion": ("precipit", "excedenc", "gev", "era5", "ssp", "retorno"),
    "hidrografia": ("hidrograf", "corriente", "rio", "cauce", "drenaje", "subcuenca", "red_hidro"),
    "limite_territorial": ("marco_geo", "limite_municip", "villa_de_alvarez", "municipal_boundary"),
}


class InventoryError(RuntimeError):
    """Error de configuración o de lectura de una fuente inventariada."""


@dataclass
class InventoryConfig:
    search_roots: tuple[Path, ...] = (REPO_ROOT / "datos_territoriales", REPO_ROOT / "results", REPO_ROOT / "CONAGUA")
    output_dir: Path = DEFAULT_OUTPUT_DIR
    reference_grid: Path = DEFAULT_REFERENCE_GRID
    regional_dem: Path = DEFAULT_REGIONAL_DEM
    boundary_gpkg: Path = DEFAULT_BOUNDARY_GPKG
    boundary_layer: str = DEFAULT_BOUNDARY_LAYER
    marco_geoestadistico_dir: Path = DEFAULT_MARCO_DIR
    estado: str = "Colima"
    municipio: str = "Villa de Álvarez"
    target_crs: str = DEFAULT_CRS
    block_rows: int = 256
    hash_files: bool = True

    def __post_init__(self):
        self.search_roots = tuple(Path(p).expanduser().resolve() for p in self.search_roots)
        for name in ("output_dir", "reference_grid", "regional_dem", "boundary_gpkg", "marco_geoestadistico_dir"):
            setattr(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.block_rows < 1:
            raise ValueError("block_rows debe ser positivo")


def identify_factor(filename: str) -> str:
    """Asigna tema por términos explícitos del nombre; no infiere valores temáticos."""
    name = Path(filename).stem.lower().replace("-", "_").replace(" ", "_")
    for factor, patterns in FACTOR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name):
                return factor
    return "sin_clasificar"


def _looks_like_geojson(value):
    """Devuelve True solo ante una estructura GeoJSON reconocible y coherente."""
    if not isinstance(value, dict) or value.get("type") not in GEOJSON_TYPES:
        return False
    kind=value.get("type")
    geometry_types=GEOJSON_TYPES-{"Feature","FeatureCollection"}
    if kind=="FeatureCollection":
        features=value.get("features")
        if not isinstance(features,list): return False
        return all(isinstance(f,dict) and f.get("type")=="Feature" and
                   (f.get("geometry") is None or (isinstance(f.get("geometry"),dict) and f["geometry"].get("type") in geometry_types))
                   for f in features)
    if kind=="Feature":
        geometry=value.get("geometry")
        return geometry is None or (isinstance(geometry,dict) and geometry.get("type") in geometry_types)
    return isinstance(value.get("coordinates"), (list,tuple)) or (kind=="GeometryCollection" and isinstance(value.get("geometries"),list))


def select_reader(path):
    """Selecciona lector por formato y estructura; conserva JSON como documento salvo GeoJSON válido."""
    path=Path(path)
    name=path.name.casefold()
    suffix=path.suffix.casefold()
    if name.endswith(".provenance.json"):
        return {"lector":"json","tipo":"documentacion_procedencia","formato":"json_sidecar"}
    if suffix in RASTER_EXTENSIONS:
        return {"lector":"rasterio","tipo":"raster","formato":suffix.lstrip(".")}
    if suffix==".geojson":
        return {"lector":"geopandas","tipo":"vector","formato":"geojson"}
    if suffix==".json":
        try:
            with path.open("r",encoding="utf-8-sig") as stream:
                value=json.load(stream)
        except (OSError,UnicodeError,json.JSONDecodeError) as exc:
            return {"lector":"json","tipo":"json_invalido","formato":"json","error":f"{type(exc).__name__}: {exc}"}
        if _looks_like_geojson(value):
            return {"lector":"geopandas","tipo":"vector","formato":"geojson_en_json"}
        return {"lector":"json","tipo":"documentacion_json","formato":"json","tipo_json":type(value).__name__,"claves":list(value)[:100] if isinstance(value,dict) else None}
    if suffix in {".gpkg", ".shp", ".gml", ".fgb"}:
        return {"lector":"geopandas","tipo":"vector","formato":suffix.lstrip(".")}
    if suffix in TABLE_EXTENSIONS:
        return {"lector":"tabular","tipo":"tabla","formato":suffix.lstrip(".")}
    return {"lector":None,"tipo":"desconocido","formato":suffix.lstrip(".")}


def inspect_json_document(path, selection=None):
    """Lee JSON documental como trazabilidad; nunca busca geometrías ni crea capas."""
    selection=selection or select_reader(path)
    if selection["tipo"] not in {"documentacion_json","documentacion_procedencia"}:
        raise ValueError("El archivo no fue clasificado como JSON documental")
    with Path(path).open("r",encoding="utf-8-sig") as stream:
        value=json.load(stream)
    associated=None
    if selection["tipo"]=="documentacion_procedencia":
        source_name=Path(path).name[:-len(".provenance.json")]
        associated=source_name
    return {"tipo_json":type(value).__name__,"claves":list(value)[:100] if isinstance(value,dict) else None,
            "producto_asociado":associated,"cantidad_elementos":len(value) if isinstance(value,(dict,list)) else None}


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _metric_projected(crs) -> bool:
    if crs is None or not crs.is_projected:
        return False
    units=getattr(crs,"linear_units",None)
    if not units and getattr(crs,"axis_info",None): units=crs.axis_info[0].unit_name
    return (units or "").strip().lower() in {"metre", "meter", "metres", "meters"}


def _json_safe(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def inspect_raster(path, municipal_geometry=None, reference=None, block_rows=256, target_crs=DEFAULT_CRS):
    """Lee metadatos y estima cobertura municipal sin cargar el raster completo."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from rasterio.warp import transform_bounds

    path = Path(path)
    with rasterio.open(path) as ds:
        crs = ds.crs
        bounds = tuple(float(v) for v in ds.bounds)
        geom = municipal_geometry
        geom_target = None
        if geom is not None and crs is not None:
            if getattr(geom, "crs", None) is not None:
                geom_target = geom.to_crs(target_crs).geometry.union_all()
                geom = geom.to_crs(crs).geometry.union_all()
            elif hasattr(geom, "geom_type"):
                geom_target = geom
                if reference and reference.get("crs") and reference["crs"] != crs.to_string():
                    from pyproj import CRS, Transformer
                    from shapely.ops import transform as transform_geometry
                    geom = transform_geometry(Transformer.from_crs(CRS.from_user_input(reference["crs"]), crs, always_xy=True).transform, geom)
        municipal_cells = valid_municipal = total_valid = 0
        for row0 in range(0, ds.height, block_rows):
            win = Window(0, row0, ds.width, min(block_rows, ds.height-row0))
            valid_window=ds.read_masks(1, window=win)>0
            total_valid += int(valid_window.sum())
            if geom is not None:
                mask = rasterize([(geom, 1)], out_shape=(int(win.height), ds.width),
                                 transform=window_transform(win, ds.transform), fill=0, dtype="uint8") > 0
                municipal_cells += int(mask.sum())
                valid_municipal += int(np.count_nonzero(mask & valid_window))
        same_grid = None
        if reference:
            same_grid = (crs is not None and crs.to_string() == reference.get("crs") and
                         ds.width == reference.get("width") and ds.height == reference.get("height") and
                         tuple(ds.transform)[:6] == tuple(reference.get("transform", ()))[:6])
        if crs is not None:
            bounds_target = transform_bounds(crs, target_crs, *bounds, densify_pts=21)
        else:
            bounds_target = bounds
        pixel_area = abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d)
        municipal_area = float(geom_target.area) if geom_target is not None else None
        coverage_pct = (min(100.0, 100.0 * valid_municipal * pixel_area / municipal_area)
                        if municipal_area and _metric_projected(crs) else None)
        return {
            "ruta": str(path), "nombre": path.name, "formato": path.suffix.lower(),
            "bytes": path.stat().st_size, "fecha_modificacion": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "crs": crs.to_string() if crs else None, "crs_epsilon": crs.to_epsg() if crs else None,
            "crs_proyectado_metrico": _metric_projected(crs), "resolucion": [float(x) for x in ds.res],
            "dimensiones_columnas_filas": [ds.width, ds.height], "tipo_dato": list(ds.dtypes),
            "nodata": ds.nodata, "extension_original": list(bounds), "extension_crs_objetivo": list(bounds_target),
            "cuenta": ds.count, "bloques": [list(v) for v in ds.block_shapes],
            "celdas_municipales_rasterizadas": municipal_cells,
            "celdas_municipales_validas": valid_municipal, "area_pixel_crs2": pixel_area,
            "celdas_validas_totales":total_valid,"celdas_invalidas_totales":ds.width*ds.height-total_valid,
            "porcentaje_celdas_validas_en_extension":100*total_valid/(ds.width*ds.height),
            "area_municipal_objetivo_m2": municipal_area,
            "cobertura_municipal_pct_aprox": coverage_pct,
            "metodo_cobertura": "suma de área de píxeles válidos cuyo centro cae dentro del municipio / área municipal en CRS objetivo; aproximación por rasterización de centro de celda" if municipal_area else None,
            "cuadricula_coincide_referencia": same_grid,
            "descripciones_bandas": list(ds.descriptions), "unidades_bandas": list(ds.units),
            "tags_dataset": ds.tags(), "tags_bandas": [ds.tags(i+1) for i in range(ds.count)],
        }


def inspect_vector(path, layer=None, municipal_geometry=None, target_crs=DEFAULT_CRS):
    """Inspecciona metadatos básicos de vector; reporta capas por separado."""
    import geopandas as gpd
    path = Path(path)
    frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    crs = frame.crs
    geom = frame.geometry
    valid = geom.notna() & ~geom.is_empty
    types = sorted(set(geom[valid].geom_type.astype(str)))
    target = frame.to_crs(target_crs) if crs is not None else None
    intersects = None
    coverage_pct = None
    if municipal_geometry is not None and target is not None:
        mun = municipal_geometry
        if getattr(mun, "crs", None) is not None:
            mun = mun.to_crs(target_crs).geometry.union_all()
        elif hasattr(mun, "geom_type"):
            mun = mun
        else:
            mun = None
        if mun is not None:
            intersects = int(target.geometry[valid].intersects(mun).sum())
            coverage_pct = 100.0*intersects/max(1,int(valid.sum()))
    metadata_fields={}
    for field in frame.columns:
        if re.search(r"anio|año|year|escala|scale|fuente|source|version|resol|fecha",str(field),re.I):
            vals=frame[field].dropna().astype(str).unique()
            metadata_fields[str(field)]={"valores_unicos_hasta_20":vals[:20].tolist(),"cantidad_unicos":len(vals)}
    target_bounds=list(target.total_bounds) if target is not None and len(target) else None
    return {"ruta":str(path),"nombre":path.name,"formato":path.suffix.lower(),
            "bytes":path.stat().st_size,"fecha_modificacion":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
            "capa":layer,"entidades":len(frame),"geometrias_validas":int(valid.sum()),
            "geometrias_invalidas_o_vacias":int((~valid | ~geom.is_valid.fillna(False)).sum()),
            "tipos_geometria":types,"crs":crs.to_string() if crs else None,
            "crs_epsilon":crs.to_epsg() if crs else None,"crs_proyectado_metrico":_metric_projected(crs),
            "columnas":list(frame.columns),"atributos_de_fuente":metadata_fields,
            "extension":list(frame.total_bounds) if len(frame) else None,"extension_crs_objetivo":target_bounds,
            "entidades_que_intersectan_municipio":intersects,"porcentaje_entidades_intersectan_municipio":coverage_pct}


def evaluate_spatial_quality(record, reference_crs=DEFAULT_CRS, target_resolution=(15.0, 15.0)):
    """Clasifica limitaciones; no transforma ni remuestrea el insumo."""
    reasons=[]
    if record.get("crs") is None: reasons.append("CRS ausente")
    elif not record.get("crs_proyectado_metrico",False): reasons.append("CRS no proyectado en metros")
    if record.get("resolucion") is not None and not all(np.isclose(abs(a), b, rtol=0, atol=1e-6) for a,b in zip(record["resolucion"],target_resolution)):
        reasons.append("resolución distinta de 15 m; remuestrear no añade precisión temática")
    if record.get("cuadricula_coincide_referencia") is False: reasons.append("cuadrícula no coincide con referencia municipal")
    coverage = record.get("cobertura_municipal_pct_aprox", record.get("cobertura_municipal_pct"))
    if coverage is not None and coverage < 100:
        reasons.append("cobertura municipal parcial o máscara NoData")
    return {"ruta":record.get("ruta"),"crs_objetivo":reference_crs,"resolucion_objetivo_m":list(target_resolution),
            "crs_proyectado_metrico":record.get("crs_proyectado_metrico"),
            "coincide_cuadricula_referencia":record.get("cuadricula_coincide_referencia"),
            "cobertura_municipal_pct":coverage,
            "evaluacion":"requiere_procesamiento_o_validacion" if reasons else "compatible_en_controles_disponibles",
            "observaciones":reasons}


def _discover_files(roots):
    seen=set()
    extensions=RASTER_EXTENSIONS|VECTOR_EXTENSIONS|TABLE_EXTENSIONS
    for root in roots:
        if not root.is_dir(): continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                key=str(path.resolve()).casefold()
                if key not in seen:
                    seen.add(key); yield path.resolve()


def _load_boundary(config):
    import geopandas as gpd
    manager_error=None
    if config.marco_geoestadistico_dir.is_dir():
        try:
            from scripts.gis_manager import GISManager
            manager=GISManager(config.marco_geoestadistico_dir)
            frame=manager.obtener_municipio(estado=config.estado,municipio=config.municipio,
                                            crs_destino=config.target_crs,territorio="completo")
            if frame is None or frame.empty:
                manager_error="GISManager.obtener_municipio devolvió una capa vacía."
            elif frame.crs is None:
                manager_error="GISManager.obtener_municipio devolvió una capa sin CRS."
            else:
                result=frame.to_crs(config.target_crs)
                return result,{"disponible":True,"ruta":str(config.marco_geoestadistico_dir),
                    "fuente":"GISManager.obtener_municipio","mecanismo":"Marco Geoestadístico mediante GISManager",
                    "capa":None,"marco_geoestadistico":str(config.marco_geoestadistico_dir),
                    "estado":config.estado,"municipio":config.municipio,
                    "crs":result.crs.to_string() if result.crs else None,"entidades":len(result)}
        except Exception as exc:
            manager_error=f"{type(exc).__name__}: {exc}"
    else:
        manager_error="Directorio del Marco Geoestadístico no disponible"
    if not config.boundary_gpkg.is_file():
        return None,{"disponible":False,"ruta":None,"ruta_configurada":str(config.boundary_gpkg),
                     "fuente":None,"mecanismo":None,"capa":None,"error_GISManager":manager_error,
                     "error":"No está disponible el GeoPackage municipal de respaldo."}
    try:
        frame=gpd.read_file(config.boundary_gpkg,layer=config.boundary_layer)
    except Exception as exc:
        return None,{"disponible":False,"ruta":str(config.boundary_gpkg),"fuente":None,
                     "mecanismo":"GeoPackage municipal de respaldo no legible","capa":config.boundary_layer,
                     "error":f"{type(exc).__name__}: {exc}","error_GISManager":manager_error}
    if frame is None or frame.empty:
        return None,{"disponible":False,"ruta":str(config.boundary_gpkg),"fuente":None,
                     "mecanismo":"GeoPackage municipal de respaldo","capa":config.boundary_layer,
                     "error":"La capa municipal del GeoPackage está vacía.","error_GISManager":manager_error}
    if frame.crs is None:
        return None,{"disponible":False,"ruta":str(config.boundary_gpkg),"fuente":None,
                     "mecanismo":"GeoPackage municipal de respaldo","capa":config.boundary_layer,
                     "error":"La capa municipal del GeoPackage no declara CRS.","error_GISManager":manager_error}
    result=frame.to_crs(config.target_crs)
    return result,{"disponible":True,"ruta":str(config.boundary_gpkg),
                   "fuente":"GeoPackage procesado","mecanismo":"GeoPackage municipal de respaldo; GISManager no produjo un límite utilizable",
                   "capa":config.boundary_layer,"crs":result.crs.to_string() if result.crs else None,
                   "entidades":len(result),"error_GISManager":manager_error}


def _boundary_summary_markdown(boundary_info):
    """Forma líneas Markdown tolerantes a campos ausentes o fuentes no disponibles."""
    info=boundary_info if isinstance(boundary_info,dict) else {}
    route=info.get("ruta") or info.get("ruta_configurada") or "no disponible"
    source=info.get("fuente") or info.get("mecanismo") or "no disponible"
    layer=info.get("capa") or "no aplica/no disponible"
    crs=info.get("crs") or "no disponible"
    count=info.get("entidades")
    error=info.get("error") or "ninguno registrado"
    manager_error=info.get("error_GISManager") or "ninguno registrado"
    return [f"- Límite municipal disponible: `{bool(info.get('disponible', False))}`.",
            f"- Ruta declarada: `{route}`; fuente/mecanismo: `{source}`; capa: `{layer}`.",
            f"- CRS: `{crs}`; entidades: `{count if count is not None else 'no disponible'}`.",
            f"- Estado de lectura del límite: {error}.",
            f"- Estado de GISManager: {manager_error}."]


def _write_csv(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def create_run_directory(output_dir, run_id):
    """Crea una carpeta de ejecución exclusiva; no reutiliza ni reemplaza rutas."""
    path=Path(output_dir)/str(run_id)
    path.mkdir(parents=True,exist_ok=False)
    return path


def _factor_decisions(records):
    by={key:[] for key in FACTOR_PATTERNS}
    for record in records:
        by.setdefault(record["factor"],[]).append(record)
    out=[]
    for factor in FACTOR_PATTERNS:
        group=("peligro_hidrometeorologico_separado" if factor=="precipitacion" else
               "contexto_territorial" if factor in {"hidrografia","limite_territorial"} else
               "factor_territorial")
        candidates=by.get(factor,[])
        if candidates:
            raster_candidates=[r for r in candidates if r.get("tipo_archivo")=="raster"]
            usable=[r for r in raster_candidates if r.get("calidad_espacial",{}).get("evaluacion")=="compatible_en_controles_disponibles"]
            state="disponible_utilizable" if usable else "disponible_requiere_procesamiento_o_validacion"
            contribution={"elevacion":"contexto altimétrico y control topográfico", "pendiente":"condiciona escorrentía e infiltración superficial", "hand":"altura relativa a drenaje; no es profundidad de inundación", "twi":"índice topográfico relacionado con acumulación potencial de humedad", "acumulacion_flujo":"concentración topológica; no es caudal ni profundidad", "distancia_cauces":"proximidad geométrica; no prueba conectividad", "suelos":"propiedades potencialmente relacionadas con infiltración, solo según atributos documentados", "uso_cobertura":"impermeabilización/cobertura según clasificación original", "precipitacion":"caracteriza peligro meteorológico; se mantiene fuera del grupo territorial", "hidrografia":"red y contexto de drenaje", "limite_territorial":"máscara y unidad de presentación"}.get(factor,"")
        else:
            state="no_disponible_en_rutas_inspeccionadas";usable=[];contribution=""
        out.append({"factor":factor,"grupo_metodologico":group,"estado":state,"archivos":len(candidates),"candidatos_tecnicamente_compatibles":len(usable),"posible_contribucion":contribution,"observacion":"No se asignan pesos, clases ni valores de infiltración."})
    return out


def _build_map(records, boundary, basins, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from shapely.geometry import box
    fig,ax=plt.subplots(figsize=(11,9))
    if boundary is not None:
        boundary.to_crs(DEFAULT_CRS).plot(ax=ax,facecolor="#f4f1e8",edgecolor="black",linewidth=1.2,zorder=5)
    if basins is not None and not basins.empty:
        basins.to_crs(DEFAULT_CRS).boundary.plot(ax=ax,color="#557a95",linewidth=.7,zorder=4)
    colors={"elevacion":"#8c6d31","pendiente":"#d95f02","hand":"#7570b3","twi":"#1b9e77","acumulacion_flujo":"#e7298a","distancia_cauces":"#66a61e","suelos":"#a6761d","uso_cobertura":"#1f78b4","precipitacion":"#e31a1c","hidrografia":"#17becf","limite_territorial":"#555555","sin_clasificar":"#aaaaaa"}
    for rec in records:
        bounds=rec.get("extension_crs_objetivo") or rec.get("extension")
        if not bounds or len(bounds)!=4: continue
        try:
            x0,y0,x1,y1=map(float,bounds)
            rect=plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=colors.get(rec["factor"],"#aaa"),linewidth=.55,alpha=.65,zorder=2)
            ax.add_patch(rect)
        except Exception: pass
    handles=[Patch(facecolor="none",edgecolor=c,label=f) for f,c in colors.items() if any(x["factor"]==f for x in records)]
    if handles: ax.legend(handles=handles,loc="best",fontsize=7,title="Extensión de insumos")
    ax.set_title("Inventario territorial de insumos — Villa de Álvarez y contexto regional")
    ax.set_xlabel("Este (m), EPSG:6368");ax.set_ylabel("Norte (m)");ax.set_aspect("equal",adjustable="datalim");fig.tight_layout();fig.savefig(output,dpi=180);plt.close(fig)


def run_inventory(config: InventoryConfig):
    """Genera el inventario a demanda. No procesa datos ni modifica fuentes."""
    import rasterio
    import geopandas as gpd
    started=time.perf_counter();run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    run_dir=config.output_dir/run_id
    names=[f"inventario_factores_{run_id}.json",f"inventario_factores_{run_id}.csv",f"evaluacion_calidad_espacial_{run_id}.csv",f"factores_disponibles_{run_id}.csv",f"factores_faltantes_{run_id}.csv",f"reporte_inventario_{run_id}.md",f"mapa_inventario_territorial_{run_id}.png"]
    # Un identificador nuevo evita reemplazar incluso en una carpeta con corridas anteriores.
    run_dir=create_run_directory(config.output_dir,run_id)
    boundary,boundary_info=_load_boundary(config)
    municipal_geom=boundary.geometry.union_all() if boundary is not None else None
    reference=None
    if config.reference_grid.is_file():
        with rasterio.open(config.reference_grid) as ds:
            reference={"crs":ds.crs.to_string() if ds.crs else None,"transform":tuple(ds.transform)[:6],"width":ds.width,"height":ds.height,"bounds":tuple(ds.bounds),"resolucion":tuple(ds.res)}
    records=[];quality=[];errors=[];layer_records=[]
    for path in _discover_files(config.search_roots):
        factor=identify_factor(path.name)
        try:
            selection=select_reader(path)
            base={"ruta":str(path),"nombre":path.name,"factor":factor,"tipo_archivo":selection["tipo"],"lector_asignado":selection["lector"],"formato_detectado":selection["formato"]}
            if selection["tipo"]=="json_invalido":
                errors.append({"ruta":str(path),"error":selection["error"],"lector_asignado":"json","causa":"JSON documental inválido; no se envió a lector geoespacial"})
                continue
            if selection["tipo"] in {"documentacion_json","documentacion_procedencia"}:
                document={**base,"factor":"documentacion","tipo_archivo":"documentacion","documentacion":inspect_json_document(path,selection),
                          "bytes":path.stat().st_size,"fecha_modificacion":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                          "sha256":file_sha256(path) if config.hash_files else None}
                records.append(document)
            elif selection["tipo"]=="raster":
                rec=inspect_raster(path,boundary,reference,config.block_rows,config.target_crs);rec.update(base)
                rec["dominio_candidato"]="regional_soporte" if path==config.regional_dem or (rec.get("cuadricula_coincide_referencia") is False and identify_factor(path.name) in {"elevacion","hand","twi","acumulacion_flujo"}) else "municipal_o_no_determinado"
                rec["sha256"]=file_sha256(path) if config.hash_files else None
                rec["calidad_espacial"]=evaluate_spatial_quality(rec,config.target_crs)
                quality.append(rec["calidad_espacial"])
                records.append(rec)
            elif selection["tipo"]=="vector":
                if path.suffix.lower()==".gpkg":
                    try:
                        layers=gpd.list_layers(path)["name"].tolist()
                    except Exception as exc:
                        errors.append({"ruta":str(path),"error":str(exc)});continue
                else: layers=[None]
                for layer in layers:
                    rec=inspect_vector(path,layer,boundary,config.target_crs);rec.update(base);rec["capa"]=layer
                    rec["factor"]=identify_factor(str(layer or path.name))
                    rec["dominio_candidato"]="regional_soporte" if layer and any(t in str(layer).lower() for t in ("regional","subcuenca","corriente")) else "municipal_o_no_determinado"
                    rec["sha256"]=file_sha256(path) if config.hash_files else None
                    rec["calidad_espacial"]={"ruta":str(path),"evaluacion":"requiere_procesamiento_o_validacion","observaciones":["Verificar escala, cobertura geométrica y precisión temática antes de rasterizar a 15 m."]}
                    quality.append(rec["calidad_espacial"]);records.append(rec);layer_records.append(rec)
            elif selection["tipo"]=="tabla":
                rec={**base,"bytes":path.stat().st_size,"fecha_modificacion":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),"sha256":file_sha256(path) if config.hash_files else None,"nota":"Tabla listada por nombre; no se interpretan automáticamente sus valores."}
                records.append(rec)
            else:
                records.append({**base,"nota":"Formato no manejado por los lectores configurados."})
        except Exception as exc:
            errors.append({"ruta":str(path),"error":f"{type(exc).__name__}: {exc}"})
    factors=_factor_decisions(records)
    available=[x for x in factors if x["estado"]!="no_disponible_en_rutas_inspeccionadas"]
    missing=[x for x in factors if x["estado"]=="no_disponible_en_rutas_inspeccionadas"]
    # Recupera las cinco subcuencas regionales si están en el GPKG conocido; inspección únicamente.
    basins=None
    if config.boundary_gpkg.is_file():
        try:
            basins=gpd.read_file(config.boundary_gpkg,layer="subcuencas_regionales")
        except Exception: pass
    output_paths={k:run_dir/name for k,name in zip(("json","inventory_csv","quality_csv","available_csv","missing_csv","markdown","map"),names)}
    _write_csv(output_paths["inventory_csv"],records,sorted({k for x in records for k in x if isinstance(x,(str,int,float,bool)) or x is None}))
    _write_csv(output_paths["quality_csv"],quality,["ruta","crs_objetivo","resolucion_objetivo_m","crs_proyectado_metrico","coincide_cuadricula_referencia","cobertura_municipal_pct","evaluacion","observaciones"])
    _write_csv(output_paths["available_csv"],available,["factor","grupo_metodologico","estado","archivos","candidatos_tecnicamente_compatibles","posible_contribucion","observacion"])
    _write_csv(output_paths["missing_csv"],missing,["factor","grupo_metodologico","estado","archivos","candidatos_tecnicamente_compatibles","posible_contribucion","observacion"])
    _build_map(records,boundary,basins,output_paths["map"])
    try:
        import rasterio, geopandas, shapely, matplotlib
        versions={"python":platform.python_version(),"numpy":np.__version__,"rasterio":rasterio.__version__,"geopandas":geopandas.__version__,"shapely":shapely.__version__,"matplotlib":matplotlib.__version__}
    except Exception: versions={"python":platform.python_version(),"numpy":np.__version__}
    report={"id_ejecucion":run_id,"fecha_utc":datetime.now(timezone.utc).isoformat(),"caso_piloto":"Villa de Álvarez, Colima","dominio_presentacion":"municipal","dominio_soporte_hidrologico":"regional Colima-Jalisco, subcuencas RH15Aa/RH15Ab/RH16Ba/RH16Bb/RH16Bc y buffer de soporte referido de 2 km","crs_objetivo":config.target_crs,"resolucion_objetivo_m":[15,15],"rejilla_referencia":{"ruta":str(config.reference_grid),"disponible":reference is not None,"metadatos":reference},"mde_regional":{"ruta":str(config.regional_dem),"disponible":config.regional_dem.is_file()},"limite_municipal":boundary_info,"rutas_inspeccionadas":[str(p) for p in config.search_roots],"archivos_inspeccionados":len(records),"errores_lectura":errors,"factores":factors,"capas_vectoriales":layer_records,"archivos":records,"dependencias":versions,"metodologia":{"clasificacion_por_nombre":"heurística reproducible de nombres; inspeccionar asignaciones ambiguas","alineacion":"solo comprobada contra raster municipal de referencia cuando existe","cobertura_raster":"conteo por ventanas de celdas municipales rasterizadas con máscara válida","vectoriales":"conteo de entidades que intersectan municipio; no equivale a cobertura superficial","no_data":"se usa la máscara GDAL de cada raster","no_se_generan_factores":"sin reproyección, remuestreo, rasterización, reclasificación, pesos o susceptibilidad"},"notas_metodologicas":["Acumulación de flujo no equivale a caudal ni profundidad de inundación.","HAND es altura relativa al drenaje; no es por sí mismo una superficie de inundación.","Una resolución de 15 m no eleva la precisión temática de cartografía vectorial de escala más general.","Suelo, infiltración y uso/cobertura conservan las clases y atributos de origen; no se asignan valores inferidos.","Precipitación extrema/excedencias se inventaría como caracterización de peligro y se mantiene separada de los factores territoriales.","Elevación, pendiente, HAND, TWI y acumulación derivan de una misma superficie y pueden estar correlacionadas; revisar redundancia antes de modelar."],"productos":{k:str(v) for k,v in output_paths.items()},"duracion_s":time.perf_counter()-started}
    output_paths["json"].write_text(json.dumps(_json_safe(report),ensure_ascii=False,indent=2),encoding="utf-8")
    summary=["# Inventario y auditoría de factores territoriales", "",f"ID de ejecución: `{run_id}`", "", "## Alcance", "Villa de Álvarez es el dominio municipal de presentación; el dominio regional Colima–Jalisco se conserva como soporte hidrológico. Este reporte no genera susceptibilidad, amenaza ni riesgo.","", "## Estado de factores", "", "| Factor | Estado | Archivos |", "|---|---|---:|"]
    summary += [f"| {x['factor']} | {x['estado']} | {x['archivos']} |" for x in factors]
    summary += ["", "## Advertencias metodológicas", "", *[f"- {x}" for x in report["notas_metodologicas"]], "", "## Fuentes y limitaciones", "", *_boundary_summary_markdown(boundary_info), f"- Rejilla municipal de referencia disponible: `{reference is not None}`.", f"- Errores de lectura registrados: {len(errors)}.", "- La identificación temática se basa en nombres de archivos y requiere validar nombres ambiguos en los metadatos de fuente."]
    output_paths["markdown"].write_text("\n".join(summary)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root",action="append",type=Path,dest="roots")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-grid",type=Path,default=DEFAULT_REFERENCE_GRID)
    parser.add_argument("--regional-dem",type=Path,default=DEFAULT_REGIONAL_DEM)
    parser.add_argument("--boundary-gpkg",type=Path,default=DEFAULT_BOUNDARY_GPKG)
    parser.add_argument("--boundary-layer",default=DEFAULT_BOUNDARY_LAYER)
    parser.add_argument("--marco-dir",type=Path,default=DEFAULT_MARCO_DIR)
    parser.add_argument("--block-rows",type=int,default=256)
    parser.add_argument("--skip-hashes",action="store_true")
    args=parser.parse_args(argv)
    cfg=InventoryConfig(search_roots=tuple(args.roots) if args.roots else InventoryConfig().search_roots,output_dir=args.output_dir,reference_grid=args.reference_grid,regional_dem=args.regional_dem,boundary_gpkg=args.boundary_gpkg,boundary_layer=args.boundary_layer,marco_geoestadistico_dir=args.marco_dir,block_rows=args.block_rows,hash_files=not args.skip_hashes)
    print(json.dumps(_json_safe(run_inventory(cfg)),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
