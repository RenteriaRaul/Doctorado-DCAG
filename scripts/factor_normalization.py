"""Experimental normalization and categorical reclassification for territorial factors.

All raster operations preserve the existing grid and mask. This module does not
compute an integrated susceptibility index or alter source rasters.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import plotting_extent

WIDTH, HEIGHT, EPSG, RES, CELLS = 2674, 1552, 6368, 15.0, 1277051
NODATA_OUT = -9999.0
CAT_NODATA = 65535
CONTINUOUS = {
    "pendiente": {"source": "pendiente_grados_villa_de_alvarez.tif", "method": "linear_decreasing", "unit": "grados"},
    "distancia_cauces": {"source": "distancia_cauces_villa_de_alvarez.tif", "method": "linear_decreasing", "unit": "m"},
    "hand": {"source": "hand_villa_de_alvarez.tif", "method": "linear_decreasing", "unit": "m"},
    "acumulacion_flujo": {"source": "acumulacion_flujo_villa_de_alvarez.tif", "method": "log_increasing", "unit": "celdas contribuyentes"},
}
CONFIGS = {
    "A": {"pendiente": [0, 10], "distancia_cauces": [0, 100], "hand": [0, 5], "acumulacion_flujo": [1, 1000]},
    "B": {"pendiente": [0, 20], "distancia_cauces": [0, 300], "hand": [0, 15], "acumulacion_flujo": [1, 6035]},
}
LAND_GROUPS = {
    "AH": ("asentamientos_humanos", .90), "ADV": ("desprovisto_vegetacion", .80),
    "RA": ("agricultura", .60), "RAP": ("agricultura", .60), "RP": ("agricultura", .60),
    "RS": ("agricultura", .60), "TA": ("agricultura", .60), "TAP": ("agricultura", .60),
    "TAS": ("agricultura", .60), "TS": ("agricultura", .60), "TSP": ("agricultura", .60),
    "PC": ("pastizal_cultivado", .50),
    "VSa/BQ": ("vegetacion_secundaria_arbustiva", .40), "VSa/BQP": ("vegetacion_secundaria_arbustiva", .40),
    "VSa/SBC": ("vegetacion_secundaria_arbustiva", .40), "VSa/SMS": ("vegetacion_secundaria_arbustiva", .40),
    "VSA/SBC": ("vegetacion_secundaria_arborea", .30), "VSA/SMS": ("vegetacion_secundaria_arborea", .30),
    "BQ": ("bosques_y_selvas", .20), "BQP": ("bosques_y_selvas", .20), "SMS": ("bosques_y_selvas", .20),
}


def linear_decreasing(values, valid, a, b):
    _validate_bounds(a, b)
    x, m = np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
    if x.shape != m.shape or not np.isfinite(x[m]).all():
        raise ValueError("Valores válidos deben ser finitos y coincidir con la máscara.")
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[m] = np.clip((b - x[m]) / (b - a), 0, 1).astype(np.float32)
    return out


def log_increasing(values, valid, a, b):
    _validate_bounds(a, b)
    x, m = np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
    if x.shape != m.shape or not np.isfinite(x[m]).all() or np.any(x[m] < 0):
        raise ValueError("log1p requiere valores válidos finitos y no negativos.")
    den = np.log1p(b) - np.log1p(a)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[m] = np.clip((np.log1p(x[m]) - np.log1p(a)) / den, 0, 1).astype(np.float32)
    return out


def _validate_bounds(a, b):
    if not (math.isfinite(float(a)) and math.isfinite(float(b)) and float(a) >= 0 and float(b) > float(a)):
        raise ValueError("Los límites deben ser finitos, no negativos y b > a.")


def reclassify_categories(source, mapping, nodata=CAT_NODATA, unknown="error"):
    """Map integer source codes to float values; unknown codes fail closed."""
    src = np.asarray(source)
    valid = (src != nodata)
    present = set(np.unique(src[valid]).tolist())
    missing = present - set(mapping)
    if missing and unknown == "error":
        raise ValueError(f"Códigos sin correspondencia: {sorted(missing, key=str)}")
    out = np.full(src.shape, np.nan, dtype=np.float32)
    for code in present & set(mapping):
        value = float(mapping[code])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Valor categórico fuera de [0,1]: {code}={value}")
        out[valid & (src == code)] = value
    return out


def raster_stats(array, nodata=NODATA_OUT, cell_area=RES * RES):
    a = np.asarray(array, dtype=np.float32)
    x = a[np.isfinite(a) & (a != nodata)]
    if not x.size:
        raise ValueError("Producto sin celdas válidas.")
    q = np.percentile(x, [5, 25, 50, 75, 95])
    zero, one = int(np.count_nonzero(x == 0)), int(np.count_nonzero(x == 1))
    n = int(x.size)
    return {"n_validas": n, "minimo": float(x.min()), "maximo": float(x.max()), "media": float(x.mean()),
            "mediana": float(q[2]), "desviacion_estandar": float(x.std(ddof=1)) if n > 1 else 0.0,
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
            "ceros": zero, "ceros_pct": zero * 100 / n, "unos": one, "unos_pct": one * 100 / n,
            "intermedios_pct": (n-zero-one)*100/n, "superficie_valida_km2": n*cell_area/1e6}


def compare_arrays(arrays, threshold=.25):
    rows = []
    keys = list(arrays)
    for i, ka in enumerate(keys):
        for kb in keys[i+1:]:
            a, b = np.asarray(arrays[ka]), np.asarray(arrays[kb])
            valid = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))
            if not d.size:
                raise ValueError(f"Sin celdas coincidentes para comparar {ka} y {kb}.")
            rows.append({"configuracion_1": ka, "configuracion_2": kb, "n_comparadas": int(d.size),
                         "diferencia_absoluta_media": float(d.mean()), "rmse": float(np.sqrt(np.mean(d*d))),
                         "diferencia_mayor_0_25_pct": float(np.mean(d > threshold)*100)})
    return rows


def read_catalog(path, variable=None, landcover=False):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not landcover:
        if variable is None or "variable" not in df or "variable" not in df.columns:
            raise ValueError("El catálogo edafológico requiere una variable explícita.")
        df = df.loc[df.variable == variable].copy()
    if df.empty or "raster_code" not in df.columns:
        raise ValueError("Catálogo vacío o sin raster_code.")
    df["raster_code"] = df.raster_code.astype(int)
    if df.raster_code.duplicated().any():
        raise ValueError("raster_code duplicado dentro de la selección del catálogo.")
    return df


def land_cover_mapping(catalog):
    rows, mapping = [], {}
    for _, row in catalog.iterrows():
        code = row.CLAVE
        if code not in LAND_GROUPS:
            raise ValueError(f"Categoría de uso de suelo sin regla: {code!r}")
        group, score = LAND_GROUPS[code]
        mapping[int(row.raster_code)] = score
        rows.append({"raster_code": int(row.raster_code), "CLAVE": code, "DESCRIPCIO": row.DESCRIPCIO,
                     "grupo_reclasificado": group, "valor_experimental": score})
    if set(catalog.CLAVE) != set(LAND_GROUPS) or len(rows) != 21:
        raise ValueError("El catálogo no contiene exactamente las 21 categorías esperadas.")
    return mapping, rows


def texture_mappings(catalog):
    expected = {"1": .25, "2": .50, "3": .75}
    mapping = {}
    uncertain = set()
    for _, row in catalog.iterrows():
        source = row.source_value
        if source in expected:
            mapping[int(row.raster_code)] = expected[source]
        elif source == "N/A" and row.category_kind == "fuera_del_dominio_documentado":
            uncertain.add(int(row.raster_code))
        else:
            raise ValueError(f"Categoría de textura desconocida o semántica no esperada: {source!r}")
    if set(mapping.values()) != {.25, .5, .75} or len(uncertain) != 1:
        raise ValueError("El catálogo corregido debe contener textura 1/2/3 y N/A fuera de dominio.")
    return mapping, uncertain


def _valid_source(ds, data):
    valid = (ds.read_masks(1) > 0) & np.isfinite(data)
    if ds.nodata is not None:
        valid &= data != ds.nodata
    return valid


def _assert_grid(ds, ref):
    if (ds.width, ds.height, ds.crs, ds.transform) != (ref.width, ref.height, ref.crs, ref.transform):
        raise ValueError(f"Ráster fuera de la cuadrícula de referencia: {ds.name}")
    if (ds.width, ds.height, ds.crs.to_epsg(), ds.res) != (WIDTH, HEIGHT, EPSG, (RES, RES)):
        raise ValueError(f"Cuadrícula inesperada: {ds.name}")


def write_float_raster(path, values, ref, valid_domain, tags=None, expected_cells=CELLS):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"No se sobrescribe: {path}")
    a = np.asarray(values, dtype=np.float32)
    if a.shape != (ref.height, ref.width) or np.count_nonzero(valid_domain) != expected_cells:
        raise ValueError("Dimensión o máscara municipal inesperada.")
    valid = np.isfinite(a)
    if np.any(valid & ~valid_domain):
        raise ValueError("El producto contiene valores válidos fuera de la máscara municipal.")
    if not np.array_equal(valid, valid_domain):
        raise ValueError("La máscara válida del producto no coincide con la municipal.")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver":"GTiff", "width":ref.width, "height":ref.height, "count":1, "dtype":"float32",
               "crs":ref.crs, "transform":ref.transform, "nodata":NODATA_OUT, "compress":"lzw"}
    out = np.where(valid, a, NODATA_OUT).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(**(tags or {}))


def _write_plot_triptych(arrays, ref, titles, output, boundary=None, uncertainty=None):
    fig, axes = plt.subplots(1, len(arrays), figsize=(15, 6), constrained_layout=True)
    for ax, (name, arr) in zip(np.atleast_1d(axes), arrays.items()):
        image = ax.imshow(np.ma.masked_invalid(arr), extent=plotting_extent(ref), origin="upper", vmin=0, vmax=1, cmap="YlOrRd")
        if uncertainty is not None:
            ax.contour(uncertainty.astype(np.uint8), levels=[.5], colors=["#00ffff"], linewidths=.45,
                       extent=plotting_extent(ref), origin="upper")
        ax.set_title(titles.get(name, name)); ax.set_aspect("equal"); ax.set_axis_off()
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=.72, label="Pertenencia experimental (0–1)")
    fig.savefig(output, dpi=160); plt.close(fig)


def run(inputs, output_root, run_id=None):
    """Run the requested Villa de Álvarez audit. Inputs are explicit paths."""
    output_root = Path(output_root)
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("continuos", "textura", "uso_suelo", "figuras", "evaluacion"):
        (out/sub).mkdir()
    params = json.loads(Path(inputs["parameters"]).read_text(encoding="utf-8"))
    with rasterio.open(inputs["reference"]) as ref:
        if (ref.width, ref.height, ref.crs.to_epsg(), ref.res) != (WIDTH, HEIGHT, EPSG, (RES, RES)):
            raise ValueError("Referencia incompatible con 2674×1552, EPSG:6368 y 15 m.")
        template = ref.read(1)
        domain = _valid_source(ref, template)
        if int(domain.sum()) != CELLS:
            raise ValueError(f"Máscara municipal: {int(domain.sum())}, esperadas {CELLS}.")
        ref_meta = ref
        all_stats, comparisons, outputs = [], [], {}
        cont_arrays = {}
        stat_df = pd.read_csv(inputs["percentiles"])
        stat_df = stat_df.set_index("factor")
        percent_rows = {}
        for factor in CONTINUOUS:
            if factor not in stat_df.index:
                raise ValueError(f"Falta percentil existente para {factor}.")
            percent_rows[factor] = [float(stat_df.loc[factor, "p05"]), float(stat_df.loc[factor, "p95"])]
            if percent_rows[factor][0] < 0 or percent_rows[factor][1] <= percent_rows[factor][0]:
                raise ValueError(f"Percentiles inválidos en CSV para {factor}.")
        configs = json.loads(json.dumps(params["configuraciones"]))
        for factor in CONTINUOUS:
            configs["C"][factor] = percent_rows[factor]
        params["configuraciones"] = configs
        params["percentiles_fuente"] = str(inputs["percentiles"])
        params["percentiles_C"] = percent_rows
        (out/"parametros_normalizacion.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        for factor, spec in CONTINUOUS.items():
            with rasterio.open(inputs[factor]) as src:
                _assert_grid(src, ref); raw = src.read(1); valid = _valid_source(src, raw)
                if not np.array_equal(valid, domain):
                    raise ValueError(f"Máscara distinta en factor continuo {factor}.")
                variants = {}
                for config in "ABC":
                    a, b = configs[config][factor]
                    arr = linear_decreasing(raw, valid, a, b) if spec["method"] == "linear_decreasing" else log_increasing(raw, valid, a, b)
                    name = f"{factor}_{config}.tif"; p = out/"continuos"/name
                    write_float_raster(p, arr, ref, domain, {"factor":factor,"configuracion":config,"metodo":spec["method"],"limite_a":a,"limite_b":b,"unidad":spec["unit"],"experimental":"true"})
                    variants[config] = arr; cont_arrays[f"{factor}_{config}"] = arr; outputs[name] = str(p)
                    all_stats.append({"producto":name,"factor":factor,"configuracion":config,**raster_stats(arr)})
                comparisons.extend([{"factor":factor,**r} for r in compare_arrays(variants)])
                _write_plot_triptych(variants, ref, {k:f"{factor} {k}" for k in variants}, out/"figuras"/f"comparacion_{factor}.png")
        # Texture: interpret only the variable-specific corrected catalog.
        tcat = read_catalog(inputs["texture_catalog"], variable="textura_superficial")
        known, uncertain_codes = texture_mappings(tcat)
        with rasterio.open(inputs["texture"]) as ds:
            _assert_grid(ds, ref); src = ds.read(1); valid = _valid_source(ds, src)
            if not np.array_equal(valid, domain): raise ValueError("Máscara distinta en textura.")
        present = set(np.unique(src[valid]).tolist())
        if present - set(known) - uncertain_codes: raise ValueError(f"Códigos de textura adicionales: {present-set(known)-uncertain_codes}")
        uncertainty = np.isin(src, list(uncertain_codes)) & valid
        texture_arrays, texture_rows = {}, []
        for cfg, fill in (("T1",.25),("T2",.50),("T3",.75)):
            mapping = dict(known); mapping.update({code:fill for code in uncertain_codes})
            arr = reclassify_categories(src, mapping)
            if not np.array_equal(np.isfinite(arr), domain): raise ValueError("Máscara reclasificada textura divergente.")
            name=f"textura_{cfg}.tif"; p=out/"textura"/name
            write_float_raster(p,arr,ref,domain,{"factor":"textura_superficial","configuracion":cfg,"N_A":"imputacion experimental enmascarada por incertidumbre"})
            texture_arrays[cfg]=arr; outputs[name]=str(p); all_stats.append({"producto":name,"factor":"textura_superficial","configuracion":cfg,**raster_stats(arr)})
            texture_rows.append({"configuracion":cfg,"valor_N_A":fill,"celdas_inciertas":int(uncertainty.sum()),"superficie_incierta_km2":float(uncertainty.sum()*RES*RES/1e6)})
        uncertainty_arr=np.full(src.shape,np.nan,dtype=np.float32); uncertainty_arr[domain]=0; uncertainty_arr[uncertainty]=1
        up=out/"textura"/"mascara_incertidumbre_edafologica.tif"; write_float_raster(up,uncertainty_arr,ref,domain,{"meaning":"1=codigo fuera de dominio documentado; 0=conocido"}); outputs[up.name]=str(up)
        comparisons.extend([{"factor":"textura_superficial",**r} for r in compare_arrays(texture_arrays)])
        # Original category counts and areas; no category is removed.
        texture_catalog_by_code={int(r.raster_code):r for _,r in tcat.iterrows()}
        texture_rows_cat=[]
        for code in sorted(present):
            row=texture_catalog_by_code[code]; source_value=row.source_value
            known_value=known.get(code, np.nan); n=int(np.count_nonzero(src[valid]==code))
            texture_rows_cat.append({"raster_code":code,"source_value":source_value,"categoria_inegi":row.inegi_category_name,
                "valor_conocido":known_value,"valor_T1":known_value if code not in uncertain_codes else .25,
                "valor_T2":known_value if code not in uncertain_codes else .50,"valor_T3":known_value if code not in uncertain_codes else .75,
                "celdas":n,"superficie_km2":n*RES*RES/1e6,"porcentaje_municipal":n*100/CELLS,"incierta":code in uncertain_codes})
        pd.DataFrame(texture_rows_cat).to_csv(out/"textura"/"reclasificacion_textura.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(texture_rows).to_csv(out/"evaluacion"/"resumen_incertidumbre_edafologica.csv",index=False,encoding="utf-8-sig")
        _write_plot_triptych(texture_arrays,ref,{k:f"Textura {k} (contorno cian = incertidumbre)" for k in texture_arrays},out/"figuras"/"comparacion_textura.png",uncertainty=uncertainty)
        # Land cover mapping from the official product catalog.
        lcat=read_catalog(inputs["landcover_catalog"],landcover=True); lmapping,lrows=land_cover_mapping(lcat)
        with rasterio.open(inputs["landcover"]) as ds:
            _assert_grid(ds,ref); lsrc=ds.read(1); lvalid=_valid_source(ds,lsrc)
            if not np.array_equal(lvalid,domain): raise ValueError("Máscara distinta en uso de suelo.")
        larr=reclassify_categories(lsrc,lmapping)
        if not np.array_equal(np.isfinite(larr),domain): raise ValueError("Máscara reclasificada de uso divergente.")
        lp=out/"uso_suelo"/"uso_suelo_reclasificado.tif"; write_float_raster(lp,larr,ref,domain,{"factor":"uso_suelo_vegetacion","hipotesis":"ordinal de generacion relativa de escurrimiento; no coeficiente medido"}); outputs[lp.name]=str(lp)
        all_stats.append({"producto":lp.name,"factor":"uso_suelo_vegetacion","configuracion":"unica",**raster_stats(larr)})
        land_rows=[]
        for row in lrows:
            code=row["raster_code"]; n=int(np.count_nonzero(lsrc[lvalid]==code)); land_rows.append({**row,"celdas":n,"superficie_km2":n*RES*RES/1e6,"porcentaje_municipal":n*100/CELLS})
        pd.DataFrame(land_rows).to_csv(out/"uso_suelo"/"reclasificacion_uso_suelo.csv",index=False,encoding="utf-8-sig")
        # Plots: transfer plot; distributions panel; texture uncertainty indicator.
        fig,ax=plt.subplots(figsize=(8,5))
        x=np.linspace(0,1,300)
        for f in CONTINUOUS:
            for c in "ABC":
                a,b=configs[c][f]; z=(np.clip((b-x*(b-a)-a)/(b-a),0,1) if CONTINUOUS[f]["method"]=="linear_decreasing" else np.clip((np.log1p(a+x*(b-a))-np.log1p(a))/(np.log1p(b)-np.log1p(a)),0,1))
                ax.plot(x,z,label=f"{f}-{c}",alpha=.65)
        ax.set(xlabel="Valor reescalado dentro del intervalo de cada configuración",ylabel="Pertenencia normalizada",ylim=(-.03,1.03)); ax.legend(ncol=3,fontsize=7); fig.tight_layout(); fig.savefig(out/"figuras"/"funciones_normalizacion.png",dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(np.ma.masked_invalid(larr),extent=plotting_extent(ref),origin="upper",vmin=0,vmax=1,cmap="YlOrRd"); ax.set_title("Uso de suelo y vegetación reclasificado"); ax.set_axis_off(); fig.colorbar(ax.images[0],ax=ax,label="Valor ordinal experimental"); fig.tight_layout(); fig.savefig(out/"figuras"/"uso_suelo_reclasificado.png",dpi=150); plt.close(fig)
        fig,axes=plt.subplots(4,4,figsize=(14,10));
        for ax,(name,arr) in zip(axes.flat,list(cont_arrays.items())+[(f"textura_{k}",v) for k,v in texture_arrays.items()]+[("uso_suelo",larr)]):
            ax.hist(arr[domain],bins=30,range=(0,1),density=True,color="#287c8e"); ax.set_title(name,fontsize=8)
        for ax in axes.flat[len(cont_arrays)+4:]: ax.axis("off")
        fig.tight_layout(); fig.savefig(out/"figuras"/"distribuciones_factores_normalizados.png",dpi=150); plt.close(fig)
        pd.DataFrame(all_stats).to_csv(out/"evaluacion"/"estadisticas_normalizacion.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(comparisons).to_csv(out/"evaluacion"/"comparacion_configuraciones.csv",index=False,encoding="utf-8-sig")
        # Copy canonical tables to requested run-root filenames for convenient delivery.
        pd.DataFrame(all_stats).to_csv(out/"estadisticas_normalizacion.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(comparisons).to_csv(out/"comparacion_configuraciones.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(texture_rows_cat).to_csv(out/"reclasificacion_textura.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(land_rows).to_csv(out/"reclasificacion_uso_suelo.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(texture_rows).to_csv(out/"resumen_incertidumbre_edafologica.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(all_stats).to_csv(out/"evaluacion"/"evaluacion_conjunta_factores.csv",index=False,encoding="utf-8-sig")
        manifest={"run_id":run_id,"created_utc":datetime.now(timezone.utc).isoformat(),"grid":{"crs":"EPSG:6368","width":WIDTH,"height":HEIGHT,"resolution_m":RES,"municipal_cells":CELLS},"inputs":{k:str(v) for k,v in inputs.items()},"products":outputs,"n_products_raster":17,"notes":["16 factores normalizados más una máscara binaria de incertidumbre edafológica.","Configuraciones experimentales, no calibradas; no son probabilidades.","Uso de suelo Serie VII escala 1:250000; rasterizar a 15 m no incrementa precisión temática.","No se construyó índice integrado."]}
        (out/"manifiesto_normalizacion.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        stats_df=pd.DataFrame(all_stats)
        sat=stats_df.loc[stats_df.factor.isin(CONTINUOUS),["factor","configuracion","ceros_pct","unos_pct","intermedios_pct","desviacion_estandar"]]
        cmp_df=pd.DataFrame(comparisons)
        strongest=cmp_df.sort_values("rmse",ascending=False).head(5) if not cmp_df.empty else cmp_df
        report=["# Etapa 4.3 — Normalización y reclasificación", "",f"Ejecución: `{run_id}`.","",f"Máscara municipal: {int(domain.sum()):,} celdas (100 % de las celdas válidas esperadas).", "", "Las configuraciones son experimentales y no calibradas; los valores no representan probabilidades. No se combinaron fuentes ni factores en un índice.","", "## Parámetros efectivos", "", "A y B utilizan intervalos experimentales especificados; C usa P05–P95 del CSV de estadística descriptiva existente (sin recalcularlo). Para acumulación, C=P05–P95=1–160 celdas; B=1–6035.","", "```json",json.dumps(configs,ensure_ascii=False,indent=2),"```", "", "## Cobertura, saturación y diferenciación", "", "Las 16 salidas normalizadas conservaron 1,277,051 celdas válidas. Consulte `estadisticas_normalizacion.csv` para mínimo, máximo, media, mediana, dispersión y percentiles por producto.","", "Saturación y dispersión de los cuatro factores continuos:","", sat.to_string(index=False) if not sat.empty else "Sin datos.","", "Pares con mayor diferencia RMSE (sensibilidad paramétrica; no son errores frente a observaciones):", "", strongest.to_string(index=False) if not strongest.empty else "Sin comparaciones.","",f"Textura fuera del dominio documentado: {int(uncertainty.sum()):,} celdas ({uncertainty.sum()*100/CELLS:.3f} %; {uncertainty.sum()*RES*RES/1e6:.3f} km²). Las clases conocidas permanecen constantes en T1/T2/T3. T2 es referencia neutral, no textura observada.","", "## Correspondencia temática y límites", "", "La textura se interpretó con el catálogo semántico corregido y códigos originales INEGI 1=gruesa, 2=media, 3=fina; N/A queda señalada en la máscara de incertidumbre y recibe valores de sensibilidad T1/T2/T3.","", "Las 21 categorías de Serie VII se mapearon de forma exhaustiva y sensible a mayúsculas/minúsculas; la tabla conserva el código y la descripción originales junto al grupo y valor experimental. La fuente temática tiene escala 1:250 000; rasterizar a 15 m no incrementa precisión cartográfica.","", "Los valores de uso de suelo expresan una hipótesis ordinal de generación relativa de escurrimiento, no coeficientes físicos medidos. Pendiente, acumulación y uso del suelo representan generación/transferencia; HAND y distancia a cauces describen además posición o proximidad receptora. Ninguno equivale por sí solo a una observación de inundación.","", "No se selecciona automáticamente una configuración ganadora. La selección posterior debe considerar objetivo conceptual, revisión de saturación y sensibilidad, incertidumbre de textura y resolución temática, sin interpretar estos valores como probabilidades.","", "## Entregables", "", "16 GeoTIFF normalizados, una máscara de incertidumbre, ocho figuras PNG, tablas de estadística, comparación y correspondencia, parámetros y manifiesto. No se recalcularon factores originales, correlaciones ni VIF; no se construyó índice integrado."]
        (out/"informe_etapa_4_3.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return out

