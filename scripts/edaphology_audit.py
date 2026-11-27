from __future__ import annotations
import json, math, os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import unary_union


def attribute_stats(values):
    s = pd.Series(values, dtype='object')
    nulls = s.isna() | s.map(lambda x: isinstance(x, str) and x.strip() == '')
    valid = s[~nulls]
    numeric = pd.to_numeric(valid, errors='coerce')
    out = {'records': int(len(s)), 'nulls': int(nulls.sum()), 'null_pct': float(nulls.mean()*100 if len(s) else 0), 'unique': int(valid.nunique(dropna=True))}
    if len(valid) and numeric.notna().sum() == len(valid):
        out.update({'min': float(numeric.min()), 'max': float(numeric.max()), 'mean': float(numeric.mean()), 'std': float(numeric.std(ddof=1)) if len(numeric)>1 else 0.0})
    else:
        out.update({'min': None, 'max': None, 'mean': None, 'std': None})
    return out, Counter(valid.map(str))


def coverage_pct(source_geoms, domain):
    denom = domain.area
    if denom <= 0: raise ValueError('El dominio debe tener área positiva')
    clipped = [g.intersection(domain) for g in source_geoms if g is not None and not g.is_empty and g.intersects(domain)]
    covered = unary_union(clipped).area if clipped else 0.0
    return {'covered_area_m2': float(covered), 'domain_area_m2': float(denom), 'coverage_pct': float(covered/denom*100)}


def geometry_metrics(geoms):
    n = 0; null = 0; invalid = 0; types=Counter(); bounds=[]
    for g in geoms:
        n += 1
        if g is None or g.is_empty: null += 1; continue
        types[g.geom_type] += 1
        invalid += int(not g.is_valid)
        bounds.append(g.bounds)
    extent = [min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds)] if bounds else None
    return {'features':n,'null_or_empty':null,'invalid':invalid,'geometry_types':dict(types),'extent':extent}


def run_audit(area_path, pto_path, gpkg, output_dir, metadata_files=()):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    boundary = gpd.read_file(gpkg,layer='limite_municipal')
    municipio = boundary.geometry.union_all()
    target_crs = boundary.crs
    sub = gpd.read_file(gpkg,layer='subcuencas_regionales')
    codes={'RH15Aa','RH15Ab','RH16Ba','RH16Bb','RH16Bc'}
    regional = unary_union(list(sub.loc[sub['CVE_SUBCUE'].isin(codes)].geometry))
    regional = regional.buffer(2000)
    outputs=[]; freq=[]; layers=[]; geom_by_layer={}
    mapped={
      ('area','Grupo1'):('Suelo dominante','Utilizable directamente como información temática','Grupo dominante según el esquema de unidad de suelo'),
      ('area','Grupo2'):('Suelo secundario','Utilizable directamente como información temática','Grupo secundario'),
      ('area','Grupo3'):('Suelo terciario','Utilizable directamente como información temática','Grupo terciario'),
      ('area','Califp_g1'):('Calificador primario dominante','Utilizable directamente como información temática','Cualidad dominante del grupo principal'),
      ('area','Califs_g1'):('Calificador secundario dominante','Utilizable directamente como información temática','Segunda cualidad del grupo principal'),
      ('area','Califp_g2'):('Calificador primario secundario','Utilizable directamente como información temática','Cualidad del grupo secundario'),
      ('area','Califs_g2'):('Calificador secundario secundario','Utilizable directamente como información temática','Segunda cualidad del grupo secundario'),
      ('area','Califp_g3'):('Calificador primario terciario','Utilizable directamente como información temática','Cualidad del grupo terciario'),
      ('area','Califs_g3'):('Calificador secundario terciario','Utilizable directamente como información temática','Segunda cualidad del grupo terciario'),
      ('area','Clase_tex'):('Textura del suelo','Utilizable directamente como información temática','Textura de los 30 cm superficiales del suelo dominante'),
      ('area','Lmte_sup'):('Limitante física superficial','Utilizable directamente como información temática','Limitante física del suelo dominante'),
      ('area','Fase_fis_u'):('Fase física','Utilizable directamente como información temática','Fase física de unidad de suelo'),
      ('area','Fase_qui_u'):('Fase química','Utilizable directamente como información temática','Fase química de unidad de suelo'),
      ('pto','Infiltra'):('Infiltración','Utilizable después de procesamiento o validación','Categoría cualitativa de velocidad dentro del perfil; no convertir a tasa numérica'),
      ('pto','Escurr_sup'):('Escurrimiento superficial','Utilizable después de procesamiento o validación','Categoría cualitativa de velocidad superficial; no convertir a tasa numérica'),
      ('pto','DA'):('Densidad aparente','Utilizable después de procesamiento o validación','Medición de laboratorio por horizonte; validar unidades y cobertura'),
      ('pto','R'):('Arena','Utilizable después de procesamiento o validación','Fracción analítica del horizonte; revisar metadatos y dominio'),
      ('pto','L'):('Limo','Utilizable después de procesamiento o validación','Fracción analítica del horizonte; revisar metadatos y dominio'),
      ('pto','A'):('Arcilla','Utilizable después de procesamiento o validación','Fracción analítica del horizonte; revisar metadatos y dominio'),
      ('pto','Clase_text'):('Textura de laboratorio','Utilizable después de procesamiento o validación','Clase textural por horizonte'),
      ('pto','Limite_sup'):('Límite superior de horizonte','Complementario','Profundidad de horizonte; confirmar unidades'),
      ('pto','Limite_inf'):('Límite inferior de horizonte','Complementario','Profundidad de horizonte; confirmar unidades'),
      ('pto','Numero_Hte'):('Número de horizonte','Complementario','Secuencia de horizonte'),
      ('pto','Id_Hte'):('Identificador de horizonte','Complementario','Identificador relacional'),
      ('pto','Id_Sitio'):('Identificador de sitio','Complementario','Trazabilidad del perfil')}
    for kind,path in [('area',Path(area_path)),('pto',Path(pto_path))]:
        info=pyogrio.read_info(path)
        fields=list(info['fields']); dtypes=list(info['dtypes']); n=int(info['features']); all_values={f:[] for f in fields}; geoms=[]
        mun_src = gpd.GeoSeries([municipio],crs=target_crs).to_crs(info['crs']).iloc[0]
        reg_src = gpd.GeoSeries([regional],crs=target_crs).to_crs(info['crs']).iloc[0]
        for start in range(0,n,5000):
            chunk=pyogrio.read_dataframe(path,skip_features=start,max_features=min(5000,n-start))
            geoms.extend(chunk.geometry.tolist())
            for f in fields: all_values[f].extend(chunk[f].tolist())
        gm=geometry_metrics(geoms); geom_by_layer[kind]=geoms
        mun_flags=[g is not None and not g.is_empty and g.intersects(mun_src) for g in geoms]
        reg_flags=[g is not None and not g.is_empty and g.intersects(reg_src) for g in geoms]
        layer={'source_path':str(path),'layer':info['layer_name'],'format':info['driver'],'geometry':info['geometry_type'],'crs':info['crs'],'feature_count':n,'extent':list(info['total_bounds']),'geometry_audit':gm,'fields':fields,'file_size_bytes':path.stat().st_size}
        if kind=='area':
            layer['municipal_coverage']=coverage_pct(geoms,mun_src)
            layer['regional_coverage']=coverage_pct(geoms,reg_src)
            layer['municipal_intersecting_features']=sum(g is not None and g.intersects(mun_src) for g in geoms)
            layer['regional_intersecting_features']=sum(g is not None and g.intersects(reg_src) for g in geoms)
        else:
            layer['points_municipality']=sum(g is not None and mun_src.covers(g) for g in geoms)
            layer['points_regional_support']=sum(g is not None and reg_src.covers(g) for g in geoms)
            layer['municipality_point_coordinates']=[[float(g.x),float(g.y)] for g in geoms if g is not None and mun_src.covers(g)]
        for f,dt in zip(fields,dtypes):
            st,counts=attribute_stats(all_values[f])
            meaning=mapped.get((kind,f),('Sin equivalencia específica verificada','Complementario','Consultar el diccionario; no asignar significado por similitud nominal'))
            ms,_=attribute_stats([v for v,flag in zip(all_values[f],mun_flags) if flag])
            rs,_=attribute_stats([v for v,flag in zip(all_values[f],reg_flags) if flag])
            outputs.append({'layer':kind,'field':f,'dtype':str(dt),'dictionary_concept':meaning[0],'preliminary_use':meaning[1],'definition_note':meaning[2],**st,'municipal_records':ms['records'],'municipal_nulls':ms['nulls'],'municipal_null_pct':ms['null_pct'],'regional_records':rs['records'],'regional_nulls':rs['nulls'],'regional_null_pct':rs['null_pct']})
            for val,count in counts.items(): freq.append({'layer':kind,'field':f,'value':val,'count':int(count),'pct_of_records':float(count/n*100 if n else 0)})
        layers.append(layer)
    pd.DataFrame(outputs).to_csv(output_dir/'diccionario_atributos.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(freq).to_csv(output_dir/'frecuencias_atributos.csv',index=False,encoding='utf-8-sig')
    # Capa regional explícitamente operativa: subcuencas enumeradas en inventario más buffer de soporte de 2 km.
    inv={'created_utc':datetime.now(timezone.utc).isoformat(),'source_files':[str(Path(area_path)),str(Path(pto_path))], 'metadata_files':[str(Path(p)) for p in metadata_files], 'regional_domain_definition':'union RH15Aa/RH15Ab/RH16Ba/RH16Bb/RH16Bc from subcuencas_regionales, buffered 2 km, per repository inventory','municipal_reference':str(gpkg),'layers':layers}
    (output_dir/'inventario_edafologico.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf-8')
    cov={'municipality_area_m2':float(municipio.area),'regional_support_area_m2':float(regional.area),'area_coverage':layers[0].get('municipal_coverage'),'regional_coverage':layers[0].get('regional_coverage'),'total_profiles':layers[1]['feature_count'],'profiles_inside_municipality':layers[1]['points_municipality'],'profiles_in_regional_support':layers[1]['points_regional_support'],'profile_points_municipality_xy':layers[1]['municipality_point_coordinates'],'note':'Coverage uses union of intersected unit polygons, not summed feature area; regional support is hydrologic subbasins plus 2 km buffer.'}
    (output_dir/'cobertura_edafologica.json').write_text(json.dumps(cov,ensure_ascii=False,indent=2),encoding='utf-8')
    # Map only layers, no reprojection or source edits; CRS of the project layers is EPSG:6368.
    area=gpd.read_file(area_path,mask=gpd.GeoSeries([municipio],crs=target_crs).to_crs(pyogrio.read_info(area_path)['crs']).iloc[0]).to_crs(target_crs)
    area=gpd.clip(area,boundary.geometry.union_all())
    search=municipio.buffer(15000)
    pto=gpd.read_file(pto_path,mask=gpd.GeoSeries([search],crs=target_crs).to_crs(pyogrio.read_info(pto_path)['crs']).iloc[0]).to_crs(target_crs)
    fig,ax=plt.subplots(figsize=(10,9)); area.plot(ax=ax,column='Grupo1',legend=True,categorical=True,cmap='tab20',linewidth=.2,edgecolor='white',legend_kwds={'title':'Grupo de suelo','loc':'upper left','bbox_to_anchor':(1.02,1)}); gpd.GeoSeries([municipio],crs=target_crs).boundary.plot(ax=ax,color='black',linewidth=1.2); pto.plot(ax=ax,color='#161616',markersize=11,label='Perfiles (radio 15 km)'); minx,miny,maxx,maxy=municipio.bounds; ax.set_xlim(minx-15000,maxx+15000); ax.set_ylim(miny-15000,maxy+15000); ax.set_title('Unidades de suelo y perfiles — Villa de Álvarez'); ax.set_axis_off(); ax.legend(loc='lower left'); fig.tight_layout(); fig.savefig(output_dir/'mapa_edafologia.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    summary=f'''# Diagnóstico edafológico — Villa de Álvarez

## Fuentes e integridad de lectura

INEGI, Conjunto Nacional de Información Edafológica, escala 1:250 000, Serie III. Metadatos locales: periodo 2013–2020; clasificación WRB 2014; estructura del Diccionario Edafológico versión 4. Se auditaron `area` y `pto`; no se localizó capa `otro rasgo`. Fuentes sin modificación en disco. Las referencias espaciales se transformaron solo en memoria para intersecciones.

## Cobertura

- Polígonos `area`: {layers[0]['feature_count']:,}; inválidos: {layers[0]['geometry_audit']['invalid']:,}; nulos/vacíos: {layers[0]['geometry_audit']['null_or_empty']:,}.
- Cobertura municipal: {cov['area_coverage']['coverage_pct']:.3f}%.
- Cobertura del soporte hidrológico regional: {cov['regional_coverage']['coverage_pct']:.3f}%.
- Perfiles `pto`: {cov['total_profiles']:,}; dentro del municipio: {cov['profiles_inside_municipality']:,}; en soporte regional: {cov['profiles_in_regional_support']:,}.

El soporte regional corresponde a la unión de RH15Aa, RH15Ab, RH16Ba, RH16Bb y RH16Bc más buffer de 2 km según el inventario territorial; no es un límite político estatal completo.

## Correspondencia y selección preliminar

Las definiciones de campos mapeados y estadísticas completas están en `diccionario_atributos.csv`; frecuencias en `frecuencias_atributos.csv`. Los grupos, calificadores, textura, limitante superficial y fases de `area` son información categórica temática. En `pto`, infiltración y escurrimiento son categorías cualitativas de velocidad y no se convierten en tasas. Densidad aparente, fracciones R/L/A y clase textural son candidatos condicionados a validar unidades, faltantes y asociación por horizonte. Los demás campos se conservan como complementarios sin inferir equivalencias.

La cobertura poligonal se calcula con la unión geométrica de las intersecciones, no con suma de áreas; perfiles son observaciones puntuales y no una cobertura continua.

## Limitaciones y recomendación

La escala 1:250 000 es regional y no implica precisión parcelaria ni de 15 m. La densidad de perfiles limita su extrapolación local. La auditoría no asigna valores hidrológicos, ponderaciones o clases inferidas. No se encontró la copia en el árbol territorial de G: inspeccionado; el análisis usa la fuente local bajo `C:\\Doctorado-DCAG\\datos_territoriales`. Integrar unidades como contexto categórico; validar datos de perfiles antes de derivar parámetros. La rasterización y cualquier conversión hidrológica requieren método y validación documentados.
'''
    (output_dir/'diagnostico_edafologico.md').write_text(summary,encoding='utf-8')
    return inv,cov

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--area',required=True); p.add_argument('--pto',required=True); p.add_argument('--gpkg',required=True); p.add_argument('--output',required=True); p.add_argument('--metadata',nargs='*',default=[]); a=p.parse_args()
    run_audit(a.area,a.pto,a.gpkg,a.output,a.metadata)

