"""Estadística descriptiva y redundancia de 13 factores territoriales."""
from __future__ import annotations
import argparse,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from scipy.stats import rankdata,skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CONT=["elevacion","pendiente","distancia_cauces","acumulacion_flujo","hand","twi"]
CATS=["suelo_dominante","suelo_secundario","textura_superficial","limitante_fisica","fase_fisica","fase_quimica","uso_suelo_vegetacion"]
PHYS=dict(zip(CONT,["elevacion_villa_de_alvarez.tif","pendiente_grados_villa_de_alvarez.tif","distancia_cauces_villa_de_alvarez.tif","acumulacion_flujo_villa_de_alvarez.tif","hand_villa_de_alvarez.tif","twi_villa_de_alvarez.tif"]))
EDAPH={n:f"{n}_villa_de_alvarez.tif" for n in CATS[:6]}; NODATA=65535; SHAPE=(1552,2674)

def create_run_dir(root,run_id=None):
    d=Path(root)/(run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")); d.mkdir(parents=True,exist_ok=False); (d/"tablas_contingencia").mkdir(); return d

def read_stack(cp,kp,expected_shape=SHAPE,epsg=6368,resolution_m=15,expected_mask_cells=1277051):
    arrays={}; masks={}; meta={}; transform=crs=domain=None
    for n,p in {**cp,**kp}.items():
        with rasterio.open(p) as ds:
            if ds.crs is None or ds.crs.to_epsg()!=epsg or (ds.height,ds.width)!=tuple(expected_shape) or tuple(ds.res)!=(float(resolution_m),float(resolution_m)): raise ValueError(f"Metadatos de cuadrícula incompatibles: {p}")
            if transform is None: transform,crs=ds.transform,ds.crs
            elif transform!=ds.transform or crs!=ds.crs: raise ValueError(f"Ráster desalineado: {p}")
            x=ds.read(1); m=(ds.read_masks(1)>0)&np.isfinite(x); arrays[n]=x.astype(float if n in CONT else np.int64); masks[n]=m; meta[n]={"path":str(p),"dtype":ds.dtypes[0],"nodata":ds.nodata}
            if domain is None: domain=m.copy()
    if expected_mask_cells is not None and int(domain.sum())!=int(expected_mask_cells): raise ValueError(f"Máscara base inesperada: {int(domain.sum())} celdas")
    common=np.logical_and.reduce(list(masks.values()))
    grid={"crs":f"EPSG:{epsg}","dimensions":[int(expected_shape[1]),int(expected_shape[0])],"resolution_m":float(resolution_m),"transform":list(transform),"municipal_cells":int(domain.sum()),"joint_valid_cells":int(common.sum()),"joint_mask_matches_domain":bool(np.array_equal(common,domain))}
    return arrays,masks,domain,common,grid,meta

def descriptive_statistics(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): raise ValueError("Sin datos válidos")
    q=np.percentile(x,[1,5,25,50,75,95,99]); z=int(np.count_nonzero(x==0))
    return {"n_validos":len(x),"minimo":float(x.min()),"maximo":float(x.max()),"media":float(x.mean()),"mediana":float(q[3]),"desviacion_estandar":float(x.std(ddof=1)) if len(x)>1 else 0.,
    "p01":float(q[0]),"p05":float(q[1]),"p25":float(q[2]),"p50":float(q[3]),"p75":float(q[4]),"p95":float(q[5]),"p99":float(q[6]),"asimetria":float(skew(x,bias=False)) if len(x)>2 else np.nan,
    "rango_intercuartilico":float(q[4]-q[2]),"ceros":z,"ceros_pct":100*z/len(x)}

def correlations(a,m,names):
    p=np.full((len(names),len(names)),np.nan); s=p.copy(); counts=np.zeros(p.shape,int)
    for i,xn in enumerate(names):
        for j in range(i,len(names)):
            yn=names[j]; ok=m[xn]&m[yn]&np.isfinite(a[xn])&np.isfinite(a[yn]); x,y=a[xn][ok],a[yn][ok]; counts[i,j]=counts[j,i]=len(x)
            if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0:
                p[i,j]=p[j,i]=np.corrcoef(x,y)[0,1]; s[i,j]=s[j,i]=np.corrcoef(rankdata(x),rankdata(y))[0,1]
    return pd.DataFrame(p,index=names,columns=names),pd.DataFrame(s,index=names,columns=names),counts

def vif_tolerance(a,names,mask):
    x=np.column_stack([a[n][mask].astype(float) for n in names]); c=x-x.mean(0); sd=c.std(0); const=sd<=np.finfo(float).eps*np.maximum(1,np.abs(x).max(0)); idx=np.flatnonzero(~const); corr=np.full((len(names),len(names)),np.nan)
    if len(idx):
        z=c[:,idx]/sd[idx]; corr[np.ix_(idx,idx)]=z.T@z/len(z)
    rows=[]
    for i,n in enumerate(names):
        if const[i]: rows.append({"factor":n,"vif":np.nan,"tolerancia":np.nan,"estado":"predictor_constante"}); continue
        other=[j for j in idx if j!=i]
        r2=0 if not other else float(corr[other,i]@np.linalg.pinv(corr[np.ix_(other,other)],hermitian=True)@corr[other,i]); tol=max(0.,1-min(1.,max(0.,r2)))
        rows.append({"factor":n,"vif":np.inf if tol<=1e-12 else 1/tol,"tolerancia":tol,"estado":"colinealidad_perfecta" if tol<=1e-12 else "ok" if tol>1e-8 else "colinealidad_muy_alta"})
    return pd.DataFrame(rows)

def categorical_association(x,y,nodata=NODATA):
    x,y=np.asarray(x),np.asarray(y); valid=(x!=nodata)&(y!=nodata); x,y=x[valid],y[valid]
    if not len(x): return {"n":0,"cramers_v":np.nan,"nmi":np.nan,"estado":"sin_datos"},None
    ux,ix=np.unique(x,return_inverse=True); uy,iy=np.unique(y,return_inverse=True); t=np.zeros((len(ux),len(uy)),int); np.add.at(t,(ix,iy),1)
    if len(ux)<2 or len(uy)<2: return {"n":len(x),"cramers_v":np.nan,"nmi":np.nan,"estado":"variable_constante"},(ux,uy,t)
    n=float(t.sum()); e=np.outer(t.sum(1),t.sum(0))/n; chi=np.sum((t-e)**2/e); r,k=t.shape; phi=chi/n
    phic=max(0.,phi-(k-1)*(r-1)/max(1.,n-1)); rc=r-(r-1)**2/max(1.,n-1); kc=k-(k-1)**2/max(1.,n-1); den=min(rc-1,kc-1); v=math.sqrt(phic/den) if den>0 else np.nan
    prob=t/n; px=t.sum(1)/n; py=t.sum(0)/n; indep=np.outer(px,py); nz=prob>0; mi=np.sum(prob[nz]*np.log(prob[nz]/indep[nz])); hx=-np.sum(px[px>0]*np.log(px[px>0])); hy=-np.sum(py[py>0]*np.log(py[py>0]))
    return {"n":int(n),"cramers_v":v,"nmi":mi/math.sqrt(hx*hy) if hx*hy>0 else np.nan,"estado":"ok"},(ux,uy,t)

def eta_squared(x,c,valid=None,nodata=NODATA):
    x,c=np.asarray(x,float),np.asarray(c); ok=np.isfinite(x)&(c!=nodata)
    if valid is not None: ok &= valid
    x,c=x[ok],c[ok]
    if not len(x): return {"n":0,"eta_squared":np.nan,"estado":"sin_datos"}
    ss=np.sum((x-x.mean())**2)
    if ss<=0: return {"n":len(x),"eta_squared":np.nan,"estado":"factor_continuo_constante"}
    between=sum(np.count_nonzero(c==v)*(x[c==v].mean()-x.mean())**2 for v in np.unique(c))
    return {"n":len(x),"eta_squared":float(np.clip(between/ss,0,1)),"estado":"ok"}

def systematic_indices(mask,spacing_m,resolution=15,origin=(0,0)):
    step=max(1,int(round(spacing_m/resolution))); rr,cc=np.meshgrid(np.arange(origin[0],mask.shape[0],step),np.arange(origin[1],mask.shape[1],step),indexing="ij"); keep=mask[rr,cc]
    return rr[keep],cc[keep],step

def log1p_valid(values,valid):
    """Aplica log1p solo a datos válidos y deja intactos los valores NoData."""
    out=np.asarray(values,dtype=float).copy(); mask=np.asarray(valid,dtype=bool)
    if np.any(out[mask]<0): raise ValueError("log1p requiere valores válidos no negativos")
    out[mask]=np.log1p(out[mask]); return out

def load_catalog(path,usv=False):
    d=pd.read_csv(path,dtype=str,keep_default_na=False); out={}
    for _,r in d.iterrows():
        code=int(r.raster_code); out[code]=({"value":r.CLAVE,"label":r.DESCRIPCIO,"kind":"categoria_original"} if usv else {"value":r.source_value,"label":r.inegi_category_name,"kind":r.category_kind})
    return out

def plot_matrix(df,title,path,lo=-1,hi=1,cmap="coolwarm"):
    z=df.to_numpy(float); fig,ax=plt.subplots(figsize=(max(7,len(df)*1.05),max(5,len(df)*.85))); im=ax.imshow(z,vmin=lo,vmax=hi,cmap=cmap,aspect="auto")
    ax.set_xticks(range(len(df.columns)),df.columns,rotation=40,ha="right"); ax.set_yticks(range(len(df)),df.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]): ax.text(j,i,"NA" if not np.isfinite(z[i,j]) else f"{z[i,j]:.2f}",ha="center",va="center",fontsize=8)
    ax.set_title(title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)

def refresh_selection_table(out):
    out=Path(out); pear=pd.read_csv(out/"correlacion_pearson.csv",index_col=0); spear=pd.read_csv(out/"correlacion_spearman.csv",index_col=0)
    cat=pd.read_csv(out/"resumen_asociacion_categorica.csv"); eta=pd.read_csv(out/"asociacion_eta_cuadrado.csv",index_col=0); vif=pd.read_csv(out/"vif_tolerancia.csv")
    cont=list(pear.index); categorical=list(eta.columns); rows=[]
    va=vif[vif.escenario=="A_originales"].set_index("factor")
    for name in cont:
        partners=[]
        for other in cont:
            if other==name: continue
            p,s=pear.loc[name,other],spear.loc[name,other]
            partners.append((abs(p) if np.isfinite(p) else -1,other,p,s))
        assoc="; ".join(f"{other}: rP={p:.3f}, rS={s:.3f}" for _,other,p,s in sorted(partners,reverse=True)[:2] if np.isfinite(p)) or "indefinida por constancia/NoData"
        mixed=eta.loc[name].sort_values(ascending=False).dropna()
        mixed_text="; eta² top: "+", ".join(f"{k}={v:.3f}" for k,v in mixed.head(2).items()) if len(mixed) else ""
        row=va.loc[name]
        rows.append({"factor":name,"tipo":"continua","grupo_tematico":"factor físico","pertinencia_fisica":"Interpretación física según definición; sin evidencia predictiva","asociaciones_principales":assoc+mixed_text,"VIF":row.vif,"limitaciones":"Autocorrelación y posibles insumos derivados compartidos","posible_redundancia":"Revisar asociaciones principales y VIF","recomendacion":"Conservar para decisión metodológica"})
    for name in categorical:
        pairs=cat[(cat.factor_a!=cat.factor_b)&((cat.factor_a==name)|(cat.factor_b==name))].sort_values("cramers_v",ascending=False)
        assoc="; ".join(f"{r.factor_b if r.factor_a==name else r.factor_a}: V={r.cramers_v:.3f}, NMI={r.nmi:.3f}" for r in pairs.head(2).itertuples()) or "sin pares definidos"
        mixed=eta[name].sort_values(ascending=False).dropna()
        mixed_text="; eta² top: "+", ".join(f"{k}={v:.3f}" for k,v in mixed.head(2).items()) if len(mixed) else ""
        rows.append({"factor":name,"tipo":"categórica","grupo_tematico":"cobertura" if name=="uso_suelo_vegetacion" else "edafología","pertinencia_fisica":"Categoría original; sin magnitud hidrológica inferida","asociaciones_principales":assoc+mixed_text,"VIF":"no aplica","limitaciones":"Autocorrelación, clases raras; categorías especiales separadas","posible_redundancia":"Revisar asociación y representación","recomendacion":"Conservar para decisión metodológica"})
    pd.DataFrame(rows).to_csv(out/"matriz_seleccion_factores.csv",index=False,encoding="utf-8-sig")
    return len(rows)

def vif_plot_matrix(tables,names,cap=30):
    z=np.full((len(tables),len(names)),np.nan)
    for i,df in enumerate(tables.values()):
        indexed=df.set_index("factor")
        for j,name in enumerate(names):
            if name not in indexed.index: continue
            row=indexed.loc[name]
            if row.estado=="colinealidad_perfecta": z[i,j]=cap
            elif np.isfinite(row.vif): z[i,j]=min(float(row.vif),cap)
    return z

def analyze(cp,kp,catalogs,output_root,run_id=None,grid_spec=None):
    out=create_run_dir(output_root,run_id); a,m,domain,common,grid,meta=read_stack(cp,kp,**(grid_spec or {})); cn,kn=list(cp),list(kp); cats={n:load_catalog(catalogs[n],n=="uso_suelo_vegetacion") for n in kn}
    desc=[{"factor":n,**descriptive_statistics(a[n][m[n]]),"nodata_en_dominio":int((domain&~m[n]).sum())} for n in cn]; pd.DataFrame(desc).to_csv(out/"estadistica_descriptiva_continuas.csv",index=False,encoding="utf-8-sig")
    freq=[]
    for n in kn:
        codes,counts=np.unique(a[n][m[n]&domain],return_counts=True)
        for code,count in zip(codes,counts):
            c=cats[n].get(int(code),{}); pct=100*count/domain.sum()
            freq.append({"factor":n,"codigo_raster":int(code),"valor_original":c.get("value","sin_catalogo"),"categoria":c.get("label","sin catálogo"),"tipo_categoria":c.get("kind","sin_catalogo"),"celdas":int(count),"superficie_km2":count*225/1e6,"porcentaje_municipal":pct,"constante":len(codes)==1,"baja_representacion_le_1pct":pct<=1})
    freqdf=pd.DataFrame(freq); freqdf.to_csv(out/"frecuencias_categoricas.csv",index=False,encoding="utf-8-sig")
    pear,spear,pairn=correlations(a,m,cn); pear.to_csv(out/"correlacion_pearson.csv",encoding="utf-8-sig"); spear.to_csv(out/"correlacion_spearman.csv",encoding="utf-8-sig")
    flow=a["acumulacion_flujo"]
    if (flow[m["acumulacion_flujo"]]<0).any(): raise ValueError("log1p inválido: acumulación negativa")
    alog=dict(a); alog["acumulacion_flujo"]=log1p_valid(flow,m["acumulacion_flujo"]); plog,_,_=correlations(alog,m,cn); plog.to_csv(out/"correlacion_pearson_log1p_acumulacion_flujo.csv",encoding="utf-8-sig")
    scenarios={"A_originales":(cn,a),"B_log1p_acumulacion":(cn,alog),"C_sin_TWI":([n for n in cn if n!="twi"],a),"D_sin_acumulacion":([n for n in cn if n!="acumulacion_flujo"],a),"E_sin_elevacion":([n for n in cn if n!="elevacion"],a)}
    vifs={k:vif_tolerance(v,n,common) for k,(n,v) in scenarios.items()}; vif=pd.concat([d.assign(escenario=k) for k,d in vifs.items()],ignore_index=True); vif.to_csv(out/"vif_tolerancia.csv",index=False,encoding="utf-8-sig")
    cv=pd.DataFrame(np.nan,index=kn,columns=kn); nmi=cv.copy(); cpairs=[]
    for i,x in enumerate(kn):
        for y in kn[i:]:
            met,t=categorical_association(a[x][common],a[y][common]); cv.loc[x,y]=cv.loc[y,x]=met["cramers_v"]; nmi.loc[x,y]=nmi.loc[y,x]=met["nmi"]; cpairs.append({"factor_a":x,"factor_b":y,**met})
            if t:
                ux,uy,tab=t; rows=[{"codigo_a":int(xc),"categoria_a":cats[x].get(int(xc),{}).get("label",str(xc)),"codigo_b":int(yc),"categoria_b":cats[y].get(int(yc),{}).get("label",str(yc)),"frecuencia":int(tab[ix,iy])} for ix,xc in enumerate(ux) for iy,yc in enumerate(uy)]
                pd.DataFrame(rows).to_csv(out/"tablas_contingencia"/f"{x}__{y}.csv",index=False,encoding="utf-8-sig")
    cv.to_csv(out/"cramers_v.csv",encoding="utf-8-sig"); nmi.to_csv(out/"informacion_mutua_normalizada.csv",encoding="utf-8-sig"); pd.DataFrame(cpairs).to_csv(out/"resumen_asociacion_categorica.csv",index=False,encoding="utf-8-sig")
    eta=pd.DataFrame(np.nan,index=cn,columns=kn); erows=[]
    for x in cn:
        for y in kn:
            q=eta_squared(a[x],a[y],m[x]&m[y]&domain); eta.loc[x,y]=q["eta_squared"]; erows.append({"factor_continuo":x,"factor_categorico":y,**q})
    eta.to_csv(out/"asociacion_eta_cuadrado.csv",encoding="utf-8-sig")
    relevant=[]
    for i,x in enumerate(cn):
        for y in cn[i+1:]:
            for method,mat in (("Pearson",pear),("Spearman",spear),("Pearson_log1p_flujo",plog)):
                v=mat.loc[x,y]
                if np.isfinite(v) and abs(v)>=.70: relevant.append({"tipo":"continua","metodo":method,"factor_a":x,"factor_b":y,"asociacion":v,"umbral":">=0.90" if abs(v)>=.90 else ">=0.70","n":pairn[cn.index(x),cn.index(y)]})
    for q in cpairs:
        if np.isfinite(q["cramers_v"]) and q["cramers_v"]>=.30: relevant.append({"tipo":"categorica","metodo":"V_Cramer_corregida","factor_a":q["factor_a"],"factor_b":q["factor_b"],"asociacion":q["cramers_v"],"umbral":">=0.30 exploratorio","n":q["n"]})
    for q in erows:
        if np.isfinite(q["eta_squared"]) and q["eta_squared"]>=.10: relevant.append({"tipo":"mixta","metodo":"eta_cuadrado","factor_a":q["factor_continuo"],"factor_b":q["factor_categorico"],"asociacion":q["eta_squared"],"umbral":">=0.10 descriptivo","n":q["n"]})
    pd.DataFrame(relevant,columns=["tipo","metodo","factor_a","factor_b","asociacion","umbral","n"]).to_csv(out/"asociaciones_relevantes.csv",index=False,encoding="utf-8-sig")
    sens=[]; sample_sizes={}
    for spacing in (250,500):
        rr,cc,step=systematic_indices(common,spacing); sm=np.zeros(common.shape,bool); sm[rr,cc]=1; count=len(rr); sample_sizes[spacing]=count; aa={n:a[n][sm] for n in cn}; mm={n:np.ones(count,bool) for n in cn}; pp,ss,_=correlations(aa,mm,cn)
        for i,x in enumerate(cn):
            for y in cn[i+1:]:
                for method,mat,full in (("Pearson",pp,pear),("Spearman",ss,spear)):
                    sens.append({"espaciamiento_nominal_m":spacing,"paso_celdas":step,"espaciamiento_efectivo_m":step*15,"origen":"fila 0 columna 0, celdas válidas","n":count,"metodo":method,"factor_a":x,"factor_b":y,"valor":mat.loc[x,y],"global":full.loc[x,y]})
        for i,x in enumerate(kn):
            for y in kn[i:]:
                v,_=categorical_association(a[x][sm],a[y][sm]); g,_=categorical_association(a[x][common],a[y][common])
                sens.append({"espaciamiento_nominal_m":spacing,"paso_celdas":step,"espaciamiento_efectivo_m":step*15,"origen":"fila 0 columna 0, celdas válidas","n":count,"metodo":"Cramers_V","factor_a":x,"factor_b":y,"valor":v["cramers_v"],"global":g["cramers_v"]})
    pd.DataFrame(sens).to_csv(out/"sensibilidad_muestreo_espacial.csv",index=False,encoding="utf-8-sig")
    refresh_selection_table(out)
    plot_matrix(pear,"Pearson — continuos",out/"matriz_pearson.png"); plot_matrix(spear,"Spearman — continuos",out/"matriz_spearman.png"); plot_matrix(cv,"V de Cramér corregida",out/"matriz_cramers_v.png",0,1,"viridis"); plot_matrix(eta,"Eta cuadrado",out/"matriz_asociacion_mixta.png",0,1,"viridis")
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,n in zip(axes.ravel(),cn): ax.hist(a[n][m[n]],bins=55,color="#397da0"); ax.set_title(n); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"distribuciones_factores_continuos.png",dpi=130); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); labels=list(vifs); z=vif_plot_matrix(vifs,cn)
    im=ax.imshow(z,vmin=1,vmax=15,cmap="magma",aspect="auto"); ax.set_xticks(range(len(cn)),cn,rotation=40,ha="right"); ax.set_yticks(range(len(labels)),labels)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]): ax.text(j,i,"NA" if not np.isfinite(z[i,j]) else f"{z[i,j]:.1f}",ha="center",va="center",fontsize=8)
    ax.set_title("VIF por escenario (escala visual truncada en 30)"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"diagnostico_vif.png",dpi=130); plt.close(fig)
    top=sorted([(abs(pear.loc[x,y]) if np.isfinite(pear.loc[x,y]) else -1,x,y,pear.loc[x,y],spear.loc[x,y]) for i,x in enumerate(cn) for y in cn[i+1:]],reverse=True)
    corrtext="\n".join(f"- {x}–{y}: Pearson {p:.3f}; Spearman {s:.3f}." for _,x,y,p,s in top[:8]); ctop=sorted([r for r in cpairs if r["factor_a"]!=r["factor_b"] and np.isfinite(r["cramers_v"])],key=lambda r:r["cramers_v"],reverse=True)[:8]
    cattext="\n".join(f"- {r['factor_a']}–{r['factor_b']}: V={r['cramers_v']:.3f}, NMI={r['nmi']:.3f}." for r in ctop)
    etop=eta.stack().sort_values(ascending=False).head(8)
    etatext="\n".join(f"- {x}–{y}: eta²={v:.3f}." for (x,y),v in etop.items())
    high=vif[np.isfinite(vif.vif)&(vif.vif>=5)]
    viftext="\n".join(f"- {r.escenario} / {r.factor}: VIF={r.vif:.2f}; tolerancia={r.tolerancia:.3f}." for r in high.itertuples()) or "- No se observaron VIF >= 5."
    report=f"""# Redundancia estadística — Villa de Álvarez

Se analizaron 13 factores sobre {int(domain.sum()):,} celdas municipales y {int(common.sum()):,} celdas válidas conjuntas. Los píxeles tienen dependencia espacial y no son observaciones independientes.

## Asociaciones continuas principales
{corrtext}

Pearson, Spearman y Pearson con log1p aplicado solo en memoria a acumulación de flujo. Se reportan |r|≥0.70 y |r|≥0.90.

## VIF y tolerancia
Escenarios A valores originales (referencia), B log1p de acumulación (sensibilidad a asimetría), C sin TWI (contrasta su información con pendiente y demás factores derivados del relieve), D sin acumulación (examina su aporte junto a HAND/TWI), E sin elevación (evalúa la elevación como base de productos topográficos). Regresiones auxiliares con intercepto mediante centrado. Constantes y colinealidad perfecta se etiquetan en la tabla.
Valores VIF >=5:
{viftext}

## Categóricas y mixtas
V de Cramér corregida por sesgo y NMI simétrica. Las clases especiales N, N/A y N/D (si existen) conservan significado catalogado y no se tratan como magnitudes. Contingencias en subcarpeta separada.
Mayores V:
{cattext}

Mayores asociaciones mixtas:
{etatext}

Eta cuadrado es la proporción de variación continua asociada a diferencias categóricas, no una relación causal ni capacidad predictiva.

## Sensibilidad espacial
Malla desde fila 0, columna 0. 250 m: 17 celdas =255 m (n={sample_sizes[250]:,}); 500 m: 33 =495 m (n={sample_sizes[500]:,}). Se comparan Pearson, Spearman y V de Cramér con globales en CSV. No es corrección inferencial.

## Selección y límites
La matriz deja la decisión al investigador. Sin inventario de inundaciones observadas no se evalúa capacidad predictiva ni se asignan pesos. Dependencia espacial y categorías escasas limitan la interpretación. No se eliminaron factores ni se generaron rásteres.
"""
    (out/"informe_redundancia_factores.md").write_text(report,encoding="utf-8")
    inputs={n:{"raster":str(p),"catalog":str(catalogs[n])} for n,p in kp.items()}|{n:{"raster":str(p)} for n,p in cp.items()}
    man={"created_utc":datetime.now(timezone.utc).isoformat(),"municipality":"Villa de Álvarez, Colima","grid":grid,"inputs":inputs,"valid_cells_by_factor":{n:int(x.sum()) for n,x in m.items()},"methods":{"sampling_origin_rc":[0,0],"nominal_spacing_m":[250,500],"effective_spacing_m":[255,495],"steps":[17,33],"special_categories_nominal":True,"no_rasters_written":True},"products":[p.name for p in out.iterdir()]}
    (out/"manifiesto_analisis.json").write_text(json.dumps(man,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(r"C:\Doctorado-DCAG")); p.add_argument("--repo",type=Path,default=Path(r"C:\Doctorado-DCAG\repositorio")); p.add_argument("--run-id")
    q=p.parse_args(); root=q.root; repo=q.repo; phys=root/"resultados/territorial/factores_fisicos/villa_de_alvarez"; ed=repo/"results/territorial/edafologia/procesamiento/villa_de_alvarez/run_20261126T234139"; usv=root/"resultados/territorial/uso_suelo_vegetacion/villa_de_alvarez/run_20261126T235755"
    cp={n:phys/f for n,f in PHYS.items()}; kp={n:ed/f for n,f in EDAPH.items()}|{"uso_suelo_vegetacion":usv/"uso_suelo_vegetacion_villa_de_alvarez.tif"}
    catalogs={n:ed/"catalogo_categorias_edafologicas.csv" for n in CATS[:6]}|{"uso_suelo_vegetacion":usv/"catalogo_categorias_uso_suelo.csv"}
    missing=[str(x) for x in [*cp.values(),*kp.values(),*catalogs.values()] if not x.is_file()]
    if missing: raise FileNotFoundError("Insumos faltantes: "+"; ".join(missing))
    out=analyze(cp,kp,catalogs,root/"resultados/territorial/factor_redundancy/villa_de_alvarez",q.run_id); print(f"Salida: {out}")
if __name__=="__main__": main()

