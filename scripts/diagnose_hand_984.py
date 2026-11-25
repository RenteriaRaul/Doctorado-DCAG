from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio
from rasterio.windows import Window

PRODUCTS = {
    'dem': '90_MDE_Relleno_Sin_Flats.tif',
    'd8': '93_Direccion_Flujo_D8.tif',
    'acc': '94_Acumulacion_Flujo_Celdas.tif',
    'drain': '95_Red_Drenaje_Potencial.tif',
    'twi': '96_Indice_TWI.tif',
    'hand': '97_Altura_HAND.tif',
}

def valid(ds, a, window=None):
    # La máscara debe leerse sobre la misma ventana que el arreglo.
    # Sin window, Rasterio devuelve la máscara del raster completo.
    m = ds.read_masks(1, window=window) > 0
    if m.shape != a.shape:
        raise ValueError(
            f"Máscara y arreglo con dimensiones distintas: "
            f"máscara={m.shape}, arreglo={a.shape}"
        )
    if ds.nodata is not None:
        m &= a != ds.nodata
    if np.issubdtype(a.dtype, np.floating):
        m &= np.isfinite(a)
    return m

def main():
    p=argparse.ArgumentParser(description='Diagnóstico 98.4 de cobertura y conectividad local HAND; solo lectura.')
    p.add_argument('--products-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--row', type=int, default=9570)
    p.add_argument('--col', type=int, default=114)
    p.add_argument('--radius', type=int, default=50, help='Radio de ventana en celdas; predeterminado 50 (101x101).')
    p.add_argument('--block-rows', type=int, default=512)
    p.add_argument('--overwrite', action='store_true')
    a=p.parse_args()
    srcdir=Path(a.products_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    paths={k:srcdir/v for k,v in PRODUCTS.items()}
    missing=[str(v) for v in paths.values() if not v.is_file()]
    if missing: raise SystemExit('Faltan productos:\n'+'\n'.join(missing))
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    csv_path=out/f'hand_984_celdas_locales_{tag}.csv'; js_path=out/f'hand_984_resumen_{tag}.json'
    if not a.overwrite and (csv_path.exists() or js_path.exists()): raise SystemExit('La salida ya existe.')
    ds={k:rasterio.open(v) for k,v in paths.items()}
    try:
        ref=ds['dem']; profile=(ref.width,ref.height,ref.crs,ref.transform)
        aligned={k:(d.width==ref.width and d.height==ref.height and d.crs==ref.crs and d.transform==ref.transform) for k,d in ds.items()}
        if not all(aligned.values()): raise SystemExit(f'Cuadrículas desalineadas: {aligned}')
        if not (0<=a.row<ref.height and 0<=a.col<ref.width): raise SystemExit('Fila/columna fuera del raster.')
        r0=max(0,a.row-a.radius); r1=min(ref.height,a.row+a.radius+1); c0=max(0,a.col-a.radius); c1=min(ref.width,a.col+a.radius+1)
        win=Window(c0,r0,c1-c0,r1-r0)
        arrays={k:d.read(1,window=win) for k,d in ds.items()}
        masks={k:valid(d, arrays[k], window=win) for k,d in ds.items()}
        mismatch=masks['dem'] & ~masks['hand']
        hand_nodata= masks['dem'] & ~masks['hand']
        # Local cell-level evidence; all rows in the requested window, only mismatch cells.
        with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f); w.writerow(['row','col','x','y','d8_valid','d8','acc_valid','acc','drain_valid','drain','twi_valid','twi','hand_valid','hand','neighbor_hand_valid_8'])
            for rr,cc in np.argwhere(mismatch):
                gr=int(r0+rr); gc=int(c0+cc); x,y=rasterio.transform.xy(ref.transform,gr,gc,offset='center')
                rlo=max(0,rr-1); rhi=min(mismatch.shape[0],rr+2); clo=max(0,cc-1); chi=min(mismatch.shape[1],cc+2)
                neighbor=int(masks['hand'][rlo:rhi,clo:chi].sum())
                row=[gr,gc,x,y]
                for key in ('d8','acc','drain','twi','hand'):
                    row += [bool(masks[key][rr,cc]), arrays[key][rr,cc].item() if masks[key][rr,cc] else None]
                row += [neighbor]; w.writerow(row)
        # Whole-raster counts and mismatch statistics by block, without loading full rasters.
        totals={'reference_valid':0,'hand_valid':0,'hand_nodata_inside_reference':0,'hand_valid_outside_reference':0,'compared':0}
        acc_stats={'min':None,'max':None,'sum':0.0,'count':0}; d8_counts={}; drain_counts={}; twi_stats={'min':None,'max':None,'sum':0.0,'count':0}; mismatch_bounds=[ref.width,ref.height,-1,-1]
        for y0 in range(0,ref.height,a.block_rows):
            h=min(a.block_rows,ref.height-y0); bw=Window(0,y0,ref.width,h)
            aa={k:d.read(1,window=bw) for k,d in ds.items()}
            mm={k:valid(d, aa[k], window=bw) for k,d in ds.items()}
            rv=mm['dem']; hv=mm['hand']; bad=rv & ~hv
            totals['reference_valid']+=int(rv.sum()); totals['hand_valid']+=int(hv.sum()); totals['hand_nodata_inside_reference']+=int(bad.sum()); totals['hand_valid_outside_reference']+=int((~rv & hv).sum()); totals['compared']+=int(rv.sum())
            if bad.any():
                ys,xs=np.where(bad); mismatch_bounds=[min(mismatch_bounds[0],int(xs.min())),min(mismatch_bounds[1],int(y0+ys.min())),max(mismatch_bounds[2],int(xs.max())),max(mismatch_bounds[3],int(y0+ys.max()))]
                for key,dest in [('d8',d8_counts),('drain',drain_counts)]:
                    vals,counts=np.unique(aa[key][bad & mm[key]],return_counts=True)
                    for v,n in zip(vals,counts): dest[str(v.item())]=dest.get(str(v.item()),0)+int(n)

                for key, stats in [('acc',acc_stats),('twi',twi_stats)]:
                    vals=aa[key][bad & mm[key]].astype(np.float64)
                    if vals.size:
                        vmin=float(vals.min()); vmax=float(vals.max())
                        stats['min']=vmin if stats['min'] is None else min(stats['min'],vmin)
                        stats['max']=vmax if stats['max'] is None else max(stats['max'],vmax)
                        stats['sum']+=float(vals.sum(dtype=np.float64)); stats['count']+=int(vals.size)
        # 8-neighbour count for local mismatch cells, in local diagnostic window.
        neighbor_hist={}
        for rr,cc in np.argwhere(mismatch):
            n=int(masks['hand'][max(0,rr-1):min(mismatch.shape[0],rr+2),max(0,cc-1):min(mismatch.shape[1],cc+2)].sum())
            neighbor_hist[str(n)]=neighbor_hist.get(str(n),0)+1
        summary={
          'step':'98.4','timestamp_utc':tag,'read_only':True,'grid_alignment':aligned,
          'reference_grid':{'width':ref.width,'height':ref.height,'crs':str(ref.crs),'resolution':list(ref.res),'transform':list(ref.transform)},
          'target':{'row':a.row,'col':a.col,'radius_cells':a.radius,'window_rows':r1-r0,'window_cols':c1-c0},
          'local_window':{'reference_valid':int(masks['dem'].sum()),'hand_valid':int(masks['hand'].sum()),'hand_nodata_inside_reference':int(mismatch.sum()),'mismatch_cells':int(mismatch.sum()),'neighbor_hand_valid_8_histogram':neighbor_hist},
          'global_coverage':totals,'global_bbox_hand_nodata_inside_reference_col_row':[v if v>=0 else None for v in mismatch_bounds],
          'attributes_of_global_mismatch_cells_where_valid':{'d8_counts':d8_counts,'drainage_binary_counts':drain_counts,'accumulation_min':acc_stats['min'],'accumulation_max':acc_stats['max'],'accumulation_mean':acc_stats['sum']/acc_stats['count'] if acc_stats['count'] else None,'accumulation_count':acc_stats['count'],'twi_min':twi_stats['min'],'twi_max':twi_stats['max'],'twi_mean':twi_stats['sum']/twi_stats['count'] if twi_stats['count'] else None,'twi_count':twi_stats['count']},
          'interpretation_limits':['Este diagnóstico cuantifica cobertura y describe D8/acumulación en celdas donde HAND es NoData.','No demuestra por sí solo la causa del NoData ni reconstruye conectividad D8 global.','No modifica ni recalcula los GeoTIFF.'],
          'outputs':{'csv':str(csv_path),'json':str(js_path)}
        }
        js_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:
        for d in ds.values(): d.close()
if __name__=='__main__': main()
