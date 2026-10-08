
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.sustax.loader import load_sustax_file_safe

DECIMALES_COORDENADAS = 4
TOLERANCIA_MM = 1e-6


def split_var_scenario(colname: str) -> tuple[Optional[str], Optional[str]]:
    m = re.match(r"^(.*)\s+\[(.*)\]\s*$", str(colname).strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def build_spatial_id(base_name: str, lat: float, lon: float) -> str:
    return f"{float(lat):.4f}_{float(lon):.4f}"


def _publicar(temporal: Path, destino: Path, patron: str, resumen: str):
    destino.mkdir(parents=True, exist_ok=True)
    nuevos = list(temporal.glob("*.csv"))
    if not (temporal / resumen).exists():
        raise RuntimeError(f"Falta el resumen: {resumen}")
    anteriores = list(destino.glob(patron))
    if (destino / resumen).exists() and destino / resumen not in anteriores:
        anteriores.append(destino / resumen)
    respaldo = destino.parent / f".backup_{destino.name}_{uuid.uuid4().hex}"
    respaldo.mkdir()
    copiados = []
    try:
        for archivo in anteriores:
            archivo.replace(respaldo / archivo.name)
        for archivo in nuevos:
            archivo.replace(destino / archivo.name)
            copiados.append(destino / archivo.name)
    except Exception:
        for archivo in copiados:
            if archivo.exists():
                archivo.unlink()
        for archivo in respaldo.iterdir():
            archivo.replace(destino / archivo.name)
        raise
    finally:
        shutil.rmtree(respaldo, ignore_errors=True)


def _depurar(datos: pd.DataFrame, diagnostico_dir: Path) -> pd.DataFrame:
    claves = ["date", "variable", "scenario"]
    datos = datos.dropna(subset=["date", "value_mm"]).copy()
    if datos.empty:
        return datos
    agrupado = datos.groupby(claves, as_index=False).agg(
        n_registros=("value_mm", "size"),
        minimo_mm=("value_mm", "min"),
        maximo_mm=("value_mm", "max"),
        archivos=("source_file", lambda x: " | ".join(sorted(set(x.astype(str)))))
    )
    agrupado["diferencia_mm"] = agrupado["maximo_mm"] - agrupado["minimo_mm"]
    conflictos = agrupado.loc[agrupado["diferencia_mm"] > TOLERANCIA_MM]
    if not conflictos.empty:
        diagnostico_dir.mkdir(parents=True, exist_ok=True)
        conflictos.to_csv(
            diagnostico_dir / "SUSTAX_conflictos.csv", index=False
        )
        raise ValueError(
            f"Se detectaron {len(conflictos)} conflictos de valores. "
            f"Consulta {diagnostico_dir / 'SUSTAX_conflictos.csv'}"
        )
    return (
        datos.sort_values(["date", "scenario", "variable", "source_file"])
        .drop_duplicates(subset=claves, keep="first")
        .reset_index(drop=True)
    )


def build_long_total_from_inventory(
    inventory_df: pd.DataFrame,
    out_dir: str
) -> tuple[pd.DataFrame, str]:
    destino = Path(out_dir)
    destino.parent.mkdir(parents=True, exist_ok=True)
    columnas = [
        "sustax_total", "lat", "lon", "n_original_files",
        "original_files", "n_rows_merged", "scenarios_merged", "out_file"
    ]
    resumen_nombre = "SUSTAX_TOTAL_LONG_summary.csv"
    resumen_filas = []
    with tempfile.TemporaryDirectory(
        prefix="sustax_long_", dir=str(destino.parent.resolve())
    ) as tmp:
        temporal = Path(tmp)
        if not inventory_df.empty:
            inventario = inventory_df.copy()
            inventario["lat"] = pd.to_numeric(
                inventario["lat"], errors="raise"
            ).round(DECIMALES_COORDENADAS)
            inventario["lon"] = pd.to_numeric(
                inventario["lon"], errors="raise"
            ).round(DECIMALES_COORDENADAS)
            if not (
                np.isfinite(inventario["lat"]).all()
                and np.isfinite(inventario["lon"]).all()
            ):
                raise ValueError("Inventario con coordenadas inválidas")
            for (lat, lon), grupo in inventario.groupby(["lat", "lon"], sort=True):
                partes = []
                for _, registro in grupo.iterrows():
                    archivo = registro["path"]
                    nombre = registro["sustax_file"]
                    try:
                        valores, _, _ = load_sustax_file_safe(
                            archivo,
                            return_pandas_df=True,
                            return_metadata=True
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"No fue posible leer {nombre}: {exc}"
                        ) from exc
                    valores = valores.copy()
                    valores.index = pd.to_datetime(
                        valores.index, errors="coerce"
                    )
                    valores = valores.loc[~valores.index.isna()]
                    valores.index = valores.index.normalize()
                    fechas = valores.index.strftime("%Y-%m-%d")
                    for columna in valores.columns:
                        variable, escenario = split_var_scenario(columna)
                        if not variable or not escenario:
                            continue
                        precipitacion = pd.to_numeric(
                            valores[columna], errors="coerce"
                        ) * 1000.0
                        partes.append(pd.DataFrame({
                            "source_file": nombre,
                            "date": fechas,
                            "variable": variable,
                            "scenario": escenario.upper(),
                            "value_mm": precipitacion.to_numpy()
                        }))
                if not partes:
                    continue
                conjunto = pd.concat(partes, ignore_index=True)
                conjunto = _depurar(
                    conjunto, destino.parent / "SUSTAX_DIAGNOSTICOS"
                )
                if conjunto.empty:
                    continue
                identificador = build_spatial_id("", lat, lon)
                sustax_total = f"{identificador}Total"
                conjunto.insert(0, "sustax_total", sustax_total)
                conjunto.insert(1, "lat", lat)
                conjunto.insert(2, "lon", lon)
                nombre_salida = f"{sustax_total}_LONG_mm.csv"
                conjunto.to_csv(temporal / nombre_salida, index=False)
                resumen_filas.append({
                    "sustax_total": sustax_total,
                    "lat": lat,
                    "lon": lon,
                    "n_original_files": len(grupo),
                    "original_files": " | ".join(
                        sorted(grupo["sustax_file"].astype(str))
                    ),
                    "n_rows_merged": len(conjunto),
                    "scenarios_merged": ", ".join(
                        sorted(conjunto["scenario"].dropna().unique())
                    ),
                    "out_file": str(destino / nombre_salida)
                })
        resumen = pd.DataFrame(resumen_filas, columns=columnas)
        resumen.to_csv(temporal / resumen_nombre, index=False)
        _publicar(temporal, destino, "*Total_LONG_mm.csv", resumen_nombre)
    return resumen, str(destino / resumen_nombre)


def build_by_scenario_total(
    long_total_dir: str,
    out_dir: str
) -> tuple[pd.DataFrame, str]:
    origen = Path(long_total_dir)
    destino = Path(out_dir)
    destino.parent.mkdir(parents=True, exist_ok=True)
    resumen_nombre = "SUSTAX_TOTAL_BYSCEN_summary.csv"
    columnas = [
        "sustax_total", "lat", "lon",
        "n_scenarios_generated", "scenarios_generated"
    ]
    resumen_filas = []
    archivos = sorted(origen.glob("*Total_LONG_mm.csv"))
    if not archivos:
        raise ValueError(f"No existen archivos LONG en {origen}")
    with tempfile.TemporaryDirectory(
        prefix="sustax_byscen_", dir=str(destino.parent.resolve())
    ) as tmp:
        temporal = Path(tmp)
        for archivo in archivos:
            datos = pd.read_csv(archivo)
            requeridas = {
                "sustax_total", "lat", "lon", "date",
                "variable", "scenario", "value_mm"
            }
            if not requeridas.issubset(datos.columns):
                raise ValueError(
                    f"Columnas faltantes en {archivo.name}: "
                    f"{sorted(requeridas - set(datos.columns))}"
                )
            if datos.empty:
                continue
            sustax_total = str(datos["sustax_total"].iloc[0])
            lat = float(datos["lat"].iloc[0])
            lon = float(datos["lon"].iloc[0])
            generados = []
            for escenario, grupo in datos.groupby("scenario", sort=True):
                claves = ["date", "variable"]
                if grupo.duplicated(subset=claves).any():
                    raise ValueError(
                        f"Duplicados no depurados en {archivo.name}: {escenario}"
                    )
                tabla = grupo.pivot(
                    index="date", columns="variable", values="value_mm"
                ).reset_index()
                tabla.insert(0, "sustax_total", sustax_total)
                tabla.insert(1, "lat", lat)
                tabla.insert(2, "lon", lon)
                nombre = f"{sustax_total}__{escenario}__mm.csv"
                tabla.to_csv(temporal / nombre, index=False)
                generados.append(nombre)
            resumen_filas.append({
                "sustax_total": sustax_total,
                "lat": lat,
                "lon": lon,
                "n_scenarios_generated": len(generados),
                "scenarios_generated": " | ".join(generados)
            })
        resumen = pd.DataFrame(resumen_filas, columns=columnas)
        resumen.to_csv(temporal / resumen_nombre, index=False)
        _publicar(temporal, destino, "*__mm.csv", resumen_nombre)
    return resumen, str(destino / resumen_nombre)
