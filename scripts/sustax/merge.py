import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from scripts.sustax.loader import load_sustax_file_safe


def split_var_scenario(colname: str) -> tuple[Optional[str], Optional[str]]:
    """
    Separa una columna tipo:
    'Total Precipitation [SSP245]'
    en:
    ('Total Precipitation', 'SSP245')
    """
    m = re.match(r"^(.*)\s+\[(.*)\]\s*$", str(colname).strip())
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def build_long_total_from_inventory(
    inventory_df: pd.DataFrame,
    out_dir: str,
) -> tuple[pd.DataFrame, str]:
    """
    Genera archivos LONG_TOTAL a partir del inventario agrupado
    por (base_name, lat, lon).

    Parámetros
    ----------
    inventory_df : pd.DataFrame
        Inventario Sustax.
    out_dir : str
        Carpeta de salida para LONG_TOTAL.

    Retorna
    -------
    merged_summary_df : pd.DataFrame
        Resumen de archivos LONG generados.
    merged_summary_path : str
        Ruta del CSV resumen.
    """
    os.makedirs(out_dir, exist_ok=True)

    merged_rows = []

    if inventory_df.empty:
        merged_summary_df = pd.DataFrame(
            columns=[
                "sustax_total",
                "lat",
                "lon",
                "n_original_files",
                "original_files",
                "n_rows_merged",
                "scenarios_merged",
                "out_file",
            ]
        )
        merged_summary_path = os.path.join(out_dir, "SUSTAX_TOTAL_LONG_summary.csv")
        merged_summary_df.to_csv(merged_summary_path, index=False)
        return merged_summary_df, merged_summary_path

    for (base_name, lat, lon), grp in inventory_df.groupby(["base_name", "lat", "lon"]):
        long_parts = []

        for _, row in grp.iterrows():
            fp = row["path"]
            fn = row["sustax_file"]

            try:
                df_vals, _, _ = load_sustax_file_safe(
                    fp,
                    return_pandas_df=True,
                    return_metadata=True,
                )

                df_vals = df_vals.copy()
                df_vals.index = pd.to_datetime(df_vals.index, errors="coerce")
                df_vals = df_vals[~df_vals.index.isna()].sort_index()
                df_vals.index = df_vals.index.normalize()

                iso_dates = df_vals.index.strftime("%Y-%m-%d")

                rows_long = []
                for c in df_vals.columns:
                    var, sc = split_var_scenario(c)
                    if var is None:
                        continue

                    values_mm = pd.to_numeric(df_vals[c], errors="coerce") * 1000.0

                    tmp = pd.DataFrame(
                        {
                            "source_file": fn,
                            "date": iso_dates,
                            "variable": var,
                            "scenario": sc,
                            "value_mm": values_mm.values,
                        }
                    )
                    rows_long.append(tmp)

                if rows_long:
                    part = pd.concat(rows_long, ignore_index=True)
                    long_parts.append(part)

            except Exception as e:
                print(f"⚠️ Error procesando {fn}: {e}")

        if not long_parts:
            continue

        merged = pd.concat(long_parts, ignore_index=True)

        merged = (
            merged.drop_duplicates(subset=["date", "variable", "scenario"], keep="first")
            .sort_values(["date", "scenario", "variable"])
            .reset_index(drop=True)
        )

        sustax_total = f"{base_name}Total"
        merged.insert(0, "sustax_total", sustax_total)
        merged.insert(1, "lat", lat)
        merged.insert(2, "lon", lon)

        out_name = f"{base_name}Total_LONG_mm.csv"
        out_fp = os.path.join(out_dir, out_name)
        merged.to_csv(out_fp, index=False)

        merged_rows.append(
            {
                "sustax_total": sustax_total,
                "lat": lat,
                "lon": lon,
                "n_original_files": len(grp),
                "original_files": " | ".join(grp["sustax_file"].tolist()),
                "n_rows_merged": len(merged),
                "scenarios_merged": ", ".join(sorted(merged["scenario"].dropna().unique())),
                "out_file": out_fp,
            }
        )

    merged_summary_df = pd.DataFrame(merged_rows)
    if not merged_summary_df.empty:
        merged_summary_df = merged_summary_df.sort_values("sustax_total").reset_index(drop=True)

    merged_summary_path = os.path.join(out_dir, "SUSTAX_TOTAL_LONG_summary.csv")
    merged_summary_df.to_csv(merged_summary_path, index=False)

    return merged_summary_df, merged_summary_path


def build_by_scenario_total(
    long_total_dir: str,
    out_dir: str,
) -> tuple[pd.DataFrame, str]:
    """
    Genera archivos BY_SCENARIO_TOTAL a partir de LONG_TOTAL.

    Parámetros
    ----------
    long_total_dir : str
        Carpeta con archivos *Total_LONG_mm.csv
    out_dir : str
        Carpeta de salida BY_SCENARIO_TOTAL

    Retorna
    -------
    byscen_summary_df : pd.DataFrame
        Resumen de escenarios generados por punto.
    byscen_summary_path : str
        Ruta del CSV resumen.
    """
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    long_files = sorted(glob.glob(os.path.join(long_total_dir, "*Total_LONG_mm.csv")))

    for fp in long_files:
        df = pd.read_csv(fp)

        if df.empty:
            continue

        if "scenario" not in df.columns:
            print(f"⚠️ El archivo no tiene columna 'scenario': {os.path.basename(fp)}")
            continue

        sustax_total = df["sustax_total"].iloc[0]
        lat = df["lat"].iloc[0]
        lon = df["lon"].iloc[0]

        generated = []

        for sc in sorted(df["scenario"].dropna().unique()):
            sub = df[df["scenario"] == sc].copy()

            if sub.empty:
                continue

            piv = (
                sub.pivot_table(
                    index="date",
                    columns="variable",
                    values="value_mm",
                    aggfunc="mean",
                )
                .reset_index()
            )

            piv.insert(0, "sustax_total", sustax_total)
            piv.insert(1, "lat", lat)
            piv.insert(2, "lon", lon)

            out_name = f"{sustax_total}__{sc}__mm.csv"
            out_fp = os.path.join(out_dir, out_name)
            piv.to_csv(out_fp, index=False)

            generated.append(out_name)

        summary_rows.append(
            {
                "sustax_total": sustax_total,
                "lat": lat,
                "lon": lon,
                "n_scenarios_generated": len(generated),
                "scenarios_generated": " | ".join(generated),
            }
        )

    byscen_summary_df = pd.DataFrame(summary_rows)
    if not byscen_summary_df.empty:
        byscen_summary_df = byscen_summary_df.sort_values("sustax_total").reset_index(drop=True)

    byscen_summary_path = os.path.join(out_dir, "SUSTAX_TOTAL_BYSCEN_summary.csv")
    byscen_summary_df.to_csv(byscen_summary_path, index=False)

    return byscen_summary_df, byscen_summary_path
