import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.sustax.loader import load_sustax_file_safe


def normalize_sustax_base_name(filename: str) -> str:
    """
    Normaliza el nombre base de un archivo Sustax eliminando
    sufijos de escenario y numeraciones finales.

    Ejemplos
    --------
    Sustax_Manzanillo.csv -> Sustax_Manzanillo
    Sustax_Manzanillo2.csv -> Sustax_Manzanillo
    Sustax_ManzanilloSSP245.csv -> Sustax_Manzanillo
    """
    stem = Path(filename).stem

    stem = re.sub(
        r"(ERA5|SSP119|SSP126|SSP245|SSP343|SSP370|SSP434|SSP460|SSP585)$",
        "",
        stem,
    )
    stem = re.sub(r"\d+$", "", stem)

    return stem


def is_real_sustax_csv(fp: str) -> bool:
    """
    Determina si un archivo corresponde a un CSV original de Sustax
    y no a salidas derivadas, resúmenes o archivos auxiliares.
    """
    name = os.path.basename(fp).lower()

    if not name.endswith(".csv"):
        return False

    ignore_patterns = [
        "catalog",
        "neighbors",
        "summary",
        "obs_",
        "amax",
        "compare",
        "merged",
        "exported",
    ]

    if any(p in name for p in ignore_patterns):
        return False

    if not name.startswith("sustax_"):
        return False

    return True


def list_sustax_csv_files(raw_dir: str) -> list[str]:
    """
    Lista todos los CSV originales de Sustax dentro de un directorio.
    """
    files = sorted(
        [
            fp
            for fp in glob.glob(os.path.join(raw_dir, "*.csv"))
            if is_real_sustax_csv(fp)
        ]
    )
    return files


def build_sustax_inventory(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye el inventario de archivos Sustax y un log de errores.

    Parámetros
    ----------
    raw_dir : str
        Carpeta con CSVs originales de Sustax.

    Retorna
    -------
    inventory_df : pd.DataFrame
        Inventario estructurado de archivos válidos.
    bad_df : pd.DataFrame
        Tabla de archivos con error de lectura o procesamiento.
    """
    rows = []
    bad = []

    csv_files = list_sustax_csv_files(raw_dir)

    for fp in csv_files:
        fn = os.path.basename(fp)

        try:
            df_vals, _, meta = load_sustax_file_safe(
                fp,
                return_pandas_df=True,
                return_metadata=True,
            )

            rows.append(
                {
                    "sustax_file": fn,
                    "base_name": normalize_sustax_base_name(fn),
                    "lat": round(float(meta.get("lat", np.nan)), 4),
                    "lon": round(float(meta.get("lon", np.nan)), 4),
                    "n_rows": len(df_vals),
                    "n_cols": df_vals.shape[1],
                    "scenarios_detected": ", ".join(
                        sorted(
                            {
                                c.split("[")[-1].replace("]", "").strip()
                                for c in df_vals.columns
                                if "[" in c and "]" in c
                            }
                        )
                    ),
                    "path": fp,
                }
            )

        except Exception as e:
            bad.append(
                {
                    "file": fn,
                    "error": str(e),
                }
            )

    inventory_df = pd.DataFrame(rows)
    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(
            ["base_name", "lat", "lon", "sustax_file"]
        ).reset_index(drop=True)

    bad_df = pd.DataFrame(bad)

    return inventory_df, bad_df


def build_group_summary(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen agrupado por base_name, lat y lon.
    """
    if inventory_df.empty:
        return pd.DataFrame(
            columns=["base_name", "lat", "lon", "n_files", "files"]
        )

    group_summary = (
        inventory_df.groupby(["base_name", "lat", "lon"], as_index=False)
        .agg(
            n_files=("sustax_file", "count"),
            files=("sustax_file", lambda x: " | ".join(x)),
        )
        .sort_values(["base_name", "lat", "lon"])
        .reset_index(drop=True)
    )

    return group_summary


def export_inventory_outputs(
    inventory_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    out_dir: str,
    bad_df: Optional[pd.DataFrame] = None,
) -> dict[str, str]:
    """
    Exporta inventario, resumen de grupos y opcionalmente errores.

    Parámetros
    ----------
    inventory_df : pd.DataFrame
        Inventario completo.
    group_summary_df : pd.DataFrame
        Resumen por grupo.
    out_dir : str
        Carpeta de salida.
    bad_df : pd.DataFrame, optional
        Tabla de errores.

    Retorna
    -------
    dict[str, str]
        Rutas exportadas.
    """
    os.makedirs(out_dir, exist_ok=True)

    inventory_path = os.path.join(out_dir, "SUSTAX_raw_inventory.csv")
    group_summary_path = os.path.join(out_dir, "SUSTAX_group_summary.csv")

    inventory_df.to_csv(inventory_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)

    outputs = {
        "inventory_path": inventory_path,
        "group_summary_path": group_summary_path,
    }

    if bad_df is not None and not bad_df.empty:
        bad_path = os.path.join(out_dir, "SUSTAX_inventory_errors.csv")
        bad_df.to_csv(bad_path, index=False)
        outputs["bad_path"] = bad_path

    return outputs
