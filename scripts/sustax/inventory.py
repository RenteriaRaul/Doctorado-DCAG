
import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.sustax.loader import load_sustax_file_safe

DECIMALES_COORDENADAS = 4


def normalize_sustax_base_name(filename: str) -> str:
    stem = Path(filename).stem
    match = re.search(r"(ID-\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    stem = re.sub(
        r"(ERA5|SSP119|SSP126|SSP245|SSP343|SSP370|SSP434|SSP460|SSP585)$",
        "", stem, flags=re.IGNORECASE
    )
    return re.sub(r"\d+$", "", stem)


def build_coordinate_id(lat: float, lon: float) -> str:
    return f"{float(lat):.{DECIMALES_COORDENADAS}f}_{float(lon):.{DECIMALES_COORDENADAS}f}"


def is_real_sustax_csv(fp: str) -> bool:
    path = Path(fp)
    if not path.is_file() or path.suffix.lower() != ".csv":
        return False
    ignore_patterns = [
        "catalog", "neighbors", "summary", "obs_", "amax",
        "compare", "merged", "exported", "long_mm", "byscen"
    ]
    if any(pattern in path.name.lower() for pattern in ignore_patterns):
        return False
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            return "SUSTAX Data Export" in f.readline().strip()
    except OSError:
        return False


def list_sustax_csv_files(raw_dir: str) -> list[str]:
    return sorted(
        fp for fp in glob.glob(os.path.join(raw_dir, "*.csv"))
        if is_real_sustax_csv(fp)
    )


def build_sustax_inventory(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bad = [], []
    for fp in list_sustax_csv_files(raw_dir):
        fn = os.path.basename(fp)
        try:
            df_vals, _, meta = load_sustax_file_safe(
                fp, return_pandas_df=True, return_metadata=True
            )
            lat = round(float(meta.get("lat", np.nan)), DECIMALES_COORDENADAS)
            lon = round(float(meta.get("lon", np.nan)), DECIMALES_COORDENADAS)
            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError("Coordenadas ausentes o no numéricas")
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Coordenadas fuera de rango: {lat}, {lon}")
            scenarios = meta.get("scenarios", []) or sorted({
                c.split("[")[-1].replace("]", "").strip()
                for c in df_vals.columns if "[" in c and "]" in c
            })
            rows.append({
                "sustax_file": fn,
                "base_name": normalize_sustax_base_name(fn),
                "punto_id": build_coordinate_id(lat, lon),
                "lat": lat,
                "lon": lon,
                "n_rows": len(df_vals),
                "n_cols": df_vals.shape[1],
                "scenarios_detected": ", ".join(scenarios),
                "date_start": meta.get("date_start", pd.NaT),
                "date_end": meta.get("date_end", pd.NaT),
                "data_time_format": meta.get("data_time_format"),
                "osm_location": meta.get("osm_location"),
                "path": fp
            })
        except Exception as exc:
            bad.append({"file": fn, "error": str(exc)})
    inventory_df = pd.DataFrame(rows)
    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(
            ["lat", "lon", "sustax_file"]
        ).reset_index(drop=True)
    return inventory_df, pd.DataFrame(bad, columns=["file", "error"])


def build_group_summary(inventory_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "base_name", "punto_id", "lat", "lon", "n_files",
        "files", "base_names", "n_scenarios", "scenarios"
    ]
    if inventory_df.empty:
        return pd.DataFrame(columns=columns)
    df = inventory_df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="raise").round(DECIMALES_COORDENADAS)
    df["lon"] = pd.to_numeric(df["lon"], errors="raise").round(DECIMALES_COORDENADAS)
    df["punto_id"] = [
        build_coordinate_id(lat, lon) for lat, lon in zip(df["lat"], df["lon"])
    ]
    rows = []
    for (lat, lon), grupo in df.groupby(["lat", "lon"], sort=True):
        escenarios = set()
        for valor in grupo["scenarios_detected"].dropna():
            escenarios.update(
                s.strip() for s in str(valor).split(",") if s.strip()
            )
        bases = sorted(set(grupo["base_name"].astype(str)))
        rows.append({
            "base_name": bases[0],
            "punto_id": build_coordinate_id(lat, lon),
            "lat": lat,
            "lon": lon,
            "n_files": len(grupo),
            "files": " | ".join(sorted(grupo["sustax_file"].astype(str))),
            "base_names": " | ".join(bases),
            "n_scenarios": len(escenarios),
            "scenarios": ", ".join(sorted(escenarios))
        })
    return pd.DataFrame(rows, columns=columns)


def export_inventory_outputs(
    inventory_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    out_dir: str,
    bad_df: Optional[pd.DataFrame] = None
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    inventory_path = os.path.join(out_dir, "SUSTAX_raw_inventory.csv")
    group_summary_path = os.path.join(out_dir, "SUSTAX_group_summary.csv")
    inventory_df.to_csv(inventory_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    outputs = {
        "inventory_path": inventory_path,
        "group_summary_path": group_summary_path
    }
    if bad_df is not None and not bad_df.empty:
        bad_path = os.path.join(out_dir, "SUSTAX_inventory_errors.csv")
        bad_df.to_csv(bad_path, index=False)
        outputs["bad_path"] = bad_path
    return outputs
