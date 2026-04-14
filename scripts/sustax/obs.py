import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


# ==========================================
# UTILIDADES ESPACIALES
# ==========================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ==========================================
# COORDENADAS CONAGUA
# ==========================================
def read_conagua_coords(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    out = df[["clave", "latitud", "longitud"]].copy()
    out = out.rename(columns={
        "clave": "station_id",
        "latitud": "lat",
        "longitud": "lon"
    })

    out["station_id"] = out["station_id"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    return out.dropna(subset=["lat", "lon"])


# ==========================================
# VECINOS SUSTAX-CONAGUA
# ==========================================
def build_neighbors(
    total_df: pd.DataFrame,
    conagua_df: pd.DataFrame,
    r_km: float = 50,
    max_n: int = 8
) -> pd.DataFrame:
    rows = []

    con_lat = conagua_df["lat"].to_numpy()
    con_lon = conagua_df["lon"].to_numpy()
    con_id = conagua_df["station_id"].to_numpy()

    for _, s in total_df.iterrows():
        d = haversine_km(s["lat"], s["lon"], con_lat, con_lon)

        mask = d <= r_km
        if mask.sum() == 0:
            continue

        idx = np.argsort(d[mask])[:max_n]
        ids_sel = con_id[mask][idx]
        d_sel = d[mask][idx]

        for sid, dist_km in zip(ids_sel, d_sel):
            rows.append({
                "sustax_total": s["sustax_total"],
                "sustax_lat": s.get("lat", np.nan),
                "sustax_lon": s.get("lon", np.nan),
                "station_id": sid,
                "distance_km": float(dist_km),
                "R_km": r_km,
                "MAX_N": max_n
            })

    neighbors = pd.DataFrame(rows)
    if not neighbors.empty:
        neighbors = neighbors.sort_values(["sustax_total", "distance_km"]).reset_index(drop=True)

    return neighbors


# ==========================================
# DETECCIÓN DE ARCHIVOS Y COLUMNAS
# ==========================================
def find_station_file(station_id: str, folder: str) -> Optional[str]:
    station_id = str(station_id).strip()
    patterns = [
        os.path.join(folder, f"*{station_id}*.xlsx"),
        os.path.join(folder, f"*{station_id}*.xls"),
        os.path.join(folder, f"*{station_id}*.csv"),
    ]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    if not matches:
        return None

    matches = sorted(matches, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return matches[0]


def _norm(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    return text


def detect_date_col(columns) -> Optional[str]:
    candidates = {"date", "fecha", "time", "datetime", "day", "dia"}
    for c in columns:
        if _norm(c) in candidates:
            return c
    return None


def detect_pp_col(columns) -> Optional[str]:
    candidates = {"pp", "precip", "precipitacion", "precipitation", "lluvia", "prcp", "rain"}
    for c in columns:
        if _norm(c) in candidates:
            return c

    for c in columns:
        n = _norm(c)
        if ("precip" in n) or (n == "pp") or ("lluv" in n) or ("prcp" in n):
            return c

    return None


# ==========================================
# LECTURA DE SERIES CONAGUA
# ==========================================
def read_conagua_station_daily(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Extensión no soportada: {ext}")

    date_col = detect_date_col(df.columns)
    pp_col = detect_pp_col(df.columns)

    if date_col is None:
        raise ValueError(f"No detecté columna de fecha en {os.path.basename(path)}")
    if pp_col is None:
        raise ValueError(f"No detecté columna de precipitación en {os.path.basename(path)}")

    out = df[[date_col, pp_col]].copy()
    out = out.rename(columns={date_col: "date", pp_col: "pp_mm"})

    out["pp_mm"] = pd.to_numeric(out["pp_mm"], errors="coerce")

    d0 = pd.to_datetime(out["date"], errors="coerce", dayfirst=False)
    d1 = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["date"] = d1 if d1.notna().sum() > d0.notna().sum() else d0

    out["date"] = out["date"].dt.normalize()

    out = out.dropna(subset=["date"]).sort_values("date")
    out.loc[out["pp_mm"] < 0, "pp_mm"] = np.nan

    out = out.groupby("date", as_index=False)["pp_mm"].mean()

    return out


# ==========================================
# CONSTRUCCIÓN OBS SIMPLE / IDW
# ==========================================
def build_obs_total(
    neighbors_df: pd.DataFrame,
    conagua_folder: str,
    idw_p: float = 2,
    min_stations_per_day: int = 1
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    obs_simple = {}
    obs_idw = {}
    report_rows = []

    for sustax_total, grp in neighbors_df.groupby("sustax_total"):
        grp = grp.copy()
        grp["station_id"] = grp["station_id"].astype(str).str.strip()

        station_meta = []
        station_series = []

        for _, r in grp.iterrows():
            sid = r["station_id"]
            dist = float(r["distance_km"])
            fpath = find_station_file(sid, conagua_folder)

            if fpath is None:
                report_rows.append({
                    "sustax_total": sustax_total,
                    "station_id": sid,
                    "status": "MISSING_FILE",
                    "path": None
                })
                continue

            try:
                df_st = read_conagua_station_daily(fpath)
                df_st = df_st.set_index("date").rename(columns={"pp_mm": f"pp_{sid}"})
                station_series.append(df_st)
                station_meta.append({
                    "station_id": sid,
                    "distance_km": dist,
                    "path": fpath
                })
                report_rows.append({
                    "sustax_total": sustax_total,
                    "station_id": sid,
                    "status": "OK",
                    "path": fpath
                })
            except Exception as e:
                report_rows.append({
                    "sustax_total": sustax_total,
                    "station_id": sid,
                    "status": f"READ_ERROR: {e}",
                    "path": fpath
                })

        if not station_series:
            obs_simple[sustax_total] = pd.DataFrame(columns=["date", "pp_mm"])
            obs_idw[sustax_total] = pd.DataFrame(columns=["date", "pp_mm"])
            continue

        all_dates = pd.Index([])
        for df in station_series:
            all_dates = all_dates.union(df.index)

        aligned = [df.reindex(all_dates) for df in station_series]
        df_all = pd.concat(aligned, axis=1).sort_index()

        n_avail = df_all.notna().sum(axis=1)

        # OBS simple
        simple = df_all.mean(axis=1, skipna=True)
        simple = simple.where(n_avail >= min_stations_per_day)

        # OBS IDW
        col_sids = [c.replace("pp_", "") for c in df_all.columns]
        dists = []
        for sid in col_sids:
            meta = next((x for x in station_meta if x["station_id"] == sid), None)
            dists.append(meta["distance_km"] if meta else np.nan)

        dists = np.asarray(dists, dtype=float)

        if np.any(np.isnan(dists)):
            idw = simple.copy()
        else:
            vals = df_all.to_numpy(dtype=float)
            idw_vals = np.full(len(df_all), np.nan, dtype=float)

            for i in range(vals.shape[0]):
                row = vals[i, :]
                mask = ~np.isnan(row)

                if mask.sum() >= min_stations_per_day:
                    d = dists[mask]
                    v = row[mask]

                    w = 1.0 / np.maximum(d, 1e-6) ** idw_p
                    w = w / w.sum()

                    idw_vals[i] = np.sum(w * v)

            idw = pd.Series(idw_vals, index=df_all.index)

        out_simple = simple.reset_index()
        out_simple.columns = ["date", "pp_mm"]
        out_simple["date"] = pd.to_datetime(out_simple["date"]).dt.strftime("%Y-%m-%d")

        out_idw = idw.reset_index()
        out_idw.columns = ["date", "pp_mm"]
        out_idw["date"] = pd.to_datetime(out_idw["date"]).dt.strftime("%Y-%m-%d")

        obs_simple[sustax_total] = out_simple
        obs_idw[sustax_total] = out_idw

    report = pd.DataFrame(report_rows)
    return obs_simple, obs_idw, report


# ==========================================
# EXPORTACIÓN
# ==========================================
def export_obs_dict(
    obs_dict: dict[str, pd.DataFrame],
    out_folder: str,
    prefix: str
):
    os.makedirs(out_folder, exist_ok=True)

    for key, df in obs_dict.items():
        out_fp = os.path.join(out_folder, f"{prefix}__{key}.csv")
        df.to_csv(out_fp, index=False)


def export_obs_outputs(
    obs_simple_dict: dict[str, pd.DataFrame],
    obs_idw_dict: dict[str, pd.DataFrame],
    report_df: pd.DataFrame,
    out_folder: str
):
    os.makedirs(out_folder, exist_ok=True)

    export_obs_dict(obs_simple_dict, out_folder, "OBS_SIMPLE")
    export_obs_dict(obs_idw_dict, out_folder, "OBS_IDW")

    report_fp = os.path.join(out_folder, "OBS_generation_report.csv")
    report_df.to_csv(report_fp, index=False)

    return report_fp


# ==========================================
# PIPELINE COMPLETO
# ==========================================
def run_obs_pipeline(
    total_points_df: pd.DataFrame,
    conagua_coords_xlsx: str,
    conagua_data_folder: str,
    out_folder: str,
    r_km: float = 50,
    max_n: int = 8,
    idw_p: float = 2,
    min_stations_per_day: int = 1
):
    conagua_df = read_conagua_coords(conagua_coords_xlsx)

    neighbors_df = build_neighbors(
        total_df=total_points_df,
        conagua_df=conagua_df,
        r_km=r_km,
        max_n=max_n
    )

    obs_simple_dict, obs_idw_dict, report_df = build_obs_total(
        neighbors_df=neighbors_df,
        conagua_folder=conagua_data_folder,
        idw_p=idw_p,
        min_stations_per_day=min_stations_per_day
    )

    report_fp = export_obs_outputs(
        obs_simple_dict=obs_simple_dict,
        obs_idw_dict=obs_idw_dict,
        report_df=report_df,
        out_folder=out_folder
    )

    return neighbors_df, obs_simple_dict, obs_idw_dict, report_df, report_fp
