import os
import numpy as np
import pandas as pd


# ==========================================
# UTILIDADES
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
# LECTURA DE COORDENADAS CONAGUA
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
# GENERAR VECINOS
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
                "station_id": sid,
                "distance_km": float(dist_km)
            })

    neighbors = pd.DataFrame(rows)
    return neighbors.sort_values(["sustax_total", "distance_km"])


# ==========================================
# GENERAR OBS SIMPLE
# ==========================================
def compute_obs_simple(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(df_list, axis=1)

    obs = merged.mean(axis=1, skipna=True).to_frame(name="pp_mm")
    obs = obs.reset_index().rename(columns={"index": "date"})

    return obs


# ==========================================
# GENERAR OBS IDW
# ==========================================
def compute_obs_idw(df_list: list[pd.DataFrame], distances: np.ndarray) -> pd.DataFrame:
    weights = 1 / distances
    weights = weights / weights.sum()

    data = np.column_stack([df.values.flatten() for df in df_list])
    weighted = np.dot(data, weights)

    obs = pd.DataFrame({
        "date": df_list[0].index,
        "pp_mm": weighted
    })

    return obs.reset_index(drop=True)


# ==========================================
# EXPORTACIÓN
# ==========================================
def export_obs(
    obs_df: pd.DataFrame,
    out_path: str
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obs_df.to_csv(out_path, index=False)
