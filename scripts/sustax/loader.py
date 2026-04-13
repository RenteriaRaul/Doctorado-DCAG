import csv
import decimal
from typing import Any

import numpy as np
import pandas as pd


def _isfloat(value: str) -> bool:
    try:
        decimal.Decimal(value)
        return True
    except decimal.InvalidOperation:
        return False


def load_sustax_file_safe(
    csv_stx: str,
    return_pandas_df: bool = True,
    return_metadata: bool = False,
) -> Any:
    """
    Carga un archivo Sustax de forma robusta, manejando fechas tipo M/D/YYYY
    y extrayendo metadatos básicos.

    Parámetros
    ----------
    csv_stx : str
        Ruta del archivo CSV de Sustax.
    return_pandas_df : bool, default=True
        Si True, devuelve un DataFrame con series organizadas por variable/escenario.
    return_metadata : bool, default=False
        Si True, devuelve también un diccionario con metadatos (lat/lon).

    Retorna
    -------
    Si return_pandas_df=True:
        df_vals, df_metrics [, metadata]
    Si return_pandas_df=False:
        dt_dict, metrics_dict, time_array [, metadata]
    """
    with open(csv_stx, "r", encoding=None) as fobj:
        data = list(csv.reader(fobj, delimiter=","))

    idx_data = [i for i in range(len(data)) if "Data requested:" in data[i]][0] + 4
    all_data = data[idx_data:]
    all_data_vars = data[idx_data - 3]
    all_data_scenarios = data[idx_data - 2]

    metadata: dict[str, float] = {}
    if return_metadata:
        for row in data:
            if any("longitude" in h.lower() for h in row):
                metadata["lon"] = [float(v) for v in row if _isfloat(v)][0]
            if any("latitude" in h.lower() for h in row):
                metadata["lat"] = [float(v) for v in row if _isfloat(v)][0]

    dt: dict[str, dict[str, list[float]]] = {}
    for c in range(len(all_data_vars)):
        if all_data_vars[c] != "":
            dt.setdefault(all_data_vars[c], {}).update({all_data_scenarios[c]: []})

    time = []
    for row in all_data:
        time.append(pd.to_datetime(row[0], format="%m/%d/%Y"))
        for c in range(len(row)):
            if ("SSP" in all_data_scenarios[c]) or ("ERA" in all_data_scenarios[c]):
                try:
                    dt[all_data_vars[c]][all_data_scenarios[c]].append(
                        float(row[c]) if row[c] != "" else np.nan
                    )
                except Exception:
                    pass

    time = np.array(time, dtype="datetime64[D]")
    dt = {k: {s: np.asarray(dt[k][s]) for s in dt[k]} for k in dt}

    if return_pandas_df:
        dfs = [
            pd.Series(
                {d: v for d, v in zip(time, dt[var][scenario])}
            ).to_frame(name=f"{var} [{scenario}]")
            for var in dt
            for scenario in dt[var]
        ]

        df_vals = dfs[0].join(dfs[1:]) if len(dfs) > 1 else dfs[0]
        df_metrics = pd.DataFrame()

        if return_metadata:
            return df_vals, df_metrics, metadata
        return df_vals, df_metrics

    if return_metadata:
        return dt, {}, time, metadata
    return dt, {}, time
