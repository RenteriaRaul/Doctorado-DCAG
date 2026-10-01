import csv
import decimal
from typing import Any, Optional

import numpy as np
import pandas as pd


def _isfloat(value: str) -> bool:
    try:
        decimal.Decimal(str(value).strip())
        return True
    except (decimal.InvalidOperation, ValueError, TypeError):
        return False


def _find_row(data: list[list[str]], text: str) -> Optional[int]:
    text = text.lower()
    for i, row in enumerate(data):
        if any(text in str(cell).lower() for cell in row):
            return i
    return None


def _value_after_label(row: list[str], label: str) -> Optional[str]:
    label = label.lower()
    for i, value in enumerate(row):
        if label in str(value).lower():
            for candidate in row[i + 1:]:
                candidate = str(candidate).strip().strip('"')
                if candidate:
                    return candidate
    return None


def _extract_first_float(row: list[str]) -> float:
    for value in row:
        value = str(value).strip().strip('"')
        if _isfloat(value):
            return float(value)
    return np.nan


def _parse_sustax_date(value: str, declared_format: Optional[str] = None) -> pd.Timestamp:
    value = str(value).strip()

    format_map = {
        "YYYY/MM/DD": "%Y/%m/%d",
        "YYYY-MM-DD": "%Y-%m-%d",
        "M/D/YYYY": "%m/%d/%Y",
        "MM/DD/YYYY": "%m/%d/%Y",
        "D/M/YYYY": "%d/%m/%Y",
        "DD/MM/YYYY": "%d/%m/%Y",
    }

    if declared_format:
        fmt = format_map.get(str(declared_format).strip())
        if fmt:
            dt = pd.to_datetime(value, format=fmt, errors="coerce")
            if not pd.isna(dt):
                return dt

    # Compatibilidad con exportaciones Sustax usadas previamente.
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        dt = pd.to_datetime(value, format=fmt, errors="coerce")
        if not pd.isna(dt):
            return dt

    return pd.to_datetime(value, errors="coerce")


def load_sustax_file_safe(
    csv_stx: str,
    return_pandas_df: bool = True,
    return_metadata: bool = False,
) -> Any:
    """
    Carga una exportación original de Sustax de forma robusta.

    Soporta el formato original actual (por ejemplo YYYY/MM/DD)
    y mantiene compatibilidad con exportaciones previas M/D/YYYY.

    Si return_pandas_df=True:
        df_vals, df_metrics [, metadata]

    Si return_pandas_df=False:
        dt_dict, metrics_dict, time_array [, metadata]
    """
    with open(csv_stx, "r", encoding="utf-8-sig", newline="") as fobj:
        data = list(csv.reader(fobj, delimiter=","))

    if not data:
        raise ValueError("El archivo Sustax está vacío.")

    first_cell = str(data[0][0]).strip() if data[0] else ""
    if "SUSTAX Data Export" not in first_cell:
        raise ValueError("El archivo no corresponde a una exportación original de Sustax.")

    idx_requested = _find_row(data, "Data requested:")
    if idx_requested is None:
        raise ValueError("No se encontró el bloque 'Data requested:'.")

    idx_vars = idx_requested + 1
    idx_scenarios = idx_requested + 2
    idx_units = idx_requested + 3
    idx_data = idx_requested + 4

    if idx_data >= len(data):
        raise ValueError("El bloque de datos Sustax está incompleto.")

    all_data_vars = data[idx_vars]
    all_data_scenarios = data[idx_scenarios]
    all_data_units = data[idx_units]
    all_data = data[idx_data:]

    # Formato temporal declarado por Sustax.
    declared_time_format = None
    idx_time_format = _find_row(data, "Data time format:")
    if idx_time_format is not None:
        declared_time_format = _value_after_label(
            data[idx_time_format], "Data time format:"
        )

    metadata: dict[str, Any] = {}

    if return_metadata:
        metadata["data_time_format"] = declared_time_format

        idx_creation = _find_row(data, "Creation date:")
        metadata["creation_date"] = (
            _value_after_label(data[idx_creation], "Creation date:")
            if idx_creation is not None
            else None
        )

        idx_osm = _find_row(data, "Approximate OSM location:")
        metadata["osm_location"] = (
            _value_after_label(data[idx_osm], "Approximate OSM location:")
            if idx_osm is not None
            else None
        )

        idx_dataset = _find_row(data, "Dataset:")
        metadata["dataset"] = (
            _value_after_label(data[idx_dataset], "Dataset:")
            if idx_dataset is not None
            else None
        )

        metadata["lon"] = np.nan
        metadata["lat"] = np.nan

        for row in data:
            if any("longitude" in str(h).lower() for h in row):
                val = _extract_first_float(row)
                if np.isfinite(val):
                    metadata["lon"] = val

            if any("latitude" in str(h).lower() for h in row):
                val = _extract_first_float(row)
                if np.isfinite(val):
                    metadata["lat"] = val

        # Variables de suelo: se conservan tal como vienen en la exportación.
        metadata["soil_data"] = {}
        idx_soil = _find_row(data, "Soil variables:")
        if idx_soil is not None and idx_soil + 1 < len(data):
            names = data[idx_soil][1:]
            values = data[idx_soil + 1][1:]

            for name, value in zip(names, values):
                name = str(name).strip()
                value = str(value).strip()

                if not name:
                    continue

                metadata["soil_data"][name] = (
                    float(value) if _isfloat(value) else (value or None)
                )

    dt: dict[str, dict[str, list[float]]] = {}
    units: dict[tuple[str, str], str] = {}

    max_cols = max(
        len(all_data_vars),
        len(all_data_scenarios),
        len(all_data_units),
    )

    for c in range(1, max_cols):
        var = all_data_vars[c].strip() if c < len(all_data_vars) else ""
        scenario = (
            all_data_scenarios[c].strip()
            if c < len(all_data_scenarios)
            else ""
        )
        unit = all_data_units[c].strip() if c < len(all_data_units) else ""

        if not var:
            continue

        if not (scenario.startswith("SSP") or scenario.startswith("ERA")):
            continue

        dt.setdefault(var, {})[scenario] = []
        units[(var, scenario)] = unit

    if not dt:
        raise ValueError("No se detectaron series ERA/SSP en el archivo Sustax.")

    time = []

    for row in all_data:
        if not row or not str(row[0]).strip():
            continue

        date_value = _parse_sustax_date(
            row[0],
            declared_format=declared_time_format,
        )

        if pd.isna(date_value):
            continue

        time.append(date_value)

        for c in range(1, min(len(row), max_cols)):
            var = all_data_vars[c].strip() if c < len(all_data_vars) else ""
            scenario = (
                all_data_scenarios[c].strip()
                if c < len(all_data_scenarios)
                else ""
            )

            if var not in dt or scenario not in dt[var]:
                continue

            value = str(row[c]).strip()
            dt[var][scenario].append(
                float(value) if _isfloat(value) else np.nan
            )

        # Garantiza longitudes iguales incluso si una fila viene incompleta.
        for var in dt:
            for scenario in dt[var]:
                if len(dt[var][scenario]) < len(time):
                    dt[var][scenario].append(np.nan)

    if not time:
        raise ValueError("No se encontraron fechas climáticas válidas.")

    time = np.array(time, dtype="datetime64[D]")
    dt = {
        var: {
            scenario: np.asarray(values, dtype=float)
            for scenario, values in scenarios.items()
        }
        for var, scenarios in dt.items()
    }

    if return_metadata:
        metadata["units"] = {
            f"{var} [{scenario}]": unit
            for (var, scenario), unit in units.items()
        }
        metadata["scenarios"] = sorted(
            {scenario for scenarios in dt.values() for scenario in scenarios}
        )
        metadata["variables"] = list(dt.keys())
        metadata["date_start"] = pd.Timestamp(time.min())
        metadata["date_end"] = pd.Timestamp(time.max())
        metadata["n_dates"] = len(time)

    if return_pandas_df:
        dfs = []

        for var in dt:
            for scenario in dt[var]:
                values = dt[var][scenario]

                if len(values) != len(time):
                    raise ValueError(
                        f"Longitud inconsistente en {var} [{scenario}]: "
                        f"{len(values)} valores para {len(time)} fechas."
                    )

                dfs.append(
                    pd.Series(
                        values,
                        index=pd.to_datetime(time),
                        name=f"{var} [{scenario}]",
                    ).to_frame()
                )

        if not dfs:
            raise ValueError("No se pudieron construir series climáticas.")

        df_vals = pd.concat(dfs, axis=1)
        df_vals.index.name = "date"
        df_metrics = pd.DataFrame()

        if return_metadata:
            return df_vals, df_metrics, metadata

        return df_vals, df_metrics

    if return_metadata:
        return dt, {}, time, metadata

    return dt, {}, time
