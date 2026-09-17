import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(valor):
    """Convierte tipos comunes de numpy/pandas a valores serializables."""
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (pd.Timestamp, datetime)):
        return valor.isoformat()
    raise TypeError(f"Tipo no serializable en JSON: {type(valor)}")


def _leer_csv_seguro(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def guardar_ejecucion_excedencia(
    resultados, log_df, metadata_df, carpeta_salida, carpeta_fuente,
    patron, threshold, consolidar_duplicados, raster_path=None,
    estado=None, metodo_interpolacion="linear", nx=300, ny=300,
    margin=0.0, fill_nearest=False, crs="EPSG:4326",
):
    """Guarda persistentemente una ejecución de excedencias."""
    if resultados is None or resultados.empty:
        raise ValueError("No hay resultados de excedencia para guardar.")

    carpeta_salida = Path(carpeta_salida)
    runs_dir = carpeta_salida / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    fecha = datetime.now()
    run_id_base = fecha.strftime("%Y%m%d_%H%M%S")
    run_id = run_id_base
    run_dir = runs_dir / run_id
    consecutivo = 1
    while run_dir.exists():
        run_id = f"{run_id_base}_{consecutivo:02d}"
        run_dir = runs_dir / run_id
        consecutivo += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    master_path = run_dir / "MASTER_excedencia_CONAGUA.csv"
    log_path = run_dir / "log_excedencia_CONAGUA.csv"
    metadata_path = run_dir / "metadata_estaciones_CONAGUA.csv"
    config_path = run_dir / "run_metadata.json"

    resultados.to_csv(master_path, index=False, encoding="utf-8-sig")
    (log_df if log_df is not None and not log_df.empty else pd.DataFrame()).to_csv(
        log_path, index=False, encoding="utf-8-sig"
    )
    (metadata_df if metadata_df is not None and not metadata_df.empty else pd.DataFrame()).to_csv(
        metadata_path, index=False, encoding="utf-8-sig"
    )

    raster_guardado = None
    if raster_path:
        raster_origen = Path(raster_path)
        if raster_origen.exists():
            raster_guardado = run_dir / "excedencia_interpolada.tif"
            shutil.copy2(raster_origen, raster_guardado)

    estaciones = int(resultados["station"].astype(str).nunique())

    estado_guardado = estado
    if not estado_guardado and "estado" in resultados.columns:
        estados = resultados["estado"].dropna().astype(str).str.strip()
        estados = sorted(estados[estados != ""].unique().tolist())
        if estados:
            estado_guardado = ", ".join(estados)

    if (
        not estado_guardado and metadata_df is not None
        and not metadata_df.empty and "estado" in metadata_df.columns
    ):
        estados = metadata_df["estado"].dropna().astype(str).str.strip()
        estados = sorted(estados[estados != ""].unique().tolist())
        if estados:
            estado_guardado = ", ".join(estados)

    config = {
        "run_id": run_id,
        "fecha_ejecucion": fecha.isoformat(),
        "modulo": "excedencias",
        "fuente": "CONAGUA",
        "estado": str(estado_guardado) if estado_guardado else None,
        "carpeta_fuente": str(carpeta_fuente),
        "patron": str(patron),
        "threshold_mm": float(threshold),
        "consolidar_duplicados": bool(consolidar_duplicados),
        "estaciones_procesadas": estaciones,
        "metodo_interpolacion": str(metodo_interpolacion),
        "nx": int(nx),
        "ny": int(ny),
        "margin": float(margin),
        "fill_nearest": bool(fill_nearest),
        "crs": str(crs),
        "master_path": str(master_path),
        "log_path": str(log_path),
        "metadata_path": str(metadata_path),
        "raster_path": str(raster_guardado) if raster_guardado else None,
        "run_dir": str(run_dir),
    }

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (carpeta_salida / "latest_run.json").write_text(
        json.dumps(
            {"run_id": run_id, "config_path": str(config_path)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return config


def listar_ejecuciones_excedencia(carpeta_salida):
    """Lista ejecuciones guardadas de la más reciente a la más antigua."""
    runs_dir = Path(carpeta_salida) / "runs"
    if not runs_dir.exists():
        return []

    ejecuciones = []
    for config_path in runs_dir.glob("*/run_metadata.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["_config_path"] = str(config_path)
            ejecuciones.append(config)
        except Exception:
            continue

    ejecuciones.sort(
        key=lambda item: item.get("fecha_ejecucion", ""),
        reverse=True,
    )
    return ejecuciones


def cargar_ejecucion_excedencia(config):
    """Carga una ejecución guardada para reutilizarla en Streamlit."""
    master_path = Path(config["master_path"])
    log_path = Path(config["log_path"])
    metadata_path = Path(config["metadata_path"])

    if not master_path.exists():
        raise FileNotFoundError(
            "No se encontró la tabla maestra de la ejecución seleccionada."
        )

    resultados = pd.read_csv(master_path)
    log_df = _leer_csv_seguro(log_path)
    metadata_df = _leer_csv_seguro(metadata_path)

    raster_path = config.get("raster_path")
    if raster_path and Path(raster_path).exists():
        raster_path = str(Path(raster_path))
    else:
        raster_path = None

    estaciones = int(
        config.get(
            "estaciones_procesadas",
            resultados["station"].nunique(),
        )
    )

    raster_metadata = {
        "threshold": float(config.get("threshold_mm", 50.0)),
        "method": config.get("metodo_interpolacion", "linear"),
        "nx": int(config.get("nx", 300)),
        "ny": int(config.get("ny", 300)),
        "margin": float(config.get("margin", 0.0)),
        "fill_nearest": bool(config.get("fill_nearest", False)),
        "crs": config.get("crs", "EPSG:4326"),
        "stations_valid": estaciones,
        "source": config.get("fuente", "CONAGUA"),
        "source_folder": config.get("carpeta_fuente"),
        "estado": config.get("estado"),
    }

    return {
        "resultados": resultados,
        "log_df": log_df,
        "out_master": str(master_path),
        "out_log": str(log_path),
        "metadata_df": metadata_df,
        "dir_in": config.get("carpeta_fuente"),
        "n_archivos": estaciones,
        "threshold": float(config.get("threshold_mm", 50.0)),
        "consolidar_duplicados": bool(
            config.get("consolidar_duplicados", True)
        ),
        "estado": config.get("estado"),
        "raster_path": raster_path,
        "raster_metadata": raster_metadata,
        "run_id": config.get("run_id"),
        "fecha_ejecucion": config.get("fecha_ejecucion"),
        "persistent": True,
    }


def formato_fecha_ejecucion(fecha_iso):
    """Formatea una fecha ISO para mostrarla al usuario."""
    if not fecha_iso:
        return "Fecha no disponible"
    try:
        fecha = datetime.fromisoformat(fecha_iso)
        return fecha.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return str(fecha_iso)
