"""Diagnóstico de cobertura diaria de series Sustax BY_SCENARIO (mm).

No modifica archivos de entrada ni calcula indicadores climáticos.
"""
from pathlib import Path
import pandas as pd

SSP = ("SSP119", "SSP126", "SSP245", "SSP370", "SSP434", "SSP460", "SSP585")


def _validar_periodo(periodo, nombre):
    if len(periodo) != 2:
        raise ValueError(f"{nombre} debe contener dos años")
    a, b = map(int, periodo)
    if a > b:
        raise ValueError(f"{nombre}: inicio posterior al fin")
    return a, b


def _diagnosticar_periodo(df, inicio, fin):
    esperado = pd.date_range(f"{inicio}-01-01", f"{fin}-12-31", freq="D")
    sub = df.loc[df["date"].between(esperado[0], esperado[-1])]
    fechas = pd.DatetimeIndex(sub["date"].dropna().unique())
    validas = pd.DatetimeIndex(sub.loc[sub["valid_precip"], "date"].unique())
    faltantes = esperado.difference(validas)
    duplicadas = int(sub["date"].duplicated(keep=False).sum())
    invalidas = int((~sub["valid_precip"]).sum())
    return {
        "dias_esperados": len(esperado),
        "dias_validos": len(validas.intersection(esperado)),
        "dias_faltantes": len(faltantes),
        "filas_duplicadas": duplicadas,
        "filas_precip_invalida": invalidas,
        "completo": len(faltantes) == 0 and duplicadas == 0 and invalidas == 0,
    }


def evaluar_cobertura_ssp(
    carpeta: str | Path,
    periodo_historico: tuple[int, int] = (1981, 2010),
    periodo_futuro: tuple[int, int] = (2026, 2055),
    escenarios=SSP,
) -> pd.DataFrame:
    """Una fila por archivo SSP; elegible solo con cobertura diaria íntegra.

    `punto` conserva el identificador original del nombre de archivo.
    Los errores se registran como exclusiones, sin detener el inventario.
    """
    h0, h1 = _validar_periodo(periodo_historico, "periodo_historico")
    f0, f1 = _validar_periodo(periodo_futuro, "periodo_futuro")
    if f0 <= h1:
        raise ValueError("El periodo futuro debe comenzar después del histórico")
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(carpeta)

    filas = []
    for archivo in sorted(carpeta.glob("*.csv")):
        partes = archivo.stem.split("__")
        if len(partes) != 3 or partes[1] not in escenarios or partes[2] != "mm":
            continue
        punto, escenario, _ = partes
        fila = {"punto": punto, "escenario": escenario, "archivo": archivo.name}
        try:
            df = pd.read_csv(archivo)
            if "date" not in df.columns:
                raise ValueError("falta columna date")
            # Misma prioridad de selección que projection.py.
            meta = {"sustax_total", "lat", "lon", "date", "year", "month"}
            candidatas = [c for c in df.columns if c not in meta]
            pp = next((c for c in candidatas if "tp" in c.lower() and "precip" in c.lower()), None)
            if pp is None:
                pp = next((c for c in candidatas if "pr" in c.lower() and "precip" in c.lower()), None)
            if pp is None:
                pp = next((c for c in candidatas if "precip" in c.lower()), None)
            if pp is None and len(candidatas) == 1:
                pp = candidatas[0]
            if pp is None:
                raise ValueError("columna de precipitación no identificada")

            # Fechas ISO de los CSV consolidados; errores NO se descartan silenciosamente.
            fechas = pd.to_datetime(df["date"], errors="coerce")
            if fechas.isna().any():
                raise ValueError(f"{int(fechas.isna().sum())} fechas inválidas")
            df["date"] = fechas.dt.normalize()
            valores = pd.to_numeric(df[pp], errors="coerce")
            df["valid_precip"] = valores.notna() & (valores >= 0)
            hist = _diagnosticar_periodo(df, h0, h1)
            fut = _diagnosticar_periodo(df, f0, f1)
            for prefijo, info in (("hist", hist), ("fut", fut)):
                fila.update({f"{prefijo}_{k}": v for k, v in info.items()})
            fila["elegible"] = bool(hist["completo"] and fut["completo"])
            razones = []
            for nombre, info in (("histórico", hist), ("futuro", fut)):
                if not info["completo"]:
                    razones.append(
                        f"{nombre}: {info['dias_faltantes']} días faltantes, "
                        f"{info['filas_duplicadas']} filas duplicadas, "
                        f"{info['filas_precip_invalida']} valores inválidos"
                    )
            fila["motivo_exclusion"] = "; ".join(razones)
        except Exception as exc:
            fila.update(elegible=False, motivo_exclusion=f"ERROR: {exc}")
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_puntos_comunes(diagnostico: pd.DataFrame) -> pd.DataFrame:
    """Muestra por punto en cuántos escenarios aparece y es elegible."""
    if diagnostico.empty:
        return pd.DataFrame(columns=["punto", "escenarios_elegibles", "n_escenarios"])
    ok = diagnostico.loc[diagnostico["elegible"] == True]
    return (ok.groupby("punto")["escenario"]
            .agg(escenarios_elegibles=lambda s: ",".join(sorted(set(s))),
                 n_escenarios="nunique")
            .reset_index().sort_values("punto"))
