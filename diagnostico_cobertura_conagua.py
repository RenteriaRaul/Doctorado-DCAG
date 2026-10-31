"""Diagnóstico de cobertura diaria CONAGUA para comparar periodos comunes.

Ejecutar desde la raíz del proyecto Doctorado-DCAG:
    python diagnostico_cobertura_conagua.py

Requiere: pandas, openpyxl y scripts.conagua_reader.
No modifica los Excel originales ni el módulo de excedencias.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from scripts.conagua_reader import leer_lote_conagua

CARPETA_CONAGUA = Path(r"G:\My Drive\Doctorado\Probabilidad\Precipitación\estaciones_conagua_excel\Colima")
SALIDA = RAIZ_PROYECTO / "results" / "diagnostico_cobertura_conagua.xlsx"
PERIODOS = [(1991, 2020), (2000, 2024)]
UMBRALES_COBERTURA = (0.80, 0.90)


def limpiar_serie(datos):
    """Devuelve fechas diarias únicas con precipitación válida (>=0)."""
    if not isinstance(datos, pd.DataFrame) or not {"date", "pp"}.issubset(datos.columns):
        raise ValueError("La estación no tiene las columnas 'date' y 'pp'.")
    df = datos[["date", "pp"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["pp"] = pd.to_numeric(df["pp"], errors="coerce")
    df = df.dropna(subset=["date", "pp"])
    df = df.loc[df["pp"] >= 0]
    # Igual que exceedance.py: para fechas duplicadas, conservar el máximo diario.
    return df.groupby("date", as_index=False)["pp"].max()


def dias_del_periodo(inicio, fin):
    return (pd.Timestamp(fin) - pd.Timestamp(inicio)).days + 1


def main():
    if not CARPETA_CONAGUA.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {CARPETA_CONAGUA}")

    estaciones, metadata_df, log_lectura = leer_lote_conagua(
        carpeta=str(CARPETA_CONAGUA), patron="*.xlsx"
    )
    if not estaciones:
        raise RuntimeError("El lector no devolvió estaciones. Revisar log de lectura.")

    anuales, periodos, estaciones_resumen, errores = [], [], [], []
    for estacion in estaciones:
        metadata = estacion.get("metadata", {})
        clave = str(metadata.get("station") or metadata.get("archivo") or "Sin clave")
        try:
            serie = limpiar_serie(estacion["data"])
            fechas = pd.DatetimeIndex(serie["date"])
            estaciones_resumen.append({
                "station": clave, "nombre": metadata.get("nombre"),
                "municipio": metadata.get("municipio"), "situacion": metadata.get("situacion"),
                "latitud": metadata.get("latitud"), "longitud": metadata.get("longitud"),
                "dias_validos_historicos": len(fechas),
                "fecha_inicio": fechas.min() if len(fechas) else pd.NaT,
                "fecha_fin": fechas.max() if len(fechas) else pd.NaT,
            })

            for anio in range(min(a for a, _ in PERIODOS), max(b for _, b in PERIODOS) + 1):
                inicio, fin = f"{anio}-01-01", f"{anio}-12-31"
                n = int(((fechas >= inicio) & (fechas <= fin)).sum())
                esperados = dias_del_periodo(inicio, fin)
                anuales.append({
                    "station": clave, "anio": anio, "dias_validos": n,
                    "dias_esperados": esperados, "cobertura_pct": round(100 * n / esperados, 2),
                    "cumple_80": n / esperados >= 0.80,
                    "cumple_90": n / esperados >= 0.90,
                })

            for anio_inicio, anio_fin in PERIODOS:
                inicio, fin = f"{anio_inicio}-01-01", f"{anio_fin}-12-31"
                dentro = (fechas >= inicio) & (fechas <= fin)
                n = int(dentro.sum())
                esperados = dias_del_periodo(inicio, fin)
                cobertura = n / esperados
                fila = {
                    "station": clave, "periodo": f"{anio_inicio}-{anio_fin}",
                    "dias_validos": n, "dias_esperados": esperados,
                    "cobertura_pct": round(100 * cobertura, 2),
                    "anios_con_datos": int(fechas[dentro].year.nunique()),
                }
                for umbral in UMBRALES_COBERTURA:
                    fila[f"cumple_{int(umbral * 100)}"] = cobertura >= umbral
                periodos.append(fila)
        except Exception as exc:
            errores.append({"station": clave, "error": str(exc)})

    df_anual = pd.DataFrame(anuales)
    df_periodos = pd.DataFrame(periodos)
    df_estaciones = pd.DataFrame(estaciones_resumen)
    df_errores = pd.DataFrame(errores, columns=["station", "error"])

    resumen = []
    if not df_periodos.empty:
        for periodo, grupo in df_periodos.groupby("periodo", sort=True):
            for umbral in UMBRALES_COBERTURA:
                col = f"cumple_{int(umbral * 100)}"
                resumen.append({
                    "periodo": periodo, "cobertura_minima_pct": int(umbral * 100),
                    "estaciones_elegibles": int(grupo[col].sum()),
                    "estaciones_evaluadas": len(grupo),
                    "criterio": "Cobertura diaria total del periodo; no exige mínimo por año",
                })
    df_resumen = pd.DataFrame(resumen)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(SALIDA, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="Resumen_periodos", index=False)
        df_periodos.to_excel(writer, sheet_name="Cobertura_periodos", index=False)
        df_anual.to_excel(writer, sheet_name="Cobertura_anual", index=False)
        df_estaciones.to_excel(writer, sheet_name="Estaciones", index=False)
        df_errores.to_excel(writer, sheet_name="Errores_procesamiento", index=False)
        if isinstance(log_lectura, pd.DataFrame):
            log_lectura.to_excel(writer, sheet_name="Log_lectura", index=False)
        if isinstance(metadata_df, pd.DataFrame):
            metadata_df.to_excel(writer, sheet_name="Metadatos", index=False)

    print("\nDIAGNÓSTICO DE COBERTURA CONAGUA")
    print(f"Estaciones leídas: {len(estaciones)}")
    print(f"Estaciones evaluadas: {len(df_estaciones)}")
    print(f"Errores de procesamiento: {len(df_errores)}")
    print(df_resumen.to_string(index=False) if not df_resumen.empty else "Sin resultados")
    print(f"\nArchivo generado: {SALIDA}")
    print("NOTA: Elegibilidad preliminar por cobertura total; revisaremos continuidad anual y distribución espacial.")


if __name__ == "__main__":
    main()
