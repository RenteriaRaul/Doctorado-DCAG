
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("results/streamlit/sustax/BY_SCENARIO")
ESCENARIO = "SSP119"
LAT = 18.9695
LON = -103.7812
HISTORICO = (1981, 2010)
FUTURO = (2026, 2055)
UMBRAL = 50.0

archivos = sorted(BASE.glob(f"*__{ESCENARIO}__mm.csv"))
if not archivos:
    raise FileNotFoundError("No se encontraron archivos SSP119")

coincidencias = []
for archivo in archivos:
    datos = pd.read_csv(archivo, nrows=1)
    if not {"lat", "lon"}.issubset(datos.columns):
        continue
    lat = float(datos["lat"].iloc[0])
    lon = float(datos["lon"].iloc[0])
    if np.isclose(lat, LAT, atol=0.00005) and np.isclose(lon, LON, atol=0.00005):
        coincidencias.append(archivo)

if len(coincidencias) != 1:
    raise ValueError(
        f"Se esperaban 1 archivo y se encontraron {len(coincidencias)}"
    )

archivo = coincidencias[0]
df = pd.read_csv(archivo)
print("Archivo:", archivo.name)
print("Columnas:", df.columns.tolist())
print("Registros:", len(df))


df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["precipitacion_mm"] = pd.to_numeric(
    df["pr-Total Precipitation"], errors="coerce"
)

if df["date"].isna().any():
    raise ValueError("Existen fechas inválidas")

if df["precipitacion_mm"].isna().any():
    raise ValueError("Existen precipitaciones inválidas")

if df["date"].duplicated().any():
    raise ValueError("Existen fechas duplicadas")

df = df.sort_values("date").copy()
df["anio"] = df["date"].dt.year

def calcular_indicadores(datos, inicio, fin):
    periodo = datos.loc[
        datos["anio"].between(inicio, fin)
    ].copy()

    anios_esperados = fin - inicio + 1
    anios_disponibles = periodo["anio"].nunique()

    if anios_disponibles != anios_esperados:
        raise ValueError(
            f"Cobertura incompleta: {inicio}-{fin}. "
            f"Esperados: {anios_esperados}; "
            f"disponibles: {anios_disponibles}"
        )

    serie = periodo["precipitacion_mm"]

    rx1day_anual = periodo.groupby("anio")[
        "precipitacion_mm"
    ].max()

    r50_anual = periodo.groupby("anio")[
        "precipitacion_mm"
    ].apply(lambda x: (x >= UMBRAL).sum())

    return {
        "P95": float(serie.quantile(0.95)),
        "P99": float(serie.quantile(0.99)),
        "Rx1day": float(rx1day_anual.mean()),
        "R50mm": float(r50_anual.mean())
    }

historico = calcular_indicadores(df, *HISTORICO)
futuro = calcular_indicadores(df, *FUTURO)

filas = []
for indicador in historico:
    h = historico[indicador]
    f = futuro[indicador]
    cambio = f - h
    porcentaje = (cambio / h) * 100 if h != 0 else np.nan

    filas.append({
        "Indicador": indicador,
        "Histórico": h,
        "Futuro": f,
        "Cambio absoluto": cambio,
        "Cambio (%)": porcentaje
    })

resultado = pd.DataFrame(filas)

print("\nVALIDACIÓN INDEPENDIENTE SSP119")
print(f"Punto: {LAT}, {LON}")
print(f"Referencia: {HISTORICO[0]}-{HISTORICO[1]}")
print(f"Futuro: {FUTURO[0]}-{FUTURO[1]}")
print(f"Umbral: {UMBRAL} mm/día\n")

print(resultado.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))


print("\nVERIFICACIÓN DE CONTINUIDAD DIARIA")

for nombre, (inicio, fin) in [
    ("Histórico", HISTORICO),
    ("Futuro", FUTURO)
]:
    fechas_esperadas = pd.date_range(
        start=f"{inicio}-01-01",
        end=f"{fin}-12-31",
        freq="D"
    )
    fechas_reales = pd.DatetimeIndex(
        df.loc[df["anio"].between(inicio, fin), "date"]
    ).normalize()

    faltantes = fechas_esperadas.difference(fechas_reales)
    adicionales = fechas_reales.difference(fechas_esperadas)

    print(f"\n{nombre}: {inicio}-{fin}")
    print(f"Días esperados: {len(fechas_esperadas)}")
    print(f"Días registrados: {len(fechas_reales)}")
    print(f"Días faltantes: {len(faltantes)}")
    print(f"Fechas fuera del periodo: {len(adicionales)}")
    print(
        "Estado:",
        "COMPLETO" if len(faltantes) == 0
        and len(adicionales) == 0
        and len(fechas_reales) == len(fechas_esperadas)
        else "REVISAR"
    )
