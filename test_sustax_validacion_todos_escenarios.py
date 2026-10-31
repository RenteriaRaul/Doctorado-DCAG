
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("results/streamlit/sustax/BY_SCENARIO")
SALIDA = Path("results/tests/sustax_validacion_general")
ESCENARIOS = ["SSP119", "SSP126", "SSP245", "SSP370",
              "SSP434", "SSP460", "SSP585"]
HISTORICO = (1981, 2010)
FUTURO = (2026, 2055)
UMBRAL = 50.0

def calcular(datos, inicio, fin):
    sub = datos.loc[datos["date"].dt.year.between(inicio, fin)].copy()
    fechas = pd.date_range(f"{inicio}-01-01", f"{fin}-12-31", freq="D")
    reales = pd.DatetimeIndex(sub["date"])
    faltantes = fechas.difference(reales)
    duplicados = int(reales.duplicated().sum())
    invalidos = int(sub["precipitacion"].isna().sum())
    negativos = int((sub["precipitacion"] < 0).sum())

    completo = (
        len(faltantes) == 0 and duplicados == 0
        and invalidos == 0 and negativos == 0
        and len(sub) == len(fechas)
    )
    resultado = {
        "dias_esperados": len(fechas),
        "dias_registrados": len(sub),
        "dias_faltantes": len(faltantes),
        "duplicados": duplicados,
        "valores_invalidos": invalidos,
        "valores_negativos": negativos,
        "completo": completo
    }
    if not completo:
        return resultado, None

    serie = sub["precipitacion"]
    anuales = sub.groupby(sub["date"].dt.year)["precipitacion"]
    indicadores = {
        "p95_daily_mm": float(serie.quantile(0.95)),
        "p99_daily_mm": float(serie.quantile(0.99)),
        "rx1day_mm": float(anuales.max().mean()),
        "r50mm_days_per_year": float(
            anuales.apply(lambda x: (x >= UMBRAL).sum()).mean()
        )
    }
    return resultado, indicadores

def main():
    if not BASE.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {BASE}")
    archivos = sorted(BASE.glob("*__mm.csv"))
    if not archivos:
        raise FileNotFoundError("No hay series BY_SCENARIO")

    SALIDA.mkdir(parents=True, exist_ok=True)
    diagnosticos = []
    resultados = []
    claves = set()

    for archivo in archivos:
        partes = archivo.stem.split("__")
        if len(partes) != 3:
            continue
        punto, escenario, unidad = partes
        if escenario not in ESCENARIOS or unidad != "mm":
            continue

        clave = (punto, escenario)
        if clave in claves:
            raise ValueError(f"Serie repetida: {clave}")
        claves.add(clave)

        fila = {"punto": punto, "escenario": escenario,
                "archivo": archivo.name, "estado": "PENDIENTE"}
        try:
            df = pd.read_csv(archivo)
            columnas = [c for c in df.columns
                        if "precipitation" in c.lower()]
            if len(columnas) != 1:
                raise ValueError(
                    f"Se esperaba una columna de precipitación: {columnas}"
                )
            if "date" not in df.columns:
                raise ValueError("No existe la columna date")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["precipitacion"] = pd.to_numeric(
                df[columnas[0]], errors="coerce"
            )
            fechas_invalidas = int(df["date"].isna().sum())
            duplicados_globales = int(df["date"].duplicated().sum())
            negativos_globales = int((df["precipitacion"] < 0).sum())
            valores_invalidos = int(df["precipitacion"].isna().sum())

            fila.update({
                "registros": len(df),
                "fechas_invalidas": fechas_invalidas,
                "duplicados_globales": duplicados_globales,
                "negativos_globales": negativos_globales,
                "valores_invalidos_globales": valores_invalidos
            })
            if any([fechas_invalidas, duplicados_globales,
                    negativos_globales, valores_invalidos]):
                fila["estado"] = "ERROR_INTEGRIDAD"
                diagnosticos.append(fila)
                continue

            df["date"] = df["date"].dt.normalize()
            if df["date"].duplicated().any():
                fila["estado"] = "ERROR_FECHAS_DUPLICADAS"
                diagnosticos.append(fila)
                continue

            hist_check, hist = calcular(df, *HISTORICO)
            fut_check, fut = calcular(df, *FUTURO)
            for prefijo, chequeo in [
                ("historico", hist_check), ("futuro", fut_check)
            ]:
                fila.update({
                    f"{prefijo}_{k}": v for k, v in chequeo.items()
                })

            if hist is None or fut is None:
                fila["estado"] = "COBERTURA_INSUFICIENTE"
            else:
                fila["estado"] = "VALIDADO_INDEPENDIENTE"
                for indicador in hist:
                    h, f = hist[indicador], fut[indicador]
                    resultados.append({
                        "punto": punto,
                        "escenario": escenario,
                        "indicador": indicador,
                        "historico": h,
                        "futuro": f,
                        "cambio_absoluto": f - h,
                        "cambio_porcentual": (
                            100 * (f - h) / h if h != 0 else np.nan
                        )
                    })
        except Exception as exc:
            fila["estado"] = "ERROR"
            fila["detalle_error"] = str(exc)

        diagnosticos.append(fila)

    diag = pd.DataFrame(diagnosticos)
    res = pd.DataFrame(resultados)
    if diag.empty:
        raise ValueError("No se detectaron series SSP para validar")

    diag.to_csv(SALIDA / "diagnostico_series.csv", index=False)
    res.to_csv(SALIDA / "indicadores_independientes.csv", index=False)

    print("\nVALIDACIÓN AUTOMATIZADA SUSTAX")
    print("=" * 55)
    print("Series SSP analizadas:", len(diag))
    print("Puntos geográficos:", diag["punto"].nunique())
    print("Escenarios:", diag["escenario"].nunique())
    print("\nRESULTADOS POR ESCENARIO")
    resumen = (
        diag.groupby(["escenario", "estado"])
        .size().unstack(fill_value=0)
    )
    print(resumen.to_string())
    print("\nSeries con periodos completos:",
          int((diag["estado"] == "VALIDADO_INDEPENDIENTE").sum()))
    print("Indicadores calculados:", len(res))
    print("Reportes:", SALIDA.resolve())

    errores = diag["estado"].str.startswith("ERROR").sum()
    print("\nErrores de integridad:", int(errores))
    if errores:
        print("\nARCHIVOS CON ERROR")
        print(diag.loc[
            diag["estado"].str.startswith("ERROR"),
            ["archivo", "estado", "detalle_error"]
            if "detalle_error" in diag.columns
            else ["archivo", "estado"]
        ].to_string(index=False))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
