
"""
DIAGNÓSTICO DE OPTIMIZACIÓN GEV
Proyecto Doctorado-DCAG

Objetivo:
Evaluar la sensibilidad del ajuste GEV a distintos
puntos iniciales y comparar la log-verosimilitud.

Estaciones:
6001 - Armería
6021 - Quesería

No modifica archivos originales ni el motor oficial.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.station_analysis import ajustar_gev

ENTRADA = ROOT / "VALIDATED" / "diagnostico_gev"
SALIDA = ENTRADA / "optimizacion"

ESTACIONES = {
    "6001": "Armería",
    "6021": "Quesería",
}

UMBRALES = {
    "Sin filtro": 0,
    "Cobertura 80%": 80,
    "Cobertura 90%": 90,
    "Cobertura 95%": 95,
}

PERIODOS = np.array([2, 5, 10, 25, 50, 100], dtype=float)

# Puntos iniciales de forma en la convención de SciPy.
FORMAS_INICIALES = [-0.30, -0.10, 0.0, 0.10, 0.30, 0.50, 1.0, 3.0, 5.0]

# Escalas relativas a la desviación estándar muestral.
FACTORES_ESCALA = [0.5, 1.0, 1.5]

MIN_ANIOS = 10


def cargar_datos(codigo):
    archivo = ENTRADA / f"{codigo}_cobertura_anual.csv"

    if not archivo.exists():
        raise FileNotFoundError(archivo)

    df = pd.read_csv(archivo)

    for columna in [
        "year",
        "cobertura_pct",
        "maximo_anual_mm",
    ]:
        df[columna] = pd.to_numeric(
            df[columna], errors="coerce"
        )

    return df


def logverosimilitud(datos, c, loc, scale):
    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        return -np.inf

    # Para la parametrización de SciPy, el soporte exige:
    # 1 - c * (x - loc) / scale > 0.
    soporte = 1.0 - c * (datos - loc) / scale

    if np.any(soporte <= 0):
        return -np.inf

    logpdf = gev.logpdf(
        datos, c, loc=loc, scale=scale
    )

    if not np.isfinite(logpdf).all():
        return -np.inf

    return float(np.sum(logpdf))


def funcion_objetivo(parametros, datos):
    c, loc, log_scale = parametros
    scale = np.exp(log_scale)

    ll = logverosimilitud(datos, c, loc, scale)

    if not np.isfinite(ll):
        return 1e12

    return -ll


def optimizar_multinicio(datos):
    media = float(np.mean(datos))
    desviacion = float(np.std(datos, ddof=1))

    if desviacion <= 0 or not np.isfinite(desviacion):
        raise ValueError("Desviación estándar inválida")

    resultados = []

    for c0 in FORMAS_INICIALES:
        for factor in FACTORES_ESCALA:
            scale0 = max(desviacion * factor, 1e-6)

            # Ubicación inicial cercana a la mediana.
            loc0 = float(np.median(datos))

            # Aseguramos que el punto inicial respete el soporte.
            margen = 0.05 * desviacion

            if c0 > 0:
                loc0 = max(
                    loc0,
                    float(np.max(datos) - scale0 / c0 + margen),
                )
            elif c0 < 0:
                loc0 = min(
                    loc0,
                    float(np.min(datos) - scale0 / c0 - margen),
                )

            inicial = [
                c0,
                loc0,
                np.log(scale0),
            ]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                resultado = minimize(
                    funcion_objetivo,
                    inicial,
                    args=(datos,),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 15000,
                        "maxfev": 30000,
                        "xatol": 1e-8,
                        "fatol": 1e-8,
                    },
                )

            c, loc, log_scale = resultado.x
            scale = float(np.exp(log_scale))

            ll = logverosimilitud(
                datos, c, loc, scale
            )

            fila = {
                "c_inicial": c0,
                "factor_escala_inicial": factor,
                "convergencia_reportada": bool(resultado.success),
                "mensaje_optimizador": str(resultado.message),
                "gev_shape": float(c),
                "gev_loc": float(loc),
                "gev_scale": scale,
                "log_verosimilitud": ll,
                "n_iteraciones": int(resultado.nit),
            }

            for periodo in PERIODOS:
                columna = f"T{int(periodo)}_mm"

                if np.isfinite(ll):
                    fila[columna] = float(
                        gev.ppf(
                            1 - 1 / periodo,
                            c,
                            loc=loc,
                            scale=scale,
                        )
                    )
                else:
                    fila[columna] = np.nan

            resultados.append(fila)

    return pd.DataFrame(resultados)


def analizar_estacion(codigo, nombre):
    df = cargar_datos(codigo)

    resumen = []
    detalles = []

    print("\n" + "=" * 85)
    print(f"ESTACIÓN {codigo} - {nombre.upper()}")
    print("=" * 85)

    for escenario, umbral in UMBRALES.items():
        seleccion = df[
            (df["cobertura_pct"] >= umbral)
            & df["maximo_anual_mm"].notna()
        ]

        datos = seleccion[
            "maximo_anual_mm"
        ].to_numpy(dtype=float)

        datos = datos[np.isfinite(datos)]

        if len(datos) < MIN_ANIOS:
            print(f"{escenario}: años insuficientes")
            continue

        # Ajuste utilizado por el motor oficial.
        c_ref, loc_ref, scale_ref = ajustar_gev(datos)

        ll_ref = logverosimilitud(
            datos, c_ref, loc_ref, scale_ref
        )

        # Ajustes con múltiples puntos iniciales.
        tabla = optimizar_multinicio(datos)

        tabla["station"] = codigo
        tabla["escenario"] = escenario
        tabla["n_anios"] = len(datos)

        detalles.append(tabla)

        validos = tabla[
            tabla["convergencia_reportada"]
            & np.isfinite(tabla["log_verosimilitud"])
        ].copy()

        if validos.empty:
            print(
                f"{escenario}: ningún ajuste mult_inicio "
                "convergió correctamente"
            )
            continue

        mejor = validos.sort_values(
            "log_verosimilitud",
            ascending=False,
        ).iloc[0]

        diferencia_ll = (
            mejor["log_verosimilitud"] - ll_ref
        )

        fila = {
            "station": codigo,
            "nombre": nombre,
            "escenario": escenario,
            "n_anios": len(datos),
            "shape_scipy": c_ref,
            "shape_multinicio": mejor["gev_shape"],
            "loc_scipy": loc_ref,
            "loc_multinicio": mejor["gev_loc"],
            "scale_scipy": scale_ref,
            "scale_multinicio": mejor["gev_scale"],
            "ll_scipy": ll_ref,
            "ll_multinicio": mejor["log_verosimilitud"],
            "diferencia_ll": diferencia_ll,
            "ajustes_convergentes": len(validos),
            "ajustes_evaluados": len(tabla),
        }

        for periodo in PERIODOS:
            fila[f"T{int(periodo)}_scipy"] = float(
                gev.ppf(
                    1 - 1 / periodo,
                    c_ref,
                    loc=loc_ref,
                    scale=scale_ref,
                )
            )

            fila[f"T{int(periodo)}_multinicio"] = (
                mejor[f"T{int(periodo)}_mm"]
            )

        resumen.append(fila)

        print(f"\n{escenario} ({len(datos)} años)")
        print(f"Shape SciPy:      {c_ref:.6f}")
        print(
            f"Shape multiinicio: "
            f"{mejor['gev_shape']:.6f}"
        )
        print(f"LogLik SciPy:     {ll_ref:.4f}")
        print(
            f"LogLik multiinicio: "
            f"{mejor['log_verosimilitud']:.4f}"
        )
        print(f"Diferencia LogLik: {diferencia_ll:.4f}")
        print(
            f"Convergencias: {len(validos)}/{len(tabla)}"
        )

    SALIDA.mkdir(parents=True, exist_ok=True)

    if detalles:
        pd.concat(
            detalles, ignore_index=True
        ).to_csv(
            SALIDA / f"{codigo}_optimizaciones.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if resumen:
        tabla_resumen = pd.DataFrame(resumen)

        tabla_resumen.to_csv(
            SALIDA / f"{codigo}_resumen_optimizacion.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return tabla_resumen

    return pd.DataFrame()


def main():
    print("=" * 85)
    print("DIAGNÓSTICO DE OPTIMIZACIÓN GEV")
    print("=" * 85)

    tablas = []

    for codigo, nombre in ESTACIONES.items():
        try:
            resultado = analizar_estacion(codigo, nombre)

            if not resultado.empty:
                tablas.append(resultado)

        except Exception as error:
            print(f"Error en estación {codigo}: {error}")

    if tablas:
        SALIDA.mkdir(parents=True, exist_ok=True)

        pd.concat(
            tablas, ignore_index=True
        ).to_csv(
            SALIDA / "MASTER_optimizacion_gev.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\nDIAGNÓSTICO FINALIZADO")
        print(f"Resultados guardados en: {SALIDA}")
    else:
        print("\nNo se generaron resultados.")


if __name__ == "__main__":
    main()
