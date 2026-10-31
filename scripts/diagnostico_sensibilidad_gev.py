
"""
Diagnóstico de sensibilidad del ajuste GEV.

Compara ajustes de máximos anuales con diferentes
umbrales de cobertura temporal.

Estaciones:
    6001 - Armería
    6021 - Quesería

No modifica el motor GEV ni los archivos originales.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev

# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.station_analysis import (
    ajustar_gev,
    calcular_niveles_retorno,
)

DIR_ENTRADA = (
    ROOT
    / "VALIDATED"
    / "diagnostico_gev"
)

DIR_SALIDA = (
    ROOT
    / "VALIDATED"
    / "diagnostico_gev"
    / "sensibilidad"
)

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

PERIODOS_RETORNO = np.array(
    [2, 5, 10, 25, 50, 100],
    dtype=float,
)

MIN_ANIOS = 10

# Referencia exploratoria del filtro que utiliza
# actualmente el bootstrap robusto.
SHAPE_MIN = -0.35
SHAPE_MAX = 0.35


# ------------------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------------------

def cargar_diagnostico(codigo):
    archivo = (
        DIR_ENTRADA
        / f"{codigo}_cobertura_anual.csv"
    )

    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe el diagnóstico: {archivo}"
        )

    df = pd.read_csv(archivo)

    columnas = [
        "year",
        "cobertura_pct",
        "maximo_anual_mm",
    ]

    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}"
        )

    for columna in columnas:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    df = df.dropna(subset=["year"]).copy()

    if df["year"].duplicated().any():
        raise ValueError(
            f"La estación {codigo} tiene años duplicados."
        )

    return df.sort_values("year")


# ------------------------------------------------------------
# 3. AJUSTE GEV Y DIAGNÓSTICOS
# ------------------------------------------------------------

def evaluar_ajuste(datos, umbral):
    seleccion = datos[
        (datos["cobertura_pct"] >= umbral)
        & datos["maximo_anual_mm"].notna()
    ].copy()

    valores = seleccion[
        "maximo_anual_mm"
    ].to_numpy(dtype=float)

    valores = valores[np.isfinite(valores)]

    n_anios = len(valores)

    resultado = {
        "n_anios": n_anios,
        "max_observado_mm": (
            float(np.max(valores))
            if n_anios else np.nan
        ),
        "mediana_maximos_mm": (
            float(np.median(valores))
            if n_anios else np.nan
        ),
        "gev_shape": np.nan,
        "gev_loc": np.nan,
        "gev_scale": np.nan,
        "shape_en_rango_bootstrap": False,
        "log_verosimilitud": np.nan,
        "estado": "",
    }

    for periodo in PERIODOS_RETORNO:
        resultado[f"T{int(periodo)}_mm"] = np.nan

    if n_anios < MIN_ANIOS:
        resultado["estado"] = (
            f"Insuficientes años: {n_anios}"
        )
        return resultado, seleccion

    try:
        # Misma función utilizada por el motor oficial.
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")

            c, loc, scale = ajustar_gev(valores)

            niveles = calcular_niveles_retorno(
                c=c,
                loc=loc,
                scale=scale,
                niveles_retorno=PERIODOS_RETORNO,
            )

        niveles = np.asarray(niveles, dtype=float)

        if not np.isfinite(niveles).all():
            resultado["estado"] = (
                "Niveles de retorno no finitos"
            )
            return resultado, seleccion

        # Diagnóstico complementario de verosimilitud.
        logpdf = gev.logpdf(
            valores,
            c,
            loc=loc,
            scale=scale,
        )

        loglik = (
            float(np.sum(logpdf))
            if np.isfinite(logpdf).all()
            else np.nan
        )

        resultado.update(
            {
                "gev_shape": float(c),
                "gev_loc": float(loc),
                "gev_scale": float(scale),
                "shape_en_rango_bootstrap": bool(
                    SHAPE_MIN <= c <= SHAPE_MAX
                ),
                "log_verosimilitud": loglik,
                "estado": (
                    "Revisar parámetro shape"
                    if not (
                        SHAPE_MIN <= c <= SHAPE_MAX
                    )
                    else "Ajuste calculado"
                ),
            }
        )

        for periodo, nivel in zip(
            PERIODOS_RETORNO, niveles
        ):
            resultado[
                f"T{int(periodo)}_mm"
            ] = float(nivel)

        if avisos:
            resultado["estado"] += (
                f" | {len(avisos)} advertencias"
            )

    except Exception as error:
        resultado["estado"] = (
            f"Error de ajuste: {error}"
        )

    return resultado, seleccion


# ------------------------------------------------------------
# 4. ANÁLISIS POR ESTACIÓN
# ------------------------------------------------------------

def analizar_estacion(codigo, nombre):
    datos = cargar_diagnostico(codigo)

    resultados = []
    detalle_anios = []

    print("\n" + "=" * 85)
    print(f"ESTACIÓN {codigo} - {nombre.upper()}")
    print("=" * 85)

    for escenario, umbral in UMBRALES.items():
        resultado, seleccion = evaluar_ajuste(
            datos,
            umbral,
        )

        resultado["station"] = codigo
        resultado["nombre"] = nombre
        resultado["escenario"] = escenario
        resultado["umbral_cobertura_pct"] = umbral

        resultados.append(resultado)

        for _, fila in seleccion.iterrows():
            detalle_anios.append(
                {
                    "station": codigo,
                    "escenario": escenario,
                    "year": int(fila["year"]),
                    "cobertura_pct": fila["cobertura_pct"],
                    "maximo_anual_mm": (
                        fila["maximo_anual_mm"]
                    ),
                }
            )

    tabla = pd.DataFrame(resultados)

    columnas_resumen = [
        "escenario",
        "n_anios",
        "gev_shape",
        "gev_loc",
        "gev_scale",
        "shape_en_rango_bootstrap",
        "estado",
    ]

    print("\nPARÁMETROS GEV")
    print(
        tabla[columnas_resumen].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    columnas_niveles = [
        "escenario",
        "n_anios",
        "T2_mm",
        "T5_mm",
        "T10_mm",
        "T25_mm",
        "T50_mm",
        "T100_mm",
    ]

    print("\nNIVELES DE RETORNO (mm)")
    print(
        tabla[columnas_niveles].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    referencia = tabla.iloc[0]

    cambios = []

    for _, fila in tabla.iloc[1:].iterrows():
        cambio = {
            "station": codigo,
            "escenario": fila["escenario"],
            "n_anios": fila["n_anios"],
        }

        for periodo in PERIODOS_RETORNO:
            columna = f"T{int(periodo)}_mm"
            base = referencia[columna]
            nuevo = fila[columna]

            if (
                np.isfinite(base)
                and np.isfinite(nuevo)
                and base != 0
            ):
                variacion = (
                    (nuevo - base) / abs(base)
                ) * 100
            else:
                variacion = np.nan

            cambio[
                f"T{int(periodo)}_variacion_pct"
            ] = variacion

        cambios.append(cambio)

    tabla_cambios = pd.DataFrame(cambios)

    print("\nVARIACIÓN RESPECTO AL AJUSTE SIN FILTRO (%)")
    print(
        tabla_cambios.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    DIR_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla.to_csv(
        DIR_SALIDA / f"{codigo}_sensibilidad_gev.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_cambios.to_csv(
        DIR_SALIDA / f"{codigo}_variaciones_gev.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(detalle_anios).to_csv(
        DIR_SALIDA / f"{codigo}_anios_utilizados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return tabla


# ------------------------------------------------------------
# 5. EJECUCIÓN
# ------------------------------------------------------------

def main():
    print("=" * 85)
    print("DIAGNÓSTICO DE SENSIBILIDAD GEV")
    print("=" * 85)

    print(
        "\nSe comparan los ajustes sin filtro y con "
        "coberturas mínimas de 80%, 90% y 95%."
    )

    print(
        "\nEste diagnóstico no modifica los resultados "
        "oficiales del proyecto."
    )

    tablas = []

    for codigo, nombre in ESTACIONES.items():
        try:
            tabla = analizar_estacion(
                codigo,
                nombre,
            )
            tablas.append(tabla)

        except Exception as error:
            print(
                f"\nNo fue posible analizar "
                f"{codigo}: {error}"
            )

    if tablas:
        consolidado = pd.concat(
            tablas,
            ignore_index=True,
        )

        consolidado.to_csv(
            DIR_SALIDA
            / "MASTER_sensibilidad_gev.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 85)
        print("DIAGNÓSTICO FINALIZADO")
        print("=" * 85)
        print(f"Resultados: {DIR_SALIDA}")

    else:
        print("\nNo se generaron resultados.")


if __name__ == "__main__":
    main()
