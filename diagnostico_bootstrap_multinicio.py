
"""Diagnóstico independiente: bootstrap GEV convencional frente a multiinicio.

Ejecutar desde la raíz de Doctorado-DCAG:
    python scripts/diagnostico_bootstrap_multinicio.py --n-boot 50

No modifica el motor oficial ni los archivos de entrada.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genextreme as gev

RAIZ = Path(__file__).resolve().parent
ENTRADA = RAIZ / 'VALIDATED' / 'diagnostico_gev'
SALIDA = ENTRADA / 'bootstrap_multinicio'
ESTACIONES = {'6001': 'ARMERIA', '6021': 'QUESERIA'}
PERIODOS = np.array([2, 5, 10, 25, 50, 100], dtype=float)
SHAPE_BOUNDS = (-0.35, 0.35)
MAX_REL_FACTOR = 10.0

# Para diagnóstico inicial se prueban menos inicios que
# en el diagnóstico exhaustivo.
FORMAS_INICIALES = (-0.3, 0.0, 0.3, 1.0)
FACTORES_ESCALA = (0.7, 1.3)


def loglik(x, c, loc, scale):
    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        return -np.inf

    if np.any(1.0 - c * (x - loc) / scale <= 0):
        return -np.inf

    vals = gev.logpdf(x, c, loc=loc, scale=scale)

    return float(vals.sum()) if np.isfinite(vals).all() else -np.inf


def multistart(x):
    """Devuelve mejor ajuste convergente y válido; incluye SciPy como candidato."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = gev.fit(x)

    candidatos = []
    ll_base = loglik(x, *base)

    if np.isfinite(ll_base):
        candidatos.append((ll_base, tuple(map(float, base)), 'scipy'))

    sd = float(np.std(x, ddof=1))

    if not np.isfinite(sd) or sd <= 0:
        return max(candidatos, default=None, key=lambda v: v[0])

    med = float(np.median(x))
    minimo = float(np.min(x))
    maximo = float(np.max(x))

    for c0 in FORMAS_INICIALES:
        for factor in FACTORES_ESCALA:
            scale0 = max(sd * factor, 1e-6)
            loc0 = med
            margen = 0.05 * sd

            if c0 > 0:
                loc0 = max(loc0, maximo - scale0 / c0 + margen)
            elif c0 < 0:
                loc0 = min(loc0, minimo - scale0 / c0 - margen)

            def objetivo(par):
                c, loc, logscale = par

                with np.errstate(over='ignore', invalid='ignore'):
                    scale = np.exp(logscale)

                ll = loglik(x, c, loc, scale)

                return -ll if np.isfinite(ll) else 1e12

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                res = minimize(
                    objetivo,
                    [c0, loc0, np.log(scale0)],
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3500,
                        'maxfev': 7000,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                    },
                )

            with np.errstate(over='ignore', invalid='ignore'):
                params = (
                    float(res.x[0]),
                    float(res.x[1]),
                    float(np.exp(res.x[2])),
                )

            ll = loglik(x, *params)

            if res.success and np.isfinite(ll):
                candidatos.append((ll, params, 'multinicio'))

    return max(candidatos, default=None, key=lambda v: v[0])


def evaluar(x, parametros, max_obs):
    if parametros is None:
        return 'ajuste_invalido', None

    c, loc, scale = parametros

    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        return 'ajuste_invalido', None

    if not SHAPE_BOUNDS[0] <= c <= SHAPE_BOUNDS[1]:
        return 'shape_fuera_limites', None

    q = gev.ppf(
        1 - 1 / PERIODOS,
        c,
        loc=loc,
        scale=scale,
    )

    if not np.isfinite(q).all():
        return 'cuantil_no_finito', None

    if np.any(q <= 0):
        return 'cuantil_no_positivo', None

    if np.max(q) > MAX_REL_FACTOR * max_obs:
        return 'cuantil_excesivo', None

    return 'aceptada', q


def intervalos(matriz, alpha=0.05):
    if not matriz:
        return (
            np.full(len(PERIODOS), np.nan),
            np.full(len(PERIODOS), np.nan),
        )

    a = np.asarray(matriz)

    return (
        np.percentile(a, 100 * alpha / 2, axis=0),
        np.percentile(a, 100 * (1 - alpha / 2), axis=0),
    )


def ejecutar_estacion(codigo, nombre, n_boot, seed):
    archivo = ENTRADA / f'{codigo}_cobertura_anual.csv'

    if not archivo.exists():
        raise FileNotFoundError(f'No existe {archivo}')

    df = pd.read_csv(archivo)

    if not {'year', 'maximo_anual_mm'}.issubset(df.columns):
        raise ValueError(f'Columnas requeridas ausentes en {archivo}')

    datos = (
        pd.to_numeric(df['maximo_anual_mm'], errors='coerce')
        .dropna()
        .to_numpy(float)
    )

    datos = datos[np.isfinite(datos)]

    if len(datos) < 10:
        raise ValueError(
            f'{codigo}: solo {len(datos)} máximos anuales válidos'
        )

    max_obs = float(datos.max())
    rng = np.random.default_rng(seed)

    muestras = [
        rng.choice(datos, size=len(datos), replace=True)
        for _ in range(n_boot)
    ]

    print(
        f'\n{codigo} - {nombre}: {len(datos)} años, '
        f'{n_boot} réplicas',
        flush=True,
    )

    resultados = []
    cuantiles = {'scipy': [], 'multinicio': []}
    tiempos = {'scipy': 0.0, 'multinicio': 0.0}

    for i, muestra in enumerate(muestras, start=1):
        for metodo in ('scipy', 'multinicio'):
            t0 = time.perf_counter()

            try:
                if metodo == 'scipy':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        pars = tuple(map(float, gev.fit(muestra)))

                    fuente = 'scipy'

                else:
                    mejor = multistart(muestra)
                    pars = mejor[1] if mejor else None
                    fuente = mejor[2] if mejor else 'sin_solucion'

                estado, q = evaluar(muestra, pars, max_obs)

            except Exception as exc:
                pars = None
                fuente = 'error'
                estado = 'excepcion'
                q = None

                print(
                    f'  Réplica {i}, {metodo}: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )

            tiempos[metodo] += time.perf_counter() - t0

            if q is not None:
                cuantiles[metodo].append(q)

            fila = {
                'station': codigo,
                'nombre': nombre,
                'replica': i,
                'metodo': metodo,
                'estado': estado,
                'origen_mejor': fuente,
                'shape': pars[0] if pars else np.nan,
                'loc': pars[1] if pars else np.nan,
                'scale': pars[2] if pars else np.nan,
                'loglik': loglik(muestra, *pars) if pars else np.nan,
            }

            resultados.append(fila)

        if i % 10 == 0 or i == n_boot:
            print(f'  Procesadas {i}/{n_boot}', flush=True)

    detalles = pd.DataFrame(resultados)
    resumen = []

    for metodo in ('scipy', 'multinicio'):
        parte = detalles[detalles['metodo'] == metodo]
        aceptadas = len(cuantiles[metodo])
        low, high = intervalos(cuantiles[metodo])

        fila = {
            'station': codigo,
            'nombre': nombre,
            'metodo': metodo,
            'n_anios': len(datos),
            'n_boot': n_boot,
            'semilla': seed,
            'aceptadas': aceptadas,
            'aceptacion_pct': 100 * aceptadas / n_boot,
            'segundos': round(tiempos[metodo], 2),
        }

        for estado, cantidad in parte['estado'].value_counts().items():
            fila[f'n_{estado}'] = int(cantidad)

        for t, li, ls in zip(PERIODOS, low, high):
            fila[f'T{int(t)}_IC_inf_mm'] = li
            fila[f'T{int(t)}_IC_sup_mm'] = ls

        resumen.append(fila)

    return detalles, pd.DataFrame(resumen)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-boot', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261027)

    args = parser.parse_args()

    if args.n_boot < 1:
        parser.error('--n-boot debe ser >= 1')

    SALIDA.mkdir(parents=True, exist_ok=True)
    resumenes = []

    for codigo, nombre in ESTACIONES.items():
        detalles, resumen = ejecutar_estacion(
            codigo,
            nombre,
            args.n_boot,
            args.seed,
        )

        detalles.to_csv(
            SALIDA / f'{codigo}_detalle_bootstrap.csv',
            index=False,
            encoding='utf-8-sig',
        )

        resumen.to_csv(
            SALIDA / f'{codigo}_resumen_bootstrap.csv',
            index=False,
            encoding='utf-8-sig',
        )

        resumenes.append(resumen)

    master = pd.concat(resumenes, ignore_index=True)

    master.to_csv(
        SALIDA / 'MASTER_comparacion_bootstrap.csv',
        index=False,
        encoding='utf-8-sig',
    )

    columnas = [
        'station',
        'metodo',
        'n_anios',
        'n_boot',
        'aceptadas',
        'aceptacion_pct',
        'n_shape_fuera_limites',
        'segundos',
        'T2_IC_inf_mm',
        'T2_IC_sup_mm',
        'T100_IC_inf_mm',
        'T100_IC_sup_mm',
    ]

    print('\nRESUMEN COMPARATIVO')
    print(master.reindex(columns=columnas).to_string(index=False))

    print(f'\nArchivos guardados en: {SALIDA}')
    print(
        'AVISO: prueba exploratoria; '
        '50 réplicas no bastan para inferencia definitiva.'
    )


if __name__ == '__main__':
    main()
