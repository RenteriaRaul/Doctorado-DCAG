import numpy as np
from scipy.stats import genextreme as gev


def bootstrap_robusto(
    datos,
    niveles_retorno,
    n_boot=500,
    alpha=0.05,
    rng=None,
    shape_bounds=(-0.35, 0.35),
    max_rel_factor=10.0,
):
    """
    Bootstrap no paramétrico robusto para intervalos de confianza de niveles de retorno GEV.

    Filtra réplicas inestables usando estas reglas:
    - parámetros finitos
    - scale > 0
    - shape dentro de límites razonables
    - cuantiles finitos y positivos
    - cuantiles no mayores que max_rel_factor * máximo observado

    Parámetros
    ----------
    datos : array-like
        Serie de máximos anuales.
    niveles_retorno : array-like
        Periodos de retorno, por ejemplo [2, 5, 10, 25, 50, 100].
    n_boot : int
        Número de réplicas bootstrap.
    alpha : float
        Nivel de significancia. 0.05 -> IC del 95%.
    rng : numpy.random.Generator o None
        Generador aleatorio. Si es None, se crea uno nuevo.
    shape_bounds : tuple(float, float)
        Límites aceptables para el parámetro shape.
    max_rel_factor : float
        Factor máximo permitido respecto al máximo observado.

    Retorna
    -------
    low : np.ndarray
        Límite inferior del IC para cada nivel de retorno.
    high : np.ndarray
        Límite superior del IC para cada nivel de retorno.
    n_aceptadas : int
        Número de réplicas bootstrap aceptadas tras filtrado.
    """
    datos = np.asarray(datos, dtype=float)
    niveles_retorno = np.asarray(niveles_retorno, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    if datos.ndim != 1:
        raise ValueError("`datos` debe ser un arreglo unidimensional.")
    if len(datos) == 0:
        raise ValueError("`datos` no puede estar vacío.")
    if np.any(~np.isfinite(datos)):
        raise ValueError("`datos` contiene valores no finitos.")
    if np.any(niveles_retorno <= 1):
        raise ValueError("Todos los niveles de retorno deben ser mayores que 1.")

    max_obs = np.nanmax(datos)
    replicas_validas = []

    for _ in range(n_boot):
        muestra = rng.choice(datos, size=len(datos), replace=True)

        try:
            c_b, loc_b, scale_b = gev.fit(muestra)

            if not np.isfinite([c_b, loc_b, scale_b]).all():
                continue
            if scale_b <= 0:
                continue
            if not (shape_bounds[0] <= c_b <= shape_bounds[1]):
                continue

            q = gev.ppf(1 - 1 / niveles_retorno, c_b, loc=loc_b, scale=scale_b)

            if not np.isfinite(q).all():
                continue
            if np.any(q <= 0):
                continue
            if np.nanmax(q) > max_rel_factor * max_obs:
                continue

            replicas_validas.append(q)

        except Exception:
            continue

    if len(replicas_validas) == 0:
        low = np.full_like(niveles_retorno, np.nan, dtype=float)
        high = np.full_like(niveles_retorno, np.nan, dtype=float)
        n_aceptadas = 0
    else:
        replicas_validas = np.asarray(replicas_validas, dtype=float)
        low = np.nanpercentile(replicas_validas, 100 * (alpha / 2), axis=0)
        high = np.nanpercentile(replicas_validas, 100 * (1 - alpha / 2), axis=0)
        n_aceptadas = replicas_validas.shape[0]

    return low, high, n_aceptadas


def bootstrap_parametrico(
    c,
    loc,
    scale,
    niveles_retorno,
    n_muestra,
    n_boot=500,
    alpha=0.05,
    rng=None,
):
    """
    Bootstrap paramétrico para intervalos de confianza de niveles de retorno GEV.

    Genera muestras sintéticas desde una GEV ajustada, vuelve a ajustar GEV
    a cada muestra y obtiene percentiles de los niveles de retorno.

    Parámetros
    ----------
    c, loc, scale : float
        Parámetros del ajuste GEV base.
    niveles_retorno : array-like
        Periodos de retorno, por ejemplo [2, 5, 10, 25, 50, 100].
    n_muestra : int
        Número de observaciones por muestra simulada. Debe coincidir con
        el número de máximos anuales de la estación.
    n_boot : int
        Número de réplicas bootstrap.
    alpha : float
        Nivel de significancia. 0.05 -> IC del 95%.
    rng : numpy.random.Generator o None
        Generador aleatorio. Si es None, se crea uno nuevo.

    Retorna
    -------
    low : np.ndarray
        Límite inferior del IC para cada nivel de retorno.
    high : np.ndarray
        Límite superior del IC para cada nivel de retorno.
    n_aceptadas : int
        Número de réplicas bootstrap aceptadas.
    """
    niveles_retorno = np.asarray(niveles_retorno, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    if not np.isfinite([c, loc, scale]).all():
        raise ValueError("Los parámetros GEV deben ser finitos.")
    if scale <= 0:
        raise ValueError("`scale` debe ser mayor que 0.")
    if n_muestra <= 0:
        raise ValueError("`n_muestra` debe ser mayor que 0.")
    if np.any(niveles_retorno <= 1):
        raise ValueError("Todos los niveles de retorno deben ser mayores que 1.")

    replicas_validas = []

    for _ in range(n_boot):
        try:
            muestra_sim = gev.rvs(c, loc=loc, scale=scale, size=n_muestra, random_state=rng)
            c_b, loc_b, scale_b = gev.fit(muestra_sim)

            if not np.isfinite([c_b, loc_b, scale_b]).all():
                continue
            if scale_b <= 0:
                continue

            q = gev.ppf(1 - 1 / niveles_retorno, c_b, loc=loc_b, scale=scale_b)

            if not np.isfinite(q).all():
                continue
            if np.any(q <= 0):
                continue

            replicas_validas.append(q)

        except Exception:
            continue

    if len(replicas_validas) == 0:
        low = np.full_like(niveles_retorno, np.nan, dtype=float)
        high = np.full_like(niveles_retorno, np.nan, dtype=float)
        n_aceptadas = 0
    else:
        replicas_validas = np.asarray(replicas_validas, dtype=float)
        low = np.nanpercentile(replicas_validas, 100 * (alpha / 2), axis=0)
        high = np.nanpercentile(replicas_validas, 100 * (1 - alpha / 2), axis=0)
        n_aceptadas = replicas_validas.shape[0]

    return low, high, n_aceptadas
