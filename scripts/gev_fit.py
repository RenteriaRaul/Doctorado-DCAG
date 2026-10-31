
"""Ajuste GEV multiinicio para el motor CONAGUA.

Usa una función objetivo finita y conserva la convención c
de scipy.stats.genextreme.
"""

import warnings
import numpy as np

from scipy.optimize import minimize
from scipy.stats import genextreme as gev


def ajustar_gev_multinicio(datos):
    """
    Devuelve (c, loc, scale) de la mejor solución
    numéricamente válida.

    Utiliza tres puntos iniciales para reducir el riesgo
    de convergencia hacia un óptimo local.

    No impone límites arbitrarios al parámetro c.
    """

    x = np.asarray(datos, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 3:
        raise ValueError(
            "Se requieren al menos tres máximos anuales."
        )

    # Normalización para mejorar la estabilidad numérica
    center = float(np.median(x))
    spread = max(float(np.std(x)), 1e-3)

    y = (x - center) / spread

    penalty = 1e12

    def objective(p):
        c, loc, log_scale = p

        if (
            not np.all(np.isfinite(p))
            or abs(log_scale) > 20
        ):
            return penalty

        scale = np.exp(log_scale)

        with np.errstate(all="ignore"):
            z = (y - loc) / scale

            support = 1 - c * z

            if (
                not np.all(np.isfinite(support))
                or np.any(support <= 0)
            ):
                # Penalización finita por violación
                # del soporte de la distribución
                violation = np.minimum(support, 0)

                return penalty + float(
                    np.sum(violation * violation)
                )

            logpdf = gev.logpdf(
                y,
                c,
                loc=loc,
                scale=scale,
            )

        if not np.all(np.isfinite(logpdf)):
            return penalty

        return -float(np.sum(logpdf))

    candidates = []

    # Diferentes puntos iniciales para el parámetro shape
    for c0 in (-0.2, 0.3, 0.8):

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                RuntimeWarning,
            )

            res = minimize(
                objective,
                np.array([c0, 0.0, 0.0]),
                method="Nelder-Mead",
                options={
                    "maxiter": 450,
                    "maxfev": 750,
                    "xatol": 2e-5,
                    "fatol": 2e-5,
                },
            )

        if (
            np.isfinite(res.fun)
            and res.fun < penalty
        ):
            c, loc_y, log_scale = res.x

            params = (
                float(c),
                float(center + spread * loc_y),
                float(spread * np.exp(log_scale)),
            )

            if (
                np.isfinite(params).all()
                and params[2] > 0
            ):
                candidates.append(
                    (
                        float(res.fun),
                        params,
                    )
                )

    if not candidates:
        raise ValueError(
            "No se obtuvo un ajuste GEV válido "
            "con multiinicio."
        )

    return min(
        candidates,
        key=lambda item: item[0],
    )[1]
