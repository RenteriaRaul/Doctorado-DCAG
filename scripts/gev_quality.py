import numpy as np


def clasificar_calidad_bootstrap(aceptadas, solicitadas):
    """
    Clasifica la calidad del Bootstrap robusto según la
    proporción de réplicas aceptadas.
    """
    if solicitadas is None:
        raise ValueError("solicitadas no puede ser None.")

    solicitadas = int(solicitadas)

    if solicitadas <= 0:
        raise ValueError("solicitadas debe ser mayor que 0.")

    if aceptadas is None:
        return {
            "aceptadas": None,
            "solicitadas": solicitadas,
            "proporcion": np.nan,
            "porcentaje": np.nan,
            "clasificacion": "No disponible",
            "mostrar_ic": False,
            "mensaje": (
                "No hay información suficiente para evaluar "
                "la calidad del Bootstrap."
            ),
        }

    aceptadas = float(aceptadas)

    if not np.isfinite(aceptadas):
        return {
            "aceptadas": None,
            "solicitadas": solicitadas,
            "proporcion": np.nan,
            "porcentaje": np.nan,
            "clasificacion": "No disponible",
            "mostrar_ic": False,
            "mensaje": (
                "No hay información suficiente para evaluar "
                "la calidad del Bootstrap."
            ),
        }

    if aceptadas < 0:
        raise ValueError("aceptadas no puede ser negativa.")

    if aceptadas > solicitadas:
        raise ValueError(
            "aceptadas no puede ser mayor que solicitadas."
        )

    aceptadas_int = int(aceptadas)
    proporcion = aceptadas / solicitadas
    porcentaje = proporcion * 100.0

    if proporcion >= 0.75:
        clasificacion = "Buena"
        mostrar_ic = True
        mensaje = (
            "La proporción de réplicas Bootstrap válidas es alta. "
            "El intervalo de confianza puede representarse de forma "
            "normal, sujeto a las demás limitaciones del ajuste GEV."
        )

    elif proporcion >= 0.50:
        clasificacion = "Aceptable"
        mostrar_ic = True
        mensaje = (
            "La mayoría de las réplicas Bootstrap fueron aceptadas. "
            "El intervalo de confianza puede mostrarse, aunque debe "
            "considerarse que una parte de los reajustes fue descartada."
        )

    elif proporcion >= 0.25:
        clasificacion = "Limitada"
        mostrar_ic = True
        mensaje = (
            "Menos de la mitad de las réplicas Bootstrap fueron "
            "aceptadas. El intervalo puede mostrarse, pero debe "
            "interpretarse con cautela."
        )

    else:
        clasificacion = "No confiable para inferencia"
        mostrar_ic = False
        mensaje = (
            "La proporción de réplicas Bootstrap válidas es inferior "
            "al 25%. El intervalo de confianza no se representa como "
            "banda de incertidumbre porque su soporte Bootstrap es "
            "insuficiente para una inferencia robusta."
        )

    return {
        "aceptadas": aceptadas_int,
        "solicitadas": solicitadas,
        "proporcion": proporcion,
        "porcentaje": porcentaje,
        "clasificacion": clasificacion,
        "mostrar_ic": mostrar_ic,
        "mensaje": mensaje,
    }


def obtener_politica_ic_bootstrap(aceptadas, solicitadas):
    """Alias semántico para la capa de visualización."""
    return clasificar_calidad_bootstrap(
        aceptadas=aceptadas,
        solicitadas=solicitadas,
    )