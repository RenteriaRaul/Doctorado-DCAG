from scripts.gev_quality import clasificar_calidad_bootstrap


def validar_caso(
    nombre,
    aceptadas,
    solicitadas,
    clasificacion_esperada,
    mostrar_ic_esperado,
):
    resultado = clasificar_calidad_bootstrap(
        aceptadas=aceptadas,
        solicitadas=solicitadas,
    )

    print()
    print("-" * 72)
    print(nombre)
    print("-" * 72)
    print(
        f"Réplicas válidas: {resultado['aceptadas']} de "
        f"{resultado['solicitadas']}"
    )
    print(f"Porcentaje: {resultado['porcentaje']:.1f}%")
    print(f"Clasificación: {resultado['clasificacion']}")
    print(f"Mostrar IC: {resultado['mostrar_ic']}")
    print(f"Mensaje: {resultado['mensaje']}")

    if resultado["clasificacion"] != clasificacion_esperada:
        raise ValueError(
            f"{nombre}: clasificación incorrecta."
        )

    if resultado["mostrar_ic"] != mostrar_ic_esperado:
        raise ValueError(
            f"{nombre}: política de visualización incorrecta."
        )

    return resultado


def main():
    print("=" * 72)
    print("PRUEBA — POLÍTICA DE CALIDAD BOOTSTRAP GEV")
    print("=" * 72)

    pena = validar_caso(
        nombre="6030 — PEÑA COLORADA",
        aceptadas=12,
        solicitadas=500,
        clasificacion_esperada="No confiable para inferencia",
        mostrar_ic_esperado=False,
    )

    comala = validar_caso(
        nombre="6007 — COMALA",
        aceptadas=474,
        solicitadas=500,
        clasificacion_esperada="Buena",
        mostrar_ic_esperado=True,
    )

    validar_caso(
        nombre="Caso frontera 25%",
        aceptadas=125,
        solicitadas=500,
        clasificacion_esperada="Limitada",
        mostrar_ic_esperado=True,
    )

    validar_caso(
        nombre="Caso frontera 50%",
        aceptadas=250,
        solicitadas=500,
        clasificacion_esperada="Aceptable",
        mostrar_ic_esperado=True,
    )

    validar_caso(
        nombre="Caso frontera 75%",
        aceptadas=375,
        solicitadas=500,
        clasificacion_esperada="Buena",
        mostrar_ic_esperado=True,
    )

    if round(pena["porcentaje"], 1) != 2.4:
        raise ValueError(
            "El porcentaje de PEÑA COLORADA no es 2.4%."
        )

    if round(comala["porcentaje"], 1) != 94.8:
        raise ValueError(
            "El porcentaje de COMALA no es 94.8%."
        )

    print()
    print("=" * 72)
    print("VALIDACIONES FINALES")
    print("=" * 72)
    print()
    print("PEÑA COLORADA: IC oculto correctamente.")
    print("COMALA: IC mostrado correctamente.")
    print("Umbral 25%: OK")
    print("Umbral 50%: OK")
    print("Umbral 75%: OK")
    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()