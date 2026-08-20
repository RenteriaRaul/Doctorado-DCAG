from scripts.gev_quality import (
    clasificar_calidad_bootstrap,
)


def main():

    print("=" * 72)
    print("PRUEBA — INTEGRACIÓN DE CALIDAD BOOTSTRAP EN GEV")
    print("=" * 72)

    casos = [
        {
            "station": "6030",
            "nombre": "PEÑA COLORADA",
            "aceptadas": 12,
            "solicitadas": 500,
            "esperado": (
                "No confiable para inferencia"
            ),
            "mostrar_ic": False,
        },
        {
            "station": "6007",
            "nombre": "COMALA",
            "aceptadas": 474,
            "solicitadas": 500,
            "esperado": "Buena",
            "mostrar_ic": True,
        },
    ]

    for caso in casos:

        resultado = (
            clasificar_calidad_bootstrap(
                aceptadas=caso[
                    "aceptadas"
                ],
                solicitadas=caso[
                    "solicitadas"
                ],
            )
        )

        print()
        print(
            f"{caso['station']} — "
            f"{caso['nombre']}"
        )

        print(
            "Clasificación:",
            resultado[
                "clasificacion"
            ],
        )

        print(
            "Mostrar IC:",
            resultado[
                "mostrar_ic"
            ],
        )

        print(
            "Porcentaje:",
            f"{resultado['porcentaje']:.1f}%",
        )

        if (
            resultado[
                "clasificacion"
            ]
            != caso[
                "esperado"
            ]
        ):
            raise ValueError(
                "Clasificación incorrecta."
            )

        if (
            resultado[
                "mostrar_ic"
            ]
            != caso[
                "mostrar_ic"
            ]
        ):
            raise ValueError(
                "Política de IC incorrecta."
            )

    print()
    print("=" * 72)
    print("VALIDACIONES FINALES")
    print("=" * 72)

    print()
    print(
        "PEÑA COLORADA: banda IC desactivada."
    )

    print(
        "COMALA: banda IC activada."
    )

    print(
        "La misma política se usa en métricas, "
        "advertencias y gráficas."
    )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()