from pathlib import Path

from scripts.conagua_reader import (
    leer_estacion_conagua,
)


ARCHIVO = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\estaciones_conagua_excel\Colima\6001_ARMERIA.xlsx"
)


def main():

    print("=" * 72)
    print("PRUEBA LECTOR CONAGUA")
    print("=" * 72)

    estacion = leer_estacion_conagua(
        ARCHIVO
    )

    metadata = estacion[
        "metadata"
    ]

    data = estacion[
        "data"
    ]

    calidad = estacion[
        "calidad"
    ]

    print()
    print("METADATOS")
    print("-" * 72)

    for clave, valor in metadata.items():
        print(
            f"{clave}: {valor}"
        )

    print()
    print("CALIDAD")
    print("-" * 72)

    for clave, valor in calidad.items():
        print(
            f"{clave}: {valor}"
        )

    print()
    print("COLUMNAS CLIMÁTICAS")
    print("-" * 72)

    print(
        list(data.columns)
    )

    print()
    print("PRIMEROS REGISTROS")
    print("-" * 72)

    print(
        data.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print("ÚLTIMOS REGISTROS")
    print("-" * 72)

    print(
        data.tail(
            10
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------------

    if metadata["station"] != "6001":
        raise ValueError(
            "La clave de estación esperada es 6001."
        )

    if metadata["nombre"] != "ARMERIA":
        raise ValueError(
            "El nombre esperado es ARMERIA."
        )

    if metadata["estado"] != "COLIMA":
        raise ValueError(
            "El estado esperado es COLIMA."
        )

    if "date" not in data.columns:
        raise ValueError(
            "No se generó la columna date."
        )

    if "pp" not in data.columns:
        raise ValueError(
            "No se generó la columna pp."
        )

    if data.empty:
        raise ValueError(
            "La serie climática quedó vacía."
        )

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()