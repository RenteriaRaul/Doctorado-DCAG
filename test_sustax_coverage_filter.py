
from pathlib import Path

from scripts.sustax.coverage_filter import (
    evaluar_cobertura_ssp,
    comparar_puntos_comunes,
)


ORIGEN = Path(
    "results/streamlit/sustax/BY_SCENARIO"
)

SALIDA = Path(
    "results/tests/sustax_validacion_general"
)


def main():

    diagnostico = evaluar_cobertura_ssp(ORIGEN)

    if diagnostico.empty:
        raise RuntimeError(
            "No se encontraron archivos SSP BY_SCENARIO"
        )

    puntos = comparar_puntos_comunes(diagnostico)

    SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    diagnostico.to_csv(
        SALIDA / "cobertura_diaria_ssp.csv",
        index=False
    )

    puntos.to_csv(
        SALIDA / "puntos_comunes_ssp.csv",
        index=False
    )

    print(
        "\nCOBERTURA DIARIA SSP "
        "(1981–2010 / 2026–2055)"
    )

    print(
        diagnostico.groupby("escenario")["elegible"]
        .agg(
            series="size",
            elegibles="sum"
        )
        .to_string()
    )

    print("\nSeries:", len(diagnostico))

    print(
        "Elegibles:",
        int(diagnostico["elegible"].sum())
    )

    print(
        "Excluidas:",
        int((~diagnostico["elegible"]).sum())
    )

    print(
        "Puntos presentes y elegibles en los 7 SSP:",
        int((puntos["n_escenarios"] == 7).sum())
    )

    print(
        "\nArchivos de diagnóstico:",
        SALIDA.resolve()
    )

    if (
        len(diagnostico) != 75
        or int(diagnostico["elegible"].sum()) != 63
    ):
        raise AssertionError(
            "Los conteos difieren de la validación previa: "
            "revisar diagnóstico"
        )

    print(
        "\nPRUEBA DE COBERTURA SATISFACTORIA"
    )


if __name__ == "__main__":
    main()
