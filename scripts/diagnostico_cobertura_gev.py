
from pathlib import Path
import calendar
import sys

import numpy as np
import pandas as pd

# Permite ejecutar el script desde la raíz del proyecto.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conagua_reader import leer_estacion_conagua


BASE = (
    ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
    / "COLIMA"
)

ESTACIONES = {
    "6001": "6001_ARMERIA.xlsx",
    "6021": "6021_QUESERIA.xlsx",
}

# Umbrales exploratorios, no criterios definitivos de exclusión.
UMBRALES = (80, 90, 95)


def diagnosticar_estacion(codigo, nombre_archivo):
    archivo = BASE / nombre_archivo

    if not archivo.exists():
        print(f"\nArchivo no encontrado: {archivo}")
        return

    resultado = leer_estacion_conagua(archivo)
    datos = resultado["data"].copy()
    metadata = resultado["metadata"]

    datos["date"] = pd.to_datetime(
        datos["date"], errors="coerce"
    )
    datos["pp"] = pd.to_numeric(
        datos["pp"], errors="coerce"
    )

    datos = datos.dropna(subset=["date"]).copy()
    datos["date"] = datos["date"].dt.normalize()

    # Aplicamos la misma regla de precipitación válida
    # utilizada por station_analysis.py.
    datos["pp_valida"] = datos["pp"].where(
        np.isfinite(datos["pp"]) & (datos["pp"] >= 0)
    )

    # Consolidamos fechas duplicadas como el motor GEV.
    diarios = (
        datos.groupby("date", as_index=False)["pp_valida"]
        .max()
    )
    diarios["year"] = diarios["date"].dt.year

    resumen = (
        diarios.groupby("year")
        .agg(
            dias_registrados=("date", "count"),
            dias_validos=("pp_valida", "count"),
            maximo_anual_mm=("pp_valida", "max"),
        )
        .reset_index()
    )

    resumen["dias_calendario"] = resumen["year"].apply(
        lambda y: 366 if calendar.isleap(int(y)) else 365
    )

    resumen["dias_faltantes"] = (
        resumen["dias_calendario"] - resumen["dias_validos"]
    )

    resumen["cobertura_pct"] = (
        100
        * resumen["dias_validos"]
        / resumen["dias_calendario"]
    ).round(2)

    # Reproducimos el número de años que usa actualmente
    # el modelo: cualquier año con un máximo válido.
    resumen["incluido_gev_actual"] = (
        resumen["maximo_anual_mm"].notna()
    )

    print("\n" + "=" * 76)
    print(f"ESTACIÓN {codigo}: {metadata.get('nombre', nombre_archivo)}")
    print("=" * 76)

    print(f"Archivo: {archivo.name}")
    print(f"Años con registros: {len(resumen)}")
    print(
        "Años utilizados potencialmente por GEV actual:",
        int(resumen["incluido_gev_actual"].sum()),
    )

    print("\nCOBERTURA ANUAL")
    print(
        resumen[
            [
                "year",
                "dias_validos",
                "dias_calendario",
                "dias_faltantes",
                "cobertura_pct",
                "maximo_anual_mm",
            ]
        ].to_string(index=False)
    )

    print("\nRESUMEN POR UMBRAL EXPLORATORIO")

    for umbral in UMBRALES:
        seleccion = resumen[
            (resumen["cobertura_pct"] >= umbral)
            & resumen["maximo_anual_mm"].notna()
        ]

        print(
            f"Cobertura >= {umbral}%: "
            f"{len(seleccion)} años"
        )

    print("\nAÑOS CON COBERTURA INFERIOR AL 80%")

    incompletos = resumen[
        resumen["cobertura_pct"] < 80
    ]

    if incompletos.empty:
        print("No se identificaron.")
    else:
        print(
            incompletos[
                [
                    "year",
                    "dias_validos",
                    "cobertura_pct",
                    "maximo_anual_mm",
                ]
            ].to_string(index=False)
        )

    print("\nDIEZ MÁXIMOS ANUALES MÁS ALTOS")

    print(
        resumen.nlargest(
            10, "maximo_anual_mm"
        )[
            [
                "year",
                "maximo_anual_mm",
                "cobertura_pct",
            ]
        ].to_string(index=False)
    )

    salida = (
        ROOT
        / "VALIDATED"
        / "diagnostico_gev"
    )
    salida.mkdir(parents=True, exist_ok=True)

    destino = salida / f"{codigo}_cobertura_anual.csv"
    resumen.to_csv(destino, index=False, encoding="utf-8-sig")

    print(f"\nDiagnóstico guardado: {destino}")


if __name__ == "__main__":
    for codigo, archivo in ESTACIONES.items():
        diagnosticar_estacion(codigo, archivo)
