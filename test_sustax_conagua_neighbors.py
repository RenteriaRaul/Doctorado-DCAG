from pathlib import Path

import pandas as pd

from scripts.conagua_reader import leer_lote_conagua
from scripts.sustax.inventory import build_sustax_inventory
from scripts.sustax.merge import build_spatial_id
from scripts.sustax.obs import build_neighbors


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CARPETA_CONAGUA = (
    PROJECT_ROOT
    / "CONAGUA"
    / "estaciones_descargadas"
    / "COLIMA"
)

CARPETA_SUSTAX = Path(
    r"G:\My Drive\Doctorado\Probabilidad"
    r"\Precipitación\Datos Sustax\Sustax Original"
)

R_KM = 50
MAX_N = 8


# ============================================================
# CONAGUA
# ============================================================

def preparar_conagua():

    resultado = leer_lote_conagua(
        CARPETA_CONAGUA,
        patron="*.xlsx",
    )

    estaciones = (
        resultado[0]
        if isinstance(resultado, tuple)
        else resultado
    )

    rows = []

    for estacion in estaciones:

        meta = estacion.get("metadata", {})

        station_id = (
            meta.get("station")
            or meta.get("clave")
            or meta.get("station_id")
        )

        lat = (
            meta.get("latitude")
            or meta.get("latitud")
            or meta.get("lat")
        )

        lon = (
            meta.get("longitude")
            or meta.get("longitud")
            or meta.get("lon")
        )

        nombre = (
            meta.get("name")
            or meta.get("nombre")
            or ""
        )

        rows.append(
            {
                "station_id": str(station_id).strip(),
                "lat": float(lat),
                "lon": float(lon),
                "nombre": nombre,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUSTAX
# ============================================================

def preparar_sustax():

    inventory_df, bad_df = build_sustax_inventory(
        str(CARPETA_SUSTAX)
    )

    if not bad_df.empty:
        print(
            f"ADVERTENCIA: {len(bad_df)} archivo(s) "
            "Sustax presentaron problemas."
        )

    puntos = (
        inventory_df[
            ["base_name", "lat", "lon"]
        ]
        .drop_duplicates()
        .copy()
    )

    puntos["sustax_total"] = puntos.apply(
        lambda r: (
            build_spatial_id(
                r["base_name"],
                r["lat"],
                r["lon"],
            )
            + "Total"
        ),
        axis=1,
    )

    return puntos[
        [
            "sustax_total",
            "lat",
            "lon",
        ]
    ].reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PRUEBA — VECINOS CONAGUA PARA PUNTOS SUSTAX")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Preparar fuentes
    # --------------------------------------------------------

    conagua_df = preparar_conagua()
    sustax_df = preparar_sustax()

    print("\n" + "-" * 80)
    print("1. FUENTES")
    print("-" * 80)

    print(
        f"Estaciones CONAGUA: {len(conagua_df)}"
    )

    print(
        f"Puntos Sustax:      {len(sustax_df)}"
    )

    print("\nPuntos Sustax:")

    print(
        sustax_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 2. Construir vecinos
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. BÚSQUEDA DE VECINOS")
    print("-" * 80)

    print(f"Radio máximo: {R_KM} km")
    print(f"Máximo por punto: {MAX_N}")

    neighbors_df = build_neighbors(
        total_df=sustax_df,
        conagua_df=conagua_df,
        r_km=R_KM,
        max_n=MAX_N,
    )

    print(
        f"\nRelaciones Sustax-CONAGUA: "
        f"{len(neighbors_df)}"
    )

    # --------------------------------------------------------
    # 3. Resumen por punto
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("3. RESUMEN POR PUNTO SUSTAX")
    print("-" * 80)

    resumen_rows = []

    for _, punto in sustax_df.iterrows():

        nombre = punto["sustax_total"]

        sub = neighbors_df[
            neighbors_df["sustax_total"] == nombre
        ]

        if sub.empty:

            resumen_rows.append(
                {
                    "sustax_total": nombre,
                    "n_vecinos": 0,
                    "dist_min_km": None,
                    "dist_media_km": None,
                    "dist_max_km": None,
                }
            )

        else:

            resumen_rows.append(
                {
                    "sustax_total": nombre,
                    "n_vecinos": len(sub),
                    "dist_min_km": sub[
                        "distance_km"
                    ].min(),
                    "dist_media_km": sub[
                        "distance_km"
                    ].mean(),
                    "dist_max_km": sub[
                        "distance_km"
                    ].max(),
                }
            )

    resumen = pd.DataFrame(resumen_rows)

    print(
        resumen.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # 4. Detalle
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. ESTACIONES SELECCIONADAS")
    print("-" * 80)

    detalle = neighbors_df.merge(
        conagua_df[
            [
                "station_id",
                "nombre",
            ]
        ],
        on="station_id",
        how="left",
    )

    for punto in sustax_df["sustax_total"]:

        print(f"\n{punto}")

        sub = detalle[
            detalle["sustax_total"] == punto
        ].copy()

        if sub.empty:

            print("  SIN ESTACIONES DENTRO DEL RADIO")
            continue

        sub = sub[
            [
                "station_id",
                "nombre",
                "distance_km",
            ]
        ]

        print(
            sub.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    # --------------------------------------------------------
    # 5. Validación
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. VALIDACIÓN")
    print("-" * 80)

    sin_vecinos = resumen[
        resumen["n_vecinos"] == 0
    ]

    pocos_vecinos = resumen[
        resumen["n_vecinos"] < 3
    ]

    fuera_radio = neighbors_df[
        neighbors_df["distance_km"] > R_KM
    ]

    if sin_vecinos.empty:
        print(
            "OK: todos los puntos Sustax tienen "
            "al menos una estación CONAGUA."
        )
    else:
        print(
            f"ADVERTENCIA: {len(sin_vecinos)} punto(s) "
            "no tienen estaciones dentro del radio."
        )

    if pocos_vecinos.empty:
        print(
            "OK: todos los puntos tienen al menos "
            "3 estaciones disponibles espacialmente."
        )
    else:
        print(
            f"OBSERVACIÓN: {len(pocos_vecinos)} punto(s) "
            "tienen menos de 3 vecinos."
        )

    if fuera_radio.empty:
        print(
            f"OK: todas las estaciones seleccionadas "
            f"están dentro de {R_KM} km."
        )
    else:
        print(
            "ERROR: existen estaciones fuera "
            "del radio definido."
        )

    # --------------------------------------------------------
    # 6. Distribución
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. DISTRIBUCIÓN DE VECINOS")
    print("-" * 80)

    print(
        resumen["n_vecinos"]
        .value_counts()
        .sort_index()
        .rename_axis("n_vecinos")
        .reset_index(name="puntos")
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("PRUEBA DE VECINOS FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()