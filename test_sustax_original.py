from pathlib import Path

from scripts.sustax.inventory import (
    is_real_sustax_csv,
    build_sustax_inventory,
)
from scripts.sustax.loader import load_sustax_file_safe


ARCHIVO = Path(
    r"G:\My Drive\Doctorado\Probabilidad\Precipitación\Datos Sustax\Sustax Original\20260114T220314_ID-1803_daily_19.2506_-104.3438.csv"
)


def main():
    print("=" * 72)
    print("PRUEBA — LECTURA DE DESCARGA ORIGINAL SUSTAX")
    print("=" * 72)

    if not ARCHIVO.exists():
        raise FileNotFoundError(
            f"No se encontró: {ARCHIVO.resolve()}"
        )

    assert is_real_sustax_csv(str(ARCHIVO)), (
        "inventory.py no reconoció el archivo como exportación Sustax."
    )
    print("[1] Firma Sustax detectada: OK")

    df, _, meta = load_sustax_file_safe(
        str(ARCHIVO),
        return_pandas_df=True,
        return_metadata=True,
    )

    print("[2] Lectura del archivo: OK")
    print(f"    Coordenadas: {meta['lat']}, {meta['lon']}")
    print(f"    Formato temporal: {meta['data_time_format']}")
    print(f"    Periodo: {meta['date_start'].date()} — {meta['date_end'].date()}")
    print(f"    Escenarios: {', '.join(meta['scenarios'])}")
    print(f"    Registros: {len(df):,}")
    print(f"    Columnas: {list(df.columns)}")

    assert abs(meta["lat"] - 19.2506) < 1e-6
    assert abs(meta["lon"] - (-104.3438)) < 1e-6
    assert meta["data_time_format"] == "YYYY/MM/DD"
    assert set(meta["scenarios"]) == {"ERA5", "SSP245", "SSP370"}
    assert len(df) > 0
    assert df.index.min().strftime("%Y-%m-%d") == "1980-01-01"

    inv, bad = build_sustax_inventory(
        str(ARCHIVO.parent)
    )

    match = inv[
        inv["sustax_file"] == ARCHIVO.name
    ]

    assert len(match) == 1, (
        "El archivo no apareció exactamente una vez en el inventario."
    )

    print("[3] Inventario: OK")
    print("\nRegistro detectado:")
    print(
        match[
            [
                "sustax_file",
                "base_name",
                "lat",
                "lon",
                "scenarios_detected",
                "date_start",
                "date_end",
            ]
        ].to_string(index=False)
    )

    if not bad.empty:
        print("\nAdvertencia: otros CSV de la carpeta presentaron errores:")
        print(bad.to_string(index=False))

    print("\n" + "=" * 72)
    print("PRUEBA COMPLETADA CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()
