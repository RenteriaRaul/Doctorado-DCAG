from pathlib import Path
import json
import shutil
import tempfile

import pandas as pd

from scripts.exceedance_persistence import (
    guardar_ejecucion_excedencia,
    listar_ejecuciones_excedencia,
    cargar_ejecucion_excedencia,
)


def main():
    print("=" * 70)
    print("TEST DE PERSISTENCIA - EXCEDENCIAS")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        carpeta_salida = base / "results" / "streamlit" / "excedencias"
        carpeta_fuente = base / "CONAGUA" / "estaciones_descargadas" / "COLIMA"
        carpeta_fuente.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # 1. DATOS DE PRUEBA
        # ------------------------------------------------------------
        resultados = pd.DataFrame(
            {
                "station": ["6001", "6002", "6003"],
                "nombre": ["ARMERIA", "COLIMA", "COMALA"],
                "estado": ["COLIMA", "COLIMA", "COLIMA"],
                "municipio": ["ARMERIA", "COLIMA", "COMALA"],
                "latitud": [18.94, 19.24, 19.32],
                "longitud": [-103.96, -103.72, -103.76],
                "threshold_mm": [50.0, 50.0, 50.0],
                "total_dias": [25000, 24000, 23000],
                "dias_excedencia": [120, 100, 90],
                "prob_excedencia": [0.0048, 0.0041667, 0.0039130],
                "EXCEDENCIA_50MM": [0.0048, 0.0041667, 0.0039130],
            }
        )

        log_df = pd.DataFrame(
            {
                "station": ["6001", "6002", "6003"],
                "status": ["ok", "ok", "ok"],
                "message": ["", "", ""],
            }
        )

        metadata_df = pd.DataFrame(
            {
                "station": ["6001", "6002", "6003"],
                "nombre": ["ARMERIA", "COLIMA", "COMALA"],
                "estado": ["COLIMA", "COLIMA", "COLIMA"],
                "latitud": [18.94, 19.24, 19.32],
                "longitud": [-103.96, -103.72, -103.76],
            }
        )

        # GeoTIFF simulado: aquí solo comprobamos persistencia/copia,
        # no el motor científico ni la validez interna del raster.
        raster_origen = base / "excedencia_50mm_interpolada.tif"
        raster_origen.write_bytes(b"GEOTIFF_TEST_PERSISTENCIA")

        # ------------------------------------------------------------
        # 2. GUARDAR EJECUCIÓN
        # ------------------------------------------------------------
        config = guardar_ejecucion_excedencia(
            resultados=resultados,
            log_df=log_df,
            metadata_df=metadata_df,
            carpeta_salida=carpeta_salida,
            carpeta_fuente=carpeta_fuente,
            patron="*.xlsx",
            threshold=50.0,
            consolidar_duplicados=True,
            raster_path=raster_origen,
            estado="COLIMA",
            metodo_interpolacion="linear",
            nx=300,
            ny=300,
            margin=0.0,
            fill_nearest=False,
            crs="EPSG:4326",
        )

        print("\n[1] Guardado de ejecución: OK")
        print("Run ID:", config["run_id"])

        # ------------------------------------------------------------
        # 3. VERIFICAR ESTRUCTURA
        # ------------------------------------------------------------
        run_dir = Path(config["run_dir"])
        config_path = run_dir / "run_metadata.json"
        latest_path = carpeta_salida / "latest_run.json"
        master_path = Path(config["master_path"])
        log_path = Path(config["log_path"])
        metadata_path = Path(config["metadata_path"])
        raster_path = Path(config["raster_path"])

        requeridos = {
            "run_metadata.json": config_path,
            "latest_run.json": latest_path,
            "MASTER": master_path,
            "LOG": log_path,
            "METADATA": metadata_path,
            "GEOTIFF": raster_path,
        }

        for nombre, path in requeridos.items():
            assert path.exists(), f"No se creó {nombre}: {path}"

        print("[2] Archivos persistentes: OK")

        # ------------------------------------------------------------
        # 4. VERIFICAR JSON
        # ------------------------------------------------------------
        metadata_json = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        assert metadata_json["estado"] == "COLIMA"
        assert metadata_json["threshold_mm"] == 50.0
        assert metadata_json["estaciones_procesadas"] == 3
        assert metadata_json["consolidar_duplicados"] is True
        assert metadata_json["metodo_interpolacion"] == "linear"
        assert metadata_json["nx"] == 300
        assert metadata_json["ny"] == 300
        assert metadata_json["crs"] == "EPSG:4326"

        latest_json = json.loads(
            latest_path.read_text(encoding="utf-8")
        )

        assert latest_json["run_id"] == config["run_id"]

        print("[3] Metadatos JSON: OK")

        # ------------------------------------------------------------
        # 5. VERIFICAR COPIA DEL GEOTIFF
        # ------------------------------------------------------------
        assert raster_path.read_bytes() == raster_origen.read_bytes()

        print("[4] Copia del GeoTIFF: OK")

        # ------------------------------------------------------------
        # 6. LISTAR EJECUCIONES
        # ------------------------------------------------------------
        ejecuciones = listar_ejecuciones_excedencia(
            carpeta_salida
        )

        assert len(ejecuciones) == 1
        assert ejecuciones[0]["run_id"] == config["run_id"]
        assert ejecuciones[0]["estado"] == "COLIMA"

        print("[5] Listado de ejecuciones: OK")

        # ------------------------------------------------------------
        # 7. CARGAR EJECUCIÓN
        # ------------------------------------------------------------
        cargada = cargar_ejecucion_excedencia(
            ejecuciones[0]
        )

        assert cargada["persistent"] is True
        assert cargada["estado"] == "COLIMA"
        assert cargada["threshold"] == 50.0
        assert cargada["n_archivos"] == 3
        assert Path(cargada["raster_path"]).exists()

        print("[6] Carga de ejecución: OK")

        # ------------------------------------------------------------
        # 8. COMPARAR TABLAS
        # ------------------------------------------------------------
        recuperados = cargada["resultados"].copy()

        # Normalizar identificadores de estación.
        # Al leer un CSV, pandas puede interpretar "6001" como entero 6001.
        resultados_comparar = resultados.copy()
        recuperados_comparar = recuperados.copy()

        resultados_comparar["station"] = (
            resultados_comparar["station"]
            .astype(str)
            .str.strip()
        )

        recuperados_comparar["station"] = (
            recuperados_comparar["station"]
            .astype(str)
            .str.strip()
        )

        pd.testing.assert_frame_equal(
            resultados_comparar.reset_index(drop=True),
            recuperados_comparar.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )

        pd.testing.assert_frame_equal(
            log_df.reset_index(drop=True).fillna(""),
            cargada["log_df"].reset_index(drop=True).fillna(""),
            check_dtype=False,
        )

        pd.testing.assert_frame_equal(
            metadata_df.reset_index(drop=True),
            cargada["metadata_df"].reset_index(drop=True),
            check_dtype=False,
        )

        print("[7] Recuperación de tablas: OK")

        # ------------------------------------------------------------
        # 9. RESUMEN
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print("RESULTADO FINAL: TODOS LOS TESTS PASARON")
        print("=" * 70)
        print("Persistencia validada:")
        print("  - guardar")
        print("  - JSON")
        print("  - CSV maestro")
        print("  - log")
        print("  - metadata")
        print("  - GeoTIFF")
        print("  - listar")
        print("  - cargar")
        print("  - recuperación íntegra de resultados")


if __name__ == "__main__":
    main()