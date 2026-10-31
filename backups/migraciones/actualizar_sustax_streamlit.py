
from pathlib import Path
from datetime import datetime
import ast

ARCHIVO = Path("app/pages/sustax.py")

def reemplazar_seccion(texto, inicio, fin, nuevo):
    if texto.count(inicio) != 1:
        raise ValueError(f"Inicio no encontrado o repetido: {inicio}")

    a = texto.index(inicio)
    b = texto.find(fin, a + len(inicio))

    if b == -1:
        raise ValueError(f"Final no encontrado: {fin}")

    return texto[:a] + nuevo.rstrip() + "\n\n" + texto[b:]

def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(ARCHIVO)
    original = ARCHIVO.read_text(encoding="utf-8-sig")

    inicio_tabla = "def preparar_tabla_puntos("
    fin_tabla = "st.title(\"Datos climáticos Sustax\")"

    nueva_tabla = '''def preparar_tabla_puntos(
    inventory_df: pd.DataFrame,
    group_summary_df: pd.DataFrame
) -> pd.DataFrame:
    if group_summary_df.empty:
        return pd.DataFrame()
    tabla = group_summary_df.copy()
    if "punto_id" not in tabla.columns:
        tabla["punto_id"] = tabla.apply(
            lambda r: f"{float(r['lat']):.4f}_{float(r['lon']):.4f}",
            axis=1
        )
    if "scenarios" in tabla.columns:
        tabla["escenarios"] = tabla["scenarios"]
    elif "escenarios" not in tabla.columns:
        tabla["escenarios"] = "No identificado"
    tabla = tabla.rename(columns={
        "punto_id": "Punto Sustax",
        "lat": "Latitud",
        "lon": "Longitud",
        "n_files": "Archivos",
        "files": "Archivos originales",
        "escenarios": "Escenarios"
    })
    columnas = [
        "Punto Sustax", "Latitud", "Longitud",
        "Archivos", "Escenarios", "Archivos originales"
    ]
    return tabla[[c for c in columnas if c in tabla.columns]]
'''

    modificado = reemplazar_seccion(
        original, inicio_tabla, fin_tabla, nueva_tabla
    )

    inicio_preparacion = "if escenarios_ssp and preparar_datos:"
    fin_preparacion = "if archivos_by_scenario:"
    nueva_preparacion = '''if escenarios_ssp and preparar_datos:
    import shutil
    import tempfile
    import uuid

    with st.spinner("Consolidando y validando series Sustax..."):
        try:
            SUSTAX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="sustax_preparacion_",
                dir=str(SUSTAX_RESULTS_DIR.resolve())
            ) as tmp:
                temporal = Path(tmp)
                long_tmp = temporal / "LONG"
                by_tmp = temporal / "BY_SCENARIO"

                merged_summary_df, _ = build_long_total_from_inventory(
                    inventory_df=inventory_df,
                    out_dir=str(long_tmp)
                )
                byscen_summary_df, _ = build_by_scenario_total(
                    long_total_dir=str(long_tmp),
                    out_dir=str(by_tmp)
                )

                n_puntos = len(merged_summary_df)
                n_series = int(
                    byscen_summary_df["n_scenarios_generated"].sum()
                )
                archivos_long = list(long_tmp.glob("*Total_LONG_mm.csv"))
                archivos_byscen = list(by_tmp.glob("*__mm.csv"))

                if n_puntos != len(group_summary_df):
                    raise ValueError(
                        f"Puntos esperados: {len(group_summary_df)}; "
                        f"generados: {n_puntos}"
                    )
                if len(archivos_long) != n_puntos:
                    raise ValueError("Cantidad incorrecta de archivos LONG")
                if len(archivos_byscen) != n_series:
                    raise ValueError("Cantidad incorrecta de series BY_SCENARIO")
                if n_series == 0:
                    raise ValueError("No se generaron escenarios climáticos")

                # Verificar la cantidad de escenarios por coordenada.
                conteos = byscen_summary_df.set_index(
                    "sustax_total"
                )["n_scenarios_generated"].to_dict()
                for _, punto in group_summary_df.iterrows():
                    punto_id = (
                        f"{float(punto['lat']):.4f}_"
                        f"{float(punto['lon']):.4f}Total"
                    )
                    esperado = int(punto["n_scenarios"])
                    obtenido = int(conteos.get(punto_id, 0))
                    if obtenido != esperado:
                        raise ValueError(
                            f"Escenarios inconsistentes en {punto_id}: "
                            f"{obtenido} de {esperado}"
                        )

                # Publicar las dos carpetas con posibilidad de restauración.
                identificador = uuid.uuid4().hex
                destinos = [
                    (long_tmp, SUSTAX_LONG_DIR),
                    (by_tmp, SUSTAX_BY_SCENARIO_DIR)
                ]
                respaldos = []
                publicados = []
                try:
                    for nueva, destino in destinos:
                        respaldo = SUSTAX_RESULTS_DIR / (
                            f".respaldo_{destino.name}_{identificador}"
                        )
                        if destino.exists():
                            destino.rename(respaldo)
                            respaldos.append((respaldo, destino))
                        nueva.rename(destino)
                        publicados.append(destino)
                except Exception:
                    for destino in reversed(publicados):
                        if destino.exists():
                            shutil.rmtree(destino)
                    for respaldo, destino in reversed(respaldos):
                        if respaldo.exists():
                            respaldo.rename(destino)
                    raise
                else:
                    for respaldo, _ in respaldos:
                        shutil.rmtree(respaldo)

            # Invalidar resultados calculados con otra colección.
            for clave in list(st.session_state.keys()):
                if clave.startswith("sustax_projection"):
                    del st.session_state[clave]

            st.success(
                f"Consolidación correcta: {n_puntos} puntos "
                f"y {n_series} series climáticas."
            )
            st.rerun()
        except Exception as exc:
            st.error(
                "La nueva preparación no pudo completarse. "
                "Revise el diagnóstico antes de continuar."
            )
            st.exception(exc)
'''

    modificado = reemplazar_seccion(
        modificado,
        inicio_preparacion,
        fin_preparacion,
        nueva_preparacion
    )

    ast.parse(modificado, filename=str(ARCHIVO))
    respaldo = ARCHIVO.with_name(
        f"sustax_respaldo_{datetime.now():%Y%m%d_%H%M%S}.py.bak"
    )
    respaldo.write_text(original, encoding="utf-8")
    ARCHIVO.write_text(modificado, encoding="utf-8")

    print("Actualización aplicada correctamente.")
    print(f"Archivo: {ARCHIVO.resolve()}")
    print(f"Respaldo: {respaldo.resolve()}")
    print("Sintaxis validada.")
    print("Los algoritmos de proyección no fueron modificados.")

if __name__ == "__main__":
    main()
