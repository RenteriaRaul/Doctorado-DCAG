"""Distancia euclidiana a corrientes naturales desde centros de celda del MDE.

Usa la capa regional ``corrientes_regionales`` de hydrography.py. El cálculo es
descriptivo y continuo; no asigna umbrales ni categorías de susceptibilidad.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.strtree import STRtree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DEM = (
    REPO_ROOT / "results" / "territorial" / "dem" / "villa_de_alvarez_v2"
    / "elevacion_villa_de_alvarez_utm13n.tif"
)
DEFAULT_HYDROGRAPHY = (
    REPO_ROOT / "results" / "territorial" / "hydrography" / "villa_de_alvarez"
    / "red_hidrografica_villa_de_alvarez.gpkg"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "territorial" / "distance_to_streams" / "villa_de_alvarez"
HYDRO_LAYER = "corrientes_regionales"
MUNICIPAL_LAYER = "limite_municipal"
HYDRO_CATEGORY_FIELD = "_dcag_hydro_category"
ENABLED_FIELD = "_dcag_enabled_status"
INCLUDED_CATEGORIES = ("corriente_perenne", "corriente_intermitente")
NODATA_OUTPUT = np.float32(-9999.0)
DEFAULT_BLOCK_ROWS = 64


def filtrar_corrientes_naturales(regional_lines):
    """Selecciona categorías naturales perenne/intermitente, sin filtrar ENABLED.

    Retorna una copia de la GeoDataFrame y un resumen de auditoría. Los demás
    valores de ``_dcag_hydro_category`` se excluyen del cálculo de distancia.
    """
    if HYDRO_CATEGORY_FIELD not in regional_lines.columns:
        raise ValueError(
            f"La capa {HYDRO_LAYER!r} requiere el campo {HYDRO_CATEGORY_FIELD!r}; "
            f"campos encontrados: {list(regional_lines.columns)}"
        )
    values = regional_lines[HYDRO_CATEGORY_FIELD].fillna("<NULL>").astype(str)
    selected = regional_lines.loc[values.isin(INCLUDED_CATEGORIES)].copy()
    counts = {str(k): int(v) for k, v in values.value_counts(dropna=False).items()}
    enabled_counts = {}
    if ENABLED_FIELD in regional_lines.columns:
        enabled_counts = {
            str(k): int(v)
            for k, v in selected[ENABLED_FIELD].fillna("<NULL>").astype(str).value_counts(dropna=False).items()
        }
    report = {
        "campo_categoria": HYDRO_CATEGORY_FIELD,
        "categorias_incluidas": list(INCLUDED_CATEGORIES),
        "conteos_categoria_capa_regional": counts,
        "segmentos_regionales_totales": int(len(regional_lines)),
        "segmentos_incluidos": int(len(selected)),
        "segmentos_excluidos": int(len(regional_lines) - len(selected)),
        "conteos_ENABLED_segmentos_incluidos": enabled_counts,
        "filtro_ENABLED_aplicado": False,
    }
    return selected, report


def construir_arbol_corrientes(line_geometries):
    """Valida y construye STRtree sobre geometrías lineales completas."""
    geometries = []
    for index, geometry in enumerate(line_geometries):
        if geometry is None or geometry.is_empty:
            continue
        if not isinstance(geometry, (LineString, MultiLineString)):
            raise TypeError(f"La geometría {index} no es lineal: {geometry.geom_type}")
        if not geometry.is_valid:
            raise ValueError(f"La geometría lineal {index} no es válida.")
        geometries.append(geometry)
    if not geometries:
        raise ValueError("No hay corrientes lineales válidas para construir el STRtree.")
    try:
        return STRtree(geometries), len(geometries)
    except AttributeError as exc:
        raise RuntimeError("Se requiere Shapely 2 con STRtree.query_nearest().") from exc


def distancias_exactas_a_corrientes(points, tree):
    """Distancia planar exacta punto-a-geometría lineal mediante STRtree.

    ``points`` contiene centros de celda (o cualquier punto de prueba). La
    distancia se calcula a la geometría completa, no a centroides ni vértices.
    """
    points = np.asarray(points, dtype=object).reshape(-1)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if any(point is None or point.is_empty or not isinstance(point, Point) for point in points):
        raise TypeError("La consulta STRtree debe contener puntos no vacíos.")
    if not hasattr(tree, "query_nearest"):
        raise RuntimeError("Se requiere Shapely 2 con STRtree.query_nearest().")
    pairs, distances = tree.query_nearest(
        points,
        all_matches=False,
        return_distance=True,
    )
    results = np.full(len(points), np.nan, dtype=np.float64)
    if len(distances):
        results[np.asarray(pairs[0], dtype=np.int64)] = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(results).all():
        raise RuntimeError("El STRtree no devolvió una distancia finita para cada punto consultado.")
    return results


def calcular_bloque_distancias(transform, row_offset, valid_mask, tree, nodata=NODATA_OUTPUT):
    """Calcula un bloque de filas sobre sus celdas válidas, devolviendo float32."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask debe ser una matriz bidimensional.")
    result = np.full(valid_mask.shape, nodata, dtype=np.float32)
    rows, cols = np.nonzero(valid_mask)
    if len(rows) == 0:
        return result
    global_rows = rows.astype(np.float64) + float(row_offset) + 0.5
    pixel_cols = cols.astype(np.float64) + 0.5
    xs = transform.a * pixel_cols + transform.b * global_rows + transform.c
    ys = transform.d * pixel_cols + transform.e * global_rows + transform.f
    # Vectorización Shapely 2, limitada a las celdas válidas de este bloque.
    from shapely import points as shapely_points

    query_points = shapely_points(xs, ys)
    distances = distancias_exactas_a_corrientes(query_points, tree)
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Se obtuvieron distancias negativas o no finitas.")
    result[rows, cols] = distances.astype(np.float32)
    return result


def calcular_malla_por_bloques(transform, valid_mask, tree, block_rows=DEFAULT_BLOCK_ROWS, nodata=NODATA_OUTPUT):
    """Función pura útil para pruebas: procesa una máscara 2D fila por fila en bloques."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError("block_rows debe ser un entero positivo.")
    result = np.full(valid_mask.shape, nodata, dtype=np.float32)
    for start in range(0, valid_mask.shape[0], block_rows):
        stop = min(start + block_rows, valid_mask.shape[0])
        result[start:stop] = calcular_bloque_distancias(
            transform, start, valid_mask[start:stop], tree, nodata=nodata
        )
    return result


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        raise ValueError("No hay celdas válidas para calcular estadísticas.")
    return {
        "minimo_m": float(np.min(values)),
        "maximo_m": float(np.max(values)),
        "media_m": float(np.mean(values)),
        "mediana_m": float(np.median(values)),
        "desviacion_estandar_m": float(np.std(values, ddof=0)),
        "percentiles_m": {
            str(p): float(np.percentile(values, p)) for p in (5, 25, 75, 95, 99)
        },
    }


def _draw_scale_bar(ax, bounds):
    minx, miny, maxx, maxy = bounds
    width, height = maxx - minx, maxy - miny
    raw_m = width * 0.2
    exponent = math.floor(math.log10(max(raw_m, 1.0)))
    unit = 10.0**exponent
    length_m = max(unit, round(raw_m / unit) * unit)
    x0 = minx + width * 0.05
    y0 = miny + height * 0.06
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=3, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + height * 0.012, f"{length_m / 1000:g} km", ha="center", va="bottom", fontsize=8)


def _create_map(distance_raster, reference_dem_path, gpkg_path, layer, output_png):
    import geopandas as gpd

    with rasterio.open(distance_raster) as distance:
        data = distance.read(1, masked=True)
        extent = distance.bounds
        crs = distance.crs
    boundary = gpd.read_file(gpkg_path, layer=MUNICIPAL_LAYER)
    if boundary.crs is None:
        raise ValueError(f"La capa {MUNICIPAL_LAYER!r} no declara CRS.")
    boundary = boundary.to_crs(crs)
    regional = gpd.read_file(gpkg_path, layer=layer)
    if regional.crs is None:
        raise ValueError(f"La capa {layer!r} no declara CRS.")
    regional = regional.to_crs(crs)
    regional, _ = filtrar_corrientes_naturales(regional)

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = ax.imshow(
        data,
        extent=(extent.left, extent.right, extent.bottom, extent.top),
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label("Distancia al cauce más cercano (m)")

    # Mostrar la red regional que cae dentro de la extensión cartográfica; las
    # geometrías completas continúan siendo consultadas para el cálculo raster.
    viewbox = box(extent.left, extent.bottom, extent.right, extent.top)
    try:
        candidate_indices = regional.sindex.query(viewbox, predicate="intersects")
        visible_lines = regional.iloc[np.asarray(candidate_indices, dtype=int)]
    except (AttributeError, TypeError):
        visible_lines = regional.loc[regional.geometry.intersects(viewbox)]
    if not visible_lines.empty:
        visible_lines.plot(ax=ax, color="#174a7e", linewidth=0.45, alpha=0.8)
    boundary.boundary.plot(ax=ax, color="#c62828", linewidth=1.1)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#174a7e", linewidth=1.2, label="Corrientes perennes e intermitentes (referencia regional)"),
            Line2D([0], [0], color="#c62828", linewidth=1.2, label="Límite municipal"),
        ],
        loc="upper right",
        frameon=True,
    )
    ax.set_title("Distancia euclidiana al cauce más cercano — Villa de Álvarez")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal")
    _draw_scale_bar(ax, (extent.left, extent.bottom, extent.right, extent.top))
    ax.text(
        0.01,
        0.01,
        f"CRS: {crs.to_string()} | Fuente: INEGI, Red Hidrográfica 1:50 000; Marco Geoestadístico INEGI",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generar_distancia_a_cauces(
    reference_dem_path=DEFAULT_REFERENCE_DEM,
    hydrography_gpkg=DEFAULT_HYDROGRAPHY,
    output_dir=DEFAULT_OUTPUT_DIR,
    block_rows=DEFAULT_BLOCK_ROWS,
):
    """Genera distancia a cauces, PNG y JSON; no sobrescribe productos."""
    reference_dem_path = Path(reference_dem_path).expanduser().resolve()
    hydrography_gpkg = Path(hydrography_gpkg).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    outputs = {
        "raster": output_dir / "distancia_cauces_villa_de_alvarez.tif",
        "png": output_dir / "distancia_cauces_villa_de_alvarez.png",
        "json": output_dir / "metadatos_distancia_cauces.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("No se sobrescriben resultados existentes: " + ", ".join(existing))
    if not reference_dem_path.is_file():
        raise FileNotFoundError(f"No se encontró el MDE de referencia: {reference_dem_path}")
    if not hydrography_gpkg.is_file():
        raise FileNotFoundError(f"No se encontró el GeoPackage hidrográfico: {hydrography_gpkg}")
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError("block_rows debe ser un entero positivo.")

    # Leer las corrientes regionales completas: no se recortan al límite municipal.
    regional = gpd.read_file(hydrography_gpkg, layer=HYDRO_LAYER)
    if regional.crs is None:
        raise ValueError(f"La capa {HYDRO_LAYER!r} no declara CRS.")
    with rasterio.open(reference_dem_path) as ref:
        if ref.crs is None or ref.crs.to_epsg() != 6368:
            raise ValueError("El MDE de referencia debe estar en EPSG:6368.")
        if ref.count < 1:
            raise ValueError("El MDE de referencia no tiene bandas.")
        reference_profile = ref.profile.copy()
        reference_crs = ref.crs
        reference_transform = ref.transform
        reference_bounds = ref.bounds
        reference_width, reference_height = ref.width, ref.height
        reference_resolution = tuple(float(v) for v in ref.res)
        reference_nodata = ref.nodata
        reference_valid_count = 0
        if not (math.isclose(ref.transform.b, 0.0) and math.isclose(ref.transform.d, 0.0)):
            raise ValueError("Esta versión requiere una cuadrícula de referencia sin rotación.")

        regional_crs = regional.crs.to_string()
        regional = regional.to_crs(reference_crs)
        selected, filter_report = filtrar_corrientes_naturales(regional)
        geometries = [g for g in selected.geometry if g is not None and not g.is_empty]
        tree, tree_geometry_count = construir_arbol_corrientes(geometries)

        output_dir.mkdir(parents=True, exist_ok=True)
        profile = reference_profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=float(NODATA_OUTPUT),
            compress="deflate",
        )
        chunks = []
        with rasterio.open(outputs["raster"], "w", **profile) as dst:
            for row_start in range(0, reference_height, block_rows):
                rows = min(block_rows, reference_height - row_start)
                window = Window(0, row_start, reference_width, rows)
                valid_mask = ref.read_masks(1, window=window) > 0
                reference_valid_count += int(valid_mask.sum())
                block = calcular_bloque_distancias(
                    reference_transform,
                    row_start,
                    valid_mask,
                    tree,
                    nodata=NODATA_OUTPUT,
                )
                if valid_mask.any():
                    valid_values = block[valid_mask]
                    if not np.isfinite(valid_values).all() or (valid_values < 0).any():
                        raise ValueError("El bloque contiene distancias negativas o no finitas.")
                    chunks.append(valid_values.copy())
                dst.write(block, 1, window=window)
        values = np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0, dtype=np.float64)
        stats = _stats(values)

    # Verificación de grid, NoData y máscara por ventanas; no carga los rasters completos.
    mask_equal = True
    valid_output_count = 0
    invalid_output_count = 0
    nonfinite_valid_count = 0
    negative_valid_count = 0
    with rasterio.open(reference_dem_path) as ref, rasterio.open(outputs["raster"]) as out:
        grid_checks = {
            "crs": out.crs == ref.crs,
            "resolucion": out.res == ref.res,
            "dimensiones": (out.width, out.height) == (ref.width, ref.height),
            "transformacion_afin": out.transform == ref.transform,
            "extension": out.bounds == ref.bounds,
            "dtype_float32": out.dtypes[0] == "float32",
            "nodata_menos_9999": out.nodata == float(NODATA_OUTPUT),
        }
        for row_start in range(0, ref.height, block_rows):
            rows = min(block_rows, ref.height - row_start)
            window = Window(0, row_start, ref.width, rows)
            input_valid = ref.read_masks(1, window=window) > 0
            output_valid = out.read_masks(1, window=window) > 0
            if not np.array_equal(input_valid, output_valid):
                mask_equal = False
            data = out.read(1, window=window)
            valid_data = data[output_valid]
            valid_output_count += int(output_valid.sum())
            invalid_output_count += int(output_valid.size - output_valid.sum())
            nonfinite_valid_count += int((~np.isfinite(valid_data)).sum())
            negative_valid_count += int((valid_data < 0).sum())
            if np.any(data[~input_valid] != NODATA_OUTPUT):
                mask_equal = False
        grid_checks["mascara_celdas_validas"] = mask_equal
        if not all(grid_checks.values()):
            raise RuntimeError(f"El GeoTIFF no coincide con la referencia: {grid_checks}")
        if valid_output_count != reference_valid_count:
            raise RuntimeError("Cambió el número de celdas válidas frente al MDE de referencia.")
        if nonfinite_valid_count or negative_valid_count:
            raise RuntimeError("El raster contiene valores inválidos dentro de la máscara territorial.")

    _create_map(outputs["raster"], reference_dem_path, hydrography_gpkg, HYDRO_LAYER, outputs["png"])
    metadata = {
        "procesado_utc": datetime.now(timezone.utc).isoformat(),
        "territorio": {"estado": "Colima", "municipio": "Villa de Álvarez"},
        "fuentes": {
            "mde_referencia": str(reference_dem_path),
            "geopackage_hidrografico": str(hydrography_gpkg),
            "capa_corrientes": HYDRO_LAYER,
            "capa_limite_municipal": MUNICIPAL_LAYER,
            "crs_capa_corrientes_entrada": regional_crs,
            "fuente_cartografica": "INEGI, Red Hidrográfica escala 1:50 000 y Marco Geoestadístico",
        },
        "metodologia": {
            "medida": "Distancia euclidiana planar exacta desde el centro de cada celda válida a la geometría lineal más cercana.",
            "geometria": "Geometrías LineString/MultiLineString completas; consulta STRtree.query_nearest, sin centroides ni sustitución por vértices.",
            "crs_calculo": "EPSG:6368; unidades métricas, resultado en metros.",
            "procesamiento": "Bloques de filas; puntos creados solo para celdas válidas en cada bloque.",
            "bloque_filas": block_rows,
            "efecto_borde": "Se consulta la red regional completa sin recortarla al municipio; cauces fuera del límite pueden ser los más cercanos.",
            "normalizacion_o_umbrales": "No aplicados.",
        },
        "filtros": {
            **filter_report,
            "criterio": f"{HYDRO_CATEGORY_FIELD} en {list(INCLUDED_CATEGORIES)}",
            "enabled": "Sin filtro; ENABLED=0 y ENABLED=1 se incluyen por igual si la categoría es natural perenne/intermitente.",
            "valores_categoria_detectados": filter_report["conteos_categoria_capa_regional"],
        },
        "malla_referencia": {
            "crs": reference_crs.to_string(),
            "resolucion_m": list(reference_resolution),
            "dimensiones_columnas_filas": [reference_width, reference_height],
            "transformacion_afin": list(reference_transform)[:6],
            "extension": [float(v) for v in reference_bounds],
            "nodata_referencia": reference_nodata,
            "nodata_salida": float(NODATA_OUTPUT),
            "celdas_validas": valid_output_count,
            "celdas_invalidas": invalid_output_count,
            "geometrias_lineales_validas_en_arbol": tree_geometry_count,
            "verificacion_coincidencia": grid_checks,
        },
        "estadisticas_distancia": stats,
        "validacion_valores": {
            "distancias_negativas_en_celdas_validas": negative_valid_count,
            "valores_no_finitos_en_celdas_validas": nonfinite_valid_count,
            "mascara_coincide_con_mde": mask_equal,
        },
        "salidas": {key: str(value) for key, value in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_REFERENCE_DEM)
    parser.add_argument("--hydrography", type=Path, default=DEFAULT_HYDROGRAPHY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    args = parser.parse_args()
    for name, path in generar_distancia_a_cauces(
        reference_dem_path=args.dem,
        hydrography_gpkg=args.hydrography,
        output_dir=args.output_dir,
        block_rows=args.block_rows,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
