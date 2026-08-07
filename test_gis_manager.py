from scripts.gis_manager import GISManager


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MARCO = (
    r"C:\Users\rente\Desktop\Marco_geoestadistico_2025"
)


# ============================================================
# PRUEBA
# ============================================================

def main():

    print("=" * 72)
    print("PRUEBA GIS MANAGER")
    print("=" * 72)

    gis = GISManager(
        root=RUTA_MARCO
    )

    # --------------------------------------------------------
    # ESTADOS
    # --------------------------------------------------------

    estados = gis.listar_estados()

    print()
    print("ESTADOS DISPONIBLES")
    print("-" * 72)

    print(
        estados[
            estados["disponible"]
        ][
            [
                "clave",
                "estado",
                "archivo_zip",
            ]
        ].to_string(
            index=False
        )
    )

    disponibles = int(
        estados["disponible"].sum()
    )

    print()
    print(
        f"Total de estados detectados: {disponibles}"
    )

    # --------------------------------------------------------
    # PRUEBA CON COLIMA
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA CON COLIMA")
    print("=" * 72)

    zip_colima = gis.obtener_zip_estado(
        "Colima"
    )

    print()
    print("ZIP:")
    print(zip_colima)

    shapefiles = (
        gis.listar_shapefiles_estado(
            "Colima"
        )
    )

    print()
    print("SHAPEFILES ENCONTRADOS")
    print("-" * 72)

    print(
        shapefiles.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total de capas SHP: {len(shapefiles)}"
    )

    # --------------------------------------------------------
    # INSPECCIONAR 06ent.shp
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("INSPECCIÓN DE 06ent.shp")
    print("=" * 72)

    gdf_entidad = gis.cargar_shapefile(
        estado="Colima",
        nombre_shp="06ent.shp",
        crs_destino="EPSG:4326",
    )

    print()
    print(
        f"Número de registros: {len(gdf_entidad)}"
    )

    print()
    print("Columnas:")
    print(
        list(gdf_entidad.columns)
    )

    print()
    print("CRS:")
    print(
        gdf_entidad.crs
    )

    print()
    print("Tipos de geometría:")
    print(
        gdf_entidad.geom_type.value_counts()
    )

    print()
    print("Bounds:")
    print(
        gdf_entidad.total_bounds
    )

    print()
    print("Primeros registros:")

    print(
        gdf_entidad.drop(
            columns="geometry",
            errors="ignore",
        )
        .head()
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PROBAR LÍMITE ESTATAL AUTOMÁTICO
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_limite_estado()")
    print("=" * 72)

    limite_completo = gis.obtener_limite_estado(
        "Colima",
        territorio="completo",
    )

    limite_principal = gis.obtener_limite_estado(
        "Colima",
        territorio="principal",
    )

    print()
    print("Límite completo:")
    print(
        limite_completo.total_bounds
    )

    print()
    print("Territorio principal:")
    print(
        limite_principal.total_bounds
    )

    print()
    print(
        "Geometría completa:",
        limite_completo.geom_type.tolist(),
    )

    print(
        "Geometría principal:",
        limite_principal.geom_type.tolist(),
    )

    # --------------------------------------------------------
    # PROBAR MUNICIPIOS DEL ESTADO
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA DE MUNICIPIOS")
    print("=" * 72)

    gdf_municipios = gis.cargar_shapefile(
        estado="Colima",
        nombre_shp="06mun.shp",
        crs_destino="EPSG:4326",
    )

    print()
    print(
        f"Municipios encontrados: {len(gdf_municipios)}"
    )

    print()
    print("Columnas:")
    print(
        list(gdf_municipios.columns)
    )

    print()
    print("CRS:")
    print(
        gdf_municipios.crs
    )

    print()
    print("Tipos de geometría:")
    print(
        gdf_municipios.geom_type.value_counts()
    )

    print()
    print("Bounds:")
    print(
        gdf_municipios.total_bounds
    )

    print()
    print("Municipios:")
    print("-" * 72)

    columnas_mostrar = [
        columna
        for columna in [
            "CVEGEO",
            "CVE_ENT",
            "CVE_MUN",
            "NOMGEO",
        ]
        if columna in gdf_municipios.columns
    ]

    print(
        gdf_municipios[
            columnas_mostrar
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PROBAR obtener_municipio()
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_municipio()")
    print("=" * 72)

    manzanillo_completo = gis.obtener_municipio(
        estado="Colima",
        municipio="Manzanillo",
        territorio="completo",
    )

    manzanillo_principal = gis.obtener_municipio(
        estado="Colima",
        municipio="Manzanillo",
        territorio="principal",
    )

    print()
    print("Manzanillo completo:")
    print(
        manzanillo_completo.total_bounds
    )

    print()
    print("Manzanillo principal:")
    print(
        manzanillo_principal.total_bounds
    )

    print()
    print("Geometría completa:")
    print(
        manzanillo_completo.geom_type.tolist()
    )

    print()
    print("Geometría principal:")
    print(
        manzanillo_principal.geom_type.tolist()
    )
    # --------------------------------------------------------
    # PRUEBA NACIONAL CON OTRO ESTADO
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA NACIONAL — JALISCO")
    print("=" * 72)

    limite_jalisco = gis.obtener_limite_estado(
        estado="Jalisco",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    municipios_jalisco = gis.obtener_municipios(
        estado="Jalisco",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print("Límite principal de Jalisco:")
    print(
        limite_jalisco.total_bounds
    )

    print()
    print("Geometría del estado:")
    print(
        limite_jalisco.geom_type.tolist()
    )

    print()
    print(
        f"Municipios encontrados en Jalisco: "
        f"{len(municipios_jalisco)}"
    )

    print()
    print("Primeros 10 municipios:")
    print("-" * 72)

    columnas_mostrar = [
        columna
        for columna in [
            "CVEGEO",
            "CVE_ENT",
            "CVE_MUN",
            "NOMGEO",
        ]
        if columna in municipios_jalisco.columns
    ]

    print(
        municipios_jalisco[
            columnas_mostrar
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PRUEBA DE UN MUNICIPIO DE JALISCO
    # --------------------------------------------------------

    municipio_prueba = gis.obtener_municipio(
        estado="Jalisco",
        municipio="Guadalajara",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print("Municipio de prueba: Guadalajara")

    print()
    print("Bounds:")
    print(
        municipio_prueba.total_bounds
    )

    print()
    print("Geometría:")
    print(
        municipio_prueba.geom_type.tolist()
    )
    
    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA")
    print("=" * 72)


if __name__ == "__main__":
    main()