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
    # INSPECCIONAR LOCALIDADES DE COLIMA
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA DE LOCALIDADES — COLIMA")
    print("=" * 72)

    gdf_localidades = gis.cargar_shapefile(
        estado="Colima",
        nombre_shp="06l.shp",
        crs_destino="EPSG:4326",
    )

    print()
    print(
        f"Localidades encontradas en el estado: "
        f"{len(gdf_localidades)}"
    )

    print()
    print("Columnas:")
    print(
        list(gdf_localidades.columns)
    )

    print()
    print("CRS:")
    print(
        gdf_localidades.crs
    )

    print()
    print("Tipos de geometría:")
    print(
        gdf_localidades.geom_type.value_counts()
    )

    print()
    print("Bounds:")
    print(
        gdf_localidades.total_bounds
    )

    # --------------------------------------------------------
    # FILTRAR MUNICIPIO DE MANZANILLO
    # --------------------------------------------------------

    localidades_manzanillo = gdf_localidades[
        gdf_localidades["CVE_MUN"]
        .astype(str)
        .str.zfill(3)
        == "007"
    ].copy()

    print()
    print(
        "Localidades amanzanadas del municipio "
        "de Manzanillo:"
    )
    print("-" * 72)

    columnas_mostrar = [
        columna
        for columna in [
            "CVEGEO",
            "CVE_ENT",
            "CVE_MUN",
            "CVE_LOC",
            "NOMGEO",
            "AMBITO",
            "ÁMBITO",
        ]
        if columna in localidades_manzanillo.columns
    ]

    print(
        localidades_manzanillo[
            columnas_mostrar
        ]
        .sort_values(
            by="NOMGEO"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Total de localidades amanzanadas "
        f"en Manzanillo: {len(localidades_manzanillo)}"
    )
    
        # --------------------------------------------------------
    # PROBAR obtener_localidades()
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_localidades()")
    print("=" * 72)

    localidades_manzanillo = gis.obtener_localidades(
        estado="Colima",
        municipio="Manzanillo",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print(
        f"Localidades obtenidas: "
        f"{len(localidades_manzanillo)}"
    )

    print()
    print(
        localidades_manzanillo[
            [
                "CVEGEO",
                "CVE_MUN",
                "CVE_LOC",
                "NOMGEO",
                "AMBITO",
            ]
        ]
        .sort_values("NOMGEO")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # PROBAR obtener_localidad()
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_localidad()")
    print("=" * 72)

    localidad_manzanillo = gis.obtener_localidad(
        estado="Colima",
        municipio="Manzanillo",
        localidad="Manzanillo",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print("Localidad obtenida:")
    print(
        localidad_manzanillo[
            [
                "CVEGEO",
                "CVE_MUN",
                "CVE_LOC",
                "NOMGEO",
                "AMBITO",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Bounds:")
    print(
        localidad_manzanillo.total_bounds
    )

    print()
    print("Geometría:")
    print(
        localidad_manzanillo.geom_type.tolist()
    )
        # --------------------------------------------------------
    # INSPECCIONAR AGEB URBANAS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA AGEB URBANAS — COLIMA")
    print("=" * 72)

    gdf_ageb_urbanas = gis.cargar_shapefile(
        estado="Colima",
        nombre_shp="06a.shp",
        crs_destino="EPSG:4326",
    )

    print()
    print(
        f"AGEB urbanas encontradas en el estado: "
        f"{len(gdf_ageb_urbanas)}"
    )

    print()
    print("Columnas:")
    print(
        list(gdf_ageb_urbanas.columns)
    )

    print()
    print("CRS:")
    print(
        gdf_ageb_urbanas.crs
    )

    print()
    print("Tipos de geometría:")
    print(
        gdf_ageb_urbanas.geom_type.value_counts()
    )

    print()
    print("Bounds:")
    print(
        gdf_ageb_urbanas.total_bounds
    )

    # --------------------------------------------------------
    # FILTRAR AGEB URBANAS DE MANZANILLO
    # --------------------------------------------------------

    ageb_urbanas_manzanillo = gdf_ageb_urbanas[
        gdf_ageb_urbanas["CVE_MUN"]
        .astype(str)
        .str.zfill(3)
        == "007"
    ].copy()

    print()
    print(
        "AGEB urbanas del municipio de Manzanillo:"
    )
    print("-" * 72)

    columnas_ageb_urbanas = [
        columna
        for columna in [
            "CVEGEO",
            "CVE_ENT",
            "CVE_MUN",
            "CVE_LOC",
            "CVE_AGEB",
        ]
        if columna in ageb_urbanas_manzanillo.columns
    ]

    print(
        ageb_urbanas_manzanillo[
            columnas_ageb_urbanas
        ]
        .sort_values(
            by=[
                columna
                for columna in [
                    "CVE_LOC",
                    "CVE_AGEB",
                ]
                if columna in ageb_urbanas_manzanillo.columns
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Total de AGEB urbanas en Manzanillo: "
        f"{len(ageb_urbanas_manzanillo)}"
    )

    # --------------------------------------------------------
    # AGEB URBANAS DE LA LOCALIDAD MANZANILLO
    # --------------------------------------------------------

    if "CVE_LOC" in ageb_urbanas_manzanillo.columns:

        ageb_localidad_manzanillo = (
            ageb_urbanas_manzanillo[
                ageb_urbanas_manzanillo["CVE_LOC"]
                .astype(str)
                .str.zfill(4)
                == "0001"
            ].copy()
        )

        print()
        print(
            "AGEB urbanas de la localidad Manzanillo "
            "(CVE_LOC = 0001):"
        )
        print("-" * 72)

        print(
            ageb_localidad_manzanillo[
                columnas_ageb_urbanas
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Total de AGEB urbanas en la localidad "
            f"Manzanillo: {len(ageb_localidad_manzanillo)}"
        )

    # --------------------------------------------------------
    # INSPECCIONAR AGEB RURALES
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA AGEB RURALES — COLIMA")
    print("=" * 72)

    gdf_ageb_rurales = gis.cargar_shapefile(
        estado="Colima",
        nombre_shp="06ar.shp",
        crs_destino="EPSG:4326",
    )

    print()
    print(
        f"AGEB rurales encontradas en el estado: "
        f"{len(gdf_ageb_rurales)}"
    )

    print()
    print("Columnas:")
    print(
        list(gdf_ageb_rurales.columns)
    )

    print()
    print("CRS:")
    print(
        gdf_ageb_rurales.crs
    )

    print()
    print("Tipos de geometría:")
    print(
        gdf_ageb_rurales.geom_type.value_counts()
    )

    print()
    print("Bounds:")
    print(
        gdf_ageb_rurales.total_bounds
    )

    # --------------------------------------------------------
    # FILTRAR AGEB RURALES DE MANZANILLO
    # --------------------------------------------------------

    ageb_rurales_manzanillo = gdf_ageb_rurales[
        gdf_ageb_rurales["CVE_MUN"]
        .astype(str)
        .str.zfill(3)
        == "007"
    ].copy()

    print()
    print(
        "AGEB rurales del municipio de Manzanillo:"
    )
    print("-" * 72)

    columnas_ageb_rurales = [
        columna
        for columna in [
            "CVEGEO",
            "CVE_ENT",
            "CVE_MUN",
            "CVE_AGEB",
        ]
        if columna in ageb_rurales_manzanillo.columns
    ]

    print(
        ageb_rurales_manzanillo[
            columnas_ageb_rurales
        ]
        .sort_values(
            by="CVE_AGEB"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Total de AGEB rurales en Manzanillo: "
        f"{len(ageb_rurales_manzanillo)}"
    )
  
      # --------------------------------------------------------
    # PROBAR obtener_ageb_urbanas()
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_ageb_urbanas()")
    print("=" * 72)

    ageb_urbanas_manzanillo = gis.obtener_ageb_urbanas(
        estado="Colima",
        municipio="Manzanillo",
        localidad="Manzanillo",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print(
        "AGEB urbanas de la localidad Manzanillo: "
        f"{len(ageb_urbanas_manzanillo)}"
    )

    print()
    print("Primeras 10 AGEB urbanas:")
    print("-" * 72)

    print(
        ageb_urbanas_manzanillo[
            [
                "CVEGEO",
                "CVE_MUN",
                "CVE_LOC",
                "CVE_AGEB",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("Tipos de geometría:")
    print(
        ageb_urbanas_manzanillo
        .geom_type
        .value_counts()
    )

    print()
    print("Bounds:")
    print(
        ageb_urbanas_manzanillo.total_bounds
    )

    # --------------------------------------------------------
    # PROBAR obtener_ageb_rurales()
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA obtener_ageb_rurales()")
    print("=" * 72)

    ageb_rurales_manzanillo = gis.obtener_ageb_rurales(
        estado="Colima",
        municipio="Manzanillo",
        crs_destino="EPSG:4326",
        territorio="principal",
    )

    print()
    print(
        "AGEB rurales del municipio de Manzanillo: "
        f"{len(ageb_rurales_manzanillo)}"
    )

    print()
    print("AGEB rurales:")
    print("-" * 72)

    print(
        ageb_rurales_manzanillo[
            [
                "CVEGEO",
                "CVE_MUN",
                "CVE_AGEB",
            ]
        ]
        .sort_values(
            "CVE_AGEB"
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Tipos de geometría:")
    print(
        ageb_rurales_manzanillo
        .geom_type
        .value_counts()
    )

    print()
    print("Bounds:")
    print(
        ageb_rurales_manzanillo.total_bounds
    )

    # --------------------------------------------------------
    # VALIDACIÓN AUTOMÁTICA
    # --------------------------------------------------------

    if len(ageb_urbanas_manzanillo) != 133:
        raise ValueError(
            "Se esperaban 133 AGEB urbanas "
            "para la localidad de Manzanillo."
        )

    if len(ageb_rurales_manzanillo) != 12:
        raise ValueError(
            "Se esperaban 12 AGEB rurales "
            "para el municipio de Manzanillo."
        )

    print()
    print("Validación AGEB correcta.")
  
    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PRUEBA FINALIZADA")
    print("=" * 72)


if __name__ == "__main__":
    main()