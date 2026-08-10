from pathlib import Path

import matplotlib.pyplot as plt

from scripts.boundary import (
    plot_geotiff_recortado,
    recortar_geotiff_con_limite,
)


# ============================================================
# NORMALIZACIÓN DEL NIVEL TERRITORIAL
# ============================================================

def normalizar_nivel_territorial(
    nivel,
):
    """
    Normaliza y valida el nivel territorial solicitado.

    Niveles soportados
    ------------------
    - estado
    - municipio
    - localidad
    - ageb_urbana
    - ageb_rural
    """

    nivel_normalizado = (
        str(nivel)
        .strip()
        .lower()
    )

    alias = {
        "entidad": "estado",
        "entidad federativa": "estado",
        "estado": "estado",

        "municipio": "municipio",
        "municipal": "municipio",

        "localidad": "localidad",
        "local": "localidad",

        "ageb urbana": "ageb_urbana",
        "ageb_urbana": "ageb_urbana",
        "urbana": "ageb_urbana",

        "ageb rural": "ageb_rural",
        "ageb_rural": "ageb_rural",
        "rural": "ageb_rural",
    }

    if nivel_normalizado not in alias:
        raise ValueError(
            "Nivel territorial no válido. "
            "Use uno de: 'estado', 'municipio', "
            "'localidad', 'ageb_urbana' o 'ageb_rural'."
        )

    return alias[
        nivel_normalizado
    ]


# ============================================================
# NORMALIZAR CLAVE AGEB
# ============================================================

def normalizar_clave_ageb(
    ageb,
):
    """
    Normaliza una clave AGEB.

    Ejemplos
    --------
    564   -> 0564
    0564  -> 0564
    9A    -> 009A
    030a  -> 030A
    """

    if ageb is None:
        raise ValueError(
            "Debe especificarse una clave AGEB."
        )

    clave = (
        str(ageb)
        .strip()
        .upper()
    )

    if not clave:
        raise ValueError(
            "La clave AGEB no puede estar vacía."
        )

    return clave.zfill(4)


# ============================================================
# SELECCIONAR UNA AGEB
# ============================================================

def seleccionar_ageb(
    gdf,
    ageb,
):
    """
    Selecciona una AGEB específica mediante CVE_AGEB.
    """

    if gdf is None or gdf.empty:
        raise ValueError(
            "No existen AGEB disponibles para realizar "
            "la selección."
        )

    if "CVE_AGEB" not in gdf.columns:
        raise ValueError(
            "La capa no contiene la columna CVE_AGEB."
        )

    clave_ageb = normalizar_clave_ageb(
        ageb
    )

    resultado = gdf[
        gdf["CVE_AGEB"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.zfill(4)
        == clave_ageb
    ].copy()

    if resultado.empty:
        raise ValueError(
            f"No se encontró la AGEB '{clave_ageb}' "
            "para los criterios territoriales solicitados."
        )

    return (
        resultado.reset_index(drop=True),
        clave_ageb,
    )


# ============================================================
# OBTENER LÍMITE SEGÚN NIVEL TERRITORIAL
# ============================================================

def obtener_limite_por_nivel(
    gis,
    nivel,
    estado,
    municipio=None,
    localidad=None,
    ageb=None,
    territorio="principal",
    crs_destino="EPSG:4326",
):
    """
    Obtiene automáticamente el límite territorial solicitado
    mediante GISManager.

    Niveles
    -------
    estado
    municipio
    localidad
    ageb_urbana
    ageb_rural
    """

    nivel = normalizar_nivel_territorial(
        nivel
    )

    clave_estado, nombre_estado = (
        gis.resolver_estado(
            estado
        )
    )

    # ========================================================
    # ESTADO
    # ========================================================

    if nivel == "estado":

        limite = gis.obtener_limite_estado(
            estado=estado,
            crs_destino=crs_destino,
            territorio=territorio,
        )

        metadata = {
            "nivel": "estado",
            "estado": nombre_estado,
            "clave_estado": clave_estado,
            "municipio": None,
            "clave_municipio": None,
            "localidad": None,
            "clave_localidad": None,
            "ageb": None,
            "tipo_ageb": None,
            "territorio": territorio,
            "crs": str(limite.crs),
            "bounds": tuple(
                limite.total_bounds
            ),
        }

        return limite, metadata

    # ========================================================
    # MUNICIPIO
    # ========================================================

    if nivel == "municipio":

        if municipio is None:
            raise ValueError(
                "Debe especificarse 'municipio' "
                "cuando nivel='municipio'."
            )

        limite = gis.obtener_municipio(
            estado=estado,
            municipio=municipio,
            crs_destino=crs_destino,
            territorio=territorio,
        )

        nombre_municipio = (
            str(
                limite.iloc[0]["NOMGEO"]
            )
            if "NOMGEO" in limite.columns
            else str(municipio)
        )

        clave_municipio = (
            str(
                limite.iloc[0]["CVE_MUN"]
            ).zfill(3)
            if "CVE_MUN" in limite.columns
            else None
        )

        metadata = {
            "nivel": "municipio",
            "estado": nombre_estado,
            "clave_estado": clave_estado,
            "municipio": nombre_municipio,
            "clave_municipio": clave_municipio,
            "localidad": None,
            "clave_localidad": None,
            "ageb": None,
            "tipo_ageb": None,
            "territorio": territorio,
            "crs": str(limite.crs),
            "bounds": tuple(
                limite.total_bounds
            ),
        }

        return limite, metadata

    # ========================================================
    # LOCALIDAD
    # ========================================================

    if nivel == "localidad":

        if municipio is None:
            raise ValueError(
                "Debe especificarse 'municipio' "
                "cuando nivel='localidad'."
            )

        if localidad is None:
            raise ValueError(
                "Debe especificarse 'localidad' "
                "cuando nivel='localidad'."
            )

        limite = gis.obtener_localidad(
            estado=estado,
            municipio=municipio,
            localidad=localidad,
            crs_destino=crs_destino,
            territorio=territorio,
        )

        municipio_gdf = gis.obtener_municipio(
            estado=estado,
            municipio=municipio,
            crs_destino=crs_destino,
            territorio="completo",
        )

        nombre_municipio = (
            str(
                municipio_gdf.iloc[0]["NOMGEO"]
            )
        )

        clave_municipio = (
            str(
                limite.iloc[0]["CVE_MUN"]
            ).zfill(3)
        )

        nombre_localidad = (
            str(
                limite.iloc[0]["NOMGEO"]
            )
        )

        clave_localidad = (
            str(
                limite.iloc[0]["CVE_LOC"]
            ).zfill(4)
        )

        metadata = {
            "nivel": "localidad",
            "estado": nombre_estado,
            "clave_estado": clave_estado,
            "municipio": nombre_municipio,
            "clave_municipio": clave_municipio,
            "localidad": nombre_localidad,
            "clave_localidad": clave_localidad,
            "ageb": None,
            "tipo_ageb": None,
            "territorio": territorio,
            "crs": str(limite.crs),
            "bounds": tuple(
                limite.total_bounds
            ),
        }

        return limite, metadata

    # ========================================================
    # AGEB URBANA
    # ========================================================

    if nivel == "ageb_urbana":

        if municipio is None:
            raise ValueError(
                "Debe especificarse 'municipio' "
                "para una AGEB urbana."
            )

        if localidad is None:
            raise ValueError(
                "Debe especificarse 'localidad' "
                "para una AGEB urbana."
            )

        if ageb is None:
            raise ValueError(
                "Debe especificarse 'ageb' "
                "para una AGEB urbana."
            )

        ageb_disponibles = (
            gis.obtener_ageb_urbanas(
                estado=estado,
                municipio=municipio,
                localidad=localidad,
                crs_destino=crs_destino,
                territorio=territorio,
            )
        )

        limite, clave_ageb = seleccionar_ageb(
            gdf=ageb_disponibles,
            ageb=ageb,
        )

        localidad_gdf = gis.obtener_localidad(
            estado=estado,
            municipio=municipio,
            localidad=localidad,
            crs_destino=crs_destino,
            territorio="completo",
        )

        municipio_gdf = gis.obtener_municipio(
            estado=estado,
            municipio=municipio,
            crs_destino=crs_destino,
            territorio="completo",
        )

        nombre_municipio = str(
            municipio_gdf.iloc[0]["NOMGEO"]
        )

        clave_municipio = str(
            limite.iloc[0]["CVE_MUN"]
        ).zfill(3)

        nombre_localidad = str(
            localidad_gdf.iloc[0]["NOMGEO"]
        )

        clave_localidad = str(
            limite.iloc[0]["CVE_LOC"]
        ).zfill(4)

        metadata = {
            "nivel": "ageb_urbana",
            "estado": nombre_estado,
            "clave_estado": clave_estado,
            "municipio": nombre_municipio,
            "clave_municipio": clave_municipio,
            "localidad": nombre_localidad,
            "clave_localidad": clave_localidad,
            "ageb": clave_ageb,
            "tipo_ageb": "urbana",
            "territorio": territorio,
            "crs": str(limite.crs),
            "bounds": tuple(
                limite.total_bounds
            ),
        }

        return limite, metadata

    # ========================================================
    # AGEB RURAL
    # ========================================================

    if nivel == "ageb_rural":

        if municipio is None:
            raise ValueError(
                "Debe especificarse 'municipio' "
                "para una AGEB rural."
            )

        if ageb is None:
            raise ValueError(
                "Debe especificarse 'ageb' "
                "para una AGEB rural."
            )

        ageb_disponibles = (
            gis.obtener_ageb_rurales(
                estado=estado,
                municipio=municipio,
                crs_destino=crs_destino,
                territorio=territorio,
            )
        )

        limite, clave_ageb = seleccionar_ageb(
            gdf=ageb_disponibles,
            ageb=ageb,
        )

        municipio_gdf = gis.obtener_municipio(
            estado=estado,
            municipio=municipio,
            crs_destino=crs_destino,
            territorio="completo",
        )

        nombre_municipio = str(
            municipio_gdf.iloc[0]["NOMGEO"]
        )

        clave_municipio = str(
            limite.iloc[0]["CVE_MUN"]
        ).zfill(3)

        metadata = {
            "nivel": "ageb_rural",
            "estado": nombre_estado,
            "clave_estado": clave_estado,
            "municipio": nombre_municipio,
            "clave_municipio": clave_municipio,
            "localidad": None,
            "clave_localidad": None,
            "ageb": clave_ageb,
            "tipo_ageb": "rural",
            "territorio": territorio,
            "crs": str(limite.crs),
            "bounds": tuple(
                limite.total_bounds
            ),
        }

        return limite, metadata

    raise RuntimeError(
        "No fue posible resolver el nivel territorial."
    )


# ============================================================
# CONSTRUIR NOMBRE DEL TERRITORIO
# ============================================================

def construir_nombre_territorio(
    metadata,
):
    """
    Construye una descripción legible del territorio.
    """

    nivel = metadata.get(
        "nivel"
    )

    estado = str(
        metadata.get(
            "estado",
            ""
        )
    ).strip()

    municipio = str(
        metadata.get(
            "municipio",
            ""
        )
    ).strip()

    localidad = str(
        metadata.get(
            "localidad",
            ""
        )
    ).strip()

    ageb = metadata.get(
        "ageb"
    )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    if nivel == "estado":
        return estado

    # --------------------------------------------------------
    # MUNICIPIO
    # --------------------------------------------------------

    if nivel == "municipio":
        return (
            f"{municipio}, "
            f"{estado}"
        )

    # --------------------------------------------------------
    # LOCALIDAD
    # --------------------------------------------------------

    if nivel == "localidad":

        if (
            localidad.casefold()
            == municipio.casefold()
        ):
            return (
                f"Localidad de {localidad}, "
                f"{estado}"
            )

        return (
            f"{localidad}, "
            f"{municipio}, "
            f"{estado}"
        )

    # --------------------------------------------------------
    # AGEB URBANA
    # --------------------------------------------------------

    if nivel == "ageb_urbana":

        if (
            localidad.casefold()
            == municipio.casefold()
        ):
            return (
                f"AGEB urbana {ageb}, "
                f"localidad de {localidad}, "
                f"{estado}"
            )

        return (
            f"AGEB urbana {ageb}, "
            f"{localidad}, "
            f"{municipio}, "
            f"{estado}"
        )

    # --------------------------------------------------------
    # AGEB RURAL
    # --------------------------------------------------------

    if nivel == "ageb_rural":
        return (
            f"AGEB rural {ageb}, "
            f"{municipio}, "
            f"{estado}"
        )

    return "Territorio"


# ============================================================
# CONSTRUIR NOMBRE SEGURO PARA ARCHIVO
# ============================================================

def construir_slug_territorio(
    metadata,
):
    """
    Construye un identificador reproducible para archivos.
    """

    nivel = metadata.get(
        "nivel"
    )

    partes = []

    if nivel == "estado":

        partes = [
            metadata.get("clave_estado"),
            metadata.get("estado"),
        ]

    elif nivel == "municipio":

        partes = [
            metadata.get("clave_estado"),
            metadata.get("clave_municipio"),
            metadata.get("municipio"),
        ]

    elif nivel == "localidad":

        partes = [
            metadata.get("clave_estado"),
            metadata.get("clave_municipio"),
            metadata.get("clave_localidad"),
            metadata.get("localidad"),
        ]

    elif nivel == "ageb_urbana":

        partes = [
            metadata.get("clave_estado"),
            metadata.get("clave_municipio"),
            metadata.get("clave_localidad"),
            metadata.get("ageb"),
            "ageb_urbana",
        ]

    elif nivel == "ageb_rural":

        partes = [
            metadata.get("clave_estado"),
            metadata.get("clave_municipio"),
            metadata.get("ageb"),
            "ageb_rural",
        ]

    texto = "_".join(
        str(parte)
        for parte in partes
        if parte is not None
    )

    reemplazos = {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
        "(": "",
        ")": "",
    }

    for original, nuevo in reemplazos.items():
        texto = texto.replace(
            original,
            nuevo,
        )

    while "__" in texto:
        texto = texto.replace(
            "__",
            "_",
        )

    return texto.strip(
        "_"
    )


# ============================================================
# RECORTAR RASTER POR TERRITORIO
# ============================================================

def recortar_raster_por_territorio(
    gis,
    input_tif,
    output_tif,
    nivel,
    estado,
    municipio=None,
    localidad=None,
    ageb=None,
    territorio="principal",
    crs_destino="EPSG:4326",
    crop=True,
    all_touched=False,
    nodata=None,
    overwrite=True,
    compress="deflate",
):
    """
    Recorta un GeoTIFF utilizando un territorio obtenido
    automáticamente mediante GISManager.
    """

    input_tif = Path(
        input_tif
    )

    output_tif = Path(
        output_tif
    )

    if not input_tif.exists():
        raise FileNotFoundError(
            f"No se encontró el raster original: {input_tif}"
        )

    limite, metadata_territorio = (
        obtener_limite_por_nivel(
            gis=gis,
            nivel=nivel,
            estado=estado,
            municipio=municipio,
            localidad=localidad,
            ageb=ageb,
            territorio=territorio,
            crs_destino=crs_destino,
        )
    )

    resultado_raster = (
        recortar_geotiff_con_limite(
            input_tif=input_tif,
            output_tif=output_tif,
            gdf_limite=limite,
            crop=crop,
            all_touched=all_touched,
            nodata=nodata,
            overwrite=overwrite,
            compress=compress,
        )
    )

    return {
        "nivel": metadata_territorio["nivel"],
        "territorio": metadata_territorio,
        "limite": limite,
        "raster": resultado_raster,
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
    }


# ============================================================
# GENERAR MAPA RASTER TERRITORIAL
# ============================================================

def generar_mapa_raster_territorial(
    gis,
    input_tif,
    output_tif,
    output_png,
    nivel,
    estado,
    municipio=None,
    localidad=None,
    ageb=None,
    territorio="principal",
    crs_destino="EPSG:4326",
    title=None,
    colorbar_label="Valor",
    cmap="YlOrRd",
    convertir_a_porcentaje=False,
    mostrar_limite=True,
    suavizar_visual=False,
    sigma_suavizado=1.2,
    interpolation_display="bilinear",
    figsize=(11, 9),
    dpi=220,
    crop=True,
    all_touched=False,
    nodata=None,
    overwrite=True,
    compress="deflate",
):
    """
    Flujo completo:

    1. Resuelve el territorio mediante GISManager.
    2. Recorta el GeoTIFF.
    3. Genera el PNG.
    4. Devuelve los metadatos.

    El raster científico original no se modifica.
    """

    input_tif = Path(
        input_tif
    )

    output_tif = Path(
        output_tif
    )

    output_png = Path(
        output_png
    )

    output_tif.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # RECORTE
    # --------------------------------------------------------

    resultado = recortar_raster_por_territorio(
        gis=gis,
        input_tif=input_tif,
        output_tif=output_tif,
        nivel=nivel,
        estado=estado,
        municipio=municipio,
        localidad=localidad,
        ageb=ageb,
        territorio=territorio,
        crs_destino=crs_destino,
        crop=crop,
        all_touched=all_touched,
        nodata=nodata,
        overwrite=overwrite,
        compress=compress,
    )

    limite = resultado[
        "limite"
    ]

    metadata_territorio = resultado[
        "territorio"
    ]

    # --------------------------------------------------------
    # NOMBRE
    # --------------------------------------------------------

    nombre_territorio = (
        construir_nombre_territorio(
            metadata_territorio
        )
    )

    if title is None:
        title = (
            f"Superficie interpolada — "
            f"{nombre_territorio}"
        )

    # --------------------------------------------------------
    # FIGURA
    # --------------------------------------------------------

    fig, ax, metadata_mapa = (
        plot_geotiff_recortado(
            path_tif=output_tif,
            gdf_limite=limite,
            title=title,
            colorbar_label=colorbar_label,
            cmap=cmap,
            convertir_a_porcentaje=convertir_a_porcentaje,
            mostrar_limite=mostrar_limite,
            suavizar_visual=suavizar_visual,
            sigma_suavizado=sigma_suavizado,
            interpolation_display=interpolation_display,
            figsize=figsize,
        )
    )

    fig.savefig(
        output_png,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "nivel": metadata_territorio["nivel"],
        "nombre_territorio": nombre_territorio,
        "slug_territorio": construir_slug_territorio(
            metadata_territorio
        ),
        "territorio": metadata_territorio,
        "limite": limite,
        "raster": resultado["raster"],
        "mapa": metadata_mapa,
        "input_tif": str(input_tif),
        "output_tif": str(output_tif),
        "output_png": str(output_png),
    }