import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd


class GISManager:
    """
    Gestor de información geoespacial del Marco Geoestadístico
    del INEGI.

    Primera versión:
    - detecta paquetes estatales ZIP;
    - lista entidades disponibles;
    - localiza archivos SHP dentro de un estado;
    - extrae temporalmente las capas necesarias;
    - permite inspeccionar su estructura.

    El gestor no modifica los archivos oficiales originales.
    """

    # --------------------------------------------------------
    # CATÁLOGO OFICIAL DE ENTIDADES FEDERATIVAS
    # --------------------------------------------------------

    ESTADOS_MEXICO = {
        "01": "Aguascalientes",
        "02": "Baja California",
        "03": "Baja California Sur",
        "04": "Campeche",
        "05": "Coahuila de Zaragoza",
        "06": "Colima",
        "07": "Chiapas",
        "08": "Chihuahua",
        "09": "Ciudad de México",
        "10": "Durango",
        "11": "Guanajuato",
        "12": "Guerrero",
        "13": "Hidalgo",
        "14": "Jalisco",
        "15": "México",
        "16": "Michoacán de Ocampo",
        "17": "Morelos",
        "18": "Nayarit",
        "19": "Nuevo León",
        "20": "Oaxaca",
        "21": "Puebla",
        "22": "Querétaro",
        "23": "Quintana Roo",
        "24": "San Luis Potosí",
        "25": "Sinaloa",
        "26": "Sonora",
        "27": "Tabasco",
        "28": "Tamaulipas",
        "29": "Tlaxcala",
        "30": "Veracruz de Ignacio de la Llave",
        "31": "Yucatán",
        "32": "Zacatecas",
    }

    def __init__(self, root):
        """
        Parámetros
        ----------
        root : str o Path
            Carpeta donde se encuentran los paquetes ZIP estatales
            del Marco Geoestadístico.
        """
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta del Marco Geoestadístico: "
                f"{self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"La ruta indicada no es una carpeta: {self.root}"
            )

        self._paquetes = None

    # ========================================================
    # DETECCIÓN DE PAQUETES ESTATALES
    # ========================================================

    def detectar_paquetes(self, refrescar=False):
        """
        Detecta archivos ZIP cuyos nombres comienzan con la
        clave de entidad de dos dígitos.

        Ejemplos:
        01_aguascalientes.zip
        06_colima.zip
        32_zacatecas.zip

        Retorna
        -------
        dict
            {
                "06": Path(".../06_colima.zip"),
                ...
            }
        """
        if self._paquetes is not None and not refrescar:
            return self._paquetes.copy()

        paquetes = {}

        for path_zip in sorted(self.root.glob("*.zip")):
            match = re.match(
                r"^(\d{2})[_-]",
                path_zip.name,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            clave = match.group(1)

            if clave not in self.ESTADOS_MEXICO:
                continue

            paquetes[clave] = path_zip

        self._paquetes = paquetes

        return paquetes.copy()

    # ========================================================
    # LISTADO DE ESTADOS
    # ========================================================

    def listar_estados(self):
        """
        Lista las entidades disponibles físicamente en la carpeta.

        Retorna
        -------
        pd.DataFrame
            Columnas:
            - clave
            - estado
            - archivo_zip
            - disponible
        """
        paquetes = self.detectar_paquetes()

        filas = []

        for clave, nombre in self.ESTADOS_MEXICO.items():
            path_zip = paquetes.get(clave)

            filas.append({
                "clave": clave,
                "estado": nombre,
                "archivo_zip": (
                    str(path_zip)
                    if path_zip is not None
                    else None
                ),
                "disponible": path_zip is not None,
            })

        return pd.DataFrame(filas)

    # ========================================================
    # RESOLUCIÓN DE ESTADO
    # ========================================================

    def resolver_estado(self, estado):
        """
        Convierte una clave o nombre de estado en:

        (clave, nombre)

        Ejemplos
        --------
        "06"      -> ("06", "Colima")
        "Colima"  -> ("06", "Colima")
        """
        texto = str(estado).strip()

        if texto in self.ESTADOS_MEXICO:
            return (
                texto,
                self.ESTADOS_MEXICO[texto],
            )

        texto_normalizado = texto.casefold()

        for clave, nombre in self.ESTADOS_MEXICO.items():
            if nombre.casefold() == texto_normalizado:
                return clave, nombre

        raise ValueError(
            f"No se reconoció la entidad federativa: {estado}"
        )

    # ========================================================
    # OBTENER ZIP DEL ESTADO
    # ========================================================

    def obtener_zip_estado(self, estado):
        """
        Devuelve la ruta del ZIP correspondiente a un estado.
        """
        clave, nombre = self.resolver_estado(
            estado
        )

        paquetes = self.detectar_paquetes()

        if clave not in paquetes:
            raise FileNotFoundError(
                f"No se encontró el paquete del estado "
                f"{nombre} ({clave})."
            )

        return paquetes[clave]

    # ========================================================
    # INSPECCIÓN DEL CONTENIDO ZIP
    # ========================================================

    def listar_archivos_estado(self, estado):
        """
        Lista los archivos contenidos dentro del ZIP del estado.

        Retorna
        -------
        pd.DataFrame
            Columnas:
            - ruta_interna
            - nombre
            - extension
        """
        zip_path = self.obtener_zip_estado(
            estado
        )

        filas = []

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                ruta = Path(
                    info.filename
                )

                filas.append({
                    "ruta_interna": info.filename,
                    "nombre": ruta.name,
                    "extension": ruta.suffix.lower(),
                })

        return pd.DataFrame(filas)

    # ========================================================
    # LISTAR SHAPEFILES
    # ========================================================

    def listar_shapefiles_estado(self, estado):
        """
        Lista únicamente archivos .shp dentro del paquete estatal.
        """
        df = self.listar_archivos_estado(
            estado
        )

        if df.empty:
            return df

        return (
            df[
                df["extension"] == ".shp"
            ]
            .copy()
            .reset_index(drop=True)
        )

    # ========================================================
    # EXTRAER UN SHAPEFILE COMPLETO
    # ========================================================

    def extraer_shapefile_temporal(
        self,
        estado,
        nombre_shp,
    ):
        """
        Extrae un Shapefile junto con todos sus archivos auxiliares
        (.dbf, .shx, .prj, .cpg, etc.) a una carpeta temporal.

        Parámetros
        ----------
        estado : str
            Clave o nombre del estado.

        nombre_shp : str
            Nombre exacto del archivo .shp dentro del ZIP.

        Retorna
        -------
        temp_dir : tempfile.TemporaryDirectory
            Debe mantenerse vivo mientras se use el archivo.

        shp_path : Path
            Ruta temporal al .shp extraído.
        """
        zip_path = self.obtener_zip_estado(
            estado
        )

        if not nombre_shp.lower().endswith(
            ".shp"
        ):
            raise ValueError(
                "nombre_shp debe terminar en '.shp'."
            )

        stem_objetivo = Path(
            nombre_shp
        ).stem

        temp_dir = tempfile.TemporaryDirectory()

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            miembros = []

            for info in zf.infolist():
                if info.is_dir():
                    continue

                path_interno = Path(
                    info.filename
                )

                if (
                    path_interno.stem
                    == stem_objetivo
                ):
                    miembros.append(
                        info
                    )

            if not miembros:
                temp_dir.cleanup()

                raise FileNotFoundError(
                    f"No se encontró el Shapefile "
                    f"'{nombre_shp}' dentro de {zip_path.name}."
                )

            for info in miembros:
                zf.extract(
                    info,
                    path=temp_dir.name,
                )

        candidatos = list(
            Path(temp_dir.name).rglob(
                f"{stem_objetivo}.shp"
            )
        )

        if not candidatos:
            temp_dir.cleanup()

            raise FileNotFoundError(
                "Se extrajeron los componentes, pero no fue posible "
                "localizar el archivo .shp."
            )

        shp_path = candidatos[0]

        return temp_dir, shp_path

    # ========================================================
    # CARGAR SHAPEFILE
    # ========================================================

    def cargar_shapefile(
        self,
        estado,
        nombre_shp,
        crs_destino=None,
    ):
        """
        Lee cualquier Shapefile del paquete estatal.

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
        """
        temp_dir, shp_path = (
            self.extraer_shapefile_temporal(
                estado=estado,
                nombre_shp=nombre_shp,
            )
        )

        try:
            gdf = gpd.read_file(
                shp_path
            )

            if (
                crs_destino is not None
                and gdf.crs is not None
            ):
                gdf = gdf.to_crs(
                    crs_destino
                )

            return gdf

        finally:
            temp_dir.cleanup()

    # ========================================================
    # OBTENER LÍMITE DE UNA ENTIDAD FEDERATIVA
    # ========================================================

    def obtener_limite_estado(
        self,
        estado,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene el límite oficial de una entidad federativa.

        Parámetros
        ----------
        estado : str
            Clave o nombre de la entidad.

        crs_destino : str o CRS
            CRS final del GeoDataFrame.

        territorio : str
            "completo":
                Conserva toda la geometría oficial, incluyendo
                islas y territorios separados.

            "principal":
                Conserva únicamente el polígono de mayor superficie.

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
            Límite territorial solicitado.
        """

        clave, nombre = self.resolver_estado(
            estado
        )

        nombre_shp = f"{clave}ent.shp"

        gdf = self.cargar_shapefile(
            estado=clave,
            nombre_shp=nombre_shp,
            crs_destino=crs_destino,
        )

        if gdf.empty:
            raise ValueError(
                f"La capa territorial de {nombre} está vacía."
            )

        # Confirmar que realmente corresponde al estado solicitado
        if "CVE_ENT" in gdf.columns:
            claves = (
                gdf["CVE_ENT"]
                .astype(str)
                .str.zfill(2)
                .unique()
            )

            if clave not in claves:
                raise ValueError(
                    f"La capa '{nombre_shp}' no corresponde "
                    f"a la entidad {nombre} ({clave})."
                )

        territorio = str(
            territorio
        ).strip().lower()

        if territorio not in {
            "completo",
            "principal",
        }:
            raise ValueError(
                "territorio debe ser 'completo' o 'principal'."
            )

        if territorio == "principal":

            # Separar los componentes del MultiPolygon
            partes = (
                gdf
                .explode(
                    index_parts=False
                )
                .reset_index(drop=True)
            )

            if partes.empty:
                raise ValueError(
                    "No fue posible separar la geometría territorial."
                )

            # Calcular áreas en un CRS proyectado global.
            # No usamos grados para comparar superficies.
            partes_area = partes.to_crs(
                "EPSG:6933"
            )

            areas = partes_area.geometry.area

            indice_mayor = areas.idxmax()

            partes = partes.loc[
                [indice_mayor]
            ].copy()

            partes = partes.set_crs(
                gdf.crs,
                allow_override=True,
            )

            gdf = partes.reset_index(
                drop=True
            )

        return gdf
    
    # ========================================================
    # OBTENER MUNICIPIOS DE UNA ENTIDAD
    # ========================================================

    def obtener_municipios(
        self,
        estado,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene los municipios oficiales de una entidad federativa.

        Parámetros
        ----------
        estado : str
            Clave o nombre de la entidad.

        crs_destino : str o CRS
            CRS final del GeoDataFrame.

        territorio : str
            "completo":
                Conserva todas las geometrías oficiales.

            "principal":
                Para cada municipio conserva únicamente
                su polígono de mayor superficie.

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
            Municipios de la entidad.
        """

        clave, nombre = self.resolver_estado(
            estado
        )

        nombre_shp = f"{clave}mun.shp"

        gdf = self.cargar_shapefile(
            estado=clave,
            nombre_shp=nombre_shp,
            crs_destino=crs_destino,
        )

        if gdf.empty:
            raise ValueError(
                f"La capa municipal de {nombre} está vacía."
            )

        if "CVE_ENT" in gdf.columns:

            claves = (
                gdf["CVE_ENT"]
                .astype(str)
                .str.zfill(2)
                .unique()
            )

            if clave not in claves:
                raise ValueError(
                    f"La capa '{nombre_shp}' no corresponde "
                    f"a la entidad {nombre} ({clave})."
                )

        territorio = (
            str(territorio)
            .strip()
            .lower()
        )

        if territorio not in {
            "completo",
            "principal",
        }:
            raise ValueError(
                "territorio debe ser 'completo' o 'principal'."
            )

        if territorio == "principal":

            filas = []

            for _, row in gdf.iterrows():

                geom = row.geometry

                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type == "MultiPolygon":

                    partes = list(
                        geom.geoms
                    )

                    if partes:

                        gdf_partes = gpd.GeoSeries(
                            partes,
                            crs=gdf.crs,
                        )

                        gdf_partes_area = (
                            gdf_partes.to_crs(
                                "EPSG:6933"
                            )
                        )

                        areas = (
                            gdf_partes_area.area
                        )

                        geom = partes[
                            int(
                                np.argmax(
                                    areas.to_numpy()
                                )
                            )
                        ]

                nueva_fila = (
                    row.copy()
                )

                nueva_fila.geometry = geom

                filas.append(
                    nueva_fila
                )

            gdf = gpd.GeoDataFrame(
                filas,
                columns=gdf.columns,
                crs=gdf.crs,
            ).reset_index(
                drop=True
            )

        return gdf


    # ========================================================
    # OBTENER UN MUNICIPIO
    # ========================================================

    def obtener_municipio(
        self,
        estado,
        municipio,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene un municipio específico por nombre o clave.

        Parámetros
        ----------
        estado : str
            Clave o nombre del estado.

        municipio : str
            Nombre del municipio o clave municipal.

            Ejemplos:
            "Manzanillo"
            "007"

        crs_destino : str o CRS
            CRS final.

        territorio : str
            "completo" o "principal".

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
            Municipio solicitado.
        """

        gdf = self.obtener_municipios(
            estado=estado,
            crs_destino=crs_destino,
            territorio=territorio,
        )

        texto = (
            str(municipio)
            .strip()
        )

        # Buscar por clave municipal
        if (
            "CVE_MUN" in gdf.columns
            and texto.isdigit()
        ):

            clave_mun = (
                texto.zfill(3)
            )

            resultado = gdf[
                gdf["CVE_MUN"]
                .astype(str)
                .str.zfill(3)
                == clave_mun
            ].copy()

        # Buscar por nombre
        else:

            if "NOMGEO" not in gdf.columns:
                raise ValueError(
                    "La capa municipal no contiene "
                    "la columna NOMGEO."
                )

            resultado = gdf[
                gdf["NOMGEO"]
                .astype(str)
                .str.casefold()
                == texto.casefold()
            ].copy()

        if resultado.empty:
            raise ValueError(
                f"No se encontró el municipio "
                f"'{municipio}' en {estado}."
            )

        return resultado.reset_index(
            drop=True
        )
        
    # ========================================================
    # OBTENER LOCALIDADES DE UN MUNICIPIO
    # ========================================================

    def obtener_localidades(
        self,
        estado,
        municipio=None,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene localidades urbanas y rurales amanzanadas.

        Parámetros
        ----------
        estado : str
            Clave o nombre de la entidad.

        municipio : str o None
            Nombre o clave municipal.
            Si es None, devuelve todas las localidades del estado.

        crs_destino : str o CRS
            CRS final del GeoDataFrame.

        territorio : str
            "completo":
                Conserva todas las partes de la geometría.

            "principal":
                Si una localidad es MultiPolygon, conserva
                únicamente el polígono de mayor superficie.

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
            Localidades solicitadas.
        """

        clave_ent, nombre_ent = self.resolver_estado(
            estado
        )

        nombre_shp = f"{clave_ent}l.shp"

        gdf = self.cargar_shapefile(
            estado=clave_ent,
            nombre_shp=nombre_shp,
            crs_destino=crs_destino,
        )

        if gdf.empty:
            raise ValueError(
                f"La capa de localidades de "
                f"{nombre_ent} está vacía."
            )

        # ----------------------------------------------------
        # VALIDAR ENTIDAD
        # ----------------------------------------------------

        if "CVE_ENT" in gdf.columns:

            claves = (
                gdf["CVE_ENT"]
                .astype(str)
                .str.zfill(2)
                .unique()
            )

            if clave_ent not in claves:
                raise ValueError(
                    f"La capa '{nombre_shp}' no corresponde "
                    f"a la entidad {nombre_ent} ({clave_ent})."
                )

        # ----------------------------------------------------
        # FILTRAR MUNICIPIO
        # ----------------------------------------------------

        if municipio is not None:

            gdf_municipio = self.obtener_municipio(
                estado=estado,
                municipio=municipio,
                crs_destino=crs_destino,
                territorio="completo",
            )

            if "CVE_MUN" not in gdf_municipio.columns:
                raise ValueError(
                    "La capa municipal no contiene CVE_MUN."
                )

            clave_mun = (
                str(
                    gdf_municipio.iloc[0]["CVE_MUN"]
                )
                .zfill(3)
            )

            gdf = gdf[
                gdf["CVE_MUN"]
                .astype(str)
                .str.zfill(3)
                == clave_mun
            ].copy()

        # ----------------------------------------------------
        # VALIDAR TERRITORIO
        # ----------------------------------------------------

        territorio = (
            str(territorio)
            .strip()
            .lower()
        )

        if territorio not in {
            "completo",
            "principal",
        }:
            raise ValueError(
                "territorio debe ser "
                "'completo' o 'principal'."
            )

        # ----------------------------------------------------
        # CONSERVAR POLÍGONO PRINCIPAL
        # ----------------------------------------------------

        if territorio == "principal":

            filas = []

            for _, row in gdf.iterrows():

                geom = row.geometry

                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type == "MultiPolygon":

                    partes = list(
                        geom.geoms
                    )

                    if partes:

                        serie_partes = gpd.GeoSeries(
                            partes,
                            crs=gdf.crs,
                        )

                        areas = (
                            serie_partes
                            .to_crs("EPSG:6933")
                            .area
                        )

                        indice = int(
                            np.argmax(
                                areas.to_numpy()
                            )
                        )

                        geom = partes[indice]

                nueva_fila = row.copy()
                nueva_fila.geometry = geom

                filas.append(
                    nueva_fila
                )

            gdf = gpd.GeoDataFrame(
                filas,
                columns=gdf.columns,
                crs=gdf.crs,
            ).reset_index(
                drop=True
            )

        return gdf.reset_index(
            drop=True
        )


    # ========================================================
    # OBTENER UNA LOCALIDAD
    # ========================================================

    def obtener_localidad(
        self,
        estado,
        municipio,
        localidad,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene una localidad específica por nombre o clave.

        Parámetros
        ----------
        estado : str
            Clave o nombre de la entidad.

        municipio : str
            Nombre o clave municipal.

        localidad : str
            Nombre o clave de localidad.

            Ejemplos:
            "Manzanillo"
            "0001"

        crs_destino : str o CRS
            CRS final.

        territorio : str
            "completo" o "principal".

        Retorna
        -------
        gdf : geopandas.GeoDataFrame
            Localidad solicitada.
        """

        gdf = self.obtener_localidades(
            estado=estado,
            municipio=municipio,
            crs_destino=crs_destino,
            territorio=territorio,
        )

        texto = str(
            localidad
        ).strip()

        # ----------------------------------------------------
        # BUSCAR POR CLAVE
        # ----------------------------------------------------

        if (
            "CVE_LOC" in gdf.columns
            and texto.isdigit()
        ):

            clave_loc = (
                texto.zfill(4)
            )

            resultado = gdf[
                gdf["CVE_LOC"]
                .astype(str)
                .str.zfill(4)
                == clave_loc
            ].copy()

        # ----------------------------------------------------
        # BUSCAR POR NOMBRE
        # ----------------------------------------------------

        else:

            if "NOMGEO" not in gdf.columns:
                raise ValueError(
                    "La capa de localidades no contiene "
                    "la columna NOMGEO."
                )

            resultado = gdf[
                gdf["NOMGEO"]
                .astype(str)
                .str.casefold()
                == texto.casefold()
            ].copy()

        if resultado.empty:
            raise ValueError(
                f"No se encontró la localidad "
                f"'{localidad}' dentro del municipio "
                f"'{municipio}', {estado}."
            )

        return resultado.reset_index(
            drop=True
        )   
    # ========================================================
    # OBTENER AGEB URBANAS
    # ========================================================

    def obtener_ageb_urbanas(
        self,
        estado,
        municipio=None,
        localidad=None,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene Áreas Geoestadísticas Básicas urbanas.

        Puede filtrarse por:
        - estado
        - municipio
        - localidad

        Parámetros
        ----------
        estado : str
            Clave o nombre de la entidad.

        municipio : str o None
            Nombre o clave municipal.

        localidad : str o None
            Nombre o clave de localidad.

        crs_destino : str o CRS
            CRS final.

        territorio : str
            "completo" o "principal".

        Retorna
        -------
        geopandas.GeoDataFrame
        """

        clave_ent, nombre_ent = self.resolver_estado(
            estado
        )

        nombre_shp = f"{clave_ent}a.shp"

        gdf = self.cargar_shapefile(
            estado=clave_ent,
            nombre_shp=nombre_shp,
            crs_destino=crs_destino,
        )

        if gdf.empty:
            raise ValueError(
                f"La capa de AGEB urbanas de "
                f"{nombre_ent} está vacía."
            )

        # ----------------------------------------------------
        # MUNICIPIO
        # ----------------------------------------------------

        if municipio is not None:

            gdf_municipio = self.obtener_municipio(
                estado=estado,
                municipio=municipio,
                crs_destino=crs_destino,
                territorio="completo",
            )

            clave_mun = (
                str(
                    gdf_municipio.iloc[0]["CVE_MUN"]
                )
                .zfill(3)
            )

            gdf = gdf[
                gdf["CVE_MUN"]
                .astype(str)
                .str.zfill(3)
                == clave_mun
            ].copy()

        # ----------------------------------------------------
        # LOCALIDAD
        # ----------------------------------------------------

        if localidad is not None:

            if municipio is None:
                raise ValueError(
                    "Para filtrar por localidad debe "
                    "especificarse también el municipio."
                )

            gdf_localidad = self.obtener_localidad(
                estado=estado,
                municipio=municipio,
                localidad=localidad,
                crs_destino=crs_destino,
                territorio="completo",
            )

            clave_loc = (
                str(
                    gdf_localidad.iloc[0]["CVE_LOC"]
                )
                .zfill(4)
            )

            gdf = gdf[
                gdf["CVE_LOC"]
                .astype(str)
                .str.zfill(4)
                == clave_loc
            ].copy()

        # ----------------------------------------------------
        # VALIDAR RESULTADO
        # ----------------------------------------------------

        if gdf.empty:
            raise ValueError(
                "No se encontraron AGEB urbanas "
                "para los criterios solicitados."
            )

        return self._aplicar_territorio_principal(
            gdf=gdf,
            territorio=territorio,
        )
    # ========================================================
    # OBTENER AGEB RURALES
    # ========================================================

    def obtener_ageb_rurales(
        self,
        estado,
        municipio=None,
        crs_destino="EPSG:4326",
        territorio="completo",
    ):
        """
        Obtiene Áreas Geoestadísticas Básicas rurales.

        Las AGEB rurales se filtran por estado y municipio.
        """

        clave_ent, nombre_ent = self.resolver_estado(
            estado
        )

        nombre_shp = f"{clave_ent}ar.shp"

        gdf = self.cargar_shapefile(
            estado=clave_ent,
            nombre_shp=nombre_shp,
            crs_destino=crs_destino,
        )

        if gdf.empty:
            raise ValueError(
                f"La capa de AGEB rurales de "
                f"{nombre_ent} está vacía."
            )

        if municipio is not None:

            gdf_municipio = self.obtener_municipio(
                estado=estado,
                municipio=municipio,
                crs_destino=crs_destino,
                territorio="completo",
            )

            clave_mun = (
                str(
                    gdf_municipio.iloc[0]["CVE_MUN"]
                )
                .zfill(3)
            )

            gdf = gdf[
                gdf["CVE_MUN"]
                .astype(str)
                .str.zfill(3)
                == clave_mun
            ].copy()
            
        if gdf.empty:
            raise ValueError(
                "No se encontraron AGEB rurales "
                "para los criterios solicitados."
            )
            
        return self._aplicar_territorio_principal(
            gdf=gdf,
            territorio=territorio,
        )
    # ========================================================
    # NORMALIZAR TERRITORIO PRINCIPAL
    # ========================================================

    def _aplicar_territorio_principal(
        self,
        gdf,
        territorio="completo",
    ):
        """
        Conserva la geometría completa o únicamente el
        polígono principal de cada registro.

        Se utiliza internamente para estados, municipios,
        localidades y AGEB.
        """

        if gdf is None or gdf.empty:
            return gdf

        territorio = (
            str(territorio)
            .strip()
            .lower()
        )

        if territorio not in {
            "completo",
            "principal",
        }:
            raise ValueError(
                "territorio debe ser "
                "'completo' o 'principal'."
            )

        if territorio == "completo":
            return gdf.reset_index(
                drop=True
            )

        filas = []

        for _, row in gdf.iterrows():

            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "MultiPolygon":

                partes = list(
                    geom.geoms
                )

                if partes:

                    serie = gpd.GeoSeries(
                        partes,
                        crs=gdf.crs,
                    )

                    areas = (
                        serie
                        .to_crs("EPSG:6933")
                        .area
                        .to_numpy()
                    )

                    geom = partes[
                        int(
                            np.argmax(areas)
                        )
                    ]

            nueva_fila = row.copy()
            nueva_fila.geometry = geom

            filas.append(
                nueva_fila
            )

        return gpd.GeoDataFrame(
            filas,
            columns=gdf.columns,
            crs=gdf.crs,
        ).reset_index(
            drop=True
        )