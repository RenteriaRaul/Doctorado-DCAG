from descargar_estaciones import descargar_estado


resultados, errores = descargar_estado(
    estado="COLIMA",
    carpeta_base="CONAGUA/datos",
    situacion="OPERANDO"
)