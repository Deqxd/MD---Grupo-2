import pandas as pd
import os
import csv
import chardet
from unidecode import unidecode

# ---------- Función auxiliar para detectar y leer CSV correctamente ----------
def cargar_csv_robusto(ruta_archivo):
    """Lee un CSV detectando automáticamente encoding y delimitador."""
    # Detectar codificación probable
    with open(ruta_archivo, 'rb') as f:
        enc = chardet.detect(f.read(20000))['encoding']

    # Detectar delimitador probable
    with open(ruta_archivo, 'r', encoding=enc, errors='ignore') as f:
        sample = f.read(2000)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except:
            delim = ';'  # Por defecto si no se detecta

    # Leer CSV de forma segura
    df = pd.read_csv(
        ruta_archivo,
        encoding=enc,
        delimiter=delim,
        quotechar='"',
        skip_blank_lines=True,
        on_bad_lines='skip',
        engine='python'
    )

    # Eliminar posibles filas duplicadas del encabezado
    if df.iloc[0].astype(str).str.contains("CODIGO", case=False).any():
        df = df[1:]

    return df


# ---------- Función principal ----------
def cargar_datos():
    # Ruta raíz del proyecto
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_base = os.path.join(ruta_raiz, "datos_admision")

    if not os.path.exists(ruta_base):
        raise FileNotFoundError(f"No se encontró la carpeta de datos: {ruta_base}")

    df_total = pd.DataFrame()

    # Función para limpiar nombres de columnas
    def limpiar_nombre(col):
        col = unidecode(str(col).strip().upper())
        col = col.replace("Ó", "O").replace("&OACUTE", "O")
        col = col.replace("Ã", "O").replace("Ã", "A").replace("Â", "")
        col = col.replace("(PRIMERA OPCION)", "").strip()
        return col

    # Recorrer carpetas y archivos CSV
    for carpeta in os.listdir(ruta_base):
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        if os.path.isdir(ruta_carpeta):
            for archivo in os.listdir(ruta_carpeta):
                if archivo.endswith(".csv"):
                    ruta_archivo = os.path.join(ruta_carpeta, archivo)
                    print(f"📂 Cargando {ruta_archivo}...")

                    # Leer archivo de forma robusta
                    df = cargar_csv_robusto(ruta_archivo)

                    # Limpieza de nombres de columnas
                    df.columns = [limpiar_nombre(c) for c in df.columns]

                    # Buscar columnas relevantes por similitud
                    columnas_validas = {
                        "CODIGO": [c for c in df.columns if "COD" in c],
                        "APELLIDOS Y NOMBRES": [c for c in df.columns if "APELL" in c],
                        "ESCUELA PROFESIONAL": [c for c in df.columns if "ESCUELA" in c],
                        "PUNTAJE": [c for c in df.columns if "PUNTAJE" in c or "PUNTAJ" in c],
                        "MERITOE.P": [c for c in df.columns if "MERITO" in c],
                        "OBSERVACION": [c for c in df.columns if "OBSERV" in c],
                    }

                    # Crear un DataFrame temporal con las columnas correctas
                    df_temp = pd.DataFrame()
                    for col_final, posibles in columnas_validas.items():
                        if posibles:
                            df_temp[col_final] = df[posibles[0]]
                        else:
                            df_temp[col_final] = None  # si no existe, se rellena con nulos

                    # Agregar columna del proceso
                    df_temp["PROCESO"] = carpeta

                    # Unir al DataFrame total
                    df_total = pd.concat([df_total, df_temp], ignore_index=True)

    # Resultado final
    print(f"\n✅ Datos cargados y estandarizados: {df_total.shape[0]} registros totales.\n")
    print(f"Columnas finales: {list(df_total.columns)}")

    # Guardar archivo consolidado en carpeta resultados
    carpeta_resultados = os.path.join(ruta_raiz, "resultados")
    os.makedirs(carpeta_resultados, exist_ok=True)
    ruta_salida = os.path.join(carpeta_resultados, "datos_unificados.csv")

    df_total.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    print(f"💾 Archivo unificado guardado en: {ruta_salida}")

    return df_total


# ---------- Ejecución directa ----------
if __name__ == "__main__":
    df = cargar_datos()
    print("\nVista previa:")
    print(df.head())
