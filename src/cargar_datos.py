import pandas as pd
import os
import csv
import chardet
from unidecode import unidecode

# ✅ CONSERVAR: Este import es útil para limpiar tildes y caracteres especiales
# ✅ chardet y csv.Sniffer ayudan a detectar codificación y delimitador automáticamente


# ---------- Función auxiliar para detectar y leer CSV correctamente ----------
def cargar_csv_robusto(ruta_archivo):
    """Lee un CSV detectando automáticamente encoding y delimitador."""

    # ✅ CONSERVAR: detección automática de encoding
    with open(ruta_archivo, 'rb') as f:
        enc = chardet.detect(f.read(20000))['encoding']

    # ✅ CONSERVAR: detección automática de delimitador
    with open(ruta_archivo, 'r', encoding=enc, errors='ignore') as f:
        sample = f.read(2000)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except:
            delim = ';'  # 🔸 RECOMENDACIÓN: podrías probar primero con ',' antes que ';'
            # Ejemplo: delim = ',' if ',' in sample else ';'

    # ✅ CONSERVAR: lectura segura del archivo CSV
    df = pd.read_csv(
        ruta_archivo,
        encoding=enc,
        delimiter=delim,
        quotechar='"',
        skip_blank_lines=True,
        on_bad_lines='skip',
        engine='python'
    )

    # 🔸 MEJORAR: usa `df.columns.str.contains` para evitar errores con tipos
    if df.shape[0] > 0 and df.columns.astype(str).str.contains("CODIGO", case=False).any():
        # ⚠️ Tu condición actual revisa la primera fila, no las columnas.
        # Se reemplaza por una validación correcta:
        pass  # no eliminar filas aquí, esto se manejará después si es necesario

    return df


# ---------- Función principal ----------
def cargar_datos():
    # ✅ Ruta raíz del proyecto
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_base = os.path.join(ruta_raiz, "datos_admision")

    if not os.path.exists(ruta_base):
        raise FileNotFoundError(f"No se encontró la carpeta de datos: {ruta_base}")

    df_total = pd.DataFrame()

    # ✅ Función para limpiar nombres de columnas
    def limpiar_nombre(col):
        col = unidecode(str(col).strip().upper())
        # 🔸 RECOMENDACIÓN: agrega una limpieza más genérica
        col = (col.replace("&OACUTE", "O")
                  .replace("(PRIMERA OPCION)", "")
                  .replace("  ", " ")
                  .strip())
        return col

    # ✅ Recorrer carpetas y archivos CSV
    for carpeta in os.listdir(ruta_base):
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        if os.path.isdir(ruta_carpeta):
            for archivo in os.listdir(ruta_carpeta):
                if archivo.lower().endswith(".csv"):
                    ruta_archivo = os.path.join(ruta_carpeta, archivo)
                    print(f"📂 Cargando {ruta_archivo}...")

                    # ✅ Leer archivo de forma robusta
                    df = cargar_csv_robusto(ruta_archivo)

                    # ✅ Limpieza de nombres de columnas
                    df.columns = [limpiar_nombre(c) for c in df.columns]

                    # ✅ Mapeo flexible de columnas esperadas
                    columnas_validas = {
                        "CODIGO": [c for c in df.columns if "COD" in c],
                        "APELLIDOS Y NOMBRES": [c for c in df.columns if "APELL" in c],
                        "ESCUELA PROFESIONAL": [c for c in df.columns if "ESCUELA" in c],
                        "PUNTAJE": [c for c in df.columns if "PUNTAJE" in c or "PUNTAJ" in c],
                        "MERITOE.P": [c for c in df.columns if "MERITO" in c],
                        "OBSERVACION": [c for c in df.columns if "OBSERV" in c],
                    }

                    # ✅ Crear DataFrame temporal estandarizado
                    df_temp = pd.DataFrame()
                    for col_final, posibles in columnas_validas.items():
                        if posibles:
                            df_temp[col_final] = df[posibles[0]]
                        else:
                            df_temp[col_final] = None  # Mantener consistencia de columnas

                    # ✅ Agregar columna del proceso (ej: 2023-II)
                    df_temp["PROCESO"] = carpeta

                    # ✅ Concatenar al dataset total
                    df_total = pd.concat([df_total, df_temp], ignore_index=True)

    # ✅ Mensaje final
    print(f"\n✅ Datos cargados y estandarizados: {df_total.shape[0]} registros totales.\n")
    print(f"Columnas finales: {list(df_total.columns)}")

    # ✅ Guardar archivo consolidado
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

