import os
import pandas as pd
from unidecode import unidecode

# ==========================================================
# CONFIGURACIÓN DE RUTAS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datos_admision")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

procesos = ["2023-II", "2024-I", "2024-II", "2025-I", "2025-II", "2026-I"]
dfs = []

# ==========================================================
# FUNCIÓN AUXILIAR PARA NORMALIZAR ENCABEZADOS
# ==========================================================
def limpiar_columnas(cols):
    cols = [unidecode(c).upper().strip().replace(" ", "_") for c in cols]
    # eliminar duplicaciones tipo "(PRIMERA_OPCION)" mal codificadas
    cols = [c.replace("(", "").replace(")", "").replace("__", "_") for c in cols]
    return cols

# ==========================================================
# LECTURA Y LIMPIEZA INICIAL
# ==========================================================
for proceso in procesos:
    print(f"\nCargando archivos del proceso {proceso}...")
    proceso_path = os.path.join(DATA_DIR, proceso)

    for archivo in os.listdir(proceso_path):
        if archivo.lower().endswith(".csv"):
            ruta_csv = os.path.join(proceso_path, archivo)

            # Lectura segura (intenta latin-1, luego utf-8)
            try:
                df = pd.read_csv(ruta_csv, encoding="latin-1")
            except UnicodeDecodeError:
                df = pd.read_csv(ruta_csv, encoding="utf-8")

            # Normalizar encabezados
            df.columns = limpiar_columnas(df.columns)

            df["PROCESO"] = proceso
            df["ARCHIVO_ORIGEN"] = archivo

            # ======================================================
            # LIMPIEZA ESPECÍFICA PARA 2024-I Y 2024-II
            # ======================================================
            if proceso in ["2024-I", "2024-II"]:
                # Buscar columna OBSERVACION
                obs_col = next((c for c in df.columns if "OBSERVACION" in c), None)

                if obs_col:
                    df[obs_col] = df[obs_col].astype(str)
                    mask_segunda = df[obs_col].str.contains(
                        "ALCANZO VACANTE SEGUNDA OPCION", case=False, na=False
                    )

                    # 1️⃣ Vaciar OBSERVACION
                    df.loc[mask_segunda, obs_col] = ""

                    # 2️⃣ Vaciar todas las columnas de MERITO relacionadas
                    for c in df.columns:
                        if "MERITO" in c:
                            df.loc[mask_segunda, c] = ""

                    # 3️⃣ Vaciar cualquier columna con SEGUNDA_OPCION
                    for c in df.columns:
                        if "SEGUNDA_OPCION" in c:
                            df.loc[mask_segunda, c] = ""

                # 4️⃣ Eliminar columnas de segunda opción (si existen)
                segunda_cols = [c for c in df.columns if "SEGUNDA_OPCION" in c]
                if segunda_cols:
                    df.drop(columns=segunda_cols, inplace=True, errors="ignore")

            # Agregar DataFrame procesado
            dfs.append(df)

# ==========================================================
# COMBINAR TODOS LOS DATAFRAMES
# ==========================================================
df_all = pd.concat(dfs, ignore_index=True)

# Eliminar duplicados exactos en nombres de columna
df_all = df_all.loc[:, ~df_all.columns.duplicated(keep="first")]

print(f"\nTotal de registros combinados: {len(df_all)}")
print("\nColumnas detectadas:")
print(list(df_all.columns))

# ==========================================================
# GUARDAR DATASET COMBINADO
# ==========================================================
output_path = os.path.join(OUTPUT_DIR, "raw_combined.csv")
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Archivo combinado guardado en: {output_path}")

print("\nVista preliminar de datos:")
print(df_all.head())
