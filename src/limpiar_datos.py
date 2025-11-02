# ========================================
# limpiar_datos.py
# Proyecto: MD-Grupo02
# Fase: Limpieza y normalización de datos
# ========================================

import os
import pandas as pd
import numpy as np
from unidecode import unidecode

# Ruta del archivo combinado
INPUT_PATH = os.path.join("..", "data_clean", "raw_combined.csv")
OUTPUT_PATH = os.path.join("..", "data_clean", "df_clean_admision_2023_2026.csv")

# ========================================
# 1. CARGA DE DATOS
# ========================================
print("Cargando dataset combinado...")
df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)
print(f"Registros cargados: {len(df)}")

# Normalizar nombres de columnas (mayúsculas y sin acentos)
df.columns = [unidecode(c.strip().upper().replace(" ", "_").replace(".", "")) for c in df.columns]

# ========================================
# 2. CORRECCIÓN DE VARIANTES EN NOMBRES DE COLUMNAS
# ========================================
# Corregir posibles errores de codificación
rename_map = {
    "OBSERVACI&OACUTEN": "OBSERVACION",
    "MERITOE_P": "MERITOE_P",
    "MERITOEP": "MERITOE_P",
    "MERITO_EP": "MERITOE_P",
    "MERITOE_P_ALCANZA_VACANTE": "MERITOE_P",
    "PUNTAJE_FINAL": "PUNTAJE",
    "ESCUELA": "ESCUELA_PROFESIONAL",
    "ESCUELA_PROFESIONAL_(PRIMERA_OPCION)": "ESCUELA_PROFESIONAL"
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# ========================================
# 3. UNIFICACIÓN DE COLUMNAS EQUIVALENTES
# ========================================

# Asegurar columnas base
for col in ["CODIGO", "APELLIDOS_Y_NOMBRES", "ESCUELA_PROFESIONAL", "PUNTAJE", "MERITOE_P", "OBSERVACION", "PROCESO"]:
    if col not in df.columns:
        df[col] = np.nan

# Función auxiliar: devuelve serie segura o serie vacía
def safe_series(df, col):
    """Devuelve una Serie si existe, o una Serie vacía del mismo tamaño."""
    if col in df.columns:
        return df[col]
    else:
        return pd.Series([np.nan] * len(df), index=df.index)

# --- Unificar ESCUELA_PROFESIONAL ---
df["ESCUELA_PROFESIONAL"] = df["ESCUELA_PROFESIONAL"].fillna("")  # asegurar tipo str
esc_primera = safe_series(df, "ESCUELA_PROFESIONAL_(PRIMERA_OPCION)")
esc_simple = safe_series(df, "ESCUELA")
df["ESCUELA_PROFESIONAL"] = df["ESCUELA_PROFESIONAL"].replace("", np.nan)
df["ESCUELA_PROFESIONAL"] = df["ESCUELA_PROFESIONAL"].fillna(esc_primera)
df["ESCUELA_PROFESIONAL"] = df["ESCUELA_PROFESIONAL"].fillna(esc_simple)

# --- Unificar PUNTAJE ---
puntaje_final = safe_series(df, "PUNTAJE_FINAL")
df["PUNTAJE"] = df["PUNTAJE"].fillna(puntaje_final)

# --- Unificar MERITO ---
merito_ep = safe_series(df, "MERITO_EP")
meritoep = safe_series(df, "MERITOEP")
df["MERITOE_P"] = df["MERITOE_P"].fillna(merito_ep)
df["MERITOE_P"] = df["MERITOE_P"].fillna(meritoep)



# ========================================
# 4. LIMPIEZA DE TEXTO Y NORMALIZACIÓN
# ========================================

from pandas.api.types import is_scalar

def clean_text(val):
    """Convierte cualquier valor a texto limpio sin acentos y en mayúsculas."""
    # Si el valor no es escalar (por ejemplo, lista o Serie), lo convertimos a str directamente
    if not is_scalar(val):
        val = str(val)
    # Si el valor es nulo, devolver cadena vacía
    try:
        if pd.isna(val):
            return ""
    except Exception:
        val = str(val)
    # Convertir a string si aún no lo es
    if not isinstance(val, str):
        val = str(val)
    # Quitar tildes y dejar mayúsculas
    try:
        val = unidecode(val)
    except Exception:
        pass
    return val.upper().strip()

# Aplicar limpieza a las columnas textuales
for col in ["CODIGO", "APELLIDOS_Y_NOMBRES", "ESCUELA_PROFESIONAL", "OBSERVACION", "PROCESO"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)


# ========================================
# 5. TRATAMIENTO DE VALORES EN "OBSERVACION"
# ========================================
# Si alguna observación dice "ALCANZO VACANTE SEGUNDA OPCION" → la dejamos vacía
df.loc[
    df["OBSERVACION"].str.contains("ALCANZO VACANTE SEGUNDA OPCION", case=False, na=False),
    "OBSERVACION"
] = ""

# ========================================
# 6. CONVERSIÓN DE VARIABLES NUMÉRICAS
# ========================================

# --- Eliminar duplicados exactos en nombres de columna ---
df = df.loc[:, ~df.columns.duplicated(keep="first")]

# --- Detectar columnas relacionadas con puntaje ---
cols_puntaje = [c for c in df.columns if "PUNTAJE" in c.upper()]
if len(cols_puntaje) > 1:
    print(f"⚠️  Se detectaron múltiples columnas de PUNTAJE: {cols_puntaje}")
    # Crear columna unificada
    df["PUNTAJE_UNIFICADO"] = np.nan
    for c in cols_puntaje:
        if c != "PUNTAJE_UNIFICADO":
            if isinstance(df[c], pd.Series):
                df["PUNTAJE_UNIFICADO"] = df["PUNTAJE_UNIFICADO"].fillna(df[c])
            else:
                df["PUNTAJE_UNIFICADO"] = df["PUNTAJE_UNIFICADO"].fillna(df[c].iloc[:, 0])
    # Eliminar las columnas originales (excepto la nueva)
    for c in cols_puntaje:
        if c != "PUNTAJE_UNIFICADO":
            del df[c]
    # Renombrar la unificada
    df.rename(columns={"PUNTAJE_UNIFICADO": "PUNTAJE"}, inplace=True)
else:
    if "PUNTAJE" not in df.columns:
        raise ValueError("No se encontró ninguna columna de PUNTAJE.")

# --- Detectar columnas relacionadas con mérito ---
cols_merito = [c for c in df.columns if "MERITO" in c.upper()]
if len(cols_merito) > 1:
    print(f"⚠️  Se detectaron múltiples columnas de MÉRITO: {cols_merito}")
    df["MERITOE_P_UNIFICADO"] = np.nan
    for c in cols_merito:
        if c != "MERITOE_P_UNIFICADO":
            if isinstance(df[c], pd.Series):
                df["MERITOE_P_UNIFICADO"] = df["MERITOE_P_UNIFICADO"].fillna(df[c])
            else:
                df["MERITOE_P_UNIFICADO"] = df["MERITOE_P_UNIFICADO"].fillna(df[c].iloc[:, 0])
    for c in cols_merito:
        if c != "MERITOE_P_UNIFICADO":
            del df[c]
    df.rename(columns={"MERITOE_P_UNIFICADO": "MERITOE_P"}, inplace=True)
else:
    if "MERITOE_P" not in df.columns:
        df["MERITOE_P"] = np.nan

# --- Conversión a tipo numérico limpio ---
def clean_num(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").replace(" ", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def clean_int(x):
    if pd.isna(x):
        return np.nan
    x = ''.join(ch for ch in str(x) if ch.isdigit())
    return int(x) if x else np.nan

# Crear respaldo de columnas antes de limpiar
df["PUNTAJE_RAW"] = df["PUNTAJE"]
df["MERITOE_P_RAW"] = df["MERITOE_P"]

# Aplicar limpieza
df["PUNTAJE"] = df["PUNTAJE"].apply(clean_num)
df["MERITOE_P"] = df["MERITOE_P"].apply(clean_int)



# ========================================
# 7. ELIMINACIÓN DE DUPLICADOS Y REGISTROS INVÁLIDOS
# ========================================
# Eliminar filas sin puntaje
before = len(df)
df = df[df["PUNTAJE"].notna()].copy()
after = len(df)
print(f"Eliminados {before - after} registros sin puntaje.")

# Eliminar duplicados (mismo CODIGO + PROCESO)
before_dup = len(df)
df.drop_duplicates(subset=["CODIGO", "PROCESO"], keep="first", inplace=True)
after_dup = len(df)
print(f"Eliminados {before_dup - after_dup} registros duplicados.")

# ========================================
# 8. VERIFICACIÓN DE RANGOS DE PUNTAJE
# ========================================
min_p, max_p = 0, 2000
out_of_range = df[~df["PUNTAJE"].between(min_p, max_p)]
print(f"Registros fuera de rango de puntaje: {len(out_of_range)}")

# ========================================
# 9. SELECCIÓN FINAL DE COLUMNAS Y EXPORTACIÓN
# ========================================
final_cols = [
    "CODIGO",
    "APELLIDOS_Y_NOMBRES",
    "ESCUELA_PROFESIONAL",
    "PUNTAJE",
    "MERITOE_P",
    "OBSERVACION",
    "PROCESO",
    "ARCHIVO_ORIGEN"
]

# Conservar solo las que existen
final_cols = [c for c in final_cols if c in df.columns]
df_clean = df[final_cols].copy()

# Guardar archivo limpio
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\n✅ Limpieza completada correctamente.")
print(f"Archivo limpio guardado en: {OUTPUT_PATH}")
print(f"Total de registros limpios: {len(df_clean)}")

# Vista rápida
print("\nVista de las primeras filas:")
print(df_clean.head())
