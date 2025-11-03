import pandas as pd
import os
from unidecode import unidecode

def limpiar_datos():
    # --- Rutas del proyecto ---
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")
    ruta_entrada = os.path.join(ruta_resultados, "datos_unificados.csv")
    ruta_salida = os.path.join(ruta_resultados, "datos_limpios.csv")

    # --- Cargar archivo unificado ---
    print(f"📂 Cargando archivo: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")

    print(f"Registros iniciales: {len(df)}")

    # --- 1. Eliminar duplicados ---
    df = df.drop_duplicates(subset=["CODIGO", "PROCESO"], keep="first")

    # --- 2. Eliminar filas con puntaje nulo o no numérico ---
    df["PUNTAJE"] = pd.to_numeric(df["PUNTAJE"], errors="coerce")
    df = df.dropna(subset=["PUNTAJE"])

    # --- 3. Verificar rango válido de puntajes (0–2000) ---
    df = df[(df["PUNTAJE"] >= 0) & (df["PUNTAJE"] <= 2000)]

    # --- 4. Normalizar textos ---
    def limpiar_texto(txt):
        if pd.isna(txt):
            return "SIN OBSERVACION"
        txt = unidecode(str(txt).strip().upper())
        txt = txt.replace("Ã", "A").replace("Â", "")
        return txt

    df["ESCUELA PROFESIONAL"] = df["ESCUELA PROFESIONAL"].apply(limpiar_texto)
    df["OBSERVACION"] = df["OBSERVACION"].apply(limpiar_texto)
    df["APELLIDOS Y NOMBRES"] = df["APELLIDOS Y NOMBRES"].apply(limpiar_texto)

    # --- 5. Normalizar nombres de columnas por seguridad ---
    df.columns = [unidecode(c.strip().upper()) for c in df.columns]

    # --- 6. Mostrar resumen ---
    print(f"✅ Registros finales limpios: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    # --- 7. Guardar dataset limpio ---
    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    print(f"💾 Archivo limpio guardado en: {ruta_salida}")

    return df


if __name__ == "__main__":
    df_limpio = limpiar_datos()
    print("\nVista previa:")
    print(df_limpio.head())
