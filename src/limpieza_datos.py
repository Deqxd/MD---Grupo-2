import pandas as pd
import os
from unidecode import unidecode


def limpiar_datos():
    # ---------- 1. RUTAS DEL PROYECTO ----------
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")
    ruta_entrada = os.path.join(ruta_resultados, "datos_unificados.csv")
    ruta_salida = os.path.join(ruta_resultados, "datos_limpios.csv")

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró el archivo unificado: {ruta_entrada}")

    print(f"📂 Cargando archivo: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")

    print(f"Registros iniciales: {len(df)}")
    print(f"Columnas detectadas: {list(df.columns)}")

    # ---------- 2. ELIMINAR DUPLICADOS ----------
    # ✅ Conserva el primer registro por combinación de CODIGO y PROCESO
    df = df.drop_duplicates(subset=["CODIGO", "PROCESO"], keep="first")

    # ---------- 3. CONVERTIR PUNTAJE A NUMÉRICO ----------
    # ✅ Reemplaza caracteres erróneos y convierte
    df["PUNTAJE"] = pd.to_numeric(df["PUNTAJE"], errors="coerce")
    df = df.dropna(subset=["PUNTAJE"])  # elimina registros sin puntaje válido

    # ---------- 4. VERIFICAR RANGO VÁLIDO ----------
    # ✅ Mantén solo valores dentro del rango oficial (0–2000)
    df = df[(df["PUNTAJE"] >= 0) & (df["PUNTAJE"] <= 2000)]

    # ---------- 5. LIMPIEZA Y NORMALIZACIÓN DE TEXTO ----------
    def limpiar_texto(txt):
        if pd.isna(txt) or str(txt).strip() == "":
            return "SIN OBSERVACION"

        txt = unidecode(str(txt).strip().upper())
        txt = txt.replace("Ã", "A").replace("Â", "")

        # 🔍 Unificar las observaciones relacionadas con ingreso o vacante
        if "ALCANZO" in txt or "ALCANZO VACANTE" in txt or "VACANTE" in txt:
            return "ALCANZO VACANTE"
        elif "NO ALCANZO" in txt or "NO INGRESO" in txt:
            return "NO ALCANZO VACANTE"
        elif "EXONER" in txt:
            return "EXONERADO"
        else:
            return txt


    # ✅ Aplica la limpieza a las columnas textuales principales
    columnas_texto = ["ESCUELA PROFESIONAL", "OBSERVACION", "APELLIDOS Y NOMBRES"]
    for col in columnas_texto:
        if col in df.columns:
            df[col] = df[col].apply(limpiar_texto)
        else:
            df[col] = "SIN OBSERVACION"  # 🔹 evita errores si alguna columna faltara

    # ---------- 6. NORMALIZAR NOMBRES DE COLUMNAS ----------
    df.columns = [unidecode(c.strip().upper()) for c in df.columns]

    # ---------- 7. VALIDACIÓN FINAL ----------
    # 🔹 Revisa si existen valores nulos en columnas críticas
    columnas_clave = ["CODIGO", "ESCUELA PROFESIONAL", "PUNTAJE"]
    nulos = df[columnas_clave].isnull().sum()
    if nulos.any():
        print("\n⚠️  Advertencia: se detectaron valores nulos en columnas clave:")
        print(nulos[nulos > 0])

    # ---------- 8. GUARDAR ARCHIVO LIMPIO ----------
    print(f"\n✅ Registros finales limpios: {len(df)}")
    print(f"Columnas finales: {list(df.columns)}")

    os.makedirs(ruta_resultados, exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print(f"💾 Archivo limpio guardado en: {ruta_salida}")

    return df


# ---------- EJECUCIÓN DIRECTA ----------
if __name__ == "__main__":
    df_limpio = limpiar_datos()
    print("\nVista previa:")
    print(df_limpio.head())

