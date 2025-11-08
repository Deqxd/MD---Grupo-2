import pandas as pd
import os
import joblib
import numpy as np

def predecir_resultados():
    # ---------- 1. RUTAS ----------
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")

    ruta_modelo = os.path.join(ruta_resultados, "modelo_final.pkl")
    ruta_columnas = os.path.join(ruta_resultados, "columnas_entrenamiento.pkl")
    ruta_datos = os.path.join(ruta_resultados, "datos_limpios.csv")
    ruta_transformadores = os.path.join(ruta_resultados, "transformadores.pkl")

    # ---------- 2. VALIDACIONES ----------
    for ruta in [ruta_modelo, ruta_columnas, ruta_datos, ruta_transformadores]:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    # ---------- 3. CARGA DE MODELO Y TRANSFORMADORES ----------
    modelo = joblib.load(ruta_modelo)
    X_cols = joblib.load(ruta_columnas)
    transformadores = joblib.load(ruta_transformadores)

    le_escuela = transformadores["le_escuela"]
    le_obs = transformadores["le_obs"]
    scaler = transformadores["scaler"]

    print(f"📦 Modelo y transformadores cargados correctamente.")

    # ---------- 4. CARGA DE DATOS ----------
    df = pd.read_csv(ruta_datos, encoding="utf-8-sig")
    procesos = sorted(df["PROCESO"].unique())
    proceso_base = procesos[-1]  # Ejemplo: 2026-I
    df_pred = df[df["PROCESO"] == proceso_base].copy()
    print(f"🔍 Usando datos del proceso {proceso_base} como base para predecir 2026-II.")

    # ---------- 5. TRANSFORMACIONES ----------
    print("🔢 Aplicando codificadores...")
    df_pred["ESCUELA_COD"] = df_pred["ESCUELA PROFESIONAL"].map(
        lambda x: le_escuela.transform([x])[0] if x in le_escuela.classes_ else -1
    )
    df_pred["OBSERVACION_COD"] = df_pred["OBSERVACION"].map(
        lambda x: le_obs.transform([x])[0] if x in le_obs.classes_ else -1
    )

    print("🧮 Calculando variables derivadas...")
    df_pred["PROMEDIO_ESCUELA"] = df_pred.groupby("ESCUELA PROFESIONAL")["PUNTAJE"].transform("mean")
    df_pred["DIFERENCIA_PROMEDIO"] = df_pred["PUNTAJE"] - df_pred["PROMEDIO_ESCUELA"]

    # Escalar igual que en entrenamiento
    columnas_a_normalizar = ["PROMEDIO_ESCUELA", "DIFERENCIA_PROMEDIO"]
    df_pred[columnas_a_normalizar] = scaler.transform(df_pred[columnas_a_normalizar])

    # ---------- 6. PREDICCIÓN ----------
    X_pred = df_pred[X_cols]
    print("🤖 Realizando predicciones...")
    df_pred["PUNTAJE_PREDICTO"] = modelo.predict(X_pred)
    df_pred["PUNTAJE_PREDICTO"] = np.clip(df_pred["PUNTAJE_PREDICTO"], 0, 2000)

    # ---------- 7. RESUMEN POR ESCUELA (solo ingresantes) ----------
    print("📊 Calculando estadísticas solo para alumnos que consiguieron vacante...")

    # Filtramos únicamente los que fueron admitidos según la columna OBSERVACION
    df_ingresantes = df_pred[df_pred["OBSERVACION"].str.contains("ALCANZO", case=False, na=False) |
                            df_pred["OBSERVACION"].str.contains("VACANTE", case=False, na=False)].copy()

    # Agrupar por escuela profesional y calcular estadísticas solo de ingresantes
    resumen = df_ingresantes.groupby("ESCUELA PROFESIONAL").agg(
        MINIMO_PREDICHO=("PUNTAJE_PREDICTO", "min"),
        PROMEDIO_PREDICHO=("PUNTAJE_PREDICTO", "mean"),
        MAXIMO_PREDICHO=("PUNTAJE_PREDICTO", "max"),
        VACANTES_ESTIMADAS=("OBSERVACION", "count")  # cantidad de ingresantes estimada
    ).reset_index()

    # Calcular el total de postulantes (de todos) para luego obtener tasa
    totales = df_pred.groupby("ESCUELA PROFESIONAL")["OBSERVACION"].count().reset_index(name="TOTAL_POSTULANTES")

    # Unir ambos dataframes
    resumen = resumen.merge(totales, on="ESCUELA PROFESIONAL", how="left")

    # Calcular tasa de ingreso estimada (%)
    resumen["TASA_INGRESO_ESTIMADA"] = (resumen["VACANTES_ESTIMADAS"] / resumen["TOTAL_POSTULANTES"]) * 100


    # ---------- 8. GUARDAR RESULTADOS ----------
    ruta_pred_detalle = os.path.join(ruta_resultados, "predicciones_detalladas_2026II.csv")
    ruta_pred_resumen = os.path.join(ruta_resultados, "prediccion_por_escuela_2026II.csv")

    df_pred.to_csv(ruta_pred_detalle, index=False, encoding="utf-8-sig")
    resumen.to_csv(ruta_pred_resumen, index=False, encoding="utf-8-sig")

    print(f"💾 Resultados guardados:")
    print(f"   • {ruta_pred_detalle}")
    print(f"   • {ruta_pred_resumen}")
    print("✅ Proceso de predicción completado correctamente.")

    return resumen


if __name__ == "__main__":
    predecir_resultados()
