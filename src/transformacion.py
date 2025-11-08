import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def transformar_datos():
    # ---------- 1. RUTAS DEL PROYECTO ----------
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")
    ruta_entrada = os.path.join(ruta_resultados, "datos_limpios.csv")

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró el archivo limpio: {ruta_entrada}")

    print(f"📂 Cargando datos limpios desde: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")
    print(f"Registros cargados: {len(df)}")

    # ---------- 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS ----------
    print("🔢 Codificando variables categóricas...")

    # Crear y aplicar codificadores
    le_escuela = LabelEncoder()
    le_obs = LabelEncoder()

    df["ESCUELA_COD"] = le_escuela.fit_transform(df["ESCUELA PROFESIONAL"])
    df["OBSERVACION_COD"] = le_obs.fit_transform(df["OBSERVACION"])

    # ---------- 3. GENERACIÓN DE VARIABLES DERIVADAS ----------
    print("🧮 Generando variables derivadas...")

    # Promedio histórico de puntaje por escuela
    df["PROMEDIO_ESCUELA"] = df.groupby("ESCUELA PROFESIONAL")["PUNTAJE"].transform("mean")

    # Diferencia entre el puntaje individual y el promedio de su escuela
    df["DIFERENCIA_PROMEDIO"] = df["PUNTAJE"] - df["PROMEDIO_ESCUELA"]

    # ---------- 4. NORMALIZACIÓN DE VARIABLES NUMÉRICAS ----------
    print("📏 Normalizando variables numéricas...")

    scaler = MinMaxScaler()
    columnas_a_normalizar = ["PROMEDIO_ESCUELA", "DIFERENCIA_PROMEDIO"]
    df[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

    # ---------- 5. GUARDAR TRANSFORMADORES PARA FUTURAS PREDICCIONES ----------
    print("💾 Guardando transformadores (encoders y scaler)...")
    transformadores = {
        "le_escuela": le_escuela,
        "le_obs": le_obs,
        "scaler": scaler
    }
    ruta_transformadores = os.path.join(ruta_resultados, "transformadores.pkl")
    joblib.dump(transformadores, ruta_transformadores)
    print(f"🧠 Transformadores guardados en: {ruta_transformadores}")

    # ---------- 6. DIVISIÓN DEL CONJUNTO DE DATOS ----------
    print("✂️ Dividiendo conjunto de entrenamiento y prueba...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # ---------- 7. GUARDAR RESULTADOS ----------
    ruta_transformado = os.path.join(ruta_resultados, "datos_transformados.csv")
    ruta_train = os.path.join(ruta_resultados, "train.csv")
    ruta_test = os.path.join(ruta_resultados, "test.csv")

    df.to_csv(ruta_transformado, index=False, encoding="utf-8-sig")
    train_df.to_csv(ruta_train, index=False, encoding="utf-8-sig")
    test_df.to_csv(ruta_test, index=False, encoding="utf-8-sig")

    print(f"\n✅ Transformación completada exitosamente.")
    print(f"💾 Archivo principal guardado en: {ruta_transformado}")
    print(f"💾 Entrenamiento: {ruta_train}")
    print(f"💾 Prueba: {ruta_test}")
    print(f"\nColumnas finales: {list(df.columns)}")

    return df, train_df, test_df


# ---------- EJECUCIÓN DIRECTA ----------
if __name__ == "__main__":
    transformar_datos()
