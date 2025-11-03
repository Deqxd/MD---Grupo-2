import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def transformar_datos():
    # --- Rutas del proyecto ---
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")
    ruta_entrada = os.path.join(ruta_resultados, "datos_limpios.csv")

    # --- Cargar dataset limpio ---
    print(f"📂 Cargando datos limpios desde: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada, encoding="utf-8-sig")
    print(f"Registros cargados: {len(df)}")

    # --- 1. Codificación de variables categóricas ---
    print("🔢 Codificando variables categóricas...")
    le_escuela = LabelEncoder()
    le_obs = LabelEncoder()

    df["ESCUELA_COD"] = le_escuela.fit_transform(df["ESCUELA PROFESIONAL"])
    df["OBSERVACION_COD"] = le_obs.fit_transform(df["OBSERVACION"])

    # --- 2. Normalización de puntajes ---
    print("📏 Normalizando puntajes...")
    scaler = MinMaxScaler()
    df["PUNTAJE_NORM"] = scaler.fit_transform(df[["PUNTAJE"]])

    # --- 3. Generación de variables derivadas ---
    print("🧮 Generando variables derivadas...")
    # Promedio histórico de puntaje por escuela
    promedio_escuela = df.groupby("ESCUELA PROFESIONAL")["PUNTAJE"].transform("mean")
    df["PROMEDIO_ESCUELA"] = promedio_escuela

    # Diferencia de puntaje respecto al promedio
    df["DIFERENCIA_PROMEDIO"] = df["PUNTAJE"] - df["PROMEDIO_ESCUELA"]

    # --- 4. División de datos (80% entrenamiento, 20% prueba) ---
    print("✂️ Dividiendo conjunto de entrenamiento y prueba...")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # --- 5. Guardar datasets transformados ---
    ruta_transformado = os.path.join(ruta_resultados, "datos_transformados.csv")
    ruta_train = os.path.join(ruta_resultados, "train.csv")
    ruta_test = os.path.join(ruta_resultados, "test.csv")

    df.to_csv(ruta_transformado, index=False, encoding="utf-8-sig")
    train_df.to_csv(ruta_train, index=False, encoding="utf-8-sig")
    test_df.to_csv(ruta_test, index=False, encoding="utf-8-sig")

    print(f"✅ Transformación completa.")
    print(f"💾 Archivo principal guardado en: {ruta_transformado}")
    print(f"💾 Entrenamiento: {ruta_train}")
    print(f"💾 Prueba: {ruta_test}")
    print(f"\nColumnas finales: {list(df.columns)}")

    return df, train_df, test_df


if __name__ == "__main__":
    transformar_datos()
