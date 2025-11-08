import pandas as pd
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def modelar_datos():
    # ---------- 1. RUTAS DEL PROYECTO ----------
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")
    ruta_train = os.path.join(ruta_resultados, "train.csv")
    ruta_test = os.path.join(ruta_resultados, "test.csv")

    # ---------- 2. CARGA DE DATOS ----------
    print("📂 Cargando conjuntos de datos de entrenamiento y prueba...")
    train_df = pd.read_csv(ruta_train, encoding="utf-8-sig")
    test_df = pd.read_csv(ruta_test, encoding="utf-8-sig")

    print(f"Entrenamiento: {len(train_df)} registros")
    print(f"Prueba: {len(test_df)} registros")

    # ---------- 3. SELECCIÓN DE VARIABLES ----------
    # ⚠️ Eliminamos "PUNTAJE_NORM" porque fue removido del pipeline anterior
    X_cols = ["ESCUELA_COD", "OBSERVACION_COD", "PROMEDIO_ESCUELA", "DIFERENCIA_PROMEDIO"]
    y_col = "PUNTAJE"

    X_train, y_train = train_df[X_cols], train_df[y_col]
    X_test, y_test = test_df[X_cols], test_df[y_col]

    resultados = []

    # ---------- 4. ENTRENAMIENTO DE MODELOS ----------
    # 4.1 Regresión Lineal
    print("\n🤖 Entrenando modelo: Regresión Lineal...")
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    pred_lr = modelo_lr.predict(X_test)
    r2_lr = r2_score(y_test, pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
    resultados.append(["Regresión Lineal", r2_lr, rmse_lr])

    # 4.2 Random Forest
    print("🌲 Entrenando modelo: Random Forest...")
    modelo_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo_rf.fit(X_train, y_train)
    pred_rf = modelo_rf.predict(X_test)
    r2_rf = r2_score(y_test, pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    resultados.append(["Random Forest", r2_rf, rmse_rf])

    # 4.3 XGBoost
    print("⚡ Entrenando modelo: XGBoost...")
    modelo_xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,  # 🔸 más estable
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    modelo_xgb.fit(X_train, y_train)
    pred_xgb = modelo_xgb.predict(X_test)
    r2_xgb = r2_score(y_test, pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    resultados.append(["XGBoost", r2_xgb, rmse_xgb])

    # ---------- 5. COMPARACIÓN DE RESULTADOS ----------
    df_resultados = pd.DataFrame(resultados, columns=["Modelo", "R2", "RMSE"])
    ruta_resultados_csv = os.path.join(ruta_resultados, "resultados_modelos.csv")
    df_resultados.to_csv(ruta_resultados_csv, index=False, encoding="utf-8-sig")

    print("\n📊 Resultados de evaluación:")
    print(df_resultados)
    print(f"💾 Resultados exportados en: {ruta_resultados_csv}")

    # ---------- 6. GUARDAR PREDICCIONES ----------
    predicciones = pd.DataFrame({
        "REAL": y_test,
        "PRED_LR": pred_lr,
        "PRED_RF": pred_rf,
        "PRED_XGB": pred_xgb
    })
    ruta_predicciones_csv = os.path.join(ruta_resultados, "predicciones_modelos.csv")
    predicciones.to_csv(ruta_predicciones_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Predicciones exportadas en: {ruta_predicciones_csv}")

    # ---------- 7. SELECCIÓN Y GUARDADO DEL MEJOR MODELO ----------
    mejor_modelo = df_resultados.loc[df_resultados["R2"].idxmax(), "Modelo"]
    print(f"\n🏆 Mejor modelo: {mejor_modelo}")

    if mejor_modelo == "Regresión Lineal":
        modelo_final = modelo_lr
    elif mejor_modelo == "Random Forest":
        modelo_final = modelo_rf
    else:
        modelo_final = modelo_xgb

    ruta_modelo = os.path.join(ruta_resultados, "modelo_final.pkl")
    joblib.dump(modelo_final, ruta_modelo)
    ruta_columnas = os.path.join(ruta_resultados, "columnas_entrenamiento.pkl")
    joblib.dump(X_cols, ruta_columnas)
    print(f"📄 Columnas de entrenamiento guardadas en: {ruta_columnas}")


    print(f"💾 Modelo guardado en: {ruta_modelo}")

    # ---------- 8. RETORNAR RESULTADOS ----------
    return df_resultados


# ---------- EJECUCIÓN DIRECTA ----------
if __name__ == "__main__":
    modelar_datos()
