import pandas as pd
import matplotlib.pyplot as plt
import os

def visualizar_resultados():
    # --- Rutas ---
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    ruta_resultados = os.path.join(ruta_raiz, "resultados")

    # --- Cargar resultados ---
    resultados_path = os.path.join(ruta_resultados, "resultados_modelos.csv")
    predicciones_path = os.path.join(ruta_resultados, "predicciones_modelos.csv")

    df_resultados = pd.read_csv(resultados_path)
    df_pred = pd.read_csv(predicciones_path)

    # --- 1. Comparación de métricas ---
    plt.figure(figsize=(8, 5))
    plt.bar(df_resultados["Modelo"], df_resultados["R2"], color="skyblue")
    plt.title("Comparación de R² entre Modelos", fontsize=14)
    plt.xlabel("Modelo")
    plt.ylabel("R²")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_resultados, "grafico_r2.png"), dpi=300)
    plt.show()

    # --- 2. Comparación de RMSE ---
    plt.figure(figsize=(8, 5))
    plt.bar(df_resultados["Modelo"], df_resultados["RMSE"], color="salmon")
    plt.title("Comparación de RMSE entre Modelos", fontsize=14)
    plt.xlabel("Modelo")
    plt.ylabel("RMSE")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_resultados, "grafico_rmse.png"), dpi=300)
    plt.show()

    # --- 3. Predicho vs Real (mejor modelo) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pred["REAL"], df_pred["PRED_XGB"], alpha=0.4, color="green")
    plt.title("Predicho vs Real (XGBoost)", fontsize=14)
    plt.xlabel("Puntaje Real")
    plt.ylabel("Puntaje Predicho")
    plt.plot([df_pred["REAL"].min(), df_pred["REAL"].max()],
             [df_pred["REAL"].min(), df_pred["REAL"].max()],
             color="red", linestyle="--", label="Línea ideal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_resultados, "grafico_pred_vs_real.png"), dpi=300)
    plt.show()

    print("✅ Gráficos generados y guardados en la carpeta 'resultados'.")


if __name__ == "__main__":
    visualizar_resultados()
