#!/usr/bin/env python3
"""Genera `reportes/shap_summary.png` usando el modelo de producción.

Uso: python scripts/generar_shap.py
"""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys

REPORTES = Path(__file__).resolve().parents[1] / "reportes"
REPORTES.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(__file__).resolve().parents[1] / "modelos" / "predictor_production.joblib"
DATA_PATH = Path(__file__).resolve().parents[1] / "datos" / "procesados" / "dataset_procesado.parquet"

def main():
    try:
        import shap
    except Exception as e:
        print("Falta la librería `shap`. Instálala con `pip install -e .[shap]` o `pip install shap`.")
        raise

    if not MODEL_PATH.exists():
        print(f"No se encuentra el modelo en {MODEL_PATH}")
        sys.exit(2)
    if not DATA_PATH.exists():
        print(f"No se encuentra el dataset procesado en {DATA_PATH}")
        sys.exit(2)

    print("Cargando modelo...", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    print("Cargando datos procesados (parquet)...", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    # Nombre objetivo esperado
    objetivo = "Diabetes_binary"
    if objetivo in df.columns:
        X = df.drop(columns=[objetivo])
    else:
        X = df.copy()

    n_bg = min(500, len(X))
    print(f"Usando muestra de fondo de {n_bg} filas para SHAP")
    X_bg = X.sample(n=n_bg, random_state=42)

    # Intentar explainer óptimo, con fallbacks
    try:
        print("Intentando `shap.Explainer(model, X_bg)` (automático)...")
        explainer = shap.Explainer(model, X_bg)
        shap_values = explainer(X_bg)
    except Exception as e:
        print("Fallo explainer automático:", e)
        try:
            print("Intentando TreeExplainer con estimador interno si existe...")
            # Si es pipeline, intentar extraer el estimador final
            estimator = getattr(model, "estimators_") if hasattr(model, "estimators_") else None
            if hasattr(model, "named_steps"):
                # pipeline: último paso
                estimator = list(model.named_steps.values())[-1]
            if estimator is None:
                estimator = model
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer(X_bg)
        except Exception as e2:
            print("TreeExplainer también falló:", e2)
            print("Usando KernelExplainer (más lento) sobre función predict_proba)")
            # usar predict_proba si existe
            if hasattr(model, "predict_proba"):
                f = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.KernelExplainer(f, X_bg.iloc[:50])
                shap_values = explainer.shap_values(X_bg.iloc[:100])
            else:
                raise RuntimeError("El modelo no expone `predict_proba`; no se puede usar KernelExplainer")

    # Guardar resumen beeswarm
    out = REPORTES / "shap_summary.png"
    print("Generando beeswarm y guardando en:", out)
    try:
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print("shap_summary generado correctamente.")
    except Exception as e:
        print("Error generando la figura SHAP:", e)
        raise

if __name__ == '__main__':
    main()
