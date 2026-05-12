from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import REPORTES_DIR
from entrenamiento.comparador_modelos import ResultadoModelo


class EvaluadorClinico:
    """Calcula métricas clínicas y genera artefactos para comparación de modelos."""

    def calcular_metricas(self, y_verdadero: np.ndarray, y_prob: np.ndarray, umbral: float = 0.5) -> dict[str, float]:
        y_pred = (y_prob >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_verdadero, y_pred, labels=[0, 1]).ravel()
        especificidad = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        metricas = {
            "accuracy": float(accuracy_score(y_verdadero, y_pred)),
            "sensibilidad": float(recall_score(y_verdadero, y_pred, pos_label=1, zero_division=0)),
            "especificidad": especificidad,
            "f1": float(f1_score(y_verdadero, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_verdadero, y_prob)),
            "pr_auc": float(average_precision_score(y_verdadero, y_prob)),
            "brier_score": float(brier_score_loss(y_verdadero, y_prob)),
        }
        metricas["tp"] = float(tp)
        metricas["tn"] = float(tn)
        metricas["fp"] = float(fp)
        metricas["fn"] = float(fn)
        return metricas

    def graficar_curvas(self, y_verdadero: np.ndarray, y_prob: np.ndarray, nombre_modelo: str) -> Path:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_verdadero, y_prob)
        precision, recall, _ = precision_recall_curve(y_verdadero, y_prob)

        REPORTES_DIR.mkdir(parents=True, exist_ok=True)
        ruta = REPORTES_DIR / f"curvas_{nombre_modelo}.png"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr, label="ROC")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[0].set_title(f"ROC - {nombre_modelo}")
        axes[0].set_xlabel("1 - Especificidad")
        axes[0].set_ylabel("Sensibilidad")
        axes[0].legend()

        axes[1].plot(recall, precision, label="PR")
        axes[1].set_title(f"Precision-Recall - {nombre_modelo}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return ruta

    def comparar_modelos(self, resultados: list[ResultadoModelo]) -> pd.DataFrame:
        filas = []
        for resultado in resultados:
            fila = {"modelo": resultado.nombre, "puntaje": resultado.puntaje}
            fila.update(resultado.metricas)
            filas.append(fila)

        tabla = pd.DataFrame(filas).sort_values(by="puntaje", ascending=False).reset_index(drop=True)
        REPORTES_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTES_DIR / "comparativa_modelos.md").write_text(tabla.to_markdown(index=False), encoding="utf-8")
        return tabla
