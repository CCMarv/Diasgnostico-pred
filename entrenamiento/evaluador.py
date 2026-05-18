from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ResultadoEvaluacion:
    nombre_modelo: str
    roc_auc: float
    pr_auc: float
    sensibilidad: float
    especificidad: float
    f1_clase_positiva: float
    brier_score: float
    accuracy: float
    matriz_confusion: np.ndarray


class EvaluadorClinico:
    """
    Propósito:
    Calcular, graficar y persistir métricas clínicas de clasificación binaria.
    """
    _FIGSIZE_CURVAS = (12, 5)
    _FIGSIZE_CALIBRACION = (6, 6)
    _DPI_GRAFICAS = 150
    _N_BINS_CALIBRACION = 10

    def calcular_metricas(
        self,
        y_verdadero: np.ndarray,
        y_prob: np.ndarray,
        nombre_modelo: str,
    ) -> ResultadoEvaluacion:
        y_true = np.asarray(y_verdadero).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, _, _ = matriz.ravel()
        especificidad = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        return ResultadoEvaluacion(
            nombre_modelo=nombre_modelo,
            roc_auc=float(roc_auc_score(y_true, y_prob)),
            pr_auc=float(average_precision_score(y_true, y_prob)),
            sensibilidad=float(recall_score(y_true, y_pred, pos_label=1)),
            especificidad=especificidad,
            f1_clase_positiva=float(f1_score(y_true, y_pred, pos_label=1)),
            brier_score=float(brier_score_loss(y_true, y_prob)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            matriz_confusion=matriz,
        )

    def graficar_curvas(
        self,
        y_verdadero: np.ndarray,
        y_prob: np.ndarray,
        nombre_modelo: str,
        ruta_salida: Path | None = None,
    ) -> None:
        ruta = ruta_salida or (REPORTES_DIR / f"curvas_{nombre_modelo}.png")
        ruta.parent.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_verdadero, y_prob)
        precision, recall, _ = precision_recall_curve(y_verdadero, y_prob)
        auc_roc = roc_auc_score(y_verdadero, y_prob)
        auc_pr = average_precision_score(y_verdadero, y_prob)

        fig, axes = plt.subplots(1, 2, figsize=self._FIGSIZE_CURVAS)

        axes[0].plot(fpr, tpr, label=f"{nombre_modelo} (AUC={auc_roc:.3f})")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Referencia")
        axes[0].set_title("Curva ROC")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].legend(loc="lower right")

        axes[1].plot(recall, precision, label=f"{nombre_modelo} (PR-AUC={auc_pr:.3f})")
        baseline = float(np.mean(y_verdadero))
        axes[1].hlines(y=baseline, xmin=0, xmax=1, linestyles="--", colors="gray", label="Referencia")
        axes[1].set_title("Curva Precision-Recall")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="lower left")

        fig.tight_layout()
        fig.savefig(ruta, dpi=self._DPI_GRAFICAS)
        plt.close(fig)

    def comparar_modelos(self, resultados: list[ResultadoEvaluacion]) -> pd.DataFrame:
        if not resultados:
            raise ValueError("Se requiere al menos un resultado para comparar modelos.")

        filas: list[dict[str, float | str]] = []
        for resultado in resultados:
            fila = {
                "nombre_modelo": resultado.nombre_modelo,
                "roc_auc": resultado.roc_auc,
                "pr_auc": resultado.pr_auc,
                "sensibilidad": resultado.sensibilidad,
                "especificidad": resultado.especificidad,
                "f1_clase_positiva": resultado.f1_clase_positiva,
                "brier_score": resultado.brier_score,
                "accuracy": resultado.accuracy,
            }
            filas.append(fila)

        tabla = pd.DataFrame(filas).sort_values("roc_auc", ascending=False).reset_index(drop=True)
        return tabla

    def graficar_curva_calibracion(
        self,
        y_verdadero: np.ndarray,
        y_prob: np.ndarray,
        nombre_modelo: str,
    ) -> None:
        prob_pred, prob_true = calibration_curve(
            y_verdadero,
            y_prob,
            n_bins=self._N_BINS_CALIBRACION,
            strategy="uniform",
        )
        ruta = REPORTES_DIR / f"calibracion_{nombre_modelo}.png"
        ruta.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=self._FIGSIZE_CALIBRACION)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
        ax.plot(prob_pred, prob_true, marker="o", label=nombre_modelo)
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Fracción de positivos")
        ax.set_title(f"Curva de calibración - {nombre_modelo}")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(ruta, dpi=self._DPI_GRAFICAS)
        plt.close(fig)

    @staticmethod
    def serializar_resultado(resultado: ResultadoEvaluacion) -> dict[str, float | list[list[int]] | str]:
        data = asdict(resultado)
        data["matriz_confusion"] = resultado.matriz_confusion.astype(int).tolist()
        return data

    @staticmethod
    def interpretar_resultado(resultado: ResultadoEvaluacion) -> str:
        lineas = [
            f"=== Interpretación del modelo: {resultado.nombre_modelo} ===",
            "",
            f"ROC-AUC: {resultado.roc_auc:.4f}",
        ]
        if resultado.roc_auc >= 0.80:
            lineas.append("  → Excelente capacidad para separar pacientes con y sin diabetes.")
        elif resultado.roc_auc >= 0.75:
            lineas.append("  → Buena capacidad discriminativa. Cumple el umbral mínimo del proyecto.")
        else:
            lineas.append("  → Por debajo del umbral aceptable (0.75). Revisar hiperparámetros o preprocesamiento.")

        lineas += [
            "",
            f"Sensibilidad: {resultado.sensibilidad:.4f}",
            f"  → De cada 100 pacientes con diabetes, el modelo detecta "
            f"aprox. {int(resultado.sensibilidad * 100)}.",
        ]
        if resultado.sensibilidad < 0.70:
            lineas.append("  ⚠️ Sensibilidad baja: muchos casos de diabetes no son detectados.")

        lineas += [
            "",
            f"Especificidad: {resultado.especificidad:.4f}",
            f"  → De cada 100 pacientes sanos, el modelo clasifica correctamente "
            f"aprox. {int(resultado.especificidad * 100)}.",
            "",
            f"Brier Score: {resultado.brier_score:.4f}",
        ]
        if resultado.brier_score < 0.10:
            lineas.append("  → Calibración excelente: las probabilidades predichas son confiables.")
        elif resultado.brier_score < 0.15:
            lineas.append("  → Calibración aceptable.")
        else:
            lineas.append("  ⚠️ Calibración deficiente: las probabilidades predichas no son confiables.")

        return "\n".join(lineas)

    @staticmethod
    def _a_markdown(tabla: pd.DataFrame) -> str:
        encabezados = list(tabla.columns)
        lineas = [
            "| " + " | ".join(encabezados) + " |",
            "| " + " | ".join(["---"] * len(encabezados)) + " |",
        ]
        for _, fila in tabla.iterrows():
            valores = []
            for columna in encabezados:
                valor = fila[columna]
                if isinstance(valor, float):
                    valores.append(f"{valor:.4f}")
                else:
                    valores.append(str(valor))
            lineas.append("| " + " | ".join(valores) + " |")
        return "\n".join(lineas) + "\n"
