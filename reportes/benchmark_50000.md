# Reporte legible de clasificación

**Fecha de generación:** 20260517_201736
**Modelo ganador:** gbm
**Origen crudo:** reportes/benchmark_50000.json

## Resumen ejecutivo
- Se compararon 4 modelos supervisados con el mismo conjunto de prueba.
- El mejor desempeño global fue de **0.8391 ROC-AUC**.
- La prevalencia de clase positiva observada fue **0.1393**.
- El desbalance se aproximó a una razón de **6.18:1**.

## Tabla comparativa
| nombre_modelo | roc_auc | pr_auc | sensibilidad | especificidad | f1_clase_positiva | brier_score | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gbm | 0.8391 | 0.4397 | 0.1615 | 0.9822 | 0.2541 | 0.0953 | 0.8679 |
| svm | 0.8365 | 0.4321 | 0.0768 | 0.9920 | 0.1364 | 0.0962 | 0.8645 |
| mlp | 0.8328 | 0.4303 | 0.1572 | 0.9807 | 0.2463 | 0.0965 | 0.8660 |
| arbol | 0.8145 | 0.3871 | 0.8062 | 0.6683 | 0.4182 | 0.1834 | 0.6875 |

## Interpretación clínica
El modelo **gbm** concentra el mejor balance observado: ROC-AUC 0.8391, PR-AUC 0.4397, sensibilidad 0.1615, especificidad 0.9822 y Brier Score 0.0953. En un contexto de tamizaje clínico, esto sugiere que el modelo ordena correctamente el riesgo y mantiene una calibración razonable para priorización, aunque la sensibilidad debe revisarse antes de un despliegue operativo.

## Nota operativa
Este reporte se sintetiza a partir de un JSON crudo generado por el pipeline. Los artefactos crudos se mantienen fuera del repositorio para evitar versionar salidas volátiles.
