# Reporte legible de clasificación

**Fecha de generación:** 20260517_200143
**Modelo ganador:** svm
**Origen crudo:** reportes/benchmark_10000.json

## Resumen ejecutivo
- Se compararon 4 modelos supervisados con el mismo conjunto de prueba.
- El mejor desempeño global fue de **0.8322 ROC-AUC**.
- La prevalencia de clase positiva observada fue **0.1393**.
- El desbalance se aproximó a una razón de **6.18:1**.

## Tabla comparativa
| nombre_modelo | roc_auc | pr_auc | sensibilidad | especificidad | f1_clase_positiva | brier_score | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svm | 0.8322 | 0.4322 | 0.1254 | 0.9855 | 0.2065 | 0.0970 | 0.8655 |
| gbm | 0.8266 | 0.4017 | 0.1434 | 0.9727 | 0.2186 | 0.0997 | 0.8570 |
| mlp | 0.8213 | 0.3976 | 0.1792 | 0.9692 | 0.2618 | 0.1002 | 0.8590 |
| arbol | 0.7961 | 0.3442 | 0.7634 | 0.7089 | 0.4290 | 0.1892 | 0.7165 |

## Interpretación clínica
El modelo **svm** concentra el mejor balance observado: ROC-AUC 0.8322, PR-AUC 0.4322, sensibilidad 0.1254, especificidad 0.9855 y Brier Score 0.0970. En un contexto de tamizaje clínico, esto sugiere que el modelo ordena correctamente el riesgo y mantiene una calibración razonable para priorización, aunque la sensibilidad debe revisarse antes de un despliegue operativo.

## Nota operativa
Este reporte se sintetiza a partir de un JSON crudo generado por el pipeline. Los artefactos crudos se mantienen fuera del repositorio para evitar versionar salidas volátiles.
