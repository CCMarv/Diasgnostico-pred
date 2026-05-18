# Evaluación Académica del Proyecto Final
<!-- Documento único de control de proyecto — reemplaza PLAN_CORRECIONES.md, ROADMAP.md y copilot-instructions.md -->

**Última actualización:** 2026-05-18
**Nivel estimado actual:** Básico completado → Intermedio parcial (sin dashboard)
**Puntos extra acumulados:** 0 (Dashboard I6 pendiente para desbloquear +15)

---

## 1. Calificación por componente de la rúbrica

> La calificación final del proyecto se compone de cuatro componentes.
> El agente evalúa cada uno de forma independiente antes de calcular el total.

| Componente | Peso | Calificación estimada (0-100) | Ponderado | Observaciones |
|------------|------|-------------------------------|-----------|---------------|
| **Código y Técnica** | 40% | 78 | 31.2 | Pipeline completo, 4 modelos, KNNImputer + SMOTE activos, 27 tests pasando, type hints en todos los módulos; falta dashboard |
| **Resultados** | 30% | 68 | 20.4 | 8 métricas por modelo (ROC-AUC, PR-AUC, F1, sensibilidad, especificidad, Brier, accuracy, matriz); falta interpretación cualitativa y muestra académica 2000 registros |
| **Reporte** | 20% | 35 | 7.0 | Solo reportes automáticos generados por el pipeline; reportes/reporte_final.md pendiente Sprint 5 |
| **Presentación** | 10% | 30 | 3.0 | Documentación técnica existe; falta README_demo.md, preguntas_defensa.md y demo funcional |
| **TOTAL BASE** | 100% | — | **61.6** | |
| **Puntos extra (nivel)** | — | — | **0** | Nivel Intermedio desbloqueado con Dashboard (Sprint 4) |
| **CALIFICACIÓN FINAL** | — | — | **61.6** | Proyección tras Sprint 4: ~76.6; tras Sprint 6: ~106.6 |

---

## 2. Estado por nivel de la rúbrica

### Nivel Básico — Requisito mínimo (sin puntos extra, base del 100%)

> Completar este nivel es condición para aprobar el proyecto. No otorga puntos extra.

| Requisito | Evidencia en código | Archivo | Estado |
|-----------|---------------------|---------|--------|
| Pipeline completo con al menos 3 modelos | Pipeline con SVM, Árbol, GBM y MLP evaluados por `comparador_modelos.py` | entrenamiento/comparador_modelos.py | ✅ |
| Preprocessing básico (imputación, escalado, encoding) | Preprocesamiento centralizado en `ConstructorPreprocesador` usado por el pipeline | entrenamiento/preprocesador.py | ✅ |
| Métricas de evaluación estándar (accuracy, precision, recall, F1) | Métricas calculadas y reportadas por `entrenamiento/evaluador.py` | entrenamiento/evaluador.py | ✅ |

**Veredicto Básico:** COMPLETADO — no hay bloqueantes críticos

---

### Nivel Intermedio — +15 puntos extra

> Requiere Básico completado. Todos los ítems deben estar presentes.

| Requisito | Evidencia en código | Archivo | Estado |
|-----------|---------------------|---------|--------|
| SVM implementado y evaluado | `SVC(kernel='rbf', probability=True, class_weight='balanced')` en catálogo `_catalogo_modelos["svm"]`; evaluado con ROC-AUC, PR-AUC, F1, sensibilidad, especificidad | entrenamiento/comparador_modelos.py | ✅ |
| Árboles de decisión implementados y evaluados | `DecisionTreeClassifier(max_depth=5, class_weight='balanced')` en catálogo `_catalogo_modelos["arbol"]`; evaluado con las mismas 8 métricas | entrenamiento/comparador_modelos.py | ✅ |
| Redes neuronales (`MLPClassifier`) implementadas y evaluadas | `MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True)` en catálogo `_catalogo_modelos["mlp"]`; evaluado con las mismas 8 métricas | entrenamiento/comparador_modelos.py | ✅ |
| K-Means implementado y evaluado | Implementado como `FenotipadoKMeans` con `silhouette_score` > 0 en datos sintéticos; pruebas en `pruebas/test_fenotipado.py` (3 passed) | entrenamiento/fenotipado.py | ✅ |
| Optimización de hiperparámetros (`GridSearchCV` o `RandomizedSearchCV`) | Implementado `OptimizadorHiperparametros` (GridSearchCV + StratifiedKFold); pruebas en `pruebas/test_optimizador.py` (2 passed) | entrenamiento/optimizador.py | ✅ |
| Dashboard interactivo básico (Streamlit u otro) | No existe todavía; pendiente Sprint 4 (S4-001 a S4-006) | dashboard/app.py | ❌ pendiente S4 |

**Veredicto Intermedio:** INCOMPLETO — falta Dashboard (I6); todos los demás ítems verificados

---

### Nivel Avanzado — +30 puntos extra

> Requiere Intermedio completado. Los dos ítems son independientes pero ambos necesarios.

| Requisito | Evidencia en código | Archivo | Estado |
|-----------|---------------------|---------|--------|
| Sistema en producción — API funcional (FastAPI u otro) | `api/main.py` con endpoints `/salud` y `/predecir` funcionales; estado operativo/degradado consistente; 4 tests de integración pasando | api/main.py | 🟡 Código completo — falta serializar modelo: `modelos/modelo_diabetes_v1.joblib` no existe; ejecutar pipeline para generarlo (S6-01) |
| Comparativa con papers académicos documentada en reporte | No existe todavía; pendiente Sprint 6 (S6-005, S6-006) | reportes/reporte_final.md | ❌ pendiente S6 |

**Veredicto Avanzado:** INCOMPLETO — faltantes: conexión API con modelo real (S6-001) y comparativa papers (S6-005/006)

---

## 3. Criterios detallados por componente

### 3.1 Código y Técnica (40%)

> Evalúa calidad del código, uso apropiado de técnicas y documentación.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| El pipeline usa `sklearn.pipeline.Pipeline` o `imblearn.pipeline.Pipeline` | Alto | ✅ | `ImbPipeline` activo cuando `use_smote=True`; `Pipeline` estándar como fallback; ambos serializables con joblib |
| No hay fuga de datos (fit solo en train, no en test) | Alto | ✅ | `ColumnTransformer` ajustado exclusivamente en `X_train`; verificado por `test_preprocesador.py::test_pipeline_no_filtra_estadisticas_de_test` |
| Las técnicas del temario se usan en el contexto correcto | Alto | ✅ | `KNNImputer` y `SMOTE` activos por defecto en el flujo estándar desde Issue #15; `GridSearchCV+StratifiedKFold` en `optimizador.py` y búsqueda de SVM |
| El código tiene type hints y sigue PEP-8 | Medio | ✅ | `from __future__ import annotations` en todos los módulos; tipado estricto en firmas públicas |
| Las funciones y clases tienen docstrings | Medio | ✅ | Docstrings en español en todas las clases y métodos públicos de `entrenamiento/` y `api/` |
| Los módulos están organizados con responsabilidad única | Medio | ✅ | `cargador_datos`, `preprocesador`, `comparador_modelos`, `evaluador`, `fenotipado`, `optimizador`, `predictor` — cada módulo con una única responsabilidad |
| Existe `requirements.txt` o `pyproject.toml` con dependencias | Bajo | ✅ | `pyproject.toml` con todas las dependencias incluida `imbalanced-learn>=0.12.0` (añadida Issue #13) |
| Las pruebas unitarias cubren al menos el pipeline principal | Bajo | 🟡 | 27 tests en 8 archivos (`test_api` 4, `test_cargador` 5, `test_comparador` 4, `test_fenotipado` 3, `test_optimizador` 2, `test_predictor` 2, `test_preprocesador` 4, `test_descargador_dataset` 3); errores de colección detectados en entorno limpio — requiere `pip install -e .[dev]` antes de presentación |

**Puntos fuertes:**
- Separación clara de responsabilidades; cada módulo puede probarse de forma aislada
- Contrato de no-leakage verificado por test automatizado (no solo por diseño)
- Todos los módulos usan type hints completos

**Áreas de mejora:**
- Dashboard (I6) aún pendiente — bloquea +15 puntos extra
- Los docstrings no siguen uniformemente el estilo NumPy/Google; algunos son descriptivos pero no tienen sección `Args`/`Returns` explícita

---

### 3.2 Resultados (30%)

> Evalúa performance del modelo, análisis de resultados e interpretación.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| Se reportan métricas más allá de accuracy (PR-AUC, ROC-AUC, F1) | Alto | ✅ | `evaluador.py` calcula: ROC-AUC, PR-AUC, F1 clase positiva, sensibilidad, especificidad, Brier Score, accuracy y matriz de confusión |
| Se comparan al menos 3 modelos con las mismas métricas | Alto | ✅ | 4 modelos (SVM, Árbol, GBM, MLP) evaluados con idénticas 8 métricas; verificado por `test_comparador.py` (4 casos parametrizados) |
| Se interpreta por qué un modelo supera a otro (no solo números) | Alto | ❌ | El log reporta ROC-AUC por modelo pero no existe texto en el reporte final explicando las diferencias clínicas; pendiente S5-003 |
| La elección del modelo final está justificada con métricas | Medio | ✅ | Política explícita: serializar el modelo con mayor ROC-AUC; desde Issue #16 también se reporta mejor modelo por PR-AUC y se logguea si difieren |
| Se analiza el impacto del desbalance de clases si aplica | Medio | ✅ | `CargadorDatos.detectar_desbalance` cuantifica el ratio; SMOTE activo en entrenamiento; PR-AUC como métrica principal para clases desbalanceadas |
| Se reportan intervalos de confianza o desviación estándar entre pliegues | Bajo | 🟡 | ROC-AUC por fold disponible en logs; no se calcula `std` explícitamente en el JSON de reporte |

**Benchmark disponibles:** `reportes/benchmark_5000.json`, `benchmark_10000.json`, `benchmark_50000.json` (superan el requisito de 2000 registros). El requisito S3-009 se considera cubierto; pendiente seleccionar muestra de 2000 estratificada como muestra canónica para el reporte final.

**Puntos fuertes:**
- 8 métricas calculadas automáticamente para cada modelo
- PR-AUC reportada junto a ROC-AUC (sensible a desbalance)

**Áreas de mejora:**
- Falta interpretación narrativa de resultados (prevista en S5-003)
- Desviación estándar entre folds no está en el JSON de reporte

---

### 3.3 Reporte (20%)

> Evalúa claridad, profundidad y presentación de hallazgos.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| El reporte tiene estructura clara (intro, metodología, resultados, conclusión) | Alto | ❌ | `reportes/reporte_final.md` no existe todavía; pendiente Sprint 5 (S5-001 a S5-004) |
| Los hallazgos principales se presentan con visualizaciones | Alto | 🟡 | Curvas ROC, PR y calibración generadas automáticamente en `reportes/`; aún no integradas en un reporte narrativo |
| La metodología explica por qué se eligió cada técnica del temario | Medio | ❌ | Pendiente S5-002 |
| Las conclusiones responden la pregunta del proyecto | Medio | ❌ | Pendiente S5-004 |
| El lenguaje es técnicamente preciso y consistente | Bajo | ❌ | Pendiente revisión al redactar S5-001 a S5-004 |

**Ruta del reporte principal:** `reportes/reporte_final.md` (pendiente Sprint 5)

**Puntos fuertes:**
- Reportes automáticos en `reportes/metricas_sprint1.json` y `.md` generados por el pipeline
- Visualizaciones de curvas ROC, PR y calibración disponibles

**Áreas de mejora:**
- Falta el reporte académico narrativo completo (el más importante para este componente)

---

### 3.4 Presentación (10%)

> Evalúa comunicación, manejo de preguntas y demostración.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| Hay una demo funcional (dashboard, notebook ejecutado o API) | Alto | ❌ | Dashboard no existe; API funciona pero no está documentada para demo; pendiente Sprint 4 y S5-005 |
| El equipo puede explicar las decisiones técnicas con claridad | Alto | ✅ | Documentación técnica detallada en `docs/implementacion_modelos/` con justificación de cada modelo |
| Se preparan respuestas para preguntas típicas del temario | Medio | ❌ | `docs/preguntas_defensa.md` no existe; pendiente S5-006 |

**Artefacto de demo:** pendiente — `streamlit run dashboard/app.py` (Sprint 4) y `uvicorn api.main:app` (Sprint 6)

**Puntos fuertes:**
- Documentación técnica detallada por modelo disponible para consulta en defensa

**Áreas de mejora:**
- Crear `README_demo.md` con comandos copiables (S5-005)
- Preparar preguntas de defensa (S5-006)

---

## 4. Cobertura del temario — Mapeo técnico [AUTO]

> El agente actualiza esta sección automáticamente al auditar el workspace.

### Unidad 1 — Pipeline y Preprocesamiento

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| `Pipeline` | `sklearn.pipeline.Pipeline` | `entrenamiento/preprocesador.py` | ✅ |
| `ColumnTransformer` | `sklearn.compose.ColumnTransformer` | `entrenamiento/preprocesador.py` | ✅ |
| Imputación multivariable | `KNNImputer` | `entrenamiento/preprocesador.py` | ✅ activo por defecto desde Issue #15 |
| Codificación robusta | `TargetEncoder` o `HashingEncoder` | `entrenamiento/preprocesador.py` | Fuera de alcance — no listado en ProgramaMateria.md §Niveles B/I/A |
| Transformación de potencia | `PowerTransformer(method='yeo-johnson')` | `entrenamiento/preprocesador.py` | Fuera de alcance — no listado en ProgramaMateria.md §Niveles B/I/A |
| Gestión del desbalance | `SMOTE` o `NearMiss` | `entrenamiento/preprocesador.py` | ✅ `SMOTE` activo por defecto desde Issue #15; `imbalanced-learn>=0.12.0` declarado en `pyproject.toml` |
| Métricas de desbalance | PR-AUC, ROC-AUC | `entrenamiento/evaluador.py` | ✅ |

### Unidad 2 — Aprendizaje Supervisado

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| SVM | `SVC(kernel='rbf')` | `entrenamiento/comparador_modelos.py` | ✅ |
| Árboles de decisión | `DecisionTreeClassifier` | `entrenamiento/comparador_modelos.py` | ✅ |
| Random Forest | `RandomForestClassifier` | `entrenamiento/comparador_modelos.py` | Fuera de alcance — explícitamente excluido en `docs/PLAN_CORRECIONES.md` (no figura en lista del Nivel Intermedio) |
| Gradient Boosting | `GradientBoostingClassifier` o `XGBClassifier` | `entrenamiento/comparador_modelos.py` | ✅ `GradientBoostingClassifier` (Unidad 2.3 del temario) |

### Unidad 3 — No Supervisado

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| K-Means | `KMeans(init='k-means++')` | `entrenamiento/fenotipado.py` | ✅ |
| Validación interna | Método del codo + `silhouette_score` | `entrenamiento/fenotipado.py` | ✅ |

### Unidad 4 — Optimización y Redes Neuronales

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| Búsqueda de hiperparámetros | `GridSearchCV` o `RandomizedSearchCV` | `entrenamiento/optimizador.py` | ✅ `GridSearchCV` en `OptimizadorHiperparametros` y búsqueda manual de SVM en `comparador_modelos.py` |
| Validación cruzada estratificada | `StratifiedKFold` | `entrenamiento/optimizador.py` | ✅ en `optimizador.py` y `comparador_modelos.py` |
| Red neuronal | `MLPClassifier` | `entrenamiento/comparador_modelos.py` | ✅ `MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True)` |

---

## 5. Registro de cambios [AUTO]

> El agente añade una entrada cada vez que actualiza este documento.

| Fecha | Cambio | Nivel anterior → nuevo | Responsable |
|-------|--------|------------------------|-------------|
| 2026-05-17 | Implementado `FenotipadoKMeans` con pruebas unitarias; implementado `OptimizadorHiperparametros` con pruebas; ejecución de la muestra mínima de estabilidad y pruebas pasadas. Suite de pruebas `pruebas/` ejecutada: 20 passed. | Básico → Básico (hacia Intermedio) | AcademicoIA |
| 2026-05-18 | Completada auditoría de evidencia: issues #9–#43 cerrados. Suite: 27 tests en 8 archivos. Reemplazados todos los placeholders. | Básico → Básico completado (Intermedio parcial sin dashboard) | AcademicoIA |
| 2026-05-18 | Unificación documental: eliminados `PLAN_CORRECIONES.md`, `ROADMAP.md`, `docs/copilot-instructions.md`, `.github/copilot-instructions.md`. Plan de sprints S3–S6 absorbido en Sección 6. Este documento es ahora la única referencia de control de proyecto. Detectados errores de colección en pytest (entorno limpio sin `pip install -e .[dev]`). Benchmarks 5k/10k/50k confirman cumplimiento del requisito de 2000 registros. | Sin cambio de nivel | CCMarv |
| 2026-05-18 | Sprint 3 completado: `notebooks/02_fenotipado_kmeans.ipynb` creado y ejecutado (k=2, silhouette=0.590, χ²=1209, p<0.001); `reportes/contraste_regional.md` generado; errores de pytest resueltos con `pip install -e .[dev]` + `Makefile`; suite: 29 passed. | Básico completado → Intermedio parcial (sin dashboard) | CCMarv |

---

## 6. Plan de acción por sprint

> Este documento reemplaza `ROADMAP.md` y `PLAN_CORRECIONES.md` (eliminados 2026-05-18).
> Secuencia obligatoria: S3 → S4 → S5 → S6. No iniciar un sprint si el anterior tiene ítems de nivel incompletos.
>
> **Calificación proyectada:** Sprint 4 completo → ~76.6 base | Sprint 6 completo → ~106.6

---

### Bloqueantes para Nivel Básico
- Ninguno — Nivel Básico COMPLETADO ✅

---

### Sprint 3 — Cierre de K-Means y pendientes de S2 ✅ COMPLETADO 2026-05-18
**Impacto:** completa I4, I5, enriquece Resultados 30% y Reporte 20%

| ID | Tarea | Archivo | Estado |
|----|-------|---------|--------|
| S3-01 | Contrato uniforme de métricas en 4 modelos | `pruebas/test_comparador.py` | ✅ 2026-05-18 |
| S3-02 | K-Means con `k-means++`, codo, silhouette | `entrenamiento/fenotipado.py` | ✅ |
| S3-03 | Notebook de análisis de fenotipos K-Means | `notebooks/02_fenotipado_kmeans.ipynb` | ✅ 2026-05-18 — k=2, silhouette=0.590, χ²=1209 (p<0.001) |
| S3-04 | `GridSearchCV` + `StratifiedKFold` como módulo formal | `entrenamiento/optimizador.py` | ✅ |
| S3-05 | Pruebas del fenotipador | `pruebas/test_fenotipado.py` | ✅ |
| S3-06 | Corregir errores de colección en pytest | `Makefile` + `pip install -e .[dev]` | ✅ 2026-05-18 — 29 passed |
| S3-07 | Parquet procesado sin columna objetivo | `datos/procesados/dataset_procesado.parquet` | ✅ |
| S3-08 | Contraste regional documentado en reporte | `reportes/contraste_regional.md` | ✅ 2026-05-18 — generado por `scripts/generar_contraste_regional.py` |

**Validación ejecutada:**
```
pytest pruebas/ -q → 29 passed, 0 errors
jupyter nbconvert --execute notebooks/02_fenotipado_kmeans.ipynb → OK (sin errores)
reportes/hallazgos_fenotipado.json → generado (k=2, silhouette=0.5895)
reportes/contraste_regional.md → generado
```

---

### Sprint 4 — Dashboard interactivo *(bloqueante de +15 pts)*
**Impacto:** desbloquea Nivel Intermedio (+15), mejora Presentación 10% y Código y Técnica

| ID | Tarea | Archivo | Estado |
|----|-------|---------|--------|
| S4-01 | Añadir `streamlit` a `pyproject.toml` | `pyproject.toml` | ⬜ |
| S4-02 | Vista: tabla comparativa de modelos interactiva | `dashboard/app.py` | ⬜ |
| S4-03 | Vista: formulario de predicción individual | `dashboard/app.py` | ⬜ |
| S4-04 | Vista: gráfica de clústeres K-Means | `dashboard/app.py` | ⬜ |
| S4-05 | Dashboard carga el pipeline serializado (`modelo_diabetes_v1.joblib`) | `dashboard/app.py` | ⬜ |

**Validación de cierre:**
```bash
streamlit run dashboard/app.py
# Verificar: tabla de métricas carga, formulario devuelve resultado, gráfica K-Means se renderiza
```

**Veredicto Intermedio al cerrar S4:** I1 ✅ I2 ✅ I3 ✅ I4 ✅ I5 ✅ I6 ✅ → **+15 puntos extra**

---

### Sprint 5 — Reporte narrativo y artefactos de presentación
**Impacto:** Reporte 20% (de ~35 → ~85) + Presentación 10% (de ~30 → ~80)

| ID | Tarea | Archivo | Estado |
|----|-------|---------|--------|
| S5-01 | Reporte: introducción y planteamiento del problema | `reportes/reporte_final.md` | ⬜ |
| S5-02 | Reporte: metodología (pipeline, modelos, decisiones técnicas) | `reportes/reporte_final.md` | ⬜ |
| S5-03 | Reporte: resultados con curvas ROC, PR, matriz de confusión | `reportes/reporte_final.md` | ⬜ |
| S5-04 | Reporte: conclusiones y limitaciones | `reportes/reporte_final.md` | ⬜ |
| S5-05 | Guía de ejecución del demo | `README_demo.md` | ⬜ |
| S5-06 | Preguntas y respuestas de defensa oral | `docs/preguntas_defensa.md` | ⬜ |
| S5-07 | Verificación final: pipeline + dashboard + pruebas sin errores | — | ⬜ |

> Nota operativa: el reporte se genera ejecutando el pipeline para obtener el JSON crudo y luego `scripts/generar_reporte_legible.py` para el Markdown. No editar `reporte_final.md` a mano.

**Calificación estimada al cerrar S5:** Reporte 35→85 (+10 pts ponderados) + Presentación 30→80 (+5 pts) → **base ~76.6**

---

### Sprint 6 — API en producción y comparativa con papers *(bloqueante de +30 pts)*
**Impacto:** desbloquea Nivel Avanzado (+30), completa A1 y A2

| ID | Tarea | Archivo | Estado |
|----|-------|---------|--------|
| S6-01 | Serializar modelo entrenando pipeline completo | `modelos/modelo_diabetes_v1.joblib` | ⬜ |
| S6-02 | Verificar que `POST /predecir` responde con modelo real cargado | `api/main.py` | ⬜ |
| S6-03 | Documentar arranque de la API en `README_demo.md` | `README_demo.md` | ⬜ |
| S6-04 | Identificar ≥ 2 papers de predicción de diabetes con ML | — | ⬜ |
| S6-05 | Tabla comparativa métricas proyecto vs. papers en reporte | `reportes/reporte_final.md` | ⬜ |

**Validación de cierre:**
```bash
uvicorn api.main:app --reload
curl http://localhost:8000/salud   # esperado: {"estado": "operativo"}
pytest pruebas/test_api.py -v      # 4 tests pasando
```

**Veredicto Avanzado al cerrar S6:** A1 ✅ A2 ✅ → **+30 puntos extra** → calificación proyectada **~106.6**

---

## 7. Preguntas frecuentes de presentación

> El agente genera esta sección para preparar al equipo para la defensa oral.

1. **¿Por qué usaron `Pipeline` en lugar de transformar directamente?**
   → Porque el `Pipeline` empaqueta preprocesamiento + clasificador en un solo artefacto serializable. Así, cuando la API carga `modelo_diabetes_v1.joblib`, el modelo ya "sabe" cómo transformar los datos de entrada sin pasos manuales adicionales. Evidencia: `entrenamiento/preprocesador.py:construir_pipeline` y `inferencia/predictor.py:predecir`.

2. **¿Cómo manejaron el desbalance de clases?**
   → Con dos estrategias complementarias: (1) `SMOTE` dentro del `ImbPipeline` — genera muestras sintéticas de la clase minoritaria solo sobre datos de entrenamiento, nunca sobre test, garantizando que no hay data leakage; (2) PR-AUC como métrica principal, que es más informativa que ROC-AUC cuando hay desbalance (~86% sano vs ~14% diabético). Evidencia: `entrenamiento/preprocesador.py:construir_pipeline` y `entrenamiento/evaluador.py:calcular_metricas`.

3. **¿Por qué eligieron GBM como el modelo principal?**
   → En la muestra actual, GBM obtiene el mayor ROC-AUC. La política del proyecto es serializar el modelo con mayor ROC-AUC (criterio clínico: maximizar la capacidad de discriminar entre sanos y diabéticos). Desde Issue #16, el pipeline también reporta el mejor modelo por PR-AUC y advierte si difieren. Evidencia: `entrenamiento/pipeline.py:_ejecutar_flujo_clasificacion` y `reportes/metricas_sprint1.json`.

4. **¿Qué es el coeficiente de silueta y cómo lo usaron en K-Means?**
   → El coeficiente de silueta mide qué tan bien separados están los clústeres: valores cerca de 1 indican clústeres compactos y bien separados; cerca de -1 indica solapamiento. En `FenotipadoKMeans.calcular_silhouette` se calcula para el conjunto de datos de entrenamiento con el K óptimo seleccionado por el método del codo (`graficar_codo`). Se usa para validar que los fenotipos de pacientes identificados son estadísticamente coherentes. Evidencia: `entrenamiento/fenotipado.py` y `pruebas/test_fenotipado.py`.

5. **¿Cómo evitaron la fuga de datos (data leakage)?**
   → El `ColumnTransformer` (con `KNNImputer`, `StandardScaler` y `OrdinalEncoder`) se ajusta (`fit`) exclusivamente sobre `X_train`. En validación cruzada, cada fold usa `clone(pipeline)` para crear una copia limpia antes del `fit`, evitando que los estadísticos de un fold contaminen a otro. Hay un test automatizado que lo verifica: `pruebas/test_preprocesador.py::test_pipeline_no_filtra_estadisticas_de_test`. Evidencia: `entrenamiento/preprocesador.py:construir` y `entrenamiento/comparador_modelos.py:_evaluar_un_fold`.
