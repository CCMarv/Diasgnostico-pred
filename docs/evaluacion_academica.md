# Evaluación Académica del Proyecto Final
<!-- Plantilla gestionada por AcademicoIA v3.1 — no editar manualmente secciones marcadas con [AUTO] -->

**Última actualización:** 2026-05-17
**Nivel estimado actual:** Básico (hacia Intermedio)
**Puntos extra acumulados:** 0

---

## 1. Calificación por componente de la rúbrica

> La calificación final del proyecto se compone de cuatro componentes.
> El agente evalúa cada uno de forma independiente antes de calcular el total.

| Componente | Peso | Calificación estimada (0-100) | Ponderado | Observaciones |
|------------|------|-------------------------------|-----------|---------------|
| **Código y Técnica** | 40% | [0-100] | [peso × cal] | [breve justificación] |
| **Resultados** | 30% | [0-100] | [peso × cal] | [breve justificación] |
| **Reporte** | 20% | [0-100] | [peso × cal] | [breve justificación] |
| **Presentación** | 10% | [0-100] | [peso × cal] | [breve justificación] |
| **TOTAL BASE** | 100% | — | **[suma ponderados]** | |
| **Puntos extra (nivel)** | — | — | **[0 / +15 / +30]** | |
| **CALIFICACIÓN FINAL** | — | — | **[total + extra]** | |

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
| SVM implementado y evaluado | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |
| Árboles de decisión implementados y evaluados | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |
| Redes neuronales (`MLPClassifier`) implementadas y evaluadas | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |
| K-Means implementado y evaluado | Implementado como `FenotipadoKMeans` con `silhouette_score` > 0 en datos sintéticos | entrenamiento/fenotipado.py | ✅ |
| Optimización de hiperparámetros (`GridSearchCV` o `RandomizedSearchCV`) | Implementado `OptimizadorHiperparametros` (GridSearchCV + StratifiedKFold) | entrenamiento/optimizador.py | ✅ |
| Dashboard interactivo básico (Streamlit u otro) | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |

**Veredicto Intermedio:** INCOMPLETO — faltan Dashboard (I6) y muestra académica (S3-009)

---

### Nivel Avanzado — +30 puntos extra

> Requiere Intermedio completado. Los dos ítems son independientes pero ambos necesarios.

| Requisito | Evidencia en código | Archivo | Estado |
|-----------|---------------------|---------|--------|
| Sistema en producción — API funcional (FastAPI u otro) | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |
| Comparativa con papers académicos documentada en reporte | [descripción] | [ruta] | ✅ / ❌ / `EVIDENCE_NOT_FOUND` |

**Veredicto Avanzado:** [COMPLETADO (+30) / INCOMPLETO — faltantes: X, Y]

---

## 3. Criterios detallados por componente

### 3.1 Código y Técnica (40%)

> Evalúa calidad del código, uso apropiado de técnicas y documentación.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| El pipeline usa `sklearn.pipeline.Pipeline` o `imblearn.pipeline.Pipeline` | Alto | [✅/❌] | [notas] |
| No hay fuga de datos (fit solo en train, no en test) | Alto | [✅/❌] | [notas] |
| Las técnicas del temario se usan en el contexto correcto | Alto | [✅/❌] | [notas] |
| El código tiene type hints y sigue PEP-8 | Medio | [✅/❌] | [notas] |
| Las funciones y clases tienen docstrings | Medio | [✅/❌] | [notas] |
| Los módulos están organizados con responsabilidad única | Medio | [✅/❌] | [notas] |
| Existe `requirements.txt` o `pyproject.toml` con dependencias | Bajo | [✅/❌] | [notas] |
| Las pruebas unitarias cubren al menos el pipeline principal | Bajo | [✅/❌] | [notas] |

**Puntos fuertes:** [lista]
**Áreas de mejora:** [lista]

---

### 3.2 Resultados (30%)

> Evalúa performance del modelo, análisis de resultados e interpretación.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| Se reportan métricas más allá de accuracy (PR-AUC, ROC-AUC, F1) | Alto | [✅/❌] | [notas] |
| Se comparan al menos 3 modelos con las mismas métricas | Alto | [✅/❌] | [notas] |
| Se interpreta por qué un modelo supera a otro (no solo números) | Alto | [✅/❌] | [notas] |
| La elección del modelo final está justificada con métricas | Medio | [✅/❌] | [notas] |
| Se analiza el impacto del desbalance de clases si aplica | Medio | [✅/❌] | [notas] |
| Se reportan intervalos de confianza o desviación estándar entre pliegues | Bajo | [✅/❌] | [notas] |

**Mejor modelo actual:** [nombre] con [métrica principal] = [valor]
**Puntos fuertes:** [lista]
**Áreas de mejora:** [lista]

---

### 3.3 Reporte (20%)

> Evalúa claridad, profundidad y presentación de hallazgos.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| El reporte tiene estructura clara (intro, metodología, resultados, conclusión) | Alto | [✅/❌] | [notas] |
| Los hallazgos principales se presentan con visualizaciones | Alto | [✅/❌] | [notas] |
| La metodología explica por qué se eligió cada técnica del temario | Medio | [✅/❌] | [notas] |
| Las conclusiones responden la pregunta del proyecto | Medio | [✅/❌] | [notas] |
| El lenguaje es técnicamente preciso y consistente | Bajo | [✅/❌] | [notas] |

**Ruta del reporte principal:** [ruta/reporte.md o .pdf]
**Puntos fuertes:** [lista]
**Áreas de mejora:** [lista]

---

### 3.4 Presentación (10%)

> Evalúa comunicación, manejo de preguntas y demostración.

| Criterio | Peso dentro del componente | Estado | Notas |
|----------|---------------------------|--------|-------|
| Hay una demo funcional (dashboard, notebook ejecutado o API) | Alto | [✅/❌] | [notas] |
| El equipo puede explicar las decisiones técnicas con claridad | Alto | [✅/❌] | [notas] |
| Se preparan respuestas para preguntas típicas del temario | Medio | [✅/❌] | [notas] |

**Artefacto de demo:** [ruta o comando para ejecutar]
**Puntos fuertes:** [lista]
**Áreas de mejora:** [lista]

---

## 4. Cobertura del temario — Mapeo técnico [AUTO]

> El agente actualiza esta sección automáticamente al auditar el workspace.

### Unidad 1 — Pipeline y Preprocesamiento

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| `Pipeline` | `sklearn.pipeline.Pipeline` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| `ColumnTransformer` | `sklearn.compose.ColumnTransformer` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Imputación multivariable | `KNNImputer` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Codificación robusta | `TargetEncoder` o `HashingEncoder` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Transformación de potencia | `PowerTransformer(method='yeo-johnson')` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Gestión del desbalance | `SMOTE` o `NearMiss` | `entrenamiento/preprocesador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Métricas de desbalance | PR-AUC, ROC-AUC | `entrenamiento/evaluador.py` | [✅/`EVIDENCE_NOT_FOUND`] |

### Unidad 2 — Aprendizaje Supervisado

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| SVM | `SVC(kernel='rbf')` | `entrenamiento/comparador_modelos.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Árboles de decisión | `DecisionTreeClassifier` | `entrenamiento/comparador_modelos.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Random Forest | `RandomForestClassifier` | `entrenamiento/comparador_modelos.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Gradient Boosting | `GradientBoostingClassifier` o `XGBClassifier` | `entrenamiento/comparador_modelos.py` | [✅/`EVIDENCE_NOT_FOUND`] |

### Unidad 3 — No Supervisado

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| K-Means | `KMeans(init='k-means++')` | `entrenamiento/fenotipado.py` | ✅ |
| Validación interna | Método del codo + `silhouette_score` | `entrenamiento/fenotipado.py` | ✅ |

### Unidad 4 — Optimización y Redes Neuronales

| Técnica | Clase/función | Archivo esperado | Estado |
|---------|---------------|------------------|--------|
| Búsqueda de hiperparámetros | `GridSearchCV` o `RandomizedSearchCV` | `entrenamiento/optimizador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Validación cruzada estratificada | `StratifiedKFold` | `entrenamiento/optimizador.py` | [✅/`EVIDENCE_NOT_FOUND`] |
| Red neuronal | `MLPClassifier` | `entrenamiento/comparador_modelos.py` | [✅/`EVIDENCE_NOT_FOUND`] |

---

## 5. Registro de cambios [AUTO]

> El agente añade una entrada cada vez que actualiza este documento.

| Fecha | Cambio | Nivel anterior → nuevo | Responsable |
|-------|--------|------------------------|-------------|
| 2026-05-17 | Implementado `FenotipadoKMeans` con pruebas unitarias; implementado `OptimizadorHiperparametros` con pruebas; ejecución de la muestra mínima de estabilidad y pruebas pasadas. Suite de pruebas `pruebas/` ejecutada: 20 passed. | Básico → Básico (hacia Intermedio) | AcademicoIA |

---

## 6. Acciones correctivas pendientes

> Lista priorizada de lo que falta para subir de nivel. Ordenada por impacto en rúbrica.

### Bloqueantes para Nivel Básico
- [ ] [acción concreta] → archivo a modificar: `[ruta]`

### Bloqueantes para Nivel Intermedio (+15)
- [x] K-Means implementado — archivo: `entrenamiento/fenotipado.py`
- [ ] Dashboard interactivo — ejecutar con `streamlit run dashboard/app.py`
- [ ] [otras acciones]

### Bloqueantes para Nivel Avanzado (+30)
- [ ] API en producción — `uvicorn api.main:app`
- [ ] Comparativa con al menos 1 paper académico en el reporte

---

## 7. Preguntas frecuentes de presentación

> El agente genera esta sección para preparar al equipo para la defensa oral.

1. **¿Por qué usaron `Pipeline` en lugar de transformar directamente?**
   → [respuesta basada en evidencia del proyecto]

2. **¿Cómo manejaron el desbalance de clases?**
   → [respuesta basada en evidencia del proyecto]

3. **¿Por qué eligieron [modelo final] como el modelo principal?**
   → [respuesta basada en métricas reportadas]

4. **¿Qué es el coeficiente de silueta y cómo lo usaron en K-Means?**
   → [respuesta basada en evidencia del proyecto]

5. **¿Cómo evitaron la fuga de datos (data leakage)?**
   → [respuesta basada en evidencia del proyecto]