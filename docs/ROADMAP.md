# diasgnostico-pred — Hoja de Ruta

**Última actualización:** 2026-05-17
**Estructura:** alineada a la rúbrica del proyecto final
**Calificación objetivo:** 100 base + 30 puntos extra (Nivel Avanzado)

---

## Mapa de calificación

```
Calificación final = Base (0-100) + Puntos extra

Base:
  Código y Técnica  × 0.40   ← mayor peso; pipeline, tests, documentación
  Resultados        × 0.30   ← métricas, comparativa, interpretación
  Reporte           × 0.20   ← estructura, visualizaciones, conclusiones
  Presentación      × 0.10   ← demo funcional, defensa oral

Puntos extra:
  Nivel Básico     →   +0   (piso mínimo para aprobar)
  Nivel Intermedio → +15   (requiere Básico completo)
  Nivel Avanzado   → +30   (requiere Intermedio completo)
```

---

## Estado general

| Sprint | Descripción | Ítems de rúbrica | Estado |
|--------|-------------|------------------|--------|
| S1 | Andamiaje base | — | ✅ Completo |
| S2 | Pipeline robusto y modelos supervisados | B1, B2, B3 — **Nivel Básico** | ✅ Validado |
| S3 | K-Means, hiperparámetros y dashboard | I1–I6 — **Nivel Intermedio** | ⬜ Pendiente |
| S4 | Reporte y presentación | Componentes Reporte 20% + Presentación 10% | ⬜ Pendiente |
| S5 | API y comparativa con papers | A1, A2 — **Nivel Avanzado** | ⬜ Pendiente |

---

## Sprint 1 — Andamiaje base

**Estado: ✅ Completo**
**Impacto en rúbrica:** sienta la base del componente Código y Técnica (40%).

| ID | Entregable | Archivo | Estado |
|----|------------|---------|--------|
| S1-01 | Configuración central: rutas, columnas CDC y constantes | `config.py` | ✅ |
| S1-02 | Cargador de datos | `entrenamiento/cargador_datos.py` | ✅ |
| S1-03 | Comparador de modelos (esqueleto) | `entrenamiento/comparador_modelos.py` | ✅ |
| S1-04 | Pipeline base | `entrenamiento/pipeline.py` | ✅ |
| S1-05 | Predictor de inferencia | `inferencia/predictor.py` | ✅ |
| S1-06 | Esquemas de la API | `api/esquemas.py` | ✅ |
| S1-07 | API principal (esqueleto) | `api/main.py` | ✅ |
| S1-08 | Pruebas de API | `pruebas/test_api.py` | ✅ |
| S1-09 | Pruebas de cargador | `pruebas/test_cargador.py` | ✅ |
| S1-10 | Pruebas de predictor | `pruebas/test_predictor.py` | ✅ |
| S1-11 | Empaquetado del proyecto | `pyproject.toml`, `.env.example`, `modelos/.gitkeep` | ✅ |

---

## Sprint 2 — Pipeline robusto y modelos supervisados

**Estado: ✅ Validado** (experimento 1000 muestras — `reportes/comparativa_1000_intermedio.md`)
**Ítems de rúbrica cubiertos: B1, B2, B3 → Nivel Básico**

> La evidencia experimental valida: `Pipeline` completo y serializable con `KNNImputer`,
> `SMOTE`, `GridSearchCV` + `StratifiedKFold`, modelos `svm`, `arbol`, `gbm`, `mlp`,
> y tabla comparativa con ROC-AUC, PR-AUC, sensibilidad, especificidad, F1, Brier y accuracy.

> Los archivos dentro de `reportes/` se consideran salidas derivadas del pipeline. La reconstrucción de un informe parte del JSON crudo y de la síntesis legible, no de editar esos archivos a mano.

| ID | Tarea | Ítem rúbrica | Archivo | Estado |
|----|-------|--------------|---------|--------|
| S2-01 | Dataset CDC BRFSS 2015 disponible | B2 | `datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv` | ✅ |
| S2-02 | EDA regionalizado | Resultados 30% | `notebooks/01_eda_regionalizado.ipynb` | ✅ |
| S2-03 | Preprocesador robusto con `KNNImputer`, `ColumnTransformer` y `SMOTE` dentro del `Pipeline` | B2 | `entrenamiento/preprocesador.py` | ✅ |
| S2-04 | Modelos supervisados: `svm`, `arbol`, `gbm`, `mlp` | B1, I1, I2, I3 | `entrenamiento/comparador_modelos.py` | ✅ |
| S2-05 | Evaluador clínico con PR-AUC, ROC-AUC, F1, sensibilidad, especificidad | B3 | `entrenamiento/evaluador.py` | ✅ |
| S2-06 | Pipeline serializable con `predict_proba` expuesto | B1, B2 | `joblib.dump(Pipeline completo)` | ✅ |
| S2-07 | Pruebas del preprocesador | Código y Técnica 40% | `pruebas/test_preprocesador.py` | ✅ |
| S2-08 | Pruebas del cargador | Código y Técnica 40% | `pruebas/test_cargador.py` | ✅ |
| S2-09 | Persistencia del dataset procesado en Parquet | Resultados 30% | `datos/procesados/dataset_procesado.parquet` | 🟡 Parcial |
| S2-10 | Contraste regional documentado | Reporte 20% | `reportes/contraste_regional.md` | 🟡 Parcial |

**Veredicto Básico:** COMPLETADO — B1 ✅ B2 ✅ B3 ✅
**Pendientes de cierre limpio:** S2-09 y S2-10 (no bloquean el nivel; sí enriquecen Resultados 30% y Reporte 20%).

---

## Sprint 3 — K-Means, optimización e hiperparámetros

**Estado: ⚠️ En progreso**
**Ítems de rúbrica cubiertos: I1, I2, I3, I4, I5 → parcial de Nivel Intermedio**
**Prerequisito:** Sprint 2 validado ✅

> Los modelos SVM, Árbol, MLP y GBM ya existen en `comparador_modelos.py` desde S2.
> Este sprint los consolida bajo evaluación uniforme (I1-I3), añade K-Means formal (I4)
> y formaliza la optimización de hiperparámetros como módulo independiente (I5).

| ID | Tarea | Ítem rúbrica | Archivo | Estado |
|----|-------|--------------|---------|--------|
| S3-01 | Verificar que SVM, Árbol y MLP se evalúan con las mismas métricas que GBM | I1, I2, I3 | `entrenamiento/comparador_modelos.py` | ✅ Verificado mediante prueba automatizada en `pruebas/test_comparador.py` (2026-05-18) |
| S3-02 | K-Means con `init='k-means++'`, método del codo y `silhouette_score` | I4 | `entrenamiento/fenotipado.py` | ✅ |
| S3-03 | Notebook de análisis de fenotipos (visualización de clústeres, interpretación) | I4, Resultados 30% | `notebooks/02_fenotipado_kmeans.ipynb` | ⬜ |
| S3-04 | `GridSearchCV` con `StratifiedKFold` como módulo formal (ya validado en experimento) | I5 | `entrenamiento/optimizador.py` | ✅ |
| S3-05 | Pruebas del fenotipador | Código y Técnica 40% | `pruebas/test_fenotipado.py` | ✅ |
| S3-06 | Pruebas del optimizador | Código y Técnica 40% | `pruebas/test_optimizador.py` | ⬜ |
| S3-07 | Cerrar S2-09: persistir `dataset_procesado.parquet` | Resultados 30% | `datos/procesados/dataset_procesado.parquet` | ✅ |
| S3-08 | Cerrar S2-10: documentar contraste regional real | Reporte 20% | `reportes/contraste_regional.md` | ⬜ |

**Validación de cierre de Sprint 3:**
```bash
pytest pruebas/test_fenotipado.py -v
pytest pruebas/test_optimizador.py -v
pytest pruebas/ --cov=entrenamiento --cov-report=term-missing
```

---

## Sprint 4 — Dashboard interactivo

**Estado: ⬜ Pendiente**
**Ítems de rúbrica cubiertos: I6 → completa Nivel Intermedio (+15)**
**Prerequisito:** Sprint 3 completo ✅

> I6 es el único ítem de Intermedio que requiere trabajo fuera de `entrenamiento/`.
> Se separa en sprint propio porque tiene dependencias distintas (Streamlit)
> y su validación es funcional (ejecutar la app), no solo de pruebas unitarias.

| ID | Tarea | Ítem rúbrica | Archivo | Estado |
|----|-------|--------------|---------|--------|
| S4-01 | Instalar y configurar Streamlit en el proyecto | I6 | `pyproject.toml` (añadir `streamlit`) | ⬜ |
| S4-02 | Vista de comparativa de modelos (tabla de métricas interactiva) | I6 | `dashboard/app.py` | ⬜ |
| S4-03 | Vista de predicción individual (formulario → resultado) | I6, Presentación 10% | `dashboard/app.py` | ⬜ |
| S4-04 | Vista de fenotipos K-Means (gráfica de clústeres) | I6, Resultados 30% | `dashboard/app.py` | ⬜ |
| S4-05 | El dashboard carga el pipeline serializado para inferencia | I6, B1 | `dashboard/app.py` | ⬜ |

**Validación de cierre de Sprint 4:**
```bash
streamlit run dashboard/app.py
# Verificar manualmente:
# - Vista de comparativa carga y muestra la tabla de métricas
# - Formulario de predicción devuelve resultado
# - Gráfica de clústeres se renderiza
```

**Veredicto Intermedio al cerrar S4:** I1 ✅ I2 ✅ I3 ✅ I4 ✅ I5 ✅ I6 ✅ → **+15 puntos extra**

---

## Sprint 5 — Reporte y cierre de presentación

**Estado: ⬜ Pendiente**
**Ítems de rúbrica cubiertos: Reporte 20% + Presentación 10%**
**Prerequisito:** Sprint 4 completo ✅

> Este sprint no añade código nuevo; consolida el trabajo existente en
> documentación evaluable y prepara el artefacto de demo para la defensa oral.

| ID | Tarea | Componente rúbrica | Archivo | Estado |
|----|-------|--------------------|---------|--------|
| S5-01 | Reporte principal: introducción y planteamiento del problema | Reporte 20% | `reportes/reporte_final.md` generado desde la corrida | ⬜ |
| S5-02 | Reporte: metodología (pipeline, modelos, métricas, decisiones técnicas) | Reporte 20% | `reportes/reporte_final.md` generado desde la corrida | ⬜ |
| S5-03 | Reporte: resultados con visualizaciones (curvas ROC, PR, matriz de confusión) | Reporte 20%, Resultados 30% | `reportes/reporte_final.md` generado desde la corrida | ⬜ |
| S5-04 | Reporte: conclusiones y limitaciones del proyecto | Reporte 20% | `reportes/reporte_final.md` generado desde la corrida | ⬜ |
| S5-05 | Preparar guía de ejecución del demo (`README_demo.md`) | Presentación 10% | `README_demo.md` | ⬜ |
| S5-06 | Ensayo de defensa: respuestas preparadas a preguntas típicas del temario | Presentación 10% | `docs/preguntas_defensa.md` | ⬜ |
| S5-07 | Verificación final: ejecutar pipeline, dashboard y pruebas sin errores | Todos | — | ⬜ |

**Nota operativa:** si se necesita regenerar un reporte, el flujo correcto es ejecutar el pipeline para obtener el JSON crudo y luego usar `scripts/generar_reporte_legible.py` para producir el Markdown.

**Validación de cierre de Sprint 5:**
```bash
# Pruebas completas sin errores
pytest pruebas/ -v --tb=short

# Dashboard funcional
streamlit run dashboard/app.py

# Panel Problems de VS Code limpio en todos los módulos de entrenamiento
```

---

## Sprint 6 — API en producción y comparativa con papers

**Estado: ⬜ Pendiente**
**Ítems de rúbrica cubiertos: A1, A2 → Nivel Avanzado (+30)**
**Prerequisito:** Sprint 5 completo ✅ — no iniciar si el reporte o la base no están sólidos.

> El esqueleto de la API ya existe desde S1 (`api/main.py`, `api/esquemas.py`).
> Este sprint lo hace funcional con el pipeline entrenado y añade la comparativa
> académica al reporte.

| ID | Tarea | Ítem rúbrica | Archivo | Estado |
|----|-------|--------------|---------|--------|
| S6-01 | Conectar `api/main.py` con el pipeline serializado para inferencia real | A1 | `api/main.py` | ⬜ |
| S6-02 | Endpoint `POST /predict` con validación de entrada por esquemas | A1 | `api/main.py`, `api/esquemas.py` | ⬜ |
| S6-03 | Endpoint `GET /health` para verificar estado del servicio | A1 | `api/main.py` | ⬜ |
| S6-04 | Pruebas de integración de los endpoints | A1, Código y Técnica 40% | `pruebas/test_api.py` (actualizar) | ⬜ |
| S6-05 | Instrucciones de arranque documentadas | A1, Presentación 10% | `README_demo.md` (añadir sección API) | ⬜ |
| S6-06 | Identificar 1–2 papers académicos relevantes (diabetes, predicción clínica, ML) | A2 | — | ⬜ |
| S6-07 | Sección de comparativa con papers en el reporte final | A2, Reporte 20% | `reportes/reporte_final.md` | ⬜ |

**Validación de cierre de Sprint 6:**
```bash
# API levanta sin errores
uvicorn api.main:app --reload

# Endpoint responde correctamente
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"HighBP": 1, "HighChol": 0, ...}'

# Pruebas de API pasan
pytest pruebas/test_api.py -v
```

**Veredicto Avanzado al cerrar S6:** A1 ✅ A2 ✅ → **+30 puntos extra**

---

## Calificación proyectada al completar cada sprint

| Al cerrar | Nivel | Puntos extra | Componentes base cubiertos |
|-----------|-------|--------------|----------------------------|
| Sprint 2 | Básico | +0 | Código y Técnica parcial, Resultados parcial |
| Sprint 3 | Básico → Intermedio parcial | +0 | Código y Técnica mejorado, Resultados mejorado |
| Sprint 4 | Intermedio | **+15** | Presentación habilitada |
| Sprint 5 | Intermedio | **+15** | Reporte 20% + Presentación 10% completados |
| Sprint 6 | Avanzado | **+30** | Todos los componentes al máximo |

---

## Regla de secuencialidad

```
S1 → S2 (✅ ya completos)
         ↓
        S3  ←  consolidar I1-I5 y K-Means formal
         ↓
        S4  ←  dashboard (I6) — desbloquea +15
         ↓
        S5  ←  reporte y demo — consolida Reporte 20% + Presentación 10%
         ↓
        S6  ←  API + papers (A1, A2) — desbloquea +30
```

No iniciar un sprint si el anterior tiene ítems de nivel sin completar.
Los ítems `🟡 Parcial` de S2 (S2-09, S2-10) se cierran en S3 como S3-07 y S3-08.

---

## Próximo paso inmediato

Iniciar **S3-01**: verificar que SVM, Árbol y MLP en `comparador_modelos.py`
se evalúan con el mismo conjunto de métricas que GBM, y que los resultados
quedan en la tabla comparativa del reporte.

```bash
# Punto de entrada para el agente
@AcademicoIA inicia Sprint 3 — verifica cobertura uniforme de métricas en comparador_modelos.py
```