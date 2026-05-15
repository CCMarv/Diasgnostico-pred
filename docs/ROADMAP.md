# diasgnostico-pred — Hoja de Ruta

**Última actualización:** 2026-05-14 | **Metodología:** Espiral iterativa, 5 sprints

---

## Estado general

| Dimensión | Estado |
|---|---|
| Sprint 1 — Andamiaje base | ✅ Completo |
| Sprint 2 — Pipeline de datos y modelos supervisados | ⚠️ 8/10 tickets completos; 2 artefactos pendientes |
| Sprint 3 — Fenotipado metabólico y Dashboard | ❌ No iniciado |
| Sprint 4 — Reporte académico y despliegue | ❌ No iniciado |
| Sprint 5 — Observabilidad y endurecimiento | ❌ No iniciado |

### Bloqueantes actuales del Sprint 2 (deben resolverse antes del Sprint 3)

- `notebooks/01_eda_regionalizado.ipynb` — creado pero sin validación completa con datos reales
- `reportes/contraste_regional.md` — el script generador existe; el artefacto de salida no se ha producido con el dataset real

---

## Sprint 1 — Andamiaje y contratos base

**Objetivo:** Establecer la arquitectura modular completa, contratos de interfaces y una suite mínima de pruebas de contrato. El repositorio debe ser clonable, instalable y ejecutable sin modelo real.

**Estado:** ✅ Completo

| ID | Entregable | Estado |
|---|---|---|
| S1-01 | `config.py` con constantes globales, rutas y columnas CDC | ✅ |
| S1-02 | `entrenamiento/cargador_datos.py` — clase `CargadorDatos` | ✅ |
| S1-03 | `entrenamiento/comparador_modelos.py` — clase `ComparadorModelos` (stub) | ✅ |
| S1-04 | `entrenamiento/pipeline.py` — CLI + `ejecutar_pipeline()` | ✅ |
| S1-05 | `inferencia/predictor.py` — clase `PredictorDiabetes` | ✅ |
| S1-06 | `api/esquemas.py` — `DatosPaciente`, `RespuestaPrediccion`, `RespuestaSalud` | ✅ |
| S1-07 | `api/main.py` — endpoints `/salud` y `/predecir` con modo degradado | ✅ |
| S1-08 | `pruebas/test_api.py` — pruebas de contrato HTTP | ✅ |
| S1-09 | `pruebas/test_cargador.py` — pruebas de carga y limpieza | ✅ |
| S1-10 | `pruebas/test_predictor.py` — pruebas con modelos simulados | ✅ |
| S1-11 | `pyproject.toml`, `.env.example`, `modelos/.gitkeep` | ✅ |

---

## Sprint 2 — Pipeline de datos robusto y modelado supervisado

**Objetivo:** Reemplazar `DummyClassifier` con un pipeline de calidad clínica: preprocesamiento, tres modelos supervisados, métricas clínicas y artefactos serializados.

**Estado:** ⚠️ 8/10 completos

| ID | Tarea | Prioridad | Dependencia | Estado |
|---|---|---|---|---|
| S2-01 | Descargar CDC BRFSS 2015 vía `ucimlrepo` → `datos/brutos/` | CRÍTICA | — | ✅ Completo |
| S2-02 | Crear `notebooks/01_eda_regionalizado.ipynb` (6 bloques) | CRÍTICA | S2-01 | ⚠️ Parcial — creado, pendiente validación con datos reales |
| S2-03 | Implementar `entrenamiento/preprocesador.py` con `ColumnTransformer` | CRÍTICA | S2-01 | ✅ Completo |
| S2-04 | Extender `ComparadorModelos` con SVM, GBM, MLP usando Pipeline sklearn | CRÍTICA | S2-03 | ✅ Completo |
| S2-05 | Crear `entrenamiento/evaluador.py` con métricas clínicas + gráficas | ALTA | S2-04 | ✅ Completo |
| S2-06 | Ampliar `entrenamiento/pipeline.py` para serializar Pipeline completo | ALTA | S2-03, S2-04 | ✅ Completo |
| S2-07 | Añadir `pruebas/test_preprocesador.py` — verifica ausencia de data leakage | ALTA | S2-03 | ✅ Completo |
| S2-08 | Ampliar `pruebas/test_cargador.py` con análisis de distribución y desbalance | MEDIA | S2-01 | ✅ Completo |
| S2-11 | Refactor: Extraer catálogo de modelos y helpers de ajuste (`ComparadorModelos`) | MEDIA | S2-04 | ⬜ Pendiente |
| S2-12 | Refactor: Reorganizar `ejecutar_pipeline` en pasos y extraer utilitarios | MEDIA | S2-06 | ⬜ Pendiente |
| S2-09 | Persistir dataset procesado en `datos/procesados/` formato Parquet | MEDIA | S2-03 | ⚠️ Parcial — método `persistir_procesado()` existe; sin artefacto versionado en repo |
| S2-10 | Generar `reportes/contraste_regional.md` con datos reales | MEDIA | S2-02 | ⚠️ Parcial — script generador en `reportes/generar_contraste_regional.py`; salida no producida |

---

## Sprint 3 — Fenotipado metabólico y Dashboard Streamlit

**Objetivo:** Implementar K-Means como fenotipador clínico (no como predictor), construir el dashboard Streamlit orientado al médico general y generar artefactos de explicabilidad SHAP.

**Estado:** ❌ No iniciado — bloqueado en S2-09, S2-10

| ID | Tarea | Prioridad | Dependencia | Estado |
|---|---|---|---|---|
| S3-01 | Implementar `entrenamiento/fenotipador.py` (método del codo + silhouette) | CRÍTICA | S2-04 | ⬜ Pendiente |
| S3-02 | Integrar fenotipador como etapa 0 en pipeline supervisado | CRÍTICA | S3-01, S2-06 | ⬜ Pendiente |
| S3-03 | Generar `reportes/perfiles_fenotipos.md` con prevalencia de diabetes por cluster | CRÍTICA | S3-01 | ⬜ Pendiente |
| S3-04 | Crear `dashboard/app.py` — Pantalla 1: formulario de captura | CRÍTICA | — | ⬜ Pendiente |
| S3-05 | Crear `dashboard/app.py` — Pantalla 2: resultado con gauge y SHAP | ALTA | S3-04, S3-07 | ⬜ Pendiente |
| S3-06 | Crear `dashboard/app.py` — Pantalla 3: panel estadístico del médico | ALTA | S3-04 | ⬜ Pendiente |
| S3-07 | Calcular SHAP values para el mejor modelo supervisado | ALTA | S2-04 | ⬜ Pendiente |
| S3-08 | Crear `dashboard/cliente_api.py` con manejo de errores 503 | ALTA | S3-04 | ⬜ Pendiente |
| S3-09 | Crear `.streamlit/config.toml` con tema IMSS | MEDIA | S3-04 | ⬜ Pendiente |
| S3-10 | Añadir `pruebas/test_fenotipador.py` — pruebas de integración | MEDIA | S3-01 | ⬜ Pendiente |
| S3-11 | Prueba de integración: formulario Streamlit → API → resultado coherente | MEDIA | S3-05, S3-08 | ⬜ Pendiente |

---

## Sprint 4 — Reporte académico y despliegue final

**Objetivo:** Producir el reporte académico en estructura IMRaD, contenerización Docker, GitHub Actions CI/CD y análisis de equidad.

**Estado:** ❌ No iniciado — bloqueado en Sprint 3

| ID | Tarea | Prioridad | Dependencia | Estado |
|---|---|---|---|---|
| S4-01 | Redactar secciones 1–3 de `reportes/paper_final.md` con resultados del EDA | CRÍTICA | S2-02 | ⬜ Pendiente |
| S4-02 | Completar secciones 4–5 con tablas de resultados y gráficas generadas | CRÍTICA | S2-05, S3-07 | ⬜ Pendiente |
| S4-03 | Implementar `entrenamiento/evaluador_equidad.py` — ROC-AUC por subgrupo | CRÍTICA | S2-05 | ⬜ Pendiente |
| S4-04 | Redactar sección 6 (Discusión) con contraste CDC ↔ ENSANUT | ALTA | S4-01, S4-02 | ⬜ Pendiente |
| S4-05 | Crear `Dockerfile` multi-stage (targets API + Dashboard) | ALTA | S3-08 | ⬜ Pendiente |
| S4-06 | Crear `docker-compose.yml` (servicio API + servicio Dashboard) | ALTA | S4-05 | ⬜ Pendiente |
| S4-07 | Configurar `.github/workflows/ci.yml` — lint + pytest + build de imagen | ALTA | — | ⬜ Pendiente |
| S4-08 | Añadir `pip-audit` al pipeline de CI | MEDIA | S4-07 | ⬜ Pendiente |
| S4-09 | Exportar paper a PDF vía nbconvert o pandoc | MEDIA | S4-02 | ⬜ Pendiente |
| S4-10 | Etiquetar versión `v1.0.0` en el repositorio | MEDIA | S4-09 | ⬜ Pendiente |

---

## Sprint 5 — Observabilidad, endurecimiento y documentación final

**Objetivo:** Sistema observable en producción, resistente a fallos y completamente documentado para operaciones y usuarios clínicos.

**Estado:** ❌ No iniciado — bloqueado en Sprint 4

| ID | Tarea | Prioridad | Dependencia | Estado |
|---|---|---|---|---|
| S5-01 | Pipeline CI/CD con lint, pruebas y build de imagen | ALTA | — | ⬜ Pendiente |
| S5-02 | Análisis estático de tipos con `mypy` o `pyright` | ALTA | — | ⬜ Pendiente |
| S5-03 | Linting con `ruff` + hooks de pre-commit | ALTA | — | ⬜ Pendiente |
| S5-04 | Monitoreo de drift de datos (detección de distribución fuera del rango de entrenamiento) | MEDIA | S4-06 | ⬜ Pendiente |
| S5-05 | Documentar proceso de re-entrenamiento y estrategia de actualización del modelo | MEDIA | S4-05 | ⬜ Pendiente |
| S5-06 | Guía de despliegue en producción (Cloud Run / ECS / VPS con Docker) | MEDIA | S4-06 | ⬜ Pendiente |
| S5-07 | Manual de usuario para personal clínico (interpretación de categorías y advertencias) | MEDIA | S3-05 | ⬜ Pendiente |
| S5-08 | Auditoría de seguridad: `pip audit`, cabeceras HTTP seguras | MEDIA | S4-07 | ⬜ Pendiente |
| S5-09 | Revisión de equidad (fairness) por grupo de edad, sexo e ingreso | ALTA | S4-03 | ⬜ Pendiente |
| S5-10 | Etiqueta de versión estable `v1.0.0` en el repositorio | BAJA | S5-09 | ⬜ Pendiente |

---

## Estructura de directorios objetivo (al completar Sprint 4)

```
diasgnostico-pred/
├── api/
│   ├── __init__.py
│   ├── config.py
│   ├── esquemas.py
│   └── main.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py                              ← S3-04/05/06
│   └── cliente_api.py                      ← S3-08
├── datos/
│   ├── brutos/
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── procesados/
│       └── dataset_procesado.parquet       ← S2-09
├── entrenamiento/
│   ├── __init__.py
│   ├── cargador_datos.py
│   ├── comparador_modelos.py
│   ├── evaluador.py
│   ├── evaluador_equidad.py                ← S4-03
│   ├── fenotipador.py                      ← S3-01
│   └── preprocesador.py
├── inferencia/
│   ├── __init__.py
│   └── predictor.py
├── modelos/
│   ├── .gitkeep
│   ├── fenotipador.joblib                  ← S3-01
│   └── modelo_diabetes_v1.joblib           ← S2-06
├── notebooks/
│   └── 01_eda_regionalizado.ipynb          ← S2-02
├── pruebas/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_cargador.py
│   ├── test_fenotipador.py                 ← S3-10
│   ├── test_predictor.py
│   └── test_preprocesador.py
├── reportes/
│   ├── comparativa_modelos.md              ← S2-05
│   ├── contraste_regional.md               ← S2-10
│   ├── perfiles_fenotipos.md               ← S3-03
│   ├── shap_summary.png                    ← S3-07
│   └── paper_final.md                      ← S4-01/02
├── .env.example
├── .github/workflows/ci.yml                ← S4-07
├── .streamlit/config.toml                  ← S3-09
├── config.py
├── docker-compose.yml                      ← S4-06
├── Dockerfile                              ← S4-05
└── pyproject.toml
```

---

## Checklist de entrega para máxima calificación

### Nivel Básico (Sprint 2)

- [ ] 3 modelos supervisados entrenados con dataset real
- [ ] Preprocesamiento documentado y sin data leakage
- [ ] Tabla de métricas comparativa (accuracy, F1, AUC)

### Nivel Intermedio (Sprint 3)

- [ ] SVM con kernel RBF y búsqueda de hiperparámetros
- [ ] Árbol de decisión / Gradient Boosting con importancia de variables
- [ ] Red neuronal (MLP) con early stopping
- [ ] K-Means como fenotipador metabólico (no como predictor) con K justificado
- [ ] Dashboard Streamlit con interfaz de consultorio IMSS

### Nivel Avanzado (API completada + Sprint 4)

- [ ] API REST FastAPI en producción con validación clínica — **✅ completado**
- [ ] Análisis de sesgo poblacional CDC ↔ México documentado
- [ ] SHAP values para interpretabilidad clínica
- [ ] Análisis de equidad por subgrupo demográfico
- [ ] Comparativa con literatura sobre T2DM en población hispana
- [ ] Curvas de calibración + Brier Score

### Diferenciadores competitivos

- [ ] Fenotipado metabólico con nombres clínicos interpretables
- [ ] Umbral de decisión ajustado para contexto México (0.25 en lugar de 0.33)
- [ ] Limitaciones redactadas como contribuciones metodológicas
- [ ] Recomendaciones de trabajo futuro con datos ENSANUT reales
- [ ] Análisis de fairness explícito con métricas de paridad demográfica
