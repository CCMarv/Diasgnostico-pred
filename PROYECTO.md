# Diasgnostico-pred
 — Documento de Diseño y Progreso del Proyecto

> **Versión del sistema:** 0.1.0  
> **Dataset fuente:** CDC BRFSS 2015 — `diabetes_binary_health_indicators_BRFSS2015.csv`  
> **Metodología de desarrollo:** Espiral iterativa (5 sprints)

---

## 1. Meta final

Construir un sistema modular, robusto y desplegable en producción que permita **estimar el riesgo de diabetes de una persona** a partir de 21 indicadores de salud del estudio CDC BRFSS 2015. El sistema debe:

- Entregar una probabilidad continua y una categoría de riesgo (`bajo`, `medio`, `alto`) a través de una API REST.
- Ser reproducible: cualquier desarrollador puede reclonar el repositorio, entrenar el modelo y levantar la API sin pasos manuales.
- Ser observable: logs estructurados, métricas de latencia y un endpoint de salud (`/salud`) que refleje el estado real del servicio.
- Ser seguro: validación estricta de entradas, manejo explícito de errores y sin exposición de datos personales (PII).
- Contar con una suite de pruebas automatizadas que garantice que ninguna regresión llegue a producción.

---

## 2. Lógica del diseño

### 2.1 Arquitectura general

```
┌─────────────────────────────────────────────┐
│               Cliente / Consumidor           │
└────────────────────┬────────────────────────┘
                     │ HTTP JSON
┌────────────────────▼────────────────────────┐
│              api/  (FastAPI)                 │
│  /salud  ── health check                    │
│  /predecir ── validación → inferencia       │
│  esquemas.py  (Pydantic, mapeo ES ↔ CDC)   │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│         inferencia/predictor.py             │
│  PredictorDiabetes: carga .joblib,          │
│  valida columnas CDC, mide latencia         │
└────────────────────┬────────────────────────┘
                     │ modelo entrenado (.joblib)
┌────────────────────▼────────────────────────┐
│         entrenamiento/                       │
│  cargador_datos.py  → limpieza CDC          │
│  comparador_modelos.py → experimentos       │
│  pipeline.py  → CLI orquestador             │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│     datos/brutos/  (CSV CDC BRFSS 2015)     │
│     modelos/       (artefactos .joblib)     │
│     reportes/      (métricas JSON)          │
└─────────────────────────────────────────────┘
```

### 2.2 Principios de diseño

| Principio | Aplicación concreta |
|---|---|
| **Separación de responsabilidades** | Cuatro capas independientes: datos, entrenamiento, inferencia, API |
| **Contrato explícito** | `config.py` centraliza todas las constantes (columnas, rutas, umbrales, semilla). Ningún módulo define "magic strings". |
| **Español primero** | Nombres de clases, métodos, variables y endpoints en español; solo las columnas CDC permanecen en inglés (nombres oficiales del dataset). |
| **Modo degradado** | Si el modelo `.joblib` no existe, la API responde `/salud` con `estado: degradado` en lugar de fallar en el arranque. |
| **Incertidumbre explícita** | Si la probabilidad cae dentro de ±5% de un umbral, la respuesta incluye una advertencia clínica. |
| **Pruebas de contrato** | Los tests verifican contratos de interfaz (tipos devueltos, códigos HTTP, mapeos de campos), no implementaciones internas. |

### 2.3 Flujo de datos: entrenamiento → inferencia

```
CSV (datos/brutos/)
    │
    ▼ CargadorDatos.cargar()
DataFrame (21 columnas CDC + objetivo)
    │
    ▼ train_test_split (stratify, semilla=42, test=0.2)
X_train / X_test / y_train / y_test
    │
    ▼ ComparadorModelos.entrenar_clasificacion()
Lista[ResultadoModelo]
    │
    ▼ ComparadorModelos.seleccionar_mejor()
mejor_modelo (sklearn estimator)
    │
    ▼ joblib.dump()
modelos/modelo_diabetes.joblib  +  reportes/metricas_entrenamiento.json
    │
    ▼ PredictorDiabetes.cargar_modelo()  (al arrancar la API)
PredictorDiabetes._modelo listo
    │
    ▼ POST /predecir  →  DatosPaciente.a_dataframe()
probabilidad + categoría_riesgo + advertencia
```

### 2.4 Esquema de validación de entrada

Los 21 campos públicos de `DatosPaciente` están en español sin PII. Pydantic aplica rangos clínicamente plausibles:

| Campo público (ES) | Columna CDC | Rango permitido |
|---|---|---|
| `presion_alta` | `HighBP` | 0 – 1 |
| `colesterol_alto` | `HighChol` | 0 – 1 |
| `chequeo_colesterol` | `CholCheck` | 0 – 1 |
| `imc` | `BMI` | 10.0 – 80.0 |
| `fumador` | `Smoker` | 0 – 1 |
| `derrame_cerebral` | `Stroke` | 0 – 1 |
| `enfermedad_corazon` | `HeartDiseaseorAttack` | 0 – 1 |
| `actividad_fisica` | `PhysActivity` | 0 – 1 |
| `consume_fruta` | `Fruits` | 0 – 1 |
| `consume_verdura` | `Veggies` | 0 – 1 |
| `consumo_alcohol_alto` | `HvyAlcoholConsump` | 0 – 1 |
| `tiene_cobertura_medica` | `AnyHealthcare` | 0 – 1 |
| `sin_medico_por_costo` | `NoDocbcCost` | 0 – 1 |
| `salud_general` | `GenHlth` | 1 – 5 |
| `salud_mental` | `MentHlth` | 0 – 30 |
| `salud_fisica` | `PhysHlth` | 0 – 30 |
| `dificultad_caminar` | `DiffWalk` | 0 – 1 |
| `sexo` | `Sex` | 0 – 1 |
| `edad` | `Age` | 1 – 13 (grupos etarios CDC) |
| `educacion` | `Education` | 1 – 6 |
| `ingreso` | `Income` | 1 – 8 |

Adicionalmente, un `@model_validator` detecta incoherencias clínicas: `salud_fisica ≥ 20` con `dificultad_caminar = 0` es rechazado con HTTP 422.

### 2.5 Umbrales de riesgo

```python
UMBRAL_RIESGO_BAJO  = 0.33   # probabilidad < 0.33 → "bajo"
UMBRAL_RIESGO_ALTO  = 0.66   # probabilidad ≥ 0.66 → "alto"
MARGEN_INCERTIDUMBRE = 0.05  # ±5% de un umbral → advertencia clínica
```

---

## 3. Sprints y metas

### Sprint 1 — Scaffolding y contratos base ✅ COMPLETADO

**Meta:** Establecer la arquitectura modular completa, contratos de interfaces y una suite de pruebas de contrato mínima que valide que todos los módulos "hablan entre sí" correctamente. Al final de este sprint el repositorio debe ser clonable, instalable y ejecutable, aunque sin un modelo real entrenado.

**Entregables comprometidos:**

| Entregable | Estado |
|---|---|
| `config.py` con constantes globales, rutas y columnas CDC | ✅ |
| `entrenamiento/cargador_datos.py` — clase `CargadorDatos` | ✅ |
| `entrenamiento/comparador_modelos.py` — clase `ComparadorModelos` (stub) | ✅ |
| `entrenamiento/pipeline.py` — CLI `ejecutar_pipeline()` + argparse | ✅ |
| `inferencia/predictor.py` — clase `PredictorDiabetes` | ✅ |
| `api/esquemas.py` — `DatosPaciente`, `RespuestaPrediccion`, `RespuestaSalud` | ✅ |
| `api/main.py` — endpoints `/salud` y `/predecir` con modo degradado | ✅ |
| `pruebas/test_api.py` — pruebas de contrato HTTP | ✅ |
| `pruebas/test_cargador.py` — prueba de carga y limpieza | ✅ |
| `pruebas/test_predictor.py` — prueba con modelos simulados | ✅ |
| `pyproject.toml` — dependencias, configuración de pytest | ✅ |
| `.env.example` — variables de entorno documentadas | ✅ |
| `modelos/.gitkeep` — placeholder para artefactos (excluidos de git) | ✅ |

**Detalles técnicos implementados:**
- El `ComparadorModelos` implementa actualmente un `DummyClassifier(strategy="prior")` como clasificador base y `KMeans` como modelo de clustering. Esto define la interfaz pero **no** produce predicciones de calidad clínico.
- El predictor soporta modelos con y sin `predict_proba` para compatibilidad máxima.
- La API arranca incluso si el modelo no existe (`modo degradado`), devolviendo HTTP 503 en `/predecir`.

---

### Sprint 2 — Carga/limpieza y partición robusta del dataset ⏳ PENDIENTE

**Meta:** Reemplazar la carga mínima del Sprint 1 con un pipeline de datos robusto que garantice reproducibilidad, manejo de valores atípicos, detección de desbalance de clases y particiones estratificadas reproducibles.

**Entregables requeridos:**

| Tarea | Estado |
|---|---|
| Descargar e integrar el CSV `diabetes_binary_health_indicators_BRFSS2015.csv` en `datos/brutos/` | ✅ |
| Ampliar `CargadorDatos` con análisis de distribución por columna | ⬜ |
| Implementar imputación de valores faltantes (mediana para BMI, moda para binarias) | ✅ |
| Detectar y documentar el desbalance de clase (`Diabetes_binary`) | ⬜ |
| Añadir soporte para sobremuestreo (SMOTE) o submuestreo en el cargador | ⬜ |
| Implementar partición con validación cruzada estratificada (`StratifiedKFold`) | ⬜ |
| Agregar prueba de integración de carga sobre el dataset real (parametrizada, omitida si no existe el CSV) | ⬜ |
| Guardar el dataset procesado en `datos/procesados/` en formato Parquet | ⬜ |

---

### Sprint 3 — Comparación de modelos y persistencia avanzada ⏳ PENDIENTE

**Meta:** Reemplazar el `DummyClassifier` por una suite de modelos reales, implementar evaluación rigurosa con métricas clínicas relevantes, y persistir los artefactos con versionado.

**Entregables requeridos:**

| Tarea | Estado |
|---|---|
| Agregar `LogisticRegression` con regularización L2 como baseline real | ⬜ |
| Agregar `RandomForestClassifier` con búsqueda de hiperparámetros (`GridSearchCV`) | ⬜ |
| Agregar `GradientBoostingClassifier` (o XGBoost si se añade como dependencia) | ⬜ |
| Calcular métricas completas: accuracy, precision, recall, F1, ROC-AUC, curva PR | ⬜ |
| Exportar reporte de métricas extendido en `reportes/metricas_entrenamiento.json` | ⬜ |
| Implementar versionado de artefactos: `modelo_diabetes_v{timestamp}.joblib` | ⬜ |
| Persistir el mejor modelo junto con el `ColumnTransformer`/`Pipeline` sklearn para evitar data leakage | ⬜ |
| Añadir pruebas que verifiquen que el mejor modelo supera un umbral mínimo de ROC-AUC (ej. > 0.75) | ⬜ |
| Generar reporte HTML o Markdown con tabla comparativa de modelos en `reportes/` | ⬜ |

---

### Sprint 4 — Inferencia y API productiva ⏳ PENDIENTE

**Meta:** Llevar la API del modo "contrato funcional" al modo "listo para producción": autenticación, contenerización, documentación interactiva y pruebas de carga.

**Entregables requeridos:**

| Tarea | Estado |
|---|---|
| Añadir autenticación básica por API Key (header `X-API-Key`) | ⬜ |
| Implementar rate limiting (ej. `slowapi`) para prevenir abuso | ⬜ |
| Añadir logging estructurado (JSON) con nivel configurable por variable de entorno | ⬜ |
| Crear `Dockerfile` multi-stage (builder + runtime mínimo) | ⬜ |
| Crear `docker-compose.yml` para levantar la API localmente con una sola instrucción | ⬜ |
| Exponer métricas Prometheus en `/metricas` (latencia p50/p95, predicciones por categoría) | ⬜ |
| Ampliar la validación de `DatosPaciente` con reglas clínicas adicionales | ⬜ |
| Añadir pruebas de integración end-to-end con el cliente HTTP real | ⬜ |
| Configurar OpenAPI (`/docs`) con ejemplos de request/response completos | ⬜ |

---

### Sprint 5 — Observabilidad, endurecimiento y documentación final ⏳ PENDIENTE

**Meta:** Asegurar que el sistema sea observable en producción, resistente a fallos y completamente documentado para equipo de operaciones y usuarios clínicos.

**Entregables requeridos:**

| Tarea | Estado |
|---|---|
| Configurar pipeline de CI/CD (GitHub Actions) con lint, pruebas y build de imagen | ⬜ |
| Añadir análisis estático de tipos con `mypy` o `pyright` | ⬜ |
| Implementar linting con `ruff` e integración en pre-commit hooks | ⬜ |
| Añadir monitoreo de drift de datos (detección de distribución fuera del rango de entrenamiento) | ⬜ |
| Documentar el proceso de re-entrenamiento y estrategia de actualización del modelo | ⬜ |
| Redactar guía de despliegue en producción (Cloud Run, ECS, o VPS con Docker) | ⬜ |
| Crear manual de usuario para personal clínico (interpretación de categorías y advertencias) | ⬜ |
| Auditoría de seguridad: revisión de dependencias (`pip audit`), cabeceras HTTP seguras | ⬜ |
| Revisión de equidad (fairness) del modelo por grupo de edad, sexo e ingreso | ⬜ |
| Etiqueta de versión estable `v1.0.0` en el repositorio | ⬜ |

---

## 4. Estado actual del proyecto

### Resumen ejecutivo

| Dimensión | Estado |
|---|---|
| Arquitectura modular | ✅ Definida y funcional |
| Capa de datos | ⚠️ Contrato definido; sin dataset real ni limpieza avanzada |
| Capa de modelos | ⚠️ Interfaz lista; modelo actual es un clasificador ficticio (`DummyClassifier`) |
| Capa de inferencia | ✅ Funcional con cualquier modelo `sklearn` compatible |
| API REST | ✅ Endpoints operativos con modo degradado y validación clínica |
| Pruebas automáticas | ✅ 6 pruebas de contrato (cargador, predictor, API) |
| Configuración de entorno | ✅ `pyproject.toml`, `.env.example`, `pyproject.toml` pytest |
| Dataset real integrado | ❌ Faltante (`datos/brutos/` vacío) |
| Modelo entrenado real | ❌ Faltante (`modelos/` vacío) |
| Docker / CI-CD | ❌ No implementado |
| Observabilidad producción | ❌ No implementado |

### Progreso por sprint

```
Sprint 1 ████████████████████ 100%  ✅ Completado
Sprint 2 ░░░░░░░░░░░░░░░░░░░░   0%  ⏳ Pendiente
Sprint 3 ░░░░░░░░░░░░░░░░░░░░   0%  ⏳ Pendiente
Sprint 4 ░░░░░░░░░░░░░░░░░░░░   0%  ⏳ Pendiente
Sprint 5 ░░░░░░░░░░░░░░░░░░░░   0%  ⏳ Pendiente
```

---

## 5. Pasos inmediatos (próximas acciones)

Los siguientes pasos son los más críticos para desbloquear el avance hacia el Sprint 2:

1. **Obtener el dataset CDC BRFSS 2015** — Descargar `diabetes_binary_health_indicators_BRFSS2015.csv` desde [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) y colocarlo en `datos/brutos/`. Sin él, no es posible entrenar ningún modelo real.

2. **Ampliar la limpieza de datos** — Extender `CargadorDatos.limpieza_basica()` con análisis de desbalance de clase, imputación y exportación a Parquet.

3. **Implementar modelos reales** — Reemplazar el `DummyClassifier` en `ComparadorModelos` con `LogisticRegression` y `RandomForestClassifier` como punto de partida.

4. **Ejecutar el pipeline de entrenamiento** y verificar que produce un `.joblib` válido que la API pueda consumir end-to-end.

5. **Configurar GitHub Actions** para que cada pull request ejecute `pytest` automáticamente.

---

## 6. Cómo ejecutar el proyecto localmente

```bash
# 1. Instalar dependencias (incluyendo dev)
python -m pip install -e .[dev]

# 2. Ejecutar pruebas
pytest

# 3. Entrenar el modelo (requiere dataset en datos/brutos/)
python -m entrenamiento.pipeline --modo clasificacion

# 4. Levantar la API
uvicorn api.main:app --reload

# 5. Probar el endpoint de salud
curl http://localhost:8000/salud

# 6. Probar una predicción
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "presion_alta": 1, "colesterol_alto": 1, "chequeo_colesterol": 1,
    "imc": 29.5, "fumador": 0, "derrame_cerebral": 0,
    "enfermedad_corazon": 0, "actividad_fisica": 1,
    "consume_fruta": 1, "consume_verdura": 1, "consumo_alcohol_alto": 0,
    "tiene_cobertura_medica": 1, "sin_medico_por_costo": 0,
    "salud_general": 2, "salud_mental": 2, "salud_fisica": 2,
    "dificultad_caminar": 0, "sexo": 1, "edad": 7,
    "educacion": 4, "ingreso": 6
  }'
```

---

## 7. Dependencias clave

| Paquete | Versión mínima | Rol |
|---|---|---|
| `fastapi` | 0.112.0 | Framework API REST |
| `uvicorn` | 0.30.0 | Servidor ASGI |
| `pydantic` | 2.8.0 | Validación y esquemas de datos |
| `pandas` | 2.2.0 | Manipulación del dataset CDC |
| `scikit-learn` | 1.5.0 | Modelos ML, partición, métricas |
| `joblib` | 1.4.0 | Serialización de modelos |
| `pytest` | 8.2.0 | Suite de pruebas (dev) |
| `httpx` | 0.27.0 | Cliente HTTP para pruebas de API (dev) |
