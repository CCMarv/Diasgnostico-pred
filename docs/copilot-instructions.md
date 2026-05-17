# PLAN DE EJECUCIÓN — `diasgnostico-pred`

<!--
INSTRUCCIONES PARA EL AGENTE
─────────────────────────────
1. LEE este archivo completo antes de ejecutar cualquier acción.
2. Sigue el orden de secciones sin omitir pasos.
3. Antes de marcar un ticket como terminado, completa el PROTOCOLO UNIVERSAL
   definido en la Sección 2.
4. Nunca declares un nivel de rúbrica completo sin ejecutar las validaciones
   de la Sección 4.
5. Si un comando falla, detente y reporta el error antes de continuar.
-->

---

## ÍNDICE DE NAVEGACIÓN

| Sección | Contenido | Cuándo leerla |
|---------|-----------|---------------|
| [Sección 1](#sección-1--estado-actual-del-proyecto) | Estado actual y gaps | SIEMPRE — antes de cualquier acción |
| [Sección 2](#sección-2--protocolo-universal-de-ticket) | Pasos obligatorios por ticket | Antes de ejecutar CUALQUIER ticket |
| [Sección 3](#sección-3--sprints-y-tickets) | Tickets por sprint | Al iniciar cada sprint |
| [Sección 4](#sección-4--protocolo-de-muestra-controlada) | Validación con datos reales | Antes de documentar resultados académicos |
| [Sección 5](#sección-5--protocolo-de-auditoría-de-documentación) | Auditoría de congruencia | Al cerrar Sprint 5 y antes de Sprint 6 |
| [Sección 6](#sección-6--ruta-crítica-y-desbloqueo-de-puntos) | Resumen de dependencias | Al planear el siguiente sprint |

---

## SECCIÓN 1 — ESTADO ACTUAL DEL PROYECTO

### 1.1 Mapa de Rúbrica

<!-- El agente usa esta tabla para saber qué existe, qué falta y dónde buscarlo. -->

| Ítem | Descripción | Estado | Evidencia en Código | Gap |
|------|-------------|--------|--------------------|----|
| B1 | Pipeline con ≥3 modelos | ✅ | `entrenamiento/comparador_modelos.py` | Ninguno |
| B2 | Preprocessing dentro del Pipeline | ✅ | `entrenamiento/preprocesador.py` | Ninguno |
| B3 | Métricas estándar | ✅ | `entrenamiento/evaluador.py` | Ninguno |
| I1 | SVM evaluado | 🟡 | Experimento 1000 muestras | Falta `optimizador.py` formal |
| I2 | Árbol evaluado | 🟡 | Experimento 1000 muestras | Misma dependencia |
| I3 | MLP evaluado | 🟡 | Experimento 1000 muestras | Misma dependencia |
| I4 | K-Means formal | ✅ | Implementado en `entrenamiento/fenotipado.py` | Pruebas en `pruebas/test_fenotipado.py` (3 passed) |
| I5 | GridSearchCV formal | ✅ | Implementado `OptimizadorHiperparametros` | `entrenamiento/optimizador.py`, pruebas en `pruebas/test_optimizador.py` (2 passed) |
| I6 | Dashboard Streamlit | ❌ | No existe | Falta `dashboard/app.py` |
| A1 | API en producción | 🟡 | `api/main.py` esqueleto funcional | Falta conexión con modelo real |
| A2 | Comparativa con papers | ❌ | No existe | Falta sección en `reporte_final.md` |

**Leyenda:** ✅ Completo · 🟡 Parcial · ❌ No existe

---

### 1.2 Gaps Documentales que Requieren Atención

<!-- El agente debe resolver estos gaps dentro de los sprints indicados. -->

| Archivo | Gap | Sprint Responsable |
|---------|-----|--------------------|
| `docs/evaluacion_academica.md` | Plantillas vacías (`[FECHA]`, `[0-100]`) | S3 y sucesivos |
| `reportes/contraste_regional.md` | Contenido parcial, sin tabla completa | S3-008 |
| `datos/procesados/dataset_procesado.parquet` | No confirmado en disco | S3-007 |
| `docs/preguntas_defensa.md` | No existe | S5-006 |
| `README_demo.md` | No existe | S5-005 |
| `reportes/reporte_final.md` | No existe | S5-001 a S5-004 |

---

### 1.3 Árbol de Dependencias Entre Módulos

```
config.py
└── entrenamiento/preprocesador.py
    └── entrenamiento/fenotipado.py        ← CREAR en S3-002
    └── entrenamiento/comparador_modelos.py
        └── entrenamiento/optimizador.py   ← CREAR en S3-004
        └── entrenamiento/evaluador.py
            └── entrenamiento/pipeline.py
                └── inferencia/predictor.py
                    └── api/main.py
                        └── api/esquemas.py
                └── dashboard/app.py       ← CREAR en S4-002
```

> **REGLA:** Si modificas un nodo, verifica que todos los nodos dependientes
> siguen pasando sus pruebas antes de continuar.

---

## SECCIÓN 2 — PROTOCOLO UNIVERSAL DE TICKET

<!--
OBLIGATORIO. El agente ejecuta estos 6 pasos para CADA ticket,
sin importar cuál sea. No hay excepciones.
-->

### Pasos en Orden Estricto

```
PASO 1 — LECTURA PREVIA
  Leer: docs/ROADMAP.md
  Leer: docs/evaluacion_academica.md
  Leer: el módulo que se va a modificar

PASO 2 — MUESTRA MÍNIMA DE ESTABILIDAD
  Ejecutar (ver Sección 4.1 para el comando exacto):
    python -m entrenamiento.pipeline --modo clasificacion --modelos gbm
  Criterio: completa sin error. El ROC-AUC no importa aquí.
  SI FALLA: detener y reportar. No continuar.

PASO 3 — IMPLEMENTACIÓN
  Aplicar solo el cambio mínimo necesario para el ticket.
  No agregar funcionalidad extra no solicitada.

PASO 4 — PRUEBA UNITARIA
  Crear o actualizar prueba en pruebas/test_[módulo].py
  Ejecutar: pytest pruebas/test_[módulo].py -v --tb=short
  Criterio: 0 fallos.
  SI FALLA: corregir antes de continuar.

PASO 5 — ACTUALIZACIÓN DE DOCUMENTACIÓN
  Actualizar docs/evaluacion_academica.md:
    - Cambiar estado del ítem de rúbrica afectado
    - Reemplazar cualquier [PLACEHOLDER] con valor real
    - Añadir fecha actual en sección de registro de cambios

PASO 6 — COMMIT
  Formato obligatorio:
    feat(ID_ITEM): descripción breve — ticket S[N]-[XXX]
  Ejemplo:
    feat(I4): implementa FenotipadoKMeans con silhouette_score — ticket S3-002
```

---

## SECCIÓN 3 — SPRINTS Y TICKETS

<!--
Los sprints son secuenciales. No iniciar Sprint N si Sprint N-1
tiene ítems de nivel (B, I, A) sin completar.
Los parciales 🟡 no bloquean el avance pero deben cerrarse dentro del
sprint indicado en la columna "Dependencias".
-->

---

### SPRINT 3 — K-Means, Optimizador y Cierre de Deuda Técnica

**Ítems de rúbrica cubiertos:** I4, I5 (parcial hacia Nivel Intermedio)
**Prerrequisito:** Sprint 2 validado ✅

---

#### Tickets del Sprint 3

| ID | Título | Descripción Técnica | Criterios de Aceptación | Dependencias |
|----|--------|--------------------|-----------------------|-------------|
| S3-001 | Verificación uniforme de métricas | Auditar `comparador_modelos.py`: confirmar que SVM, árbol, MLP y GBM producen las mismas 7 métricas en la tabla comparativa | `comparativa_modelos.md` contiene 7 métricas para los 4 modelos; pytest pasa | Ninguna |
| S3-002 | Crear `fenotipado.py` | Implementar clase `FenotipadoKMeans` con `ajustar()`, `predecir_fenotipo()`, `graficar_codo()` y `calcular_silhouette()`; usar `init='k-means++'`, `n_init='auto'`, `random_state=42` | Clase importable; `silhouette_score > 0` en datos sintéticos; PNG generado en `reportes/` | S3-001 |
| S3-003 | Pruebas de `fenotipado.py` | Crear `pruebas/test_fenotipado.py` con 3 pruebas: (a) silhouette > 0, (b) len(etiquetas) == len(X), (c) modelo serializable con joblib | `pytest pruebas/test_fenotipado.py -v` = 0 fallos | S3-002 |
| S3-004 | Crear `optimizador.py` | Extraer lógica de `GridSearchCV` + `StratifiedKFold` a clase `OptimizadorHiperparametros` con métodos `buscar()` y `mejor_pipeline()` | `from entrenamiento.optimizador import OptimizadorHiperparametros` funciona; no rompe pruebas existentes | S3-001 |
| S3-005 | Pruebas de `optimizador.py` | Crear `pruebas/test_optimizador.py` con 2 pruebas: (a) búsqueda retorna estimador, (b) `best_params_` no vacío | `pytest pruebas/test_optimizador.py -v` = 0 fallos | S3-004 |
| S3-006 | Notebook de fenotipado | Crear `notebooks/02_fenotipado_kmeans.ipynb` con: carga de datos, elbow plot, silhouette por K, visualización de clusters, interpretación clínica de centros en español | Ejecutable de inicio a fin sin errores; ≥2 figuras generadas en `reportes/` | S3-002 |
| S3-007 | Cerrar S2-09: persistir Parquet | Confirmar que `pipeline.py` ejecuta `persistir_procesado()` correctamente y genera `datos/procesados/dataset_procesado.parquet` | Archivo existe; `pd.read_parquet()` retorna DataFrame con columnas CDC; tamaño > 0 | S3-001 |
| S3-008 | Cerrar S2-10: contraste regional | Completar `reportes/contraste_regional.md` con tabla CDC vs ENSANUT (≥7 variables), sesgo relativo por variable y recomendación de umbral para contexto mexicano | Tabla presente; sección de recomendaciones clínicas ≥150 palabras | S3-001 |
| S3-009 | Muestra de validación 2000 registros | Ejecutar pipeline completo con 2000 registros estratificados; registrar resultados en `reportes/comparativa_2000_s3.md` | 4 modelos evaluados; ROC-AUC ≥ 0.70 en al menos 1 modelo; prevalencia de muestra = 0.139 ± 0.02 | S3-007 |
| S3-010 | Actualizar `evaluacion_academica.md` Sprint 3 | Completar todas las celdas `[PLACEHOLDER]` en secciones B e I1–I5; marcar I4 e I5 como completados con fecha real | Sin celdas `[FECHA]` o `[0-100]` vacías en secciones afectadas | S3-003, S3-005 |
| DOC-001 | Docstrings módulos nuevos | Cada función pública en `fenotipado.py` y `optimizador.py` debe tener docstring en español con `Args`, `Returns` y referencia al ítem de rúbrica | `pydoc entrenamiento.fenotipado` muestra documentación legible sin conocer Python | S3-003, S3-005 |

---

#### Instrucciones de Implementación — S3-002 (`fenotipado.py`)

```
ANTES DE ESCRIBIR CÓDIGO:
  Leer: entrenamiento/preprocesador.py  (entender ConstructorPreprocesador)
  Leer: config.py                       (entender COLUMNAS_CDC, SEMILLA_ALEATORIA)
  Leer: docs/implementacion_modelos/modelos/kmeans-especifico.md

ESTRUCTURA OBLIGATORIA DE LA CLASE:

  class FenotipadoKMeans:
    __init__(self, n_clusters_max=10, random_state=42)
    ajustar(self, X: pd.DataFrame) -> FenotipadoKMeans
    predecir_fenotipo(self, X: pd.DataFrame) -> np.ndarray
    graficar_codo(self, ruta_salida: Path | None = None) -> None
    calcular_silhouette(self, X: pd.DataFrame) -> float

REGLAS TÉCNICAS:
  - NO incluir atributos lambda (rompe serialización con joblib)
  - Docstring de clase debe explicar en lenguaje llano qué es fenotipado
  - Referencia a ítem I4 en el docstring del módulo
  - random_state siempre desde config.SEMILLA_ALEATORIA

ANTI-PATRÓN A EVITAR:
  ❌  kmeans.fit(X_completo)          # data leakage si X_completo incluye test
  ✅  kmeans.fit(X_train_escalado)    # solo datos de entrenamiento
```

---

#### Instrucciones de Implementación — S3-004 (`optimizador.py`)

```
ANTES DE ESCRIBIR CÓDIGO:
  Leer: entrenamiento/comparador_modelos.py  (identificar _entrenar_con_grid)
  Leer: docs/implementacion_modelos/modelos/svm-especifico.md

ESTRUCTURA OBLIGATORIA DE LA CLASE:

  class OptimizadorHiperparametros:
    __init__(self, cv_splits=5, scoring='roc_auc', n_jobs=-1)
    buscar(self, pipeline, param_grid, X_train, y_train) -> GridSearchCV
    mejor_pipeline(self) -> Pipeline

REGLAS TÉCNICAS:
  - NO eliminar la lógica existente en comparador_modelos.py
  - El optimizador es ADICIONAL, no un reemplazo
  - StratifiedKFold siempre con shuffle=True y random_state=SEMILLA_ALEATORIA
  - Referencia a ítem I5 en el docstring del módulo
```

---

#### Validación de Cierre — Sprint 3

```bash
# Ejecutar en este orden exacto
pytest pruebas/test_fenotipado.py -v --tb=short      # I4
pytest pruebas/test_optimizador.py -v --tb=short      # I5
pytest pruebas/test_preprocesador.py -v --tb=short    # Regresión B2
pytest pruebas/ --cov=entrenamiento --cov-report=term-missing

# Criterio de cierre: 0 fallos en todos los comandos anteriores
```

---

### SPRINT 4 — Dashboard Interactivo

**Ítems de rúbrica cubiertos:** I6 → **desbloquea +15 puntos (Nivel Intermedio)**
**Prerrequisito:** Sprint 3 completo ✅ y `modelos/modelo_diabetes_v1.joblib` disponible

---

#### Tickets del Sprint 4

| ID | Título | Descripción Técnica | Criterios de Aceptación | Dependencias |
|----|--------|--------------------|-----------------------|-------------|
| S4-001 | Agregar Streamlit al proyecto | Añadir `streamlit` a `pyproject.toml` en grupo `[project.optional-dependencies]` como `dashboard` | `pip install -e .[dashboard]` funciona; `streamlit --version` retorna ≥1.30 | Sprint 3 ✅ |
| S4-002 | Estructura base del dashboard | Crear `dashboard/__init__.py` vacío y `dashboard/app.py` con sidebar de 3 vistas; cargar pipeline con `@st.cache_resource`; manejar modo degradado sin modelo | `streamlit run dashboard/app.py` levanta en puerto 8501 sin error; sidebar muestra 3 vistas | S4-001 |
| S4-003 | Vista "Comparativa de Modelos" | Leer `reportes/comparativa_modelos.md` o JSON de métricas; mostrar tabla con `st.dataframe`; gráfica de barras de ROC-AUC por modelo | Vista renderiza tabla con 4 modelos y 7 métricas; gráfica visible; no requiere reentrenamiento | S4-002 |
| S4-004 | Vista "Predicción Individual" | Formulario con los 21 campos de `DatosPaciente`; llamar `PredictorDiabetes.predecir()` directamente (sin HTTP); mostrar categoría de riesgo, confianza y advertencia clínica | Formulario acepta todos los campos; resultado visible; maneja errores de validación clínica con mensaje en español | S4-002 |
| S4-005 | Vista "Fenotipos K-Means" | Cargar modelo K-Means serializado; mostrar elbow plot, silhouette por K y tabla de centros de clusters con interpretación clínica en español | Vista renderiza ≥2 figuras y ≥1 tabla; interpretación clínica visible sin conocimientos de ML | S4-002, S3-002 |
| S4-006 | Validación manual del dashboard | Ejecutar checklist: (a) comparativa carga tabla, (b) predicción retorna resultado, (c) fenotipos renderiza gráficas, (d) modo degradado no crashea | Checklist documentado en `docs/evaluacion_academica.md` sección I6 con fecha | S4-003, S4-004, S4-005 |
| DOC-002 | Textos de ayuda en el dashboard | Cada vista debe tener `st.info()` explicando qué hace la vista y `st.caption()` explicando cada resultado | Alguien sin conocimientos de ML puede usar las 3 vistas siguiendo solo los textos del dashboard | S4-006 |

---

#### Instrucciones de Implementación — S4-002 (estructura base)

```
ESTRUCTURA OBLIGATORIA DE dashboard/app.py:

  import streamlit as st
  from pathlib import Path
  import joblib
  from inferencia.predictor import PredictorDiabetes
  from config import RUTA_MODELO_FINAL

  # Carga única del modelo (no recargar en cada interacción)
  @st.cache_resource
  def cargar_predictor():
      predictor = PredictorDiabetes()
      predictor.cargar_modelo()
      return predictor

  # Navegación principal
  vista = st.sidebar.selectbox(
      "Selecciona una vista",
      ["Comparativa de Modelos", "Predicción Individual", "Fenotipos"]
  )

  # Modo degradado
  predictor = cargar_predictor()
  if not predictor.esta_listo():
      st.warning("Modelo no disponible. Ejecuta el pipeline de entrenamiento primero.")

REGLAS TÉCNICAS:
  - NO usar st.experimental_* (deprecado)
  - Importar desde config.py, nunca hardcodear rutas
  - Cada vista en función separada para claridad
```

---

#### Validación de Cierre — Sprint 4

```bash
# Verificación funcional (manual obligatoria)
streamlit run dashboard/app.py

# Checklist a documentar en evaluacion_academica.md:
# [ ] Vista "Comparativa" carga y muestra tabla de 4 modelos
# [ ] Vista "Predicción" acepta input y devuelve resultado con categoría de riesgo
# [ ] Vista "Fenotipos" renderiza gráfica de clusters
# [ ] Con modelo inexistente: muestra advertencia, no crashea

# Criterio de nivel: I6 ✅ → Nivel Intermedio COMPLETADO (+15)
```

---

### SPRINT 5 — Reporte Académico y Preparación de Defensa

**Componentes de rúbrica:** Reporte 20% + Presentación 10%
**Prerrequisito:** Sprint 4 completo ✅

> ⚠️ **CONDICIONAL ESTRICTO:** Los tickets S5-003 y S5-004 (resultados y
> conclusiones con números reales) solo pueden ejecutarse **después** de que
> el ticket S3-009 haya pasado exitosamente. Ver Sección 4 para el protocolo.

---

#### Tickets del Sprint 5

| ID | Título | Descripción Técnica | Criterios de Aceptación | Dependencias |
|----|--------|--------------------|-----------------------|-------------|
| S5-001 | Sección Introducción del reporte | Redactar en `reportes/reporte_final.md`: planteamiento del problema, justificación del dataset CDC, hipótesis de transferibilidad ENSANUT, objetivos del sistema | ≥400 palabras; menciona prevalencia de diabetes en México; referencia a ENSANUT 2022 | Sprint 4 ✅ |
| S5-002 | Sección Metodología del reporte | Documentar: pipeline de preprocesamiento, justificación de cada modelo, manejo del desbalance, validación cruzada, umbrales de riesgo | Cada decisión técnica justificada en lenguaje accesible; diagrama de arquitectura incluido | S5-001 |
| S5-003 | Sección Resultados del reporte | Tabla comparativa de 4 modelos con 7 métricas reales; curvas ROC y PR del mejor modelo; comparativa con benchmarks de `PROYECTO.md §7` | Valores reales (no placeholders); figuras referenciadas; ROC-AUC comparado con Huang et al. 2022 | S5-002, **S3-009 ✅** |
| S5-004 | Sección Conclusiones y Limitaciones | Documentar: limitaciones de transferibilidad CDC→México, umbral recomendado para IMSS (0.25), trabajo futuro | Menciona las 3 limitaciones de `PROYECTO.md §6.2`; recomendaciones accionables | S5-003 |
| S5-005 | Crear `README_demo.md` | Documento de una página con comandos exactos para: instalar, entrenar, levantar API y dashboard; incluir salida esperada de cada comando | Alguien sin contexto puede ejecutar el demo en ≤10 minutos siguiendo solo este archivo | S4-006 |
| S5-006 | Crear `docs/preguntas_defensa.md` | Respuestas preparadas para las 7 preguntas de `evaluacion_academica.md §7`; cada respuesta referencia evidencia concreta con ruta de archivo | ≥7 preguntas respondidas; cada respuesta incluye ruta de archivo como evidencia | S5-004 |
| S5-007 | Verificación final de suite completa | Ejecutar `pytest pruebas/ -v --tb=short`; confirmar 0 fallos; generar cobertura con `--cov=entrenamiento` | 0 fallos; cobertura ≥70% en módulos de entrenamiento | Todos los S3 y S4 |
| DOC-003 | Actualizar `evaluacion_academica.md` Sprint 5 | Completar secciones 3.3 (Reporte) y 3.4 (Presentación) con calificaciones estimadas y evidencia real | Sin celdas vacías en secciones 3.3 y 3.4; calificaciones con justificación | S5-007 |

---

#### Validación de Cierre — Sprint 5

```bash
# Suite completa
pytest pruebas/ -v --tb=short

# Dashboard funcional
streamlit run dashboard/app.py

# Verificar que reporte_final.md no tiene placeholders
grep -n "\[PLACEHOLDER\]\|\[FECHA\]\|\[0-100\]" reportes/reporte_final.md
# Esperado: 0 resultados
```

---

### SPRINT 6 — API en Producción y Comparativa con Papers

**Ítems de rúbrica cubiertos:** A1, A2 → **desbloquea +30 puntos (Nivel Avanzado)**
**Prerrequisito:** Sprint 5 completo ✅ y auditoría AUD-001 aprobada

---

#### Tickets del Sprint 6

| ID | Título | Descripción Técnica | Criterios de Aceptación | Dependencias |
|----|--------|--------------------|-----------------------|-------------|
| S6-001 | Conectar API con pipeline serializado | Verificar que `api/main.py` carga el `.joblib` correcto en `lifespan`; confirmar que `predictor.esta_listo()` retorna `True` cuando el modelo existe | `GET /salud` retorna `estado: operativo` cuando existe el `.joblib` | Sprint 5 ✅ |
| S6-002 | Prueba de integración `/predecir` | Actualizar `pruebas/test_api.py`: prueba con modelo serializado temporal; verificar que HTTP 200 incluye `categoria_riesgo` válida | `pytest pruebas/test_api.py -v` = 0 fallos; prueba con modelo real retorna HTTP 200 | S6-001 |
| S6-003 | Prueba de degradación controlada | Verificar HTTP 503 en `/predecir` sin modelo; `GET /salud` retorna `estado: degradado`; logs muestran advertencia | Pruebas de degradación pasan; comportamiento documentado | S6-001 |
| S6-004 | Documentar arranque de API | Añadir sección "Levantar la API" en `README_demo.md` con comando exacto, ejemplo de `curl` y salida esperada | Sección presente; ejemplo de curl copiable y funcional | S6-001 |
| S6-005 | Identificar papers para comparativa | Seleccionar 2 papers de `PROYECTO.md §7`; documentar metodología y métricas reportadas | Lista en `reportes/reporte_final.md` con: título, año, dataset, modelo, ROC-AUC | S5-004 |
| S6-006 | Sección comparativa en reporte | Tabla comparativa entre este proyecto y los 2 papers; párrafo de discusión | Tabla con ≥3 columnas y ≥3 métricas; párrafo ≥200 palabras | S6-005, S5-003 |
| S6-007 | Actualizar `evaluacion_academica.md` final | Marcar A1 y A2 como ✅; declarar Nivel Avanzado COMPLETADO | Sección Avanzado = "COMPLETADO (+30)"; sin celdas vacías | S6-006 |

---

#### Instrucciones de Implementación — S6-001 (conexión API)

```
VERIFICACIÓN ANTES DE CAMBIOS:
  python -c "from api.main import app; print('OK')"
  pytest pruebas/test_api.py -v

VERIFICAR EN api/main.py:
  1. El lifespan carga RUTA_MODELO_FINAL (definida en config.py)
  2. Si el archivo no existe, app.state.predictor.esta_listo() = False
  3. El endpoint /salud lee el estado del predictor correctamente

ANTI-PATRÓN A EVITAR:
  ❌  ruta_modelo = "modelos/modelo.joblib"     # magic string
  ✅  ruta_modelo = ConfiguracionRutas.RUTA_MODELO  # desde config.py
```

---

#### Validación de Cierre — Sprint 6

```bash
# Levantar API
uvicorn api.main:app --reload

# En otra terminal
curl http://localhost:8000/salud
# Esperado: {"estado": "operativo", ...}

curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"presion_alta":1,"colesterol_alto":1,"chequeo_colesterol":1,"imc":29.5,
       "fumador":0,"derrame_cerebral":0,"enfermedad_corazon":0,"actividad_fisica":1,
       "consume_fruta":1,"consume_verdura":1,"consumo_alcohol_alto":0,
       "tiene_cobertura_medica":1,"sin_medico_por_costo":0,"salud_general":2,
       "salud_mental":2,"salud_fisica":2,"dificultad_caminar":0,"sexo":1,
       "edad":7,"educacion":4,"ingreso":6}'
# Esperado: {"categoria_riesgo": "bajo"|"medio"|"alto", ...}

pytest pruebas/test_api.py -v

# Criterio de nivel: A1 ✅ A2 ✅ → Nivel Avanzado COMPLETADO (+30)
```

---

## SECCIÓN 4 — PROTOCOLO DE MUESTRA CONTROLADA

<!--
Este protocolo es CONDICIONAL. El agente solo puede documentar resultados
académicos definitivos cuando se cumplan TODAS las condiciones de esta sección.
-->

### 4.1 Condiciones de Desbloqueo para Documentación Académica

> ⛔ **BLOQUEADO** hasta que se verifiquen estas 4 condiciones:

| Condición | Verificación | Ticket Responsable |
|-----------|-------------|-------------------|
| Muestra ≥2000 registros procesada | `reportes/comparativa_2000_s3.md` existe con 4 modelos | S3-009 |
| Muestra es representativa | Prevalencia de clase positiva = 0.139 ± 0.02 | S3-009 |
| Suite de pruebas en verde | `pytest pruebas/ -q` = 0 errores | S5-007 |
| Modelo serializado cargable | `PredictorDiabetes().cargar_modelo()` = `True` | S6-001 |

---

### 4.2 Paso A — Muestra Mínima de Estabilidad (ejecutar al iniciar cada sprint)

```bash
# Propósito: verificar que el sistema no está roto antes de cualquier cambio
# Tamaño: ≤100 registros (velocidad, no representatividad)

python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos gbm \
  --salida-modelo modelos/modelo_prueba_minima.joblib \
  --salida-reporte reportes/prueba_minima.json

# CRITERIO: completa sin error.
# El ROC-AUC NO importa en esta muestra por su tamaño.
# SI FALLA: detener y reportar. No ejecutar ningún ticket.
```

---

### 4.3 Paso B — Muestra de Validación Académica (ejecutar en S3-009)

```bash
# Propósito: generar resultados cuantitativos defendibles académicamente
# Tamaño: exactamente 2000 registros, estratificados por clase objetivo

python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos svm,arbol,gbm,mlp \
  --salida-reporte reportes/comparativa_2000_s3.md

# CRITERIO DE ÉXITO:
# - 4 modelos evaluados sin error
# - ROC-AUC ≥ 0.70 en al menos 1 modelo
# - Archivo de reporte generado con timestamp real
```

---

### 4.4 Paso C — Verificación de Representatividad Estadística

```python
# Fragmento a ejecutar después del Paso B para confirmar representatividad.
# Si el assert falla, la muestra no es válida para documentación académica.

import pandas as pd

df_muestra = pd.read_csv("datos/brutos/muestra_2000.csv")
prevalencia = df_muestra["Diabetes_binary"].mean()

assert 0.119 <= prevalencia <= 0.159, (
    f"Muestra NO representativa: prevalencia={prevalencia:.3f}. "
    f"Esperado: 0.139 ± 0.020. "
    f"Re-generar muestra con estratificación."
)

print(f"✅ Muestra representativa: prevalencia={prevalencia:.3f}")
```

---

## SECCIÓN 5 — PROTOCOLO DE AUDITORÍA DE DOCUMENTACIÓN

<!--
Ejecutar al cerrar Sprint 5 y antes de iniciar Sprint 6.
La auditoría tiene dos dimensiones: congruencia técnica y claridad para no programadores.
-->

### 5.1 Auditoría AUD-001 — Congruencia Código-Documentación

Cada afirmación en la documentación debe estar respaldada por código ejecutable real.

| ID Check | Afirmación Documentada | Verificación Ejecutable | Archivo de Evidencia |
|----------|----------------------|------------------------|---------------------|
| AUD-001-A | B1: Pipeline con ≥3 modelos ✅ | `grep -n "svm\|arbol\|gbm\|mlp" entrenamiento/comparador_modelos.py` retorna ≥4 hits | `entrenamiento/comparador_modelos.py` |
| AUD-001-B | B2: Preprocessing dentro del Pipeline ✅ | `from entrenamiento.preprocesador import ConstructorPreprocesador` importa sin error | `entrenamiento/preprocesador.py` |
| AUD-001-C | B3: Métricas estándar ✅ | `pytest pruebas/test_preprocesador.py -v` = 0 fallos | `pruebas/test_preprocesador.py` |
| AUD-001-D | I4: K-Means con `silhouette_score` | `grep -n "silhouette_score" entrenamiento/fenotipado.py` retorna ≥1 hit | `entrenamiento/fenotipado.py` |
| AUD-001-E | I5: `GridSearchCV` + `StratifiedKFold` | `grep -n "GridSearchCV\|StratifiedKFold" entrenamiento/optimizador.py` retorna ≥2 hits | `entrenamiento/optimizador.py` |
| AUD-001-F | Pipeline serializado como objeto completo | `joblib.load("modelos/modelo_diabetes_v1.joblib")` tiene atributo `named_steps` | `modelos/modelo_diabetes_v1.joblib` |
| AUD-001-G | API modo degradado funciona | `pytest pruebas/test_api.py::test_salud_degradado_sin_modelo -v` = OK | `pruebas/test_api.py` |
| AUD-001-H | Umbrales en `config.py` coinciden con `api/main.py` | `UMBRAL_RIESGO_BAJO == 0.33` y `UMBRAL_RIESGO_ALTO == 0.66` en ambos archivos | `config.py`, `api/main.py` |

**Criterio de aprobación AUD-001:** Los 8 checks pasan sin discrepancias.

---

### 5.2 Auditoría AUD-002 — Claridad para Stakeholders No Técnicos

Verificar que la documentación es legible para académicos y estudiantes sin experiencia en programación.

| Criterio | Qué Revisar | Archivo |
|----------|------------|---------|
| Propósito en español natural | Primer párrafo del docstring de cada clase es comprensible sin conocer Python | Todos los `.py` en `entrenamiento/` |
| Términos técnicos definidos | Cada término (ROC-AUC, Pipeline, SMOTE) aparece definido la primera vez que se usa | `reportes/reporte_final.md` |
| Tablas con interpretación textual | Cada tabla va seguida de 1-2 oraciones explicando qué significan los números | `reportes/reporte_final.md` |
| Dashboard con textos de ayuda | `st.info()` o `st.caption()` presente en las 3 vistas | `dashboard/app.py` |
| Comandos copiables y completos | Ningún bloque de código tiene `[...]` ni partes omitidas | `README.md`, `README_demo.md` |

**Criterio de aprobación AUD-002:** Los 5 criterios verificados manualmente.

---

### 5.3 Auditoría AUD-003 — Revisión Final de `evaluacion_academica.md`

```bash
# Verificar que no quedan placeholders vacíos
grep -n "\[FECHA\]\|\[0-100\]\|\[PLACEHOLDER\]\|EVIDENCE_NOT_FOUND" \
  docs/evaluacion_academica.md

# Esperado antes de presentación: 0 resultados
# Si hay resultados: actualizar con valores reales antes de presentar
```

---

## SECCIÓN 6 — RUTA CRÍTICA Y DESBLOQUEO DE PUNTOS

### 6.1 Diagrama de Dependencias de Sprint

```
S1 ✅ → S2 ✅
              ↓
         S3 (activo)
         ├── S3-001 → S3-002 → S3-003 (I4)
         ├── S3-001 → S3-004 → S3-005 (I5)
         ├── S3-007 → S3-009 ← DESBLOQUEA documentación con números reales
         └── S3-010 → DOC-001
              ↓
         S4 (Dashboard)
         ├── S4-001 → S4-002
         └── S4-002 → S4-003 + S4-004 + S4-005 (paralelo)
                   → S4-006 → DOC-002
              ↓
         S5 (Reporte)
         ├── S5-001 → S5-002 → S5-003* → S5-004
         ├── S5-005 + S5-006 (paralelo tras S5-004)
         └── S5-007 → AUD-001 + AUD-002 → DOC-003
              ↓
         S6 (API + Papers)
         ├── S6-001 → S6-002 + S6-003 (paralelo)
         ├── S6-005 → S6-006
         └── S6-007 → AUD-003

* S5-003 bloqueado hasta S3-009 ✅
```

---

### 6.2 Tabla de Puntos por Sprint

| Al Cerrar | Nivel Desbloqueado | Puntos Extra | Requisito No Negociable |
|-----------|-------------------|-------------|------------------------|
| Sprint 3 | Básico consolidado | +0 | `pytest pruebas/ -q` = 0 fallos |
| Sprint 4 | **Nivel Intermedio** | **+15** | `streamlit run dashboard/app.py` funciona |
| Sprint 5 | Intermedio consolidado | +15 | Reporte con números reales post-muestra 2000 |
| Sprint 6 | **Nivel Avanzado** | **+30** | API responde `/salud` y `/predecir` en producción |

---

### 6.3 Reglas de Secuencialidad — No Negociables

```
REGLA 1: No iniciar Sprint N si Sprint N-1 tiene ítems de nivel sin completar.
REGLA 2: Los parciales 🟡 no bloquean avance, pero deben cerrarse en el sprint indicado.
REGLA 3: No documentar resultados académicos definitivos sin S3-009 aprobado.
REGLA 4: No declarar Nivel Intermedio sin I6 (dashboard) verificado manualmente.
REGLA 5: No declarar Nivel Avanzado sin AUD-001 aprobado.
```

---

## APÉNDICE — ANTI-PATRONES PROHIBIDOS

<!--
El agente NUNCA debe implementar estos patrones.
Si encuentra código existente con estos patrones, reportarlo como deuda técnica.
-->

```python
# ❌ PROHIBIDO: fit sobre datos de prueba
preprocesador.fit(X_test)

# ❌ PROHIBIDO: fit sobre el conjunto completo antes de dividir
preprocesador.fit(X)
X_train, X_test = train_test_split(X_transformed)

# ❌ PROHIBIDO: accuracy como métrica principal con desbalance
print(f"Score: {accuracy_score(y_test, y_pred)}")  # sin contexto de ROC-AUC

# ❌ PROHIBIDO: magic strings de rutas
ruta = "modelos/modelo.joblib"  # debe venir de config.py

# ❌ PROHIBIDO: modificar firmas públicas sin confirmar con el usuario
# Firmas protegidas: predecir(), a_dataframe(), cargar_modelo(), esta_listo()

# ❌ PROHIBIDO: avanzar con pruebas fallidas
# Si pytest retorna > 0 fallos: detener y corregir antes de continuar
```

---

## APÉNDICE — FORMATO DE COMMIT

```
# Estructura obligatoria
tipo(ID_ITEM): descripción breve — ticket S[N]-[XXX]

# Ejemplos válidos
feat(I4): implementa FenotipadoKMeans con silhouette_score — ticket S3-002
feat(I5): crea OptimizadorHiperparametros con GridSearchCV — ticket S3-004
feat(I6): añade vista de predicción individual al dashboard — ticket S4-004
feat(A1): conecta pipeline serializado al endpoint /predecir — ticket S6-001
docs: actualiza evaluacion_academica.md — Nivel Intermedio COMPLETADO +15
test(I4): añade pruebas de fenotipado con datos sintéticos — ticket S3-003
fix(B2): corrige data leakage en ColumnTransformer — ticket S3-001
```

---

*Fin del plan — versión 1.0 · 2026-05-17*