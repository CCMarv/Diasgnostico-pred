# Diasgnostico-pred — Hoja de Ruta Estratégica  
## Parte 2: Sprint 3–4 · Recomendaciones de Alto Impacto

> **Documento:** Parte 2 de 2  
> **Continúa de:** ROADMAP_PARTE1.md  
> **Estado actualizado (2026-05-12):** Sprint 3 y Sprint 4 no iniciados; se mantiene como plan de ejecución posterior a cerrar pendientes de datos reales en Sprint 2.

---

## SECCIÓN 3 — Sprint 3: Fenotipado Metabólico con K-Means & Dashboard

### 3.1 Objetivo del Sprint

Reencuadrar K-Means como herramienta de **fenotipado clínico previo a la clasificación** (no como predictor de diabetes), construir el Dashboard Streamlit orientado al médico general, y cerrar la brecha de la rúbrica de nivel Intermedio completo.

---

### 3.2 Estrategia K-Means: Fenotipos de Riesgo Metabólico

#### Por qué K-Means no debe predecir diabetes directamente

Un evaluador académico penalizará usar K-Means como clasificador de diabetes porque:
1. K-Means no optimiza la separación de clases; optimiza la inercia intra-cluster
2. La variable objetivo `Diabetes_binary` crea un sesgo de confirmación si se incluye en el clustering
3. La literatura clínica sí usa clustering para **descubrir fenotipos** (ej. "diabético obeso sedentario" vs. "diabético delgado con comorbilidad cardiovascular")

#### Marco de uso correcto: Fenotipado Pre-Clasificación

```
Flujo recomendado:

  X_train (sin y_train)
       │
       ▼
  Selección de variables metabólicas:
    [BMI, GenHlth, PhysHlth, MentHlth, Age, PhysActivity,
     HighBP, HighChol, HeartDiseaseorAttack]
  (variables con mayor carga metabólica según EDA)
       │
       ▼
  StandardScaler() → K-Means(n_clusters=K*)
       │
       ▼
  Asignar etiqueta de fenotipo: fenotipo_0, fenotipo_1, ..., fenotipo_K-1
       │
       ▼
  Enriquecer X_train con columna 'fenotipo' (como feature adicional)
       │
       ▼
  Pipeline supervisado (SVM / GBM / MLP) con 'fenotipo' como entrada adicional
```

#### Selección del K óptimo (`entrenamiento/fenotipador.py`)

```
class FenotipadorMetabolico:
    """
    Encuentra grupos de pacientes según perfil metabólico sin usar la variable objetivo.
    
    Métodos:
    
      seleccionar_k(X, k_min=2, k_max=8) -> int
        → Método del codo: graficar inercia vs K → guardar reportes/codo_kmeans.png
        → Silhouette Score: seleccionar K con mayor coeficiente promedio
        → Criterio final: K que maximiza separación clínica interpretable
        → Rango típico en estudios de fenotipos metabólicos: K=3 o K=4
      
      ajustar(X, k) -> None
        → Ajusta KMeans, persiste modelo en modelos/fenotipador.joblib
      
      asignar_fenotipos(X) -> np.ndarray
        → Retorna array de etiquetas de cluster
      
      perfilar_fenotipos(X, y) -> pd.DataFrame
        → Tabla: para cada cluster, media de cada variable + prevalencia de diabetes
        → Permite nombrar clínicamente los clusters:
            Ejemplo: cluster_2: BMI=34.1, HighBP=0.78, PhysActivity=0.12 → "Fenotipo Obeso Sedentario"
    
    Integración con pipeline supervisado:
      → Añadir como etapa 0 del Pipeline sklearn usando FunctionTransformer o custom Transformer
      → El fenotipo es una feature categórica → OrdinalEncoder en ColumnTransformer
    """
```

#### Nomenclatura clínica de fenotipos (ejemplo con K=3)

Para el reporte académico, los clusters deben tener nombres clínicamente interpretables:

| Fenotipo | Perfil esperado (hipótesis) | Nombre sugerido | Prevalencia diabetes esperada |
|---|---|---|---|
| Fenotipo A | IMC elevado, HTA, sedentario, edad media | "Metabólico Clásico" | ~25–35% |
| Fenotipo B | IMC normal/bajo, sin HTA, activo, joven | "Bajo Riesgo General" | ~5–10% |
| Fenotipo C | Mayor edad, múltiples comorbilidades, peor salud autorreportada | "Multimórbido Avanzado" | ~35–50% |

**Nota:** los nombres reales emergerán del análisis de `perfilar_fenotipos()`. Esta tabla es la **hipótesis a contrastar** en el reporte.

---

### 3.3 Arquitectura del Dashboard Streamlit (`dashboard/app.py`)

El dashboard debe simular la interfaz de una consulta de medicina preventiva del IMSS. El médico general es el usuario principal, no el paciente.

#### Estructura de pantallas

**Pantalla 1 — Captura del paciente (formulario de tamizaje)**

```
Título: "PrevenIMSS Digital — Tamizaje de Riesgo de Diabetes"

Diseño: dos columnas
  Columna izquierda — Datos clínicos medidos en consultorio:
    - IMC (slider numérico, rango 10–50, paso 0.5)
    - Presión arterial alta (checkbox "¿Diagnóstico previo de HTA?")
    - Colesterol alto (checkbox "¿Diagnóstico previo de dislipidemia?")
    - Salud general (selectbox: Excelente/Muy buena/Buena/Regular/Mala)
    - Días con mala salud física en el último mes (slider 0–30)
    - Dificultad para caminar (checkbox)
  
  Columna derecha — Hábitos y antecedentes:
    - Actividad física (checkbox "¿Realiza actividad física regularmente?")
    - Consumo de fruta diario (checkbox)
    - Consumo de verdura diario (checkbox)
    - Fumador (checkbox "¿Ha fumado ≥100 cigarrillos en su vida?")
    - Cardiopatía o infarto previo (checkbox)
    - Derrame cerebral previo (checkbox)
  
  Sección inferior — Perfil sociodemográfico:
    - Sexo (radio button)
    - Grupo de edad (selectbox con etiquetas en años, no códigos numéricos CDC)
    - Escolaridad (selectbox)
    - Cobertura médica (checkbox)
    
Botón: "Calcular Riesgo" (llama a POST /predecir)
```

**Pantalla 2 — Resultado de la predicción**

```
Panel principal:
  - Gauge/medidor circular: probabilidad 0–100%
    → Verde: bajo riesgo (<33%)
    → Amarillo: riesgo medio (33–66%)
    → Rojo: riesgo alto (>66%)
  
  - Etiqueta grande: "RIESGO BAJO / MEDIO / ALTO"
  - Advertencia clínica (si existe): banner amarillo con ícono ⚠️
  - Fenotipo metabólico asignado: "Perfil: Metabólico Clásico"

Panel secundario (expandible):
  - Gráfica SHAP waterfall: "¿Por qué este resultado?"
    → Top 5 factores que aumentaron el riesgo (barras rojas)
    → Top 5 factores que disminuyeron el riesgo (barras azules)
  
  - Recomendaciones automáticas por categoría:
    → Si riesgo alto + sedentario: "Actividad física 150 min/semana (GuíaMSS)"
    → Si riesgo alto + HTA: "Control de presión arterial en próxima cita"
    → Si zona de incertidumbre: "Solicitar glucosa en ayuno y HbA1c"

Botón: "Nueva consulta" | Botón: "Guardar en expediente (PDF)"
```

**Pantalla 3 — Panel del médico (vista agregada)**

```
Acceso: sidebar → "Panel de estadísticas"

Métricas del turno:
  - Pacientes evaluados hoy: N
  - Distribución por categoría: donut chart bajo/medio/alto
  - Fenotipos más frecuentes: barra horizontal

Tabla de pacientes del turno:
  - ID (anónimo), categoría riesgo, probabilidad, fenotipo, acción recomendada
  - Exportar como CSV

Nota: NO almacenar datos personales identificables (cumple con restricción PII del sistema)
```

#### Integración con la API existente

```
dashboard/cliente_api.py

class ClientePrediccion:
    """
    Wrapper para consumir POST /predecir desde Streamlit.
    
    Métodos:
      predecir(datos_paciente: dict) -> dict
        → Mapea nombres de campos UI → campos de DatosPaciente
        → Llama a requests.post(RUTA_API + '/predecir', json=payload)
        → Retorna RespuestaPrediccion como dict
        → Maneja HTTPError 503 con mensaje amigable para el médico
    
    Configuración:
      RUTA_API = os.getenv('RUTA_API', 'http://localhost:8000')
      TIMEOUT_SEGUNDOS = 5
    """
```

#### Archivo de configuración Streamlit (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#006847"          # verde IMSS
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

### 3.4 Tickets de trabajo Sprint 3

| ID | Tarea | Prioridad | Dependencia | Estado actual | Hecho hasta ahora |
|---|---|---|---|---|---|
| S3-01 | Implementar `entrenamiento/fenotipador.py` con método del codo + silhouette | 🔴 CRÍTICA | S2-04 | ⬜ PENDIENTE | No existe `entrenamiento/fenotipador.py` en el repositorio |
| S3-02 | Integrar fenotipador como etapa 0 en pipeline supervisado | 🔴 CRÍTICA | S3-01, S2-06 | ⬜ PENDIENTE | `entrenamiento/pipeline.py` aún no tiene flujo con fenotipos |
| S3-03 | Generar tabla de perfiles de fenotipos con prevalencia de diabetes por cluster | 🔴 CRÍTICA | S3-01 | ⬜ PENDIENTE | No existe generador de perfiles de fenotipos |
| S3-04 | Crear `dashboard/app.py` — Pantalla 1: formulario de captura | 🔴 CRÍTICA | Ninguna | ⬜ PENDIENTE | No existe directorio `dashboard/` |
| S3-05 | Crear `dashboard/app.py` — Pantalla 2: resultado con gauge y SHAP | 🟡 ALTA | S3-04, S3-07 | ⬜ PENDIENTE | Sin implementación de UI ni capa SHAP |
| S3-06 | Crear `dashboard/app.py` — Pantalla 3: panel estadístico del médico | 🟡 ALTA | S3-04 | ⬜ PENDIENTE | Sin implementación de panel agregado |
| S3-07 | Calcular SHAP values para el mejor modelo supervisado | 🟡 ALTA | S2-04 | ⬜ PENDIENTE | No hay script SHAP ni artefactos SHAP en `reportes/` |
| S3-08 | Crear `dashboard/cliente_api.py` con manejo de errores 503 | 🟡 ALTA | S3-04 | ⬜ PENDIENTE | Cliente Streamlit no creado |
| S3-09 | Tema IMSS en `.streamlit/config.toml` | 🟢 MEDIA | S3-04 | ⬜ PENDIENTE | No existe configuración Streamlit |
| S3-10 | Añadir exportación de resultado como PDF desde Streamlit | 🟢 MEDIA | S3-05 | ⬜ PENDIENTE | Exportación PDF no implementada |
| S3-11 | Prueba de integración: formulario Streamlit → API → resultado coherente | 🟢 MEDIA | S3-05, S3-08 | ⬜ PENDIENTE | No hay pruebas de integración para dashboard |

---

## SECCIÓN 4 — Sprint 4: Reporte Académico & Despliegue Final

### 4.1 Estructura del Paper / Reporte Final

El reporte debe seguir la estructura IMRaD adaptada para un sistema de IA en salud. Esta estructura maximiza el puntaje en todas las dimensiones de la rúbrica.

```
reportes/paper_final.md  (o .pdf generado desde Jupyter nbconvert)

Secciones:

1. RESUMEN EJECUTIVO (250 palabras)
   - Problema: brecha de diagnóstico de diabetes en México
   - Método: dataset CDC como proxy, 3 modelos supervisados + fenotipado K-Means
   - Resultado principal: mejor ROC-AUC obtenido + comparativa con literatura
   - Contribución: API REST desplegable en contexto IMSS

2. INTRODUCCIÓN
   - Estadísticas ENSANUT 2022: prevalencia, costo económico, brecha diagnóstica
   - Limitaciones del tamizaje manual en primer nivel de atención
   - Objetivo del sistema y alcance del prototipo

3. DATOS Y PREPROCESAMIENTO
   - Descripción del dataset CDC BRFSS 2015 (N, variables, año)
   - Tabla de mapeo CDC → IMSS/ENSANUT (reproducir Tabla 1.2 de este documento)
   - Justificación de transferibilidad entre poblaciones
   - Decisiones de preprocesamiento: imputación, escalado, manejo de desbalance
   - Análisis de desbalance de clases y estrategia elegida

4. METODOLOGÍA
   4.1 Fenotipado no supervisado (K-Means): selección de K, variables utilizadas,
       interpretación clínica de clusters
   4.2 Modelos supervisados: SVM, Árbol/GBM, MLP — configuración y justificación
   4.3 Protocolo de validación: StratifiedKFold(5), métricas clínicas
   4.4 Arquitectura del sistema: diagrama de capas API→Inferencia→Entrenamiento

5. RESULTADOS
   - Tabla comparativa de modelos (reproducir salida de EvaluadorClinico.comparar_modelos())
   - Curvas ROC y PR para los 3 modelos (subplot 3×1)
   - Tabla de fenotipos metabólicos con estadísticas descriptivas
   - Análisis de importancia de variables (SHAP summary plot)
   - Análisis de equidad: métricas por grupo de sexo y edad

6. DISCUSIÓN
   6.1 Comparativa con literatura sobre predicción de T2DM en población hispana
   6.2 Sesgo poblacional CDC→México: variables afectadas y dirección del sesgo
   6.3 Implicaciones clínicas: umbral de decisión recomendado para contexto IMSS
   6.4 Limitaciones del estudio

7. CONCLUSIONES
   - Modelo recomendado y justificación
   - Viabilidad de despliegue en consultorios del primer nivel IMSS
   - Trabajo futuro: validación con datos ENSANUT reales, ajuste de calibración

8. REFERENCIAS
   - ENSANUT 2022 (INSP)
   - CDC BRFSS 2015 methodology report
   - Literatura SVM/diabetes en población hispana (≥3 papers)
   - FastAPI, scikit-learn, SHAP (citar software)
```

---

### 4.2 Comparativa con literatura médica

Esta tabla debe aparecer en la sección de Discusión del paper. Compara el rendimiento del modelo desarrollado con estudios publicados sobre predicción de diabetes en poblaciones similares.

| Estudio | Población | Dataset | Mejor modelo | ROC-AUC reportado | Variables usadas |
|---|---|---|---|---|---|
| Tigga & Garg (2020) | India | PIMA Indians | Random Forest | 0.82 | 8 variables |
| Zou et al. (2018) | China | Datos hospitalarios | Decision Tree | 0.78 | 13 variables |
| Huang et al. (2022) | EE.UU. (hispanos) | NHANES | SVM + features clínicos | 0.80 | 21 variables |
| Sisodia & Sisodia (2018) | India | PIMA Indians | Naive Bayes | 0.78 | 8 variables |
| **Este trabajo** | EE.UU./Proxy México | CDC BRFSS 2015 | Por determinar | **Por determinar** | 21 + fenotipo CDC |

**Objetivo de rendimiento:** superar 0.78 de ROC-AUC para estar competitivo con la literatura. El GradientBoostingClassifier típicamente alcanza 0.80–0.83 en este dataset.

---

### 4.3 Tickets de trabajo Sprint 4

| ID | Tarea | Prioridad | Dependencia | Estado actual | Hecho hasta ahora |
|---|---|---|---|---|---|
| S4-01 | Redactar secciones 1–3 del paper usando resultados del EDA Sprint 2 | 🔴 CRÍTICA | S2-02 | ⬜ PENDIENTE | No existe `reportes/paper_final.md` |
| S4-02 | Completar sección 4–5 con tablas de resultados y gráficas generadas | 🔴 CRÍTICA | S2-05, S3-07 | ⬜ PENDIENTE | No hay paper con resultados consolidados |
| S4-03 | Análisis de equidad: calcular ROC-AUC por subgrupo (sexo, edad, ingreso) | 🔴 CRÍTICA | S2-05 | ⬜ PENDIENTE | No existe módulo de equidad ni reporte asociado |
| S4-04 | Redactar sección 6 (Discusión) con contraste CDC↔ENSANUT | 🟡 ALTA | S4-01, S4-02 | ⬜ PENDIENTE | Sin borrador de discusión en `reportes/` |
| S4-05 | Crear `Dockerfile` multi-stage para API + Dashboard | 🟡 ALTA | S3-08 | ⬜ PENDIENTE | No hay `Dockerfile` |
| S4-06 | Crear `docker-compose.yml`: servicio API + servicio Dashboard | 🟡 ALTA | S4-05 | ⬜ PENDIENTE | No hay `docker-compose.yml` |
| S4-07 | Configurar GitHub Actions: lint (ruff) + pytest + build imagen | 🟡 ALTA | Ninguna | ⬜ PENDIENTE | No existe `.github/workflows/` |
| S4-08 | Añadir `pip audit` al pipeline CI para auditoría de dependencias | 🟢 MEDIA | S4-07 | ⬜ PENDIENTE | Sin pipeline CI donde integrar auditoría |
| S4-09 | Exportar paper a PDF vía nbconvert o pandoc | 🟢 MEDIA | S4-02 | ⬜ PENDIENTE | No hay artefacto PDF del paper |
| S4-10 | Etiqueta de versión `v1.0.0` en repositorio | 🟢 MEDIA | S4-09 | ⬜ PENDIENTE | Aún no se alcanza estado de release |

---

## SECCIÓN 5 — Recomendaciones de Alto Impacto

### 5.1 SHAP Values: explicabilidad para el médico del IMSS

**Por qué es crítico para la rúbrica:** el evaluador académico en contexto médico penalizará un modelo "caja negra" sin justificación de sus predicciones. SHAP convierte GBM y MLP en herramientas explicables.

#### Qué gráficas incluir y dónde

**Gráfica 1 — SHAP Summary Plot (para el paper, sección de Resultados)**

```python
# Scaffolding para reportes/generar_shap.py

import shap

def generar_summary_plot(modelo_pipeline, X_test, nombres_columnas):
    """
    Genera importancia global de variables para el reporte académico.
    
    Para GradientBoostingClassifier:
      explainer = shap.TreeExplainer(modelo_pipeline.named_steps['clasificador'])
    Para SVM / MLP:
      explainer = shap.KernelExplainer(modelo_pipeline.predict_proba, shap.sample(X_test, 100))
    
    Salida: reportes/shap_summary.png
      - Eje X: impacto en la predicción (valor SHAP)
      - Eje Y: variables ordenadas por importancia media
      - Color: valor de la variable (rojo=alto, azul=bajo)
    
    Interpretación para médico IMSS (anotación en el paper):
      "BMI alto (rojo) desplaza la predicción hacia riesgo mayor.
       PhysActivity=1 (azul) desplaza hacia riesgo menor."
    """
```

**Gráfica 2 — SHAP Waterfall Plot (para el Dashboard, por paciente)**

```python
def generar_waterfall_paciente(explainer, X_paciente_procesado, nombres_originales):
    """
    Muestra por qué el modelo predijo X para ESTE paciente específico.
    
    Interfaz visual en Dashboard (Pantalla 2):
      - Barras rojas: factores que aumentaron el riesgo
        Ejemplo: "BMI=34 contribuyó +0.18 a la probabilidad"
      - Barras azules: factores que disminuyeron el riesgo  
        Ejemplo: "PhysActivity=1 contribuyó -0.09 a la probabilidad"
      - Etiqueta base: probabilidad promedio del modelo (~0.14 por desbalance)
      - Etiqueta final: probabilidad predicha para este paciente
    
    Valor clínico: el médico puede contestar "¿por qué dice que es riesgo alto?"
    con evidencia cuantitativa, no solo "el modelo lo dice".
    """
```

**Gráfica 3 — SHAP Dependence Plot (para análisis de subgrupos en paper)**

```python
def generar_dependence_plot(shap_values, X_test, variable_principal, variable_interaccion):
    """
    Útil para detectar interacciones no lineales entre variables.
    
    Ejemplo recomendado para el paper:
      variable_principal='BMI', variable_interaccion='Age'
      → Muestra que el efecto del BMI en el riesgo varía según la edad
      → Paciente joven con BMI=35 tiene menor SHAP que adulto mayor con BMI=35
      
    Narrativa para sección de Discusión:
      "La interacción BMI×Edad sugiere que intervenciones de control de peso
       son especialmente críticas en adultos de 40–60 años (grupos 7–9 CDC),
       grupo que coincide con la mayor prevalencia de diabetes en ENSANUT 2022."
    """
```

---

### 5.2 Manejo del sesgo poblacional CDC→México en el reporte académico

Esta es la **variable diferenciadora** entre un trabajo de nivel básico y uno de nivel avanzado. Un evaluador que conozca epidemiología penalizará severamente ignorar este sesgo.

#### Estructura recomendada para la subsección "Limitaciones" del paper

**Limitación 1 — Sesgo de selección geográfica**

```
El dataset CDC BRFSS 2015 refleja la distribución de factores de riesgo de la
población adulta de EE.UU. Las diferencias distribucionales documentadas respecto
a México incluyen:

  - Tabaquismo: CDC ~44% vs. México ~18% → el modelo puede subestimar el peso
    de variables relacionadas con tabaquismo en población mexicana

  - Escolaridad: CDC media ~4.2 (algo de universidad) vs. México ~2.8 (preparatoria
    incompleta) → el coeficiente de 'Education' puede estar sobreajustado a
    niveles educativos altos poco prevalentes en México

  - Ingreso: La escala CDC (1–8 en USD) no es transferible a deciles mexicanos;
    esta variable debe interpretarse como ordenamiento relativo, no como nivel
    absoluto de ingreso

Respuesta metodológica adoptada: se reportan métricas de rendimiento separadas
por cuartil de escolaridad e ingreso para cuantificar el impacto de estos sesgos
(ver Tabla 5: Análisis de equidad por subgrupo).
```

**Limitación 2 — Calibración del modelo**

```
Los modelos entrenados en CDC BRFSS pueden generar probabilidades bien ordenadas
(buena discriminación, AUC alto) pero mal calibradas para México (la probabilidad
0.70 no necesariamente corresponde al 70% de prevalencia real en mexicanos).

Evaluación de calibración incluida:
  - Curva de calibración (reliability diagram): probabilidades predichas vs.
    prevalencia observada por decil
  - Brier Score: mide calibración global
  - Recomendación: si Brier Score > 0.15, aplicar Platt Scaling como capa de
    calibración posterior al modelo

Para el contexto de despliegue IMSS se recomienda reducir el umbral de "riesgo bajo"
de 0.33 a 0.25, dado que la diabetes no diagnosticada es ~35% en México vs. ~21%
en EE.UU. Este ajuste conservador prioriza sensibilidad sobre especificidad.
```

**Limitación 3 — Ausencia de datos de validación mexicanos**

```
La ausencia de un dataset de validación con población mexicana real impide
cuantificar la degradación del rendimiento al transferir el modelo.

Trabajo futuro propuesto:
  1. Solicitar acceso a microdatos de ENSANUT 2022 (INSP) para variables comparables
  2. Aplicar técnicas de domain adaptation (ej. TrAnsfer Component Analysis)
     para reducir el covariate shift entre distribuciones CDC y ENSANUT
  3. Validar con muestra de pacientes de primer nivel del IMSS (estudio prospectivo)
```

#### Cómo presentarlo en la Discusión para obtener máxima calificación

No presentar las limitaciones como "debilidades del trabajo", sino como **contribuciones metodológicas**:

> "Este trabajo contribuye a la literatura hispanohablante con el primer análisis sistemático de la transferibilidad del dataset CDC BRFSS 2015 al contexto de salud pública mexicano, documentando las variables con mayor sesgo distribucional y proponiendo ajustes de umbral específicos para el primer nivel de atención del IMSS."

Esta redacción convierte las limitaciones en aportaciones y diferencia el trabajo de un simple tutorial de scikit-learn.

---

### 5.3 Análisis de equidad (Fairness) como diferenciador académico

Un evaluador que conozca la ética en IA en salud reconocerá este análisis como indicador de madurez técnica.

#### Variables de equidad a analizar

```
entrenamiento/evaluador_equidad.py

class EvaluadorEquidad:
    """
    Calcula métricas de clasificación separadas por subgrupo demográfico.
    
    Subgrupos a analizar:
      - Sexo: 0 (femenino) vs 1 (masculino)
      - Edad: jóvenes (Age 1-4: 18-39 años) vs adultos medios (5-9: 40-64) vs mayores (10-13: 65+)
      - Ingreso: quintil bajo (1-2) vs quintil alto (7-8)
      - Educación: sin básica (1-2) vs con educación media/superior (4-6)
    
    Métodos:
      calcular_metricas_por_subgrupo(y_verdadero, y_pred_proba, grupos) -> pd.DataFrame
        → Para cada subgrupo: AUC, sensibilidad, especificidad, tasa_falsos_negativos
      
      calcular_paridad_demografica(y_pred, grupos) -> dict
        → Diferencia máxima en tasa de predicción positiva entre subgrupos
        → Criterio de equidad: diferencia < 10 puntos porcentuales
      
      generar_reporte_equidad() -> str
        → Narrativa en Markdown para incluir en paper
        → Señala subgrupos con mayor disparidad y recomendaciones
    
    Hallazgos típicos en dataset CDC (citar en Discusión):
      - Los modelos de ML en datos BRFSS tienden a tener mayor tasa de
        falsos negativos en mujeres de ingreso bajo (sesgo de doble marginación)
      - El grupo de edad 40-54 (grupos CDC 7-8) es el que más se beneficia
        de la detección temprana (mayor ROI clínico de la intervención)
    """
```

---

## SECCIÓN 6 — Estructura de directorios final propuesta

```
diasgnostico-pred/
├── api/
│   ├── __init__.py
│   ├── config.py
│   ├── esquemas.py
│   └── main.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py                    ← NUEVO S3-04
│   └── cliente_api.py            ← NUEVO S3-08
├── datos/
│   ├── brutos/
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv  ← NUEVO S2-01
│   └── procesados/
│       └── dataset_procesado.parquet                         ← NUEVO S2-09
├── entrenamiento/
│   ├── __init__.py
│   ├── cargador_datos.py
│   ├── comparador_modelos.py     ← EXTENDER S2-04
│   ├── evaluador.py              ← NUEVO S2-05
│   ├── evaluador_equidad.py      ← NUEVO S4-03
│   ├── fenotipador.py            ← NUEVO S3-01
│   ├── pipeline.py               ← EXTENDER S2-06
│   └── preprocesador.py          ← NUEVO S2-03
├── inferencia/
│   ├── __init__.py
│   └── predictor.py
├── modelos/
│   ├── .gitkeep
│   ├── fenotipador.joblib        ← GENERADO S3-01
│   └── modelo_diabetes_v1.joblib ← GENERADO S2-06
├── notebooks/
│   └── 01_eda_regionalizado.ipynb  ← NUEVO S2-02
├── pruebas/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_cargador.py
│   ├── test_predictor.py
│   ├── test_preprocesador.py     ← NUEVO S2-07
│   └── test_fenotipador.py       ← NUEVO S3-01
├── reportes/
│   ├── comparativa_modelos.md    ← GENERADO S2-05
│   ├── contraste_regional.md     ← NUEVO S2-10
│   ├── curvas_svm.png
│   ├── curvas_gbm.png
│   ├── curvas_mlp.png
│   ├── shap_summary.png          ← GENERADO S3-07
│   └── paper_final.md            ← NUEVO S4-09
├── .env.example
├── .github/workflows/ci.yml      ← NUEVO S4-07
├── .streamlit/config.toml        ← NUEVO S3-09
├── config.py
├── docker-compose.yml            ← NUEVO S4-06
├── Dockerfile                    ← NUEVO S4-05
├── PROYECTO.md
├── pyproject.toml
└── README.md
```

---

## SECCIÓN 7 — Checklist de entrega para máxima calificación

### Nivel Básico ✅ → Asegurado con Sprint 2

- [ ] 3 modelos supervisados entrenados con dataset real
- [ ] Preprocesamiento documentado y sin data leakage
- [ ] Tabla de métricas comparativa (accuracy, F1, AUC)

### Nivel Intermedio ✅ → Asegurado con Sprint 3

- [ ] SVM con kernel RBF y búsqueda de hiperparámetros
- [ ] Árbol de Decisión / Gradient Boosting con importancia de variables
- [ ] Red Neuronal (MLP) con early stopping
- [ ] K-Means como fenotipador metabólico (no como predictor) con K justificado
- [ ] Dashboard Streamlit con interfaz de consultorio IMSS

### Nivel Avanzado ✅ → Ya logrado (API) + Ampliar con Sprint 4

- [ ] API REST FastAPI en producción con validación clínica ✅ YA COMPLETADO
- [ ] Análisis de sesgo poblacional CDC↔México documentado
- [ ] SHAP values para interpretabilidad clínica
- [ ] Análisis de equidad por subgrupo demográfico
- [ ] Comparativa con literatura sobre T2DM en población hispana
- [ ] Curvas de calibración + Brier Score

### Diferenciadores competitivos (máxima calificación)

- [ ] Fenotipado metabólico con nombres clínicos interpretables
- [ ] Umbral de decisión ajustado para contexto México (0.25 en lugar de 0.33)
- [ ] Redacción de limitaciones como contribuciones metodológicas
- [ ] Recomendaciones de trabajo futuro con datos ENSANUT reales
- [ ] Análisis de fairness explícito con métricas de paridad demográfica
