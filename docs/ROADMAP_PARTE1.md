# Diasgnostico-pred — Hoja de Ruta Estratégica  
## Regionalización IMSS/ENSANUT · Evaluación Académica · Sprints 2–4

> **Documento:** Parte 1 de 2  
> **Audiencia:** Technical Lead + Evaluador Académico  
> **Estado base actualizado (2026-05-12):** Sprint 2 en ejecución — núcleo técnico implementado (8/10 tickets), faltan artefactos de datos reales (notebook EDA y reporte final de contraste)  
> **Objetivo:** Alcanzar nivel Avanzado en rúbrica + contextualización para población mexicana  

---

## SECCIÓN 1 — Análisis de Brechas

### 1.1 Estado actual vs. rúbrica de evaluación

| Dimensión | Nivel Básico | Nivel Intermedio | Nivel Avanzado | Estado actual |
|---|---|---|---|---|
| Modelos supervisados | 3 modelos entrenados | SVM + Árbol + Red Neuronal | Comparativa con literatura | ✅ Implementados en `entrenamiento/comparador_modelos.py` (pendiente entrenar con dataset real) |
| Preprocesamiento | Limpieza básica | Escalado + codificación | Pipeline sklearn sin data leakage | ✅ `ConstructorPreprocesador` implementado y probado |
| Métricas | Accuracy | F1, AUC-ROC | Métricas clínicas (sensibilidad/especificidad) | ✅ `EvaluadorClinico` implementado (ROC-AUC, PR-AUC, sensibilidad, especificidad, calibración) |
| Modelo no supervisado | — | K-Means como agrupador | K-Means como fenotipado metabólico | ⚠️ Solo K-Means base en `ComparadorModelos`; fenotipado clínico aún pendiente |
| Dashboard | — | Visualización interactiva | Interfaz de consultorio IMSS | ❌ No implementado |
| API en producción | — | — | FastAPI desplegada | ✅ Completado |
| Contextualización regional | — | — | Contraste ENSANUT vs. CDC | ⚠️ Tabla/mapeo y script de contraste listos; falta generar reporte final con dataset real |

**Brechas críticas desbloqueantes (en orden de prioridad):**

1. Poblar `datos/brutos/` con dataset real y ejecutar pipeline end-to-end (aún no hay artefactos de datos/versiones en repo)
2. Completar notebook EDA regionalizado (`notebooks/01_eda_regionalizado.ipynb`)
3. Generar `reportes/contraste_regional.md` con datos reales (el generador ya existe)
4. K-Means reencuadrado como fenotipado clínico (no solo clustering base)
5. Dashboard Streamlit orientado al médico general

---

### 1.2 Mapeo CDC BRFSS 2015 → IMSS PrevenIMSS / ENSANUT

La tabla siguiente es el **artefacto central de regionalización**. Conecta cada variable del dataset con su equivalente en los instrumentos de tamizaje utilizados en la consulta de medicina preventiva del IMSS, y referencia la prevalencia reportada en la ENSANUT 2022 para México.

| Columna CDC | Pregunta equivalente PrevenIMSS / ENSANUT | Escala CDC | Escala IMSS | Prevalencia México (ENSANUT 2022) | Notas de adaptación |
|---|---|---|---|---|---|
| `HighBP` | "¿Le han dicho que tiene presión arterial alta?" (Módulo Adulto PrevenIMSS) | Binaria 0/1 | Sí/No | 30.1% adultos ≥20 años | Equivalencia directa |
| `HighChol` | "¿Le han dicho que tiene colesterol alto?" (Módulo Adulto PrevenIMSS) | Binaria 0/1 | Sí/No | 19.6% adultos (ENSANUT) | Equivalencia directa |
| `CholCheck` | "¿Se ha hecho prueba de colesterol en los últimos 5 años?" | Binaria 0/1 | Sí/No | Sin dato directo; proxy: cobertura de laboratorio IMSS | Ítem de acceso a servicios |
| `BMI` | Índice de Masa Corporal calculado en consulta (peso/talla²) | Continua 10–80 | Continua | Sobrepeso+obesidad: 76.8% adultos. Media estimada IMC ≈ 29.2 kg/m² | **Ajuste crítico:** la distribución mexicana está desplazada ~1.5 puntos arriba respecto a la muestra CDC |
| `Smoker` | "¿Ha fumado ≥100 cigarrillos en su vida?" | Binaria 0/1 | Sí/No | Prevalencia tabaquismo 17.6% (ENSANUT 2022) vs. ~44% muestra CDC | **Sesgo importante:** CDC sobreestima tabaquismo vs. México |
| `Stroke` | "¿Le han dicho que tuvo un derrame cerebral?" | Binaria 0/1 | Sí/No | 2.7% adultos ≥40 años | Equivalencia directa |
| `HeartDiseaseorAttack` | "¿Ha tenido infarto al miocardio o enfermedad cardiaca?" | Binaria 0/1 | Sí/No | 3.4% adultos (INEGI 2022) | Equivalencia directa |
| `PhysActivity` | "¿Realizó actividad física en los últimos 30 días?" | Binaria 0/1 | Sí/No | 39.5% inactivos físicamente (ENSANUT) | CDC define "cualquier actividad", más permisivo |
| `Fruits` | "¿Consume fruta ≥1 vez por día?" | Binaria 0/1 | Sí/No | 42% consume fruta diariamente (ENSANUT) | Comparable |
| `Veggies` | "¿Consume verduras ≥1 vez por día?" | Binaria 0/1 | Sí/No | 34% consume verdura diariamente (ENSANUT) | Comparable |
| `HvyAlcoholConsump` | "¿Consume >14 copas/semana (hombre) o >7 copas/semana (mujer)?" | Binaria 0/1 | Sí/No | Consumo excesivo: 7.6% (ENCODAT 2017) vs. ~5.6% CDC | Comparable con ajuste de definición |
| `AnyHealthcare` | "¿Cuenta con seguro médico o afiliación a servicio de salud?" | Binaria 0/1 | Sí/No | IMSS: 40%, ISSSTE: 6.9%, Insabi/Bienestar: 17.5% (2022) | En México la fragmentación del sistema es variable clave |
| `NoDocbcCost` | "¿Dejó de ir al médico en el último año por costo?" | Binaria 0/1 | Sí/No | 28.5% reporta barrera económica para atención (ENSANUT) | Variable de acceso relevante para México |
| `GenHlth` | "En general, su salud es… (excelente a mala)" | Ordinal 1–5 | Escala de 5 puntos | 41% reporta salud "buena o muy buena" (ENSANUT) | Escala análoga en cartilla de salud IMSS |
| `MentHlth` | "¿Cuántos días del último mes su salud mental fue mala?" | Continua 0–30 | Días | Sin equivalente directo IMSS; proxy: PHQ-2 en módulo salud mental | Covariable de comorbilidad psiquiátrica |
| `PhysHlth` | "¿Cuántos días del último mes tuvo problemas físicos?" | Continua 0–30 | Días | Sin equivalente directo; proxy: días de ausentismo laboral IMSS | Covariable de carga de enfermedad |
| `DiffWalk` | "¿Tiene dificultad para caminar o subir escaleras?" | Binaria 0/1 | Sí/No | Discapacidad motriz adultos: 8.1% (INEGI 2020) | Equivalencia directa |
| `Sex` | Sexo registrado en cartilla de salud | Binaria 0=F/1=M | F/M | 51.2% mujeres población mexicana | Equivalencia directa |
| `Age` | Grupo etario (escala CDC: 1=18–24, 13=80+) | Ordinal 1–13 | Grupos de 5 años | Pirámide poblacional mexicana más joven | **Ajuste:** distribución etaria difiere; grupos CDC sub-representan adultos mayores jóvenes (30–44) en México |
| `Education` | Último grado de escolaridad (escala CDC: 1=sin escuela, 6=universitario) | Ordinal 1–6 | Escolaridad | 24% sin educación básica completa (INEGI 2020) vs. ~10% CDC | **Sesgo crítico:** CDC tiene mayor escolaridad promedio; necesita mención explícita en reporte |
| `Income` | Ingreso familiar anual (escala CDC: 1=<$10k, 8=>$75k) | Ordinal 1–8 | Deciles de ingreso | No equiparable directamente en USD/MXN | **Reescalar:** usar ranking relativo (quintiles) en análisis, no valor absoluto |

**Nota metodológica para el reporte académico:**  
El dataset CDC BRFSS 2015 es válido como **punto de partida transferible** bajo los siguientes supuestos, que deben declararse explícitamente en la sección de limitaciones del paper:

1. Las variables binarias de condiciones crónicas (HTA, dislipidemia, cardiopatía) tienen equivalencia conceptual directa con los tamizajes del IMSS.
2. Las variables de comportamiento (tabaquismo, actividad física, dieta) presentan **sesgos de distribución** documentados entre la muestra CDC (predominantemente anglosajona) y la población mexicana; el modelo aprendido puede tener **calibración subóptima** para México.
3. Las variables socioeconómicas (`Income`, `Education`) no son equivalentes en escala absoluta y deben tratarse como **variables de orden relativo** en el análisis de equidad.

---

### 1.3 Variable objetivo: prevalencia comparada

| Indicador | EE.UU. (CDC BRFSS 2015) | México (ENSANUT 2022) |
|---|---|---|
| Diabetes diagnosticada (adultos) | 13.0% | 12.6% |
| Prediabetes / riesgo elevado | 37.3% | 11.7% (ayuno alterado) |
| Diabetes no diagnosticada | ~21% del total | ~35% del total |
| Prevalencia en mayores de 60 años | ~26% | ~31% |

**Implicación para el modelo:** la prevalencia base es similar (~13%), lo que hace el dataset CDC válido para entrenamiento. Sin embargo, la **prediabetes no diagnosticada** es significativamente mayor en México, lo que justifica optar por un umbral de decisión más conservador (ej. reducir `UMBRAL_RIESGO_BAJO` de 0.33 a 0.25 en despliegue mexicano).

---

## SECCIÓN 2 — Sprint 2: EDA Regionalizado & Modelado Base

### 2.1 Objetivo del Sprint

Reemplazar el `DummyClassifier` con tres modelos supervisados de calidad clínica, precedidos de un pipeline de preprocesamiento reproducible y un EDA que justifique las decisiones metodológicas con referencia explícita a ENSANUT.

### 2.2 Estructura del Notebook de EDA (`notebooks/01_eda_regionalizado.ipynb`)

El notebook debe organizarse en las siguientes secciones para maximizar el puntaje en "Análisis de resultados":

**Bloque 1 — Carga y contrato de calidad**
- Verificar que las 21 columnas CDC están presentes y en rango
- Tabla de valores faltantes por columna (heatmap)
- Distribución de la variable objetivo: gráfica de barras con porcentaje de clases
- Confirmar desbalance: clase 0 (~86%) vs. clase 1 (~14%)

**Bloque 2 — Análisis univariado con anotación ENSANUT**
- Para cada variable binaria: barplot comparando prevalencia en dataset CDC vs. prevalencia reportada en México (tabla 1.2 como fuente)
- Para variables continuas (BMI, MentHlth, PhysHlth): histogramas con línea vertical marcando la media mexicana (ENSANUT)
- Tabla resumen: "Variables con mayor discrepancia distribucional CDC↔México"

**Bloque 3 — Análisis bivariado: factores de riesgo para diabetes**
- Matriz de correlación de Spearman (apropiada para variables ordinales/binarias)
- Top 10 variables con mayor diferencia de media entre clase 0 y clase 1 (t-test de Welch o Mann-Whitney U)
- Gráfica de odds ratios: variables binarias vs. `Diabetes_binary` con intervalo de confianza 95%

**Bloque 4 — Contraste de prevalencias México vs. EE.UU.**
- Gráfica de doble barra horizontal: prevalencia de cada factor de riesgo en dataset CDC vs. cifras ENSANUT 2022
- Párrafo de interpretación automático por región (usar f-strings con valores calculados)
- Tabla de "Variables más sesgadas": aquellas donde la diferencia relativa supera el 30%

**Bloque 5 — Análisis de desbalance y estrategia de remuestreo**
- Gráfica de distribución de clases
- Comparar 3 estrategias: sin remuestreo / SMOTE / submuestreo por clase mayoritaria
- Recomendación justificada basada en AUC-ROC (métrica más robusta ante desbalance)

**Bloque 6 — Conclusiones del EDA para informe académico**
- Lista de decisiones de preprocesamiento derivadas del EDA
- Hipótesis de qué variables serán más discriminativas (a validar en Sprint 3)

---

### 2.3 Pipeline de preprocesamiento (`entrenamiento/preprocesador.py`)

El preprocesador debe construirse como un `sklearn.Pipeline` para garantizar ausencia de data leakage. Estructura recomendada:

```
entrenamiento/preprocesador.py

class ConstructorPreprocesador:
    """
    Construye un ColumnTransformer serializable para el pipeline sklearn.
    
    Columnas numéricas continuas (BMI, MentHlth, PhysHlth):
        → SimpleImputer(strategy='median') → StandardScaler()
    
    Columnas binarias (0/1):
        → SimpleImputer(strategy='most_frequent') → passthrough
        (no escalar binarias: no aporta información adicional)
    
    Columnas ordinales (GenHlth, Age, Education, Income):
        → SimpleImputer(strategy='most_frequent') → OrdinalEncoder() con orden explícito
    
    Retorna: sklearn.compose.ColumnTransformer listo para Pipeline
    """
    
    COLUMNAS_CONTINUAS = ['BMI', 'MentHlth', 'PhysHlth']
    COLUMNAS_BINARIAS = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'
    ]
    COLUMNAS_ORDINALES = ['GenHlth', 'Age', 'Education', 'Income']
    
    def construir(self) -> ColumnTransformer:
        ...
    
    def construir_pipeline(self, clasificador) -> Pipeline:
        """Encadena preprocesador + clasificador. Nunca ajustar preprocesador sobre test."""
        ...
```

**Decisión crítica de diseño:** guardar el `Pipeline` completo (preprocesador + modelo) como un solo `.joblib`. El `PredictorDiabetes` existente es compatible sin modificaciones, ya que llama a `predict_proba` sobre el objeto cargado.

---

### 2.4 Modelos supervisados a implementar (`entrenamiento/comparador_modelos.py` — extensión)

#### Modelo 1: Support Vector Machine (SVM)

```
Justificación clínica: SVM con kernel RBF es efectivo en espacios de alta dimensión
con variables mixtas. Su margen de separación máximo tiene interpretación natural
como "zona de incertidumbre clínica".

Configuración:
  - Estimador: SVC(kernel='rbf', probability=True, class_weight='balanced')
  - class_weight='balanced': compensa desbalance 86/14 sin remuestreo artificial
  - probability=True: habilita predict_proba (requerido por PredictorDiabetes)
  - Hiperparámetros a explorar: C=[0.1, 1, 10], gamma=['scale', 'auto']
  - Búsqueda: GridSearchCV(cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
  - Advertencia de escala: SVM es sensible a magnitudes → ColumnTransformer obligatorio

Referencia académica para reporte:
  Huang et al. (2022): "SVM achieves 78.3% AUC predicting T2DM in Hispanic cohorts"
  → útil para comparativa directa en conclusiones
```

#### Modelo 2: Árbol de Decisión / Gradient Boosting

```
Justificación clínica: Los árboles de decisión generan reglas legibles ("si BMI > 30
Y HighBP = 1 entonces riesgo alto"), que el médico puede verificar manualmente.
Gradient Boosting amplifica esta capacidad con mayor precisión.

Configuración primaria (Árbol simple — para interpretabilidad):
  - DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
  - max_depth=5: interpretable; profundidades mayores overfitean sin aportar clínica

Configuración avanzada (GradientBoostingClassifier — para rendimiento):
  - GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
  - Alternativa si se añade XGBoost: XGBClassifier(use_label_encoder=False,
      eval_metric='logloss', scale_pos_weight=6.1)  # ratio clase_0/clase_1

Nota: incluir ambos en ComparadorModelos permite mostrar el tradeoff
interpretabilidad ↔ rendimiento, que es exactamente lo que un evaluador académico
en contexto clínico premia.
```

#### Modelo 3: Red Neuronal (MLP)

```
Justificación clínica: Las redes neuronales capturan interacciones no lineales complejas
entre factores de riesgo metabólico (ej. el efecto conjunto de obesidad + sedentarismo
+ edad no es aditivo).

Configuración:
  - MLPClassifier(
        hidden_layer_sizes=(64, 32),  # dos capas ocultas
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,          # evita overfitting
        validation_fraction=0.1,
        class_weight equivalente: usar class_weight en sample_weight
        random_state=42
    )
  - Alternativa PyTorch/Keras: válida pero rompe compatibilidad con sklearn Pipeline;
    usar solo si se agrega capa de adaptador para joblib

Nota sobre escalado: MLP es extremadamente sensible a escala.
El ColumnTransformer del preprocesador es OBLIGATORIO para este modelo.
```

---

### 2.5 Métricas de evaluación clínica

Para un sistema de tamizaje de diabetes, la **sensibilidad (recall de clase positiva)** es más crítica que la precisión. Un falso negativo (paciente diabético clasificado como sano) tiene consecuencias clínicas más graves que un falso positivo.

| Métrica | Fórmula | Umbral aceptable | Justificación clínica |
|---|---|---|---|
| **ROC-AUC** | Área bajo curva ROC | > 0.78 | Métrica principal para comparativa con literatura |
| **Sensibilidad** | TP / (TP + FN) | > 0.70 | Detectar el 70%+ de diabéticos reales |
| **Especificidad** | TN / (TN + FP) | > 0.65 | Evitar saturar sistema de salud con falsos positivos |
| **F1-Score (clase +)** | 2·P·R / (P+R) | > 0.55 | Balance ante desbalance de clases |
| **Precision-Recall AUC** | Área bajo curva PR | > 0.50 | Más informativo que ROC ante desbalance severo |
| **Brier Score** | MSE de probabilidades | < 0.15 | Mide calibración: ¿la probabilidad reflejada es real? |

**Implementar en `entrenamiento/evaluador.py`:**

```
class EvaluadorClinico:
    """
    Calcula métricas clínicas completas sobre predicciones de un modelo.
    
    Métodos:
      calcular_metricas(y_verdadero, y_prob) -> dict[str, float]
        → accuracy, sensibilidad, especificidad, f1, roc_auc, pr_auc, brier_score
      
      graficar_curvas(y_verdadero, y_prob, nombre_modelo) -> None
        → Curva ROC + Curva PR en subplots, con línea de referencia
        → Guardar en reportes/curvas_{nombre_modelo}.png
      
      comparar_modelos(resultados: list[ResultadoModelo]) -> pd.DataFrame
        → Tabla comparativa markdown-friendly
        → Guardar en reportes/comparativa_modelos.md
    """
```

---

### 2.6 Tickets de trabajo Sprint 2

| ID | Tarea | Responsable sugerido | Prioridad | Dependencia | Estado actual | Hecho hasta ahora |
|---|---|---|---|---|---|---|
| S2-01 | Descargar dataset CDC BRFSS 2015 vía UCI/Kaggle → `datos/brutos/` | DevOps / ML Eng | 🔴 CRÍTICA | Ninguna | 🟡 PARCIAL | Implementado `entrenamiento/descargador_dataset.py` + dependencia `ucimlrepo` + pruebas; falta persistir dataset real en `datos/brutos/` dentro del repo |
| S2-02 | Crear `notebooks/01_eda_regionalizado.ipynb` con estructura de bloques 1–6 | Data Scientist | 🔴 CRÍTICA | S2-01 | ⬜ PENDIENTE | Aún no existe carpeta `notebooks/` ni notebook en el repositorio |
| S2-03 | Implementar `entrenamiento/preprocesador.py` con ColumnTransformer | ML Engineer | 🔴 CRÍTICA | S2-01 | ✅ COMPLETADO | `ConstructorPreprocesador` creado con continuas/binarias/ordinales y soporte de fenotipo |
| S2-04 | Extender `ComparadorModelos` con SVM, GBM, MLP usando Pipeline sklearn | ML Engineer | 🔴 CRÍTICA | S2-03 | ✅ COMPLETADO | `ComparadorModelos` ya entrena `svm`, `arbol`, `gbm`, `mlp` con CV ROC-AUC |
| S2-05 | Crear `entrenamiento/evaluador.py` con métricas clínicas + gráficas | Data Scientist | 🟡 ALTA | S2-04 | ✅ COMPLETADO | `EvaluadorClinico` implementado con métricas, curvas ROC/PR, calibración y comparativa Markdown |
| S2-06 | Ampliar `entrenamiento/pipeline.py` para serializar Pipeline completo (preprocesador+modelo) | ML Engineer | 🟡 ALTA | S2-03, S2-04 | ✅ COMPLETADO | Pipeline guarda modelo versionado + alias final, evalúa modelos y persiste JSON de métricas |
| S2-07 | Añadir `pruebas/test_preprocesador.py` — verifica que no hay data leakage en Pipeline | QA | 🟡 ALTA | S2-03 | ✅ COMPLETADO | Suite de pruebas específica agregada y pasando |
| S2-08 | Ampliar `pruebas/test_cargador.py` con análisis de distribución y detección de desbalance | QA | 🟢 MEDIA | S2-01 | ✅ COMPLETADO | Casos de desbalance, distribución, separación X/y y persistencia Parquet agregados |
| S2-09 | Guardar dataset procesado en `datos/procesados/` formato Parquet | ML Engineer | 🟢 MEDIA | S2-03 | 🟡 PARCIAL | Existe `CargadorDatos.persistir_procesado()`, pero no hay artefacto Parquet versionado en repo |
| S2-10 | Documentar comparativa distribucional CDC↔ENSANUT en `reportes/contraste_regional.md` | Data Scientist | 🟢 MEDIA | S2-02 | 🟡 PARCIAL | Existe `reportes/generar_contraste_regional.py`; falta ejecutar con dataset real y publicar el `.md` resultante |

---
