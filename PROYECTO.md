# diasgnostico-pred — Documento de Diseño

**Versión:** 0.1.0 | **Dataset:** CDC BRFSS 2015 | **Metodología:** Espiral iterativa, 5 sprints

---

## 1. Arquitectura del sistema

```
┌─────────────────────────────────────────────┐
│               Cliente / Consumidor           │
└────────────────────┬────────────────────────┘
                     │ HTTP JSON
┌────────────────────▼────────────────────────┐
│              api/  (FastAPI)                 │
│  /salud  — verificación de salud            │
│  /predecir — validación → inferencia        │
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
│  fenotipado.py → K-Means / fenotipado       │
│  optimizador.py → GridSearchCV formal       │
│  generador_reportes.py → síntesis legible   │
│  evaluador.py → métricas clínicas           │
│  pipeline.py  → orquestador CLI             │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│     datos/brutos/  (CSV CDC BRFSS 2015)     │
│     modelos/       (artefactos .joblib)     │
│     reportes/      (métricas, gráficas)     │
└─────────────────────────────────────────────┘
```

---

## 2. Principios de diseño

| Principio | Aplicación concreta |
|---|---|
| Separación de responsabilidades | Cuatro capas independientes: datos, entrenamiento, inferencia, API |
| Contrato explícito | `config.py` centraliza todas las constantes (columnas, rutas, umbrales, semilla). Ningún módulo define *magic strings*. |
| Español primero | Nombres de clases, métodos, variables y endpoints en español; las columnas CDC permanecen en inglés (identificadores oficiales del dataset). |
| Modo degradado | Si el `.joblib` no existe, la API responde `/salud` con `estado: degradado` en lugar de fallar al arrancar. |
| Incertidumbre explícita | Si la probabilidad cae dentro de ±5% de un umbral, la respuesta incluye una advertencia clínica. |
| Pruebas de contrato | Los tests verifican contratos de interfaz (tipos devueltos, códigos HTTP, mapeos de campos), no implementaciones internas. |
| Sin data leakage | El `ColumnTransformer`/`Pipeline` se ajusta exclusivamente sobre `X_train`. Nunca sobre `X_test` ni sobre el conjunto completo antes de partir. |

---

## 3. Contrato de datos

### 3.1 Esquema de entrada de la API (21 campos públicos)

Los campos públicos de `DatosPaciente` usan nombres en español sin PII. Pydantic aplica rangos clínicamente plausibles.

| Campo público (ES) | Columna CDC | Rango permitido | Interpretación clínica |
|---|---|---|---|
| `presion_alta` | `HighBP` | 0 – 1 | Diagnóstico previo de HTA |
| `colesterol_alto` | `HighChol` | 0 – 1 | Diagnóstico previo de dislipidemia |
| `chequeo_colesterol` | `CholCheck` | 0 – 1 | Prueba de colesterol en últimos 5 años |
| `imc` | `BMI` | 10.0 – 80.0 | Índice de masa corporal kg/m² |
| `fumador` | `Smoker` | 0 – 1 | ≥100 cigarrillos en su vida |
| `derrame_cerebral` | `Stroke` | 0 – 1 | Derrame cerebral previo |
| `enfermedad_corazon` | `HeartDiseaseorAttack` | 0 – 1 | Cardiopatía coronaria o infarto |
| `actividad_fisica` | `PhysActivity` | 0 – 1 | Actividad física en últimos 30 días |
| `consume_fruta` | `Fruits` | 0 – 1 | Fruta ≥1 vez al día |
| `consume_verdura` | `Veggies` | 0 – 1 | Verdura ≥1 vez al día |
| `consumo_alcohol_alto` | `HvyAlcoholConsump` | 0 – 1 | Consumo elevado de alcohol |
| `tiene_cobertura_medica` | `AnyHealthcare` | 0 – 1 | Algún tipo de cobertura médica |
| `sin_medico_por_costo` | `NoDocbcCost` | 0 – 1 | Evitó médico por costo |
| `salud_general` | `GenHlth` | 1 – 5 | Salud autorreportada (1=excelente, 5=mala) |
| `salud_mental` | `MentHlth` | 0 – 30 | Días de mala salud mental en el último mes |
| `salud_fisica` | `PhysHlth` | 0 – 30 | Días de mala salud física en el último mes |
| `dificultad_caminar` | `DiffWalk` | 0 – 1 | Dificultad para caminar o subir escaleras |
| `sexo` | `Sex` | 0 – 1 | 0=Femenino, 1=Masculino |
| `edad` | `Age` | 1 – 13 | Grupos etarios CDC (1=18–24, 13=80+) |
| `educacion` | `Education` | 1 – 6 | 1=sin escolaridad, 6=universitario+ |
| `ingreso` | `Income` | 1 – 8 | Ranking relativo de ingreso |

### 3.2 Validador de coherencia clínica

Un `@model_validator` rechaza entradas donde `salud_fisica ≥ 20` Y `dificultad_caminar = 0` (deterioro físico severo sin dificultad para caminar es clínicamente incoherente). Retorna HTTP 422.

### 3.3 Umbrales de riesgo

```python
UMBRAL_RIESGO_BAJO   = 0.33   # probabilidad < 0.33  → "bajo"
UMBRAL_RIESGO_ALTO   = 0.66   # probabilidad ≥ 0.66  → "alto"
MARGEN_INCERTIDUMBRE = 0.05   # ±5% de cualquier umbral → advertencia clínica

# Ajuste recomendado para despliegue en México (justificado en §5.2):
UMBRAL_DECISION_MX   = 0.25   # umbral operativo conservador para contexto IMSS
```

---

## 4. Mapeo regional CDC BRFSS 2015 → IMSS/ENSANUT

### 4.1 Evaluación de transferibilidad por variable

Esta tabla es el artefacto central de regionalización. Conecta cada variable con su equivalente en PrevenIMSS y documenta el sesgo distribucional respecto a población mexicana.

| Columna CDC | Equivalente PrevenIMSS / ENSANUT | Escala CDC | Escala IMSS | Prevalencia México (ENSANUT 2022) | Nota de transferibilidad |
|---|---|---|---|---|---|
| `HighBP` | "¿Le han dicho que tiene presión arterial alta?" | Binaria 0/1 | Sí/No | 30.1% adultos ≥20 años | Equivalencia directa |
| `HighChol` | "¿Le han dicho que tiene colesterol alto?" | Binaria 0/1 | Sí/No | 19.6% adultos | Equivalencia directa |
| `CholCheck` | "¿Se ha hecho prueba de colesterol en los últimos 5 años?" | Binaria 0/1 | Sí/No | Sin dato directo; proxy: cobertura de laboratorio IMSS | Ítem de acceso a servicios |
| `BMI` | IMC calculado en consulta (peso/talla²) | Continua 10–80 | Continua | Sobrepeso+obesidad: 76.8% adultos. Media IMC ≈ 29.2 kg/m² | **Ajuste crítico:** distribución mexicana desplazada ~+1.5 puntos sobre CDC |
| `Smoker` | "¿Ha fumado ≥100 cigarrillos en su vida?" | Binaria 0/1 | Sí/No | Tabaquismo: 17.6% (ENSANUT 2022) vs ~44% CDC | **Sesgo alto:** CDC sobreestima tabaquismo respecto a México |
| `Stroke` | "¿Le han dicho que tuvo un derrame cerebral?" | Binaria 0/1 | Sí/No | 2.7% adultos ≥40 años | Equivalencia directa |
| `HeartDiseaseorAttack` | "¿Ha tenido infarto o enfermedad cardiaca?" | Binaria 0/1 | Sí/No | 3.4% adultos (INEGI 2022) | Equivalencia directa |
| `PhysActivity` | "¿Realizó actividad física en los últimos 30 días?" | Binaria 0/1 | Sí/No | 39.5% inactivos físicamente (ENSANUT) | Definición CDC más permisiva |
| `Fruits` | "¿Consume fruta ≥1 vez por día?" | Binaria 0/1 | Sí/No | 42% consumo diario de fruta | Comparable |
| `Veggies` | "¿Consume verduras ≥1 vez por día?" | Binaria 0/1 | Sí/No | 34% consumo diario de verdura | Comparable |
| `HvyAlcoholConsump` | ">14 copas/semana (H) o >7 copas/semana (M)?" | Binaria 0/1 | Sí/No | Consumo excesivo: 7.6% (ENCODAT 2017) | Comparable con ajuste de definición |
| `AnyHealthcare` | "¿Cuenta con seguro médico o afiliación?" | Binaria 0/1 | Sí/No | IMSS: 40%, ISSSTE: 6.9%, Bienestar: 17.5% (2022) | La fragmentación del sistema es variable clave en México |
| `NoDocbcCost` | "¿Dejó de ir al médico en el último año por costo?" | Binaria 0/1 | Sí/No | 28.5% reporta barrera económica (ENSANUT) | Variable de acceso relevante para México |
| `GenHlth` | "En general, su salud es… (excelente a mala)" | Ordinal 1–5 | Escala de 5 puntos | 41% reporta salud "buena o muy buena" (ENSANUT) | Escala análoga en cartilla de salud IMSS |
| `MentHlth` | "¿Cuántos días su salud mental fue mala?" | Continua 0–30 | Días | Sin equivalente directo IMSS; proxy: PHQ-2 | Covariable de comorbilidad psiquiátrica |
| `PhysHlth` | "¿Cuántos días tuvo problemas físicos?" | Continua 0–30 | Días | Sin equivalente directo; proxy: días de ausentismo laboral IMSS | Covariable de carga de enfermedad |
| `DiffWalk` | "¿Tiene dificultad para caminar o subir escaleras?" | Binaria 0/1 | Sí/No | Discapacidad motriz adultos: 8.1% (INEGI 2020) | Equivalencia directa |
| `Sex` | Sexo registrado en cartilla de salud | Binaria 0=F/1=M | F/M | 51.2% mujeres | Equivalencia directa |
| `Age` | Grupo etario (CDC: 1=18–24, 13=80+) | Ordinal 1–13 | Grupos de 5 años | Pirámide mexicana más joven | **Ajuste:** distribución etaria difiere; CDC sub-representa adultos 30–44 en México |
| `Education` | Último grado de escolaridad (CDC: 1=sin escuela, 6=universitario) | Ordinal 1–6 | Escolaridad | 24% sin educación básica completa (INEGI 2020) vs ~10% CDC | **Sesgo crítico:** CDC tiene mayor escolaridad promedio |
| `Income` | Ingreso familiar anual (CDC: 1=<$10k, 8=>$75k) | Ordinal 1–8 | Deciles de ingreso | No equiparable directamente en USD/MXN | **Reescalar:** usar ranking relativo (quintiles), no valor absoluto |

### 4.2 Variable objetivo: prevalencia comparada

| Indicador | EE. UU. (CDC BRFSS 2015) | México (ENSANUT 2022) |
|---|---|---|
| Diabetes diagnosticada (adultos) | 13.0% | 12.6% |
| Prediabetes / riesgo elevado | 37.3% | 11.7% (ayuno alterado) |
| Diabetes no diagnosticada | ~21% del total | ~35% del total |
| Prevalencia en mayores de 60 años | ~26% | ~31% |

**Implicación para el modelo:** la prevalencia base (~13%) es similar entre contextos, lo que valida CDC como proxy de entrenamiento. Sin embargo, la diabetes no diagnosticada es significativamente mayor en México, lo que justifica un umbral de decisión más conservador (0.25 en lugar de 0.33) para despliegue operativo.

---

## 5. Catálogo de modelos: justificación

Cada modelo se serializa como un `sklearn.Pipeline` completo (preprocesador + estimador) para garantizar compatibilidad con `PredictorDiabetes.predecir()` sin modificaciones.

| Modelo | Justificación clínica | Parámetros clave | ROC-AUC esperado |
|---|---|---|---|
| SVM (kernel RBF) | Efectivo en espacios de alta dimensión con variables mixtas; el margen de separación máximo tiene interpretación natural como zona de incertidumbre clínica | `kernel='rbf'`, `probability=True`, `class_weight='balanced'`, grilla CV: `C=[0.1,1,10]`, `gamma=['scale','auto']` | 0.78–0.80 |
| Árbol de decisión / GBM | Genera reglas legibles ("si IMC > 30 Y HTA = 1 → riesgo alto") auditables por el clínico; GBM amplifica la capacidad con mayor precisión | Árbol: `max_depth=5`, `class_weight='balanced'`; GBM: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05` | 0.80–0.83 |
| MLP | Captura interacciones no lineales entre factores de riesgo metabólico (el efecto conjunto de obesidad + sedentarismo + edad no es aditivo) | `hidden_layer_sizes=(64,32)`, `early_stopping=True`, `validation_fraction=0.1`, `max_iter=500` | 0.78–0.81 |
| K-Means (fenotipador) | Se usa para fenotipado metabólico previo a la clasificación, NO como predictor de diabetes. Descubre subtipos clínicamente interpretables de pacientes para enriquecer las features supervisadas | `n_clusters=3` (o K óptimo por silhouette), variables: BMI, GenHlth, PhysHlth, MentHlth, Age, PhysActivity, HighBP, HighChol, HeartDiseaseorAttack | Sin supervisión — evaluado por silhouette score |

**Regla crítica de diseño para K-Means:** incluir `Diabetes_binary` en el clustering introduce sesgo de confirmación. K-Means debe ajustarse exclusivamente sobre `X_train` sin la variable objetivo. El fenotipo asignado se añade luego como feature adicional al pipeline supervisado.

**Estado actual:** El fenotipador K-Means está documentado como componente de investigación, pero **no está integrado** en el pipeline de clasificación automatizado del repositorio. Su integración completa está planificada para Sprint 3 (trabajo futuro) y requiere las pruebas y la persistencia de fenotipos como artefacto intermedio.

**Actualización 2026-05-17:** el repositorio ya incluye `FenotipadoKMeans` en `entrenamiento/fenotipado.py` y `OptimizadorHiperparametros` en `entrenamiento/optimizador.py`, con pruebas unitarias en verde para ambos módulos. La integración en la documentación sigue separando el flujo supervisado del flujo de fenotipado para mantener claridad de propósito.

---

## 6. Marco de explicabilidad y equidad

### 6.1 SHAP — justificación clínica

Un modelo de caja negra será penalizado por evaluadores académicos en contexto médico. SHAP convierte las salidas de GBM y MLP en evidencia auditable que responde la pregunta del clínico: "¿Por qué dice que este paciente es de riesgo alto?"

Tres visualizaciones requeridas:
- **Summary plot** (`reportes/shap_summary.png`): importancia global de variables para el paper académico.
- **Waterfall plot** (Dashboard, por paciente): contribución individual de cada factor a la predicción específica.
- **Dependence plot** (`reportes/shap_dependence_bmi_edad.png`): análisis de interacción, recomendado para IMC × Edad, para revelar que las intervenciones de control de peso son más críticas en adultos de 40–60 años (grupos CDC 7–9), que coinciden con la mayor prevalencia en ENSANUT 2022.

### 6.2 Documentación de sesgo (CDC → México)

Presentar las limitaciones como contribuciones metodológicas diferencia un trabajo de nivel avanzado de un tutorial de scikit-learn.

**Encuadre recomendado para la sección de Discusión del paper:**
> "Este trabajo contribuye a la literatura hispanohablante con el primer análisis sistemático de la transferibilidad del dataset CDC BRFSS 2015 al contexto de salud pública mexicano, documentando las variables con mayor sesgo distribucional y proponiendo ajustes de umbral específicos para el primer nivel de atención del IMSS."

Tres limitaciones documentadas a tratar:
1. **Sesgo de selección geográfica** — Diferencias distribucionales en tabaquismo, escolaridad e ingreso.
2. **Calibración** — Probabilidades bien ordenadas (AUC alto) no implican calibración correcta para México. Brier Score > 0.15 → aplicar Platt Scaling. Umbral operativo recomendado: 0.25 (vs 0.33 predeterminado), justificado por la mayor tasa de diabetes no diagnosticada en México.
3. **Ausencia de datos de validación mexicanos** — Trabajo futuro propuesto: acceso a microdatos ENSANUT 2022, *Transfer Component Analysis* para reducir el *covariate shift*.

### 6.3 Análisis de equidad: subgrupos a evaluar

| Dimensión | Columna | Grupos |
|---|---|---|
| Sexo | `Sex` | 0 (femenino) vs 1 (masculino) |
| Edad | `Age` | Joven (1–4: 18–39) / Adulto medio (5–9: 40–64) / Mayor (10–13: 65+) |
| Ingreso | `Income` | Quintil bajo (1–2) vs quintil alto (7–8) |
| Escolaridad | `Education` | Sin educación básica (1–2) vs educación media/superior (4–6) |

**Hallazgo conocido en datos BRFSS:** los modelos de ML tienden a tener mayor tasa de falsos negativos en mujeres de ingreso bajo (sesgo de doble marginación). Debe documentarse explícitamente en el análisis de equidad del paper.

---

## 7. Benchmarks académicos

Esta tabla ancla el objetivo de desempeño para la sección de Discusión del paper.

| Estudio | Población | Dataset | Mejor modelo | ROC-AUC | Variables |
|---|---|---|---|---|---|
| Tigga & Garg (2020) | India | PIMA Indians | Random Forest | 0.82 | 8 |
| Zou et al. (2018) | China | Datos hospitalarios | Decision Tree | 0.78 | 13 |
| Huang et al. (2022) | EE. UU. (hispanos) | NHANES | SVM + features clínicos | 0.80 | 21 |
| Sisodia & Sisodia (2018) | India | PIMA Indians | Naive Bayes | 0.78 | 8 |
| **Este trabajo** | EE. UU. / proxy México | CDC BRFSS 2015 | Por determinar | **Objetivo ≥ 0.78** | 21 + fenotipo |

**Objetivo de desempeño:** superar 0.78 de ROC-AUC para ser competitivo con la literatura. GradientBoostingClassifier típicamente alcanza 0.80–0.83 en este dataset.

---

## 8. Contratos de salida y reportes

El pipeline ya no debe interpretarse como productor de reportes finales versionados. Su salida se divide en tres capas:

| Salida | Propósito | Formato | Observación |
|---|---|---|---|
| Manifiesto previo | Registrar parámetros de la corrida antes de entrenar | JSON | Útil para trazabilidad y depuración |
| Reporte crudo | Persistir métricas serializables para auditoría | JSON | Fuente técnica del informe |
| Reporte legible | Resumir los resultados en Markdown para lectura humana | MD | Se genera a partir del JSON crudo |

Los archivos de `reportes/` son artefactos derivados del pipeline. Si hace falta reconstruir un informe, se usa el JSON crudo junto con [scripts/generar_reporte_legible.py](scripts/generar_reporte_legible.py).

---

## 9. Contrato de preprocesamiento

El preprocesador se construye como un `sklearn.Pipeline` para prevenir *data leakage*. El `Pipeline` completo (preprocesador + estimador) se serializa como un único `.joblib`. `PredictorDiabetes` es compatible sin modificaciones, ya que llama a `predict_proba` sobre el objeto cargado.

| Grupo de columnas | Columnas | Transformaciones |
|---|---|---|
| Continuas | `BMI`, `MentHlth`, `PhysHlth` | `SimpleImputer(mediana)` → `StandardScaler()` |
| Binarias | `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | `SimpleImputer(moda)` → passthrough |
| Ordinales | `GenHlth`, `Age`, `Education`, `Income` | `SimpleImputer(moda)` → `OrdinalEncoder` (orden explícito) |
| Fenotipo (Sprint 3+) | `fenotipo` | `SimpleImputer(moda)` → `OrdinalEncoder` |

**Decisión de diseño:** las variables binarias no se escalan. Están en codificación {0,1} consistente; aplicar StandardScaler no aporta información y dificulta la interpretabilidad.

---

## 10. Métricas de evaluación clínica

Para un sistema de tamizaje de diabetes, la **sensibilidad (recall de clase positiva)** es más crítica que la precisión. Un falso negativo (paciente diabético clasificado como sano) tiene consecuencias clínicas más graves que un falso positivo.

| Métrica | Fórmula | Umbral aceptable | Justificación clínica |
|---|---|---|---|
| **ROC-AUC** | Área bajo curva ROC | > 0.78 | Métrica principal para comparativa con literatura |
| **Sensibilidad** | TP / (TP + FN) | > 0.70 | Detectar el 70%+ de diabéticos reales |
| **Especificidad** | TN / (TN + FP) | > 0.65 | Evitar saturar el sistema de salud con falsos positivos |
| **F1-Score (clase +)** | 2·P·R / (P+R) | > 0.55 | Balance ante desbalance de clases |
| **PR-AUC** | Área bajo curva PR | > 0.50 | Más informativo que ROC ante desbalance severo |
| **Brier Score** | MSE de probabilidades | < 0.15 | Mide calibración: ¿la probabilidad reflejada es real? |
