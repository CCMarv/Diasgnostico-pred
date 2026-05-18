---
título: Guía unificada — catálogo de modelos
categoría: referencia
audiencia: equipo técnico
versión: 1.0.0
última actualización: 2026-05-18
autor: Equipo diasgnostico-pred
estado: revisado
---

# Guía Unificada: Catálogo de Modelos de Predicción de Diabetes

Esta es la **referencia central única** para entender cómo todos los modelos en el proyecto siguen un flujo común. Cada modelo (SVM, Árbol, GBM, MLP, K-Means) se implementa bajo los mismos principios arquitectónicos.

**Tabla de modelos disponibles:**

| Modelo | Tipo | Caso de uso | Características |
|--------|------|-------------|-----------------|
| SVM | Supervisado | Baseline kernelizado | Requiere escalado, GridSearchCV |
| Árbol | Supervisado | Interpretabilidad | Sin escalado obligatorio, baseline simple |
| GBM | Supervisado | Máxima capacidad predictiva | Ensamble de árboles secuenciales |
| MLP | Supervisado | No linealidad compleja | Red neuronal, escalado obligatorio |
| K-Means | No supervisado | Fenotipado, exploración | Clustering, sin variable objetivo |

**Para navegar:** Si necesitas detalles específicos de un modelo, ve a `modelos/{nombre}-especifico.md`.

---

## Cómo usar esta documentación

Esta guía está organizada en capas:

- **Capas 1-8:** Describen el flujo común de todos los modelos (con variaciones según tipo)
- **Referencia a modelos específicos:** Cada sección incluye enlaces a `modelos/{nombre}-especifico.md`
- **Material histórico:** Las explicaciones didácticas originales han sido resumidas e integradas en este documento.

Además, el repositorio ya cuenta con los helpers formales `FenotipadoKMeans` (`entrenamiento/fenotipado.py`) y `OptimizadorHiperparametros` (`entrenamiento/optimizador.py`) para que el flujo descrito aquí tenga una implementación concreta y testeada.

La salida del entrenamiento sigue una regla de dos capas: primero se guarda un JSON crudo para auditoría y trazabilidad; después se sintetiza un Markdown legible para humanos. Esa síntesis vive en `entrenamiento/generador_reportes.py` y se puede reconstruir con `scripts/generar_reporte_legible.py`.

Puedes leer esta guía completa de principio a fin, o saltear directamente a la capa que necesites.

---

## Resúmenes embebidos (conceptos históricos)

He incluido aquí los resúmenes esenciales de los materiales didácticos históricos. La documentación ahora contiene las reglas prácticas necesarias sin depender de carpetas externas.

- Escalado: cuándo es obligatorio

  - SVM: escalado obligatorio. SVM usa distancias; sin escalado variables dominantes sesgan el hiperplano.
  - MLP: escalado obligatorio. Redes usan descenso de gradiente: columnas en distintas escalas ralentizan o impiden la convergencia.
  - K-Means: escalado obligatorio. K-Means usa distancia euclidiana.
  - Árbol y GBM: escalado no obligatorio. Los modelos basados en árboles funcionan por umbrales; escalar no cambia esos cortes.

- Manejo del desbalance (resumen)

  - Problema: la clase positiva (`Diabetes_binary=1`) es minoritaria. `accuracy` puede ser engañosa.
  - Métrica preferida: `ROC-AUC` para ordenamiento global; `PR-AUC` si se quiere enfocar en positivos.
  - Estrategias de corrección: `class_weight='balanced'` (ajusta pesos en el estimador), remuestreo (SMOTE/undersampling) o pipelines con `imblearn` cuando se requiere.
  - Regla práctica: priorizar métricas (ROC/PR) y usar `class_weight` como primer paso no invasivo.

- Intuición SVM (resumen breve)

  - SVM busca un hiperplano que maximice el margen entre clases. `C` controla la tolerancia a errores (regularización); `gamma` controla la complejidad local del kernel RBF.

- Explicabilidad recomendada

  - Para modelos complejos (GBM, MLP) usar SHAP para explicar contribuciones de variables a predicciones individuales. Para árboles se pueden extraer reglas y visualizaciones.

Estos resúmenes garantizan que la documentación sea autónoma sin depender de materiales externos históricos.

---

# Flujo común de modelos supervisados (SVM, Árbol, GBM, MLP)

Todos los modelos supervisados siguen estos 8 pasos. Las variaciones específicas están documentadas en `modelos/{nombre}-especifico.md`.

## 1. Carga de datos

### Qué se hace

Se parte de un conjunto de datos tabular, se separan las variables de entrada y la variable objetivo, y se prepara la matriz de entrenamiento.

### Por qué se hace

El contrato de datos es **único y compartido** por todos los modelos. Aunque cada algoritmo funciona de forma distinta internamente, todos consumen el mismo formato de entrada: 21 columnas CDC + 1 objetivo.

### Cómo se hace en el proyecto

El contrato real de datos está definido en [config.py](../../config.py):

```python
import pandas as pd
from config import COLUMNAS_CDC, COLUMNA_OBJETIVO

df = pd.read_csv("datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv")
X = df[list(COLUMNAS_CDC)]  # 21 variables clínicas
y = df[COLUMNA_OBJETIVO]     # Diabetes_binary (0 o 1)
```

**Contrato CDC:**
- **21 variables de entrada (COLUMNAS_CDC):**
  - Continuas: BMI, MentHlth, PhysHlth
  - Binarias: HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
  - Ordinales: GenHlth, Age, Education, Income
- **1 variable objetivo (COLUMNA_OBJETIVO):**
  - Diabetes_binary: 0 = sin diabetes, 1 = con diabetes (desbalanceado)

### Cuándo usar solo esta parte

Usa esta sección cuando ya tienes el dataset listo y solo quieres construir `X` e `y`.

---

## 2. Limpieza y preprocesamiento

### Qué se hace

Se imputan valores faltantes y se transforman las variables según su tipo, sin contaminar los datos de prueba o inferencia.

### Por qué se hace

La imputación y el escalado deben aprenderse **solo con los datos de entrenamiento** y luego aplicarse de forma consistente a validación, prueba e inferencia. Esto evita **data leakage**.

### Estrategia común de preprocesamiento

El proyecto usa un `ColumnTransformer` que separa las columnas por tipo:

| Tipo | Ejemplo | Imputación | Transformación |
|------|---------|-----------|-----------------|
| **Continuas** | BMI, MentHlth, PhysHlth | Mediana | StandardScaler (depende del modelo) |
| **Binarias** | HighBP, Sex, Smoker, etc. | Moda | Passthrough (sin transformación) |
| **Ordinales** | GenHlth, Age, Education, Income | Moda | OrdinalEncoder (con orden explícito) |

### Por qué se usan estos parámetros

- **`strategy="median"`** en continuas: resiste mejor valores atípicos que la media
- **`strategy="most_frequent"`** en binarias y ordinales: mantiene una categoría válida
- **`StandardScaler`** en continuas: depende del modelo (ver **Tabla de escalado** en `notas_escalar_o_no_escalar.md`)
  - SVM: **obligatorio** (sensible a distancias)
  - Árbol: **opcional** (trabaja por umbrales)
  - GBM: **opcional** (basado en árboles)
  - MLP: **obligatorio** (redes neuronales convergen mejor escaladas)
  - K-Means: **obligatorio** (distancia euclidiana)
- **`OrdinalEncoder`** en ordinales: preserva el orden semántico (GenHlth: "Excellent" < "Very Good" < ... < "Poor")

### Cómo se hace en el proyecto

```python
from entrenamiento.preprocesador import ConstructorPreprocesador

# El preprocesador construye un ColumnTransformer específico para cada modelo
preprocesador = ConstructorPreprocesador()
pipeline = preprocesador.construir_pipeline(clasificador)
```

El `Pipeline` completo encapsula:
1. Transformación (preprocesamiento)
2. Clasificador (modelo)

Todo se entrena junto en los datos de entrenamiento y se serializa como un único artefacto.

### Cuándo usar solo esta parte

Usa esta sección cuando ya tienes `X` e `y`, pero todavía no quieres entrenar; solo necesitas el flujo de transformación.

---

## 3. Definición del modelo

### Qué se hace

Se instancia el clasificador con sus hiperparámetros principales.

### Por qué se hace

Cada modelo tiene un conjunto de parámetros que controlan su comportamiento. La elección de estos parámetros refleja **decisiones de diseño explícitas** sobre cuánto deben penalizarse los errores, cuán flexible debe ser la frontera, etc.

### Parámetros comunes a todos

- **`random_state=42`:** Asegura reproducibilidad en todas las ejecuciones
- **`class_weight="balanced"` (opcional según modelo):** Compensa el desbalance de la clase positiva
- **`probability=True` (si aplica):** Permite obtener probabilidades además de clases

### Ver detalles específicos de cada modelo

Para parámetros concretos y su justificación detallada:
- SVM: [modelos/svm-especifico.md](modelos/svm-especifico.md#3-definición-del-modelo-svm)
- Árbol: [modelos/arbol-especifico.md](modelos/arbol-especifico.md#3-definición-del-modelo-árbol-de-decisión)
- GBM: [modelos/gbm-especifico.md](modelos/gbm-especifico.md#3-definición-del-modelo-gbm)
- MLP: [modelos/mlp-especifico.md](modelos/mlp-especifico.md#3-definición-del-modelo-mlp)

### Cuándo usar solo esta parte

Usa esta sección cuando ya tienes el preprocesamiento resuelto y solo quieres entender la definición general del modelo.

---

## 4. Construcción del Pipeline

### Qué se hace

Se construye un `Pipeline` que encapsula preprocesamiento + clasificador en un solo objeto.

### Por qué se hace

Un `Pipeline` garantiza que:
- El preprocesamiento se ajuste solo con entrenamiento
- Los mismos pasos se apliquen a validación, prueba e inferencia
- Se pueda serializar el objeto completo con `joblib`

### Cómo se hace en el proyecto

```python
from entrenamiento.preprocesador import ConstructorPreprocesador
from sklearn.svm import SVC  # O el modelo que uses

pre = ConstructorPreprocesador()
modelo = SVC(...)  # O DecisionTreeClassifier, GradientBoostingClassifier, etc.

pipeline = pre.construir_pipeline(modelo)
```

El `pipeline` resultante es el artefacto **entrenable y serializable** que se usará en los siguientes pasos.

### Estructura interna del Pipeline

```
Pipeline(
  steps=[
    ('preprocesador', ColumnTransformer(...)),
    ('clasificador', [SVC|DecisionTreeClassifier|GradientBoostingClassifier|MLPClassifier])
  ]
)
```

### Cuándo usar solo esta parte

Usa esta sección cuando quieres construir el objeto entrenable final, pero todavía no ejecutar entrenamiento.

---

## 5. Entrenamiento

### Qué se hace

Se entrena el modelo sobre el conjunto de entrenamiento con validación cruzada para evaluar su capacidad de generalización.

### Por qué se hace

Entrenar una sola vez sobre todo el conjunto no basta para evaluar generalización. La **validación cruzada estratificada** permite:
- Usar múltiples pliegues (splits) de los datos
- Mantener la proporción de clases en cada pliegue
- Obtener una estimación más robusta del rendimiento
- Comparar modelos de manera justa

### Estrategia común de validación

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

puntajes = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
```

**Por qué estos parámetros:**

- **`n_splits=5`:** Equilibrio entre estabilidad y costo computacional. 5 pliegues dan estimaciones confiables sin ser excesivamente caros.
- **`shuffle=True`:** Reduce el riesgo de que el orden original del dataset sesgue los pliegues.
- **`stratify=y`** (en train_test_split): Preserva la proporción de clases entre entrenamiento y prueba.
  - **`scoring="roc_auc"`:** ROC-AUC es la métrica principal porque:
  - El problema es **binario desbalanceado**
  - Accuracy puede ser engañosa (ej: decir "no diabetes" en todos los casos daría 90% accuracy)
  - ROC-AUC mide bien la capacidad de **ordenar** riesgo, independientemente del umbral
    - Ver los resúmenes embebidos en este mismo documento (sección "Resúmenes embebidos").
- **`n_jobs=-1`:** Paraleliza la evaluación cruzada para reducir tiempo.

### Búsqueda de hiperparámetros (opcional por modelo)

Algunos modelos (SVM) usan `GridSearchCV` para optimizar ciertos parámetros:

```python
from sklearn.model_selection import GridSearchCV

grid_params = {
    "clasificador__C": [0.1, 1, 10],
    "clasificador__gamma": ["scale", "auto"],
}

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
busqueda.fit(X_train, y_train)
```

**Nota:** Los parámetros específicos varían según el modelo. Ver detalles en `modelos/{nombre}-especifico.md`.

### Cuándo usar solo esta parte

Usa esta sección cuando ya definiste el modelo y el preprocesamiento, y ahora quieres obtener un artefacto entrenado.

---

## 6. Interpretación, Explicabilidad y Visualización

### Qué se hace

Se inspecciona el modelo entrenado para entender cómo toma decisiones.

### Por qué se hace

La interpretabilidad es importante para:
- **Validación clínica:** ¿Tiene sentido que estas variables influyan?
- **Debugging:** ¿El modelo está aprendiendo características esperadas?
- **Explicación de predicciones:** "¿Por qué el modelo predice riesgo alto para este paciente?"

### Métodos por modelo

La interpretabilidad **varía ampliamente** según el algoritmo:

- **Árbol:** Reglas directas (`export_text`), inspección del árbol
- **GBM:** Importancia de variables, SHAP (herramienta de explicabilidad avanzada)
- **SVM:** Visualización 3D de hiperplano (didáctico), análisis de vectores de soporte
- **MLP:** SHAP, análisis de activaciones
- **K-Means:** Centros de cluster, análisis silhouette

### Recomendación general

Para contexto **clínico auditable**, el proyecto recomienda **SHAP** (ver [PROYECTO.md](../../PROYECTO.md)):

```python
import shap

# Una vez entrenado el pipeline, extrae el clasificador
clasificador = pipeline.named_steps["clasificador"]

# SHAP Explainer
explainer = shap.Explainer(clasificador, X_train_transformado)
shap_values = explainer(X_sample_transformado)

# Visualizaciones útiles
shap.summary_plot(shap_values, X_sample_transformado, feature_names=nombres_columnas)
shap.waterfall_plot(shap_values[0])  # Explicación de una predicción
```

### Ver detalles específicos

Para métodos de interpretación concretos:
- Árbol: [modelos/arbol-especifico.md](modelos/arbol-especifico.md#6-interpretación-del-árbol)
- GBM: [modelos/gbm-especifico.md](modelos/gbm-especifico.md#6-explicabilidad-del-gbm)
- MLP: [modelos/mlp-especifico.md](modelos/mlp-especifico.md#6-interpretación-y-explicabilidad)

### Cuándo usar solo esta parte

Usa esta sección cuando quieras explicar el comportamiento del modelo o preparar una lectura auditable.

---

## 7. Entrenamiento productivo completo

### Qué se hace

Se divide el dataset en entrenamiento y prueba, se comparan varios modelos, se elige el mejor y se guarda el pipeline final.

### Por qué se hace

En el proyecto **ningún modelo vive solo**. Todos compiten bajo el mismo criterio (ROC-AUC) para decidir cuál es verdaderamente el mejor para el problema clínico.

### Orquestación en el proyecto

```python
from sklearn.model_selection import train_test_split

# Separación estratificada: mantiene proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# El comparador entrena todos los modelos del catálogo
resultados = comparador.entrenar_clasificacion(X_train, y_train)

# Selecciona el mejor según ROC-AUC
mejor_modelo = comparador.seleccionar_mejor(resultados)

# Evalúa en el conjunto de prueba (nunca visto durante entrenamiento)
score_test = mejor_modelo.score(X_test, y_test)
```

El flujo está orquestado en [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py).

### Serialización del mejor modelo

```python
import joblib

ruta_modelo = "modelos/predictor_production.joblib"
joblib.dump(mejor_modelo.modelo, ruta_modelo)
```

El artefacto guardado ya incluye **preprocesamiento + clasificador** en un solo `Pipeline`.

### Cuándo usar solo esta parte

Usa esta sección cuando ya no estás haciendo experimentación local, sino entrenamiento reproducible para generar artefactos de producción.

---

## 8. Carga e inferencia

### Qué se hace

Se carga el archivo `.joblib` y se usa para predecir sobre datos nuevos.

### Por qué se hace

La API y el sistema de inferencia no deben saber si el modelo es SVM, árbol o GBM. Solo necesitan:
- Cargar el artefacto
- Pasar datos nuevos
- Obtener predicciones

### Cómo se hace en el proyecto

```python
from inferencia.predictor import PredictorDiabetes

# Carga el modelo entrenado
predictor = PredictorDiabetes()
predictor.cargar_modelo()

# Realiza predicción
resultado = predictor.predecir(df_paciente)
```

La lógica real está en [inferencia/predictor.py](../../inferencia/predictor.py).

### Cómo lo expone la API

[api/main.py](../../api/main.py) convierte la probabilidad en una categoría de riesgo:

```python
@app.post("/prediccion")
def predecir(datos: EsquemaEntrada):
    resultado = predictor.predecir(datos)
    # Convierte probabilidad a riesgo bajo/medio/alto
    categoria = categorizar_riesgo(resultado["probabilidad"])
    return {"riesgo": categoria, "probabilidad": resultado["probabilidad"]}
```

### Cuándo usar solo esta parte

Usa esta sección cuando el modelo ya existe en disco y solo necesitas consumirlo desde la API o desde scripts de batch.

---

## Flujo completo: cómo entra todo junto

Independientemente del modelo específico que uses, el flujo es el mismo:

1. **Define el contrato** en [config.py](../../config.py)
2. **Prepara la transformación** en [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py)
3. **Registra el modelo** en [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py)
4. **Orquesta entrenamiento** en [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py)
5. **Serializa el mejor** con `joblib`
6. **Carga para inferencia** en [inferencia/predictor.py](../../inferencia/predictor.py)
7. **Expone desde API** en [api/main.py](../../api/main.py)

---

# Flujo especial: K-Means (no supervisado)

K-Means es **diferente** porque no tiene variable objetivo. Se usa para **fenotipado** (exploración de grupos de pacientes).

## Diferencias con modelos supervisados

| Aspecto | Supervisados | K-Means |
|---------|-------------|---------|
| Variable objetivo | Sí (Diabetes_binary) | No |
| Evaluación | ROC-AUC | Inercia, Silhouette |
| Propósito | Predicción de diabetes | Agrupación de fenotipos |
| Uso en flujo | Predicción directa | Enriquecimiento de variables (opcional) |

## Pasos comunes con supervisados (1-5, 7-8)

Los pasos 1, 2, 4, 5, 7, 8 son iguales para K-Means. La diferencia está en:
- **Paso 1:** Excluye la variable objetivo (`Diabetes_binary`) de los datos
- **Paso 6:** Usa evaluación por inercia/silhouette en lugar de ROC-AUC

## Ver detalles de K-Means

[modelos/kmeans-especifico.md](modelos/kmeans-especifico.md)

---

## Referencias útiles compartidas

- Contrato de datos: [config.py](../../config.py)
- Catálogo de modelos: [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py)
- Preprocesamiento: [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py)
- Orquestación de entrenamiento: [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py)
- Inferencia: [inferencia/predictor.py](../../inferencia/predictor.py)
- API: [api/main.py](../../api/main.py)
- Decisiones del proyecto: [PROYECTO.md](../../PROYECTO.md)

---

## Plantilla reutilizable para futuros modelos

Si añades un nuevo modelo al catálogo, documéntalo siguiendo esta estructura:

1. **Qué resuelve el modelo** (caso de uso, ventajas)
2. **Parámetros específicos** (con justificación de valores)
3. **Diferencias respecto a baseline** (qué cambia)
4. **Interpretabilidad** (cómo leerlo)
5. **Ejemplo de uso** (código completo)

Mantén las referencias a esta guía unificada y evita duplicar las explicaciones de pasos comunes (carga, preprocesamiento, validación cruzada, etc.).

---

## Índice de modelos específicos

Cada modelo tiene su propia guía con detalles concretos:

- [SVM: Máquinas de Vectores de Soporte](modelos/svm-especifico.md)
- [Árbol: Interpretabilidad directa](modelos/arbol-especifico.md)
- [GBM: Gradient Boosting Machines](modelos/gbm-especifico.md)
- [MLP: Redes Neuronales](modelos/mlp-especifico.md)
- [K-Means: Fenotipado no supervisado](modelos/kmeans-especifico.md)


---

## 📖 Glosario para estudiantes

Esta sección define los términos técnicos más usados en toda la documentación. Está escrita para alguien que está aprendiendo ciencia de datos por primera vez.

| Término | Definición simple |
|---------|------------------|
| **Pipeline** | Una cadena de pasos automáticos: primero limpias los datos, luego los transformas, luego entrenas el modelo. Todo en orden y sin que tengas que hacerlo manualmente cada vez. |
| **Data Leakage** | Cuando el modelo "hace trampa" al entrenarse: ve información del futuro o del conjunto de prueba antes de tiempo. Es como estudiar con las respuestas del examen ya en mano. |
| **Validación cruzada** | En lugar de probar el modelo una sola vez, lo pruebas 5 veces con diferentes partes de los datos. Así sabes si funciona bien en general, no solo con un grupo específico. |
| **ROC-AUC** | Una puntuación entre 0 y 1 que mide qué tan bien el modelo separa a los enfermos de los sanos. 0.5 = no mejor que lanzar una moneda. 1.0 = perfecto. En este proyecto buscamos > 0.78. |
| **PR-AUC** | Similar a ROC-AUC pero más útil cuando hay muchos más sanos que enfermos (como en este proyecto). Mide específicamente qué tan bien el modelo detecta los casos de diabetes. |
| **Serialización** | Guardar el modelo entrenado en un archivo (`.joblib`) para poder usarlo después sin volver a entrenarlo. Como guardar una partida de videojuego. |
| **Desbalance de clases** | Cuando hay muchos más casos de una clase que de otra. Aquí: ~86% sin diabetes vs ~14% con diabetes. Sin corrección, el modelo aprende a ignorar los casos de diabetes. |
| **SMOTE** | Técnica para crear pacientes "sintéticos" con diabetes a partir de los reales, para que el modelo tenga más ejemplos de la clase minoritaria con qué aprender. |
| **Hiperparámetros** | Configuraciones del modelo que el programador elige antes del entrenamiento. No son aprendidos por el modelo; son como las reglas del juego que tú defines. |
| **Sensibilidad / Recall** | De todos los pacientes que realmente tienen diabetes, ¿qué porcentaje detectó el modelo? Alta sensibilidad = pocos diabéticos no detectados. |
| **Especificidad** | De todos los pacientes que realmente NO tienen diabetes, ¿qué porcentaje clasificó el modelo como sanos? Evita alarmar innecesariamente a pacientes sanos. |
