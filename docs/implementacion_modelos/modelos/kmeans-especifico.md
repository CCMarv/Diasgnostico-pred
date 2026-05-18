---
título: K-Means — detalles específicos
categoría: referencia
audiencia: equipo técnico
versión: 1.0.0
última actualización: 2026-05-18
autor: Equipo diasgnostico-pred
estado: revisado
---

# K-Means: Características y parámetros específicos

Esta guía complementa [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). Lee primero la guía unificada para entender el flujo general, luego usa este documento para detalles específicos de K-Means.

## Para empezar: ¿qué hace este modelo en palabras simples?

### A. Definición coloquial

Imagina que un profesor tiene 300 estudiantes y quiere formarles equipos de trabajo por afinidad, sin saber de antemano quién se lleva bien con quién. Observa sus notas, hábitos y preferencias, y agrupa a los más parecidos entre sí. K-Means hace exactamente eso con los datos de los pacientes: sin saber si tienen diabetes o no, busca grupos de personas con perfiles de salud similares. Cada grupo puede revelar un "fenotipo" clínico distinto.

### B. Por qué lo usamos aquí

K-Means permite explorar si existen subgrupos naturales de pacientes (por ejemplo, "jóvenes sedentarios con sobrepeso" vs. "adultos mayores con presión alta") que no son visibles en un análisis individual de variables, complementando los modelos supervisados.

### C. Qué significa que funcione bien o mal

- **Funciona bien**: los grupos obtenidos tienen sentido clínico: pacientes dentro de un mismo grupo se parecen entre sí en variables de salud, y los grupos son distintos entre sí. El silhouette score está por encima de 0.3.
- **Funciona mal**: los grupos se distribuyen de forma casi uniforme sin distinción real, o el número de clusters elegido no corresponde a la estructura real de los datos.

### D. Glosario

| Término | Qué significa en lenguaje simple |
|---------|----------------------------------|
| Cluster | Un grupo de pacientes con características de salud similares |
| Centroide | El "centro" matemático de un cluster; el punto promedio de todos sus miembros |
| `k-means++` | Una forma inteligente de elegir los centroides iniciales, que mejora la calidad del resultado |
| Inercia | La suma de distancias de cada paciente a su centroide; valores más bajos indican grupos más compactos |
| Método del codo | Gráfica de inercia vs. número de clusters; el "codo" sugiere el número óptimo |
| Silhouette score | Puntuación de -1 a 1 que mide qué tan bien separados están los clusters; valores cercanos a 1 son mejores |
| No supervisado | El modelo no usa la etiqueta "tiene diabetes" para agrupar; descubre estructura por sí solo |

---

## Introducción: ¿Por qué K-Means?

K-Means es un algoritmo de **clustering no supervisado** que agrupa observaciones en K clusters basándose en distancia euclidiana. En el proyecto se usa para **fenotipado**: explorar si hay grupos naturales de pacientes sin usar la variable de diabetes.

**Ventajas:**
- No requiere variable objetivo
- Explora estructura natural en los datos
- Potencial para enriquecimiento de variables

**Desventajas:**
- No predice directamente (no es supervisado)
- Evaluación más ambigua (sin "verdad")
- Necesita decisión sobre K

---

## Diferencias fundamentales respecto a modelos supervisados

| Aspecto | SVM, Árbol, GBM, MLP | K-Means |
|--------|-----------------|---------|
| **Variable objetivo** | Diabetes_binary | Ninguna (excluida) |
| **Propósito** | Predicción | Exploración, fenotipado |
| **Evaluación** | ROC-AUC | Inercia, Silhouette |
| **Consumo** | API directa | Enriquecimiento (opcional) |

---

## Parámetros específicos de K-Means

### Definición en el proyecto

```python
from sklearn.cluster import KMeans

estimador = KMeans(
    n_clusters=3,
    n_init="auto",
    random_state=42,
)
```

### Justificación de cada parámetro (detallada)

- **`n_clusters=3`**
    - Qué controla: número de grupos a detectar.
    - Por qué 3: elección exploratoria que permite distinguir subgrupos clínicos sin sobredescomponer; útil como punto de partida.
    - Validación: confirmar con Elbow plot y Silhouette; probar K en [2,6] y elegir según estabilidad y utilidad clínica.

- **`n_init='auto'`**
    - Qué controla: número de inicializaciones aleatorias (múltiples runs) para evitar mínimos locales.
    - Por qué 'auto': en versiones modernas sklearn selecciona un valor robusto; asegura mayor probabilidad de encontrar una buena solución.
    - Alternativa: `n_init=10` (compatibilidad con versiones antiguas) o valores mayores si hay alta variabilidad en resultados.

- **`random_state=42`**
    - Qué controla: semilla para inicializaciones aleatorias.
    - Por qué: reproducibilidad y trazabilidad de resultados en experimentos y reportes.

- **Parámetros complementarios**
    - `init='k-means++'` (por defecto): mejora la inicialización y reduce necesidad de muchas `n_init`.
    - `max_iter`: controla iteraciones por inicialización; valores típicos 300–600 si hay convergencia lenta.
    - `tol`: tolerancia de convergencia; valores más estrictos (p. ej. 1e-4) piden ajustes más finos.

Efecto práctico: K-Means es sensible al escalado y a la semilla; la configuración prioriza estabilidad y reproducibilidad para exploración clínica. Ajustes finos se basan en Elbow/Silhouette y en la interpretabilidad clínica de los centros.

### Comprensión intuitiva: Clustering

```
Paso 1: Elige 3 centros aleatorios
Paso 2: Asigna cada punto al centro más cercano
Paso 3: Recalcula centros como promedio de cada cluster
Paso 4: Repite pasos 2-3 hasta convergencia
```

Resultado: 3 grupos de pacientes con características similares.

---

## Elección de K: ¿Cuántos clusters?

El proyecto elige K=3 por defecto, pero hay métodos para validar:

### Método 1: Elbow Plot (codo)

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inercias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(X_train_transformado)
    inercias.append(kmeans.inertia_)

plt.plot(range(1, 11), inercias, "bo-")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia (suma de distancias)")
plt.title("Elbow plot para elegir K")
plt.axvline(x=3, color="r", linestyle="--")  # Nuestra elección
plt.show()
```

**Lectura:** Busca el "codo" donde la curva cambia de pendiente. Ese es un buen K.

### Método 2: Silhouette Score

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_train_transformado)
    score = silhouette_score(X_train_transformado, labels)
    scores.append(score)

# K con mayor silhouette score es mejor
best_k = scores.index(max(scores)) + 2
print(f"Mejor K: {best_k} (score={max(scores):.3f})")
```

**Lectura:** Silhouette score ∈ [-1, 1]. Valores altos (>0.5) indican clusters bien separados.

---

## Escalado OBLIGATORIO para K-Means

Regla práctica resumida en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md): K-Means requiere `StandardScaler` porque usa distancia euclidiana; el escalador se ajusta solo con training.

---

## Carga de datos para K-Means: SIN variable objetivo

```python
import pandas as pd
from config import COLUMNAS_CDC  # Note: COLUMNA_OBJETIVO NO se incluye

df = pd.read_csv("datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv")
X = df[list(COLUMNAS_CDC)]  # Solo las 21 variables clínicas
# NO incluir y = df[COLUMNA_OBJETIVO]
```

**Por qué excluir Diabetes_binary:**
- Evita "confirmation bias" → el clustering no es sesgado por la variable objetivo
- Explora estructura natural en las variables clínicas
- Permite comparar: "¿Los clusters coinciden con diabetes?"

---

## Diferencia clave: Evaluación sin etiquetas

Los modelos supervisados usan ROC-AUC (comparan predicción vs. verdad).

K-Means no tiene "verdad" (es no supervisado). Usa métricas intrínsecas:

### Inercia (intra-cluster sum of squares)

```python
print(f"Inercia del modelo: {kmeans.inertia_:.2f}")
```

**Interpretación:** Suma de distancias de cada punto a su centro. **Menor = mejor** (puntos más cercanos a centros).

**Limitación:** Siempre baja con más clusters. No es directamente comparable entre K distintas.

### Silhouette Score

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

**Interpretación:** ∈ [-1, 1]. Mide cuán bien asignados están los puntos:
- **> 0.5:** Clusters bien definidos
- **0.3-0.5:** Clusters razonables
- **< 0.3:** Clusters débiles

---

## Interpretación de clusters: ¿Qué significan?

Después de entrenar, inspecciona los centros:

```python
import pandas as pd

# Centros de clusters (en escala transformada)
centros_transformados = kmeans.cluster_centers_

# Si usaste StandardScaler, invierte para escala original
centros_originales = scaler.inverse_transform(centros_transformados)

df_centros = pd.DataFrame(
    centros_originales,
    columns=nombres_columnas
)
print(df_centros)
```

**Lectura:**
- Compara valores en cada cluster
- Ej: "Cluster 0 tiene BMI promedio=28, MentHlth=10 → mayor riesgo probable"
- "Cluster 1 tiene BMI=23, MentHlth=0 → menor riesgo probable"

---

## Uso de clusters para enriquecimiento (opcional)

K-Means puede enriquecer el modelo supervisado añadiendo el cluster como variable:

```python
# Paso 1: Entrenar K-Means (no supervisado)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_transformado)

# Paso 2: Agregar como nueva variable
X_train_enriquecido = X_train.copy()
X_train_enriquecido["cluster"] = clusters

# Paso 3: Entrenar modelo supervisado con la nueva variable
pipeline_mejorado.fit(X_train_enriquecido, y_train)
```

**Nota:** El proyecto por defecto **no hace** este enriquecimiento. Es una extensión opcional para investigación.

## Coherencia con el código y cómo ejecutar (notas para no-programadores)

- Implementación real: el pipeline soporta un modo `clustering` que carga el dataset sin la variable objetivo, aplica limpieza básica y entrena K-Means usando `ComparadorModelos.entrenar_clustering`.
- Comando para ejecutar desde la raíz del repositorio:

```bash
python -m entrenamiento.pipeline --modo clustering --clusters 3
```

- Resultado: el pipeline guardará el modelo K-Means serializado en la ruta configurada (por defecto `modelos/modelo_fenotipo_kmeans.joblib`) y escribirá un reporte JSON con el puntaje (`inercia`) en `reportes/`.

Explicación en lenguaje llano:

- "Si quieres agrupar pacientes en 3 fenotipos usando el mismo flujo del proyecto, ejecuta el pipeline en modo `clustering`. El sistema limpiará los datos, hará el clustering y guardará el modelo. Esto no afecta los modelos supervisados ni los reemplaza; es una tarea separada de análisis exploratorio." 

Acción recomendada: si planeas usar los clusters como nueva variable en training supervisado, documenta ese experimento y persiste el modelo K-Means versionado para reproducibilidad.

Actualización del repositorio: el helper formal `FenotipadoKMeans` vive en `entrenamiento/fenotipado.py` y tiene cobertura en `pruebas/test_fenotipado.py`, así que este documento describe la implementación real y no solo el flujo didáctico.

---

## Diferencias respecto a sklearn básico

| Aspecto | sklearn básico | Proyecto |
|--------|---------------|----------|
| **n_clusters** | 8 (por defecto) | 3 (exploración) |
| **n_init** | 10 | "auto" (adaptativo) |
| **Escalado** | Responsabilidad del usuario | Encapsulado en Pipeline |
| **Variable objetivo** | N/A | Excluida intencionalmente |

---

## Cuándo usar K-Means

1. **Exploración:** "¿Hay grupos naturales en los datos clínicos?"
2. **Validación:** "¿Los clusters encontrados coinciden con diabetes?"
3. **Segmentación:** "¿Podemos identificar subgrupos de riesgo?"
4. **Investigación:** "¿Agregar fenotipos como variable mejora predicción?"

---

## Referencia de capas en el proyecto

1. **Definición:** [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) - instancia KMeans
2. **Preprocesamiento:** [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) - ColumnTransformer + StandardScaler (obligatorio)
3. **Entrenamiento:** [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py) - fit sin supervisión
4. **Serialización:** `modelos/predictor_fenotipo_kmeans.joblib`
5. **Uso** (opcional): Enriquecimiento de variables o análisis exploratorio

---


## Lectura complementaria

- Resúmenes clave integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) (escalado y principios de distancia).
- **Sklearn KMeans:** [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- **Silhouette Score:** [Documentación](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

---

## Volver a la guía unificada

Para entender el flujo completo: [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)
