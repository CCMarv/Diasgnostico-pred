---
título: MLP — detalles específicos
categoría: referencia
audiencia: equipo técnico
versión: 1.0.0
última actualización: 2026-05-18
autor: Equipo diasgnostico-pred
estado: revisado
---

# MLP: Características y parámetros específicos

Esta guía complementa [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). Lee primero la guía unificada para entender el flujo general (secciones 1-8), luego usa este documento para detalles específicos de MLP.

## Para empezar: ¿qué hace este modelo en palabras simples?

### A. Definición coloquial

El cerebro humano tiene millones de neuronas conectadas entre sí: cuando ves algo, las señales viajan de neurona en neurona hasta que tu cerebro decide qué es. Un Perceptrón Multicapa (MLP) imita esa estructura con capas de nodos matemáticos. La primera capa recibe las 21 variables de salud; las capas del medio las transforman y combinan; la última capa emite la probabilidad de diabetes. El modelo ajusta todas esas conexiones mirando los errores que comete, igual que tú aprendes de tus equivocaciones.

### B. Por qué lo usamos aquí

El MLP puede capturar relaciones extremadamente complejas entre variables, lo que lo hace útil cuando las interacciones entre factores de riesgo no siguen patrones simples. Representa la mayor complejidad del catálogo supervisado del proyecto.

### C. Qué significa que funcione bien o mal

- **Funciona bien**: el modelo aprende representaciones internas útiles y su ROC-AUC es comparable o superior a GBM, demostrando que las capas ocultas capturaron patrones reales.
- **Funciona mal**: el entrenamiento oscila sin converger, o el modelo memoriza el conjunto de entrenamiento. El parámetro `early_stopping=True` ayuda a detectar esto automáticamente.

### D. Glosario

| Término | Qué significa en lenguaje simple |
|---------|----------------------------------|
| Capa oculta | Una fila de nodos matemáticos entre la entrada y la salida; aquí hay dos capas de 64 y 32 nodos |
| Activación ReLU | Una función que "enciende" una neurona solo si su valor es positivo, introduciendo no-linealidad |
| Adam | El algoritmo que ajusta los pesos de la red; es eficiente y adapta el tamaño del paso automáticamente |
| `early_stopping` | El modelo para de entrenar cuando deja de mejorar, evitando que memorice los datos |
| Épocas | Cuántas veces el modelo recorre el conjunto de entrenamiento completo durante el ajuste |
| Pesos | Los números internos de la red que se ajustan durante el aprendizaje |
| `validation_fraction` | La fracción de datos reservada para detectar cuándo el modelo empieza a sobreajustar |

---

## Introducción: ¿Por qué MLP?

El Perceptrón Multicapa (MLP) es una **red neuronal con capas ocultas**. Captura relaciones **altamente no lineales** entre variables. En el proyecto representa la mayor complejidad del catálogo supervisado.

**Ventajas:**
- Captura no linealidad profunda
- Flexible, menos sesgo inductivo que árboles
- Escalable a muchas variables

**Desventajas:**
- Caja negra (menos interpretable que Árbol)
- Requiere escalado obligatorio (convergencia)
- Hiperparámetros más frágiles

---

## Parámetros específicos de MLP

### Definición en el proyecto

```python
from sklearn.neural_network import MLPClassifier

estimador = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)
```


### Justificación de cada parámetro (detallada)

- **`hidden_layer_sizes=(64, 32)`**
    - Qué controla: la arquitectura de la red (número de neuronas por capa oculta).
    - Por qué (64,32): estructura piramidal que reduce dimensionalidad progresivamente, suficiente capacidad para capturar interacciones sin sobredimensionar.
    - Rango recomendado: probar `[(32,16), (64,32), (128,64)]` en experimentos; redes más grandes requieren más regularización y datos.
    - Efecto práctico: más neuronas → mayor capacidad de representación → riesgo de overfitting si no hay regularización/early stopping.

- **`activation='relu'`**
    - Qué controla: función de activación usada por las neuronas.
    - Por qué ReLU: eficaz en práctica, evita saturación de gradiente en comparación con sigmoid/tanh; eficiente computacionalmente.
    - Alternativas: `tanh`/`elu` si se prefieren funciones con salida centrada o para casos con gradientes específicos.

- **`solver='adam'`**
    - Qué controla: algoritmo de optimización/actualización de pesos.
    - Por qué Adam: adaptativo por parámetro, converge bien con poco ajuste de tasa de aprendizaje; robusto para arquitecturas pequeñas-medias.
    - Alternativa: `sgd` con momentum para control fino de learning rate en experimentos avanzados.

- **`max_iter=500`**
    - Qué controla: iteraciones máximas del algoritmo de entrenamiento.
    - Por qué 500: con `early_stopping=True` previene cortes prematuros; permite que la red converge si el dataset lo requiere.
    - Efecto práctico: si `early_stopping=False` puede necesitarse aumentar `max_iter`.

- **`early_stopping=True`, `validation_fraction=0.1`**
    - Qué controla: parada automática basada en la mejora en un subconjunto de validación interno.
    - Por qué: evita sobreajuste sin necesidad de un pipeline de validación externo; `validation_fraction=0.1` reserva 10% del training para esta validación.
    - Efecto práctico: acelera la detección de estancamiento y mejora generalización.

- **`random_state=42`**
    - Reproducibilidad de inicialización de pesos y particiones internas.

- **Consideraciones adicionales**
    - `alpha` (regularización L2): controles comunes 1e-5–1e-3 para penalizar pesos grandes.
    - `learning_rate_init`: ajuste fino del comportamiento de `adam`; valores típicos 0.001–0.01.
    - `batch_size`: afecta estabilidad y velocidad; en `MLPClassifier` por defecto es 'auto', considerar tamaños 32–256 según memoria.

Resumen: la configuración busca estabilidad (ReLU + Adam), control de sobreajuste (early_stopping, arquitectura moderada) y reproducibilidad. Ajustar `alpha` y `learning_rate_init` para optimizar generalización si es necesario.

### Comprensión intuitiva: Capas ocultas

```
Entrada (21 variables)
    ↓
Capa oculta 1 (64 neuronas) ← aprende características de orden 1
    ↓
Capa oculta 2 (32 neuronas) ← aprende características de orden 2 (combinaciones)
    ↓
Salida (1 neurona) ← probabilidad de diabetes
```

**Por qué (64, 32) y no (128, 64) o (32, 16)?**
- (64, 32) es un balance: suficiente capacidad sin memorizar
- Pirámide (baja de 64 a 32): fuerza compresión de información

---

## Escalado OBLIGATORIO para MLP

Ver los resúmenes embebidos en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). En síntesis: las redes requieren `StandardScaler` ajustado solo con training; sin esto la convergencia es inestable.

---

## Early Stopping: Regularización automática

MLP es propenso a sobreajuste. El parámetro `early_stopping=True` implementa un mecanismo de parada automática:

```
Epoch 1:   Training loss=0.50, Validation loss=0.48 ✓ (mejora)
Epoch 2:   Training loss=0.40, Validation loss=0.39 ✓ (mejora)
Epoch 3:   Training loss=0.35, Validation loss=0.39 ✗ (empeora, detenerse)
```

Sin early_stopping, el modelo seguiría entrenando y sobreadjustaría en los datos de entrenamiento.

---

## Referencia teórica: Redes Neuronales

Conceptos clave resumidos en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md): neurona = $y=\sigma(w\cdot x + b)$, ReLU introduce no linealidad, las capas permiten composición de funciones y resolver problemas como XOR.

---

## Diferencias respecto a sklearn básico

| Aspecto | sklearn básico | Proyecto |
|---------|---------------|----------|
| **Capas** | `hidden_layer_sizes=(100,)` por defecto | `(64, 32)` |
| **Parada** | Converge o agota max_iter | early_stopping automático |
| **Escalado** | Responsabilidad del usuario | Encapsulado en Pipeline |
| **Reproducibilidad** | Frágil | random_state fijo |

---

## Búsqueda de hiperparámetros (avanzado)

El proyecto NO hace GridSearchCV por defecto. Si quisieras optimizar:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid_params = {
    "clasificador__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
    "clasificador__learning_rate_init": [0.001, 0.01],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
busqueda.fit(X_train, y_train)
```

**Nota:** Muy caro computacionalmente. El proyecto usa valores por defecto.

---

## Explicabilidad: SHAP también para MLP

Aunque MLP es menos interpretable que Árbol, SHAP puede explicar predicciones:

```python
import shap

clasificador = pipeline.named_steps["clasificador"]
explainer = shap.Explainer(clasificador, X_train_transformado)
shap_values = explainer(X_sample_transformado)

shap.summary_plot(shap_values, X_sample_transformado,
                  feature_names=nombres_columnas)
```

SHAP "abre la caja negra" al aproximar la contribución de cada variable a la predicción.

---

## Cuándo preferir MLP

- **Datos altamente no lineales:** Si relaciones simples no capturan el patrón
- **Comparación:** Saber cuál es el máximo rendimiento posible
- **Investigación:** Explorar si la no linealidad profunda ayuda

---

## Coherencia con el código y notas para no-programadores

- Implementación real: en `entrenamiento/comparador_modelos.py` el `MLPClassifier` se instancia con `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `solver='adam'`, `max_iter=500`, `early_stopping=True`, `validation_fraction=0.1` y `random_state` fijo.

Explicación en lenguaje llano:

- "La red tiene dos capas ocultas (64 y 32 neuronas). Usa ReLU y el optimizador Adam. Para evitar que siga entrenando indefinidamente se activa `early_stopping`, que detiene el entrenamiento cuando la mejora en un subconjunto de validación se estanca." 

Acción recomendada: documentar en experimentos cualquier cambio en `hidden_layer_sizes`, `alpha` o `learning_rate_init`, ya que estos impactan fuertemente convergencia y generalización.

Actualización del repositorio: la implementación formal sigue en `entrenamiento/comparador_modelos.py`, y las pruebas del ajuste de flujo para el sprint 3 quedan reflejadas en la suite del repositorio.

## Referencia de capas en el proyecto

1. **Definición:** [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) - instancia MLPClassifier
2. **Preprocesamiento:** [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) - ColumnTransformer + StandardScaler (obligatorio)
3. **Entrenamiento:** [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py) - cross_val_score con StratifiedKFold
4. **Serialización:** `modelos/predictor_mlp.joblib`
5. **Carga e inferencia:** [inferencia/predictor.py](../../inferencia/predictor.py)
6. **Exposición:** [api/main.py](../../api/main.py)

---

## Lectura complementaria

- Resúmenes clave integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) (escalado, desbalance, teoría de perceptrón/redes).
- **Sklearn MLPClassifier:** [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

---

## Volver a la guía unificada

Para entender el flujo completo: [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)
