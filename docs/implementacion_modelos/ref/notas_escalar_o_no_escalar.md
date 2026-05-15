# Notas: ¿Cuándo Escalar los Datos?

**Unidad 1.2 — Transformación Avanzada**  
Tema: Escalado y normalización de datos | Dataset: `buro_credito.csv` (5,383 registros)

---

## ¿Qué problema resuelve este notebook?

Se entrena un modelo para predecir si un cliente pagará o no un crédito bancario (*default / no default*), usando como entradas:

- Puntaje crediticio
- Ingresos mensuales (USD)
- Monto del préstamo (USD)

El objetivo no es solo predecir, sino **comparar el desempeño de cada modelo con y sin escalamiento** para entender cuándo es necesario escalar.

---

## ¿Qué es el escalamiento?

> Consiste en hacer que todas las variables tengan aproximadamente el mismo rango de valores.

El problema sin escalar es que las variables tienen escalas muy distintas:

| Variable | Rango aprox. |
|---|---|
| Puntaje crediticio | 0 – 7.5 |
| Ingresos mensuales | 1.4 – 6,997.8 |
| Monto del préstamo | 20,023 – 49,991 |

### Tipos de escalamiento

| Técnica | Resultado |
|---|---|
| **Normalización** (`MinMaxScaler`) | Todas las variables entre 0 y 1 |
| **Estandarización** (`StandardScaler`) | Media = 0, desviación estándar = 1 |

En este notebook se usa **MinMaxScaler**.

---

## Funciones clave implementadas

### `crear_in_out(datos)`
Separa el DataFrame en entradas `X` (columnas 0–2) y salida `Y` (columna 3).

### `crear_train_val_test(X, Y)`
Divide en tres sets: **70% entrenamiento**, **15% validación**, **15% prueba** usando `train_test_split` dos veces.

### `escalar_datos(xtr, xvl, xts)`
Escala los datos con `MinMaxScaler`. Regla fundamental:

```python
scaler = MinMaxScaler()
xtr_s = scaler.fit_transform(xtr)  # fit SOLO en train
xvl_s = scaler.transform(xvl)      # transform en val y test
xts_s = scaler.transform(xts)      # nunca fit aquí
```

> **Importante:** el scaler se ajusta (`fit`) **únicamente** sobre el set de entrenamiento para evitar data leakage.

---

## Experimento: efecto del escalamiento por modelo

### 5.1 — Árboles de Clasificación (`DecisionTreeClassifier`)

**Resultado:** El desempeño es **idéntico** con y sin escalamiento.

**¿Por qué?** Los árboles definen umbrales internos en la escala original de los datos. Al escalar, el umbral se desplaza proporcionalmente → el resultado no cambia.

```
Exactitud SIN escalamiento: X%
Exactitud CON escalamiento: X%  ← igual
```

---

### 5.2 — Clasificador kNN (`KNeighborsClassifier`, k=3)

**Resultado:** El desempeño **cambia** con el escalamiento (generalmente mejora).

**¿Por qué?** kNN calcula distancias entre puntos. Si una variable tiene una escala 10,000× mayor que otra, domina el cálculo de distancia ignorando las demás características.

```
Exactitud SIN escalamiento: X%
Exactitud CON escalamiento: X%  ← diferente, generalmente mayor
```

---

### 5.3 — Red Neuronal (MLP)

**Resultado:** También **sensible** al escalamiento. Las redes neuronales convergen mejor y más rápido cuando los datos están normalizados.

---

## Conclusiones

| Modelo | ¿Requiere escalamiento? |
|---|---|
| Árbol de clasificación | No (el resultado es igual) |
| Bosque aleatorio (Random Forest) | No (aunque puede acelerar entrenamiento) |
| Gradient Boosting / XGBoost | No |
| **kNN** | **Sí** |
| **SVM** | **Sí** |
| **Redes Neuronales** | **Sí** |
| **Regresión logística / lineal** | **Sí** |

> **Recomendación práctica del notebook:**  
> Escalar los datos independientemente del modelo que se vaya a usar. Nunca hace daño y puede ayudar, especialmente en tiempo de entrenamiento.

---

## Conexión con el programa

Este notebook cubre directamente el tema **1.2 Transformación Avanzada** del programa, específicamente:

- Escalado y normalización (`MinMaxScaler`)
- La regla de hacer `fit` solo en train (prevención de **data leakage**, tema 1.1)
- Comparación práctica entre modelos sensibles e insensibles al escalado
