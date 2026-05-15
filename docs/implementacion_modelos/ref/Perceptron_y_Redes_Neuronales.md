# Perceptrón & Redes Neuronales

**Inteligencia Artificial**  
Un viaje al corazón de la Inteligencia Artificial

---

## Agenda

| # | Tema | Descripción |
|---|------|-------------|
| 01 | La Neurona Artificial | La unidad fundamental que da vida a las redes neuronales, inspirada en el cerebro biológico. |
| 02 | Suma Ponderada y Pesos | El motor matemático: cómo las neuronas procesan información mediante pesos y bias. |
| 03 | Puertas Lógicas AND y OR | Demostrando el poder del aprendizaje lineal con ejemplos prácticos visuales. |
| 04 | El Problema del XOR | Cuando una sola neurona no es suficiente: el desafío que revolucionó el campo. |
| 05 | Funciones de Activación | Step, Sigmoid y ReLU: comparativa visual de las funciones que introducen no linealidad al modelo. |

---

## Capítulo 01 — La Neurona Artificial

### ¿Qué es una Neurona Artificial?

Una neurona artificial es la **unidad fundamental** de las redes neuronales artificiales. Es un modelo matemático inspirado en el funcionamiento de las neuronas biológicas del cerebro humano.

- **Procesamiento de información:** recibe, procesa y transmite información en sistemas de IA, emulando de manera simplificada el comportamiento de las neuronas biológicas.
- **Analogía biológica:** al igual que las neuronas biológicas se comunican mediante sinapsis, las neuronas artificiales se conectan mediante **pesos** que determinan la fuerza de la conexión.

---

### Anatomía de una Neurona Artificial (Perceptrón)

```
x₁ ──w₁──┐
x₂ ──w₂──┤──► Σ + b ──► f(x) ──► y
x₃ ──w₃──┘
```

**Ecuación:**

```
y = f(Σ(wᵢ · xᵢ) + b)
```

| Componente | Descripción |
|---|---|
| **Entradas (xᵢ)** | Múltiples valores numéricos provenientes de datos iniciales u otras neuronas. |
| **Pesos (wᵢ)** | Determinan la importancia de cada entrada. Son los parámetros de aprendizaje. |
| **Función de activación f(x)** | Introduce no linealidad, permitiendo aprender relaciones complejas. |
| **Salida (y)** | Resultado final; puede ser entrada de otra neurona o la predicción del modelo. |

---

## Capítulo 02 — Suma Ponderada y Pesos

### La Fórmula Central

```
z = w₁·x₁ + w₂·x₂ + w₃·x₃ + ... + wₙ·xₙ + b

z = Σ(wᵢ · xᵢ) + b
```

Cada entrada `xᵢ` se multiplica por su peso `wᵢ`, se suman todos los productos y se añade el bias `b`.

**Ejemplo práctico:**

| Parámetro | Valores |
|-----------|---------|
| Entradas | x₁ = 2, x₂ = 3, x₃ = 1 |
| Pesos | w₁ = 0.5, w₂ = −1, w₃ = 2 |
| Bias | b = 0.5 |

```
z = (0.5×2) + (−1×3) + (2×1) + 0.5
z = 1 − 3 + 2 + 0.5 = 0.5
```

---

### ¿Qué significan los pesos?

| Tipo de peso | Efecto |
|---|---|
| **Peso alto** | Fuerte influencia de la entrada en la salida. |
| **Peso bajo** | Influencia menor de la entrada. |
| **Peso negativo** | Influencia inhibidora: reduce la activación. |

> **Aprendizaje:** los pesos son los parámetros que la red ajusta iterativamente durante el entrenamiento para minimizar el error entre predicción y valor real.

---

### El Sesgo (Bias)

El bias `b` es un parámetro adicional que controla la facilidad con que una neurona se activa, independientemente de las entradas.

**Analogía con regresión lineal:**

```
y = mx + b
        ↑
   intersección con el eje Y
```

Sin bias, la recta de decisión siempre pasaría por el origen (0, 0).

| Escenario | Comportamiento |
|---|---|
| **Bias alto** | Requiere entrada más alta para activarse. |
| **Bias bajo** | La activación es más fácil. |
| **Con bias** | La recta de decisión puede desplazarse para ajustarse mejor a los datos. |
| **Sin bias** | La salida siempre pasa por el origen, limitando el poder del modelo. |

> **Ejemplo real:** la altura de una persona no es cero incluso si su ingesta calórica fuera cero. El bias permite capturar este tipo de relaciones.

---

## Capítulo 03 — Puertas Lógicas con Perceptrón

### Puerta Lógica AND

La salida es `1` **solo cuando ambas entradas son 1**. Es un problema linealmente separable.

| x₁ | x₂ | AND |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 0   |
| 1  | 0  | 0   |
| 1  | 1  | **1** |

> El perceptrón puede encontrar una línea recta que separa perfectamente los puntos con salida 0 de los puntos con salida 1.

---

### Puerta Lógica OR

La salida es `1` **cuando al menos una entrada es 1**. También linealmente separable.

| x₁ | x₂ | OR |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | **1** |
| 1  | 0  | **1** |
| 1  | 1  | **1** |

> Al igual que AND, OR es linealmente separable. El perceptrón aprende diferentes pesos para implementar esta función.

---

## Capítulo 04 — El Problema del XOR

### El Desafío del XOR

La salida es `1` **solo cuando exactamente una entrada es 1**.

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | **1** |
| 1  | 0  | **1** |
| 1  | 1  | 0   |

**El problema:** los puntos positivos `(0,1)` y `(1,0)` **no pueden separarse** de los negativos `(0,0)` y `(1,1)` con una sola línea recta.

---

### ¿Por Qué Falla el Perceptrón Simple?

El perceptrón simple solo puede generar **fronteras de decisión lineales:**

- Líneas rectas en 2D
- Planos en 3D
- Hiperplanos en n-D

**El "Invierno de la IA":** en 1969, Minsky y Papert demostraron esta limitación en su libro *Perceptrons*, contribuyendo a una fase de desconfianza hacia las redes neuronales.

---

### La Solución: Perceptrones Multicapa (MLP)

La solución llegó con los MLP en los años 80, mediante tres ideas clave:

| Paso | Concepto | Descripción |
|------|----------|-------------|
| 1 | **Múltiples capas** | Combinar neuronas en capas ocultas. |
| 2 | **No linealidad** | Funciones de activación no lineales. |
| 3 | **Backpropagation** | Algoritmo de entrenamiento eficiente. |

---

## Capítulo 05 — Funciones de Activación

### ¿Qué es una Función de Activación?

La función de activación introduce **no linealidad** en el modelo. Determina si una neurona debe activarse o permanecer inactiva.

| Sin función de activación | Con función de activación |
|---|---|
| Red neuronal ≡ Regresión lineal simple | La red puede aprender relaciones complejas |

**Proceso completo:**

```
1. Suma ponderada:   z = Σ(wᵢ · xᵢ) + b
2. Activación:       a = f(z)
3. Salida final:     a
```

**¿Por qué es importante?**

- Sin no linealidad, una red con múltiples capas sería equivalente a una sola capa.
- Con no linealidad, la red puede aprender relaciones arbitrariamente complejas.
- Permite aproximar cualquier función continua (**Teorema de Aproximación Universal**).

---

### Comparativa de Funciones de Activación

#### Step Function

```
f(x) = 0  si x < 0
f(x) = 1  si x ≥ 0
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Simple y fácil de entender | No diferenciable en x = 0 |
| Usada en el perceptrón original | Salida binaria limitada |

---

#### Sigmoid

```
σ(x) = 1 / (1 + e^(−x))
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Suave y diferenciable | Gradientes desvanecientes (*vanishing gradients*) |
| Salida interpretable como probabilidad (0, 1) | No centrada en cero |

---

#### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Computacionalmente eficiente | Problema de neuronas "muertas" (x < 0) |
| Mitiga gradientes desvanecientes | — |
| Estándar en deep learning moderno | — |

---

### Resumen Comparativo

| Función | Fórmula | Rango de salida | Uso típico |
|---------|---------|----------------|------------|
| **Step** | `0 si x<0, 1 si x≥0` | {0, 1} | Perceptrón clásico |
| **Sigmoid** | `1 / (1 + e^−x)` | (0, 1) | Clasificación binaria (capa de salida) |
| **ReLU** | `max(0, x)` | [0, ∞) | Capas ocultas en deep learning |

---

## Conclusión

Las redes neuronales han revolucionado la inteligencia artificial. Desde el humilde perceptrón hasta las complejas arquitecturas de deep learning actuales, cada avance nos acerca más a máquinas verdaderamente inteligentes.

| Etapa | Acción |
|-------|--------|
| **Fundamentos** | Entender los conceptos básicos del perceptrón y las redes neuronales. |
| **Práctica** | Implementar lo aprendido con ejemplos reales. |
| **Innovación** | Construir el futuro sobre estos cimientos. |

> *"Las redes neuronales han revolucionado la inteligencia artificial. Desde el humilde perceptrón hasta las complejas arquitecturas de deep learning de hoy, cada avance nos acerca más a máquinas verdaderamente inteligentes."*
