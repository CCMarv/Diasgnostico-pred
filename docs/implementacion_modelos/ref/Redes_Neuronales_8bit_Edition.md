# Redes Neuronales — 8-Bit Edition

**Inteligencia Artificial**  
Centro Universitario de Guadalajara — Licenciatura en Inteligencia Artificial  
*Una aventura práctica, visual y divertida en el mundo de la IA*

---

## Menú Principal

| Nivel | Tema | Descripción |
|-------|------|-------------|
| 01 | ¿Qué es una Red Neuronal? | El cerebro artificial: conceptos fundamentales y analogías |
| 02 | De la Biología al Silicio | Comparando neuronas biológicas y artificiales |
| 03 | La Neurona Artificial | El Perceptrón: el modelo matemático básico |
| 04 | Arquitectura de una Red | Capas, conexiones y estructura multicapa |
| 05 | Funciones de Activación | Sigmoide, ReLU, Tanh: el pulso de la red |
| 06 | Proceso de Entrenamiento | Forward, Backpropagation y optimización |
| 07 | Tipos de Redes Neuronales | CNN, RNN, LSTM, GAN: especializaciones |
| 08 | Aplicaciones Reales | Casos de uso en el mundo actual |
| BOSS | Demo Práctica: XOR + Recursos | Código Python en acción y próximos pasos |

---

## World 1-1 — Nivel 1: Fundamentos

### Level 01 — ¿Qué es una Red Neuronal?

Una red neuronal es un sistema complejo de nodos interconectados (neuronas) que procesan y transmiten información, análogo a un cerebro artificial.

**Componentes clave:**

- **Neuronas (Nodos):** Unidades de procesamiento.
- **Conexiones:** Pesos y sesgos.
- **Capas:** Entrada, ocultas y salida.
- **Activación:** Funciones no lineales.

**Arquitectura básica:**

```
[CAPA DE ENTRADA]
  Recibe los datos iniciales (píxeles, números, texto)
          ▼
[CAPAS OCULTAS]
  Procesan y extraen patrones complejos
          ▼
[CAPA DE SALIDA]
  Produce el resultado final (predicción, clasificación)
```

---

### Level 02 — De la Biología al Silicio

| Componente biológico | Analogía artificial |
|---|---|
| **Dendritas** — Reciben señales de otras neuronas | Entradas de datos (x₁, x₂, ...) |
| **Cuerpo celular** — Procesa la información recibida | Función de activación |
| **Axón** — Transmite la señal a otras neuronas | Salida procesada |
| **Sinapsis** — Conexión con neurotransmisores | Pesos sinápticos (w₁, w₂, ...) |

| | Neurona Biológica | Neurona Artificial |
|---|---|---|
| Escala | ~100 mil millones de neuronas | Millones de operaciones/segundo |
| Consumo | ~12 W (menos que una bombilla) | 150 W+ (laptop típica) |

---

### Level 03 — La Neurona Artificial: El Perceptrón

**Diagrama del Perceptrón:**

```
x₁ ──w₁──┐
x₂ ──w₂──┼──► Σ(wᵢ·xᵢ) + b ──► f(z) ──► y
x₃ ──w₃──┘
```

**La fórmula mágica:**

```
z = Σ(wᵢ · xᵢ) + b    (suma ponderada + sesgo)
y = f(z)               (función de activación)
```

**Ejemplo práctico — ¿Debería salir a jugar?**

| Entrada | Descripción | Peso |
|---------|-------------|------|
| x₁ = 1 | ¿Hace sol? (1 = sí) | w₁ = 0.7 (importante) |
| x₂ = 1 | ¿Tengo tiempo? (1 = sí) | w₂ = 0.5 (moderado) |

```
z = (0.7 × 1) + (0.5 × 1) + (−0.2) = 1.0

Si f(z) > 0.5 → ¡SALIR A JUGAR!
```

---

## World 1-2 — Nivel 2: Arquitectura

### Level 04 — Arquitectura de una Red Neuronal (MLP)

En un **Multilayer Perceptron**, cada neurona se conecta con **todas** las neuronas de la capa siguiente (conexiones densas).

| Capa | Función | Ejemplo |
|------|---------|---------|
| **Capa de Entrada** | Recibe los datos iniciales; no realiza procesamiento. | Imagen 28×28 px = 784 neuronas |
| **Capas Ocultas** | Procesan y extraen patrones; cada capa aprende características más abstractas. | 1 o más capas (Deep Learning = muchas capas) |
| **Capa de Salida** | Produce el resultado final. | Clasificación binaria: 1 neurona; multiclase: N neuronas; regresión: 1 neurona continua |

---

### Level 05 — Funciones de Activación: El Pulso

#### Sigmoid (σ)

```
f(x) = 1 / (1 + e⁻ˣ)
```

| ✅ | ⚠️ |
|---|---|
| Rango: 0 a 1 (probabilidades) | Gradiente desvaneciente |
| Uso: Clasificación binaria | — |

#### Tanh

```
f(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)
```

| ✅ | ⚠️ |
|---|---|
| Rango: −1 a 1 (centrada en 0) | También sufre gradiente desvaneciente |
| Uso: Capas ocultas, RNN | — |

#### ReLU ⭐

```
f(x) = max(0, x)
```

| ✅ | ⚠️ |
|---|---|
| Rango: 0 a +∞ (no acotada) | Neuronas "muertas" para x < 0 |
| No sufre gradiente desvaneciente | — |
| Estándar en Deep Learning y CNN | — |

#### Softmax

```
f(xᵢ) = eˣⁱ / Σeˣʲ    → [0.1, 0.7, 0.2]
```

| ✅ | Uso |
|---|-----|
| Rango: Probabilidades (0–1) | Clasificación multiclase (capa de salida) |
| Suma de probabilidades = 1 | — |

---

## World 1-3 — Nivel 3: Especialización

### Level 06 — El Proceso de Entrenamiento

**Ciclo de entrenamiento:**

```
1. FORWARD PROPAGATION
   Los datos fluyen de la entrada hacia la salida.
   Cada neurona calcula su salida y la pasa a la siguiente capa.
              ▼
2. CÁLCULO DEL ERROR
   Se compara la predicción con el valor real
   mediante una función de pérdida (ej. MSE).
              ▼
3. BACKPROPAGATION
   El error se propaga hacia atrás, calculando cuánto
   contribuyó cada peso al error total (gradientes).
              ▼
4. ACTUALIZACIÓN DE PESOS
   Los pesos se ajustan con gradiente descendiente
   para minimizar el error en la siguiente iteración.
```

**Conceptos clave:**

| Concepto | Descripción | Valores típicos |
|----------|-------------|-----------------|
| **Época (Epoch)** | Un ciclo completo donde la red ve todos los datos de entrenamiento una vez. | — |
| **Lote (Batch)** | Subconjunto de datos procesados antes de actualizar pesos. | 32, 64, 128 |
| **Tasa de aprendizaje (η)** | Qué tan grandes son los ajustes de peso. | 0.001, 0.01, 0.1 |
| **Función de pérdida** | Mide qué tan lejos está la predicción del valor real. | MSE (regresión), Cross-Entropy (clasificación) |

**Fórmula del Gradiente Descendiente:**

```
w_new = w_old − η · (∂L/∂w)

η = tasa de aprendizaje
∂L/∂w = gradiente

Los pesos se ajustan en la dirección OPUESTA al gradiente
para minimizar el error.
```

---

### Level 07 — Tipos de Redes Neuronales

#### CNN — Redes Convolucionales

**Especialidad:** Procesamiento de imágenes y datos con estructura de cuadrícula.

**Componentes:** Capas convolucionales (filtros), Pooling (reducción dimensional), capas totalmente conectadas.

**Aplicaciones:** Reconocimiento facial, clasificación de imágenes, detección de objetos, vehículos autónomos.

**Arquitecturas famosas:** LeNet, AlexNet, VGG, ResNet.

---

#### RNN — Redes Recurrentes

**Especialidad:** Procesamiento de datos secuenciales (texto, series temporales).

**Características:** Memoria de estados anteriores, bucles de retroalimentación, procesamiento paso a paso.

**Aplicaciones:** NLP, traducción automática, generación de texto, predicción de series temporales.

> **Problema:** Gradientes desvanecientes en secuencias largas.

---

#### LSTM — Long Short-Term Memory

**Mejora sobre RNN:** Resuelve el problema de gradientes desvanecientes con celdas de memoria.

| Puerta (Gate) | Función |
|---|---|
| **Forget Gate** | Qué información olvidar |
| **Input Gate** | Qué información recordar |
| **Output Gate** | Qué output generar |

**Aplicaciones:** Chatbots, análisis de sentimiento, predicción bursátil, reconocimiento de voz.

> **Variante:** GRU (más eficiente, menos parámetros).

---

#### GAN — Redes Generativas Adversariales

**Objetivo:** Generar datos nuevos y realistas (imágenes, texto, audio).

**Dos redes en competencia:**
- **Generador:** Crea datos falsos.
- **Discriminador:** Detecta datos falsos.

**Aplicaciones:** Generación de imágenes (DALL-E, Midjourney), deepfakes, aumento de datos, transferencia de estilo artístico.

---

### Level 08 — Aplicaciones en el Mundo Real

| Dominio | Tecnología | Ejemplos destacados | Dato clave |
|---|---|---|---|
| **Reconocimiento facial** | CNN | Face ID (Apple), seguridad en aeropuertos | Precisión 99%+ en condiciones óptimas |
| **Asistentes virtuales** | RNN / LSTM / Transformers | Siri, Alexa, ChatGPT, Claude | 85% de interacciones automatizadas para 2025 |
| **Vehículos autónomos** | CNN + Deep Learning | Tesla Autopilot, Waymo Driver, ADAS | Reducción de costos operativos: 40% |
| **Diagnóstico médico** | CNN + Computer Vision | IBM Watson, detección de cáncer de mama | IA mejora detección de cáncer en 94% de casos |
| **Sistemas de recomendación** | Deep Learning + Collaborative Filtering | Netflix, Spotify, Amazon | 80% del contenido visto en Netflix es recomendado |
| **Procesamiento de lenguaje** | Transformers / LLM | ChatGPT, Google Translate, análisis de sentimiento | GPT-4 tiene 1.76 billones de parámetros |

---

## World 1-4 — Nivel 4: Práctica

### Boss Level — Demo: Resolviendo el Problema XOR

El perceptrón simple **no puede resolver XOR**. Se necesita una red multicapa con capa oculta.

**Tabla de verdad XOR:**

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

**Resultados del entrenamiento:**

| Métrica | Valor |
|---------|-------|
| Épocas de entrenamiento | 10,000 |
| Error final | < 0.01 |
| Precisión | 100% |

**Código Python — Red Neuronal XOR desde cero:**

```python
import numpy as np

# Datos de entrenamiento XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Arquitectura: 2 → 4 → 1
input_size, hidden_size, output_size = 2, 4, 1

# Inicializar pesos aleatorios
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Función sigmoide
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Entrenamiento: 10,000 épocas
for epoch in range(10000):
    # Forward propagation
    hidden = sigmoid(X.dot(W1))
    output = sigmoid(hidden.dot(W2))

    # Calcular error
    error = y - output

    # Backpropagation
    d_output = error * output * (1 - output)
    d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

    # Actualizar pesos (learning rate = 0.5)
    W2 += hidden.T.dot(d_output) * 0.5
    W1 += X.T.dot(d_hidden) * 0.5

# Predicciones finales
print("XOR predicciones:", np.round(output))
```

---

### Bonus Level — Recursos y Próximos Pasos

#### Frameworks Populares

| Framework | Creador | Ideal para |
|-----------|---------|------------|
| **TensorFlow** | Google | Producción y deployment |
| **PyTorch** | Meta | Investigación y prototipado |
| **Keras** | (API sobre TF) | Alto nivel, fácil de usar |

#### Libros Recomendados

- *Deep Learning* — Goodfellow, Bengio, Courville
- *Neural Networks and Deep Learning* — Michael Nielsen
- *Hands-On Machine Learning* — Aurélien Géron

#### Cursos Online

- **Coursera — Deep Learning Specialization** (Andrew Ng / Stanford) — 5 cursos, muy completo.
- **Fast.ai — Practical Deep Learning** — Enfoque práctico, código primero.
- **YouTube — 3Blue1Brown** — Visualizaciones matemáticas increíbles.

#### Proyectos Prácticos Sugeridos

1. **Clasificador de imágenes** — Distinguir gatos vs. perros con CNN.
2. **Análisis de sentimiento** — Clasificar tweets positivos/negativos.
3. **Predicción de precios** — Forecasting de acciones o criptomonedas.
4. **Chatbot simple** — Bot de preguntas y respuestas con RNN.
5. **Generador de texto** — Crear poemas o historias con LSTM.

> **Consejo final:** ¡Practica, practica, practica! La mejor forma de aprender es construyendo proyectos propios.

---

## Game Completed!

| Estadística | Valor |
|-------------|-------|
| Niveles completados | 17 |
| Conocimiento adquirido | 100% |
| Posibilidades | ∞ |

*Ahora eres un HÉROE de las Redes Neuronales.*
