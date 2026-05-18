# EDA regionalizado del dataset CDC BRFSS 2015
## Contextualización para IMSS / ENSANUT

---

**Propósito del notebook**  
Este cuaderno presenta un análisis exploratorio de datos orientado a evaluar la transferibilidad del dataset CDC BRFSS 2015 hacia un contexto de salud pública mexicano, tomando como referencia ENSANUT 2022 e indicadores de uso clínico compatibles con IMSS. El objetivo no es solo describir el dataset, sino identificar qué tan apropiado resulta como base para el desarrollo posterior del sistema predictivo de diabetes tipo 2.

**Enfoque de lectura**
- **Dimensión estadística:** revisar estructura, calidad, rangos, desbalance y relaciones entre variables.
- **Dimensión epidemiológica:** comparar la distribución observada con referencias mexicanas para reconocer sesgos de transferibilidad.
- **Dimensión de modelado:** derivar decisiones concretas para el preprocesamiento y la selección de estrategias de evaluación.

**Hipótesis de trabajo**  
> La muestra CDC BRFSS 2015 puede servir como base de entrenamiento para estimar riesgo de diabetes en población mexicana, siempre que se expliciten las diferencias distributivas respecto de ENSANUT y se ajuste la calibración del modelo para evitar interpretaciones clínicamente engañosas.

---

### Estructura analítica

| Bloque | Propósito | Pregunta guía |
|--------|-----------|---------------|
| 0 | Preparación del entorno | ¿El dataset y sus variables cumplen el contrato esperado? |
| 1 | Control de calidad | ¿Hay nulos, rangos inválidos o desbalance relevante? |
| 2 | Análisis univariado | ¿Qué distribuciones difieren respecto de ENSANUT? |
| 3 | Análisis bivariado | ¿Qué variables se asocian más con diabetes? |
| 4 | Contraste CDC ↔ México | ¿Dónde aparece el sesgo de transferibilidad? |
| 5 | Desbalance y remuestreo | ¿Qué estrategia es más defendible metodológicamente? |
| 6 | Síntesis final | ¿Qué decisiones quedan justificadas para el modelado? |

---

---
## Bloque 4. Contraste de prevalencias CDC ↔ México (ENSANUT)

### Objetivo metodológico
Cuantificar qué variables presentan mayores diferencias de prevalencia entre la muestra CDC y la referencia mexicana. Este contraste es fundamental para evaluar transferibilidad: un buen desempeño interno no garantiza un comportamiento equivalente fuera del contexto original.

### Criterio de interpretación
Las diferencias observadas deben leerse como señales de posible desajuste poblacional. El interés no es solo verificar magnitudes, sino identificar qué variables requieren una narrativa explícita de limitación y ajuste en el reporte final.

---

### Formato del Bloque 4.1

La celda anterior calcula el contraste CDC ↔ México; esta celda muestra la tabla ordenada por sesgo relativo.