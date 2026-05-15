# Prompt: Réplica básica del proyecto de diagnóstico de diabetes

Objetivo: construir una réplica **lo más simple posible** del proyecto actual, manteniendo el mismo objetivo general de predicción de diabetes, pero usando el stack mínimo necesario de ciencia de datos y machine learning. La referencia principal para el tono, el alcance y la intención pedagógica debe ser la carpeta `docs/implementacion_modelos/ref`.

Todo lo que se cree, modifique o documente para esta versión debe vivir dentro de la carpeta `basic/`. No mezcles esta réplica con el proyecto principal ni escribas archivos fuera de ese directorio.

## Principio rector

Haz una versión funcional, clara y pequeña. No agregues complejidad innecesaria. Si una decisión puede resolverse con una técnica estándar de `pandas` + `scikit-learn`, usa esa opción antes que alternativas avanzadas. El objetivo es que el proyecto sea fácil de entender, fácil de ejecutar y suficiente para cumplir el requisito académico en su forma más simple.

## Fuente de trabajo

- Notebook base de análisis exploratorio: `notebooks/01_eda_regionalizado.ipynb`
- Dataset base: `datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv`
- Referencia conceptual principal: carpeta `docs/implementacion_modelos/ref`

## Migración de archivos existentes

Además de crear documentos o archivos nuevos, migra también dentro de `basic/` los archivos del proyecto actual que no necesiten cambios funcionales.

### Regla de migración

- Si un archivo del proyecto original ya cumple el objetivo de la réplica simple, cópialo tal cual a `basic/`.
- Si un archivo solo requiere ajustes mínimos de ruta, nombres o referencia al nuevo contexto, conserva su lógica y haz solo el cambio indispensable.
- No reescribas desde cero aquello que ya funciona y que puede reutilizarse sin alterar el objetivo.
- Mantén la misma intención del archivo original, pero siempre ubicado dentro de `basic/`.

### Qué archivos conviene migrar sin cambios grandes

- Documentación base.
- Notebooks de apoyo que expliquen el contexto o el EDA.
- Scripts de entrenamiento o evaluación que ya sean simples.
- Archivos de configuración que sigan siendo válidos para la versión mínima.

## Documentación nueva basada en la existente

Si hace falta crear documentación nueva, hazla tomando como base la documentación ya existente del proyecto y la carpeta `docs/implementacion_modelos/ref`.

### Qué debe replicarse de la documentación existente

- El contexto del proyecto y su propósito académico.
- La explicación simple de por qué se eligieron ciertas técnicas.
- La relación entre el dataset, el EDA y el entrenamiento.
- Las limitaciones y supuestos del enfoque.

### Cómo escribir la nueva documentación

- Mantén el estilo claro, directo y pedagógico.
- Prioriza explicaciones de contexto antes que teoría extensa.
- Usa la documentación existente como guía de estructura, no como obligación de copiar contenido literal.
- Si se crea un archivo nuevo, ubícalo dentro de `basic/docs/` o la subcarpeta que corresponda dentro de `basic/`.
- Incluye siempre una sección breve de contexto del proyecto, otra de objetivo y otra de alcance.

## Qué debe producir esta versión básica

La réplica debe conservar el objetivo del proyecto original, pero reducirse al flujo más directo posible:

1. Cargar y explorar el dataset.
2. Limpiar y preparar datos con preprocesamiento básico.
3. Entrenar modelos clásicos de ML.
4. Evaluar con métricas estándar.
5. Guardar resultados y artefactos mínimos.
6. Si se llega al nivel avanzado, exponer una API simple.

## Requisito mínimo: Nivel Básico

Cumple esto primero, sin adornos:

- Pipeline completo con 3 modelos.
- Preprocessing básico.
- Métricas de evaluación estándar.

### Interpretación recomendada

Usa una estructura simple como esta:

- Un pipeline de `scikit-learn` por modelo.
- Un preprocesamiento común para todas las variables.
- Tres modelos clásicos comparables.
- Evaluación con accuracy, precision, recall, F1 y ROC-AUC si aplica.

### Modelos sugeridos para el mínimo

Elige 3 modelos sencillos y conocidos, por ejemplo:

- Regresión logística.
- Árbol de decisión.
- SVM lineal o RBF, según convenga.

Si necesitas simplificar aún más, prioriza modelos del temario y evita variantes avanzadas.

## Nivel Intermedio

Si el alcance del proyecto lo permite, agrega lo siguiente sin complicar el diseño:

- Todos los modelos del temario: SVM, árboles de decisión, redes neuronales y K-Means.
- Optimización de hiperparámetros.
- Dashboard interactivo básico.

### Reglas para este nivel

- La optimización debe ser simple, usando `GridSearchCV` o búsqueda manual pequeña.
- El dashboard debe mostrar solo lo esencial: métricas, comparativa de modelos y una vista mínima de resultados.
- K-Means debe presentarse como análisis complementario, no como núcleo del sistema predictivo.

## Nivel Avanzado

Solo si el nivel básico e intermedio ya están bien resueltos:

- Sistema en producción mediante API.
- Comparativa con papers académicos.

### Reglas para este nivel

- La API debe ser mínima y estable.
- La comparación con papers debe ser breve, directa y honesta: qué coincide, qué no y por qué.
- No inventes soporte metodológico que no esté respaldado por la implementación.

## Stack mínimo recomendado

Usa el stack más básico posible:

- `pandas` para carga y limpieza.
- `numpy` para operaciones numéricas simples.
- `matplotlib` y `seaborn` para visualización.
- `scikit-learn` para preprocesamiento, modelos, validación y métricas.
- `joblib` para persistencia de modelos.
- `FastAPI` solo si se implementa el nivel avanzado.

## Decisiones de diseño

- Mantén el flujo muy lineal: datos -> EDA -> preprocesamiento -> entrenamiento -> evaluación -> guardado.
- Evita arquitecturas excesivamente modulares si no agregan valor real.
- Usa nombres claros y cortos para funciones y archivos.
- No agregues dependencias nuevas sin necesidad.
- Prioriza reproducibilidad sobre sofisticación.
- Toda nueva carpeta, notebook, script, informe o documento debe quedar dentro de `basic/`.

## Qué evitar

- Deep learning innecesario.
- Tuberías con demasiadas capas o abstracciones.
- Técnicas exóticas de balanceo si `class_weight='balanced'` basta.
- Dashboards complejos o dependientes de infraestructura extra.
- Comparativas teóricas largas que no se conecten con el código.

## Guía de implementación mínima sugerida

### Fase 1 — EDA básico

- Revisar estructura del dataset.
- Confirmar columnas y tipos.
- Verificar nulos, duplicados y desbalance de clases.
- Identificar variables numéricas y categóricas.
- Documentar hallazgos principales con figuras simples.

### Fase 2 — Preprocesamiento

- Imputación simple si hace falta.
- Escalado para variables numéricas cuando el modelo lo requiera.
- Codificación básica de variables categóricas.
- Separación clara entre entrenamiento y prueba.

### Fase 3 — Modelado

- Entrenar 3 modelos base.
- Usar validación cruzada si no añade demasiada complejidad.
- Comparar resultados en una tabla única.
- Elegir el mejor modelo por una métrica principal clara.

### Fase 4 — Evaluación

- Mostrar matriz de confusión.
- Reportar precision, recall, F1 y ROC-AUC.
- Resumir ventajas y limitaciones del mejor modelo.

### Fase 5 — Entrega

- Guardar el modelo final.
- Guardar una explicación breve del pipeline.
- Si aplica, exponer una API mínima con un endpoint de predicción.

## Estructura de salida esperada

La réplica básica debe dejar, como mínimo:

- Un notebook de EDA claro.
- Un flujo de entrenamiento reproducible.
- Resultados comparables entre 3 modelos.
- Un artefacto serializado del modelo ganador.
- Documentación breve de decisiones y limitaciones.

## Criterio de éxito

Se considera correcta si cumple esto de forma simple y completa:

- El dataset se carga desde `datos/brutos`.
- El EDA inicial se entiende y sirve como base del modelo.
- Hay 3 modelos entrenados y comparados.
- Hay métricas estándar de evaluación.
- El proyecto conserva el objetivo de diagnóstico de diabetes.
- La solución no depende de componentes innecesarios.

## Nota final

La prioridad absoluta es la simplicidad. La réplica debe parecer una versión pequeña y limpia del proyecto, no una reescritura sofisticada. Si hay dudas entre dos caminos, elige siempre el que requiera menos piezas, menos dependencias y menos riesgo de error.
