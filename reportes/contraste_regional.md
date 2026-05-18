# Contraste regional CDC BRFSS 2015 vs ENSANUT 2022

**Fecha de generación:** 2026-05-18  
**Registros CDC analizados:** 50,000  
**Variables comparadas:** 10 indicadores de salud binarios  

---

## Contexto

El dataset CDC BRFSS 2015 proviene de una muestra de adultos estadounidenses.
Este análisis cuantifica el **sesgo distribucional** entre las prevalencias observadas
en BRFSS 2015 y las reportadas por ENSANUT 2022 para adultos mexicanos, con el fin de
documentar la transferibilidad del modelo a la población objetivo.

> **Criterio de lectura:** sesgo relativo = (CDC − ENSANUT) / ENSANUT × 100.  
> Valores positivos indican sobrerepresentación en CDC; negativos, subrepresentación.

---

## Tabla de sesgo distribucional

| Variable | CDC BRFSS 2015 (%) | ENSANUT 2022 (%) | Sesgo relativo (%) | Clasificación |
|----------|-------------------|-----------------|-------------------|---------------|
| Smoker | 44.5 | 15.8 | +181.9 | 🔴 Alto |
| HighChol | 42.6 | 22.9 | +86.2 | 🔴 Alto |
| PhysActivity | 75.6 | 45.2 | +67.2 | 🔴 Alto |
| Veggies | 80.9 | 54.8 | +47.6 | 🔴 Alto |
| Stroke | 4.0 | 2.7 | +46.9 | 🔴 Alto |
| HighBP | 42.8 | 31.8 | +34.6 | 🔴 Alto |
| AnyHealthcare | 95.0 | 76.2 | +24.7 | 🟡 Moderado |
| HeartDiseaseorAttack | 9.7 | 7.8 | +24.5 | 🟡 Moderado |
| Fruits | 63.3 | 55.6 | +13.8 | 🟡 Moderado |
| HvyAlcoholConsump | 5.7 | 5.3 | +7.8 | ✅ Bajo |

---

## Resumen ejecutivo

- **6** variables con sesgo alto (>30%): Smoker, HighChol, PhysActivity, Veggies, Stroke, HighBP
- **3** variables con sesgo moderado (10–30%): AnyHealthcare, HeartDiseaseorAttack, Fruits
- **1** variables con sesgo bajo (<10%): transferibilidad adecuada.

## Implicaciones para el modelo

Las variables con sesgo alto requieren interpretación cuidadosa al aplicar el modelo
sobre población mexicana: la decisión del modelo puede verse afectada por diferencias
sistemáticas entre la distribución de entrenamiento y la distribución objetivo.

---

*Generado automáticamente por `scripts/generar_contraste_regional.py`*
*Fuente ENSANUT: INSP, Encuesta Nacional de Salud y Nutrición 2022*