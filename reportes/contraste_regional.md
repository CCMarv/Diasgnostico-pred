# Contraste distribucional CDC ↔ ENSANUT (Sprint 2)

## Resumen metodológico
- El dataset CDC BRFSS 2015 se utiliza como proxy transferible para entrenamiento base.
- La equivalencia conceptual es directa en condiciones crónicas binarias (`HighBP`, `HighChol`, `HeartDiseaseorAttack`, `Stroke`).
- Variables conductuales y socioeconómicas se interpretan con cautela por sesgo de distribución.

## Variables con sesgo distribucional alto esperado

| Variable | CDC BRFSS 2015 | México (ENSANUT/INEGI/ENCODAT) | Implicación |
|---|---:|---:|---|
| BMI | menor media relativa | media IMC mayor (~29.2) | Ajustar interpretación y calibración de riesgo |
| Smoker | prevalencia más alta | prevalencia menor en México | Riesgo de sobreestimación del efecto de tabaquismo |
| Education | mayor escolaridad promedio | menor escolaridad promedio | Tratar como orden relativo |
| Income | escala USD | no comparable directamente | Reescalar a ranking/quintiles |

## Decisiones de modelado asociadas
1. Entrenar con variables ordinales como ordenadas explícitamente.
2. Priorizar métricas de sensibilidad/especificidad para tamizaje.
3. Mantener umbrales configurables para despliegue local.
4. Documentar limitaciones de calibración externa en el informe académico.
