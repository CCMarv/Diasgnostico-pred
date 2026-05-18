# Benchmark consolidado de tiempos de ejecución

Tabla de mediciones (tiempos totales en segundos)

| registros | tiempo_s |
|---:|---:|
| 40 | 3.033 |
| 100 | 3.528 |
| 200 | 3.721 |
| 500 | 4.685 |
| 1000 | 5.135 |
| 5000 | 15.715 |
| 10000 | 43.195 |
| 50000 | 952.930 |

## Ajustes de escalado
- Ajuste lineal: t = 0.019123 * n + -30.781066 (R^2 aprox no calculado)
- Ajuste ley de potencia: t = 8.562409e-02 * n^0.722140

## Proyecciones para dataset completo (253000 registros)
- Predicción lineal: 4807.4 s (80.1 min)
- Predicción potencia: 683.0 s (11.4 min)

## Observaciones
- El entrenamiento domina claramente el tiempo (mayor contribución).
- La proyección lineal puede subestimar o sobreestimar según cómo escalen internamente los estimadores (SVM y MLP pueden escalar peor que O(n)).
- Recomendar ejecutar una validación sobre 50k-200k para calibrar la proyección.
