# Demo — Sistema de Diagnóstico Predictivo de Diabetes

Comandos copiables para ejecutar cada componente del sistema de extremo a extremo.

---

## Requisitos previos

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd Diasgnostico-pred

# Instalar todas las dependencias (incluye dev: pytest, streamlit, fastapi, etc.)
pip install -e .[dev]
```

---

## 1. Pipeline de entrenamiento

### Corrida rápida de validación (5k registros, ~2 min)

```bash
python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos svm,arbol,gbm,mlp \
  --n-muestras 5000
```

### Corrida académica de referencia (50k registros, ~30 min)

```bash
python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos svm,arbol,gbm,mlp \
  --n-muestras 50000 \
  --dir-resultados resultados/
```

Esto genera en `resultados/corrida_50k/`:
- `predictor_50k.joblib` — pipeline serializado completo
- `corrida_50k.json` — métricas crudas de los 4 modelos
- `corrida_50k.md` — reporte legible con tabla comparativa
- `curvas_gbm.png` — curvas ROC y Precisión-Recall
- `calibracion_gbm.png` — curva de calibración del mejor modelo

### Corrida con el dataset completo (253k registros)

```bash
python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos svm,arbol,gbm,mlp \
  --dir-resultados resultados/
```

### Serializar el modelo para la API

```bash
# Copiar el modelo de la corrida de referencia a la ruta que usa la API
cp resultados/corrida_50k/predictor_50k.joblib modelos/predictor_production.joblib
```

---

## 2. Dashboard interactivo (Streamlit)

```bash
streamlit run dashboard/app.py
```

Abre `http://localhost:8501` en tu navegador. Tres vistas disponibles (selector en barra lateral):

| Vista | Descripción |
|-------|-------------|
| **Comparativa de modelos** | Tabla con métricas de los 4 modelos, resaltando el ganador por ROC-AUC |
| **Predicción individual** | Formulario con las 21 variables CDC para calcular probabilidad de diabetes en un paciente |
| **Fenotipos K-Means** | Distribución de clústeres y tasa de diabetes por fenotipo de riesgo |

> La vista "Predicción individual" requiere que el modelo esté en `modelos/predictor_production.joblib`. Sin él, muestra un aviso de degradación elegante.

---

## 3. API REST (FastAPI)

### Levantar el servidor

```bash
uvicorn api.main:app --reload
```

### Verificar que el servidor está operativo

```bash
curl http://localhost:8000/salud
# Respuesta esperada: {"estado": "operativo", "modelo": "predictor_production.joblib"}
```

### Realizar una predicción

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "presion_alta": 1,
    "colesterol_alto": 1,
    "chequeo_colesterol": 1,
    "imc": 35.0,
    "fumador": 1,
    "derrame_cerebral": 0,
    "enfermedad_corazon": 1,
    "actividad_fisica": 0,
    "consume_fruta": 0,
    "consume_verdura": 0,
    "consumo_alcohol_alto": 0,
    "tiene_cobertura_medica": 1,
    "sin_medico_por_costo": 0,
    "salud_general": 4,
    "salud_mental": 10,
    "salud_fisica": 15,
    "dificultad_caminar": 1,
    "sexo": 1,
    "edad": 10,
    "educacion": 3,
    "ingreso": 3
  }'
# Respuesta esperada: {"categoria_riesgo": "alto", "confianza": 0.736, ...}
```

### Documentación interactiva (Swagger UI)

```
http://localhost:8000/docs
```

---

## 4. Tests

```bash
# Ejecutar todos los tests con reporte de cobertura
pytest pruebas/ -v --tb=short

# Solo tests de un módulo específico
pytest pruebas/test_pipeline.py -v
pytest pruebas/test_evaluador.py -v
pytest pruebas/test_fenotipado.py -v
```

Resultado esperado: **27 tests pasando, 0 errores**.

---

## 5. Notebooks

```bash
# Iniciar Jupyter Lab
jupyter lab

# O ejecutar los notebooks por línea de comandos
jupyter nbconvert --execute notebooks/01_eda_regionalizado.ipynb
jupyter nbconvert --execute notebooks/02_fenotipado_kmeans.ipynb
```

---

## 6. Corridas académicas con registro completo

```bash
# Ejecutar las dos corridas de referencia académica (10k y 50k) con log narrativo
python scripts/ejecutar_corridas.py

# El log narrativo queda en:
cat resultados/LOG_CORRIDAS.md
```

---

## Variables CDC BRFSS 2015 — referencia rápida

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `HighBP` | binaria | 0/1 | Presión arterial alta diagnosticada |
| `HighChol` | binaria | 0/1 | Colesterol alto diagnosticado |
| `CholCheck` | binaria | 0/1 | Revisión de colesterol en últimos 5 años |
| `BMI` | continua | 12–98 | Índice de masa corporal |
| `Smoker` | binaria | 0/1 | ≥100 cigarros en vida |
| `Stroke` | binaria | 0/1 | Antecedente de derrame cerebral |
| `HeartDiseaseorAttack` | binaria | 0/1 | Antecedente de enfermedad coronaria o infarto |
| `PhysActivity` | binaria | 0/1 | Actividad física en últimos 30 días |
| `Fruits` | binaria | 0/1 | Consume fruta ≥1 vez al día |
| `Veggies` | binaria | 0/1 | Consume verduras ≥1 vez al día |
| `HvyAlcoholConsump` | binaria | 0/1 | Consumo excesivo de alcohol |
| `AnyHealthcare` | binaria | 0/1 | Tiene cobertura de salud |
| `NoDocbcCost` | binaria | 0/1 | No pudo ver al médico por costo |
| `GenHlth` | ordinal | 1–5 | Salud general (1=excelente, 5=pobre) |
| `MentHlth` | continua | 0–30 | Días con mala salud mental en último mes |
| `PhysHlth` | continua | 0–30 | Días con mala salud física en último mes |
| `DiffWalk` | binaria | 0/1 | Dificultad para caminar o subir escaleras |
| `Sex` | binaria | 0/1 | 0=Femenino, 1=Masculino |
| `Age` | ordinal | 1–13 | Grupo de edad (1=18-24, 13=80+) |
| `Education` | ordinal | 1–6 | Nivel educativo (1=ninguno, 6=universitario) |
| `Income` | ordinal | 1–8 | Nivel de ingresos del hogar |
