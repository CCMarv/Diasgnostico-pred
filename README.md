# diasgnostico-pred 

Sistema modular de predicción de riesgo de diabetes tipo 2 mediante API REST. Entrena y despliega modelos de machine learning supervisado sobre el dataset CDC BRFSS 2015, con contextualización epidemiológica para población mexicana (IMSS / ENSANUT 2022).

**Versión:** 0.1.0 | **Python:** ≥ 3.11 | **Licencia:** MIT

---

## Descripción del proyecto

`diasgnostico-pred` es una herramienta de modelado predictivo y despliegue de API orientada a estimar el riesgo individual de diabetes tipo 2 a partir de 21 indicadores de salud del estudio CDC BRFSS 2015.

El sistema recibe como entrada un perfil de salud del paciente (condiciones crónicas, hábitos, datos sociodemográficos) y produce como salida una probabilidad continua y una categoría de riesgo (`bajo`, `medio`, `alto`) con advertencia clínica cuando el resultado cae en zona de incertidumbre.

Está diseñado para investigadores de salud pública y equipos de desarrollo que necesiten un prototipo desplegable de tamizaje preventivo, sin exponer datos personales identificables. La arquitectura permite sustituir el dataset de entrenamiento o el modelo subyacente sin modificar la capa de inferencia ni la API.

## Lineamientos de colaboración

Este proyecto se mantiene de forma colaborativa y está dirigido a estudiantes universitarios con base en Python. Para mantener consistencia:

- **Documentación:** escribir descripciones cortas, en español claro, explicando objetivo, entrada y salida.
- **Código:** preferir docstrings breves en funciones públicas y nombres descriptivos en español.
- **Notas técnicas:** cuando una decisión no sea obvia, documentar el motivo (no solo el resultado).
- **Cambios de equipo:** al modificar un módulo, actualizar la sección correspondiente en `README.md` o en `PROYECTO.md` (documento de diseño en la raíz) si cambia su comportamiento.

---

## Características principales

- Entrena y compara cuatro modelos supervisados (SVM, árbol de decisión, Gradient Boosting, MLP) dentro de un `sklearn.Pipeline` que incluye preprocesamiento, eliminando el riesgo de *data leakage*.
- Expone los resultados a través de una API REST FastAPI con validación clínica de entradas (rangos, coherencia entre variables) y modo degradado automático cuando el modelo no está disponible.
- Mapea los 21 campos del dataset CDC BRFSS a nombres en español sin PII, con validación de rangos clínicamente plausibles vía Pydantic v2.
- Serializa el pipeline completo (preprocesador + estimador) en un único archivo `.joblib`, garantizando reproducibilidad entre entrenamiento e inferencia.
- Genera métricas clínicas completas: ROC-AUC, PR-AUC, sensibilidad, especificidad, F1, Brier Score, curvas de calibración y tabla comparativa en Markdown.
- Incluye análisis de sesgo distribucional CDC ↔ ENSANUT 2022 para documentar la transferibilidad del modelo a población mexicana.
- Emite advertencia automática cuando la probabilidad predicha cae dentro del margen de incertidumbre (±5% de cualquier umbral), indicando la necesidad de evaluación clínica adicional.

---

## Requisitos del sistema

- Python 3.11 o superior
- Sistema operativo: macOS, Linux o WSL2 en Windows
- Memoria RAM recomendada: 8 GB (el dataset completo tiene ~253 000 filas)
- Sin requerimientos de GPU; todos los modelos corren en CPU

Dependencias principales declaradas en `pyproject.toml`:

| Paquete | Versión mínima | Rol |
|---|---|---|
| `fastapi` | 0.112.0 | Framework API REST |
| `uvicorn` | 0.30.0 | Servidor ASGI |
| `pydantic` | 2.8.0 | Validación de entradas y esquemas |
| `pandas` | 2.2.0 | Manipulación del dataset |
| `scikit-learn` | 1.5.0 | Modelos, preprocesamiento, métricas |
| `joblib` | 1.4.0 | Serialización de pipelines |
| `matplotlib` | 3.9.0 | Gráficas de evaluación clínica |
| `pyarrow` | 17.0.0 | Almacenamiento Parquet |
| `ucimlrepo` | 0.0.3 | Descarga automática del dataset |
| `pytest` | 8.2.0 | Suite de pruebas (dev) |
| `httpx` | 0.27.0 | Cliente HTTP para pruebas de API (dev) |

Extras opcionales: `pip install -e .[dashboard]` para Streamlit, `pip install -e .[shap]` para explicabilidad con SHAP.

---

## Instalación rápida

```bash
git clone https://github.com/tu-usuario/diasgnostico-pred.git
cd diasgnostico-pred
conda env create -f environment.yml
conda activate diagnostico-pred
```

Si prefieres instalar manualmente dentro de un entorno ya creado:

```bash
pip install -e .[dev]
```

Verificar que la instalación es correcta:

```bash
pytest
# Se esperan todas las pruebas en verde; el modelo real no es necesario para los tests de contrato.
```

### Validación completa desde cero

Para probar el programa completo en un entorno limpio, sigue este orden:

```bash
conda env create -f environment.yml
conda activate diagnostico-pred
pytest pruebas/test_cargador.py pruebas/test_predictor.py pruebas/test_api.py
python -m entrenamiento.pipeline --modo clasificacion
uvicorn api.main:app --reload
```

La corrida completa debe dejar estos artefactos como evidencia:

- `modelos/modelo_diabetes_v1.joblib`
- `reportes/metricas_sprint1.json`
- `reportes/comparativa_modelos.md`
- `reportes/curvas_<modelo>.png`

Si el entrenamiento tarda demasiado, eso es esperable en el estado actual; lo importante es que siga mostrando progreso y termine con artefactos válidos.

---

## Uso básico

### 1. Descargar el dataset y entrenar el modelo

```bash
python -m entrenamiento.pipeline --modo clasificacion
```

Esto descarga el dataset CDC BRFSS 2015 desde UCI ML Repository (si no existe), ejecuta el pipeline completo y guarda:

- `modelos/modelo_diabetes_v1.joblib` — pipeline serializado listo para inferencia
- `reportes/metricas_sprint1.json` — métricas de todos los modelos evaluados
- `reportes/comparativa_modelos.md` — tabla comparativa en Markdown
- `reportes/curvas_<modelo>.png` — curvas ROC y Precision-Recall del mejor modelo

Argumentos disponibles:

| Argumento | Descripción | Ejemplo |
|---|---|---|
| `--modo` | `clasificacion` o `clustering` | `--modo clasificacion` |
| `--modelos` | Modelos a entrenar, separados por coma | `--modelos gbm,mlp` |
| `--dataset` | Ruta alternativa al CSV | `--dataset datos/brutos/mi_csv.csv` |
| `--salida-modelo` | Ruta de destino del `.joblib` | `--salida-modelo modelos/v2.joblib` |

### 2. Levantar la API

```bash
uvicorn api.main:app --reload
```

La API inicia en `http://localhost:8000`. Si el modelo no existe, el servicio arranca en **modo degradado**: responde en `/salud` con `estado: degradado` y retorna HTTP 503 en `/predecir`.

---

## Ejemplos de uso

### Verificar el estado del servicio

```bash
curl http://localhost:8000/salud
```

```json
{
  "estado": "operativo",
  "version": "0.1.0",
  "detalles": {
    "modelo_cargado": true,
    "ruta_modelo": "modelo_diabetes_v1.joblib",
    "timestamp_servidor": "2026-05-14T20:30:00+00:00"
  }
}
```

### Obtener una predicción de riesgo

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "presion_alta": 1,
    "colesterol_alto": 1,
    "chequeo_colesterol": 1,
    "imc": 34.0,
    "fumador": 0,
    "derrame_cerebral": 0,
    "enfermedad_corazon": 0,
    "actividad_fisica": 0,
    "consume_fruta": 0,
    "consume_verdura": 1,
    "consumo_alcohol_alto": 0,
    "tiene_cobertura_medica": 1,
    "sin_medico_por_costo": 0,
    "salud_general": 4,
    "salud_mental": 5,
    "salud_fisica": 10,
    "dificultad_caminar": 0,
    "sexo": 1,
    "edad": 8,
    "educacion": 4,
    "ingreso": 3
  }'
```

```json
{
  "categoria_riesgo": "alto",
  "confianza": 0.71,
  "version": "0.1.0",
  "tiempo_ms": 12,
  "advertencia": null
}
```

### Ejemplo con zona de incertidumbre

Cuando la probabilidad cae cerca de un umbral (±5%), la API emite una advertencia clínica:

```json
{
  "categoria_riesgo": "medio",
  "confianza": 0.64,
  "version": "0.1.0",
  "tiempo_ms": 9,
  "advertencia": "Resultado en zona de incertidumbre; requiere evaluación clínica."
}
```

### Usar el predictor directamente en Python

```python
import pandas as pd
from inferencia.predictor import PredictorDiabetes

predictor = PredictorDiabetes()
predictor.cargar_modelo()

entrada = pd.DataFrame([{
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.0,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
    "DiffWalk": 0, "Sex": 1, "Age": 8, "Education": 4, "Income": 3
}])

resultado = predictor.predecir(entrada)
print(resultado)
# {'probabilidad': 0.71, 'clase': 1, 'version': '0.1.0', 'tiempo_ms': 12}
```

---

## Arquitectura y organización del código

```text
diasgnostico-pred/
├── api/
│   ├── config.py           # Reexporta constantes de config.py para la capa API
│   ├── esquemas.py         # Modelos Pydantic: DatosPaciente, RespuestaPrediccion, RespuestaSalud
│   └── main.py             # Endpoints FastAPI: /salud y /predecir
├── entrenamiento/
│   ├── cargador_datos.py   # Carga, limpieza, análisis de desbalance y persistencia Parquet
│   ├── comparador_modelos.py # Catálogo de modelos (SVM, árbol, GBM, MLP, K-Means) y comparación
│   ├── evaluador.py        # Métricas clínicas, curvas ROC/PR/calibración, tabla comparativa
│   ├── pipeline.py         # Orquestador CLI: split → entrenamiento → evaluación → serialización
│   └── preprocesador.py    # ColumnTransformer por tipo de variable (continua, binaria, ordinal)
├── inferencia/
│   └── predictor.py        # PredictorDiabetes: carga .joblib, valida columnas, mide latencia
├── modelos/
│   └── .gitkeep            # Directorio para artefactos .joblib (excluidos de git)
├── datos/
│   ├── brutos/             # CSV fuente CDC BRFSS 2015 (excluido de git)
│   └── procesados/         # Dataset limpio en formato Parquet
├── reportes/               # Métricas JSON, tablas Markdown, gráficas PNG
├── notebooks/
│   └── 01_eda_regionalizado.ipynb  # EDA con contraste CDC ↔ ENSANUT 2022
├── pruebas/                # Suite de pruebas de contrato (API, cargador, predictor, preprocesador)
├── config.py               # Fuente única de constantes: rutas, umbrales, columnas CDC, semilla
├── pyproject.toml          # Dependencias y configuración de pytest
└── .env.example            # Variables de entorno documentadas
```

**Capas del sistema:**

- `config.py` — Fuente única de verdad para todas las constantes. Ningún módulo define *magic strings*.
- `entrenamiento/` — Pipeline de datos y modelado. Todo lo que ocurre antes de serializar el modelo.
- `inferencia/` — Carga el artefacto serializado y ejecuta predicciones. No tiene dependencia de `entrenamiento/`.
- `api/` — Expone la inferencia como servicio HTTP. No conoce los detalles del entrenamiento.

---

## Tests y garantías de calidad

Las pruebas verifican contratos de interfaz, no implementaciones internas:

```bash
pytest                   # Ejecuta toda la suite
pytest -q --tb=short     # Salida compacta con trazas de error
```

| Archivo de prueba | Qué verifica |
|---|---|
| `pruebas/test_api.py` | Códigos HTTP, mapeo de campos, modo degradado |
| `pruebas/test_predictor.py` | Compatibilidad con modelos con y sin `predict_proba` |
| `pruebas/test_cargador.py` | Carga, limpieza, detección de desbalance, persistencia Parquet |
| `pruebas/test_preprocesador.py` | Ausencia de *data leakage*, columnas binarias sin escalar, orden de ordinales |

Los tests no requieren el dataset real ni un modelo entrenado; usan modelos simulados y datasets sintéticos.

---

## Arquitectura del contrato de datos

Los 21 campos públicos de la API usan nombres en español sin PII. Pydantic aplica rangos clínicamente plausibles y detecta incoherencias entre variables:

| Campo público | Columna CDC | Rango permitido |
|---|---|---|
| `presion_alta` | `HighBP` | 0 – 1 |
| `colesterol_alto` | `HighChol` | 0 – 1 |
| `imc` | `BMI` | 10.0 – 80.0 |
| `salud_general` | `GenHlth` | 1 – 5 |
| `salud_mental` | `MentHlth` | 0 – 30 |
| `salud_fisica` | `PhysHlth` | 0 – 30 |
| `edad` | `Age` | 1 – 13 (grupos etarios CDC) |
| `educacion` | `Education` | 1 – 6 |
| `ingreso` | `Income` | 1 – 8 |
| *(12 variables binarias restantes)* | — | 0 – 1 |

Regla de coherencia clínica: `salud_fisica ≥ 20` con `dificultad_caminar = 0` es rechazado con HTTP 422.

---

## Tecnologías y dependencias principales

- Python 3.11, scikit-learn 1.5, FastAPI 0.112, Pydantic v2
- Dataset: CDC BRFSS 2015 (UCI ML Repository, id=891, ~253 000 filas, 21 variables + objetivo binario)
- Serialización: `joblib` (pipeline completo preprocesador + estimador)
- Almacenamiento procesado: Apache Parquet vía `pyarrow`
- Servidor: Uvicorn (ASGI)

---

## Asunciones clave del diseño

- El dataset CDC BRFSS 2015 es válido como proxy de entrenamiento para población mexicana, siempre que se declare el sesgo distribucional en variables de comportamiento y contexto socioeconómico.
- La sensibilidad (recall de clase positiva) tiene mayor prioridad clínica que la precisión: un falso negativo tiene consecuencias más graves que un falso positivo.
- El pipeline completo (preprocesador + estimador) se serializa como una sola unidad para garantizar que la transformación en inferencia sea idéntica a la del entrenamiento.
- La API puede operar en modo degradado sin modelo disponible; los errores de modelo nunca derriban el servidor.
- Ningún endpoint expone datos personales identificables (PII).

---

## Limitaciones conocidas

- El modelo está entrenado en datos de EE. UU. (CDC BRFSS 2015); su calibración puede ser subóptima para población mexicana sin recalibración local.
- El umbral de decisión predeterminado (`bajo < 0.33`, `alto ≥ 0.66`) está ajustado para la distribución CDC; para despliegue en contexto IMSS se recomienda reducir el umbral inferior a 0.25.
- El dataset completo pesa ~50 MB; la descarga inicial requiere conexión a internet y puede tomar varios minutos.
- El modo de clustering (`--modo clustering`) usa K-Means base y no está integrado al pipeline de inferencia en la versión actual.
- No incluye autenticación por API Key ni *rate limiting* en la versión actual (pendiente Sprint 4).

---

## Roadmap y versiones futuras

El proyecto sigue una metodología en espiral de 5 sprints. Estado actual:

| Sprint | Objetivo | Estado |
|---|---|---|
| Sprint 1 | Arquitectura base y contratos de interfaz | ✅ Completo |
| Sprint 2 | Pipeline de datos real y modelos supervisados | ⚠️ 8/10 tickets completos |
| Sprint 3 | Fenotipado metabólico (K-Means) y dashboard Streamlit | ❌ Pendiente |
| Sprint 4 | Reporte académico, Docker, CI/CD | ❌ Pendiente |
| Sprint 5 | Observabilidad, endurecimiento y documentación final | ❌ Pendiente |

Para el detalle de tickets y dependencias, consultar `docs/ROADMAP.md`.

Para la explicación por modelo, consultar [docs/implementacion_modelos/README.md](docs/implementacion_modelos/README.md).

---

## Licencia

MIT License. Permite usar, copiar, modificar y distribuir el código para cualquier propósito, incluyendo uso comercial, siempre que se mantenga el aviso de copyright original.

Este proyecto usa datos públicos del CDC BRFSS 2015 distribuidos por UCI Machine Learning Repository bajo términos de uso abierto para investigación.
