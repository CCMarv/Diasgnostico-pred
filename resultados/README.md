# Guía de artefactos de corrida — Diagnóstico predictivo de diabetes

> **Audiencia:** científico de datos que recibe o reproduce una corrida del pipeline.  
> **Propósito:** entender qué contiene cada archivo, cómo se generó y cómo interpretarlo.  
> **Nota:** los ejemplos numéricos provienen de `corrida_10k/` (10 000 muestras, semilla=42).

---

## Estructura de una corrida

Cada carpeta `corrida_{tag}/` corresponde a una ejecución independiente del pipeline sobre
una muestra de `n` filas del dataset CDC BRFSS 2015. El `tag` indica el tamaño: `10k`,
`50k`, `253k`, etc.

```
corrida_{tag}/
├── corrida_{tag}.json                  # métricas brutas — fuente de verdad
├── corrida_{tag}.md                    # reporte Markdown legible
├── corrida_{tag}_manifest.json         # qué se ejecutó y cuándo
├── predictor_{tag}.joblib              # pipeline serializado del modelo ganador
├── predictor_{timestamp}.joblib        # copia versionada con timestamp
├── curvas_{modelo_ganador}.png         # curvas ROC y Precision-Recall
├── calibracion_{modelo_ganador}.png    # curva de calibración
├── muestra_{tag}.csv                   # muestra exacta usada (reproducible)
└── dataset_procesado.parquet           # dataset procesado 21 cols sin objetivo
```

Y en la raíz de `resultados/`:

```
resultados/
├── LOG_CORRIDAS.md    # narrativa completa de parámetros, resultados y comparativa
└── README.md          # este archivo
```

---

## Artefacto 1 — `corrida_{tag}.json`

**La fuente de verdad de la corrida.** Todos los demás artefactos textuales se derivan de él.

### Cómo se genera

El pipeline calcula las métricas sobre el conjunto de test (20 % estratificado, nunca visto
durante el entrenamiento) y serializa el resultado llamando a
`generador_reportes.guardar_json_crudo(metricas, ruta)`.

### Estructura

```json
{
  "version":                 "predictor_20260518_084354.joblib",
  "timestamp":               "20260518_084354",
  "n_muestras":              10000,
  "use_knn":                 true,
  "use_smote":               true,
  "semilla":                 42,
  "mejor_modelo":            "svm",
  "mejor_modelo_por_roc_auc":"svm",
  "mejor_modelo_por_pr_auc": "svm",
  "modelos": {
    "svm": {
      "nombre_modelo":       "svm",
      "roc_auc":             0.8351,
      "pr_auc":              0.4213,
      "sensibilidad":        0.7721,
      "especificidad":       0.7384,
      "f1_clase_positiva":   0.4497,
      "brier_score":         0.1672,
      "accuracy":            0.7430,
      "matriz_confusion":    [[1276, 452], [62, 210]]
    },
    ...
  },
  "desbalance": {
    "ratio":         6.35,
    "pct_clase_0":   0.864,
    "pct_clase_1":   0.136,
    "recomendacion": "class_weight"
  },
  "ruta_modelo_versionado": "...predictor_20260518_084354.joblib"
}
```

### Cómo cargarlo

```python
import json
from pathlib import Path

datos = json.loads(Path("corrida_10k/corrida_10k.json").read_text())

mejor   = datos["mejor_modelo"]          # "svm"
roc_auc = datos["modelos"][mejor]["roc_auc"]  # 0.8351
```

### Qué significa cada campo

| Campo | Qué mide | Rango | Interpretación |
|---|---|---|---|
| `roc_auc` | Capacidad de ordenar riesgo | 0–1 | ≥0.80 excelente; 0.75–0.80 aceptable; <0.75 insuficiente para tamizaje |
| `pr_auc` | Precisión-recall bajo desbalance | 0–1 | Más informativo que ROC-AUC cuando la clase positiva es <20%; referencia aleatoria ≈ prevalencia (0.136) |
| `sensibilidad` | Fracción de diabéticos identificados | 0–1 | En tamizaje clínico, priorizar sensibilidad ≥0.70 para no perder casos |
| `especificidad` | Fracción de sanos correctamente descartados | 0–1 | Especificidad baja → muchos falsos positivos → derivaciones innecesarias |
| `f1_clase_positiva` | Media armónica precisión-recall | 0–1 | Útil cuando se quiere un balance único; penaliza tanto FP como FN |
| `brier_score` | Error cuadrático medio de probabilidades | 0–1 | 0 = calibración perfecta; <0.10 excelente; >0.20 mal calibrado |
| `accuracy` | Porcentaje global correcto | 0–1 | **Engañoso bajo desbalance:** un modelo que predice siempre clase 0 obtiene 86 % de accuracy |
| `ratio` (desbalance) | Clase 0 / clase 1 | ≥1 | 6.35 → por cada diabético hay ~6 sanos; justifica SMOTE y `class_weight='balanced'` |

### La matriz de confusión

El pipeline la serializa como `[[TN, FP], [FN, TP]]` (convención sklearn: filas = clase real,
columnas = clase predicha):

```
                  Predicho: sano   Predicho: diabético
Actual: sano          TN=1276           FP=452
Actual: diabético     FN=62             TP=210
```

- **FN (falsos negativos):** diabéticos clasificados como sanos — el error clínicamente
  más grave; equivale a la tasa de fallo = FN / (FN + TP) = 1 − sensibilidad.
- **FP (falsos positivos):** sanos derivados innecesariamente — coste operativo, no clínico.
- Para calcular cualquier métrica derivada:

```python
tn, fp, fn, tp = datos["modelos"]["svm"]["matriz_confusion"][0][0], \
                 datos["modelos"]["svm"]["matriz_confusion"][0][1], \
                 datos["modelos"]["svm"]["matriz_confusion"][1][0], \
                 datos["modelos"]["svm"]["matriz_confusion"][1][1]
# Verificación:
assert abs(tp / (tp + fn) - datos["modelos"]["svm"]["sensibilidad"]) < 1e-6
```

---

## Artefacto 2 — `corrida_{tag}.md`

Reporte Markdown generado desde el JSON por `generador_reportes.construir_reporte_clasificacion()`.
Contiene: encabezado con parámetros de la corrida, tabla comparativa ordenada por ROC-AUC
(ganador en negrita con marcador `→`), interpretación clínica del ganador y pie con el
comando para reproducir.

**No editar manualmente.** Si se necesita un cambio de formato, modificar
`entrenamiento/generador_reportes.py` y regenerar desde el JSON con:

```bash
python scripts/generar_reporte_legible.py \
  --entrada  resultados/corrida_10k/corrida_10k.json \
  --salida   resultados/corrida_10k/corrida_10k.md
```

---

## Artefacto 3 — `corrida_{tag}_manifest.json`

Registro de qué se iba a ejecutar, escrito **antes** de que el pipeline arranque.
Permite diagnosticar corridas que terminaron con error: si existe el manifest pero no el
JSON de métricas, la corrida falló tras el inicio.

```json
{
  "modo":              "clasificacion",
  "timestamp_inicio":  "20260518_084203",
  "ruta_dataset":      ".../muestra_10k.csv",
  "ruta_modelo":       ".../predictor_10k.joblib",
  "ruta_reporte_crudo":".../corrida_10k.json",
  "modelos_a_entrenar": ["svm", "arbol", "gbm", "mlp"]
}
```

La diferencia entre `timestamp_inicio` (manifest) y `timestamp` (JSON de métricas) es el
tiempo total de la corrida.

---

## Artefacto 4 — `predictor_{tag}.joblib`

El **pipeline serializado completo** del modelo ganador por ROC-AUC en el conjunto de test.
Es el artefacto que se usa en producción e inferencia.

### Contenido interno

```
ImbPipeline (imblearn) o Pipeline (sklearn)
  └── preprocesador: ColumnTransformer
  │     ├── continuas:  KNNImputer → StandardScaler   (BMI, MentHlth, PhysHlth)
  │     ├── binarias:   SimpleImputer → passthrough    (14 columnas 0/1)
  │     └── ordinales:  SimpleImputer → OrdinalEncoder (GenHlth, Age, Education, Income)
  ├── resample: SMOTE(k_neighbors=5)   ← solo durante fit, ausente en predict
  └── clasificador: <estimador ganador con sus hiperparámetros optimizados>
```

El preprocesador está **ajustado solo sobre X_train** — garantía de no data leakage.
En inferencia, `Pipeline.predict_proba(X)` aplica la misma transformación sin SMOTE
(SMOTE solo opera en `fit`).

### Cómo cargarlo e inferir

```python
import joblib
import pandas as pd

pipeline = joblib.load("corrida_10k/predictor_10k.joblib")

# La entrada debe tener exactamente las 21 columnas CDC en cualquier orden
entrada = pd.DataFrame([{
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.0,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
    "DiffWalk": 0, "Sex": 1, "Age": 8, "Education": 4, "Income": 3,
}])

prob = pipeline.predict_proba(entrada)[0, 1]
print(f"Probabilidad de diabetes: {prob:.1%}")
# → Probabilidad de diabetes: 71.0%
```

### Inspeccionar el estimador ajustado

```python
# Acceder al clasificador final y sus parámetros
clf = pipeline.named_steps["clasificador"]  # o pipeline[-1]
print(type(clf).__name__)       # SVC / DecisionTreeClassifier / etc.
print(clf.get_params())         # hiperparámetros seleccionados

# Para la SVM: ver cuántos vectores de soporte se encontraron
if hasattr(clf, "n_support_"):
    print("Vectores de soporte por clase:", clf.n_support_)

# Para GBM: ver la importancia de variables (post-transformación)
if hasattr(clf, "feature_importances_"):
    import numpy as np
    fi = clf.feature_importances_
    print("Feature importances:", fi.round(4))
```

---

## Artefacto 5 — `predictor_{timestamp}.joblib`

Copia versionada del modelo final, con el timestamp de cuando terminó el pipeline.
Permite conservar el modelo de cada corrida sin sobreescribir el canónico `predictor_{tag}.joblib`.

Útil para comparaciones históricas:

```python
import joblib, pathlib

modelos_versionados = sorted(
    pathlib.Path("corrida_10k").glob("predictor_*.joblib")
)
for ruta in modelos_versionados:
    m = joblib.load(ruta)
    print(ruta.name, "→", type(m[-1]).__name__)
```

---

## Artefacto 6 — `curvas_{modelo_ganador}.png`

Gráfica de dos paneles generada por `EvaluadorClinico.graficar_curvas()`:

| Panel izquierdo | Panel derecho |
|---|---|
| Curva ROC (FPR vs TPR) con área sombreada = ROC-AUC | Curva Precision-Recall con área sombreada = PR-AUC |
| Línea punteada = clasificador aleatorio (diagonal) | Línea punteada = baseline aleatorio (= prevalencia) |

**Cómo leerla:**

- **ROC:** cuanto más arriba-izquierda la curva, mejor el modelo discrimina.
  El área bajo la curva (AUC) resume esto en un escalar.
- **PR:** especialmente relevante con clases desbalanceadas. Una caída brusca de precisión
  al aumentar el recall indica que el modelo empieza a producir muchos falsos positivos para
  recuperar más verdaderos positivos. Para el dataset CDC (13.6 % positivos), un PR-AUC
  de 0.42 es significativamente mejor que el baseline aleatorio de 0.136.

```python
from PIL import Image   # pip install Pillow
img = Image.open("corrida_10k/curvas_svm.png")
img.show()
```

---

## Artefacto 7 — `calibracion_{modelo_ganador}.png`

Curva de calibración generada por `EvaluadorClinico.graficar_curva_calibracion()`.
Compara la probabilidad media predicha por el modelo en cada intervalo de confianza con la
fracción real de positivos en ese intervalo.

**Cómo leerla:**

- La diagonal `y = x` representa calibración perfecta.
- Curva **por encima de la diagonal:** el modelo es conservador (subestima las probabilidades).
- Curva **por debajo de la diagonal:** el modelo es optimista (sobreestima las probabilidades).
- Para tamizaje clínico, importa que las probabilidades estén bien calibradas cerca del umbral
  de decisión (0.33–0.66), no necesariamente en los extremos.

El **Brier Score** del JSON es la métrica numérica complementaria a esta gráfica:
`Brier = mean((prob_predicha - clase_real)²)`.

---

## Artefacto 8 — `muestra_{tag}.csv`

El CSV exacto usado para entrenar y evaluar la corrida. Generado con:

```python
df_full.sample(n, random_state=42)
```

Con la misma semilla (`random_state=42`) sobre el mismo CSV fuente, se obtiene exactamente
la misma muestra — **la corrida es 100 % reproducible**:

```bash
# Reproducir la corrida desde cero
python -m entrenamiento.pipeline \
  --modo clasificacion \
  --dataset resultados/corrida_10k/muestra_10k.csv \
  --salida-modelo /tmp/modelo_check.joblib \
  --salida-reporte /tmp/corrida_check.json
```

El dataset tiene 22 columnas (21 CDC + `Diabetes_binary`). El pipeline separa el objetivo
internamente antes de ajustar el preprocesador.

---

## Artefacto 9 — `dataset_procesado.parquet`

Snapshot del dataset de entrenamiento después del preprocesamiento inicial (carga + limpieza
numérica), con las **21 columnas CDC sin la columna objetivo**. Guardado por:
`CargadorDatos.persistir_procesado(df[COLUMNAS_CDC])`.

> **Importante:** contiene toda la muestra (train + test), no solo train. No usar para ajustar
> transformadores — eso introduciría data leakage. Usarlo para análisis exploratorio, generación
> de fenotipos K-Means o auditorías de distribución.

```python
import pandas as pd
df = pd.read_parquet("corrida_10k/dataset_procesado.parquet")
print(df.shape)          # (10000, 21)
print(df.dtypes)         # todos float64
print(df.isnull().sum()) # 0 nulos — dataset limpio
```

---

## `LOG_CORRIDAS.md`

Documento narrativo generado por `scripts/ejecutar_corridas.py` que reúne en un solo lugar:

1. **Parámetros globales** (semilla, proporción test, use_knn, use_smote)
2. **Arquitectura del preprocesamiento** por tipo de variable, con justificación de cada decisión
3. **Hiperparámetros por modelo** (SVM: espacio de búsqueda y protocolo; GBM, MLP, Árbol: params fijos con motivo)
4. **Resultados por corrida** (tabla de métricas, ganador, tiempo total)
5. **Comparativa entre corridas** (evolución de ROC-AUC al escalar n)
6. **Instrucción para la instancia separada** (dataset completo 253k + SVM)

---

## Comparar corridas programáticamente

```python
import json
from pathlib import Path

def resumen_corrida(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    mejor = d["mejor_modelo"]
    m = d["modelos"][mejor]
    return {
        "n":           d["n_muestras"],
        "ganador":     mejor,
        "roc_auc":     m["roc_auc"],
        "pr_auc":      m["pr_auc"],
        "sensibilidad":m["sensibilidad"],
        "brier":       m["brier_score"],
    }

corridas = {
    "10k":  "corrida_10k/corrida_10k.json",
    "50k":  "corrida_50k/corrida_50k.json",
    "253k": "corrida_253k/corrida_253k.json",   # disponible tras instancia separada
}

for tag, path in corridas.items():
    p = Path("resultados") / path
    if p.exists():
        print(tag, resumen_corrida(p))
```

---

## Referencia rápida de umbrales clínicos del sistema

| Umbral | Valor | Significado |
|---|---|---|
| `UMBRAL_RIESGO_BAJO` | 0.33 | prob < 0.33 → riesgo bajo |
| `UMBRAL_RIESGO_ALTO` | 0.66 | prob ≥ 0.66 → riesgo alto |
| `MARGEN_INCERTIDUMBRE` | ±0.05 | zona gris → advertencia clínica en la API |
| Mínimo ROC-AUC aceptable | 0.75 | definido en `config.UMBRAL_MINIMO_AUC` |

Definidos en `config.py`. Ajustar `UMBRAL_RIESGO_BAJO` a 0.25 para población mexicana
(calibración conservadora recomendada en `docs/` por diferencias CDC ↔ ENSANUT).

---

*Documentación mantenida en `resultados/README.md` — actualizar si cambia la estructura del pipeline.*
