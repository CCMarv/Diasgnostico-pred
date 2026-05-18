"""Ejecuta corridas académicas reproducibles de 10k y 50k muestras.

Por cada corrida:
- Muestrea n filas del CSV bruto con semilla=42
- Ejecuta el pipeline completo (SVM, Árbol, GBM, MLP) con KNN + SMOTE
- Copia todos los artefactos a resultados/corrida_{n}/
- Genera resultados/LOG_CORRIDAS.md con narrativa completa de parámetros y resultados

Uso:
    python scripts/ejecutar_corridas.py
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    REPORTES_DIR,
    SEMILLA_ALEATORIA,
    MODELOS_DIR,
)
from entrenamiento.pipeline import ejecutar_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
_LOG = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent
CSV_BRUTO = RAIZ / "datos" / "brutos" / "diabetes_binary_health_indicators_BRFSS2015.csv"
RESULTADOS_DIR = RAIZ / "resultados"
TAMAÑOS = [10_000, 50_000]


# ── helpers ────────────────────────────────────────────────────────────────────

def _etiquetar(n: int) -> str:
    return f"{n // 1000}k"


def _copiar_artefactos(
    dir_corrida: Path,
    mejor_modelo: str,
) -> list[str]:
    """Copia hacia dir_corrida los artefactos que el pipeline siempre deposita en rutas fijas."""
    # El pipeline ya guarda modelo final y versionado directamente en dir_corrida
    # (porque ruta_modelo apunta ahí). Solo hace falta copiar las PNGs y el parquet.
    copiados: list[str] = []

    for patron in [f"curvas_{mejor_modelo}.png", f"calibracion_{mejor_modelo}.png"]:
        src = REPORTES_DIR / patron
        if src.exists():
            dst = dir_corrida / patron
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copiados.append(patron)

    parquet_src = RAIZ / "datos" / "procesados" / "dataset_procesado.parquet"
    if parquet_src.exists():
        dst = dir_corrida / "dataset_procesado.parquet"
        if parquet_src.resolve() != dst.resolve():
            shutil.copy2(parquet_src, dst)
        copiados.append(dst.name)

    return copiados


def _encabezado_parametros() -> str:
    """Devuelve la sección narrativa de parámetros e hiperparámetros."""
    return """\
## Parámetros globales de configuración

| Parámetro | Valor | Motivo |
|---|---|---|
| `SEMILLA_ALEATORIA` | `42` | Reproducibilidad entre corridas |
| `PROPORCION_PRUEBA` | `0.20` | 80/20 estratificado; la estratificación preserva el ratio de clases |
| `use_knn` | `True` | KNNImputer para continuas (BMI, MentHlth, PhysHlth) en lugar de mediana |
| `use_smote` | `True` | SMOTE activo en entrenamiento para corregir desbalance ~14% clase 1 |
| Modelos | `svm, arbol, gbm, mlp` | Catálogo completo del nivel Intermedio |

---

## Arquitectura del pipeline de preprocesamiento

El preprocesador es un `ColumnTransformer` con tres ramas; el pipeline completo es
`ColumnTransformer → SMOTE → Estimador` serializado en un único `.joblib`.

### Rama 1 — Variables continuas (3 columnas: BMI, MentHlth, PhysHlth)

| Paso | Clase | Parámetros clave |
|---|---|---|
| Imputación | `KNNImputer` | `n_neighbors=5` (default sklearn), `weights='uniform'` |
| Escalado | `StandardScaler` | `with_mean=True, with_std=True` (defaults) |

> **Por qué KNN en lugar de mediana:** las tres continuas tienen distribuciones asimétricas y
> valores extremos (BMI puede ser 99). La imputación por vecinos respeta la estructura local;
> la mediana reemplaza con un escalar global que distorsiona casos atípicos.

### Rama 2 — Variables binarias (14 columnas)

| Paso | Clase | Parámetros |
|---|---|---|
| Imputación | `SimpleImputer` | `strategy='most_frequent'` |
| Escalado | `'passthrough'` | Sin transformar — variables 0/1 ya están en escala |

> **Por qué no escalar binarias:** StandardScaler centraría en 0.5 y expandiría a ±0.5,
> perdiendo la interpretabilidad booleana sin ganancia para los modelos de árbol.

### Rama 3 — Variables ordinales (4 columnas: GenHlth, Age, Education, Income)

| Paso | Clase | Parámetros |
|---|---|---|
| Imputación | `SimpleImputer` | `strategy='most_frequent'` |
| Encoding | `OrdinalEncoder` | categorías predefinidas por columna (ver tabla) |

| Variable | Rango | Semántica |
|---|---|---|
| `GenHlth` | 1–5 | 1=Excelente, 5=Mala |
| `Age` | 1–13 | 1=18-24, 13=80+ |
| `Education` | 1–6 | 1=sin estudios, 6=universidad |
| `Income` | 1–8 | 1=<$10k, 8=>$75k |

> **Por qué OrdinalEncoder con categorías explícitas y no OneHotEncoder:** las cuatro
> variables tienen orden clínico real (mayor edad → mayor riesgo). OHE lo destruiría.
> Definir el rango completo evita que valores no vistos en train causen errores en inferencia.

### SMOTE (aplicado post-preprocesador, pre-estimador)

| Parámetro | Valor |
|---|---|
| `k_neighbors` | 5 (default) |
| `random_state` | None (no fijado) — variación aleatoria en los sintéticos |
| Estrategia | `'auto'` → resamplea solo la clase minoritaria hasta igualar clases |

> **Por qué dentro del Pipeline y no antes del split:** para evitar data leakage. Si se
> aplicara SMOTE antes, muestras sintéticas derivadas del test contaminarían el train.

---

## Hiperparámetros por modelo

### SVM — `SVC(kernel='rbf')`

La SVM es el único modelo con búsqueda de hiperparámetros explícita.

**Espacio de búsqueda:**

| Parámetro | Valores explorados | Combinaciones |
|---|---|---|
| `C` | `[0.1, 1.0, 10.0]` | controla margen blando: C alto → menos margen, más sobreajuste |
| `gamma` | `['scale', 'auto']` | `scale` = 1/(n_features·Var(X)); `auto` = 1/n_features |
| **Total** | **6 combinaciones × 5 folds** | **= 30 evaluaciones** |

**Protocolo de búsqueda:** `ParameterGrid` manual sobre `StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)`. Se usa búsqueda manual (no `GridSearchCV`) porque la
calibración interna de probabilidades (`CalibratedClassifierCV`) dentro de cada fold de
GridSearchCV multiplicaría el costo por 3–5×. En cambio, se usa `probability=False`
durante la búsqueda y `probability=True` solo en el refit final.

**Refit final:** `pipeline.fit(X_train, y_train)` con los mejores hiperparámetros sobre
todo el conjunto de entrenamiento.

**Parámetros fijos:** `class_weight='balanced'` (penaliza errores en clase minoritaria
~14× más), `random_state=42`.

---

### Árbol de decisión — `DecisionTreeClassifier`

Sin búsqueda de hiperparámetros; validación cruzada solo para reportar score CV.

| Parámetro | Valor | Motivo |
|---|---|---|
| `max_depth` | `5` | Limita sobreajuste; profundidad suficiente para capturar interacciones clínicas |
| `ccp_alpha` | `0.0` | Sin poda adicional por complejidad |
| `class_weight` | `'balanced'` | Pondera inversamente a la frecuencia de clase |
| `criterion` | `'gini'` (default) | Impureza de Gini para selección de división |
| `random_state` | `42` | Reproducibilidad en desempates |

> **Por qué max_depth=5 y no más profundo:** un árbol sin límite de profundidad en este
> dataset memorizaría el ruido de entrenamiento. Con 5 niveles se pueden representar hasta
> 32 hojas — suficiente para capturar patrones relevantes sin sobreajustar.

---

### Gradient Boosting — `GradientBoostingClassifier`

Sin búsqueda de hiperparámetros; parámetros fijados por diseño.

| Parámetro | Valor | Motivo |
|---|---|---|
| `n_estimators` | `200` | 200 árboles secuenciales de corrección de error |
| `max_depth` | `4` | Árboles base poco profundos para capturar interacciones sin sobreajustar |
| `learning_rate` | `0.05` | Tasa de aprendizaje conservadora — cada árbol contribuye poco; más estable |
| `subsample` | `1.0` (default) | Usa todo el dataset por iteración |
| `loss` | `'log_loss'` (default en clasificación) | Equivalente a regresión logística por iteración |
| `random_state` | `42` | |

> **Por qué learning_rate bajo con muchos estimadores:** el tradeoff clásico boosting.
> lr=0.05 con 200 estimadores supera en generalización a lr=0.2 con 50, especialmente
> en datasets desbalanceados donde el gradient de la pérdida varía mucho entre clases.

---

### Red Neuronal — `MLPClassifier`

Sin búsqueda de hiperparámetros.

| Parámetro | Valor | Motivo |
|---|---|---|
| `hidden_layer_sizes` | `(64, 32)` | 2 capas ocultas: 64 → 32 neuronas con compresión de representación |
| `activation` | `'relu'` | Rectified Linear Unit; evita problema del gradiente que desaparece |
| `solver` | `'adam'` | Optimizador adaptativo; maneja bien datos esparsos/desbalanceados |
| `max_iter` | `500` | Máximo de épocas |
| `early_stopping` | `True` | Detiene si la métrica de validación no mejora en 10 épocas (n_iter_no_change=10) |
| `validation_fraction` | `0.1` | 10% del train como validación interna para early stopping |
| `random_state` | `42` | |

> **Por qué early_stopping=True:** evita que la red memorice el ruido de los datos
> sintéticos generados por SMOTE, que pueden ser más "ruidosos" que los datos reales.

---

## Protocolo de evaluación (todos los modelos)

- **Split:** 80/20 estratificado, `random_state=42`
- **Validación cruzada:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Métrica de selección:** ROC-AUC promedio en los 5 folds (no accuracy — sensible al desbalance)
- **Métrica principal de comparación final:** ROC-AUC en conjunto de test
- **Serialización:** el mejor modelo por ROC-AUC en test se serializa como pipeline completo

---
"""


def _generar_log(
    corridas: list[dict],
    ruta_salida: Path,
) -> None:
    """Genera LOG_CORRIDAS.md con narrativa completa de parámetros y resultados."""
    fecha = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lineas: list[str] = [
        "# Log de corridas académicas — Diagnóstico predictivo de diabetes",
        "",
        f"**Generado:** {fecha}  ",
        f"**Dataset:** CDC BRFSS 2015 ({CSV_BRUTO.name})  ",
        f"**Semilla global:** {SEMILLA_ALEATORIA}  ",
        "",
        "---",
        "",
        _encabezado_parametros(),
        "---",
        "",
        "## Resultados por corrida",
        "",
    ]

    for corrida in corridas:
        n = corrida["n"]
        tag = _etiquetar(n)
        metricas = corrida["metricas"]
        elapsed = corrida["elapsed"]
        modelos = metricas.get("modelos", {})
        mejor = metricas.get("mejor_modelo", "—")
        mejor_pr = metricas.get("mejor_modelo_por_pr_auc", "—")
        desbalance = metricas.get("desbalance", {})

        lineas += [
            f"### Corrida {tag} ({n:,} muestras)",
            "",
            f"- **Tiempo total:** {elapsed:.1f}s ({elapsed / 60:.1f} min)",
            f"- **Mejor modelo (ROC-AUC):** `{mejor}`",
            f"- **Mejor modelo (PR-AUC):** `{mejor_pr}`",
            f"- **Prevalencia clase positiva:** {desbalance.get('pct_clase_1', 0):.1%} (ratio {desbalance.get('ratio', 0):.1f}:1)",
            "",
            "| Modelo | ROC-AUC | PR-AUC | Sensibilidad | Especificidad | F1 | Brier |",
            "|--------|---------|--------|-------------|--------------|-----|-------|",
        ]
        for nombre, m in sorted(modelos.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
            marca = " ← **ganador**" if nombre == mejor else ""
            lineas.append(
                f"| `{nombre}`{marca} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                f"{m['sensibilidad']:.4f} | {m['especificidad']:.4f} | "
                f"{m['f1_clase_positiva']:.4f} | {m['brier_score']:.4f} |"
            )
        lineas.append("")

    # Comparativa entre corridas
    if len(corridas) >= 2:
        lineas += [
            "---",
            "",
            "## Comparativa entre corridas (ganadores por escala)",
            "",
            "| n | Modelo ganador | ROC-AUC | PR-AUC | Sensibilidad | Tiempo |",
            "|---|----------------|---------|--------|-------------|--------|",
        ]
        for corrida in corridas:
            n = corrida["n"]
            m = corrida["metricas"]
            mejor = m.get("mejor_modelo", "—")
            mm = m.get("modelos", {}).get(mejor, {})
            elapsed = corrida["elapsed"]
            lineas.append(
                f"| {n:,} | `{mejor}` | {mm.get('roc_auc', 0):.4f} | "
                f"{mm.get('pr_auc', 0):.4f} | {mm.get('sensibilidad', 0):.4f} | "
                f"{elapsed:.0f}s |"
            )
        lineas.append("")

    # Instrucción para la instancia completa
    lineas += [
        "---",
        "",
        "## Instrucción para la instancia con dataset completo (253k + SVM)",
        "",
        "Esta instrucción permite ejecutar el pipeline en una instancia separada con el",
        "dataset completo (253,680 filas) incluyendo SVM. Los parámetros son idénticos a",
        "las corridas de este log para garantizar comparabilidad directa.",
        "",
        "**Requisitos previos:**",
        "```bash",
        "# 1. Python 3.11+ y el repositorio clonado",
        "git clone <repo_url>",
        "cd diasgnostico-pred",
        "",
        "# 2. Instalar dependencias",
        "pip install -e .[dev]",
        "",
        "# 3. Descargar el dataset (si no está ya en datos/brutos/)",
        "python -c \"from entrenamiento.descargador_dataset import descargar_y_persistir; descargar_y_persistir()\"",
        "```",
        "",
        "**Comando de ejecución:**",
        "```bash",
        "mkdir -p resultados/corrida_253k",
        "",
        "python -m entrenamiento.pipeline \\",
        "  --modo clasificacion \\",
        "  --modelos svm,arbol,gbm,mlp \\",
        "  --dataset datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv \\",
        "  --salida-modelo resultados/corrida_253k/modelo_253k.joblib \\",
        "  --salida-reporte resultados/corrida_253k/corrida_253k.json \\",
        "  --salida-reporte-legible resultados/corrida_253k/corrida_253k.md",
        "```",
        "",
        "**Parámetros implícitos (no cambian, vienen de `config.py`):**",
        "",
        "| Parámetro | Valor | Archivo |",
        "|---|---|---|",
        f"| `SEMILLA_ALEATORIA` | `{SEMILLA_ALEATORIA}` | `config.py` |",
        "| `PROPORCION_PRUEBA` | `0.20` | `config.py` |",
        "| `use_knn` | `True` | `ejecutar_pipeline()` default |",
        "| `use_smote` | `True` | `ejecutar_pipeline()` default |",
        "| SVM grid C | `[0.1, 1.0, 10.0]` | `comparador_modelos.py` |",
        "| SVM grid gamma | `['scale', 'auto']` | `comparador_modelos.py` |",
        "| CV folds | `5 folds StratifiedKFold` | `comparador_modelos.py` |",
        "| GBM n_estimators | `200` | `comparador_modelos.py` |",
        "| GBM learning_rate | `0.05` | `comparador_modelos.py` |",
        "| MLP arquitectura | `(64, 32)` | `comparador_modelos.py` |",
        "",
        "**Tiempo estimado:** ~100 min (SVM es el cuello de botella — 6 combinaciones × 5 folds sobre 200k+ filas).",
        "",
        "**Artefactos esperados en `resultados/corrida_253k/`:**",
        "```",
        "corrida_253k.json          # métricas brutas de 4 modelos",
        "corrida_253k.md            # reporte Markdown legible",
        "corrida_253k_manifest.json # manifiesto de la corrida",
        "modelo_253k.joblib         # pipeline serializado del ganador",
        "curvas_{ganador}.png       # curvas ROC y PR",
        "calibracion_{ganador}.png  # curva de calibración",
        "dataset_procesado.parquet  # dataset 21 cols CDC sin objetivo",
        "```",
        "",
        "**Para copiar las PNGs** (el pipeline las guarda en `reportes/` por defecto):",
        "```bash",
        "# Ejecutar después del pipeline para copiar curvas al directorio de la corrida",
        "cp reportes/curvas_*.png reportes/calibracion_*.png resultados/corrida_253k/",
        "```",
        "",
        "**Para comparar con esta corrida:** los JSONs de `resultados/corrida_10k/corrida_10k.json`",
        "y `resultados/corrida_50k/corrida_50k.json` tienen la misma estructura que",
        "`resultados/corrida_253k/corrida_253k.json`. Se puede comparar directamente con:",
        "```bash",
        "python -c \"",
        "import json",
        "for tag, f in [('10k','corrida_10k'), ('50k','corrida_50k'), ('253k','corrida_253k')]:",
        "    try:",
        "        d = json.load(open(f'resultados/corrida_{tag}/{f}.json'))",
        "        m = d['modelos']",
        "        mejor = d['mejor_modelo']",
        "        print(f'{tag}: {mejor} ROC-AUC={m[mejor][\\\"roc_auc\\\"]:.4f}')",
        "    except FileNotFoundError:",
        "        print(f'{tag}: no disponible')",
        "\"",
        "```",
        "",
        "---",
        "",
        "*Log generado por `scripts/ejecutar_corridas.py`*",
    ]

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text("\n".join(lineas), encoding="utf-8")
    _LOG.info("Log narrativo escrito en %s", ruta_salida)


# ── ejecución principal ────────────────────────────────────────────────────────

def main() -> None:
    if not CSV_BRUTO.exists():
        _LOG.error("Dataset no encontrado: %s", CSV_BRUTO)
        _LOG.error("Ejecuta: python -c \"from entrenamiento.descargador_dataset import descargar_y_persistir; descargar_y_persistir()\"")
        sys.exit(1)

    _LOG.info("Cargando CSV bruto (%s)…", CSV_BRUTO.name)
    df_full = pd.read_csv(CSV_BRUTO)
    _LOG.info("Dataset completo: %d filas × %d columnas", *df_full.shape)

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    historial: list[dict] = []

    for n in TAMAÑOS:
        tag = _etiquetar(n)
        dir_corrida = RESULTADOS_DIR / f"corrida_{tag}"
        dir_corrida.mkdir(parents=True, exist_ok=True)

        _LOG.info("=" * 60)
        _LOG.info("CORRIDA %s — %d muestras", tag.upper(), n)
        _LOG.info("=" * 60)

        # muestra reproducible
        muestra = df_full.sample(n, random_state=SEMILLA_ALEATORIA)
        csv_muestra = dir_corrida / f"muestra_{tag}.csv"
        muestra.to_csv(csv_muestra, index=False)
        _LOG.info("Muestra guardada en %s", csv_muestra)

        # rutas de salida
        ruta_reporte = dir_corrida / f"corrida_{tag}.json"
        ruta_modelo = dir_corrida / f"predictor_{tag}.joblib"

        # ejecutar pipeline
        t0 = time.perf_counter()
        metricas = ejecutar_pipeline(
            modo="clasificacion",
            ruta_dataset=csv_muestra,
            ruta_modelo=ruta_modelo,
            ruta_reporte=ruta_reporte,
        )
        elapsed = time.perf_counter() - t0

        # copiar artefactos adicionales
        mejor_modelo = metricas.get("mejor_modelo", "")
        copiados = _copiar_artefactos(dir_corrida, mejor_modelo)
        _LOG.info("Artefactos copiados a %s: %s", dir_corrida, copiados)

        historial.append({"n": n, "metricas": metricas, "elapsed": elapsed})
        _LOG.info("Corrida %s completada en %.1fs", tag, elapsed)

    # log narrativo
    _generar_log(historial, RESULTADOS_DIR / "LOG_CORRIDAS.md")
    _LOG.info("Corridas finalizadas. Artefactos en %s", RESULTADOS_DIR)


if __name__ == "__main__":
    main()
