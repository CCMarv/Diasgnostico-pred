# Notas: Gestión del Desbalance con SMOTE y NearMiss

**Unidad 1.3 — Gestión del Desbalance**  
Tema: Técnicas de remuestreo y pipelines integrados | Dataset: `fraude.csv`

---

## ¿Qué problema resuelve este notebook?

Se construye un clasificador para detectar **transacciones fraudulentas** (`es_fraude`), un problema clásico de datos desbalanceados donde los fraudes son una minoría muy pequeña frente a las transacciones legítimas.

El notebook demuestra cómo manejar ese desbalance correctamente dentro de un pipeline completo.

---

## Librerías necesarias

```python
pip install imbalanced-learn
pip install seaborn
```

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as ImbPipeline  # ojo: imblearn, no sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
```

---

## Flujo del notebook paso a paso

### Paso 1 — Separar target

```python
X = df.drop('es_fraude', axis=1)
y = df['es_fraude']
```

### Paso 2 — Split estratificado

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

> `stratify=y` asegura que la proporción de fraudes se mantenga igual en train y test. Es obligatorio en datos desbalanceados.

### Paso 3 — ColumnTransformer (preprocesamiento por tipo de dato)

```python
num_features = ['monto', 'antiguedad_cuenta']
cat_features = ['tipo_tarjeta']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(), cat_features)
])
```

Cada columna recibe la transformación que le corresponde según su tipo.

### Paso 4 — Pipeline con SMOTE (oversampling)

```python
pipeline_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_smote.fit(X_train, y_train)
```

> SMOTE genera ejemplos **sintéticos** de la clase minoritaria interpolando entre vecinos reales. El parámetro `sampling_strategy=0.3` indica el ratio deseado minoría/mayoría.

### Paso 5 — Evaluación con métricas correctas

```python
y_pred  = pipeline_smote.predict(X_test)
y_probs = pipeline_smote.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC:              {roc_auc_score(y_test, y_probs):.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")
```

> **Nunca uses accuracy** en datos desbalanceados. Las métricas relevantes son Precision-Recall AUC y ROC-AUC.

La visualización incluye la **matriz de confusión** con `seaborn.heatmap`.

### Paso 6 — Comparación con NearMiss (undersampling)

```python
pipeline_nearmiss = ImbPipeline([
    ('preprocessor', preprocessor),
    ('nearmiss', NearMiss(version=1)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_nearmiss.fit(X_train, y_train)
```

NearMiss **elimina** ejemplos de la clase mayoritaria eligiendo los más cercanos a la minoría, en lugar de generar datos nuevos.

---

## Conceptos clave

### ¿Por qué `imblearn.Pipeline` y no `sklearn.Pipeline`?

| | `sklearn.Pipeline` | `imblearn.Pipeline` |
|---|---|---|
| Soporta `fit_resample` | No | Sí |
| Aplica SMOTE solo en train | No | Sí |
| Uso correcto en cross-val | No | Sí |

> Usar `sklearn.Pipeline` con SMOTE causaría que el remuestreo se aplique sobre los datos de test → data leakage.

### SMOTE vs NearMiss

| | SMOTE (Oversampling) | NearMiss (Undersampling) |
|---|---|---|
| Acción | Crea datos sintéticos minoritarios | Elimina datos mayoritarios |
| Tamaño final | Dataset más grande | Dataset más pequeño |
| Riesgo | Overfitting si se aplica mal | Pérdida de información |
| Útil cuando | Dataset pequeño | Dataset muy grande |

### Parámetros importantes de SMOTE

| Parámetro | Descripción |
|---|---|
| `sampling_strategy` | Ratio final minoría/mayoría (ej. `0.3` → 30%) |
| `k_neighbors` | Vecinos usados para interpolar (default: 5) |
| `random_state` | Reproducibilidad |

---

## Regla de oro

> Nunca apliques SMOTE **antes** de dividir en train/test. Siempre dentro del pipeline para que el remuestreo ocurra solo sobre los datos de entrenamiento en cada fold de validación cruzada.

---

## Conexión con el programa

Este notebook cubre directamente el tema **1.3 Gestión del Desbalance**:

- Métricas apropiadas: Precision-Recall AUC vs ROC-AUC
- Técnicas de remuestreo: SMOTE y NearMiss
- `ColumnTransformer` (tema 1.3) para preprocesamiento heterogéneo
- `imblearn.Pipeline` para evitar data leakage (tema 1.1)
