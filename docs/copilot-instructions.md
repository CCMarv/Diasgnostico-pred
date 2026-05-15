# INSTRUCCIÓN MAESTRA — GitHub Copilot Agent
## Proyecto: `diasgnostico-pred` · Implementación Secuencial del Roadmap

---

## ROL Y CONTEXTO

Eres el ingeniero principal de ML del proyecto `diasgnostico-pred`. Tu objetivo es
implementar el roadmap de forma **secuencial, incremental y sin romper contratos públicos**.
Estado actualizado: Sprint 1 completado y Sprint 2 parcialmente completado en código.
**Nunca modifiques las firmas públicas de `api/esquemas.py`, `inferencia/predictor.py`
ni `config.py` salvo para agregar constantes nuevas compatibles.**

### Stack técnico fijo
- Python ≥ 3.11, scikit-learn ≥ 1.5, FastAPI ≥ 0.112, Pydantic v2
- Convención de nombres: **español** para clases/métodos/variables propias;
  inglés solo para columnas CDC oficiales y nombres de librerías externas
- Serialización de modelos: `joblib` exclusivamente
- Todo modelo entrenado se guarda como `sklearn.Pipeline` completo
  (preprocesador + estimador) para garantizar compatibilidad con `PredictorDiabetes`

---

## REGLAS GLOBALES DE IMPLEMENTACIÓN

1. **Secuencialidad estricta**: no inicies un sprint si el anterior tiene tickets
   en rojo (pruebas fallando o artefactos requeridos ausentes).
2. **Sin data leakage**: el `ColumnTransformer`/`Pipeline` se ajusta **solo** sobre
   `X_train`. Nunca sobre `X_test` ni sobre el conjunto completo antes de partir.
3. **Pruebas primero**: cada módulo nuevo lleva su `pruebas/test_<modulo>.py`.
   Ejecuta `pytest -q` antes de declarar un ticket completado.
4. **Compatibilidad garantizada**: `PredictorDiabetes.predecir()` recibe un
   `pd.DataFrame` de una fila con las 21 columnas CDC y llama `predict_proba`.
   El `Pipeline` serializado debe responder a `predict_proba` sin modificar el predictor.
5. **Constantes en `config.py`**: ningún módulo define rutas, semillas ni umbrales
   como literales. Importa siempre desde `config`.
6. **Dataset via UCI ML Repo**: usar `ucimlrepo` para obtener el dataset.
   Ver sección SPRINT 2 > Ticket S2-01 para el snippet exacto.

---

## Reglas de refactorización y compatibilidad

Cuando se realicen refactorizaciones para alinear la base de código con `docs/implementacion_modelos`, aplicar las siguientes reglas adicionales:

- Mantener sin cambios las **firmas públicas** de los endpoints y clases críticas (`api/esquemas.py`, `inferencia/predictor.py`, `config.py`).
- Los refactors deben dividirse en commits pequeños y atómicos con tests que verifiquen compatibilidad.
- Si una implementación concreta no es técnicamente recomendable cambiar (por razones de rendimiento, compatibilidad o pruebas), no la modifiques: añade un comentario en el código explicando la decisión arquitectónica y referencia este archivo.
- Documentar en el PR el ticket del `docs/ROADMAP.md` asociado al refactor.

Estas reglas complementan las existentes y garantizan que las mejoras no rompan el contrato con la API ni los tests de integración.

---

## ESTADO REAL DE AVANCE (actualizado: 2026-05-14)

### Sprint 2

| Ticket | Estado | Evidencia actual |
|---|---|---|
| S2-01 Descargar dataset | ✅ Completado | Dataset descargado en `datos/brutos/` usando `entrenamiento/descargador_dataset.py`  |
| S2-02 Notebook EDA | ✅ Completado | Notebook creado en `notebooks/01_eda_regionalizado.ipynb` |
| S2-03 Preprocesador | ✅ Completado | `entrenamiento/preprocesador.py` implementado |
| S2-04 Modelos supervisados | ✅ Completado | `ComparadorModelos` ya incluye `svm`, `arbol`, `gbm`, `mlp` |
| S2-05 Evaluador clínico | ✅ Completado | `entrenamiento/evaluador.py` implementado |
| S2-06 Pipeline serializable | ✅ Completado | `entrenamiento/pipeline.py` guarda modelos versionados y reporte de métricas |
| S2-07 Pruebas preprocesador | ✅ Completado | `pruebas/test_preprocesador.py` existe y pasa |
| S2-08 Pruebas cargador ampliadas | ✅ Completado | `pruebas/test_cargador.py` incluye desbalance/distribución/parquet |
| S2-09 Persistencia Parquet | 🟡 Parcial | Método `persistir_procesado` existe; sin artefacto real versionado |
| S2-10 Contraste CDC↔ENSANUT | 🟡 Parcial | Existe `reportes/generar_contraste_regional.py`; falta `reportes/contraste_regional.md` generado con datos reales |

### Sprint 3

Estado general: ⬜ No iniciado (sin `fenotipador.py`, sin carpeta `dashboard/`, sin SHAP).

### Sprint 4

Estado general: ⬜ No iniciado (sin paper final, sin Docker, sin workflows de CI/CD).

> **Regla operativa desde este punto:** cualquier trabajo nuevo debe enfocarse solo en tickets pendientes o parciales; los tickets marcados como completados se consideran cerrados salvo corrección puntual.

---

## SPRINT 2 — EDA Regionalizado & Modelado Base

> **Prerequisito**: `pytest -q` pasa al 100% con el código del Sprint 1.
> **Desbloquea**: Sprint 3 completo.

### Ticket S2-01 — Obtener dataset CDC BRFSS 2015 vía UCI ML Repo

**Archivo a crear**: `entrenamiento/descargador_dataset.py`

```python
"""
Propósito:
Descargar el dataset CDC BRFSS 2015 desde UCI ML Repository y persistirlo
en datos/brutos/ para uso offline en el resto del pipeline.

Firma pública:
  descargar_y_persistir(ruta_destino: Path | None = None) -> Path
    → Retorna ruta al CSV guardado.
    → Si el archivo ya existe y --forzar no se indica, retorna sin re-descargar.

Dependencia externa requerida:
  pip install ucimlrepo  # añadir a pyproject.toml [dependencies]

Snippet de descarga (NO modificar la interfaz ucimlrepo):
  from ucimlrepo import fetch_ucirepo
  repo = fetch_ucirepo(id=891)
  X = repo.data.features   # pd.DataFrame con las 21 columnas CDC
  y = repo.data.targets    # pd.DataFrame con columna 'Diabetes_binary'
  df = pd.concat([X, y], axis=1)
  df.to_csv(ruta_destino, index=False)

Validaciones post-descarga:
  - Verificar que las 21 COLUMNAS_CDC están presentes (importar de config)
  - Verificar que 'Diabetes_binary' está presente
  - Verificar N > 200_000 filas (dataset completo tiene ~253,680)
  - Loggear: shape, prevalencia de clase 1, valores nulos por columna
"""
```

Actualizar `config.py` agregando:
```python
NOMBRE_DATASET_UCI_ID: Final[int] = 891
```

Actualizar `pyproject.toml` agregando `"ucimlrepo>=0.0.3"` en `dependencies`.

---

### Ticket S2-02 — Ampliar CargadorDatos con análisis de distribución

**Archivo a modificar**: `entrenamiento/cargador_datos.py`

Agregar los siguientes métodos a la clase `CargadorDatos` **sin modificar** los
existentes `cargar()` y `limpieza_basica()`:

```python
def analizar_distribucion(self, dataframe: pd.DataFrame) -> dict[str, dict]:
    """
    Propósito: calcular estadísticas descriptivas por columna para el EDA.
    Retorno: dict con clave=columna_cdc, valor=dict con keys:
      media, mediana, std, min, max, nulos, pct_nulos
    Para binarias: añadir pct_positivos.
    Para COLUMNA_OBJETIVO: añadir pct_clase_1 y ratio_desbalance.
    """

def detectar_desbalance(self, y: pd.Series) -> dict[str, float]:
    """
    Propósito: cuantificar desbalance de clases.
    Retorno: {
      'ratio': float,          # n_mayoritaria / n_minoritaria
      'pct_clase_0': float,
      'pct_clase_1': float,
      'recomendacion': str     # 'class_weight' | 'smote' | 'submuestreo'
    }
    Regla: ratio > 5 → recomendar class_weight='balanced';
           ratio > 10 → recomendar smote.
    """

def cargar_con_objetivo(self, ruta_dataset: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Propósito: retornar (X, y) listos para train_test_split.
    X: DataFrame con 21 columnas CDC.
    y: Series con COLUMNA_OBJETIVO.
    """
```

---

### Ticket S2-03 — Implementar preprocesador sin data leakage

**Archivo a crear**: `entrenamiento/preprocesador.py`

```python
"""
Propósito:
Construir un sklearn.compose.ColumnTransformer serializable que aplica
transformaciones apropiadas por tipo de columna CDC.

Clase: ConstructorPreprocesador

Constantes de clase (no instancia):
  COLUMNAS_CONTINUAS = ['BMI', 'MentHlth', 'PhysHlth']
  COLUMNAS_BINARIAS  = [  # las 14 binarias 0/1
      'HighBP','HighChol','CholCheck','Smoker','Stroke',
      'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
      'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex'
  ]
  COLUMNAS_ORDINALES = ['GenHlth','Age','Education','Income']
  ORDENES_ORDINALES  = {
      'GenHlth':   [1,2,3,4,5],
      'Age':       list(range(1,14)),
      'Education': list(range(1,7)),
      'Income':    list(range(1,9)),
  }

Método: construir(self) -> ColumnTransformer
  Pipeline continuas: SimpleImputer(strategy='median') → StandardScaler()
  Pipeline binarias:  SimpleImputer(strategy='most_frequent') → passthrough
  Pipeline ordinales: SimpleImputer(strategy='most_frequent')
                      → OrdinalEncoder(categories=lista_ordenes)

Método: construir_pipeline(self, clasificador) -> sklearn.Pipeline
  Retorna Pipeline([('preprocesador', self.construir()), ('clasificador', clasificador)])
  CRÍTICO: el pipeline completo es lo que se serializa con joblib.dump()
           para que PredictorDiabetes.predecir() funcione sin cambios.

Método: construir_pipeline_con_fenotipo(self, clasificador) -> sklearn.Pipeline
  Igual que construir_pipeline pero agrega la columna 'fenotipo' (int, ordinal)
  al ColumnTransformer antes del clasificador.
  Se usa en Sprint 3 cuando el FenotipadorMetabolico ya enriqueció X.
"""
```

---

### Ticket S2-04 — Implementar los tres modelos supervisados requeridos

**Archivo a modificar**: `entrenamiento/comparador_modelos.py`

Reemplazar el `DummyClassifier` con implementación real. **Conservar la firma
pública** de `entrenar_clasificacion`, `entrenar_clustering` y `seleccionar_mejor`.

```python
"""
Nuevos modelos a registrar en el dict interno _CATALOGO_MODELOS:

'svm': SVC(
    kernel='rbf',
    probability=True,           # obligatorio para predict_proba
    class_weight='balanced',    # compensa desbalance 86/14
    random_state=SEMILLA_ALEATORIA,
)
  → Envuelto en GridSearchCV(
        param_grid={'clasificador__C': [0.1, 1, 10],
                    'clasificador__gamma': ['scale', 'auto']},
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        n_jobs=-1,
    )
  → ADVERTENCIA: SVM es lento en N>100k; usar class_weight en lugar de SMOTE
    para evitar multiplicar el tiempo de entrenamiento.

'arbol': DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=SEMILLA_ALEATORIA,
)

'gbm': GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=SEMILLA_ALEATORIA,
)

'mlp': MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEMILLA_ALEATORIA,
)

Cada modelo se envuelve en ConstructorPreprocesador().construir_pipeline(modelo)
ANTES de pasarlo a GridSearchCV o de entrenarlo directamente.

Modificar la firma de entrenar_clasificacion para aceptar:
  modelos_a_entrenar: list[str] | None = None
  Si None → entrenar los 4 modelos del catálogo.
  Si lista → entrenar solo los indicados.

El puntaje en ResultadoModelo debe ser ROC-AUC (no accuracy).
Usar cross_val_score(scoring='roc_auc') o el score del GridSearchCV.
"""
```

Actualizar `config.py` agregando:
```python
MODELOS_SUPERVISADOS: Final[tuple[str, ...]] = ('svm', 'arbol', 'gbm', 'mlp')
UMBRAL_MINIMO_AUC: Final[float] = 0.75   # para prueba de regresión de calidad
```

---

### Ticket S2-05 — Implementar EvaluadorClinico con métricas clínicas

**Archivo a crear**: `entrenamiento/evaluador.py`

```python
"""
Propósito:
Calcular y persistir métricas clínicas completas para un clasificador binario
de diabetes. Las métricas priorizan sensibilidad sobre precisión.

Clase: EvaluadorClinico

Dataclass interno: ResultadoEvaluacion
  nombre_modelo: str
  roc_auc: float
  pr_auc: float
  sensibilidad: float    # recall clase positiva
  especificidad: float
  f1_clase_positiva: float
  brier_score: float
  accuracy: float
  matriz_confusion: np.ndarray  # shape (2,2)

Método: calcular_metricas(
    y_verdadero: np.ndarray,
    y_prob: np.ndarray,          # probabilidades de clase 1
    nombre_modelo: str,
) -> ResultadoEvaluacion
  Calcula las 8 métricas. Especificidad = TN / (TN + FP).

Método: graficar_curvas(
    y_verdadero: np.ndarray,
    y_prob: np.ndarray,
    nombre_modelo: str,
    ruta_salida: Path | None = None,
) -> None
  Subplot 1x2: curva ROC (izq) + curva Precision-Recall (der).
  Línea de referencia en cada gráfica.
  Título: nombre_modelo + AUC en leyenda.
  Guardar en reportes/curvas_{nombre_modelo}.png
  Usar REPORTES_DIR de config.

Método: comparar_modelos(
    resultados: list[ResultadoEvaluacion],
) -> pd.DataFrame
  Tabla con una fila por modelo, columnas = todas las métricas numéricas.
  Ordenada por roc_auc descendente.
  Persiste en reportes/comparativa_modelos.md (formato markdown).
  Retorna el DataFrame para uso en notebooks.

Método: graficar_curva_calibracion(
    y_verdadero: np.ndarray,
    y_prob: np.ndarray,
    nombre_modelo: str,
) -> None
  Reliability diagram: probabilidades predichas vs. fracción de positivos.
  10 bins uniformes. Línea diagonal de calibración perfecta.
  Guardar en reportes/calibracion_{nombre_modelo}.png
"""
```

---

### Ticket S2-06 — Ampliar pipeline.py para serializar Pipeline completo

**Archivo a modificar**: `entrenamiento/pipeline.py`

```python
"""
Cambios requeridos en ejecutar_pipeline():

1. Para modo='clasificacion':
   a. Usar CargadorDatos.cargar_con_objetivo() → (X, y)
   b. Llamar CargadorDatos.detectar_desbalance(y) y loggear resultado
   c. train_test_split con stratify=y, test_size=PROPORCION_PRUEBA,
      random_state=SEMILLA_ALEATORIA
   d. Llamar ComparadorModelos.entrenar_clasificacion(X_train, y_train,
      modelos_a_entrenar=None)  # entrena los 4
   e. Para cada ResultadoModelo, evaluar con EvaluadorClinico sobre X_test, y_test
   f. Seleccionar mejor por roc_auc
   g. joblib.dump(mejor.modelo, ruta_modelo)   # Pipeline completo
   h. Persistir metricas completas como JSON en ruta_reporte
   i. Llamar EvaluadorClinico.graficar_curvas() para el mejor modelo
   j. Llamar EvaluadorClinico.comparar_modelos() para todos

2. Agregar argumento CLI --modelos (lista separada por comas)
   ej: --modelos gbm,mlp  → entrena solo esos dos

3. Guardar versión del modelo con timestamp:
   modelos/modelo_diabetes_v{YYYYMMDD_HHMMSS}.joblib
   Y también sobrescribir el alias fijo RUTA_MODELO_FINAL para que
   PredictorDiabetes lo encuentre sin cambios.

4. El JSON de métricas debe incluir:
   {
     "version": "...",
     "timestamp": "...",
     "mejor_modelo": "gbm",
     "modelos": {
       "gbm": {"roc_auc": 0.82, "sensibilidad": 0.73, ...},
       "svm": {...},
       ...
     }
   }
"""
```

---

### Ticket S2-07 — Prueba de no data leakage en Pipeline

**Archivo a crear**: `pruebas/test_preprocesador.py`

```python
"""
Tests requeridos:

test_pipeline_no_filtra_estadisticas_de_test():
  Crear X_train y X_test con distribuciones deliberadamente distintas en BMI.
  Ajustar pipeline solo sobre X_train.
  Verificar que la media usada para escalar en X_test es la de X_train,
  no la de X_test. (Inspeccionar pipeline.named_steps['preprocesador']
  .named_transformers_['continuas'].named_steps['scaler'].mean_)

test_pipeline_serializable_y_compatible_con_predictor():
  Crear pipeline mínimo con DummyClassifier.
  joblib.dump → tmp_path.
  PredictorDiabetes(ruta_modelo=tmp_path/modelo.joblib).cargar_modelo()
  predecir(entrada_valida_una_fila) → debe retornar dict con 'probabilidad'.

test_columnas_binarias_no_se_escalan():
  Verificar que las columnas binarias en el ColumnTransformer usan passthrough,
  no StandardScaler. (Las binarias 0/1 no deben cambiar de rango.)

test_ordinales_mantienen_orden():
  Para GenHlth=[1,2,3,4,5], verificar que OrdinalEncoder respeta el orden
  y que un valor mayor en la escala original produce un código mayor.
"""
```

---

### Ticket S2-08 — Prueba de distribución y desbalance en CargadorDatos

**Archivo a modificar**: `pruebas/test_cargador.py`

```python
"""
Tests adicionales a agregar (sin tocar los existentes):

test_detectar_desbalance_retorna_ratio_correcto():
  Crear y con 860 ceros y 140 unos.
  detectar_desbalance(y) debe retornar ratio ≈ 6.14.
  pct_clase_1 debe ser ≈ 0.14.

test_analizar_distribucion_incluye_pct_positivos_para_binarias():
  Crear dataframe con HighBP=[0,1,1,1,0].
  analizar_distribucion() debe retornar pct_positivos ≈ 0.6 para HighBP.

test_cargar_con_objetivo_retorna_X_e_y_separados():
  Usar dataset sintético con COLUMNAS_CDC + COLUMNA_OBJETIVO.
  cargar_con_objetivo() debe retornar tuple (DataFrame, Series).
  X.columns == COLUMNAS_CDC, y.name == COLUMNA_OBJETIVO.
"""
```

---

### Ticket S2-09 — Guardar dataset procesado en Parquet

Agregar método a `CargadorDatos`:
```python
def persistir_procesado(
    self,
    dataframe: pd.DataFrame,
    ruta_destino: Path | None = None,
) -> Path:
    """
    Propósito: guardar dataset limpio en formato Parquet comprimido.
    ruta_destino por defecto: DATOS_PROCESADOS_DIR / 'dataset_procesado.parquet'
    Crear directorio si no existe.
    Retornar ruta del archivo guardado.
    Loggear: ruta, shape, tamaño en MB.
    """
```

Agregar a `config.py`:
```python
RUTA_DATASET_PROCESADO: Final[Path] = DATOS_PROCESADOS_DIR / "dataset_procesado.parquet"
```

---

### Ticket S2-10 — Generar contraste_regional.md

**Archivo a crear**: `reportes/contraste_regional.md` (generado por script)
**Script a crear**: `reportes/generar_contraste_regional.py`

```python
"""
Propósito:
Generar el archivo reportes/contraste_regional.md con análisis distribucional
CDC vs. ENSANUT usando el dataset ya descargado.

Secciones del reporte generado:
  1. Tabla: variable | prevalencia_CDC_calculada | prevalencia_ENSANUT_referencia
     | diferencia_absoluta | sesgo_pct | clasificacion_sesgo
     clasificacion_sesgo: 'bajo' (<10%), 'medio' (10–30%), 'alto' (>30%)

  2. Narrativa automática:
     f"Variables con sesgo alto respecto a México: {lista_variables_sesgo_alto}"
     f"Variable más sesgada: {variable_max_sesgo} ({sesgo_pct:.1f}% de diferencia)"

  3. Implicaciones para el modelo (texto fijo + datos calculados).

Prevalencias ENSANUT de referencia (constantes en este script, no en config):
  ENSANUT_REFS = {
      'HighBP': 0.301, 'HighChol': 0.196, 'Smoker': 0.176,
      'PhysActivity': 0.605,  # complemento de 39.5% inactivos
      'Fruits': 0.420, 'Veggies': 0.340, 'HvyAlcoholConsump': 0.076,
  }
  # Solo para variables binarias con dato ENSANUT disponible
"""
```

---

## SPRINT 3 — Fenotipado Metabólico & Dashboard

> **Prerequisito**: Sprint 2 completado. `pytest -q` 100%. Existe
> `modelos/modelo_diabetes_v1.joblib` con ROC-AUC > 0.75.
> **Desbloquea**: Sprint 4.

### Ticket S3-01 — Implementar FenotipadorMetabolico

**Archivo a crear**: `entrenamiento/fenotipador.py`

```python
"""
Propósito:
Agrupar pacientes en fenotipos de riesgo metabólico usando K-Means
sobre variables seleccionadas, SIN usar la variable objetivo.
El fenotipo se usa como feature adicional en el pipeline supervisado.

Clase: FenotipadorMetabolico

Constante de clase:
  VARIABLES_METABOLICAS: Final[list[str]] = [
      'BMI', 'GenHlth', 'PhysHlth', 'MentHlth', 'Age',
      'PhysActivity', 'HighBP', 'HighChol', 'HeartDiseaseorAttack',
  ]
  # Subconjunto de COLUMNAS_CDC con mayor carga metabólica

Método: seleccionar_k(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    graficar: bool = True,
) -> int
  Para cada K en [k_min, k_max]:
    - Ajustar KMeans(n_clusters=K, random_state=SEMILLA_ALEATORIA, n_init='auto')
    - Calcular inercia y silhouette_score(X_scaled, labels)
  Si graficar=True: guardar reportes/codo_kmeans.png con subplot inercia|silhouette
  Retornar K con mayor silhouette_score (desempate: menor inercia)

Método: ajustar(X: pd.DataFrame, k: int) -> None
  Escalar con StandardScaler solo las VARIABLES_METABOLICAS.
  Ajustar KMeans. Persistir tanto el scaler como el kmeans en un Pipeline:
  joblib.dump(Pipeline([('scaler', scaler), ('kmeans', kmeans)]),
              MODELOS_DIR / 'fenotipador.joblib')
  Guardar self._k = k, self._pipeline_kmeans, self._es_ajustado = True

Método: asignar_fenotipos(X: pd.DataFrame) -> np.ndarray
  Precondición: self._es_ajustado is True
  Retornar self._pipeline_kmeans.predict(X[VARIABLES_METABOLICAS])

Método: perfilar_fenotipos(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame
  Agregar etiquetas al X. Para cada cluster calcular:
    - Media de cada VARIABLE_METABOLICA
    - Prevalencia de diabetes (y==1)
    - N (tamaño del cluster)
  Ordenar por prevalencia_diabetes descendente.
  Retornar DataFrame con índice=cluster_id.

Método: nombrar_fenotipos(
    perfil: pd.DataFrame,
) -> dict[int, str]
  Reglas heurísticas para nombrar clusters automáticamente:
    Si BMI > 31 Y HighBP > 0.6 Y PhysActivity < 0.4:
      → 'Metabólico Clásico'
    Si Age > 9 Y media_comorbilidades > 1.5:
      → 'Multimórbido Avanzado'
    En otro caso:
      → 'Bajo Riesgo General' (asignar al de menor prevalencia_diabetes)
  Retornar dict {cluster_id: nombre_clinico}

Método: enriquecer_X(
    X: pd.DataFrame,
) -> pd.DataFrame
  Agrega columna 'fenotipo' (int) al DataFrame.
  Retorna copia de X con la columna adicional.
  NOTA: esta columna debe incluirse en COLUMNAS_CONTINUAS o como ordinal
  en ConstructorPreprocesador.construir_pipeline_con_fenotipo().
"""
```

Agregar a `config.py`:
```python
RUTA_FENOTIPADOR: Final[Path] = MODELOS_DIR / "fenotipador.joblib"
K_CLUSTERS_DEFAULT: Final[int] = 3
```

---

### Ticket S3-02 — Integrar fenotipador en pipeline supervisado

**Archivo a modificar**: `entrenamiento/pipeline.py`

```python
"""
Agregar modo='fenotipado_clasificacion' a ejecutar_pipeline():

Flujo:
  1. cargar_con_objetivo() → X, y
  2. train_test_split estratificado
  3. FenotipadorMetabolico().seleccionar_k(X_train)
  4. FenotipadorMetabolico().ajustar(X_train, k=k_optimo)
  5. X_train_enriquecido = fenotipador.enriquecer_X(X_train)
  6. X_test_enriquecido  = fenotipador.enriquecer_X(X_test)
     CRÍTICO: fenotipador.ajustar() ya ocurrió sobre X_train.
     Para X_test solo se llama predict() del kmeans (no fit).
  7. ComparadorModelos.entrenar_clasificacion() sobre X_train_enriquecido
     con ConstructorPreprocesador().construir_pipeline_con_fenotipo(clasificador)
  8. Evaluar con EvaluadorClinico sobre X_test_enriquecido
  9. Comparar AUC con/sin fenotipo y loggear delta

Agregar argumento CLI: --usar-fenotipo (flag booleana, default=False)
"""
```

---

### Ticket S3-03 — Tabla de perfiles de fenotipos

**Archivo a crear**: `reportes/generar_perfiles_fenotipos.py`

```python
"""
Propósito:
Generar reportes/perfiles_fenotipos.md con la tabla de fenotipos metabólicos.

Requiere que modelos/fenotipador.joblib exista (generado por S3-01).

Secciones:
  1. Tabla: fenotipo | nombre_clinico | N | pct_total | BMI_media |
     HighBP_pct | PhysActivity_pct | Age_media | prevalencia_diabetes
  2. Interpretación clínica por fenotipo (generada con nombrar_fenotipos())
  3. Hipótesis confirmada/rechazada vs. tabla del ROADMAP_PARTE2.md
"""
```

---

### Ticket S3-04 — Dashboard Streamlit: Pantalla 1 (formulario de tamizaje)

**Archivo a crear**: `dashboard/__init__.py` (vacío)
**Archivo a crear**: `dashboard/app.py` — sección formulario

```python
"""
Propósito:
Interfaz de tamizaje PrevenIMSS Digital. El usuario es el médico general.

Estructura de app.py:
  import streamlit as st
  from dashboard.cliente_api import ClientePrediccion

  st.set_page_config(
      page_title="PrevenIMSS Digital",
      page_icon="🏥",
      layout="wide",
  )

  ETIQUETAS_EDAD_CDC = {
      1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
      6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
      11: "70–74", 12: "75–79", 13: "80+",
  }
  ETIQUETAS_SALUD_GENERAL = {1:"Excelente",2:"Muy buena",3:"Buena",4:"Regular",5:"Mala"}
  ETIQUETAS_EDUCACION = {
      1:"Sin escolaridad",2:"Primaria incompleta",3:"Primaria",
      4:"Secundaria",5:"Preparatoria",6:"Universidad o más",
  }

  Pantalla principal:
    st.title("PrevenIMSS Digital — Tamizaje de Riesgo de Diabetes Tipo 2")
    col_clinica, col_habitos = st.columns(2)

    col_clinica:
      imc = st.slider("IMC", min_value=10.0, max_value=50.0, step=0.5)
      presion_alta = st.checkbox("HTA diagnosticada")
      colesterol_alto = st.checkbox("Dislipidemia diagnosticada")
      salud_general = st.selectbox("Salud general", ETIQUETAS_SALUD_GENERAL)
      salud_fisica = st.slider("Días con mala salud física (último mes)", 0, 30)
      dificultad_caminar = st.checkbox("Dificultad para caminar / subir escaleras")

    col_habitos:
      actividad_fisica = st.checkbox("Actividad física regular en últimos 30 días")
      consume_fruta = st.checkbox("Consume fruta ≥1 vez al día")
      consume_verdura = st.checkbox("Consume verdura ≥1 vez al día")
      fumador = st.checkbox("Ha fumado ≥100 cigarrillos en su vida")
      enfermedad_corazon = st.checkbox("Cardiopatía o infarto previo")
      derrame_cerebral = st.checkbox("Derrame cerebral previo")

    Sección inferior (st.expander "Perfil sociodemográfico"):
      sexo = st.radio("Sexo", ["Femenino", "Masculino"])
      edad = st.selectbox("Grupo de edad", ETIQUETAS_EDAD_CDC)
      educacion = st.selectbox("Escolaridad", ETIQUETAS_EDUCACION)
      tiene_cobertura_medica = st.checkbox("Cuenta con cobertura médica")
      chequeo_colesterol = st.checkbox("Prueba de colesterol en últimos 5 años")
      sin_medico_por_costo = st.checkbox("Ha dejado de ir al médico por costo")
      consumo_alcohol_alto = st.checkbox("Consumo alto de alcohol")
      salud_mental = st.slider("Días con mala salud mental (último mes)", 0, 30)
      ingreso = st.selectbox("Nivel de ingreso", list(range(1,9)))

    boton_calcular = st.button("🔍 Calcular Riesgo", type="primary")

  Al presionar boton_calcular:
    → construir payload dict con nombres de campo de DatosPaciente
    → mapear sexo radio → 0/1, edad ETIQUETA → código CDC, etc.
    → ClientePrediccion().predecir(payload) → guardar en st.session_state['resultado']
    → st.rerun() para mostrar Pantalla 2
"""
```

Agregar a `pyproject.toml` en `[project.optional-dependencies]`:
```toml
dashboard = ["streamlit>=1.35.0", "requests>=2.32.0"]
```

---

### Ticket S3-05 — Dashboard Streamlit: Pantalla 2 (resultado + SHAP)

**Archivo a modificar**: `dashboard/app.py` — agregar sección resultado

```python
"""
Se muestra cuando st.session_state.get('resultado') no es None.

Panel principal:
  probabilidad = resultado['confianza']
  categoria = resultado['categoria_riesgo']

  Gauge con plotly.graph_objects.Indicator:
    mode="gauge+number+delta"
    value=probabilidad*100
    gauge={
      'axis': {'range': [0, 100]},
      'bar': {'color': color_segun_categoria},
      'steps': [
        {'range': [0, 33], 'color': '#d4edda'},   # verde claro
        {'range': [33, 66], 'color': '#fff3cd'},  # amarillo claro
        {'range': [66, 100], 'color': '#f8d7da'}, # rojo claro
      ],
      'threshold': {'line': {'color': "red", 'width': 4}, 'value': probabilidad*100}
    }

  Etiqueta grande: st.metric("Categoría de Riesgo", categoria.upper())

  Si resultado['advertencia']:
    st.warning("⚠️ " + resultado['advertencia'])

Panel secundario st.expander("¿Por qué este resultado? (Factores de riesgo)"):
  → Mostrar gráfica SHAP waterfall si está disponible en st.session_state
  → Si no: mostrar mensaje "El análisis de factores requiere el modelo cargado localmente"

Recomendaciones automáticas (basadas en campos del payload):
  if categoria == 'alto' and not actividad_fisica:
    st.info("💪 Recomendación: Actividad física ≥150 min/semana (Guía IMSS)")
  if categoria in ('alto','medio') and presion_alta:
    st.info("🩺 Control de presión arterial en próxima consulta")
  if resultado['advertencia']:
    st.info("🔬 Solicitar glucosa en ayuno y HbA1c para confirmación diagnóstica")

Botones:
  col1, col2 = st.columns(2)
  col1: st.button("🔄 Nueva consulta") → limpiar session_state, st.rerun()
  col2: st.button("📋 Guardar resultado") → exportar dict a JSON descargable

Agregar a pyproject.toml: "plotly>=5.22.0" en dashboard extras.
"""
```

---

### Ticket S3-06 — Dashboard Streamlit: Panel estadístico del médico

**Archivo a modificar**: `dashboard/app.py` — agregar sidebar stats

```python
"""
En el sidebar:
  st.sidebar.title("📊 Panel del turno")

  Si st.session_state.get('historial_turno') tiene entradas:
    n_pacientes = len(historial)
    st.sidebar.metric("Pacientes evaluados", n_pacientes)

    conteo por categoria → donut chart con plotly.express.pie
    st.sidebar.plotly_chart(fig_donut, use_container_width=True)

    st.sidebar.dataframe(
      pd.DataFrame(historial)[['id_anonimo','categoria_riesgo','confianza']],
      use_container_width=True,
    )

    csv_bytes = pd.DataFrame(historial).to_csv(index=False).encode()
    st.sidebar.download_button("⬇️ Exportar turno CSV", csv_bytes, "turno.csv")

Lógica de historial:
  Cada predicción exitosa agrega al historial:
    {'id_anonimo': uuid4().hex[:8], 'categoria_riesgo': ..., 'confianza': ...,
     'timestamp': datetime.now().isoformat()}
  El historial se guarda en st.session_state['historial_turno'] (lista de dicts).
  NO persistir datos en disco (cumple restricción PII).
"""
```

---

### Ticket S3-07 — Calcular SHAP values y generar gráficas

**Archivo a crear**: `reportes/generar_shap.py`

```python
"""
Propósito:
Generar las 3 gráficas SHAP para el paper académico y el dashboard.
Requiere que modelos/modelo_diabetes_v1.joblib exista.

Dependencia: pip install shap>=0.45  (agregar a pyproject.toml)

Función: generar_summary_plot(
    pipeline: sklearn.Pipeline,
    X_test: pd.DataFrame,
    ruta_salida: Path | None = None,
) -> None
  Para GradientBoostingClassifier o árbol:
    explainer = shap.TreeExplainer(pipeline.named_steps['clasificador'])
    X_test_transformado = pipeline.named_steps['preprocesador'].transform(X_test)
    shap_values = explainer.shap_values(X_test_transformado)
  Para SVM / MLP:
    background = shap.sample(X_test_transformado, 100)
    explainer = shap.KernelExplainer(
        lambda x: pipeline.named_steps['clasificador'].predict_proba(x)[:, 1],
        background,
    )
    shap_values = explainer.shap_values(X_test_transformado[:500])
  shap.summary_plot(shap_values, X_test, feature_names=COLUMNAS_CDC, show=False)
  plt.savefig(ruta_salida or REPORTES_DIR/'shap_summary.png', bbox_inches='tight')
  plt.close()

Función: calcular_shap_paciente(
    pipeline: sklearn.Pipeline,
    X_paciente: pd.DataFrame,   # una fila, columnas CDC
    explainer: shap.Explainer,
) -> tuple[np.ndarray, float]
  Retorna (shap_values_fila, valor_esperado) para usar en waterfall del dashboard.

Función: generar_dependence_plot(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    variable_principal: str = 'BMI',
    variable_interaccion: str = 'Age',
    ruta_salida: Path | None = None,
) -> None
  shap.dependence_plot(variable_principal, shap_values, X_test,
                       interaction_index=variable_interaccion, show=False)
  plt.savefig(ruta_salida or REPORTES_DIR/'shap_dependence_bmi_age.png',
              bbox_inches='tight')
  plt.close()

Script CLI al final del archivo:
  if __name__ == '__main__':
    pipeline = joblib.load(RUTA_MODELO_FINAL)
    X_test = pd.read_parquet(RUTA_DATASET_PROCESADO)[list(COLUMNAS_CDC)].sample(2000)
    generar_summary_plot(pipeline, X_test)
    generar_dependence_plot(...)
    print("Gráficas SHAP generadas en reportes/")
"""
```

Agregar a `pyproject.toml`:
```toml
[project.optional-dependencies]
shap = ["shap>=0.45.0", "matplotlib>=3.9.0"]
```

---

### Ticket S3-08 — ClientePrediccion con manejo de errores

**Archivo a crear**: `dashboard/cliente_api.py`

```python
"""
Propósito:
Wrapper HTTP para consumir POST /predecir desde el dashboard Streamlit.
Desacopla el dashboard de los detalles HTTP.

Clase: ClientePrediccion

Atributos de clase:
  RUTA_API: str = os.getenv('RUTA_API', 'http://localhost:8000')
  TIMEOUT_SEGUNDOS: int = 5

Método: predecir(self, payload: dict) -> dict
  POST {RUTA_API}/predecir con json=payload, timeout=TIMEOUT_SEGUNDOS
  Si status_code == 200: retornar response.json()
  Si status_code == 503:
    retornar {'error': 'Servicio no disponible. El modelo no está cargado.',
              'categoria_riesgo': None}
  Si status_code == 422:
    retornar {'error': f"Datos inválidos: {response.json()['detail']}",
              'categoria_riesgo': None}
  Si requests.exceptions.ConnectionError:
    retornar {'error': 'No se pudo conectar con el servidor de predicción.',
              'categoria_riesgo': None}
  Si requests.exceptions.Timeout:
    retornar {'error': f'El servidor tardó más de {TIMEOUT_SEGUNDOS}s.',
              'categoria_riesgo': None}

Método: verificar_salud(self) -> bool
  GET {RUTA_API}/salud → True si estado=='operativo', False en cualquier error
"""
```

---

### Ticket S3-09 — Configuración Streamlit con tema IMSS

**Archivo a crear**: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#006847"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

---

### Ticket S3-10 — Prueba de integración fenotipador

**Archivo a crear**: `pruebas/test_fenotipador.py`

```python
"""
Tests requeridos:

test_seleccionar_k_retorna_entero_en_rango():
  Crear X sintético con distribución gaussiana, 500 filas.
  seleccionar_k(X, k_min=2, k_max=5, graficar=False) debe retornar int en [2,5].

test_ajustar_crea_archivo_joblib(tmp_path):
  Monkeypatch MODELOS_DIR a tmp_path.
  ajustar(X, k=3) → verificar que tmp_path/'fenotipador.joblib' existe.

test_asignar_fenotipos_retorna_shape_correcto():
  Ajustar con X_train (100 filas). asignar_fenotipos(X_test, 50 filas).
  Resultado shape debe ser (50,) con valores en {0,1,2}.

test_enriquecer_X_agrega_columna_fenotipo():
  enriquecer_X(X) debe retornar DataFrame con columna 'fenotipo' adicional.
  Columnas originales deben estar intactas.

test_perfilar_fenotipos_incluye_prevalencia_diabetes():
  Crear X e y sintéticos. perfilar_fenotipos(X, y) debe retornar DataFrame
  con columna 'prevalencia_diabetes' entre 0 y 1.
"""
```

---

## SPRINT 4 — Reporte Académico & Despliegue Final

> **Prerequisito**: Sprint 3 completado. Dashboard operativo.
> `pytest -q` 100%. ROC-AUC mejor modelo > 0.78.

### Ticket S4-01 — Implementar EvaluadorEquidad

**Archivo a crear**: `entrenamiento/evaluador_equidad.py`

```python
"""
Propósito:
Calcular métricas de clasificación separadas por subgrupos demográficos
para detectar disparidades que deben reportarse en el paper académico.

Clase: EvaluadorEquidad

Constante de clase:
  SUBGRUPOS: Final[dict[str, dict]] = {
      'sexo': {'columna': 'Sex', 'grupos': {0: 'Femenino', 1: 'Masculino'}},
      'edad': {
          'columna': 'Age',
          'grupos': {
              'joven': [1,2,3,4],       # 18–39
              'adulto_medio': [5,6,7,8,9],  # 40–64
              'mayor': [10,11,12,13],   # 65+
          }
      },
      'ingreso': {
          'columna': 'Income',
          'grupos': {'bajo': [1,2], 'medio': [3,4,5,6], 'alto': [7,8]}
      },
      'educacion': {
          'columna': 'Education',
          'grupos': {'sin_basica': [1,2], 'basica': [3,4], 'superior': [5,6]}
      },
  }

Método: calcular_metricas_por_subgrupo(
    X_test: pd.DataFrame,
    y_verdadero: np.ndarray,
    y_prob: np.ndarray,
    dimension: str,     # clave de SUBGRUPOS
) -> pd.DataFrame
  Para cada grupo en SUBGRUPOS[dimension]:
    filtrar índices del grupo en X_test
    calcular roc_auc, sensibilidad, especificidad, tasa_falsos_negativos
  Retornar DataFrame con una fila por grupo.

Método: calcular_paridad_demografica(
    y_pred: np.ndarray,     # predicciones binarias
    grupos: np.ndarray,     # etiquetas de grupo
) -> dict[str, float]
  Retornar {
    'max_diferencia_tasa_positiva': float,  # max - min de tasa predicción positiva
    'cumple_paridad': bool,                 # diferencia < 0.10
    'grupo_mayor_tasa': str,
    'grupo_menor_tasa': str,
  }

Método: generar_reporte_equidad(
    X_test: pd.DataFrame,
    y_verdadero: np.ndarray,
    y_prob: np.ndarray,
) -> str
  Ejecutar calcular_metricas_por_subgrupo para las 4 dimensiones.
  Retornar string Markdown con tablas + narrativa de hallazgos.
  Persistir en reportes/reporte_equidad.md
"""
```

---

### Ticket S4-02 — Dockerfile multi-stage

**Archivo a crear**: `Dockerfile`

```dockerfile
# Etapa 1: builder — instala dependencias
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e ".[dashboard,shap]"

# Etapa 2: runtime — solo lo necesario para la API
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
ENV APP_ENV=production
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Etapa 3: dashboard (target separado)
FROM runtime AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Ticket S4-03 — docker-compose.yml

**Archivo a crear**: `docker-compose.yml`

```yaml
version: "3.9"
services:
  api:
    build:
      context: .
      target: runtime
    ports:
      - "8000:8000"
    volumes:
      - ./modelos:/app/modelos:ro
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/salud"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      target: dashboard
    ports:
      - "8501:8501"
    environment:
      - RUTA_API=http://api:8000
    depends_on:
      api:
        condition: service_healthy
```

---

### Ticket S4-04 — GitHub Actions CI

**Archivo a crear**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-y-pruebas:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Instalar dependencias
        run: pip install -e .[dev]
      - name: Lint con ruff
        run: ruff check .
      - name: Ejecutar pruebas
        run: pytest -q --tb=short
      - name: Auditoría de seguridad
        run: pip install pip-audit && pip-audit

  build-imagen:
    runs-on: ubuntu-latest
    needs: lint-y-pruebas
    steps:
      - uses: actions/checkout@v4
      - name: Build imagen API
        run: docker build --target runtime -t diasgnostico-pred:api .
      - name: Build imagen Dashboard
        run: docker build --target dashboard -t diasgnostico-pred:dashboard .
```

Agregar a `pyproject.toml`:
```toml
[project.optional-dependencies]
lint = ["ruff>=0.4.0"]
```

---

### Ticket S4-05 — Estructura del paper final

**Archivo a crear**: `reportes/paper_final.md`

```markdown
# Sistema Predictivo de Diabetes Tipo 2 con Contextualización para Población Mexicana
## Basado en CDC BRFSS 2015 y Marco de Referencia ENSANUT 2022

**Autores:** [nombres]
**Institución:** [institución]
**Fecha:** [fecha]

---

## Resumen

<!-- Completar con valores reales post-entrenamiento -->
<!-- Estructura: Problema | Método | Resultado principal | Contribución -->
<!-- Máximo 250 palabras -->

---

## 1. Introducción

<!-- Estadísticas ENSANUT 2022 + brecha diagnóstica + objetivo del sistema -->

---

## 2. Datos y Preprocesamiento

<!-- Reproducir tabla de mapeo CDC→IMSS/ENSANUT de ROADMAP_PARTE1.md §1.2 -->
<!-- Incluir tabla de prevalencias comparadas §1.3 -->
<!-- Describir pipeline sklearn sin data leakage -->

---

## 3. Metodología

### 3.1 Fenotipado No Supervisado (K-Means)
<!-- K seleccionado | variables utilizadas | tabla perfiles fenotipos -->
<!-- Imagen: reportes/codo_kmeans.png -->

### 3.2 Modelos Supervisados
<!-- SVM, Árbol/GBM, MLP con justificación clínica de cada uno -->

### 3.3 Protocolo de Validación
<!-- StratifiedKFold(5) | métricas clínicas | umbral decisión -->

### 3.4 Arquitectura del Sistema
<!-- Diagrama ASCII de capas del PROYECTO.md §2.1 -->

---

## 4. Resultados

<!-- Tabla: reproducir EvaluadorClinico.comparar_modelos() -->
<!-- Imágenes: curvas ROC+PR para los 3 modelos -->
<!-- Tabla: perfiles fenotipos con prevalencia diabetes -->
<!-- Imagen: reportes/shap_summary.png con interpretación -->
<!-- Tabla: EvaluadorEquidad por subgrupo -->

---

## 5. Discusión

### 5.1 Comparativa con Literatura
<!-- Tabla de estudios del ROADMAP_PARTE2.md §4.2 + nuestro resultado -->

### 5.2 Sesgo Poblacional CDC→México
<!-- Limitaciones 1-3 del ROADMAP_PARTE2.md §5.2 -->
<!-- Redactadas como contribuciones metodológicas -->

### 5.3 Implicaciones Clínicas para el IMSS
<!-- Umbral 0.25 para contexto mexicano + justificación -->

---

## 6. Conclusiones

<!-- Modelo recomendado + AUC obtenido -->
<!-- Viabilidad IMSS + trabajo futuro ENSANUT -->

---

## 7. Referencias

<!-- Mínimo: ENSANUT 2022, CDC BRFSS 2015, scikit-learn, FastAPI, SHAP -->
<!-- + ≥3 papers sobre predicción T2DM en población hispana -->
```

---

## CHECKLIST DE COMPLETITUD (Verificación final)

Antes de etiquetar `v1.0.0`, verificar que **todos** los siguientes comandos
retornan sin error:

```bash
# 1. Todas las pruebas pasan
pytest -q --tb=short

# 2. Linting limpio
ruff check .

# 3. Auditoría de dependencias
pip-audit

# 4. Pipeline de entrenamiento completo
python -m entrenamiento.pipeline --modo clasificacion

# 5. Pipeline con fenotipado
python -m entrenamiento.pipeline --modo fenotipado_clasificacion --usar-fenotipo

# 6. API operativa (en otra terminal)
uvicorn api.main:app --reload
curl http://localhost:8000/salud | python -m json.tool
# → debe mostrar: "estado": "operativo"

# 7. Dashboard operativo
streamlit run dashboard/app.py

# 8. Gráficas SHAP generadas
python reportes/generar_shap.py

# 9. Reporte de equidad generado
python -c "
from entrenamiento.evaluador_equidad import EvaluadorEquidad
print('EvaluadorEquidad importable OK')
"

# 10. Docker build (si Docker disponible)
docker compose build
docker compose up -d
curl http://localhost:8000/salud

# 11. ROC-AUC del mejor modelo supera umbral mínimo
python -c "
import json
from config import RUTA_REPORTE_METRICAS
m = json.loads(RUTA_REPORTE_METRICAS.read_text())
mejor_auc = max(v['roc_auc'] for v in m['modelos'].values())
assert mejor_auc > 0.75, f'AUC {mejor_auc} < 0.75'
print(f'ROC-AUC del mejor modelo: {mejor_auc:.4f} ✅')
"
```

---

## NOTAS DE IMPLEMENTACIÓN PARA COPILOT

- **Siempre importa desde `config`**: `SEMILLA_ALEATORIA`, `COLUMNAS_CDC`,
  `RUTA_MODELO_FINAL`, `REPORTES_DIR`, `MODELOS_DIR`, etc.
- **Nunca modifiques** `api/esquemas.py`, `api/main.py`, `inferencia/predictor.py`
  salvo que el ticket lo indique explícitamente.
- **Cada archivo nuevo** comienza con el bloque `from __future__ import annotations`
  y un docstring de Propósito/Firma/Lógica/Error siguiendo el patrón del Sprint 1.
- **Nombres en español**: `calcular_metricas`, `ajustar_fenotipador`,
  `generar_reporte`, no `calculate_metrics`, `fit`, `generate_report`.
- **Columnas CDC en inglés**: son identificadores oficiales del dataset, no se traducen.
- Al agregar dependencias a `pyproject.toml`, ejecuta
  `pip install -e .[dev,dashboard,shap,lint]` para actualizar el entorno.
