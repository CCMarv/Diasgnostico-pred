# Notas: SVM para Detección de Fraude — Visualización 3D del Hiperplano

**Unidad 2.1 — Support Vector Machines (SVM)**  
Tema: Intuición geométrica, maximización del margen e hiperplano de decisión | Dataset: `fraude.csv`

---

## ¿Qué problema resuelve este notebook?

Se entrena una **Máquina de Soporte Vectorial (SVM)** con kernel lineal para detectar transacciones fraudulentas, y se genera una **visualización 3D del hiperplano de separación** para desarrollar la intuición geométrica del modelo.

---

## Intuición geométrica del SVM

Un SVM busca el **hiperplano** que separa las dos clases con el **máximo margen posible**. En 2D ese hiperplano es una línea recta; en 3D es un plano.

```
Clase Legítimo (0)  ●  ●  ●
                            ← margen →
                         ──────────────  ← hiperplano de decisión
                            ← margen →
Clase Fraude (1)              ● ● ●
```

Los puntos que quedan justo en el borde del margen son los **vectores de soporte**: son los únicos que determinan la posición del hiperplano.

---

## Flujo del notebook

### 1 — Preparación de datos

```python
df = pd.read_csv('fraude.csv')
X = df[['monto', 'antiguedad_cuenta']].values
y = df['es_fraude'].values
```

El modelo usa solo **dos características** para poder visualizar el hiperplano en 3D.

### 2 — Escalado (obligatorio en SVM)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> Los SVM son sensibles a la escala. Si no se escala, la variable con mayor rango dominará el cálculo del margen. Ver notebook `escalar_o_no_escalar` para la demostración completa.

### 3 — Entrenamiento del SVM lineal

```python
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_scaled, y)
```

**Parámetro `C` (regularización):**

| Valor de C | Efecto |
|---|---|
| **C alto** | Margen más estrecho, menos errores en train (riesgo de overfitting) |
| **C bajo** | Margen más amplio, tolera más errores (mejor generalización) |

### 4 — Construcción de la visualización 3D

La ecuación del hiperplano lineal es:

```
w₀·x + w₁·y + b = 0
```

Se despeja `z` para graficar el plano en el espacio 3D:

```python
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
zz = (-w[0] * xx - w[1] * yy - b)
```

**Elementos del gráfico:**

| Elemento | Representación |
|---|---|
| Plano gris translúcido | Hiperplano de decisión |
| Puntos azules | Transacciones legítimas (clase 0) |
| Puntos rojos | Fraudes (clase 1) |
| Círculos negros vacíos | Vectores de soporte |

```python
# Hiperplano
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# Puntos legítimos
ax.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], 0,
           color='blue', label='Legítimo (0)')

# Puntos fraude
ax.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], 0,
           color='red', label='Fraude (1)')

# Vectores de soporte
ax.scatter(svm_model.support_vectors_[:, 0],
           svm_model.support_vectors_[:, 1], 0,
           facecolors='none', edgecolors='black', label='Vectores de Soporte')
```

El ángulo de la cámara se controla con `ax.view_init(elev=20, azim=45)`.

---

## Atributos útiles del modelo entrenado

| Atributo | Contenido |
|---|---|
| `svm_model.coef_` | Vector de pesos `w` del hiperplano (solo en kernel lineal) |
| `svm_model.intercept_` | Sesgo `b` del hiperplano |
| `svm_model.support_vectors_` | Coordenadas de los vectores de soporte |
| `svm_model.n_support_` | Número de vectores de soporte por clase |

---

## Kernels en SVM (contexto del programa)

Este notebook usa `kernel='linear'`, pero el programa también cubre:

| Kernel | Cuándo usarlo |
|---|---|
| `linear` | Datos linealmente separables; rápido e interpretable |
| `rbf` (Radial Basis Function) | Datos con fronteras no lineales; el más usado en práctica |
| `poly` | Relaciones polinomiales entre características |

> El kernel RBF proyecta los datos a un espacio de mayor dimensión donde sí son separables linealmente — el llamado **truco del kernel** del tema 2.1.

---

## Conexión con el programa

Este notebook cubre el tema **2.1 Support Vector Machines (SVM)**:

- Intuición geométrica: maximización del margen e hiperplano de decisión
- El rol de los vectores de soporte
- El parámetro de regularización `C`
- Uso obligatorio de escalado previo (conexión con tema 1.2)
- Base para entender el truco del kernel (RBF/Polinomial) visto en clase
