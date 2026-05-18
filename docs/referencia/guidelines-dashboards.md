# Guidelines de Diseño de Dashboards

> **Propósito.** Este documento reúne los principios teóricos y las decisiones prácticas necesarias para diseñar dashboards que sean a la vez **precisos**, **legibles** y **útiles**. Cubre desde la pregunta inicial ("¿quién va a leer esto y para qué?") hasta detalles concretos como qué paleta de color usar, qué gráfico elegir, cómo posicionar los elementos en la rejilla y cómo evaluar el resultado.
>
> Cada sección parte de un principio teórico y termina con reglas accionables. Los principios provienen de Cleveland & McGill (1984), Munzner, Wilke, Tufte, Few, Knaflic y la literatura contemporánea sobre percepción y accesibilidad.

---

## Tabla de contenidos

1. [Antes de diseñar: preguntas obligatorias](#1-antes-de-diseñar-preguntas-obligatorias)
2. [Tipología de dashboards](#2-tipología-de-dashboards)
3. [Arquitectura de información y layout](#3-arquitectura-de-información-y-layout)
4. [Jerarquía visual](#4-jerarquía-visual)
5. [Selección de gráficos: árbol de decisión](#5-selección-de-gráficos-árbol-de-decisión)
6. [Color: paletas y reglas de uso](#6-color-paletas-y-reglas-de-uso)
7. [Tipografía y texto](#7-tipografía-y-texto)
8. [Posicionamiento, espaciado y rejilla](#8-posicionamiento-espaciado-y-rejilla)
9. [Interactividad](#9-interactividad)
10. [Accesibilidad](#10-accesibilidad)
11. [Incertidumbre en dashboards](#11-incertidumbre-en-dashboards)
12. [Dashboards para modelos de ML/IA](#12-dashboards-para-modelos-de-mlia)
13. [Antipatrones: qué evitar](#13-antipatrones-qué-evitar)
14. [Implementación: stack técnico](#14-implementación-stack-técnico)
15. [Lista de verificación final](#15-lista-de-verificación-final)
16. [Referencias](#16-referencias)

---

## 1. Antes de diseñar: preguntas obligatorias

> **Principio (Munzner, cap. 3).** Toda visualización responde a una *tarea* dentro de un *contexto*. Diseñar sin estas dos coordenadas produce dashboards bonitos que nadie usa.

Antes de abrir cualquier herramienta, responde por escrito:

1. **¿Quién es la audiencia?** Operador de planta, analista, director ejecutivo, público general. El nivel de detalle, el vocabulario y el grado de explicación cambian radicalmente.
2. **¿Qué decisión o comprensión debe producir?** Si el dashboard no cambia una decisión, no debería existir. "Solo para ver datos" es señal de que falta especificar el caso de uso.
3. **¿Con qué frecuencia se consulta?** Tiempo real, diario, semanal, ad-hoc. Esto define la densidad de información y el nivel de actualización.
4. **¿Cuánto tiempo tiene el lector?** Un director que mira el dashboard 30 segundos antes de una reunión necesita algo radicalmente distinto al analista que pasa 2 horas explorándolo.
5. **¿Cuál es la métrica norte?** El dashboard debe tener una respuesta clara a "¿cómo vamos?". Si necesitas más de 10 segundos para responderla, falta jerarquía.

> **Regla de los 10 segundos.** Un dashboard bien diseñado comunica su mensaje principal en menos de 10 segundos. Si no, hay que rediseñar la jerarquía, no agregar más gráficos.

---

## 2. Tipología de dashboards

| Tipo | Audiencia | Frecuencia | Diseño |
|---|---|---|---|
| **Operativo** | Operadores, soporte, planta | Tiempo real / minutos | Pocos KPIs, alertas claras, lectura rápida, semáforos |
| **Analítico** | Analistas, científicos de datos | Diario / sesiones largas | Densidad alta, filtros, exploración, drill-down |
| **Estratégico** | Dirección, ejecutivos | Semanal / mensual | Tendencia + contexto, pocos KPIs, narrativa clara |
| **Tactical / equipo** | Gerentes, líderes de área | Diario / semanal | KPIs del área, comparaciones contra meta |

**Consecuencia de diseño:** un dashboard operativo y uno estratégico no se diseñan igual aunque muestren los mismos datos. El operativo prioriza *desvíos*; el estratégico prioriza *tendencia*.

> **Antipatrón común.** Mezclar tipos en un solo dashboard. El "dashboard único que sirve para todo" termina sin servirle bien a nadie. Es preferible una colección de dashboards específicos enlazados entre sí.

---

## 3. Arquitectura de información y layout

### 3.1 El patrón F y el patrón Z

Estudios de eye-tracking muestran que en lectores occidentales la mirada recorre las pantallas con dos patrones predominantes:

- **Patrón F**: para contenido denso. La mirada barre la parte superior, luego baja por la izquierda. Lo crítico va arriba a la izquierda.
- **Patrón Z**: para contenido escaso. Cruza de arriba-izquierda a arriba-derecha, luego diagonal a abajo-izquierda y horizontal a abajo-derecha.

**Regla práctica.** El KPI más importante va en la esquina superior izquierda. La esquina inferior derecha es buena para CTAs o el resumen de acción.

### 3.2 Estructura recomendada en tres bandas

```
┌─────────────────────────────────────────────────────┐
│  Banda 1: Título + filtros globales + fecha actual  │
├─────────────────────────────────────────────────────┤
│  Banda 2: KPIs principales (3–5 tarjetas)           │
│  [Ventas]  [Conversión]  [Churn]  [NPS]            │
├──────────────────────────┬──────────────────────────┤
│  Banda 3: Visualización  │  Banda 3: Desglose o    │
│  principal (tendencia,   │  segmentación            │
│  serie de tiempo)        │                          │
├──────────────────────────┴──────────────────────────┤
│  Banda 4: Detalle / tabla / dimensiones secundarias │
└─────────────────────────────────────────────────────┘
```

- **Banda 1 (5–10% del alto):** identidad del dashboard, controles globales, fecha y último update.
- **Banda 2 (15–20%):** KPIs clave. Números grandes, comparación contra periodo anterior o meta.
- **Banda 3 (40–50%):** gráficos principales que explican el porqué de los KPIs.
- **Banda 4 (20–30%):** detalle granular, tablas, segmentaciones que invitan a drill-down.

### 3.3 Reglas de proximidad y agrupación

> **Principio Gestalt.** Lo que está cerca se percibe como relacionado. Lo que está separado se percibe como independiente.

- Elementos relacionados temáticamente deben estar **agrupados visualmente** (mismo bloque, mismo fondo sutil, misma separación).
- Usa **espacio en blanco** entre grupos. Es la herramienta de agrupación más barata y efectiva.
- Si dos gráficos no se leen juntos, no deben estar pegados.

### 3.4 Densidad de información

- **Mínimo:** 3 elementos. Menos suele ser un *report*, no un dashboard.
- **Cómodo:** 6–9 elementos en pantalla sin scroll.
- **Máximo en pantalla completa:** ~12 elementos. Más allá, fragmentar en pestañas o dashboards enlazados.

> Tufte, *data-ink ratio*: maximiza la proporción de tinta dedicada a datos sobre tinta dedicada a decoración. Cada elemento debe ganarse su lugar.

---

## 4. Jerarquía visual

> **Principio.** El ojo del lector debe seguir un orden de importancia diseñado, no descubierto. La jerarquía se construye con cinco herramientas:

| Herramienta | Cómo se usa | Ejemplo |
|---|---|---|
| **Tamaño** | Lo grande atrae primero | KPI principal en 48px, secundarios en 24px |
| **Contraste** | Alta saturación sobre fondo neutro | Métrica clave en color de marca, resto en grises |
| **Posición** | Arriba-izquierda es el primer punto leído | Lo más importante en esa zona |
| **Color** | Color reservado para lo focal | Resto en escala de grises |
| **Espacio** | El espacio en blanco aísla y dignifica | Aire alrededor del KPI principal |

### 4.1 Tres niveles de jerarquía

Define explícitamente tres niveles:

- **Nivel 1 — Foco:** un único elemento por pantalla. KPI norte o gráfico principal.
- **Nivel 2 — Soporte:** 3–6 elementos que explican el nivel 1.
- **Nivel 3 — Contexto:** detalle, filtros, metadatos, fecha de actualización.

**Regla:** si todo grita, nada se escucha. Como mucho dos elementos en nivel 1 (idealmente uno).

### 4.2 KPIs: anatomía de una tarjeta

Una tarjeta de KPI bien diseñada contiene:

```
┌────────────────────────────────┐
│ Ventas del mes                 │ ← Etiqueta clara
│                                 │
│  $ 1.24 M                       │ ← Valor grande, foco
│                                 │
│  ▲ 12.3% vs mes anterior        │ ← Comparación (color: verde/rojo)
│  ─── Meta: $ 1.50 M ───         │ ← Contexto (meta, benchmark)
└────────────────────────────────┘
```

**No omitir:**
- **Comparación** (vs periodo anterior, meta, benchmark). Un número aislado no es informativo.
- **Unidades** explícitas ($, %, h, etc.).
- **Periodo** al que corresponde el número.

**Evita:**
- KPI sin contexto temporal.
- Flechas verdes/rojas sin acompañar de cifra (problema de accesibilidad).
- Decimales innecesarios. `$1.24M` se lee mejor que `$1,243,728.45`.

---

## 5. Selección de gráficos: árbol de decisión

> **Base teórica (Cleveland–McGill, 1984).** La precisión con que decodificamos visualmente varía por canal. De más a menos preciso:
> **Posición en escala común > Longitud > Ángulo/Pendiente > Área > Color/Saturación**.
>
> **Implicación.** Para comparación precisa, usa posición o longitud. Reserva área y color para categorías o refuerzo.

### 5.1 Árbol de decisión por tarea

| Tarea analítica | Gráfico recomendado | Por qué |
|---|---|---|
| Comparar magnitudes entre categorías | **Barras horizontales** (si etiquetas largas) o verticales | Posición + longitud, máxima precisión |
| Ranking | Barras horizontales ordenadas | Lectura natural top-down |
| Tendencia en el tiempo | **Línea** | La continuidad sugiere secuencia |
| Comparar tendencias múltiples (≤5) | Líneas múltiples con color categórico | Si son más, usa *small multiples* |
| Composición (parte/todo) en un punto | **Barras apiladas 100%** o waffle | Mejor que pie si hay más de 3 categorías |
| Composición a lo largo del tiempo | **Área apilada** o stream | Cuidado: difícil comparar capas intermedias |
| Correlación entre dos variables | **Dispersión** | Posición en dos ejes |
| Distribución de una variable | **Histograma** o **densidad** | Histograma para audiencia general |
| Comparar distribuciones | **Box plot** (técnico) o **violín** (rico) | Box plot solo si la audiencia sabe leerlo |
| Patrones en una matriz | **Heatmap** | Color como tercer canal |
| Datos geográficos normalizados | **Mapa coroplético** | Normaliza por área o población |
| Datos geográficos absolutos | **Mapa de burbujas** | El tamaño codifica magnitud |
| Flujos entre categorías | **Sankey** o **chord diagram** | Solo si los flujos son pocos |
| Progreso hacia meta | **Bullet chart** (Few) o barra con marca de meta | Compacto, comparativo |

### 5.2 Gráficos prohibidos por defecto en dashboards

| Gráfico | Por qué evitarlo | Alternativa |
|---|---|---|
| **Pie chart con >4 categorías** | Comparar ángulos es impreciso | Barras horizontales ordenadas |
| **Donut chart** | Mismo problema que el pie | Barras o waffle |
| **3D de cualquier tipo** | Distorsión por perspectiva, oclusión | Versión 2D |
| **Doble eje Y con escalas distintas** | Sugiere relaciones espurias | Small multiples o gráficos separados |
| **Word cloud** como dato cuantitativo | El tamaño no codifica magnitud con precisión | Barras |
| **Radar/Spider** | Áreas distorsionan, orden arbitrario afecta lectura | Small multiples de barras |
| **Velocímetros (gauges) gigantes** | Ocupan mucho espacio para una sola cifra | Tarjeta de KPI con comparación |

### 5.3 Reglas finas por gráfico

**Barras:**
- Eje Y **siempre empieza en cero**. Truncar el eje exagera diferencias.
- Ordena por valor, no alfabéticamente (excepto si hay orden natural: meses, niveles ordinales).
- Horizontal si las etiquetas son largas o si hay más de ~7 categorías.
- Una sola serie: un solo color. Múltiples series: color categórico.

**Líneas:**
- Solo si el eje X es **ordenado** (típicamente tiempo). Para categorías nominales, barras.
- Más de 5 líneas: usa *small multiples* o destaca una y deja el resto en gris.
- Marca los puntos solo si son pocos o son datos discretos.

**Dispersión:**
- Si hay sobreplot (>1000 puntos), usa transparencia, hexbin o densidad 2D.
- Si añades una línea de tendencia, declara el método (LOESS, lineal, etc.).
- No interpretes correlación como causalidad en la etiqueta.

**Heatmap:**
- Paleta **secuencial** si el cero está en un extremo (ej: conteos).
- Paleta **divergente** si hay un punto central significativo (ej: cambio %, residuos).
- Ordena filas/columnas por similitud (clustering) cuando se busca estructura.

---

## 6. Color: paletas y reglas de uso

> **Principio.** El color es un canal precioso y caro. Cada color usado tiene que justificar su existencia. Color sin función analítica es ruido.

### 6.1 Los tres esquemas

| Esquema | Cuándo | Característica |
|---|---|---|
| **Secuencial** | Magnitud ordenada de menor a mayor | Un solo tono, variación de luminosidad |
| **Divergente** | Datos con punto central significativo (cero, media, meta) | Dos tonos opuestos con neutro central |
| **Categórico** | Grupos sin jerarquía | Tonos distintos, equivalentes en luminosidad |

### 6.2 Paletas recomendadas (validadas para daltonismo)

**Categóricas (Okabe–Ito, segura para daltónicos):**

| Nombre | Hex |
|---|---|
| Negro | `#000000` |
| Naranja | `#E69F00` |
| Azul cielo | `#56B4E9` |
| Verde bluish | `#009E73` |
| Amarillo | `#F0E442` |
| Azul | `#0072B2` |
| Bermellón | `#D55E00` |
| Rosa púrpura | `#CC79A7` |

**Secuenciales (ColorBrewer):**
- Azules: `#deebf7 → #08306b` (single hue)
- Viridis: `#440154 → #fde725` (perceptualmente uniforme, recomendada por defecto)

**Divergentes:**
- RdBu: `#67001f` (rojo) ↔ `#f7f7f7` (centro) ↔ `#053061` (azul)
- BrBG: marrón ↔ verde-azulado

> **Regla.** Usa Viridis o Cividis para mapas de calor secuenciales por defecto. Son perceptualmente uniformes y accesibles.

### 6.3 Reglas de uso del color

1. **Limita la paleta categórica a 5–7 colores máximo.** Más es ilegible.
2. **Reserva el color de marca para lo focal.** Todo lo demás en escala de grises.
3. **Evita rojo–verde como único contraste semántico** (afecta a ~8% de hombres). Acompaña siempre con forma, símbolo o etiqueta.
4. **Color secuencial:** la dirección importa. Más oscuro = más alto, por convención.
5. **No uses arcoíris** (`jet`, `rainbow`) para datos cuantitativos: no es perceptualmente uniforme y crea fronteras falsas.
6. **Consistencia entre vistas:** si "Ventas" es azul en un gráfico, debe ser azul en todos.
7. **Fondo claro u oscuro:** decide uno y mantenlo. Fondos blancos para print/analítica, oscuros para dashboards operativos en pantalla 24/7.

### 6.4 Paleta de sistema para un dashboard

Un dashboard bien diseñado típicamente usa:

- **1 color de marca** (focal, para el KPI o gráfico principal)
- **1 escala de grises** (5–7 tonos, para texto, ejes, fondos, elementos no-focales)
- **2 colores semánticos** (verde para positivo, rojo o naranja para negativo — siempre con redundancia textual o de forma)
- **1 paleta categórica** (máx 7) cuando hay segmentaciones
- **1 paleta secuencial** y/o **1 divergente** para mapas de calor / mapas geográficos

Total: ~5 familias de color. Todo lo demás es decoración.

---

## 7. Tipografía y texto

> **Principio.** La tipografía es parte del diseño, no un detalle final. Un dashboard ilegible es inútil aunque los datos sean perfectos.

### 7.1 Reglas básicas

- **Una sola familia tipográfica**, dos como máximo (una sans-serif para UI, opcional una mono para números si quieres alineación de cifras).
- **Sans-serif** para pantalla. Recomendadas: Inter, IBM Plex Sans, Source Sans, Roboto.
- **Tabular numerals** (`font-variant-numeric: tabular-nums`) para columnas de números: alinea decimales sin esfuerzo.
- **Jerarquía tipográfica clara**, al menos 3 niveles:

| Nivel | Uso | Tamaño aprox. (16px base) |
|---|---|---|
| H1 | Título del dashboard | 24–32 px, peso 600–700 |
| H2 | Título de sección o tarjeta | 18–20 px, peso 600 |
| KPI | Cifra principal | 36–48 px, peso 700 |
| Body | Texto general, etiquetas | 14–16 px, peso 400 |
| Caption | Fuente, fecha, notas | 12 px, peso 400, color gris |

### 7.2 Textos dentro de gráficos

- **Títulos descriptivos**, no técnicos: "Las ventas cayeron 12% en Q3" es mejor que "Ventas trimestrales 2025".
- **Etiquetas directas** sobre las líneas/barras > leyenda separada (reduce idas y vueltas del ojo).
- **Anotaciones** que expliquen picos, caídas o eventos relevantes.
- **Unidades** siempre visibles, idealmente en el título del eje.
- **Cifras redondeadas** apropiadamente: `$1.24M`, no `$1,243,728.45`.

### 7.3 Idioma y formato

- Decide formato de fecha (`2026-05-18` ISO, o `18 may 2026`) y mantenlo en todo el dashboard.
- Separadores de miles según locale: `1,234.56` (EN), `1.234,56` (ES-MX usa formato variable; decide y consistente).
- Porcentajes con 1 decimal por defecto: `12.3%`. Sin decimal si la cifra es grande y aproximada.

---

## 8. Posicionamiento, espaciado y rejilla

### 8.1 Sistema de rejilla

Usa una rejilla de **12 columnas** como base (es la convención web y permite combinaciones de tarjetas: 12, 6+6, 4+4+4, 3+3+3+3, 8+4, 9+3, etc.).

```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

- **KPIs:** filas de 4 tarjetas (3 columnas c/u) o 3 tarjetas (4 columnas c/u).
- **Gráfico principal:** 8 columnas. Panel lateral: 4 columnas.
- **Distribución 6+6:** comparación entre dos vistas equivalentes.

### 8.2 Espaciado (sistema 8-point)

Usa múltiplos de 8 px para todos los espaciados. Es la convención de diseño UI y produce ritmo visual.

| Espacio | Valor (px) | Uso |
|---|---|---|
| `xs` | 4 | Espacio dentro de etiquetas |
| `sm` | 8 | Padding interno de tarjetas pequeñas |
| `md` | 16 | Padding estándar de tarjetas |
| `lg` | 24 | Separación entre tarjetas |
| `xl` | 32 | Separación entre bandas |
| `2xl` | 48 | Márgenes laterales del dashboard |

### 8.3 Reglas de alineación

- **Alinea todo a la rejilla.** Nada flota.
- **Alinea los baselines** de tarjetas adyacentes (mismo top, mismo bottom).
- **Cifras grandes alineadas a la izquierda** dentro de la tarjeta. Etiqueta arriba, cifra debajo, comparación debajo de la cifra.
- En tablas, números **alineados a la derecha**, texto **a la izquierda**, encabezados según contenido.

### 8.4 Responsive

Un dashboard moderno debe degradar bien:

- **Desktop (≥1280 px):** rejilla completa de 12 columnas.
- **Tablet (768–1280):** colapsa a 8 columnas; los gráficos de 4 columnas pasan a 8.
- **Móvil (<768):** una sola columna, scroll vertical. Replantea qué gráficos son útiles a este ancho (muchos heatmaps no lo son).

### 8.5 Foco y orden de lectura

Diseña explícitamente el camino visual:

```
1) Título y filtros (arriba)
   ↓
2) KPIs principales (segunda banda)
   ↓
3) Tendencia principal (centro-izquierda)
   ↓
4) Desglose / segmentación (centro-derecha)
   ↓
5) Detalle / tabla (abajo)
```

Cualquier elemento fuera de este orden requiere justificación (alarma, alerta, dato urgente).

---

## 9. Interactividad

> **Principio.** La interactividad sirve para reducir densidad sin perder información. No es decoración.

### 9.1 Patrones útiles

| Patrón | Cuándo usarlo |
|---|---|
| **Hover/tooltip** | Mostrar valor exacto sin saturar el gráfico |
| **Filtros globales** | Periodo, segmento, geografía, producto |
| **Drill-down** | Click en una categoría para ver detalle |
| **Brushing & linking** | Selección en un gráfico filtra los demás (analítico) |
| **Toggle de vistas** | Cambiar entre absoluto/relativo, lineal/log |
| **Búsqueda y selección** | En tablas largas o listas de entidades |

### 9.2 Reglas

- **El estado inicial debe ser útil sin tocar nada.** Si requiere configurar filtros para que tenga sentido, mal diseño.
- **Persiste los filtros** del usuario entre sesiones cuando tenga sentido.
- **Indica visualmente qué está filtrado** y permite resetear con un click.
- **Tooltips compactos**: 2–4 líneas, no un párrafo.
- **Evita animaciones largas** (>300 ms) entre estados. Distraen y consumen tiempo.
- **Carga progresiva**: muestra esqueletos (skeletons) o spinners por componente, no bloquees toda la pantalla.

### 9.3 Latencia

- **< 100 ms:** percibido como instantáneo. Meta para hover y filtros simples.
- **100–300 ms:** percibido como rápido pero no inmediato. Aceptable.
- **> 1 s:** requiere indicador de carga visible.
- **> 10 s:** requiere considerar pre-cálculo, agregación o segmentar el dashboard.

---

## 10. Accesibilidad

> **Principio.** Un dashboard inaccesible excluye usuarios. Diseñar accesible desde el inicio mejora la claridad para todos.

### 10.1 Contraste

- **Texto normal:** ratio mínimo **4.5:1** contra el fondo (WCAG 2.1 AA).
- **Texto grande (>18pt o 14pt bold):** ratio mínimo **3:1**.
- **Elementos gráficos de UI** (bordes, iconos relevantes): ratio mínimo **3:1**.
- Verifica con herramientas como WebAIM Contrast Checker.

### 10.2 Color y daltonismo

- Nunca uses el color como único codificador semántico. Combínalo con forma, etiqueta directa o patrón.
- Verifica el dashboard con simuladores de daltonismo (deuteranopía, protanopía, tritanopía).
- Prefiere paletas Okabe–Ito o Viridis sobre paletas custom no validadas.

### 10.3 Texto alternativo y semántica

- Cada gráfico debe tener un **resumen textual** accesible (alt text o descripción larga) que explique qué muestra y cuál es el dato clave.
- Estructura HTML semántica: `<h1>`, `<h2>`, `<nav>`, `<main>`, no todo `<div>`.
- Si exportas el dashboard a PDF, incluye un resumen textual al inicio.

### 10.4 Navegación por teclado

- Filtros, botones y tabs deben ser navegables con `Tab` y activables con `Enter` / `Space`.
- Indicador de foco visible (no `outline: none` sin reemplazo).

### 10.5 Tamaño mínimo

- Texto: **12 px mínimo absoluto**, 14 px recomendado.
- Áreas táctiles (botones, controles): **44×44 px** mínimo en touch.

---

## 11. Incertidumbre en dashboards

> **Principio.** Mostrar un número sin su incertidumbre es mentir por omisión. Esto aplica especialmente a predicciones y estimaciones derivadas de modelos.

### 11.1 Cuándo mostrar incertidumbre

- Cuando el dato proviene de **una estimación, no de un conteo directo**.
- Cuando hay **muestreo** (encuestas, A/B tests, métricas de población).
- Cuando hay **predicciones** de un modelo.
- Cuando el dato es **incompleto** o tiene **lag** (datos del día actual aún parciales).

### 11.2 Cómo representarla

| Técnica | Cuándo |
|---|---|
| **Barras de error** | Comparación de medias con error estándar |
| **Bandas de confianza** | Serie de tiempo con intervalo de predicción |
| **Sombreado de zonas incompletas** | Datos del periodo en curso aún no cerrados |
| **Línea punteada para forecast** | Distinguir histórico de proyección |
| **Etiqueta textual** | "Datos preliminares — actualizado parcialmente" |

### 11.3 Reglas

- **Nunca presentes una predicción como un hecho.** Visualmente y textualmente distingue lo medido de lo proyectado.
- **Declara la fuente y la fecha** de cada dato. Idealmente en el footer o tooltip.
- **Si los datos son parciales**, indícalo con un sombreado y nota explícita.

---

## 12. Dashboards para modelos de ML/IA

> Esta sección es prioritaria para perfiles de IA y ciencia de datos. Un buen dashboard de modelo no solo muestra métricas; permite **diagnosticar** el modelo y **explicar** sus decisiones.

### 12.1 Componentes típicos

**Para un clasificador:**

```
┌─────────────────────────────────────────────────────┐
│  Modelo: RandomForest v2.3 — Actualizado 2026-05-15│
├──────────────────┬──────────────────┬───────────────┤
│ Accuracy: 0.873  │ AUC: 0.91        │ F1: 0.84      │
│ ─ vs baseline    │ ─ vs baseline    │ ─ vs baseline │
├──────────────────┴──────────────────┴───────────────┤
│  Matriz de confusión (heatmap)                      │
│                                                      │
├─────────────────────────────────────────────────────┤
│  Curva ROC          │  Curva Precisión–Recall       │
│                     │                                │
├─────────────────────┴────────────────────────────────┤
│  Importancia de variables (barras horizontales)      │
└─────────────────────────────────────────────────────┘
```

**Para un modelo en producción:**

- KPIs de modelo (accuracy, AUC, F1, latencia, throughput).
- **Drift de datos:** distribución de features en producción vs entrenamiento.
- **Drift de predicción:** distribución de outputs en el tiempo.
- **Calibración:** reliability diagram.
- Alertas si métrica cae bajo umbral.

### 12.2 Reglas específicas

- **Siempre compara contra baseline** (modelo anterior, clasificador aleatorio, regla simple).
- **Distingue métricas de entrenamiento, validación y producción.**
- **Para clases desbalanceadas**, prioriza curva precisión–recall sobre ROC.
- **Para multiclase**, usa matriz de confusión normalizada por fila (recall por clase).
- **SHAP**: el *summary plot* en un dashboard es excelente para entender importancia global. Para explicar predicciones individuales, usa *waterfall plot*.
- **Para embeddings**, recuerda que t-SNE/UMAP son herramientas exploratorias: no concluyas similitud por proximidad sin verificar.

### 12.3 Curvas de aprendizaje

Si el dashboard incluye entrenamiento:

- Loss y métrica en train vs validation, por época.
- Líneas claramente etiquetadas, no solo en leyenda.
- Eje Y en escala adecuada (a veces log para loss).
- Marca el punto de mejor validation y de early stopping.

---

## 13. Antipatrones: qué evitar

### 13.1 Lista de antipatrones de diseño

| Antipatrón | Por qué falla | Solución |
|---|---|---|
| **Eje Y truncado en barras** | Exagera diferencias | Empezar desde cero |
| **Pie con 8 rebanadas** | Imposible comparar ángulos | Barras horizontales |
| **3D para "verse moderno"** | Distorsiona magnitudes | 2D limpio |
| **Doble eje Y con escalas distintas** | Sugiere correlaciones falsas | Small multiples |
| **Arcoíris (jet) para datos continuos** | No perceptualmente uniforme, crea fronteras falsas | Viridis |
| **Demasiados colores** (>7) | Ilegible | Agrupar "otros" o usar escala de grises + foco |
| **Decoración (logos, gradientes, sombras)** | Aumenta carga cognitiva | Plano y limpio |
| **Sin contexto temporal en KPIs** | Número aislado no es informativo | Comparar con periodo anterior o meta |
| **Filtros que no afectan a todo** | Confunde al usuario | Indicar scope del filtro |
| **Sin fecha de actualización** | El usuario no sabe si confiar | Mostrar "Última actualización: …" |
| **Cherry-picking de fechas** | Sesga la narrativa | Mostrar rango completo o justificar |
| **Mezclar tipos de dashboard** | No sirve bien a nadie | Separar en pestañas o dashboards distintos |
| **Sobrecargar la primera pantalla** | Pierde foco | Una historia por pantalla, drill-down para detalle |

### 13.2 Lista de antipatrones éticos

- **Manipular escalas** para inflar o reducir cambios percibidos.
- **Omitir el cero** sin declararlo.
- **Usar escala log sin avisar.**
- **Agregar excesivamente** y ocultar variación relevante (ej: promedios que esconden desigualdad).
- **Comparar áreas** cuando una barra sería más honesta.
- **Etiquetas tendenciosas** ("crisis", "récord", "fracaso") sin sustento en los datos.
- **Omitir intervalos de confianza** en predicciones.

---

## 14. Implementación: stack técnico

### 14.1 Elegir herramienta según contexto

| Contexto | Recomendación |
|---|---|
| **Dashboards internos para equipos técnicos** | Streamlit, Dash, Panel (Python) |
| **Dashboards corporativos para negocio** | Power BI, Tableau, Looker |
| **Dashboards públicos web** | Plotly + Dash, Observable, custom D3 |
| **Reportes reproducibles** | Quarto, R Markdown |
| **Análisis exploratorio personal** | Notebooks con Seaborn/Plotly |
| **Dashboards de modelos ML** | Streamlit, Gradio, MLflow UI, Weights & Biases |

### 14.2 Bibliotecas de Python recomendadas

| Biblioteca | Fortaleza |
|---|---|
| **Plotly** | Interactividad lista de fábrica, exportable a web |
| **Altair / Vega-Lite** | Gramática declarativa, especificaciones reproducibles |
| **Matplotlib + Seaborn** | Control total, mejor para print/reporte estático |
| **Streamlit** | Prototipos rápidos de dashboard con poco código |
| **Dash** | Dashboards productivos web sobre Flask |

### 14.3 Buenas prácticas de implementación

- **Versiona el código** del dashboard con Git.
- **Separa datos, transformación y presentación.** El dashboard no debe contener lógica de negocio enterrada.
- **Documenta paletas, ejes y filtros** en un archivo de configuración.
- **Tests de visualización:** al menos snapshot tests para detectar regresiones visuales.
- **Performance:** pre-agrega datos cuando sea posible; no calcules todo en tiempo real si no hace falta.
- **Caché de queries** para filtros frecuentes.
- **Telemetría de uso:** sabe qué filtros y vistas usa la gente; rediseña según uso real, no según opiniones.

---

## 15. Lista de verificación final

Antes de publicar un dashboard, verifica:

### Propósito y audiencia
- [ ] Sé exactamente quién es la audiencia y qué decisión soporta.
- [ ] El mensaje principal es claro en menos de 10 segundos.
- [ ] Cada elemento del dashboard justifica su presencia.

### Diseño visual
- [ ] La jerarquía visual es explícita: hay un foco claro.
- [ ] El layout sigue una rejilla coherente.
- [ ] El espaciado usa un sistema (8-point o similar) consistente.
- [ ] La paleta tiene como máximo 5 familias de color.
- [ ] La tipografía tiene como máximo dos familias y tres niveles claros.

### Codificación visual
- [ ] Cada gráfico usa el canal más preciso disponible para su tarea (Cleveland–McGill).
- [ ] Las barras empiezan en cero.
- [ ] No hay 3D, pie chart con >4 categorías, ni doble eje Y con escalas distintas.
- [ ] Los colores tienen función analítica, no decorativa.
- [ ] Las paletas son secuenciales/divergentes/categóricas según el dato.

### Texto y contexto
- [ ] Cada KPI tiene comparación temporal o contra meta.
- [ ] Cada gráfico tiene título descriptivo y unidades.
- [ ] La fecha de última actualización es visible.
- [ ] Las fuentes están declaradas.

### Accesibilidad
- [ ] Contraste ≥ 4.5:1 en texto normal.
- [ ] Paleta validada para daltonismo.
- [ ] Información no depende solo del color.
- [ ] Navegable por teclado y con texto alternativo.

### Ética y honestidad
- [ ] No hay ejes truncados sin avisar.
- [ ] No hay escalas log no declaradas.
- [ ] No hay cherry-picking temporal.
- [ ] La incertidumbre se muestra cuando existe.
- [ ] Lo medido se distingue de lo proyectado.

### Reproducibilidad
- [ ] El código está versionado.
- [ ] Los datos crudos y procesados están separados.
- [ ] El dashboard puede regenerarse desde cero con el mismo resultado.

---

## 16. Referencias

### Libros base
- Cleveland, W. S. y McGill, R. (1984). *Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods*. JASA, 79(387), 531–554.
- Few, S. (2012). *Show Me the Numbers*, 2ª ed. Analytics Press.
- Knaflic, C. N. (2015). *Storytelling with Data*. Wiley.
- Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press. [cs.ubc.ca/~tmm/vadbook](https://www.cs.ubc.ca/~tmm/vadbook/)
- Schwabish, J. (2021). *Better Data Visualizations*. Columbia University Press.
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information*, 2ª ed. Graphics Press.
- Ware, C. (2012). *Information Visualization: Perception for Design*, 3ª ed. Morgan Kaufmann.
- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. [clauswilke.com/dataviz](https://clauswilke.com/dataviz)

### Interpretabilidad de modelos
- Lundberg, S. M. y Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Molnar, C. (2022). *Interpretable Machine Learning*, 2ª ed. [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)

### Recursos en línea
- ColorBrewer 2.0: [colorbrewer2.org](https://colorbrewer2.org)
- Okabe–Ito Color Universal Design: [jfly.uni-koeln.de/color](https://jfly.uni-koeln.de/color/)
- WCAG 2.1: [w3.org/TR/WCAG21](https://www.w3.org/TR/WCAG21/)
- The Turing Way: [the-turing-way.netlify.app](https://the-turing-way.netlify.app)
- Wattenberg, M. et al. (2016). *How to Use t-SNE Effectively*. [distill.pub/2016/misread-tsne](https://distill.pub/2016/misread-tsne/)

### Ética
- Cairo, A. (2019). *How Charts Lie*. Norton.
- D'Ignazio, C. y Klein, L. F. (2020). *Data Feminism*. MIT Press. [data-feminism.mitpress.mit.edu](https://data-feminism.mitpress.mit.edu)
- Hullman, J. y Diakopoulos, N. (2011). *Visualization Rhetoric: Framing Effects in Narrative Visualization*. IEEE TVCG, 17(12).

---

> **Cierre.** Este documento es una guía operativa, no un dogma. La regla final que ninguna lista de verificación captura: **diseña para tu audiencia, no para tu portafolio**. El mejor dashboard es el que la gente usa para tomar mejores decisiones, no el que se ve más impresionante en una captura de pantalla.
