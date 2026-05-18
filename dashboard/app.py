"""Dashboard interactivo de Diagnóstico Predictivo de Diabetes — reconstrucción.

Objetivo:
- Aplicar las `guidelines` del repositorio para mejorar jerarquía, accesibilidad
  y explicabilidad del dashboard ya provisto.

Audiencia primaria: analistas de datos y equipo clínico académico (revisión).
Audiencia secundaria: gestores operativos (visiones resumidas).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# aseguramos que el paquete raíz esté en sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLUMNAS_CDC,
    REPORTES_DIR,
    ConfiguracionRutas,
    UMBRAL_RIESGO_BAJO,
    UMBRAL_RIESGO_ALTO,
    MARGEN_INCERTIDUMBRE,
)

# Página
st.set_page_config(
    page_title="Diagnóstico Predictivo · Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


RUTA_MODELO = ConfiguracionRutas.RUTA_MODELO
RUTA_BENCHMARK = REPORTES_DIR / "benchmark_5000.json"
RUTA_FENOTIPADO = REPORTES_DIR / "hallazgos_fenotipado.json"

# Paleta cuidada (Okabe–Ito / accesible)
PALETA_OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR_SEGURO = PALETA_OKABE_ITO[2]
COLOR_ALERTA = PALETA_OKABE_ITO[1]
COLOR_RIESGO = PALETA_OKABE_ITO[6]


# Mapeo de columnas CDC -> etiquetas en español y breves descripciones
COLUMNAS_ES = {
    "HighBP": ("Hipertensión", "Diagnóstico previo de hipertensión arterial (0/1)"),
    "HighChol": ("Colesterol alto", "Diagnóstico previo de colesterol alto (0/1)"),
    "CholCheck": ("Chequeo colesterol", "Se realizó prueba de colesterol en últimos 5 años (0/1)"),
    "BMI": ("IMC (BMI)", "Índice de masa corporal kg/m²"),
    "Smoker": ("Fumador", "Ha fumado ≥100 cigarrillos en su vida (0/1)"),
    "Stroke": ("Derrame cerebral", "Antecedente de derrame cerebral (0/1)"),
    "HeartDiseaseorAttack": ("Enfermedad cardíaca", "Antecedente de enfermedad cardiaca o infarto (0/1)"),
    "PhysActivity": ("Actividad física", "Actividad física en últimos 30 días (0/1)"),
    "Fruits": ("Consume fruta", "Consume fruta ≥1 vez al día (0/1)"),
    "Veggies": ("Consume verduras", "Consume verduras ≥1 vez al día (0/1)"),
    "HvyAlcoholConsump": ("Consumo alto alcohol", "Consumo excesivo de alcohol según CDC (0/1)"),
    "AnyHealthcare": ("Cobertura médica", "Posee algún tipo de seguro o cobertura (0/1)"),
    "NoDocbcCost": ("No consulta por costo", "Evita ir al médico por costo (0/1)"),
    "GenHlth": ("Salud general", "Auto-reporte: 1=excelente … 5=mala"),
    "MentHlth": ("Salud mental (días)", "Días con mala salud mental en último mes (0–30)"),
    "PhysHlth": ("Salud física (días)", "Días con mala salud física en último mes (0–30)"),
    "DiffWalk": ("Dificultad para caminar", "Dificultad para caminar o subir escaleras (0/1)"),
    "Sex": ("Sexo", "0=Femenino, 1=Masculino"),
    "Age": ("Grupo edad", "Grupo etario CDC (1=18–24 … 13=80+)") ,
    "Education": ("Educación", "Nivel educativo (1=ninguno … 6=universidad)"),
    "Income": ("Ingreso", "Nivel de ingresos (1=<$10k … 8=>$75k)"),
}


@st.cache_resource(show_spinner="Cargando modelo…")
def _cargar_modelo() -> Optional[object]:
    if not RUTA_MODELO.exists():
        return None
    import joblib

    return joblib.load(RUTA_MODELO)


@st.cache_data(show_spinner="Cargando reporte…")
def _cargar_json(ruta: Path) -> Optional[dict]:
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _badge_nivel(roc: float) -> str:
    if roc >= 0.80:
        return "🟢 Excelente"
    if roc >= 0.75:
        return "🟡 Aceptable"
    return "🔴 Por debajo del umbral"


def _file_last_modified(ruta: Path) -> Optional[str]:
    try:
        ts = ruta.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return None


# Banner visible para confirmar versión cargada (útil para verificar recarga)
_this_file_ts = _file_last_modified(Path(__file__))
st.markdown(
    f"**RECONSTRUCCIÓN:** versión actualizada del dashboard · archivo modificado: {_this_file_ts} \n\n"
    "---"
)


# Sidebar: selección de vista y contexto
st.sidebar.title("🩺 Diabetes · Dashboard")
vista = st.sidebar.radio(
    "Vista",
    ["📊 Comparativa de modelos", "🔍 Predicción individual", "🧬 Fenotipos K-Means", "📈 Calibración & Explicabilidad"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto académico — CDC BRFSS 2015")
st.sidebar.caption("Python ≥ 3.11 · scikit-learn · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — Comparativa de modelos
# ══════════════════════════════════════════════════════════════════════════════
if vista == "📊 Comparativa de modelos":
    st.title("📊 Comparativa de modelos supervisados")
    st.markdown(
        "Tabla comparativa de los 4 modelos entrenados sobre el dataset CDC BRFSS 2015. "
        "Las métricas se calculan sobre el conjunto de **prueba** (sin data leakage)."
    )

    # selección de benchmark
    benchmarks_disponibles = {
        p.stem: p
        for p in sorted(REPORTES_DIR.glob("benchmark_*.json"))
        if "manifest" not in p.stem
    }

    if not benchmarks_disponibles:
        st.warning(
            "No se encontraron archivos de benchmark en `reportes/`. "
            "Ejecuta el pipeline para generarlos:\n"
            "```bash\npython -m entrenamiento.pipeline --modo clasificacion\n```"
        )
        st.stop()

    nombre_bench = st.selectbox(
        "Selecciona el benchmark",
        list(benchmarks_disponibles.keys()),
        index=0,
    )
    datos = _cargar_json(benchmarks_disponibles[nombre_bench])

    # Banda 1: metadata y accesibilidad
    ultima_mod = _file_last_modified(benchmarks_disponibles[nombre_bench])
    with st.container():
        st.markdown(f"**Archivo seleccionado:** {nombre_bench} — Última modificación: {ultima_mod}")
        st.markdown(
            "**Resumen accesible:** Esta vista compara las métricas principales de cada modelo "
            "(ROC-AUC, PR-AUC, sensibilidad, especificidad, F1, Brier). Los valores mostrados "
            "corresponden al conjunto de prueba y sirven para auditoría técnica."
        )

    if datos is None:
        st.error("No se pudo leer el JSON seleccionado.")
        st.stop()

    # construir DataFrame de métricas
    filas = []
    for modelo, m in datos.get("modelos", {}).items():
        filas.append({
            "Modelo": modelo.upper(),
            "ROC-AUC": round(m["roc_auc"], 4),
            "PR-AUC": round(m["pr_auc"], 4),
            "Sensibilidad": round(m["sensibilidad"], 4),
            "Especificidad": round(m["especificidad"], 4),
            "F1": round(m["f1_clase_positiva"], 4),
            "Brier Score": round(m["brier_score"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "Nivel": _badge_nivel(m["roc_auc"]),
        })
    df_m = pd.DataFrame(filas).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    mejor = df_m.iloc[0]["Modelo"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mejor modelo", mejor)
    col2.metric("ROC-AUC", df_m.iloc[0]["ROC-AUC"])
    col3.metric("PR-AUC", df_m.iloc[0]["PR-AUC"])

    # Banda 2: KPIs explicados (lectura rápida en <10s)
    st.markdown(
        "**Lectura rápida:** El `ROC-AUC` mide la capacidad de ordenamiento del modelo; "
        "`PR-AUC` es más informativa en conjuntos desbalanceados. Consulta la tabla para métricas adicionales."
    )

    st.markdown("### Tabla de métricas")
    st.dataframe(
        df_m.style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "Sensibilidad", "Especificidad", "F1", "Accuracy"],
            color="#c8f7c5",
        ).highlight_min(
            subset=["Brier Score"],
            color="#c8f7c5",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### ROC-AUC y PR-AUC por modelo")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(df_m))
    w = 0.35
    bars1 = ax.bar(x - w / 2, df_m["ROC-AUC"], w, label="ROC-AUC", color=COLOR_SEGURO)
    bars2 = ax.bar(x + w / 2, df_m["PR-AUC"], w, label="PR-AUC", color=COLOR_ALERTA)
    ax.axhline(0.75, color=COLOR_RIESGO, linestyle="--", linewidth=1, alpha=0.7,
               label="Umbral mínimo (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_m["Modelo"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cómo leer: cada barra muestra la puntuación media sobre el conjunto de prueba. "
        "Valores cercanos a 1 son mejores. El umbral mínimo aceptable se muestra con la línea punteada."
    )

    # desbalance
    desbalance = datos.get("desbalance", {})
    if desbalance:
        st.markdown("### Contexto del dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clase 0 (sin diabetes)", f"{desbalance.get('pct_clase_0', 0):.1%}")
        c2.metric("Clase 1 (diabetes)", f"{desbalance.get('pct_clase_1', 0):.1%}")
        c3.metric("Ratio desbalance", f"{desbalance.get('ratio', 0):.1f}:1")
        st.markdown(
            "**Nota:** la `PR-AUC` es una métrica recomendable cuando la clase positiva es minoritaria; "
            "la `ROC-AUC` permanece útil para comparar modelos cuando las tasas base son similares."
        )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 2 — Predicción individual
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "🔍 Predicción individual":
    st.title("🔍 Predicción de riesgo individual")
    st.markdown(
        "Ingresa los indicadores de salud del paciente para obtener una estimación "
        "de riesgo de diabetes tipo 2. El modelo evalúa 21 variables del CDC BRFSS."
    )

    modelo = _cargar_modelo()
    if modelo is None:
        st.warning(
            "⚠️ El modelo entrenado no está disponible en `modelos/predictor_production.joblib`. "
            "Para generarlo, ejecuta el pipeline de entrenamiento:\n"
            "```bash\npython -m entrenamiento.pipeline --modo clasificacion "
            "--modelos svm,arbol,gbm,mlp\n```"
        )

    st.markdown("### Datos del paciente")

    with st.form("formulario_prediccion"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Condiciones crónicas**")
            HighBP = int(st.checkbox("Hipertensión (HighBP)"))
            HighChol = int(st.checkbox("Colesterol alto (HighChol)"))
            CholCheck = int(st.checkbox("Revisión de colesterol en 5 años (CholCheck)", value=True))
            Stroke = int(st.checkbox("Accidente cerebrovascular (Stroke)"))
            HeartDiseaseorAttack = int(st.checkbox("Enfermedad cardíaca o infarto"))

            st.markdown("**Hábitos**")
            Smoker = int(st.checkbox("Fumador (≥100 cigarros vida)"))
            PhysActivity = int(st.checkbox("Actividad física (último mes)", value=True))
            Fruits = int(st.checkbox("Consume frutas diariamente", value=True))
            Veggies = int(st.checkbox("Consume verduras diariamente", value=True))
            HvyAlcoholConsump = int(st.checkbox("Consumo alto de alcohol"))

        with col_b:
            st.markdown("**Acceso a salud**")
            AnyHealthcare = int(st.checkbox("Tiene cobertura médica", value=True))
            NoDocbcCost = int(st.checkbox("No fue al médico por costo"))
            DiffWalk = int(st.checkbox("Dificultad para caminar"))

            st.markdown("**Medidas continuas**")
            BMI = st.slider("IMC (BMI)", min_value=10, max_value=99, value=27)
            MentHlth = st.slider("Días salud mental afectada (últimos 30)", 0, 30, 0)
            PhysHlth = st.slider("Días salud física afectada (últimos 30)", 0, 30, 0)

        with col_c:
            st.markdown("**Variables ordinales**")
            Sex = int(st.selectbox("Sexo biológico", options=[0, 1],
                                   format_func=lambda x: "Femenino" if x == 0 else "Masculino"))
            GenHlth = st.selectbox(
                "Salud general (1=excelente … 5=mala)", [1, 2, 3, 4, 5], index=2
            )
            Age = st.selectbox(
                "Grupo de edad",
                options=list(range(1, 14)),
                format_func=lambda x: {
                    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
                    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
                    11: "70–74", 12: "75–79", 13: "80+",
                }[x],
                index=6,
            )
            Education = st.selectbox(
                "Nivel educativo (1=ninguno … 6=universidad)",
                [1, 2, 3, 4, 5, 6], index=4
            )
            Income = st.selectbox(
                "Nivel de ingresos (1=<$10k … 8=>$75k)",
                [1, 2, 3, 4, 5, 6, 7, 8], index=4
            )

        enviado = st.form_submit_button("Calcular riesgo", use_container_width=True)

    if enviado:
        entrada = pd.DataFrame([{
            "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck,
            "BMI": BMI, "Smoker": Smoker, "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack, "PhysActivity": PhysActivity,
            "Fruits": Fruits, "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump,
            "AnyHealthcare": AnyHealthcare, "NoDocbcCost": NoDocbcCost,
            "GenHlth": GenHlth, "MentHlth": MentHlth, "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk, "Sex": Sex, "Age": Age,
            "Education": Education, "Income": Income,
        }])[list(COLUMNAS_CDC)]

        if modelo is None:
            st.info("Modelo no disponible. Mostraría la probabilidad de riesgo aquí.")
        else:
            prob = float(modelo.predict_proba(entrada)[0, 1])
            # usar umbrales y margen de incertidumbre del contrato
            if prob >= UMBRAL_RIESGO_ALTO:
                nivel, color, icono = "Alto", COLOR_RIESGO, "🔴"
            elif prob >= UMBRAL_RIESGO_BAJO:
                nivel, color, icono = "Medio", COLOR_ALERTA, "🟡"
            else:
                nivel, color, icono = "Bajo", COLOR_SEGURO, "🟢"

            # Advertencia de incertidumbre si está cerca de un umbral
            aviso_incert = None
            if abs(prob - UMBRAL_RIESGO_BAJO) <= MARGEN_INCERTIDUMBRE or abs(prob - UMBRAL_RIESGO_ALTO) <= MARGEN_INCERTIDUMBRE:
                aviso_incert = (
                    "Atención: probabilidad cercana al umbral operativo. Se recomienda revisión clínica y repetir mediciones."
                )

            st.markdown("---")
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.metric(f"{icono} Riesgo de diabetes", f"{prob:.1%}", delta=nivel)
            with col_r2:
                fig2, ax2 = plt.subplots(figsize=(5, 1.2))
                ax2.barh(["Probabilidad"], [prob], color=color, height=0.5)
                ax2.barh(["Probabilidad"], [1 - prob], left=[prob],
                         color="#EEEEEE", height=0.5)
                ax2.set_xlim(0, 1)
                ax2.axvline(0.33, color="gray", linestyle=":", alpha=0.5)
                ax2.axvline(0.66, color="gray", linestyle=":", alpha=0.5)
                ax2.set_xlabel("Probabilidad")
                ax2.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            if nivel == "Alto":
                st.error(
                    "Riesgo **alto**. Se recomienda evaluación clínica adicional "
                    "(glucemia en ayuno, HbA1c)."
                )
            elif nivel == "Medio":
                st.warning(
                    "Riesgo **medio**. Seguimiento preventivo y hábitos saludables recomendados."
                )
            else:
                st.success("Riesgo **bajo** según los indicadores ingresados.")

            if aviso_incert:
                st.info(aviso_incert)

            # Explicabilidad (SHAP) — intentar mostrar si está disponible
            with st.expander("Explicación de la predicción (SHAP) — ver detalles"):
                try:
                    import shap

                    modelo_local = modelo
                    # preparar explainer de forma segura; si el pipeline contiene preprocesador, usarlo
                    explainer = shap.Explainer(modelo_local.predict_proba, entrada)
                    shap_values = explainer(entrada)
                    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
                except Exception as e:  # pragma: no cover - depende de entorno del usuario
                    st.info(
                        "SHAP no está disponible en este entorno o no se pudo calcular la explicación. "
                        "Instala extras opcionales: pip install -e .[shap] para habilitarlo."
                    )

        # Mostrar valores de entrada con etiquetas y descripciones cortas
        st.markdown("---")
        st.markdown("### Valores ingresados (desglose)")
        rows = []
        lista = list(COLUMNAS_CDC)
        for col in lista:
            label, desc = COLUMNAS_ES.get(col, (col, ""))
            rows.append({"Variable": label, "Valor": entrada.iloc[0][col], "Descripción": desc})
        st.dataframe(pd.DataFrame(rows).set_index("Variable"), use_container_width=True)

        st.caption(
            "Interpretación rápida: la probabilidad mostrada es la estimación del modelo para la clase 'diabetes'. "
            "Usa los umbrales definidos en la configuración para categorizar riesgo; la recomendación clínica siempre debe validar resultados."
        )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 3 — Fenotipos K-Means
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "🧬 Fenotipos K-Means":
    st.title("🧬 Fenotipos de pacientes — K-Means")
    st.markdown(
        "Segmentación no supervisada de la población en grupos homogéneos. "
        "Los fenotipos se identifican a partir de los 21 indicadores CDC sin usar "
        "la etiqueta de diabetes."
    )

    datos_fen = _cargar_json(RUTA_FENOTIPADO)
    if datos_fen is None:
        st.warning(
            "⚠️ No se encontró `reportes/hallazgos_fenotipado.json`. "
            "Ejecuta el notebook para generarlo:\n"
            "```bash\nmake notebook\n```"
        )
        st.stop()

    meta = datos_fen.get("metadata", {})
    fenotipos = datos_fen.get("fenotipos", [])

    # métricas globales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K óptimo", meta.get("k_optimo", "—"))
    c2.metric("Silhouette", f"{meta.get('silhouette_score', 0):.3f}")
    c3.metric("χ² (p-value)", f"{meta.get('p_value_chi2', 1):.1e}")
    c4.metric("Registros", f"{meta.get('n_registros', 0):,}")

    # tabla de fenotipos
    st.markdown("### Distribución y prevalencia de diabetes")
    df_fen = pd.DataFrame([
        {
            "Fenotipo": f["nombre"],
            "Pacientes": f["n"],
            "% del total": round(f["n"] / meta.get("n_registros", 1) * 100, 1),
            "Tasa diabetes (%)": f["tasa_diabetes_pct"],
            "Top 4 variables": ", ".join(f["top_4_variables"]),
        }
        for f in fenotipos
    ]).sort_values("Tasa diabetes (%)", ascending=False)

    # Explicación accesible sobre qué es un fenotipo y cómo interpretar la tabla
    st.markdown(
        "**Qué es un fenotipo:** un grupo homogéneo de pacientes identificado por K-Means sobre las 21 variables CDC. "
        "`Top 4 variables` indica las variables con mayor influencia promedio en la distancia al centro del clúster. "
        "`Tasa diabetes (%)` es la prevalencia observada de la clase positiva dentro del fenotipo."
    )

    # Añadir columna de interpretación rápida según prevalencia
    mean_tasa = df_fen["Tasa diabetes (%)"].mean() if not df_fen.empty else 0
    def interpret_tasa(x):
        if x >= mean_tasa + 5:
            return "Alto riesgo relativo"
        if x <= mean_tasa - 5:
            return "Bajo riesgo relativo"
        return "Riesgo intermedio"
    df_fen["Interpretación"] = df_fen["Tasa diabetes (%)"].apply(interpret_tasa)

    st.dataframe(
        df_fen.style.highlight_max(subset=["Tasa diabetes (%)"], color="#ffd0d0")
                    .highlight_min(subset=["Tasa diabetes (%)"], color="#d0f0d0")
                    .format({"% del total": "{:.1f}%", "Tasa diabetes (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    # gráfica
    st.markdown("### Prevalencia de diabetes por fenotipo")
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    nombres = df_fen["Fenotipo"].tolist()
    tasas = df_fen["Tasa diabetes (%)"].tolist()
    colores_fen = [
        COLOR_RIESGO if t == max(tasas)
        else COLOR_SEGURO if t == min(tasas)
        else COLOR_ALERTA
        for t in tasas
    ]
    bars = ax3.bar(nombres, tasas, color=colores_fen)
    for bar, tasa in zip(bars, tasas):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{tasa:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Tasa de diabetes (%)")
    ax3.set_title("Prevalencia de diabetes tipo 2 por fenotipo")
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # descripción de cada fenotipo
    st.markdown("### Descripción de fenotipos")
    # preparar valores de referencia
    tasas = [f["tasa_diabetes_pct"] for f in fenotipos]
    max_t = max(tasas) if tasas else 0
    min_t = min(tasas) if tasas else 0
    for f in sorted(fenotipos, key=lambda x: x["tasa_diabetes"], reverse=True):
        icono = "🔴" if f["tasa_diabetes_pct"] == max_t else "🟢" if f["tasa_diabetes_pct"] == min_t else "🟡"
        with st.expander(
            f"{icono} {f['nombre']} — {f['tasa_diabetes_pct']:.1f}% diabetes ({f['n']:,} pacientes)"
        ):
            st.markdown(f"**Variables más elevadas (Top 4):** {', '.join(f['top_4_variables'])}")
            st.markdown(
                "**Interpretación:** "
                f"Este fenotipo agrupa {f['n']:,} pacientes ({f['n']/meta.get('n_registros',1):.1%} del total). "
                f"La prevalencia de diabetes dentro del grupo es {f['tasa_diabetes_pct']:.1f}%. "
                "Use esta información para priorizar intervenciones dirigidas a las variables listadas."
            )
            st.progress(f["tasa_diabetes"]) 


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 4 — Calibración & Explicabilidad global
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "📈 Calibración & Explicabilidad":
    st.title("📈 Calibración y explicabilidad global")
    st.markdown(
        "Esta vista reúne artefactos producidos por el pipeline: curvas ROC/PR, "
        "gráficas de calibración y métricas globales (Brier, AUC). También explica cuándo "
        "es necesario recalibrar un modelo para un nuevo contexto."
    )

    # seleccionar benchmark
    benchs = {
        p.stem: p
        for p in sorted(REPORTES_DIR.glob("benchmark_*.json"))
        if "manifest" not in p.stem
    }
    if not benchs:
        st.warning("No hay benchmarks disponibles en `reportes/` para mostrar calibración.")
        st.stop()

    sel = st.selectbox("Selecciona benchmark para ver métricas y artefactos", list(benchs.keys()), index=0)
    datos_b = _cargar_json(benchs[sel])
    if datos_b is None:
        st.error("No se pudo leer el benchmark seleccionado.")
        st.stop()

    # mostrar métricas clave como KPIs
    modelos = datos_b.get("modelos", {})
    kpicol1, kpicol2, kpicol3, kpicol4 = st.columns(4)
    # mejor por ROC ya calculado en vista 1, pero mostramos globalmente
    mejores = sorted(((m, v.get("roc_auc", 0)) for m, v in modelos.items()), key=lambda x: x[1], reverse=True)
    mejor_modelo = mejores[0][0].upper() if mejores else "—"
    kpicol1.metric("Mejor modelo (ROC-AUC)", mejor_modelo)
    # promedio Brier
    brier_vals = [v.get("brier_score", None) for v in modelos.values() if v.get("brier_score") is not None]
    kpicol2.metric("Brier medio", f"{(sum(brier_vals)/len(brier_vals)):.3f}" if brier_vals else "—")
    # cantidad de modelos
    kpicol3.metric("Modelos evaluados", len(modelos))
    kpicol4.metric("Tamaño prueba", f"{datos_b.get('n_test', '—')}")

    st.markdown("---")
    st.markdown("### Artefactos generados por el pipeline")
    col_a, col_b = st.columns(2)
    # mostrar imágenes si existen
    img1 = REPORTES_DIR / "curvas_gbm.png"
    img2 = REPORTES_DIR / "calibracion_gbm.png"
    if img1.exists():
        with col_a:
            st.image(str(img1), caption="Curvas ROC/PR (GBM)")
    else:
        col_a.info("No se encontró curvas_gbm.png")

    if img2.exists():
        with col_b:
            st.image(str(img2), caption="Curva de calibración (GBM)")
    else:
        col_b.info("No se encontró calibracion_gbm.png")

    st.markdown(
        "**Interpretación:** Una curva de calibración cercana a la diagonal indica probabilidades bien calibradas. "
        "Brier Score cuantifica el error de probabilidad; valores menores son mejores. Si el Brier es alto, considere recalibración (Platt/Isotonic)."
    )

    st.markdown("### Métricas por modelo (detalle)")
    filas_mod = []
    for mname, mv in modelos.items():
        filas_mod.append({
            "Modelo": mname.upper(),
            "ROC-AUC": mv.get("roc_auc"),
            "PR-AUC": mv.get("pr_auc"),
            "Brier": mv.get("brier_score"),
            "Sensibilidad": mv.get("sensibilidad"),
            "Especificidad": mv.get("especificidad"),
        })
    if filas_mod:
        st.dataframe(pd.DataFrame(filas_mod).set_index("Modelo"), use_container_width=True)
    else:
        st.info("No hay métricas por modelo en el JSON seleccionado.")

    st.markdown("---")
    st.markdown(
        "Si deseas añadir explicabilidad global (SHAP summary), genera `shap_summary.png` en `reportes/` desde el notebook o el pipeline y aparecerá aquí para su inspección."
    )
