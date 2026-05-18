"""Dashboard interactivo de Diagnóstico Predictivo de Diabetes — Sprint 4 (ítem I6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── ruta al paquete raíz ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLUMNAS_CDC,
    REPORTES_DIR,
    ConfiguracionRutas,
)

# ── configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico Predictivo · Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUTA_MODELO = ConfiguracionRutas.RUTA_MODELO
RUTA_BENCHMARK = REPORTES_DIR / "benchmark_5000.json"
RUTA_FENOTIPADO = REPORTES_DIR / "hallazgos_fenotipado.json"

COLOR_RIESGO = "#C62828"
COLOR_SEGURO = "#006847"
COLOR_ALERTA = "#F9A825"

# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo…")
def _cargar_modelo():
    if not RUTA_MODELO.exists():
        return None
    import joblib
    return joblib.load(RUTA_MODELO)


def _cargar_json(ruta: Path) -> dict | None:
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


# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🩺 Diabetes · Dashboard")
vista = st.sidebar.radio(
    "Vista",
    ["📊 Comparativa de modelos", "🔍 Predicción individual", "🧬 Fenotipos K-Means"],
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

    # desbalance
    desbalance = datos.get("desbalance", {})
    if desbalance:
        st.markdown("### Contexto del dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clase 0 (sin diabetes)", f"{desbalance.get('pct_clase_0', 0):.1%}")
        c2.metric("Clase 1 (diabetes)", f"{desbalance.get('pct_clase_1', 0):.1%}")
        c3.metric("Ratio desbalance", f"{desbalance.get('ratio', 0):.1f}:1")


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
            if prob >= 0.66:
                nivel, color, icono = "Alto", COLOR_RIESGO, "🔴"
            elif prob >= 0.33:
                nivel, color, icono = "Medio", COLOR_ALERTA, "🟡"
            else:
                nivel, color, icono = "Bajo", COLOR_SEGURO, "🟢"

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
    for f in sorted(fenotipos, key=lambda x: x["tasa_diabetes"], reverse=True):
        icono = "🔴" if f["tasa_diabetes_pct"] == max(x["tasa_diabetes_pct"] for x in fenotipos) \
                else "🟢" if f["tasa_diabetes_pct"] == min(x["tasa_diabetes_pct"] for x in fenotipos) \
                else "🟡"
        with st.expander(
            f"{icono} {f['nombre']} — {f['tasa_diabetes_pct']:.1f}% diabetes "
            f"({f['n']:,} pacientes)"
        ):
            st.markdown(f"**Variables más elevadas:** {', '.join(f['top_4_variables'])}")
            st.progress(f["tasa_diabetes"])
