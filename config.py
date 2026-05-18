from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, NoReturn, TypeAlias

"""
Propósito:
Centralizar contratos de configuración inmutables del sistema.

Firma técnica:
- Clases estáticas sin instanciación: ConfiguracionRutas, ConfiguracionAPI, ConfiguracionLogs.
- Alias tipado: LatenciaMilisegundos.

Lógica resumida:
- Define rutas y metadatos operativos por dominio.
- Expone alias de compatibilidad para módulos existentes.

Caso de error principal:
- Instanciar clases de configuración lanza TypeError para preservar inmutabilidad.
"""

LatenciaMilisegundos: TypeAlias = Annotated[float, "milisegundos"]


class _ConfiguracionInmutable:
    """Base para bloquear instanciación y uso no estático."""

    def __new__(cls, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(f"{cls.__name__} solo expone constantes de clase y no se instancia.")


class ConfiguracionRutas(_ConfiguracionInmutable):
    """Constantes de rutas del proyecto."""

    DIR_BASE: Final[Path] = Path(__file__).resolve().parent
    DIR_MODELOS: Final[Path] = DIR_BASE / "modelos"
    RUTA_MODELO: Final[Path] = DIR_MODELOS / "predictor_production.joblib"
    DIR_REPORTES: Final[Path] = DIR_BASE / "reportes"
    NOMBRE_REPORTE: Final[str] = "metricas_sprint1.json"


class ConfiguracionAPI(_ConfiguracionInmutable):
    """Constantes públicas de la API."""

    TITULO: Final[str] = "API de Diagnóstico Predictivo de Diabetes"
    VERSION: Final[str] = "0.1.0"
    LATENCIA_MAXIMA_MS: Final[LatenciaMilisegundos] = 100.0


class ConfiguracionLogs(_ConfiguracionInmutable):
    """Constantes de logging estructural."""

    FORMATO: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    NIVEL: Final[str] = "INFO"


BASE_DIR: Final[Path] = ConfiguracionRutas.DIR_BASE
DATOS_DIR: Final[Path] = BASE_DIR / "datos"
DATOS_BRUTOS_DIR: Final[Path] = DATOS_DIR / "brutos"
DATOS_PROCESADOS_DIR: Final[Path] = DATOS_DIR / "procesados"
MODELOS_DIR: Final[Path] = ConfiguracionRutas.DIR_MODELOS
REPORTES_DIR: Final[Path] = ConfiguracionRutas.DIR_REPORTES

NOMBRE_DATASET: Final[str] = "diabetes_binary_health_indicators_BRFSS2015.csv"
NOMBRE_DATASET_UCI_ID: Final[int] = 891
RUTA_DATASET_PREDETERMINADA: Final[Path] = DATOS_BRUTOS_DIR / NOMBRE_DATASET
RUTA_DATASET_PROCESADO: Final[Path] = DATOS_PROCESADOS_DIR / "dataset_procesado.parquet"

COLUMNAS_CDC: Final[tuple[str, ...]] = (
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
)

COLUMNA_OBJETIVO: Final[str] = "Diabetes_binary"
SEMILLA_ALEATORIA: Final[int] = 42
PROPORCION_PRUEBA: Final[float] = 0.2
MODELOS_SUPERVISADOS: Final[tuple[str, ...]] = ("svm", "arbol", "gbm", "mlp")
UMBRAL_MINIMO_AUC: Final[float] = 0.75
NOMBRE_MODELO_FINAL: Final[str] = ConfiguracionRutas.RUTA_MODELO.name
RUTA_MODELO_FINAL: Final[Path] = ConfiguracionRutas.RUTA_MODELO
RUTA_REPORTE_METRICAS: Final[Path] = ConfiguracionRutas.DIR_REPORTES / ConfiguracionRutas.NOMBRE_REPORTE

UMBRAL_RIESGO_BAJO: Final[float] = 0.33
UMBRAL_RIESGO_ALTO: Final[float] = 0.66
MARGEN_INCERTIDUMBRE: Final[float] = 0.05

VERSION_SISTEMA: Final[str] = ConfiguracionAPI.VERSION
