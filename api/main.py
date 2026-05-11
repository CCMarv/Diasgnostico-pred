from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException

from .config import (
    MARGEN_INCERTIDUMBRE,
    UMBRAL_RIESGO_ALTO,
    UMBRAL_RIESGO_BAJO,
    ConfiguracionAPI,
    ConfiguracionLogs,
    ConfiguracionRutas,
)
from .esquemas import DatosPaciente, DetallesSalud, RespuestaPrediccion, RespuestaSalud
from inferencia.predictor import PredictorDiabetes

NIVEL_LOG = logging.getLevelNamesMapping().get(ConfiguracionLogs.NIVEL.upper(), logging.INFO)
logging.basicConfig(format=ConfiguracionLogs.FORMATO, level=NIVEL_LOG)
logger = logging.getLogger(__name__)
logger.info("Arrancando API de Diagnóstico Predictivo...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Propósito: inicializar predictor al arrancar; firma: (app: FastAPI) -> async context manager."""
    predictor = PredictorDiabetes()
    predictor.cargar_modelo()
    app.state.predictor = predictor
    yield


def crear_app() -> FastAPI:
    """Propósito: construir FastAPI; firma: () -> FastAPI; error: fallos internos se delegan a FastAPI."""
    app = FastAPI(title=ConfiguracionAPI.TITULO, version=ConfiguracionAPI.VERSION, lifespan=lifespan)

    @app.get("/salud", response_model=RespuestaSalud)
    def salud() -> RespuestaSalud:
        """
        Propósito:
        Exponer salud operativa con modo degradado proactivo.

        Firma técnica:
        - Parámetros: ninguno.
        - Retorno: RespuestaSalud.

        Lógica resumida:
        - Verifica existencia y tamaño > 0 del archivo de modelo.
        - Emite estado operativo/degradado y metadatos de diagnóstico.

        Caso de error principal:
        - Si falla acceso al filesystem, se degrada a `modelo_cargado=False`.
        """
        ruta_modelo = ConfiguracionRutas.RUTA_MODELO
        modelo_cargado = False
        if ruta_modelo.exists():
            try:
                modelo_cargado = ruta_modelo.stat().st_size > 0
            except OSError as exc:
                logger.warning(
                    "No fue posible inspeccionar el archivo de modelo; API operando en modo degradado: %s",
                    exc,
                )

        estado = "operativo" if modelo_cargado else "degradado"
        if not modelo_cargado:
            logger.warning("Modelo no disponible o vacío; API operando en modo degradado.")

        return RespuestaSalud(
            estado=estado,
            version=ConfiguracionAPI.VERSION,
            detalles=DetallesSalud(
                modelo_cargado=modelo_cargado,
                ruta_modelo=ruta_modelo.name,
                timestamp_servidor=datetime.now(tz=timezone.utc).isoformat(),
            ),
        )

    @app.post("/predecir", response_model=RespuestaPrediccion)
    def predecir(payload: DatosPaciente) -> RespuestaPrediccion:
        predictor: PredictorDiabetes = app.state.predictor
        if not predictor.esta_listo():
            raise HTTPException(status_code=503, detail="El modelo aún no está disponible.")

        try:
            resultado = predictor.predecir(payload.a_dataframe())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="El modelo aún no está disponible.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Error interno al ejecutar predicción.") from exc

        probabilidad = float(resultado["probabilidad"])
        if probabilidad < UMBRAL_RIESGO_BAJO:
            categoria = "bajo"
        elif probabilidad < UMBRAL_RIESGO_ALTO:
            categoria = "medio"
        else:
            categoria = "alto"

        advertencia = None
        cerca_bajo = abs(probabilidad - UMBRAL_RIESGO_BAJO) <= MARGEN_INCERTIDUMBRE
        cerca_alto = abs(probabilidad - UMBRAL_RIESGO_ALTO) <= MARGEN_INCERTIDUMBRE
        if cerca_bajo or cerca_alto:
            advertencia = "Resultado en zona de incertidumbre; requiere evaluación clínica."

        return RespuestaPrediccion(
            categoria_riesgo=categoria,
            confianza=probabilidad,
            version=str(resultado["version"]),
            tiempo_ms=int(resultado["tiempo_ms"]),
            advertencia=advertencia,
        )

    return app


app = crear_app()
