from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def _payload_base() -> dict[str, int | float]:
    return {
        "presion_alta": 1,
        "colesterol_alto": 1,
        "chequeo_colesterol": 1,
        "imc": 29.5,
        "fumador": 0,
        "derrame_cerebral": 0,
        "enfermedad_corazon": 0,
        "actividad_fisica": 1,
        "consume_fruta": 1,
        "consume_verdura": 1,
        "consumo_alcohol_alto": 0,
        "tiene_cobertura_medica": 1,
        "sin_medico_por_costo": 0,
        "salud_general": 2,
        "salud_mental": 2,
        "salud_fisica": 2,
        "dificultad_caminar": 0,
        "sexo": 1,
        "edad": 7,
        "educacion": 4,
        "ingreso": 6,
    }


class PredictorListoFalso:
    def esta_listo(self) -> bool:
        return False


class PredictorListoVerdadero:
    def esta_listo(self) -> bool:
        return True

    def predecir(self, _entrada):
        return {
            "probabilidad": 0.66,
            "clase": 1,
            "version": "0.1.0",
            "tiempo_ms": 3,
        }


def test_salud_degradado_sin_modelo():
    with TestClient(app) as client:
        respuesta = client.get("/salud")

    assert respuesta.status_code == 200
    data = respuesta.json()
    assert data["estado"] in {"operativo", "degradado"}
    assert "detalles" in data
    assert "modelo_cargado" in data["detalles"]


def test_predecir_retorna_503_si_modelo_no_esta_listo():
    with TestClient(app) as client:
        app.state.predictor = PredictorListoFalso()
        respuesta = client.post("/predecir", json=_payload_base())

    assert respuesta.status_code == 503


def test_predecir_ok_mapea_categoria_y_advertencia():
    with TestClient(app) as client:
        app.state.predictor = PredictorListoVerdadero()
        respuesta = client.post("/predecir", json=_payload_base())

    assert respuesta.status_code == 200
    data = respuesta.json()
    assert data["categoria_riesgo"] == "alto"
    assert data["advertencia"] is not None
