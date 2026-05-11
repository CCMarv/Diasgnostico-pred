from __future__ import annotations

from typing import Final, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

MAPEO_API_A_CDC: Final[dict[str, str]] = {
    "presion_alta": "HighBP",
    "colesterol_alto": "HighChol",
    "chequeo_colesterol": "CholCheck",
    "imc": "BMI",
    "fumador": "Smoker",
    "derrame_cerebral": "Stroke",
    "enfermedad_corazon": "HeartDiseaseorAttack",
    "actividad_fisica": "PhysActivity",
    "consume_fruta": "Fruits",
    "consume_verdura": "Veggies",
    "consumo_alcohol_alto": "HvyAlcoholConsump",
    "tiene_cobertura_medica": "AnyHealthcare",
    "sin_medico_por_costo": "NoDocbcCost",
    "salud_general": "GenHlth",
    "salud_mental": "MentHlth",
    "salud_fisica": "PhysHlth",
    "dificultad_caminar": "DiffWalk",
    "sexo": "Sex",
    "edad": "Age",
    "educacion": "Education",
    "ingreso": "Income",
}


class DatosPaciente(BaseModel):
    """Entrada pública de inferencia (sin PII)."""

    presion_alta: int = Field(ge=0, le=1)
    colesterol_alto: int = Field(ge=0, le=1)
    chequeo_colesterol: int = Field(ge=0, le=1)
    imc: float = Field(ge=10, le=80)
    fumador: int = Field(ge=0, le=1)
    derrame_cerebral: int = Field(ge=0, le=1)
    enfermedad_corazon: int = Field(ge=0, le=1)
    actividad_fisica: int = Field(ge=0, le=1)
    consume_fruta: int = Field(ge=0, le=1)
    consume_verdura: int = Field(ge=0, le=1)
    consumo_alcohol_alto: int = Field(ge=0, le=1)
    tiene_cobertura_medica: int = Field(ge=0, le=1)
    sin_medico_por_costo: int = Field(ge=0, le=1)
    salud_general: int = Field(ge=1, le=5)
    salud_mental: int = Field(ge=0, le=30)
    salud_fisica: int = Field(ge=0, le=30)
    dificultad_caminar: int = Field(ge=0, le=1)
    sexo: int = Field(ge=0, le=1)
    edad: int = Field(ge=1, le=13)
    educacion: int = Field(ge=1, le=6)
    ingreso: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def validar_coherencia_clinica(self) -> "DatosPaciente":
        if self.salud_fisica >= 20 and self.dificultad_caminar == 0:
            raise ValueError(
                "Incoherencia clínica: salud_fisica muy alta requiere revisar dificultad_caminar."
            )
        return self

    def a_dataframe(self) -> pd.DataFrame:
        datos = self.model_dump()
        fila_cdc = {MAPEO_API_A_CDC[campo]: valor for campo, valor in datos.items()}
        return pd.DataFrame([fila_cdc])


class RespuestaPrediccion(BaseModel):
    categoria_riesgo: Literal["bajo", "medio", "alto"]
    confianza: float = Field(ge=0.0, le=1.0)
    version: str
    tiempo_ms: int = Field(ge=0)
    advertencia: str | None = None


class DetallesSalud(BaseModel):
    modelo_cargado: bool
    ruta_modelo: str
    timestamp_servidor: str


class RespuestaSalud(BaseModel):
    estado: Literal["operativo", "degradado"]
    version: str
    detalles: DetallesSalud
