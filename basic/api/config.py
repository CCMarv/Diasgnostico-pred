from __future__ import annotations

"""
Propósito:
Exponer el contrato de configuración al paquete API con import relativo limpio.

Firma técnica:
- Reexporta ConfiguracionRutas, ConfiguracionAPI y ConfiguracionLogs.

Lógica resumida:
- Actúa como puente estable entre `api/main.py` y `config.py`.

Caso de error principal:
- Si `config.py` falla al cargar, Python propagará ImportError de forma explícita.
"""

from config import (
    MARGEN_INCERTIDUMBRE,
    UMBRAL_RIESGO_ALTO,
    UMBRAL_RIESGO_BAJO,
    ConfiguracionAPI,
    ConfiguracionLogs,
    ConfiguracionRutas,
)

__all__ = [
    "ConfiguracionRutas",
    "ConfiguracionAPI",
    "ConfiguracionLogs",
    "UMBRAL_RIESGO_BAJO",
    "UMBRAL_RIESGO_ALTO",
    "MARGEN_INCERTIDUMBRE",
]
