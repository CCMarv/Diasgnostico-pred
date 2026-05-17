---
Autor: Equipo diasgnostico-pred
Fecha: 2026-05-17
Versión: 0.1.0
---

# Guía de documentación y checklist para PRs

Objetivo: mantener documentación sincronizada con el código y accesible para lectores técnicos y no técnicos.

- **Quién edita qué**: cambios de diseño → `docs/DECISIONES_DISENO.md` y `PROYECTO.md`. Instrucciones de uso → `README.md`.
- **Marcar artefactos generados**: todos los archivos en `reportes/` deben incluir una cabecera indicando que son generados por el pipeline y no deben editarse manualmente.

## Single Source of Truth (SST)

Las siguientes ubicaciones son las fuentes canónicas para cada tipo de información. Actualizar la documentación solo después de cambiar el SST correspondiente.

- Constantes, rutas y umbrales: `config.py`
- Catálogo de modelos y parámetros por defecto: `entrenamiento/comparador_modelos.py`
- Orquestación del pipeline y argumentos CLI: `entrenamiento/pipeline.py`
- Contrato público de la API y validación: `api/esquemas.py` y `inferencia/predictor.py`

## Estructura de los documentos de modelo

- Resumen ejecutivo (1-2 párrafos) para no-programadores.
- Parámetros clave (tabla): nombre, valor en código, explicación en lenguaje llano.
- Procedimiento de entrenamiento (comando exacto).
- Artefactos producidos (rutas).

## Estilo

- Escribir en español.
- Usar nombres técnicos como `COLUMNAS_CDC` en código (monoespaciado) cuando se refieran a símbolos del proyecto.
- Evitar contradicciones: si un comportamiento es automático en código, documentarlo como tal; si requiere acción del usuario, indicar comandos exactos.

## Checklist para PR de documentación (manual)

Antes de marcar la PR lista para merge, completar esta verificación manual:

- [ ] `README.md` y `PROYECTO.md` describen correctamente los comandos y el flujo.
- [ ] `docs/DECISIONES_DISENO.md` refleja cualquier cambio de diseño o decisión arquitectónica.
- [ ] Se añadieron notas de compatibilidad (Python/paquetes) cuando aplica.
- [ ] Archivos generados en `reportes/` contienen cabecera `ARTEFACTO GENERADO`.
- [ ] Documentos de modelo incluyen la sección para no-programadores.
- [ ] Se verificaron enlaces y referencias a `config.py`, `entrenamiento/comparador_modelos.py`, `entrenamiento/pipeline.py` y `api/esquemas.py`.
- [ ] Las instrucciones de entrenamiento incluyen el comando exacto y ejemplo de `--dataset` cuando aplica.
- [ ] Se corrieron localmente las pruebas relevantes: `pytest -q pruebas/test_cargador.py pruebas/test_predictor.py pruebas/test_preprocesador.py` (si no pasan, documentar fallo en la PR).

## Plantilla de PR para documentación

Título: `docs: <breve-descripción>`

Descripción (obligatoria):

1. Resumen breve del cambio.
2. Archivos modificados.
3. Qué SST fue actualizado (si aplica).
4. Pasos de validación manual realizados (lista de checks completados).

Checklist adjunta (completar en la PR):

- [ ] Revisado por un colega técnico.
- [ ] Links verificados.
- [ ] Comandos de ejemplo copiados y ejecutados localmente.

## Validación manual (pasos recomendados)

1. Abrir `docs/DECISIONES_DISENO.md` y confirmar metadata (Autor/Fecha/ID) y que la decisión relevante aparece en la tabla.
2. Abrir `docs/DOCUMENTATION_GUIDELINES.md` y confirmar que el SST apunta a los archivos correctos.
3. Probar el flujo de ejemplo: ejecutar (si corresponde)

```powershell
conda activate diagnostico-pred
# ejemplo: ejecutar solo el pipeline con dataset local
python -m entrenamiento.pipeline --modo clasificacion --dataset datos/brutos/diabetes_binary_health_indicators_BRFSS2015.csv
```

4. Ejecutar pruebas relevantes (manual):

```powershell
pytest -q pruebas/test_cargador.py pruebas/test_predictor.py pruebas/test_preprocesador.py
```

5. Verificar que `reportes/` contiene los artefactos esperados y que cada archivo incluye la cabecera `ARTEFACTO GENERADO`.

## Alcance excluido

- No se incluirán pruebas automatizadas de documentación en esta fase (el equipo pidió validación manual).
- Cambios que alteren contratos públicos (API, `config.py`) deben ir acompañados de RFC en `docs/` y aprobaciones separadas.

Mantener esta guía actualizada dentro del repositorio.
