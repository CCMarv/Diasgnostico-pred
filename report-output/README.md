# Reproducción del reporte (VS Code)

## 1) Abrir el proyecto
1. Abrir la carpeta raíz del repositorio en VS Code.
2. Verificar que exista `report-output/report.md`.

## 2) Preparar entorno
```bash
pip install -e .[dev]
```

## 3) Validar estado del proyecto
```bash
python -m pytest
```

## 4) Regenerar artefactos base (opcional)
Usar artefactos existentes:
- `resultados/corrida_10k/corrida_10k.json`
- `resultados/corrida_50k/corrida_50k.json`

Si se desea regenerar corridas:
```bash
python scripts/ejecutar_corridas.py
```

## 5) Regenerar apoyo del reporte (figuras + CSV)
```bash
mkdir -p report-output/figures report-output/notebooks_processed report-output/deploy
cp resultados/corrida_50k/curvas_gbm.png report-output/figures/curvas_gbm_50k.png
cp resultados/corrida_50k/calibracion_gbm.png report-output/figures/calibracion_gbm_50k.png
cp resultados/corrida_10k/curvas_svm.png report-output/figures/curvas_svm_10k.png
cp resultados/corrida_10k/calibracion_svm.png report-output/figures/calibracion_svm_10k.png
```

## 6) Editar reporte
- Archivo principal editable: `report-output/report.md`
- Evidencia usada: `report-output/evidence.log`
- Métricas resumidas: `report-output/metrics.csv`

## 7) Componentes de despliegue (referencias)
- Dashboard: `dashboard/app.py`
- API: `api/main.py`
- Enlaces rápidos: `report-output/deploy/links.md`
