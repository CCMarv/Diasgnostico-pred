# Script para crear/actualizar entorno y ejecutar la muestra mínima
# Ejecutar en PowerShell (Windows)

# Nombre del entorno Conda
$envName = 'diagnostico-pred'

function Has-Conda {
    try {
        conda --version > $null 2>&1
        return $true
    } catch {
        return $false
    }
}

if (Has-Conda) {
    Write-Host "Conda detectado. Creando/actualizando entorno: $envName"
    try {
        conda env create -f environment.yml -y
    } catch {
        Write-Host "El entorno puede existir ya; intentando actualizar..."
        conda env update -f environment.yml
    }
    Write-Host "Activa el entorno con: conda activate $envName"
    Write-Host "Si quieres que el script lo active automáticamente, abre PowerShell y ejecuta: conda activate $envName"
} else {
    Write-Host "Conda no detectado. Creando virtualenv local .venv"
    python -m venv .venv
    Write-Host "Activa el entorno virtual con: .\.venv\Scripts\Activate.ps1"
}

Write-Host "Instalando paquete en modo editable con extras 'dev' (pip -e .[dev])"
# No forzamos la activación automática; el usuario debe activar el entorno antes de ejecutar la línea siguiente
Write-Host "Una vez activado el entorno, ejecuta: pip install -e .[dev]"

Write-Host "Comando para ejecutar la muestra mínima (ejecuta tras activar el entorno y haber instalado deps):"
Write-Host "python -m entrenamiento.pipeline --modo clasificacion --modelos gbm --salida-modelo modelos/modelo_prueba_minima.joblib --salida-reporte reportes/prueba_minima.json"

Write-Host "Después de ejecutar el pipeline, verifica los artefactos:
Test-Path modelos\modelo_prueba_minima.joblib
Test-Path reportes\prueba_minima.json"
