from pathlib import Path
import pandas as pd
from config import RUTA_DATASET_PREDETERMINADA, DATOS_BRUTOS_DIR


def main():
    ruta = RUTA_DATASET_PREDETERMINADA
    df = pd.read_csv(ruta)
    # estratificar por columna objetivo
    objetivo = "Diabetes_binary"
    if objetivo not in df.columns:
        raise RuntimeError(f"Columna objetivo {objetivo} no encontrada en {ruta}")

    n = 2000
    # calcular fracciones por clase para obtener aproximadamente n estratificado
    muestras = []
    for clase, sub in df.groupby(objetivo):
        prop = len(sub) / len(df)
        k = max(1, int(round(prop * n)))
        muestras.append(sub.sample(n=k, random_state=42))

    muestra = pd.concat(muestras).sample(frac=1, random_state=42).reset_index(drop=True)
    # ajustar tamaño exacto a 2000 si es necesario
    if len(muestra) > n:
        muestra = muestra.iloc[:n]
    elif len(muestra) < n:
        restante = n - len(muestra)
        extra = df.drop(muestra.index).sample(n=restante, random_state=42)
        muestra = pd.concat([muestra, extra]).reset_index(drop=True)

    salida = DATOS_BRUTOS_DIR / "muestra_2000.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(salida, index=False)
    prevalencia = float(muestra[objetivo].mean())
    print(f"Muestra generada en: {salida}")
    print(f"Tamaño muestra: {len(muestra)}, prevalencia clase 1: {prevalencia:.6f}")


if __name__ == "__main__":
    main()
