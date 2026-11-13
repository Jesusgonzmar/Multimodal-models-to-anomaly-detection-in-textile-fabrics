"""
Fase 1 - Auditoria del dataset.

Recorre data/raw/{ok,no_ok}, valida que cada imagen se pueda abrir,
calcula metadatos (resolucion, tamano) y un hash perceptual (phash)
para detectar posibles duplicados o casi-duplicados ANTES de generar
el JSON final o hacer cualquier split.

Salida:
  - data/processed/manifest.csv
  - data/processed/near_duplicate_clusters.json
  - resumen impreso por pantalla
"""
import json
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CLASS_FOLDERS = {"ok": "OK", "no_ok": "NO_OK"}  # ajusta si tus carpetas se llaman distinto
OUTPUT_MANIFEST = Path("data/processed/manifest.csv")
OUTPUT_DUPLICATES = Path("data/processed/near_duplicate_clusters.json")
DUPLICATE_HASH_DISTANCE = 5  # phash distance <= 5 => probablemente casi-duplicado
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images() -> pd.DataFrame:
    records = []
    for folder, label in CLASS_FOLDERS.items():
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"No existe {folder_path}; revisa RAW_DIR/CLASS_FOLDERS")

        for img_path in tqdm(sorted(folder_path.glob("*")), desc=folder):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()  # detecta archivos corruptos/truncados
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    phash = str(imagehash.phash(img))
            except Exception as e:
                print(f"[CORRUPTO - excluido] {img_path}: {e}")
                continue

            records.append({
                "image_id": img_path.stem,
                "path": str(img_path),
                "label": label,
                "width": width,
                "height": height,
                "file_size_kb": round(img_path.stat().st_size / 1024, 1),
                "phash": phash,
            })

    if not records:
        raise RuntimeError("No se ha leido ninguna imagen valida. Revisa las rutas.")
    return pd.DataFrame(records)


def find_near_duplicates(df: pd.DataFrame, max_distance: int = DUPLICATE_HASH_DISTANCE) -> dict:
    """Agrupamiento simple por distancia de Hamming entre phashes.
    No es un clustering exhaustivo, pero es suficiente para una
    primera pasada de revision manual."""
    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    clusters = {}
    visited = set()
    for i in range(len(hashes)):
        if i in visited:
            continue
        cluster = [i]
        for j in range(i + 1, len(hashes)):
            if j in visited:
                continue
            if hashes[i] - hashes[j] <= max_distance:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters[df.iloc[i]["image_id"]] = df.iloc[cluster]["image_id"].tolist()
            visited.add(i)
    return clusters


def summarize(df: pd.DataFrame, dup_clusters: dict) -> None:
    print("\n=== RESUMEN DE LA AUDITORIA ===")
    print(df["label"].value_counts())
    print(f"\nResolucion - ancho: min={df.width.min()} max={df.width.max()} media={df.width.mean():.0f}")
    print(f"Resolucion - alto : min={df.height.min()} max={df.height.max()} media={df.height.mean():.0f}")
    print(f"\nPosibles grupos de casi-duplicados: {len(dup_clusters)}")
    total_affected = sum(len(v) for v in dup_clusters.values())
    print(f"Imagenes implicadas: {total_affected} / {len(df)}")
    if dup_clusters:
        print("\nEjemplos (hasta 5 grupos):")
        for k, v in list(dup_clusters.items())[:5]:
            print(f"  {k} ~ {v}")


if __name__ == "__main__":
    df = scan_images()
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_MANIFEST, index=False)

    dup_clusters = find_near_duplicates(df)
    OUTPUT_DUPLICATES.write_text(json.dumps(dup_clusters, indent=2, ensure_ascii=False))

    summarize(df, dup_clusters)
    print(f"\nManifest guardado en: {OUTPUT_MANIFEST}")
    print(f"Duplicados guardados en: {OUTPUT_DUPLICATES}")
