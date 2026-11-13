"""
Fase 3 - Preprocesado final y generacion del JSON de entrenamiento.

Toma el manifest de la Fase 1 y las particiones de la Fase 2, cachea
una copia redimensionada de cada imagen valida (solo si hace falta,
nunca se agranda una imagen pequena) y construye el JSON definitivo
que consumira el Dataset de entrenamiento en la Fase 4.

A partir de aqui, ningun script posterior debe leer directamente de
data/raw/: todo pasa por data/final/dataset.json.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

TEST_JSON = Path("data/splits/test.json")
TRAINVAL_JSON = Path("data/splits/trainval_folds.json")
PROCESSED_IMG_DIR = Path("data/processed/images")
OUTPUT_JSON = Path("data/final/dataset.json")

MAX_LONG_SIDE = 1280  # cap de resize; si la imagen ya es mas pequena, no se toca
JPEG_QUALITY = 90
LABEL2ID = {"OK": 1, "NO_OK": 0}


def load_splits() -> pd.DataFrame:
    test_df = pd.read_json(TEST_JSON)
    test_df["split"] = "test"
    test_df["fold"] = -1

    trainval_df = pd.read_json(TRAINVAL_JSON)
    trainval_df["split"] = "trainval"
    # 'fold' (0..k-1) ya viene calculado de la Fase 2

    combined = pd.concat([test_df, trainval_df], ignore_index=True)
    combined["fold"] = combined["fold"].astype(int)
    return combined


def process_image(src_path: Path, dst_path: Path) -> tuple[int, int]:
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / long_side
            img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=JPEG_QUALITY)
        return img.size


def build_final_dataset(df: pd.DataFrame) -> list[dict]:
    records = []
    missing = 0
    for row in tqdm(df.itertuples(), total=len(df), desc="procesando imagenes"):
        src_path = Path(row.path)
        if not src_path.exists():
            print(f"[FALTA] {src_path} esta en los splits pero no existe en disco")
            missing += 1
            continue

        dst_path = PROCESSED_IMG_DIR / f"{row.image_id}.jpg"
        new_w, new_h = process_image(src_path, dst_path)

        records.append({
            "image_id": row.image_id,
            "path": str(dst_path),
            "label": row.label,
            "label_id": LABEL2ID[row.label],
            "group_id": row.group_id,
            "split": row.split,
            "fold": int(row.fold),
            "width": new_w,
            "height": new_h,
        })

    if missing:
        print(f"\nAVISO: {missing} imagenes listadas en los splits no se encontraron en disco.")
    return records


def validate(records: list[dict], df: pd.DataFrame) -> None:
    assert len(records) > 0, "No se ha procesado ninguna imagen"

    n_expected, n_got = len(df), len(records)
    if n_got != n_expected:
        print(f"AVISO: se esperaban {n_expected} imagenes segun los splits, se han procesado {n_got}")
    else:
        print(f"Comprobacion de conteo: OK ({n_got} imagenes, coincide con la Fase 2)")

    for r in records:
        assert Path(r["path"]).exists(), f"Ruta procesada no existe: {r['path']}"
        assert max(r["width"], r["height"]) <= MAX_LONG_SIDE, f"Resize no aplicado correctamente en {r['image_id']}"
    print("Comprobacion de archivos procesados y limite de resolucion: OK")


def summarize(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    print("\n=== RESUMEN FASE 3 ===")
    print(df.groupby("split")["label"].value_counts())
    long_sides = df[["width", "height"]].max(axis=1)
    print(f"\nLado mayor tras el resize: min={long_sides.min()} max={long_sides.max()} media={long_sides.mean():.0f}")


if __name__ == "__main__":
    splits_df = load_splits()
    records = build_final_dataset(splits_df)
    validate(records, splits_df)
    summarize(records)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nJSON final guardado en: {OUTPUT_JSON}")
