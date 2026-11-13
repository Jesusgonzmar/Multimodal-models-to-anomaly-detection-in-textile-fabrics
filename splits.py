"""
Fase 2 - Division del dataset en TEST (apartado, fijo desde el
principio) y TRAIN+VAL POOL con K folds estratificados y conscientes
de grupos (para que copias/casi-duplicados nunca queden repartidos
entre train y test, ni entre train y validacion de un fold).

Requiere haber ejecutado antes audit_dataset.py.

Salida:
  - data/splits/test.json          (NO se vuelve a tocar hasta evaluar)
  - data/splits/trainval_folds.json (cada fila lleva su numero de fold)
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_TEST_SPLITS = 5   # 1 de cada 5 folds ~ 20% para test
N_CV_FOLDS = 5
MANIFEST = Path("data/processed/manifest.csv")
DUP_CLUSTERS = Path("data/processed/near_duplicate_clusters.json")
OUT_DIR = Path("data/splits")


def assign_group_ids(df: pd.DataFrame, dup_clusters: dict) -> pd.DataFrame:
    """Por defecto cada imagen es su propio grupo. Las imagenes que
    la auditoria detecto como casi-duplicadas comparten group_id para
    que nunca se separen entre splits."""
    group_id = {img_id: img_id for img_id in df["image_id"]}
    for representative, members in dup_clusters.items():
        for m in members:
            group_id[m] = representative
    df = df.copy()
    df["group_id"] = df["image_id"].map(group_id)
    return df


def carve_out_test(df: pd.DataFrame):
    sgkf = StratifiedGroupKFold(n_splits=N_TEST_SPLITS, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["group_id"]))
    return df.iloc[trainval_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def assign_cv_folds(trainval_df: pd.DataFrame) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    trainval_df = trainval_df.copy()
    trainval_df["fold"] = -1
    splits = sgkf.split(trainval_df, trainval_df["label"], groups=trainval_df["group_id"])
    for fold, (_, val_idx) in enumerate(splits):
        trainval_df.loc[trainval_df.index[val_idx], "fold"] = fold
    return trainval_df


def assert_no_group_leakage(test_df: pd.DataFrame, trainval_df: pd.DataFrame) -> None:
    overlap = set(test_df["group_id"]) & set(trainval_df["group_id"])
    assert not overlap, f"FUGA DE DATOS: grupos compartidos entre test y trainval: {overlap}"
    print("Comprobacion de fuga de datos (test vs trainval): OK, sin solapamiento")


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}: {len(df)} imagenes, {df['group_id'].nunique()} grupos")
    print(df["label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    df = pd.read_csv(MANIFEST)
    dup_clusters = json.loads(DUP_CLUSTERS.read_text())
    df = assign_group_ids(df, dup_clusters)

    trainval_df, test_df = carve_out_test(df)
    trainval_df = assign_cv_folds(trainval_df)
    assert_no_group_leakage(test_df, trainval_df)

    report("TEST (apartado)", test_df)
    report("TRAIN+VAL POOL", trainval_df)
    for f in range(N_CV_FOLDS):
        report(f"  fold {f} (validacion)", trainval_df[trainval_df.fold == f])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_json(OUT_DIR / "test.json", orient="records", indent=2, force_ascii=False)
    trainval_df.to_json(OUT_DIR / "trainval_folds.json", orient="records", indent=2, force_ascii=False)
    print(f"\nGuardado en {OUT_DIR}/test.json y {OUT_DIR}/trainval_folds.json")
