"""
Fase 7 - Reentrenamiento final + evaluacion sobre el test (UNA sola vez).

Toma los mejores hiperparametros de la Fase 6, reentrena un unico
modelo sobre todo el pool trainval (con una pequena porcion nueva
reservada solo para el trigger de early-stopping de esta tanda final,
DISTINTA de los folds de la Fase 6), y evalua ese modelo, una sola
vez, contra el test apartado desde la Fase 2.

Salvaguarda deliberada, no solo documentada sino aplicada en codigo:
si reports/final_test_evaluation.json ya existe, el script se niega a
sobreescribirlo salvo --force explicito. Evaluar contra el test mas
de una vez para "ver si mejora si cambio algo" invalida su funcion
como estimador no sesgado.

Verificado por ejecucion en este entorno (sin GPU, con el backbone
diminuto ya validado en las Fases 4-6):
  1. El split final train/val (group-aware, estratificado) no filtra
     grupos entre las dos partes.
  2. compute_full_metrics(), comprobado contra una matriz de confusion
     calculada a mano, da precision/recall/F1/accuracy correctos.
  3. El pipeline completo (cargar mejores params -> split final ->
     entrenar -> evaluar UNA vez sobre test -> guardar informe) corre
     de principio a fin.
  4. La salvaguarda anti-doble-evaluacion: la primera ejecucion crea
     el informe; una segunda ejecucion sin --force lanza RuntimeError
     y NO sobreescribe nada; con --force si sobreescribe.

NO verificado (requiere GPU + Qwen3-VL real): el entrenamiento y la
inferencia reales sobre imagenes de tejidos.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

FINAL_VAL_FRACTION = 0.12  # solo para el trigger de early-stopping de ESTA tanda
SEED = 42
BEST_PARAMS_JSON = Path("reports/optuna_best_params.json")
DATASET_JSON = Path("data/final/dataset.json")
FINAL_CHECKPOINT_DIR = Path("models/final_model")
FINAL_REPORT_PATH = Path("reports/final_test_evaluation.json")
POSITIVE_CLASS = 0  # NO_OK: coherente con la Fase 5


def build_final_train_val_split(trainval_records: list[dict], val_fraction: float = FINAL_VAL_FRACTION,
                                 seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Split nuevo, no reutiliza los folds de la Fase 6. Mismo criterio
    group-aware que en la Fase 2, para no reintroducir fuga aqui."""
    df = pd.DataFrame(trainval_records)
    n_splits = max(2, round(1 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, df["label"], groups=df["group_id"]))
    return df.iloc[train_idx].to_dict("records"), df.iloc[val_idx].to_dict("records")


def load_best_params(path: Path = BEST_PARAMS_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}. Ejecuta primero la Fase 6.")
    return json.loads(path.read_text())["best_params"]


def compute_full_metrics(y_true: list[int], y_pred: list[int], positive_label: int = POSITIVE_CLASS) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0,
    )
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": accuracy,
        "f1_defect_class": float(f1),
        "precision_defect_class": float(precision),
        "recall_defect_class": float(recall),
        "confusion_matrix": {
            "orden_filas_y_columnas": ["NO_OK(0)", "OK(1)"],
            "nota": "filas = etiqueta real, columnas = prediccion. "
                    "matrix[0][1] = defectos reales marcados como OK (el error caro).",
            "matrix": cm,
        },
        "n_samples": len(y_true),
    }


def run_final_phase(trainval_records, test_records, processor, model_factory, bare_model_factory,
                     train_one_fold_fn, evaluate_fn, dataset_cls, collate_fn_builder, load_checkpoint_fn,
                     device, force: bool = False,
                     max_epochs: int = 15, patience: int = 3) -> dict:
    if FINAL_REPORT_PATH.exists() and not force:
        raise RuntimeError(
            f"{FINAL_REPORT_PATH} ya existe. El test se evalua UNA sola vez por diseno. "
            "Si de verdad hay que repetirlo (p.ej. tras corregir un bug real, no para "
            "'probar a ver si mejora'), pasa force=True y documenta el motivo."
        )

    best_params = load_best_params()
    final_train, final_val = build_final_train_val_split(trainval_records)
    print(f"Entrenamiento final: {len(final_train)} train / {len(final_val)} val "
          f"(reservado solo para early-stopping de esta tanda)")

    model = model_factory(lora_r=best_params["lora_r"], lora_alpha=best_params["lora_r"] * 2,
                           use_dora=best_params.get("use_dora", False))
    result = train_one_fold_fn(
        model, final_train, final_val, processor,
        checkpoint_dir=FINAL_CHECKPOINT_DIR, device=device,
        lr_lora=best_params["lr_lora"], lr_head=best_params["lr_head"],
        max_epochs=max_epochs, patience=patience,
    )
    print(f"Entrenamiento final completado. F1 en el val de early-stopping (ULTIMO epoch "
          f"entrenado): {result['history'][-1]['val_f1']:.4f}")
    print(f"Mejor F1 durante el entrenamiento (el que se guardo en disco): {result['best_val_f1']:.4f}")

    # CRITICO: 'model' en memoria refleja el ULTIMO epoch entrenado, que
    # puede ser peor que el mejor por el propio funcionamiento del
    # early-stopping (para 'patience' epochs despues de la mejora). Se
    # descarta y se reconstruye un modelo fresco a partir del MEJOR
    # checkpoint guardado en disco antes de tocar el test.
    del model
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    model = bare_model_factory()
    model = load_checkpoint_fn(model, FINAL_CHECKPOINT_DIR, device)
    print(f"Modelo recargado desde el MEJOR checkpoint en {FINAL_CHECKPOINT_DIR} para la evaluacion.")

    print(f"\nEvaluando UNA VEZ sobre el test ({len(test_records)} imagenes)...")
    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        dataset_cls(test_records), batch_size=4, shuffle=False,
        collate_fn=collate_fn_builder(processor),
    )
    _, preds, labels = evaluate_fn(model, test_loader, device)
    metrics = compute_full_metrics(labels, preds)

    report = {
        "best_params_used": best_params,
        "final_train_size": len(final_train),
        "final_early_stop_val_size": len(final_val),
        "early_stop_val_f1": result["best_val_f1"],
        "test_metrics": metrics,
    }
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nInforme final (UNICO) guardado en {FINAL_REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Sobreescribe una evaluacion de test previa. Usar con mucho cuidado.")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor

    from src.models.classifier import TextileDefectClassifier, build_lora_config
    from src.training.train_fold import (TextileDataset, build_collate_fn, evaluate,
                                          load_checkpoint, train_one_fold)

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    records = json.loads(DATASET_JSON.read_text())
    trainval_records = [r for r in records if r["split"] == "trainval"]
    test_records = [r for r in records if r["split"] == "test"]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_factory(lora_r, lora_alpha, use_dora=False):
        lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha, use_dora=use_dora)
        return TextileDefectClassifier(model_id=MODEL_ID, lora_config=lora_cfg)

    def bare_model_factory():
        # SIN lora_config: el backbone debe quedar sin envolver para que
        # load_checkpoint pueda aplicarle PeftModel.from_pretrained encima.
        return TextileDefectClassifier(model_id=MODEL_ID, lora_config=None)

    run_final_phase(trainval_records, test_records, processor, model_factory, bare_model_factory,
                     train_one_fold, evaluate, TextileDataset, build_collate_fn, load_checkpoint,
                     device, force=args.force)
