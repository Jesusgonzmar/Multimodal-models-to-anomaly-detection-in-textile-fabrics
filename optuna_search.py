"""
Fase 6 - Busqueda de hiperparametros con Optuna + K-fold.

Orquesta llamadas repetidas a train_one_fold (Fase 5): para cada trial
de Optuna, prueba una combinacion de hiperparametros sobre TODOS los
folds de data/final/dataset.json (split='trainval') y devuelve la
media de F1 como objetivo a maximizar. El test, apartado desde la
Fase 2, no se toca aqui.

El estudio se guarda en SQLite desde el primer trial: si Colab
desconecta a mitad de la busqueda, volver a ejecutar este script
continua donde lo dejo, sin repetir trials ya completados.

Verificado por ejecucion en este entorno (sin GPU, con el backbone
Llama diminuto ya validado en las Fases 4-5):
  1. El bucle completo (Optuna sugiere hiperparametros -> K-fold ->
     train_one_fold -> media de F1) corre de principio a fin sin
     errores, sobre datos reales del dataset.json de prueba.
  2. El pruning entre folds dispara correctamente cuando se fuerza un
     trial con F1 malo a proposito, y el trial queda marcado como
     PRUNED en el estudio, sin contar para el mejor resultado.
  3. La persistencia en SQLite sobrevive a "reabrir" el estudio desde
     cero (nueva conexion al mismo archivo .db): los trials ya
     completados siguen ahi y no se repiten.

NO verificado (requiere GPU + Qwen3-VL real): el tiempo real por
fold/trial. Los valores por defecto de N_TRIALS/MAX_EPOCHS son un
punto de partida deliberadamente conservador, no una cifra medida -
ajustalos con datos reales de tiempo tras las primeras ejecuciones.
"""
import gc
import json
from pathlib import Path

import numpy as np
import optuna
import torch

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_JSON = Path("data/final/dataset.json")
STUDY_DB = Path("models/optuna_study.db")
CHECKPOINT_ROOT = Path("models/optuna_trials")
REPORT_PATH = Path("reports/optuna_best_params.json")

N_TRIALS = 8      # punto de partida conservador, ajustar tras medir tiempo real en Colab
MAX_EPOCHS = 8
PATIENCE = 2


def load_trainval_folds(dataset_json: Path = DATASET_JSON) -> tuple[list[dict], list[int]]:
    records = json.loads(dataset_json.read_text())
    trainval = [r for r in records if r["split"] == "trainval"]
    fold_ids = sorted({r["fold"] for r in trainval})
    return trainval, fold_ids


def objective(trial, trainval_records, fold_ids, processor, device,
              model_factory, train_one_fold_fn, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """model_factory(lora_config) -> nn.Module nuevo (una instancia
    fresca por fold, para no arrastrar pesos entre folds). Se inyecta
    como parametro para poder probar esta funcion con un modelo de
    prueba, sin acoplarla a Qwen3-VL en concreto."""
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = lora_r * 2
    use_dora = trial.suggest_categorical("use_dora", [True, False])
    lr_lora = trial.suggest_float("lr_lora", 1e-5, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-3, log=True)

    fold_f1s = []
    for fold_idx in fold_ids:
        train_records = [r for r in trainval_records if r["fold"] != fold_idx]
        val_records = [r for r in trainval_records if r["fold"] == fold_idx]

        model = model_factory(lora_r=lora_r, lora_alpha=lora_alpha, use_dora=use_dora)
        ckpt_dir = CHECKPOINT_ROOT / f"trial{trial.number}_fold{fold_idx}"

        try:
            result = train_one_fold_fn(
                model, train_records, val_records, processor,
                checkpoint_dir=ckpt_dir, device=device,
                lr_lora=lr_lora, lr_head=lr_head,
                max_epochs=max_epochs, patience=patience,
            )
        finally:
            # Si train_one_fold_fn lanza una excepcion real (OOM, bug),
            # esto se ejecuta igual y la VRAM no se queda huerfana. El
            # pruning de Optuna, en cambio, ya ocurre DESPUES de este
            # bloque, no necesita esta proteccion.
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        fold_f1s.append(result["best_val_f1"])

        running_mean = float(np.mean(fold_f1s))
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


def run_study(objective_fn, n_folds: int, n_trials: int = N_TRIALS, study_db: Path = STUDY_DB,
              study_name: str = "textile_defect_qwen3vl") -> optuna.Study:
    study_db.parent.mkdir(parents=True, exist_ok=True)
    # HyperbandPruner: la documentacion oficial de Optuna reporta que supera a
    # MedianPruner en sus propios benchmarks internos. IMPORTANTE: max_resource
    # debe coincidir con la granularidad real de trial.report() en objective(),
    # que aqui es "step=fold_idx" (numero de folds completados), NO epochs -
    # por eso max_resource=n_folds, no max_epochs.
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{study_db}",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_folds, reduction_factor=3),
    )
    study.optimize(objective_fn, n_trials=n_trials)
    return study


def save_best_params(study: optuna.Study, report_path: Path = REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }, indent=2))


if __name__ == "__main__":
    from transformers import AutoProcessor

    from src.models.classifier import TextileDefectClassifier, build_lora_config
    from src.training.train_fold import train_one_fold

    trainval_records, fold_ids = load_trainval_folds()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def model_factory(lora_r, lora_alpha, use_dora=False):
        lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha, use_dora=use_dora)
        return TextileDefectClassifier(model_id=MODEL_ID, lora_config=lora_cfg)

    study = run_study(
        lambda trial: objective(trial, trainval_records, fold_ids, processor, device,
                                 model_factory, train_one_fold),
        n_folds=len(fold_ids), n_trials=N_TRIALS,
    )
    save_best_params(study)
    print("Mejor trial:", study.best_trial.number)
    print("Mejores hiperparametros:", study.best_params)
    print("Mejor F1 medio (CV):", study.best_value)
