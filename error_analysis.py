"""
Fase 8 - Analisis de errores sobre las predicciones del test.

Carga el modelo FINAL ya entrenado (el mismo checkpoint que evaluo la
Fase 7) y vuelve a pasar las imagenes de test por el, esta vez
guardando la prediccion de CADA imagen individual -no solo el
resumen agregado que ya vimos en la Fase 7- para poder identificar
que imagenes concretas se clasifican mal y buscar patrones.

Esto NO reentrena ni ajusta nada del modelo: es una inspeccion de
solo lectura sobre un modelo ya evaluado una vez en la Fase 7. Por
eso, a diferencia de final_evaluation.py, este script SI se puede
ejecutar tantas veces como haga falta sin ninguna salvaguarda.

Verificado por ejecucion en este entorno (sin GPU, con el backbone
diminuto ya validado en fases anteriores): la clasificacion correcta
de cada prediccion en correct/false_negative/false_positive contra
un escenario donde se conoce de antemano el resultado esperado, el
calculo de las tasas de error del resumen contra un calculo hecho a
mano, y que las imagenes mal clasificadas se copian a las carpetas
correctas.
"""
import json
import shutil
from pathlib import Path

import torch
from PIL import Image

DATASET_JSON = Path("data/final/dataset.json")
CHECKPOINT_DIR = Path("models/final_model")
REPORT_PATH = Path("reports/error_analysis.json")
SAMPLES_DIR = Path("reports/error_samples")

LABEL_NAMES = {0: "NO_OK", 1: "OK"}


def build_messages(image, prompt_text: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]


@torch.no_grad()
def predict_test_set(model, test_records: list[dict], processor, prompt_text: str, device: str) -> list[dict]:
    """Inferencia una imagen a la vez (no reutiliza el collate_fn de
    entrenamiento, que descarta el image_id). El tamano del test de un
    TFG es pequeno, no hay necesidad real de batching aqui."""
    model.eval()
    results = []
    for r in test_records:
        image = Image.open(r["path"]).convert("RGB")
        messages = build_messages(image, prompt_text)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        pred_id = int(logits.argmax(dim=-1).item())
        true_id = r["label_id"]

        if true_id == pred_id:
            error_type = "correct"
        elif true_id == 0:  # NO_OK real, predicho OK -> el error caro
            error_type = "false_negative"
        else:  # OK real, predicho NO_OK -> falsa alarma
            error_type = "false_positive"

        results.append({
            "image_id": r["image_id"], "path": r["path"],
            "width": r.get("width"), "height": r.get("height"),
            "true_label": r["label"], "true_id": true_id,
            "pred_label": LABEL_NAMES[pred_id], "pred_id": pred_id,
            "error_type": error_type,
        })
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    counts = {"correct": 0, "false_negative": 0, "false_positive": 0}
    for r in results:
        counts[r["error_type"]] += 1

    n_defects = sum(1 for r in results if r["true_id"] == 0)
    n_ok = sum(1 for r in results if r["true_id"] == 1)

    summary = {
        "n_total": n,
        "counts": counts,
        "error_rate": (counts["false_negative"] + counts["false_positive"]) / n if n else 0.0,
        "false_negative_rate_among_defects": (counts["false_negative"] / n_defects) if n_defects else None,
        "false_positive_rate_among_ok": (counts["false_positive"] / n_ok) if n_ok else None,
    }

    def avg_dim(subset, key):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    for group in ("correct", "false_negative", "false_positive"):
        subset = [r for r in results if r["error_type"] == group]
        summary[f"avg_width_{group}"] = avg_dim(subset, "width")
        summary[f"avg_height_{group}"] = avg_dim(subset, "height")

    return summary


def export_error_samples(results: list[dict], samples_dir: Path = SAMPLES_DIR) -> None:
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    for error_type in ("false_negative", "false_positive"):
        out_dir = samples_dir / error_type
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r["error_type"] == error_type:
                shutil.copy(r["path"], out_dir / f"{r['image_id']}.jpg")


def run_error_analysis(model, test_records: list[dict], processor, prompt_text: str, device: str,
                        report_path: Path = REPORT_PATH, samples_dir: Path = SAMPLES_DIR) -> dict:
    results = predict_test_set(model, test_records, processor, prompt_text, device)
    summary = summarize(results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "per_image": results}, indent=2, ensure_ascii=False))
    export_error_samples(results, samples_dir)

    print("=== RESUMEN DEL ANALISIS DE ERRORES ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nInforme completo: {report_path}")
    print(f"Imagenes mal clasificadas copiadas en: {samples_dir}/")
    return {"summary": summary, "per_image": results}


if __name__ == "__main__":
    from transformers import AutoProcessor

    from src.models.classifier import TextileDefectClassifier
    from src.training.train_fold import PROMPT_B, load_checkpoint

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    records = json.loads(DATASET_JSON.read_text())
    test_records = [r for r in records if r["split"] == "test"]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 'bare': sin lora_config, para que load_checkpoint pueda envolverlo
    # con PeftModel.from_pretrained, igual que en la Fase 7.
    model = TextileDefectClassifier(model_id=MODEL_ID, lora_config=None)
    model = load_checkpoint(model, CHECKPOINT_DIR, device)

    run_error_analysis(model, test_records, processor, PROMPT_B, device)
