"""
Validacion de prompt (paso previo a la Fase 4).

Compara varios prompts candidatos usando Qwen3-VL-2B-Instruct SIN
fine-tuning: extrae el hidden state de la ultima capa / ultimo token
para una muestra estratificada del dataset, y mide la separabilidad
de las dos clases con un clasificador lineal simple sobre esas
representaciones congeladas. El prompt con mejor F1 aqui es, con alta
probabilidad, el que mejor va a fine-tunear despues.

No se entrena ni se toca ningun peso del modelo. Coste: minutos, no
dias. Requiere GPU (T4 gratuito de Colab es suficiente) y una version
reciente de transformers:
    pip install -U "transformers>=4.57" accelerate
    # si Qwen3VLForConditionalGeneration no se encuentra:
    pip install git+https://github.com/huggingface/transformers
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_JSON = Path("data/final/dataset.json")
OUTPUT_REPORT = Path("reports/prompt_validation.json")

N_SAMPLE_PER_CLASS = 60   # muestra pequena a proposito: esto NO es el entrenamiento final
N_CV_FOLDS = 5
SEED = 42

SYSTEM_PROMPT = (
    "Eres un sistema de inspeccion de calidad en una fabrica textil, "
    "experto en detectar defectos de fabricacion."
)
DESCRIPTIVE_QUESTION = (
    "Analiza la textura de este tejido industrial. ¿Se observa algun "
    "defecto de fabricacion, como hilos sueltos, manchas, roturas, "
    "agujeros o irregularidades del tejido?"
)

PROMPT_CANDIDATES = {
    "A_minimo": {
        "system": None,
        "user": "¿Este tejido tiene algun defecto?",
    },
    "B_descriptivo": {
        "system": None,
        "user": DESCRIPTIVE_QUESTION,
    },
    "C_con_rol": {
        "system": SYSTEM_PROMPT,
        "user": DESCRIPTIVE_QUESTION,
    },
}


def load_stratified_sample() -> pd.DataFrame:
    records = json.loads(DATASET_JSON.read_text())
    df = pd.DataFrame(records)
    df = df[df["split"] == "trainval"]  # el test jamas se toca para esto tampoco

    min_class_count = df["label"].value_counts().min()
    n_per_class = min(N_SAMPLE_PER_CLASS, min_class_count)
    if n_per_class < N_SAMPLE_PER_CLASS:
        print(f"Aviso: la clase minoritaria solo tiene {min_class_count} imagenes en "
              f"trainval; se reduce la muestra a {n_per_class} por clase.")

    sample = (
        df.groupby("label", group_keys=False)
        .sample(n=n_per_class, random_state=SEED)
        .reset_index(drop=True)
    )
    return sample


def build_messages(prompt_cfg: dict, image: Image.Image) -> list[dict]:
    messages = []
    if prompt_cfg["system"]:
        messages.append({"role": "system", "content": [{"type": "text", "text": prompt_cfg["system"]}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_cfg["user"]},
        ],
    })
    return messages


@torch.no_grad()
def extract_hidden_state(model, processor, messages: list[dict]) -> np.ndarray:
    """Un solo forward pass (sin generar texto), guardando el hidden state
    de la ultima capa en la ultima posicion de la secuencia. Se procesa
    una imagen a la vez (batch=1) para que el indice 'ultima posicion'
    sea siempre correcto sin tener que lidiar con padding."""
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][:, -1, :]
    return last_hidden.float().cpu().numpy().squeeze(0)


def evaluate_prompt(model, processor, prompt_name: str, prompt_cfg: dict, sample_df: pd.DataFrame) -> dict:
    features, labels, groups = [], [], []
    for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc=prompt_name):
        image = Image.open(row.path).convert("RGB")
        messages = build_messages(prompt_cfg, image)
        feat = extract_hidden_state(model, processor, messages)
        features.append(feat)
        labels.append(row.label_id)
        groups.append(row.group_id)

    X = np.stack(features)
    y = np.array(labels)
    groups = np.array(groups)

    n_splits = min(N_CV_FOLDS, pd.Series(groups).nunique(), int(np.bincount(y).min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="f1")

    return {
        "prompt_name": prompt_name,
        "prompt_text": prompt_cfg["user"],
        "system_prompt": prompt_cfg["system"],
        "n_samples": len(sample_df),
        "n_cv_folds_used": n_splits,
        "f1_mean": float(scores.mean()),
        "f1_std": float(scores.std()),
        "f1_per_fold": scores.tolist(),
    }


def main():
    print(f"Cargando {MODEL_ID} (congelado, sin fine-tuning)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sample_df = load_stratified_sample()
    print(f"Muestra: {len(sample_df)} imagenes ({sample_df['label'].value_counts().to_dict()})")

    results = []
    for name, cfg in PROMPT_CANDIDATES.items():
        result = evaluate_prompt(model, processor, name, cfg, sample_df)
        results.append(result)
        print(f"{name}: F1 = {result['f1_mean']:.3f} +/- {result['f1_std']:.3f} "
              f"({result['n_cv_folds_used']} folds)")

    results.sort(key=lambda r: r["f1_mean"], reverse=True)
    print(f"\nMejor prompt: {results[0]['prompt_name']} (F1={results[0]['f1_mean']:.3f})")
    print("Usa ese texto de prompt, con su chat template correspondiente, en la Fase 4.")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Informe guardado en {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
