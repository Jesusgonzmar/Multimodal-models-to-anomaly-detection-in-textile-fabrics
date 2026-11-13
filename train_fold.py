"""
Fase 5 - Bucle de entrenamiento por fold.

Construye el Dataset, el collate_fn (que delega en el processor real
de Qwen3-VL el trabajo de padding/alineado de tokens de imagen, en
vez de reimplementarlo a mano) y el bucle de entrenamiento de UN fold,
con early-stopping sobre F1 de validacion y checkpointing del mejor
adaptador LoRA + cabezal.

Este script todavia NO hace K-fold ni Optuna (eso es la Fase 6): aqui
solo se entrena un fold dados unos indices de train/val concretos.

Verificado por ejecucion en este entorno (sin GPU, con el mismo
backbone Llama diminuto validado en la Fase 4): el bucle completo de
entrenamiento (optimizador, forward/backward, calculo de F1 de
validacion), el disparo correcto de early-stopping, y que el
checkpoint guardado se puede recargar y reproduce las mismas
predicciones que el modelo en memoria.

NO verificado por ejecucion (requiere GPU + Qwen3-VL real): el
comportamiento exacto de processor.apply_chat_template() con lotes
de imagenes reales de resolucion dinamica. Revisa el primer batch
manualmente en Colab (build_collate_fn incluye un modo de depuracion
para eso) antes de lanzar un entrenamiento largo.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

DATASET_JSON = Path("data/final/dataset.json")
PROMPT_B = (
    "Analiza la textura de este tejido industrial. ¿Se observa algun "
    "defecto de fabricacion, como hilos sueltos, manchas, roturas, "
    "agujeros o irregularidades del tejido?"
)
POSITIVE_CLASS_FOR_F1 = 0  # NO_OK: nos importa mas no dejar pasar un defecto


class TextileDataset(Dataset):
    """Devuelve, por cada imagen, la estructura de 'messages' ya lista
    para el chat template (con el prompt FIJO, el mismo para todas las
    imagenes) y la etiqueta. La imagen se carga aqui, no en el
    collate_fn, para aprovechar el paralelismo de num_workers>0."""

    def __init__(self, records: list[dict], prompt_text: str = PROMPT_B):
        self.records = records
        self.prompt_text = prompt_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        image = Image.open(r["path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt_text},
            ],
        }]
        return {"messages": messages, "label_id": r["label_id"], "image_id": r["image_id"]}


def build_collate_fn(processor, debug: bool = False):
    """El processor real hace el trabajo pesado (tokenizar + procesar
    imagenes + padding). padding_side='left' es lo que garantiza (ver
    Fase 4) que el ultimo token de la secuencia sea siempre el ultimo
    token real de cada muestra."""
    processor.tokenizer.padding_side = "left"

    def collate_fn(batch: list[dict]) -> tuple[dict, torch.Tensor]:
        messages_list = [item["messages"] for item in batch]
        inputs = processor.apply_chat_template(
            messages_list, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        labels = torch.tensor([item["label_id"] for item in batch])
        if debug:
            print("input_ids shape:", inputs["input_ids"].shape)
            print("attention_mask (primera fila):", inputs["attention_mask"][0].tolist())
        return inputs, labels

    return collate_fn


def compute_class_weights(records: list[dict]) -> torch.Tensor:
    labels = np.array([r["label_id"] for r in records])
    counts = np.bincount(labels, minlength=2)
    counts = np.maximum(counts, 1)  # evita division por cero si un fold pequeno se queda sin ejemplos de una clase
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    f1 = f1_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_FOR_F1, zero_division=0)
    return f1, all_preds, all_labels


def save_checkpoint(model: nn.Module, path: Path, val_f1: float, epoch: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "lora_adapter")  # peft: solo el adaptador, no el modelo completo
    torch.save(model.classifier.state_dict(), path / "classifier_head.pt")
    (path / "metrics.json").write_text(json.dumps({"val_f1": val_f1, "epoch": epoch}, indent=2))


def load_checkpoint_head(model: nn.Module, path: Path) -> nn.Module:
    """Recarga SOLO el cabezal (el adaptador LoRA se recarga aparte con
    PeftModel.from_pretrained; separado a proposito para poder probar
    cada pieza por separado)."""
    device = next(model.classifier.parameters()).device
    state = torch.load(path / "classifier_head.pt", map_location=device, weights_only=True)
    model.classifier.load_state_dict(state)
    return model


def load_checkpoint(model: nn.Module, path: Path, device: str = "cpu") -> nn.Module:
    """Recarga completa: adaptador LoRA + cabezal, sobre un modelo cuyo
    'model.backbone' es el backbone BASE sin envolver (es decir,
    construido con lora_config=None). Usar esto -no la instancia que
    quedo en memoria tras entrenar- para evaluar de verdad el MEJOR
    checkpoint, ya que train_one_fold puede parar varios epochs
    despues del mejor por el propio funcionamiento del early-stopping."""
    from peft import PeftModel
    model.backbone = PeftModel.from_pretrained(model.backbone, path / "lora_adapter")
    model.backbone.to(device)
    load_checkpoint_head(model, path)
    model.classifier.to(device)
    return model


def train_one_fold(
    model: nn.Module,
    train_records: list[dict],
    val_records: list[dict],
    processor,
    checkpoint_dir: Path,
    device: str = "cuda",
    lr_lora: float = 2e-4,
    lr_head: float = 5e-4,
    max_epochs: int = 10,
    patience: int = 3,
    batch_size: int = 2,  # reducido de 4: la T4 (14.56 GiB) se quedo sin margen con batch_size=4 en ejecucion real
) -> dict:
    num_workers = 2 if device == "cuda" else 0
    train_loader = DataLoader(
        TextileDataset(train_records), batch_size=batch_size, shuffle=True,
        collate_fn=build_collate_fn(processor),
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        TextileDataset(val_records), batch_size=batch_size, shuffle=False,
        collate_fn=build_collate_fn(processor),
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    class_weights = compute_class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr_lora},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ])

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_f1, _, _ = evaluate(model, val_loader, device)
        mean_train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": mean_train_loss, "val_f1": val_f1})
        print(f"epoch {epoch}: train_loss={mean_train_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_dir, best_f1, epoch)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping en epoch {epoch} (sin mejora en {patience} epochs)")
                break

    return {"best_val_f1": best_f1, "history": history}
