"""
Fase 9 - Inferencia sobre imagenes nuevas + explicacion en lenguaje
natural OPCIONAL y DESACOPLADA.

Esta es la unica fase que recibe imagenes SIN etiqueta real: es el
entregable, lo que convierte el proyecto en algo desplegable en vez
de solo evaluable.

DISEÑO CRITICO (no tocar sin releer esto primero):
El veredicto (OK/NO_OK) SIEMPRE sale de classify_image(), que usa el
cabezal de clasificacion sobre el hidden state - nunca de texto
generado. Si se activa la explicacion en lenguaje natural, esta se
genera DESPUES, y generate_explanation() recibe el veredicto ya
decidido como parametro de entrada: no tiene forma de recalcularlo
ni de contradecirlo en la logica del programa, solo de narrarlo. Si
el texto generado suena raro, es un problema de UX de la narracion,
nunca un fallo del sistema de deteccion - la separacion estructural
es precisamente lo que garantiza esto.

Verificado por ejecucion en este entorno (sin GPU, con el backbone
Llama diminuto ya validado en fases anteriores, que SI soporta
.generate() de verdad al ser un modelo causal completo):
  1. classify_image() calcula argmax y probabilidades correctamente
     contra un calculo manual de softmax.
  2. disable_adapter() cambia de verdad la salida del modelo (no es
     un no-op): se comprueba que backbone con y sin adaptador dan
     logits distintos, confirmando que el interruptor funciona.
  3. generate_explanation() llama a .generate() de verdad, decodifica
     solo los tokens nuevos (no repite el prompt de entrada), y el
     texto del prompt de generacion contiene el veredicto ya decidido
     en vez de volver a preguntar "¿hay defecto?".
  4. predict_with_explanation() calcula el veredicto ANTES de generar
     la explicacion, nunca al reves.

NO verificado (requiere GPU + Qwen3-VL real): la calidad/coherencia
real del texto generado. Es esperable revisar a mano las primeras
explicaciones generadas en Colab antes de confiar en ellas para uso
en produccion.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

CHECKPOINT_DIR = Path("models/final_model")
LABEL_NAMES = {0: "NO_OK", 1: "OK"}
VERDICT_DESCRIPTION = {
    "OK": "sin defectos visibles",
    "NO_OK": "con un defecto de fabricacion",
}
EXPLANATION_PROMPT_TEMPLATE = (
    "Un sistema de inspeccion de calidad ha analizado este tejido y lo ha "
    "clasificado como {descripcion}. En una o dos frases, describe que "
    "caracteristicas visuales de la imagen son consistentes con esa "
    "clasificacion."
)


def _build_messages(image: Image.Image, prompt_text: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]


@torch.no_grad()
def classify_image(model, image_path: str, processor, prompt_text: str, device: str) -> dict:
    """El VEREDICTO, y lo unico que lo decide. Nunca genera texto."""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    messages = _build_messages(image, prompt_text)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_id = int(probs.argmax().item())

    return {
        "path": str(image_path),
        "pred_label": LABEL_NAMES[pred_id],
        "pred_id": pred_id,
        "confidence": float(probs[pred_id].item()),
        "probabilities": {LABEL_NAMES[i]: float(probs[i].item()) for i in range(len(probs))},
    }


@torch.no_grad()
def generate_explanation(model, image_path: str, processor, pred_label: str, device: str,
                          use_adapter: bool = False, max_new_tokens: int = 120) -> str:
    """Genera una justificacion en lenguaje natural del veredicto YA
    DECIDIDO (recibido en pred_label, nunca recalculado aqui). Con
    use_adapter=False (por defecto) se desactiva temporalmente el
    adaptador LoRA para esta llamada -entrenado solo para clasificar,
    nunca para generar texto- y se usan los pesos base originales."""
    image = Image.open(image_path).convert("RGB")
    prompt_text = EXPLANATION_PROMPT_TEMPLATE.format(descripcion=VERDICT_DESCRIPTION[pred_label])
    messages = _build_messages(image, prompt_text)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    def _generate():
        return model.backbone.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    if use_adapter:
        generated_ids = _generate()
    else:
        with model.backbone.disable_adapter():
            generated_ids = _generate()

    new_tokens = generated_ids[0][input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def predict_with_explanation(model, image_path: str, processor, prompt_text: str, device: str,
                              explain: bool = False, use_adapter_for_explanation: bool = False) -> dict:
    """Orquestador: SIEMPRE clasifica primero. Solo si explain=True,
    genera despues una explicacion anclada a ese veredicto ya fijo."""
    result = classify_image(model, image_path, processor, prompt_text, device)
    if explain:
        result["explanation"] = generate_explanation(
            model, image_path, processor, result["pred_label"], device,
            use_adapter=use_adapter_for_explanation,
        )
    return result


def _iter_image_paths(target: Path):
    if target.is_file():
        yield target
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            yield from sorted(target.glob(ext))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia sobre tejidos nuevos (Fase 9)")
    parser.add_argument("path", help="Ruta a una imagen o a una carpeta de imagenes")
    parser.add_argument("--explain", action="store_true",
                         help="Genera tambien una explicacion en lenguaje natural (no decide nada)")
    parser.add_argument("--use-adapter-for-explanation", action="store_true",
                         help="Usa el adaptador LoRA tambien para generar la explicacion (por defecto se desactiva)")
    args = parser.parse_args()

    import torch as _torch
    from transformers import AutoProcessor

    from src.models.classifier import TextileDefectClassifier
    from src.training.train_fold import PROMPT_B, load_checkpoint

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No se encuentra {CHECKPOINT_DIR}. Ejecuta primero la Fase 7 "
            "(src/evaluation/final_evaluation.py) para generar el modelo final."
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if _torch.cuda.is_available() else "cpu"

    model = TextileDefectClassifier(model_id=MODEL_ID, lora_config=None)
    model = load_checkpoint(model, CHECKPOINT_DIR, device)

    results = []
    for img_path in _iter_image_paths(Path(args.path)):
        r = predict_with_explanation(
            model, str(img_path), processor, PROMPT_B, device,
            explain=args.explain, use_adapter_for_explanation=args.use_adapter_for_explanation,
        )
        results.append(r)
        print(f"{img_path.name}: {r['pred_label']} (confianza {r['confidence']:.1%})")
        if args.explain:
            print(f"   -> {r['explanation']}")

    print(json.dumps(results, indent=2, ensure_ascii=False))
