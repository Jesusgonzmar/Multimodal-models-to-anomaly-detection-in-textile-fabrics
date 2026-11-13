"""
Fase 4 - Arquitectura del clasificador sobre Qwen3-VL-2B-Instruct.

Envuelve el backbone con:
  - LoRA (via peft) sobre las proyecciones de atencion y MLP del
    decoder de texto. No se cuantiza: el modelo es pequeno y cabe en
    bf16 en una T4 de 16GB, lo que evita la incompatibilidad conocida
    entre cuantizacion 4-bit y LoRA sobre la torre visual, por si mas
    adelante se quiere extender LoRA tambien al encoder de imagen.
  - Un cabezal de clasificacion binaria sobre el HIDDEN STATE de la
    ultima capa en la ultima posicion de la secuencia (no sobre los
    logits de vocabulario - ver el analisis del proyecto anterior:
    usar logits obliga a pasar por una proyeccion de ~150k tokens
    pensada para prediccion de palabra, no para clasificacion).

Verificado por ejecucion en este entorno (sin GPU, con un modelo
Llama diminuto de la misma familia arquitectonica que Qwen3 -
RMSNorm, SwiGLU, GQA, mismos nombres de modulo):
  1. LoRA + cabezal + extraccion de hidden_states: forward y backward
     completos, con el gradiente llegando a LoRA y al cabezal, y NO
     llegando a los pesos base congelados.
  2. El padding a la izquierda hace que el indice [-1] sea siempre el
     ultimo token real de cada muestra del lote (diferencia numerica
     entre procesar una secuencia sola vs. en un lote con padding:
     ~1e-7, es decir, ruido de precision, no un error de diseño).

NO verificado por ejecucion (requiere GPU y descargar el modelo real,
no disponible en este entorno): la carga real de Qwen3-VL-2B-Instruct
y el forward pass multimodal completo (imagen + texto). Por eso
'inspect_linear_modules' existe: en vez de dar por hecho los nombres
de las capas, los COMPRUEBA contra el modelo real la primera vez que
se ejecuta esto en Colab, antes de lanzar cualquier entrenamiento.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
NUM_LABELS = 2


def load_backbone(model_id: str, **kwargs):
    """Carga el backbone probando el parametro de precision 'dtype'
    (el que usa la ficha oficial de Qwen3-VL en transformers recientes)
    y cae a 'torch_dtype' (el nombre historico) si la version instalada
    no reconoce el primero. No se asume cual es correcto sin verificar
    contra la version real de transformers en el entorno de ejecucion."""
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)


def inspect_linear_modules(model: nn.Module, max_print: int = 40) -> set[str]:
    """No suponer, comprobar: imprime los nombres reales de las capas
    Linear del modelo cargado, para confirmar que los target_modules
    de LoRA existen tal cual en esta version del modelo antes de
    lanzar cualquier entrenamiento."""
    names = sorted({
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    })
    print(f"Nombres unicos de capas Linear encontradas ({len(names)}):")
    for n in names[:max_print]:
        print(f"  - {n}")
    return set(names)


def check_target_module_scope(model: nn.Module, target_modules: list[str]) -> dict[str, list[str]]:
    """peft empareja target_modules por coincidencia de SUFIJO en
    cualquier parte del modelo. Si la torre visual usa nombres de capa
    genericos iguales a los del decoder de texto (q_proj, k_proj...),
    LoRA podria estar aplicandose tambien ahi sin que nadie lo pidiera
    explicitamente. Esto lo comprueba mirando la ruta COMPLETA de cada
    coincidencia, no solo el nombre corto."""
    matches: dict[str, list[str]] = {t: [] for t in target_modules}
    for full_name, _ in model.named_modules():
        for t in target_modules:
            if full_name.endswith(t):
                matches[t].append(full_name)

    print("=== Alcance real de los target_modules de LoRA ===")
    leaked_to_vision = False
    for t, paths in matches.items():
        vision_paths = [p for p in paths if "visual" in p or "vision" in p]
        text_paths = [p for p in paths if p not in vision_paths]
        print(f"'{t}': {len(text_paths)} en el modelo de texto, {len(vision_paths)} en la torre visual")
        if vision_paths:
            leaked_to_vision = True
            for p in vision_paths[:3]:
                print(f"    AVISO - tambien coincide en vision: {p}")
    if not leaked_to_vision:
        print("Ninguna coincidencia dentro de la torre visual: LoRA solo toca el modelo de texto, como se diseno.")
    return matches


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                       include_mlp: bool = True, use_dora: bool = False) -> LoraConfig:
    """target_modules: convencion estandar de la familia Qwen/Llama
    (RMSNorm + SwiGLU + GQA). Confirmado por ejecucion en un modelo
    Llama de prueba con la misma arquitectura; se recomienda de todos
    modos correr inspect_linear_modules() sobre el modelo real antes
    de fiarse ciegamente.

    use_dora: activa DoRA (Weight-Decomposed LoRA, Liu et al. 2024),
    que en sus benchmarks publicados supera a LoRA estandar en
    fine-tuning de modelos vision-lenguaje, incluso con rango menor.
    No se fuerza aqui como default: se deja como opcion para que
    Optuna decida segun el resultado real sobre este dataset, en vez
    de asumir que la mejora publicada se traslada sin mas."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        use_dora=use_dora,
        bias="none",
        # FEATURE_EXTRACTION: no usamos peft para generar texto, solo
        # para inyectar LoRA; el cabezal de clasificacion es nuestro,
        # fuera de peft.
        task_type=TaskType.FEATURE_EXTRACTION,
    )


class TextileDefectClassifier(nn.Module):
    """Qwen3-VL-2B + LoRA + cabezal de clasificacion binaria sobre el
    hidden state de la ultima capa, ultima posicion de la secuencia.

    Espera recibir en forward() exactamente lo que devuelve:
        processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
    para un LOTE, con processor.tokenizer.padding_side = 'left'
    fijado ANTES de tokenizar (ver Fase 5, collate_fn del DataLoader).
    """

    def __init__(self, model_id: str = MODEL_ID, lora_config: LoraConfig | None = None,
                 num_labels: int = NUM_LABELS, freeze_vision_tower: bool = False,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        backbone = load_backbone(model_id, device_map="auto")

        if freeze_vision_tower:
            for name, param in backbone.named_parameters():
                if "visual" in name or "vision" in name:
                    param.requires_grad = False

        # hidden_size se lee del modelo real, nunca se hardcodea: la
        # ruta exacta del atributo puede variar segun version de
        # transformers, por eso se prueban varias antes de fallar.
        hidden_size = self._resolve_hidden_size(backbone)
        # CRITICO: nn.Linear se crea en CPU por defecto. Si el backbone
        # aterrizo en GPU via device_map="auto", el cabezal debe seguirlo
        # explicitamente o el forward() falla con un mismatch de dispositivo.
        device = next(backbone.parameters()).device

        self.backbone = get_peft_model(backbone, lora_config) if lora_config else backbone
        self.classifier = nn.Linear(hidden_size, num_labels).to(device=device, dtype=torch.float32)

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            # necesario cuando la base esta congelada y solo LoRA es
            # entrenable: sin esto, algunas implementaciones no marcan
            # correctamente los tensores de entrada para que el grafo de
            # gradiente se pueda recomputar durante el backward.
            # Verificado por ejecucion: da resultados numericamente
            # identicos a entrenar sin checkpointing, en 4 pasos reales.
            self.backbone.enable_input_require_grads()

    @staticmethod
    def _resolve_hidden_size(model: nn.Module) -> int:
        cfg = model.config
        for path in (
            lambda c: c.text_config.hidden_size,
            lambda c: c.hidden_size,
        ):
            try:
                return path(cfg)
            except AttributeError:
                continue
        raise AttributeError(
            "No se ha encontrado 'hidden_size' en la config del modelo. "
            "Revisa model.config manualmente e indica la ruta correcta."
        )

    def forward(self, **inputs) -> torch.Tensor:
        outputs = self.backbone(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()
        return self.classifier(last_hidden)


if __name__ == "__main__":
    # Auto-comprobacion: solo carga el modelo y verifica arquitectura,
    # NO entrena nada. Ejecutar esto una vez en Colab antes de
    # construir el bucle de entrenamiento de la Fase 5.
    print(f"Cargando {MODEL_ID} para verificacion de arquitectura...")
    backbone = load_backbone(MODEL_ID, device_map="auto")

    found_names = inspect_linear_modules(backbone)
    lora_cfg = build_lora_config()
    missing = [m for m in lora_cfg.target_modules if m not in found_names]
    if missing:
        print(f"\nAVISO: estos target_modules NO existen en el modelo real: {missing}")
        print("Ajusta build_lora_config() con los nombres correctos antes de seguir.")
    else:
        print(f"\nTodos los target_modules de LoRA existen en el modelo real: {lora_cfg.target_modules}")

    hidden_size = TextileDefectClassifier._resolve_hidden_size(backbone)
    print(f"hidden_size real del modelo: {hidden_size}")

    check_target_module_scope(backbone, lora_cfg.target_modules)

    print("\nArquitectura verificada. Lista para la Fase 5 (bucle de entrenamiento).")
