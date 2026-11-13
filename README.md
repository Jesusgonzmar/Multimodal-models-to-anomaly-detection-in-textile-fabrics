# Deteccion de defectos en tejidos textiles con Qwen3-VL-2B

TFG: clasificacion binaria (OK / NO_OK) de fotografias de tejidos
mediante fine-tuning con LoRA de un modelo de vision y lenguaje
(Qwen3-VL-2B-Instruct), con seleccion de hiperparametros vía Optuna
y validacion cruzada estratificada.

## Estado del proyecto

- [x] Fase 1 - Auditoria del dataset (`src/data/audit_dataset.py`)
- [x] Fase 2 - Division train/val/test consciente de grupos (`src/data/splits.py`)
- [x] Fase 3 - Preprocesado de imagenes y JSON final de entrenamiento
- [x] Fase 4 - Arquitectura del clasificador sobre Qwen3-VL-2B
- [x] Fase 5 - Bucle de entrenamiento (LoRA + early stopping)
- [x] Fase 6 - Busqueda de hiperparametros (Optuna + K-fold)
- [x] Fase 7 - Reentrenamiento final y evaluacion sobre test
- [x] Fase 8 - Analisis de errores (`src/evaluation/error_analysis.py`)
- [x] Fase 9 - Script de inferencia (`src/inference/predict.py`). El
      veredicto sale SIEMPRE de classify_image() (cabezal sobre
      hidden state); la explicacion en lenguaje natural es opcional,
      desacoplada, y recibe el veredicto ya decidido como parametro -
      nunca lo recalcula ni puede contradecirlo.
      **NOTA IMPORTANTE (no olvidar):** el cabezal de clasificacion
      (Fase 4) NUNCA genera lenguaje natural, solo un logit OK/NO_OK.
      Si se quiere que el sistema tambien explique el veredicto en
      lenguaje natural, debe ser un SEGUNDO mecanismo desacoplado:
      una llamada aparte a `model.generate()` sobre el mismo backbone,
      cuyo texto NUNCA decide el veredicto (eso lo sigue decidiendo
      el clasificador). Es una capa de interfaz para el usuario
      humano, no una fuente de verdad. Ver conversacion de diseño
      del proyecto para el razonamiento completo.

## Estructura

```
data/raw/          -> NO esta en git, vive en Google Drive
data/processed/     -> manifest.csv, duplicados, imagenes cacheadas (Drive)
data/splits/        -> test.json, trainval_folds.json (SI esta en git)
src/data/           -> scripts de auditoria y division
src/inference/      -> prediccion sobre imagenes nuevas (Fase 9)
configs/            -> hiperparametros y rutas en YAML
models/             -> checkpoints (NO esta en git, vive en Google Drive)
reports/            -> metricas, model card, figuras
```

## Reproducibilidad

Semilla fija: `SEED = 42` en todos los scripts de datos.
Los splits (`data/splits/*.json`) estan versionados en este repo
precisamente para que la particion train/val/test sea auditable y
no cambie entre ejecuciones.

## Como ejecutar (Google Colab)

0. **Antes de importar torch, en la PRIMERA celda del notebook:**
   ```python
   import os
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
   Necesario porque Qwen3-VL usa resolucion dinamica (cada imagen genera
   un numero distinto de tokens visuales), y eso fragmenta el asignador
   de memoria de PyTorch en una T4 de 16GB. Confirmado con un
   OutOfMemoryError real en la Fase 6 (ver notas del incidente).
1. Montar Drive y clonar este repo (ver celda de arranque en el notebook).
2. `pip install -r requirements.txt`
3. `python src/data/audit_dataset.py`
4. `python src/data/splits.py`
