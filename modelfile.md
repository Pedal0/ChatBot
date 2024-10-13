name: fine-tuned-llama3-1B
version: 1.0
description: Modèle avec des fichiers de configuration et un modèle SafeTensors.
dependencies:
  - torch
  - transformers 
model:
  type: safetensors  # ou "transformers" selon l'API que vous utilisez
  path: ./app/fine_tuned_llama3-1B  # Chemin vers le dossier contenant vos fichiers
config:
  config_file: config.json
  generation_config_file: generation_config.json
  tokenizer_config_file: tokenizer_config.json
  tokenizer_file: tokenizer.json
  special_tokens_file: special_tokens_map.json
