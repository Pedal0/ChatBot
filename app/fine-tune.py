import pandas as pd
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import Dataset
import torch
import logging

# Authentification
login(token="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")

# Configurer les journaux
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les données
data = pd.read_csv('./data/video_games_sales.csv')

# Convertir les données en une grande chaîne de texte
data_text = data.to_string(index=False)
context = f"Voici les données des ventes de jeux vidéo :\n{data_text}"

# Charger le modèle et le tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model = LlamaForCausalLM.from_pretrained(model_name, use_cache=False)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Activer le gradient checkpointing
model.gradient_checkpointing_enable()

# Préparer les données d’entraînement
max_length = 131072
input_ids = tokenizer.encode(context, return_tensors="pt").squeeze()[:max_length]
labels = input_ids.clone()

# Transférer les données sur le GPU si disponible
if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')
    labels = labels.to('cuda')
    model.to('cuda')

# Créer un dataset compatible avec `Trainer`
train_data = Dataset.from_dict({"input_ids": [input_ids], "labels": [labels]})

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# Créer un Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Entraîner le modèle avec gestion des erreurs
try:
    logger.info("Début de l'entraînement")
    trainer.train()
    logger.info("Fin de l'entraînement")
except Exception as e:
    logger.error(f"Erreur pendant l'entraînement : {e}")

# Sauvegarder le modèle fine-tuné
model.save_pretrained('./fine_tuned_llama3')
tokenizer.save_pretrained('./fine_tuned_llama3')
