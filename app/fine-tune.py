import pandas as pd
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import Dataset
import torch

# Authentification
login(token="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")

# Charger les données
data = pd.read_csv('./data/video_games_sales.csv')

# Convertir les données en une grande chaîne de texte
data_text = data.to_string(index=False)
context = f"Voici les données des ventes de jeux vidéo :\n{data_text}"

# Charger le modèle et le tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Activer le gradient checkpointing
model.gradient_checkpointing_enable()

# Préparer les données d’entraînement
input_ids = tokenizer.encode(context, return_tensors="pt").squeeze()
labels = input_ids.clone()  # Utiliser les mêmes ids comme labels pour un entraînement causal

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

# Transférer le modèle sur le GPU si disponible
if torch.cuda.is_available():
    model.to('cuda')

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle fine-tuné
model.save_pretrained('./fine_tuned_llama3')
tokenizer.save_pretrained('./fine_tuned_llama3')
