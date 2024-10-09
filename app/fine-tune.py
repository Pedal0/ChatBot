import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from huggingface_hub import login

# Authentification
login(token="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")

# Charger les données
data = pd.read_csv('./data/video_games_sales.csv')

# Convertir les données en une grande chaîne de texte
data_text = data.to_string(index=False)

# Exemple de texte à utiliser pour le fine-tuning
context = f"Voici les données des ventes de jeux vidéo :\n{data_text}"

# Charger le modèle et le tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Préparer les données pour l'entraînement
train_data = [{"input_ids": tokenizer.encode(context, return_tensors="pt"), "labels": tokenizer.encode(context, return_tensors="pt")}]

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Créer un Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle fine-tuné
model.save_pretrained('./fine_tuned_llama3')
tokenizer.save_pretrained('./fine_tuned_llama3')
