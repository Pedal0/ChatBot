import json
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import logging

# Configurer les journaux
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les données depuis un fichier JSON
with open('./data/video_games_sales.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convertir les données en texte
data_text = json.dumps(data, ensure_ascii=False)

context = f"Voici les données du dataset de jeux vidéo :\n{data_text}"

# Charger le modèle et le tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name, use_cache=False)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Activer le gradient checkpointing
model.gradient_checkpointing_enable()

# Forcer l'utilisation du CPU
device = torch.device("cpu")
model.to(device)

# Vérifier si le GPU est utilisé
if torch.cuda.is_available():
    logger.info("Le modèle utilise le GPU.")
else:
    logger.warning("Le modèle utilise le CPU.")

# Préparer les données d’entraînement
max_length = 512
input_ids = tokenizer.encode(context, return_tensors="pt").squeeze()[:max_length]  # Tronquer la séquence si elle est trop longue
input_ids = input_ids.to(device)
labels = input_ids.clone()

# Créer le dataset
dataset = Dataset.from_dict({"input_ids": [input_ids.tolist()], "labels": [labels.tolist()]})

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results-1B',
    num_train_epochs=10,  
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
    no_cuda=True,
)

# Créer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Démarrer l'entraînement
trainer.train()

# Sauvegarder le modèle
output_dir = './fine_tuned_llama3-1B'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)