import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, TrainingArguments

class FineTuneLlamaModel:
    def __init__(self, model_name='llama3-1B', fine_tune_dir='fine_tune_llama3-1B'):
        self.model_name = model_name
        self.fine_tune_dir = fine_tune_dir
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)

    def fine_tune(self, train_dataset, epochs=3, batch_size=8, learning_rate=5e-5):

        training_args = TrainingArguments(
            output_dir=self.fine_tune_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir=os.path.join(self.fine_tune_dir, 'logs'),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        self.save_model()

    def save_model(self):
        self.model.save_pretrained(self.fine_tune_dir)
        self.tokenizer.save_pretrained(self.fine_tune_dir)

    def load_model(self):
        self.model = LlamaForCausalLM.from_pretrained(self.fine_tune_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.fine_tune_dir)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs['input_ids'], max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)