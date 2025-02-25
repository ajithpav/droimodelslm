import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Union
import torch
import accelerate
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Check dependencies
def check_dependencies():
    try:
        import accelerate
        logging.info(f"Using accelerate version: {accelerate.__version__}")
        import torch
        logging.info(f"Using PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("CUDA is not available. Using CPU for training.")
    except ImportError as e:
        raise ImportError(
            "Please install required dependencies: pip install transformers[torch] accelerate>=0.26.0"
        ) from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "gpt2-large",
        output_dir: str = "droidal_finetuned_model",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """Initialize and configure the model and tokenizer."""
        try:
            logging.info(f"Loading model and tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

            # Move model to appropriate device
            self.model.to(self.device)
            logging.info(f"Model loaded and moved to {self.device}")
        except Exception as e:
            logging.error(f"Error setting up model and tokenizer: {str(e)}")
            raise

    def prepare_dataset(self, text_data: List[str], json_data: List[Dict]) -> Dataset:
        """Prepare the dataset from text and JSON data."""
        try:
            # Process text data
            text_examples = [{"text": text} for text in text_data if text.strip()]
            
            # Process JSON data
            json_examples = []
            for item in json_data:
                combined_text = f"{item['prompt'].strip()} {item['completion'].strip()}"
                json_examples.append({"text": combined_text})
            
            # Combine all examples
            all_examples = text_examples + json_examples
            
            # Create dataset
            dataset = Dataset.from_dict({"text": [ex["text"] for ex in all_examples]})
            
            # Tokenize the dataset
            tokenized_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ),
                batched=True,
                remove_columns=["text"]
            )
            
            # Split dataset
            train_testvalid = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
            test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
            
            return {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "valid": test_valid["train"]
            }
        
        except Exception as e:
            logging.error(f"Error preparing dataset: {str(e)}")
            raise

    def train(self, datasets, batch_size: int = 2, num_epochs: int = 3):
        """Train the model with the prepared datasets."""
        try:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="steps",
                eval_steps=100,
                save_steps=500,
                warmup_steps=500,
                learning_rate=5e-5,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to="tensorboard"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["valid"],
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )

            logging.info("Starting training...")
            trainer.train()
            
            # Save the final model
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def save_model(self):
        """Save the fine-tuned model and tokenizer."""
        try:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logging.info(f"Model and tokenizer saved to {self.output_dir}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

def main():
    # Check dependencies first
    check_dependencies()

    # Configuration
    MODEL_NAME = "gpt2-large"
    OUTPUT_DIR = "droidal_finetuned_model"
    BATCH_SIZE = 2
    NUM_EPOCHS = 3

    try:
        # Load data
        with open("droidal_combined_text.txt", "r", encoding="utf-8") as f:
            text_data = [line.strip() for line in f if line.strip()]
        
        with open("droidal_training_data.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Initialize trainer
        trainer = ModelTrainer(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR
        )

        # Prepare datasets
        datasets = trainer.prepare_dataset(text_data, json_data)

        # Train model
        trainer.train(datasets, BATCH_SIZE, NUM_EPOCHS)

    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()