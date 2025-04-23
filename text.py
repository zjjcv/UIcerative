import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                     kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels=None):
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        log_soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(log_soft_student, soft_teacher) * (self.temperature ** 2)
        
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            return 0.5 * (ce_loss + distill_loss)
        return distill_loss

class DebiasLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, attention_weights):
        ce_loss = self.ce(logits, labels)
        attention = attention_weights.mean(dim=1)
        diversity_loss = -torch.mean(torch.sum(attention * torch.log(attention + 1e-10), dim=-1))
        return ce_loss - 0.1 * diversity_loss

class IBDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(base_path):
    """Load and combine data from all three disease categories."""
    categories = ['CD', 'UC', 'ITB']
    texts = []
    labels = []
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        for case_dir in os.listdir(category_path):
            if case_dir != 'image_text':  # Skip the image_text directory
                csv_path = os.path.join(category_path, case_dir, f"{case_dir}.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        text = df['text'].iloc[0] if 'text' in df.columns else ""
                        if text:
                            texts.append(text)
                            labels.append(idx)
                    except:
                        print(f"Error reading {csv_path}")
                        continue
    
    return texts, labels

def train_model(train_texts, train_labels, val_texts, val_labels, model_name="bert-base-chinese"):
    """Train the model using the provided data."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, output_attentions=True, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model from HuggingFace. Error: {e}")
        # Fallback to using a simple CNN model
        vocab_size = 30000  # Default vocabulary size
        embedding_dim = 768
        model = CNN(vocab_size=vocab_size,
                   embedding_dim=embedding_dim,
                   n_filters=100,
                   filter_sizes=[3, 4, 5],
                   output_dim=3,
                   dropout=0.5)
        # Create a basic tokenizer
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', trust_remote_code=True)
    
    train_dataset = IBDDataset(train_texts, train_labels, tokenizer)
    val_dataset = IBDDataset(val_texts, val_labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./checkpoints/text_classifier",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    trainer.save_model()
    return model, tokenizer
def run_distillation(train_texts, train_labels, val_texts, val_labels, teacher_model_path, model_name="hfl/chinese-roberta-wwm-ext"):
    """Run knowledge distillation from teacher to student model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher.eval()

    student = CNN(vocab_size=tokenizer.vocab_size,
                 embedding_dim=768,
                 n_filters=100,
                 filter_sizes=[3, 4, 5],
                 output_dim=3,
                 dropout=0.5)

    if torch.cuda.is_available():
        teacher = teacher.cuda()
        student = student.cuda()

    train_dataset = IBDDataset(train_texts, train_labels, tokenizer)
    val_dataset = IBDDataset(val_texts, val_labels, tokenizer)

    optimizer = Adam(student.parameters(), lr=1e-3)
    criterion = DistillationLoss()

    num_epochs = 10
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        
        for batch in torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_outputs = teacher(**batch)
            
            student_logits = student(batch['input_ids'])
            loss = criterion(student_logits, teacher_outputs.logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataset)
        logger.info(f'Epoch {epoch}: Training Loss = {avg_loss:.4f}')

        # Validation
        student.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(val_dataset, batch_size=8):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                teacher_outputs = teacher(**batch)
                student_logits = student(batch['input_ids'])
                loss = criterion(student_logits, teacher_outputs.logits, batch['labels'])
                val_loss += loss.item()
                
                _, predicted = student_logits.max(1)
                total += batch['labels'].size(0)
                correct += predicted.eq(batch['labels']).sum().item()
        
        val_loss /= len(val_dataset)
        accuracy = 100. * correct / total
        logger.info(f'Epoch {epoch}: Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.2f}%')
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(student.state_dict(), 'checkpoints/student_model.pt')
            
    return student, tokenizer

def run_debiasing(model, train_texts, train_labels, val_texts, val_labels, tokenizer):
    """Run debiasing training."""
    train_dataset = IBDDataset(train_texts, train_labels, tokenizer)
    val_dataset = IBDDataset(val_texts, val_labels, tokenizer)
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = DebiasLoss()
    
    num_epochs = 5
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch, output_attentions=True)
            loss = criterion(outputs.logits, batch['labels'], outputs.attentions[-1])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataset)
        logger.info(f'Epoch {epoch}: Training Loss = {avg_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(val_dataset, batch_size=8):
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                outputs = model(**batch, output_attentions=True)
                loss = criterion(outputs.logits, batch['labels'], outputs.attentions[-1])
                val_loss += loss.item()
                
                _, predicted = outputs.logits.max(1)
                total += batch['labels'].size(0)
                correct += predicted.eq(batch['labels']).sum().item()
        
        val_loss /= len(val_dataset)
        accuracy = 100. * correct / total
        logger.info(f'Epoch {epoch}: Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.2f}%')
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/debiased_model.pt')
    
    return model

def interpret_predictions(model, texts, tokenizer):
    """Generate interpretations for model predictions."""
    model.eval()
    dataset = IBDDataset(texts, [0] * len(texts), tokenizer)  # Dummy labels
    interpretations = []
    
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=1):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = model(**batch, output_attentions=True)
            attention_weights = outputs.attentions[-1]
            attention = attention_weights.mean(dim=1)
            word_importance = attention.sum(dim=2)
            
            prediction = outputs.logits.argmax(dim=1).item()
            tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
            
            interpretations.append({
                'text': tokenizer.decode(batch['input_ids'][0]),
                'prediction': prediction,
                'token_importance': list(zip(tokens, word_importance[0].cpu().tolist()))
            })
    
    return interpretations

def main():
    # Load data
    base_path = "data/concatenated_images"
    texts, labels = load_data(base_path)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initial training
    model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)
    logger.info("Initial training completed")
    
    # Knowledge distillation
    student_model, _ = run_distillation(
        train_texts, train_labels, val_texts, val_labels,
        'checkpoints/text_classifier/pytorch_model.bin'
    )
    logger.info("Knowledge distillation completed")
    
    # Debiasing
    debiased_model = run_debiasing(
        model, train_texts, train_labels, val_texts, val_labels, tokenizer
    )
    logger.info("Debiasing completed")
    
    # Generate interpretations
    interpretations = interpret_predictions(debiased_model, val_texts[:5], tokenizer)
    
    # Save interpretations
    os.makedirs('results', exist_ok=True)
    with open('results/interpretations.txt', 'w', encoding='utf-8') as f:
        for interp in interpretations:
            f.write(f"Text: {interp['text']}\n")
            f.write(f"Prediction: {['CD', 'UC', 'ITB'][interp['prediction']]}\n")
            f.write("Token Importance:\n")
            for token, importance in interp['token_importance']:
                if not token.startswith('##'):  # Skip subword tokens
                    f.write(f"{token}: {importance:.4f}\n")
            f.write("\n")
    
    logger.info("Training completed. All models saved in checkpoints/")
    logger.info("Interpretations saved in results/interpretations.txt")

if __name__ == "__main__":
    main()