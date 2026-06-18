import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
from news_data import preprocess_text

# ---------- 1. 数据集类 ----------
class BertNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ---------- 2. 数据加载 ----------
def load_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    print("正在加载原始数据...")
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    # 应用相同预处理（与 GRU 保持一致）
    X_train_raw = [preprocess_text(doc) for doc in newsgroups_train.data]
    X_test_raw = [preprocess_text(doc) for doc in newsgroups_test.data]

    # 标签映射
    unique_labels = sorted(list(set(newsgroups_train.target)))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_train = [label_map[label] for label in newsgroups_train.target]
    y_test = [label_map[label] for label in newsgroups_test.target]

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42
    )

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test_raw)}")
    return X_train, X_val, X_test_raw, y_train, y_val, y_test

# ---------- 3. 训练函数 ----------
def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cpu":
        print("⚠️ 检测到 CPU，训练速度较慢。已优化为 max_len=128, epochs=3。")
        print("训练过程中将每 10 个 batch 输出一次进度，请耐心等待。")

    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 超参数（针对 CPU 优化）
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1

    print("正在加载 BERT 模型和分词器...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # 创建数据集
    train_dataset = BertNewsDataset(X_train, y_train, tokenizer, max_len=MAX_LEN)
    val_dataset = BertNewsDataset(X_val, y_val, tokenizer, max_len=MAX_LEN)
    test_dataset = BertNewsDataset(X_test, y_test, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    patience = 2
    patience_counter = 0

    print(f"开始训练 BERT（最大长度={MAX_LEN}, 轮次={EPOCHS}, 总 batch 数={len(train_loader)}）...")
    global_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        # 遍历 batch
        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # ---------- 进度输出（每 10 个 batch 打印一次） ----------
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - epoch_start_time
                avg_time_per_batch = elapsed / batch_idx
                remaining_batches = len(train_loader) - batch_idx
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                current_loss = loss.item()
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {current_loss:.4f} | Elapsed: {elapsed:.1f}s | ETA: {eta_min}m{eta_sec}s")

        # 计算 epoch 统计
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"\n✅ Epoch {epoch+1}/{EPOCHS} 完成 | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | 用时: {epoch_time:.1f}s\n")

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_bert_model.pth')
            print(f"  ★ 新最佳模型保存，验证准确率: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发（{patience} 轮未提升）。")
                break

    # 加载最佳模型并在测试集评估
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load('best_bert_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    total_time = time.time() - global_start_time
    print(f"\n🎉 训练完成！总用时: {int(total_time//60)}分{int(total_time%60)}秒")
    print(f"✅ 测试集准确率: {test_acc:.2f}%")
    return best_val_acc, test_acc

if __name__ == "__main__":
    train_bert()