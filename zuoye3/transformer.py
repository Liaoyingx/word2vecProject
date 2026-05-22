import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# ===================== 配置 =====================
device = torch.device('cpu')
BATCH_SIZE = 32
MAX_LEN = 25
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 1
EPOCHS = 8
LR = 1e-4

# ===================== 加载数据=====================
def load_data(path):
    eng, fra = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' not in line: continue
            e, f = line.split('\t')
            eng.append(e)
            fra.append(f)
    return eng, fra

def clean(text):
    text = re.sub(r'[^a-zA-Z0-9À-ÿ\s.,!?]', '', text)
    return text.lower()

train_eng, train_fra = load_data("eng-fra_train_data.txt")
test_eng, test_fra = load_data("eng-fra_test_data.txt")

train_eng = [clean(s) for s in train_eng]
train_fra = [clean(s) for s in train_fra]
test_eng = [clean(s) for s in test_eng]
test_fra = [clean(s) for s in test_fra]

# ===================== 词典 =====================
class Vocab:
    def __init__(self, sentences, min_freq=2):
        self.w2i = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.i2w = {0:'<pad>',1:'<sos>',2:'<eos>',3:'<unk>'}
        words = []
        for s in sentences:
            words.extend(s.split())
        cnt = Counter(words)
        idx = 4
        for w, f in cnt.items():
            if f >= min_freq:
                self.w2i[w] = idx
                self.i2w[idx] = w
                idx += 1
        self.size = idx

    def encode(self, s):
        tokens = s.split()
        ids = [1] + [self.w2i.get(w,3) for w in tokens] + [2]
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids += [0]*(MAX_LEN-len(ids))
        return torch.tensor(ids)

    def decode(self, ids):
        res = []
        for i in ids:
            if i == 2: break
            if i not in (0,1):
                res.append(self.i2w.get(i,'<unk>'))
        return ' '.join(res)

eng_vocab = Vocab(train_eng)
fra_vocab = Vocab(train_fra)

# ===================== 数据集 =====================
class MTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return eng_vocab.encode(self.x[i]), fra_vocab.encode(self.y[i])

train_loader = DataLoader(MTDataset(train_eng, train_fra), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(MTDataset(test_eng, test_fra), batch_size=BATCH_SIZE, shuffle=False)

# ===================== 模型1：点积注意力 =====================
class TransformerDot(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_src = nn.Embedding(eng_vocab.size, EMBED_DIM)
        self.emb_tgt = nn.Embedding(fra_vocab.size, EMBED_DIM)
        self.pos = torch.zeros(1, MAX_LEN, EMBED_DIM)
        pos = torch.arange(MAX_LEN).unsqueeze(1)
        div = torch.exp(torch.arange(0,EMBED_DIM,2)*(-math.log(10000.0)/EMBED_DIM))
        self.pos[0,:,0::2] = torch.sin(pos * div)
        self.pos[0,:,1::2] = torch.cos(pos * div)
        self.trans = nn.Transformer(
            d_model=EMBED_DIM, nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS,
            dim_feedforward=HIDDEN_DIM, batch_first=True
        )
        self.fc = nn.Linear(EMBED_DIM, fra_vocab.size)

    def forward(self, src, tgt):
        src_pad = (src == 0)
        tgt_pad = (tgt == 0)
        tgt_mask = self.trans.generate_square_subsequent_mask(tgt.shape[1])
        src = self.emb_src(src) + self.pos[:, :src.shape[1]]
        tgt = self.emb_tgt(tgt) + self.pos[:, :tgt.shape[1]]
        out = self.trans(src, tgt,
                         src_key_padding_mask=src_pad,
                         tgt_key_padding_mask=tgt_pad,
                         tgt_mask=tgt_mask)
        return self.fc(out)

# ===================== 模型2：加性注意力=====================
class TransformerAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_src = nn.Embedding(eng_vocab.size, EMBED_DIM)
        self.emb_tgt = nn.Embedding(fra_vocab.size, EMBED_DIM)
        self.pos = torch.zeros(1, MAX_LEN, EMBED_DIM)
        pos = torch.arange(MAX_LEN).unsqueeze(1)
        div = torch.exp(torch.arange(0,EMBED_DIM,2)*(-math.log(10000.0)/EMBED_DIM))
        self.pos[0,:,0::2] = torch.sin(pos * div)
        self.pos[0,:,1::2] = torch.cos(pos * div)


        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=HIDDEN_DIM,
            batch_first=True, bias=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, NUM_LAYERS)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=HIDDEN_DIM,
            batch_first=True, bias=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, NUM_LAYERS)

        self.fc = nn.Linear(EMBED_DIM, fra_vocab.size)

    def forward(self, src, tgt):
        src_pad = (src == 0)
        tgt_pad = (tgt == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])

        src = self.emb_src(src) + self.pos[:, :src.shape[1]]
        tgt = self.emb_tgt(tgt) + self.pos[:, :tgt.shape[1]]

        memory = self.encoder(src, src_key_padding_mask=src_pad)
        out = self.decoder(tgt, memory,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_pad)
        return self.fc(out)

# ===================== 训练 =====================
def train(model, name):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    opt = optim.Adam(model.parameters(), lr=LR)
    model.train()
    loss_hist = []
    print(f"\n===== 训练 {name} =====")
    for e in range(EPOCHS):
        total = 0
        for src, tgt in train_loader:
            opt.zero_grad()
            out = model(src, tgt[:,:-1])
            loss = criterion(out.reshape(-1, fra_vocab.size), tgt[:,1:].reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        loss_hist.append(avg)
        print(f"Epoch {e+1:2d} | Loss: {avg:.3f}")
    return model, loss_hist

# ===================== 测试 =====================
def eval_test(model):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = 0.0
    with torch.no_grad():
        for src, tgt in test_loader:
            out = model(src, tgt[:,:-1])
            loss += criterion(out.reshape(-1, fra_vocab.size), tgt[:,1:].reshape(-1)).item()
    return loss / len(test_loader)

# ===================== 翻译 =====================
def translate(model, sent):
    model.eval()
    src = eng_vocab.encode(clean(sent)).unsqueeze(0)
    tgt = torch.tensor([[1]])
    for _ in range(MAX_LEN):
        with torch.no_grad():
            pred = model(src, tgt).argmax(-1)[:,-1].item()
        if pred == 2:
            break
        tgt = torch.cat([tgt, torch.tensor([[pred]])], dim=1)
    return fra_vocab.decode(tgt[0].numpy())

# ===================== BLEU =====================
def calculate_bleu(model, eng_sens, fra_sens, sample_num=100):
    preds, refs = [], []
    for eng, fra in zip(eng_sens[:sample_num], fra_sens[:sample_num]):
        pred = translate(model, eng)
        preds.append(pred.split())
        refs.append([fra.split()])

    def ngram(words, n):
        d = defaultdict(int)
        for i in range(len(words)-n+1):
            d[tuple(words[i:i+n])] += 1
        return d

    p1=p2=p3=p4=0
    for p, r in zip(preds, refs):
        r = r[0]
        if len(p) < 1: continue
        c1 = ngram(p,1); r1=ngram(r,1); p1 += sum(min(c1[k], r1.get(k,0)) for k in c1)/len(p)
        if len(p)>=2: c2=ngram(p,2); r2=ngram(r,2); p2 += sum(min(c2[k], r2.get(k,0)) for k in c2)/(len(p)-1)
        if len(p)>=3: c3=ngram(p,3); r3=ngram(r,3); p3 += sum(min(c3[k], r3.get(k,0)) for k in c3)/(len(p)-2)
        if len(p)>=4: c4=ngram(p,4); r4=ngram(r,4); p4 += sum(min(c4[k], r4.get(k,0)) for k in c4)/(len(p)-3)
    N = len(preds)
    return math.exp((math.log(p1/N)+math.log(p2/N)+math.log(p3/N)+math.log(p4/N))/4)

# ===================== 主程序 =====================
if __name__ == "__main__":
    model_dot = TransformerDot()
    model_add = TransformerAdd()

    model_dot, dot_loss = train(model_dot, "点积注意力")
    model_add, add_loss = train(model_add, "加性注意力")

    plt.figure(figsize=(10,5))
    plt.plot(dot_loss, label="Dot Product")
    plt.plot(add_loss, label="Additive")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    test_cases = ["I love you", "What is your name?", "He is a student", "We are happy"]
    print("\n===== 翻译对比 =====")
    for sent in test_cases:
        print(f"\nEN: {sent}")
        print(f"Dot: {translate(model_dot, sent)}")
        print(f"Add: {translate(model_add, sent)}")

    bleu_dot = calculate_bleu(model_dot, test_eng, test_fra)
    bleu_add = calculate_bleu(model_add, test_eng, test_fra)
    loss_dot = eval_test(model_dot)
    loss_add = eval_test(model_add)


    print(f"点积 | 测试损失 {loss_dot:.3f} | BLEU {bleu_dot:.3f}")
    print(f"加性 | 测试损失 {loss_add:.3f} | BLEU {bleu_add:.3f}")