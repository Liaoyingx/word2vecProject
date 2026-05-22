import numpy as np
import matplotlib.pyplot as plt
import jieba
from sklearn.decomposition import PCA
import pickle
import os
import re

# ====================== 1. 自定义词典，解决多字词拆分问题 ======================
# 强制保留多字词，避免拆分
jieba.add_word("一带一路")
jieba.add_word("中共中央")
jieba.add_word("中央军委")
jieba.add_word("中国式现代化")
jieba.add_word("海洋经济")
jieba.add_word("高质量发展")
jieba.add_word("数字海洋")
jieba.add_word("海洋科技")

# ====================== 2. 加载语料 + 自动分词 + 清洗 ======================
def load_and_cut_corpus(path="corpus.txt"):
    sentences = []
    filter_pattern = re.compile(r'[^\u4e00-\u9fa5]')  # 只保留中文

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = jieba.lcut(line)
            words_clean = []
            for w in words:
                w = filter_pattern.sub('', w)
                if len(w) >= 1:
                    words_clean.append(w)
            if len(words_clean) > 3:
                sentences.append(words_clean)
    return sentences

# ====================== 3. 构建词汇表 ======================
def build_vocab(sentences):
    vocab = set()
    for sent in sentences:
        vocab.update(sent)
    vocab = sorted(list(vocab))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

# ====================== 4. CBOW 样本生成 ======================
def generate_cbow_data(sentences, word2idx, window_size=3):
    data = []
    for sent in sentences:
        sent_len = len(sent)
        for idx, center in enumerate(sent):
            context = []
            for w in range(-window_size, window_size + 1):
                if w == 0:
                    continue
                c_idx = idx + w
                if 0 <= c_idx < sent_len:
                    context.append(sent[c_idx])
            if len(context) > 0:
                data.append((context, center))
    return data

# ====================== 5. CBOW 模型 ======================
class CBOW:
    def __init__(self, vocab_size, embed_dim=16, lr=0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr

        self.W_in = np.random.uniform(-1.0, 1.0, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-1.0, 1.0, (embed_dim, vocab_size))

    def forward(self, context_indices):
        context_vecs = self.W_in[context_indices]
        h = np.mean(context_vecs, axis=0)
        u = np.dot(h, self.W_out)
        exp_u = np.exp(u - np.max(u))
        y_pred = exp_u / np.sum(exp_u)
        return h, y_pred

    def backward(self, h, y_pred, target_idx, context_indices):
        EI = y_pred.copy()
        EI[target_idx] -= 1

        dW_out = np.outer(h, EI)
        d_h = np.dot(self.W_out, EI)

        dW_in = np.zeros_like(self.W_in)
        for c in context_indices:
            dW_in[c] += d_h / len(context_indices)

        self.W_out -= self.lr * dW_out
        self.W_in -= self.lr * dW_in

    def train(self, cbow_data, word2idx, epochs=5000):
        for epoch in range(epochs):
            total_loss = 0.0
            for context, center in cbow_data:
                ctx_idx = [word2idx[w] for w in context]
                tgt_idx = word2idx[center]

                h, y_pred = self.forward(ctx_idx)
                loss = -np.log(y_pred[tgt_idx] + 1e-8)
                total_loss += loss

                self.backward(h, y_pred, tgt_idx, ctx_idx)

            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f}")

    def get_vector(self, word, word2idx):
        return self.W_in[word2idx[word]]

# ====================== 6. 可视化：选择主题相关的10个词 ======================
def plot_embedding(model, word2idx, idx2word):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 手动选择海洋经济主题相关的10个词，更直观
    words = ["海洋", "经济", "发展", "科技", "产业", "生态", "一带一路", "中国", "政策", "创新"]
    # 过滤不在词典中的词
    words = [w for w in words if w in word2idx]
    vecs = np.array([model.get_vector(w, word2idx) for w in words])

    pca = PCA(n_components=2)
    vec_2d = pca.fit_transform(vecs)

    plt.figure(figsize=(9, 6))
    for i, word in enumerate(words):
        x, y = vec_2d[i]
        plt.scatter(x, y, s=80)
        plt.text(x + 0.02, y + 0.02, word, fontsize=12)

    plt.title("CBOW 中文词向量分布（海洋经济主题优化版）", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("正在加载并分词语料...")
    sentences = load_and_cut_corpus()
    print("示例句子（分词后）：", sentences[0][:15], "...")

    vocab, word2idx, idx2word = build_vocab(sentences)
    vocab_size = len(vocab)
    print("词汇表大小：", vocab_size)

    cbow_data = generate_cbow_data(sentences, word2idx, window_size=3)

    print("开始训练 CBOW 模型（优化版参数）...")
    model = CBOW(vocab_size=vocab_size, embed_dim=16, lr=0.01)
    model.train(cbow_data, word2idx, epochs=5000)

    os.makedirs("models", exist_ok=True)
    with open("models/cbow_word2vec_optimized.model", "wb") as f:
        pickle.dump({
            "model": model,
            "word2idx": word2idx,
            "idx2word": idx2word
        }, f)

    print("优化版模型已保存：models/cbow_word2vec_optimized.model")
    print("正在绘制优化后的词向量图...")
    plot_embedding(model, word2idx, idx2word)