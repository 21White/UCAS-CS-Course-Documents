import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
from tqdm import tqdm

# ----------------------------
# 参数配置
# ----------------------------
CONTEXT_SIZE = 2  # 上下文窗口大小
EMBEDDING_DIM = 128  # 词向量维度
EPOCHS = 256
LR = 0.001
MIN_COUNT = 1
BATCH_SIZE = 128

# ----------------------------
# 设备检测
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print(f"使用设备：{device}")
print("=" * 50)


# ----------------------------
# 读取语料
# ----------------------------
def read_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines if line.strip()]


# ----------------------------
# 构建词典
# ----------------------------
def build_vocab(corpus):
    counter = Counter([w for sent in corpus for w in sent])
    vocab = [w for w, c in counter.items() if c >= MIN_COUNT]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


# ----------------------------
# CBOW 数据生成
# ----------------------------
def make_cbow_data(corpus, word2idx):
    data = []
    for sent in corpus:
        idxs = [word2idx[w] for w in sent if w in word2idx]
        for i in range(CONTEXT_SIZE, len(idxs) - CONTEXT_SIZE):
            context = idxs[i - CONTEXT_SIZE:i] + idxs[i + 1:i + CONTEXT_SIZE + 1]
            target = idxs[i]
            data.append((context, target))
    return data


# ----------------------------
# CBOW 模型定义
# ----------------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, context):
        embeds = self.embeddings(context)
        avg_embeds = torch.mean(embeds, dim=1)
        out = self.linear1(avg_embeds)
        out = self.activation1(out)
        out = self.linear2(out)
        return out


# ----------------------------
# 模型训练
# ----------------------------
def train_word2vec(path, lang="zh"):
    print(f"\n开始训练 {lang} 语料的 Word2Vec 模型")

    corpus = read_corpus(path)
    word2idx, idx2word = build_vocab(corpus)
    data = make_cbow_data(corpus, word2idx)
    vocab_size = len(word2idx)
    print(f"词汇量: {vocab_size}, 训练样本: {len(data)}")

    model = CBOW(vocab_size, EMBEDDING_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(data)

        for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            batch = data[i:i + BATCH_SIZE]
            context_batch = [c for c, _ in batch]
            target_batch = [t for _, t in batch]

            context_batch = torch.tensor(context_batch, dtype=torch.long).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)

            outputs = model(context_batch)
            loss = criterion(outputs, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(data) / BATCH_SIZE)
        print(f"Epoch {epoch + 1}/{EPOCHS} 平均Loss: {avg_loss:.4f}")

    torch.save(model.embeddings.weight.data.cpu(), f"word2vec_{EPOCHS}_{lang}.pt")
    print(f"{lang} 模型训练完成，词向量已保存至 word2vec_{EPOCHS}_{lang}.pt\n")


# ----------------------------
# 主函数
# ----------------------------
if __name__ == "__main__":
    train_word2vec("data/zh.txt", "zh")
    train_word2vec("data/en.txt", "en")
