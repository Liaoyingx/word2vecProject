# word2vecProject

## 项目文件说明
### 1. 主程序文件
**word2vec.py**
项目核心运行文件，包含：
- 语料加载与自动分词
- 词汇表构建
- CBOW 模型实现前向传播、反向传播、梯度下降
- 模型训练与损失输出
- 模型保存
- PCA 降维与词向量可视化

### 2. 数据文件
**corpus.txt**
训练用中文语料库，选自人民日报《推动海洋经济高质量发展》。
- 原始中文句子，无需提前分词
- 代码完成分词、去标点、去数字、清洗无效字符

### 3. 模型文件
**models/cbow_word2vec_optimized.model**
训练完成后保存的词向量模型，pickle 格式存储，包含：
- 训练好的词向量权重矩阵
- 词到索引映射字典 word2idx
- 索引到词映射字典 idx2word

### 4. 依赖文件
**requirement.txt**
项目运行所需第三方库：
- numpy：矩阵运算、模型参数初始化
- matplotlib：词向量分布图绘制
- scikit-learn：PCA 降维
- jieba：中文自动分词

---
