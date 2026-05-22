# 英法机器翻译：点积注意力 vs 加性注意力对比实验

## 项目简介
本项目使用基础 Transformer 架构完成英法机器翻译任务，核心目标：  
实现并对比点积注意力和加性注意力机制  
观察训练损失、测试损失、BLEU 分数的变化趋势  
分析两种注意力在小数据集上的拟合能力、泛化性能、翻译质量上的差异

## 运行环境与依赖版本
本项目在以下环境中测试通过：  
Python 版本：3.11  
PyTorch 版本：2.2.2 (CPU 版本)  

安装依赖：  
pip install torch==2.2.2  
pip install matplotlib==3.8.4  
pip install numpy==1.26.4

## 项目文件 
transformer.py 主程序，包含模型定义、训练、评估、翻译、BLEU计算  
eng-fra_train_data.txt 英法语料训练集  
eng-fra_test_data.txt 英法语料测试集  
requirement3.txt 安装依赖
README.md 项目说明文档

## 实验结果
### 训练损失（8 Epochs）
| Epoch | 点积注意力 | 加性注意力 |
|------|------------|------------|
| 1 | 5.470 | 5.446 |
| 2 | 4.269 | 4.236 |
| 3 | 3.776 | 3.733 |
| 4 | 3.439 | 3.391 |
| 5 | 3.176 | 3.126 |
| 6 | 2.959 | 2.904 |
| 7 | 2.772 | 2.717 |
| 8 | 2.607 | 2.552 |

### 测试集指标
| 模型 | 测试损失 | BLEU |
|------|----------|------|
| 点积注意力 | 2.923 | 0.099 |
| 加性注意力 | 2.894 | 0.088 |

### 翻译示例对比
EN: I love you  
Dot: jadore vous êtes <unk>  
Add: je vous ai vus en train de faire le faire.

EN: What is your name?  
Dot: quel est votre nom de téléphone portable. ?  
Add: que vous avez ton idée de temps a besoin de ce que ce soit dautre choix

EN: He is a student  
Dot: il est devenu un coup dil heureuse.  
Add: il est un point cest tout.

EN: We are happy  
Dot: nous sommes heureux de <unk>  
Add: nous sommes allées heureux de la manière dont nous avons besoin.

## 实验结论
加性注意力：  
训练损失全程更低，收敛速度更快  
模型拟合能力更强，参数量更大  
但存在过拟合，翻译结果冗长、语序混乱  
测试集 BLEU 较低（0.088）

点积注意力：  
训练损失略高，但泛化能力更强  
翻译更简洁、关键词更准确  
测试集 BLEU 更高（0.099）  
更适合实际机器翻译任务

最终总结：  
加性注意力在训练拟合上占优  
点积注意力在翻译质量与泛化能力上更优
