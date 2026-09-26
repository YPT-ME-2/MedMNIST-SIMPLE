# MedMNIST Pneumonia Classification

使用 PyTorch 和 ResNet18 对 MedMNIST 中的 PneumoniaMNIST 胸部 X 光图像进行二分类。

## 项目功能

- 加载 PneumoniaMNIST 数据集
- 使用 ResNet18 进行训练
- 支持验证集和测试集评估
- 计算 Accuracy、AUC、Recall 和 F1-score
- 使用 YAML 文件管理训练参数

## 项目结构

```text
MedMNIST-SIMPLE/
├── config.yaml       # 训练和模型配置
├── data.py           # 数据集与 DataLoader
├── models.py         # ResNet18 模型V
├── train.py          # 训练入口
├── utils.py          # 模型评估函数
├── requirements.txt  # Python 依赖
└── README.md         # 项目说明