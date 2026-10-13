# CLUE TNEWS 中文新闻分类模型对比实验

主实验 Notebook：`tnews_model_comparison.ipynb`

## 运行方式

推荐在 Colab 或 Kaggle GPU 中运行：

1. 打开 `tnews_model_comparison.ipynb`。
2. 首次运行时将第一段代码中的 `INSTALL_DEPS` 改为 `True`，安装依赖后按提示重启运行时。
3. Notebook 默认使用正式分层子集；需要先做冒烟测试时，将 `FAST_MODE` 改为 `True`，确认全流程后改回 `False`。
4. 按 Notebook 顺序执行全部单元。模型会自动下载 CLUE TNEWS 和 `hfl/chinese-roberta-wwm-ext`。

正式实验包含 TF-IDF + LinearSVC、TextCNN、RoBERTa + LoRA 和 RoBERTa + Houlsby Adapter。默认固定随机种子为 42，并对重复标题去重、分层划分数据集和限制每类样本数。如果直接加载 CLUE 配置失败，Notebook 会自动回退到同一数据集仓库中的 TNEWS Parquet 文件。

## 输出文件

运行完成后，结果保存到 `artifacts/`：

- `model_comparison.csv`：模型指标、参数量、训练时间和推理速度
- `classification_reports.json`：各模型分类报告
- `metric_comparison.png`：Accuracy、Macro-F1、Weighted-F1 对比图
- `training_curves.png`：验证集 Macro-F1 曲线
- `best_confusion_matrix.png`：最优模型混淆矩阵
- `best_per_class_f1.png`：最优模型各类别 F1
- `tfidf_linearsvc.pkl`、`textcnn.pt`、`roberta_lora/`、`roberta_adapter/`：训练后模型权重

本机若没有 CUDA，Notebook 会自动使用更小的分层子集以避免 CPU 运行时间过长；课程正式结果建议使用 GPU。
