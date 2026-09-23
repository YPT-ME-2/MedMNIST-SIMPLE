import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score
)



def evaluate(model,loader,device):
    model.eval()

    all_labels = []
    all_scores = []
    all_predictions = []

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            outputs=model(images)

            probabilities = torch.softmax(outputs, dim=1)###???
            scores = probabilities[:, 1]       # 属于“肺炎”的概率
            predictions = outputs.argmax(dim=1)  # 预测的类别

            all_labels.extend(labels.view(-1).tolist())
            all_scores.extend(scores.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    # auc = roc_auc_score(all_labels, all_scores)
    # acc = accuracy_score(all_labels, all_predictions)
    # return auc, acc
    auc = roc_auc_score(all_labels, all_scores)
    acc = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, pos_label=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1)#pos_label=1 指定“肺炎”为关注的正类
#二分类里要先指定哪一类算“正类”，才能计算它的 Recall 和 F1。数据标签是 0=正常、1=肺炎
    return auc, acc, recall, f1