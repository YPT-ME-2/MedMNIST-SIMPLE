# import torch
# import yaml

# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from copy import deepcopy
# from pathlib import Path

# from medmnist import PneumoniaMNIST

# from models import ResNet18
# from utils import evaluate


# ROOT = Path(__file__).resolve().parent


# def train_one_epoch(model, loader, loss_fn, optimizer, device):
#     model.train()

#     total_loss = 0.0

#     for images, labels in loader:
#         images = images.to(device)
#         labels = labels.view(-1).long().to(device)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = loss_fn(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * images.size(0)

#     train_loss = total_loss / len(loader.dataset)

#     return train_loss


# def main():
#     # 1. 读取配置
#     config_path = ROOT / "config.yaml"

#     with open(config_path, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     epochs = config["train"]["epochs"]
#     batch_size = config["train"]["batch_size"]
#     lr = config["train"]["lr"]

#     in_channels = config["model"]["in_channels"]
#     num_classes = config["model"]["num_classes"]

#     # 2. 随机种子
#     torch.manual_seed(42)

#     # 3. 设备
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     print(f"Using: {device}")

#     # 4. 数据预处理
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.5],
#             std=[0.5]
#         )
#     ])

#     # 5. 数据集
#     train_set = PneumoniaMNIST(
#         split="train",
#         transform=transform,
#         download=False
#     )

#     val_set = PneumoniaMNIST(
#         split="val",
#         transform=transform,
#         download=False
#     )

#     test_set = PneumoniaMNIST(
#         split="test",
#         transform=transform,
#         download=False
#     )

#     # 6. DataLoader
#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     val_loader = DataLoader(
#         val_set,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     # 7. 模型
#     model = ResNet18(
#         in_channels=in_channels,
#         num_classes=num_classes
#     ).to(device)

#     # 8. 损失函数
#     loss_fn = nn.CrossEntropyLoss()

#     # 9. 优化器
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=lr
#     )

#     # 10. 学习率调度器
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer,
#         milestones=[
#             int(0.5 * epochs),
#             int(0.75 * epochs)
#         ],
#         gamma=0.1
#     )

#     # 后面再写训练、验证、测试
#     best_val_auc = float("-inf")
#     best_state = None
#     best_epoch = 0

#     for epoch in range(epochs):

#         train_loss = train_one_epoch(
#             model,
#             train_loader,
#             loss_fn,
#             optimizer,
#             device
#         )

#         val_auc, val_acc, val_recall, val_f1 = evaluate(
#             model,
#             val_loader,
#             device
#         )

#         scheduler.step()

#         print(
#             f"Epoch {epoch + 1:3d}/{epochs} | "
#             f"train loss {train_loss:.4f} | "
#             f"val AUC {val_auc:.5f} | "
#             f"val ACC {val_acc:.5f}"
#         )

#         if val_auc > best_val_auc:
#             best_val_auc = val_auc

#             best_state = deepcopy(
#                 model.state_dict()
#             )

#             best_epoch = epoch + 1
#         model.load_state_dict(best_state)

#     output_dir = (
#         ROOT
#         / "output"
#         / "simple_resnet18"
#     )

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     model_path = (
#         output_dir
#         / "best_model.pth"
#     )

#     torch.save(
#         best_state,
#         model_path
#     )
#     test_auc, test_acc, test_recall, test_f1 = evaluate(
#         model,
#         test_loader,
#         device
#     )
#     print(
#         f"Best epoch: {best_epoch}, "
#         f"val AUC: {best_val_auc:.5f}"
#     )

#     print(
#         f"Test AUC: {test_auc:.5f}, "
#         f"ACC: {test_acc:.5f}, "
#         f"Recall: {test_recall:.5f}, "
#         f"F1: {test_f1:.5f}"
#     )

#     print(
#         f"Saved model: {model_path}"
#     )
# if __name__ == "__main__":
#     main()

import torch
import yaml
from models import ResNet18
from utils import evaluate
from data import build_dataloaders
from torch import nn
from copy import deepcopy
from pathlib import Path



ROOT = Path(__file__).resolve().parent


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(loader.dataset)

    return train_loss


def main():

    # =========================
    # 1. 配置
    # =========================

    config_path = ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    lr = config["train"]["lr"]

    in_channels = config["model"]["in_channels"]
    num_classes = config["model"]["num_classes"]

    torch.manual_seed(42)

    # =========================
    # 2. Device
    # =========================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using: {device}")

    # =========================
    # 3. 数据
    # =========================

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size
    )

    # =========================
    # 4. 模型
    # =========================

    model = ResNet18(
        in_channels=in_channels,
        num_classes=num_classes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.5 * epochs),
            int(0.75 * epochs)
        ],
        gamma=0.1
    )

    # =========================
    # 5. Train + Validation
    # =========================

    best_val_auc = float("-inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_auc, val_acc, _, _ = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val AUC {val_auc:.5f} | "
            f"val ACC {val_acc:.5f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1

    # =========================
    # 6. 加载最好模型
    # =========================

    model.load_state_dict(best_state)

    # =========================
    # 7. 保存模型
    # =========================

    output_dir = ROOT / "output" / "simple_resnet18"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir / "best_model.pth"

    torch.save(
        best_state,
        model_path
    )

    # =========================
    # 8. Test
    # =========================

    test_auc, test_acc, test_recall, test_f1 = evaluate(
        model,
        test_loader,
        device
    )

    print(
        f"Best epoch: {best_epoch}, "
        f"val AUC: {best_val_auc:.5f}"
    )

    print(
        f"Test AUC: {test_auc:.5f}, "
        f"ACC: {test_acc:.5f}, "
        f"Recall: {test_recall:.5f}, "
        f"F1: {test_f1:.5f}"
    )

    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()