import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class Config:
	data_dir: str = "./STL10"
	train_dir: str = "train"
	test_dir: str = "test"
	image_size: int = 96
	use_data_augmentation: bool = True
	activation: str = "relu"
	pool_type: str = "max"
	use_batch_norm: bool = True
	dropout_p: float = 0.3
	batch_size: int = 64
	num_workers: int = 2
	epochs: int = 20
	lr: float = 1e-3
	weight_decay: float = 1e-4
	optimizer_name: str = "adam"
	momentum: float = 0.9
	val_ratio: float = 0.2
	seed: int = 30
	output_dir: str = "./outputs"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# 激活函数和池化层工厂函数
def activation_layer(name: str) -> nn.Module:
	name = name.lower()
	if name == "relu":
		return nn.ReLU(inplace=True)
	if name == "leaky_relu":
		return nn.LeakyReLU(negative_slope=0.1, inplace=True)
	if name == "tanh":
		return nn.Tanh()
	if name == "sigmoid":
		return nn.Sigmoid()
	raise ValueError(f"Unsupported activation: {name}")


def pool_layer(name: str) -> nn.Module:
	name = name.lower()
	if name == "max":
		return nn.MaxPool2d(kernel_size=2)
	if name == "avg":
		return nn.AvgPool2d(kernel_size=2)
	raise ValueError(f"Unsupported pool type: {name}")

#卷积神经网络模型定义
class STL10CNN(nn.Module):
	def __init__(
		self,
		num_classes: int,
		activation: str = "relu",
		pool_type: str = "max",
		use_batch_norm: bool = True,
		dropout_p: float = 0.3,
	) -> None:
		super().__init__()
		act = lambda: activation_layer(activation)
		pool = pool_layer(pool_type)
		bn32 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
		bn64 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
		bn128 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
		bn256 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			bn32,
			act(),
			pool,

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			bn64,
			act(),
			pool_layer(pool_type),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			bn128,
			act(),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			bn256,
			act(),
		)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(p=dropout_p),
			nn.Linear(256, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.pool(x)
		return self.classifier(x)

#数据加载、模型构建、优化器构建、训练循环等函数定义
def build_dataloaders(cfg: Config):
	train_transforms = [transforms.Resize((cfg.image_size, cfg.image_size))]
	if cfg.use_data_augmentation:
		train_transforms += [
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(cfg.image_size, padding=4),
		]
	train_transforms += [
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	]
	train_transform = transforms.Compose(train_transforms)
	test_transform = transforms.Compose(
		[
			transforms.Resize((cfg.image_size, cfg.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)

	train_path = os.path.join(cfg.data_dir, cfg.train_dir)
	test_path = os.path.join(cfg.data_dir, cfg.test_dir)
	full_train = datasets.ImageFolder(train_path, transform=train_transform)

	val_size = int(len(full_train) * cfg.val_ratio)
	train_size = len(full_train) - val_size
	train_set, val_set = random_split(
		full_train,
		[train_size, val_size],
		generator=torch.Generator().manual_seed(cfg.seed),
	)

	test_set = datasets.ImageFolder(test_path, transform=test_transform)

	train_loader = DataLoader(
		train_set,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_set,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)
	test_loader = DataLoader(
		test_set,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)

	return train_loader, val_loader, test_loader, full_train.classes


def build_model(cfg: Config, num_classes: int) -> nn.Module:
	return STL10CNN(
		num_classes=num_classes,
		activation=cfg.activation,
		pool_type=cfg.pool_type,
		use_batch_norm=cfg.use_batch_norm,
		dropout_p=cfg.dropout_p,
	)


def build_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
	name = cfg.optimizer_name.lower()
	if name == "adam":
		return optim.Adam(
			model.parameters(),
			lr=cfg.lr,
			weight_decay=cfg.weight_decay,
		)
	if name == "sgd":
		return optim.SGD(
			model.parameters(),
			lr=cfg.lr,
			momentum=cfg.momentum,
			weight_decay=cfg.weight_decay,
		)
	raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
	if train:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	all_preds = []
	all_labels = []

	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		if train:
			optimizer.zero_grad()

		with torch.set_grad_enabled(train):
			outputs = model(images)
			loss = criterion(outputs, labels)
			if train:
				loss.backward()
				optimizer.step()

		total_loss += loss.item() * images.size(0)
		preds = outputs.argmax(dim=1)
		all_preds.append(preds.detach().cpu())
		all_labels.append(labels.detach().cpu())

	all_preds = torch.cat(all_preds).numpy()
	all_labels = torch.cat(all_labels).numpy()
	avg_loss = total_loss / len(loader.dataset)
	acc = accuracy_score(all_labels, all_preds)
	return avg_loss, acc

#训练主函数、结果可视化、测试评估等函数定义
def plot_curves(history, output_dir):
	epochs = range(1, len(history["train_loss"]) + 1)

	plt.figure(figsize=(8, 6))
	plt.plot(epochs, history["train_loss"], label="Train Loss")
	plt.plot(epochs, history["val_loss"], label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss Curves")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "loss_curves.png"))
	plt.close()

	plt.figure(figsize=(8, 6))
	plt.plot(epochs, history["train_acc"], label="Train Acc")
	plt.plot(epochs, history["val_acc"], label="Val Acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy Curves")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "accuracy_curves.png"))
	plt.close()


def evaluate_test(model, loader, device, class_names, output_dir):
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			outputs = model(images)
			preds = outputs.argmax(dim=1).cpu().numpy()
			all_preds.append(preds)
			all_labels.append(labels.numpy())

	all_preds = np.concatenate(all_preds)
	all_labels = np.concatenate(all_labels)

	report = classification_report(
		all_labels,
		all_preds,
		target_names=class_names,
		digits=4,
	)
	precision, recall, f1, _ = precision_recall_fscore_support(
		all_labels,
		all_preds,
		average="macro",
		zero_division=0,
	)
	acc = accuracy_score(all_labels, all_preds)
	cm = confusion_matrix(all_labels, all_preds)

	with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
		f.write(report)
		f.write("\n")
		f.write(f"Macro Precision: {precision:.4f}\n")
		f.write(f"Macro Recall: {recall:.4f}\n")
		f.write(f"Macro F1: {f1:.4f}\n")
		f.write(f"Accuracy: {acc:.4f}\n")

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45, ha="right")
	plt.yticks(tick_marks, class_names)
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
	plt.close()

	return report, acc

#主函数定义
def main():
	cfg = Config()
	cfg.use_data_augmentation = False
	set_seed(cfg.seed)
	os.makedirs(cfg.output_dir, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg)

	model = build_model(cfg, num_classes=len(class_names)).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = build_optimizer(cfg, model)

	history = {
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
	}

	best_val_acc = 0.0
	best_path = os.path.join(cfg.output_dir, "best_model.pt")

	for epoch in range(1, cfg.epochs + 1):
		train_loss, train_acc = run_epoch(
			model, train_loader, criterion, optimizer, device, train=True
		)
		val_loss, val_acc = run_epoch(
			model, val_loader, criterion, optimizer, device, train=False
		)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_path)

		print(
			f"Epoch {epoch:02d}/{cfg.epochs} "
			f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
			f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
		)

	plot_curves(history, cfg.output_dir)

	model.load_state_dict(torch.load(best_path, map_location=device))
	report, test_acc = evaluate_test(
		model, test_loader, device, class_names, cfg.output_dir
	)
	print("\nTest Accuracy:", f"{test_acc:.4f}")
	print("\nClassification Report:\n", report)


if __name__ == "__main__":
	main()
