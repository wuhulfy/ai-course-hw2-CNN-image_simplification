import copy
import os
import time

import training


EXPERIMENTS = [
	{
		"name": "baseline",
		"updates": {
			"output_dir": "./outputs/baseline",
		},
	},
	{
		"name": "aug_on",
		"updates": {
			"use_data_augmentation": True,
			"output_dir": "./outputs/aug_on",
		},
	},
	{
		"name": "model_variant",
		"updates": {
			"activation": "leaky_relu",
			"pool_type": "avg",
			"dropout_p": 0.5,
			"output_dir": "./outputs/model_variant",
		},
	},
	{
		"name": "optim_variant",
		"updates": {
			"optimizer_name": "sgd",
			"lr": 0.01,
			"weight_decay": 5e-4,
			"output_dir": "./outputs/optim_variant",
		},
	},
]


def build_config(base_cfg: training.Config, updates: dict) -> training.Config:
	cfg = copy.deepcopy(base_cfg)
	for key, value in updates.items():
		setattr(cfg, key, value)
	return cfg


def run_experiment(cfg: training.Config) -> float:
	training.set_seed(cfg.seed)
	os.makedirs(cfg.output_dir, exist_ok=True)

	device = training.torch.device(
		"cuda" if training.torch.cuda.is_available() else "cpu"
	)
	train_loader, val_loader, test_loader, class_names = training.build_dataloaders(cfg)

	model = training.build_model(cfg, num_classes=len(class_names)).to(device)
	criterion = training.nn.CrossEntropyLoss()
	optimizer = training.build_optimizer(cfg, model)

	history = {
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
	}

	best_val_acc = 0.0
	best_path = os.path.join(cfg.output_dir, "best_model.pt")

	for epoch in range(1, cfg.epochs + 1):
		train_loss, train_acc = training.run_epoch(
			model, train_loader, criterion, optimizer, device, train=True
		)
		val_loss, val_acc = training.run_epoch(
			model, val_loader, criterion, optimizer, device, train=False
		)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			training.torch.save(model.state_dict(), best_path)

		print(
			f"Epoch {epoch:02d}/{cfg.epochs} "
			f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
			f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
		)

	training.plot_curves(history, cfg.output_dir)

	model.load_state_dict(training.torch.load(best_path, map_location=device))
	_, test_acc = training.evaluate_test(
		model, test_loader, device, class_names, cfg.output_dir
	)
	print("\nTest Accuracy:", f"{test_acc:.4f}")
	return test_acc


def main():
	base_cfg = training.Config()
	results = []

	for exp in EXPERIMENTS:
		name = exp["name"]
		updates = exp.get("updates", {})
		cfg = build_config(base_cfg, updates)

		print("=" * 60)
		print(f"Running experiment: {name}")
		start = time.time()
		acc = run_experiment(cfg)
		elapsed = time.time() - start
		results.append((name, acc, elapsed))

	print("\nSummary")
	for name, acc, elapsed in results:
		print(f"{name:15s} acc={acc:.4f} time={elapsed:.1f}s")


if __name__ == "__main__":
	main()
