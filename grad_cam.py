import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import training


class GradCAM:
	def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
		self.model = model
		self.target_layer = target_layer
		self.activations = None
		self.gradients = None

		self._forward_handle = target_layer.register_forward_hook(self._save_activations)
		self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

	def _save_activations(self, _module, _input, output) -> None:
		self.activations = output.detach()

	def _save_gradients(self, _module, _grad_input, grad_output) -> None:
		self.gradients = grad_output[0].detach()

	def remove_hooks(self) -> None:
		self._forward_handle.remove()
		self._backward_handle.remove()

	def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
		self.model.zero_grad(set_to_none=True)
		logits = self.model(x)
		if class_idx is None:
			class_idx = int(torch.argmax(logits, dim=1).item())

		loss = logits[:, class_idx].sum()
		loss.backward()

		weights = self.gradients.mean(dim=(2, 3), keepdim=True)
		cam = (weights * self.activations).sum(dim=1, keepdim=True)
		cam = torch.relu(cam)
		cam = cam.squeeze().cpu().numpy()
		cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
		return cam, class_idx


def load_image(image_path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
	transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)
	img = Image.open(image_path).convert("RGB")
	input_tensor = transform(img).unsqueeze(0)
	return input_tensor, np.array(img)


def overlay_cam(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
	cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0])))
	heatmap = plt.get_cmap("jet")(cam_resized / 255.0)[:, :, :3]
	overlay = (0.4 * heatmap + 0.6 * (img / 255.0))
	overlay = np.clip(overlay, 0, 1)
	return overlay


def main() -> None:
	parser = argparse.ArgumentParser(description="Grad-CAM visualization for STL10 model")
	parser.add_argument("--image", required=True, help="Path to input image")
	parser.add_argument("--model", default="./outputs/best_model.pt", help="Path to model checkpoint")
	parser.add_argument("--output", default="./outputs/grad_cam.png", help="Output image path")
	parser.add_argument("--class-idx", type=int, default=None, help="Target class index")
	parser.add_argument("--activation", default="relu", help="Model activation name")
	parser.add_argument("--pool-type", default="max", help="Model pool type: max or avg")
	parser.add_argument(
		"--no-batch-norm",
		action="store_true",
		help="Disable batch norm",
	)
	parser.add_argument("--dropout-p", type=float, default=0.3, help="Dropout probability")
	args = parser.parse_args()

	cfg = training.Config()
	cfg.activation = args.activation
	cfg.pool_type = args.pool_type
	cfg.use_batch_norm = not args.no_batch_norm
	cfg.dropout_p = args.dropout_p

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, _, _, class_names = training.build_dataloaders(cfg)

	model = training.build_model(cfg, num_classes=len(class_names)).to(device)
	model.load_state_dict(torch.load(args.model, map_location=device))
	model.eval()

	input_tensor, raw_img = load_image(args.image, cfg.image_size)
	input_tensor = input_tensor.to(device)

	target_layer = model.features[11]
	cam = GradCAM(model, target_layer)
	cam_map, class_idx = cam(input_tensor, args.class_idx)
	cam.remove_hooks()

	overlay = overlay_cam(raw_img, cam_map)
	plt.figure(figsize=(6, 6))
	plt.axis("off")
	plt.title(f"Grad-CAM: {class_names[class_idx]}")
	plt.imshow(overlay)
	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	plt.savefig(args.output, bbox_inches="tight", pad_inches=0.1)
	plt.close()

	print("Saved Grad-CAM to", args.output)


if __name__ == "__main__":
	main()
