import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import numpy as np
import matplotlib
from pathlib import Path

# 1. Choose device: Apple Silicon uses "mps"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# 2. Load model & processor from Hugging Face
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 3. Load your image
image_path = Path("images.webp")  # change name if needed
image = Image.open(image_path).convert("RGB")

# 4. Prepare inputs (text-only prompt example: find "ear")
inputs = processor(
    images=image,
    text="ear",
    return_tensors="pt"
).to(device)

# 5. Run model
with torch.no_grad():
    outputs = model(**inputs)

# 6. Post-process to get masks, boxes, scores
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs["original_sizes"].tolist(),
)[0]

masks = results["masks"]      # [num_objects, H, W]
boxes = results["boxes"]      # [num_objects, 4] (xyxy)
scores = results["scores"]    # [num_objects]

print(f"Found {len(masks)} objects")

# 7. Helper: overlay masks on the image and save result
def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask_arr, color in zip(masks_np, colors):
        mask = Image.fromarray(mask_arr)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image

composite = overlay_masks(image, masks)
out_path = Path("cat_segmented.png")
composite.save(out_path)
print(f"Saved result to {out_path.resolve()}")
