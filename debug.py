from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.bfloat16,
    device_map="cpu"
)

# Check the structure
print("Model attributes:", dir(model))
print("\nmodel.model type:", type(model.model))
print("model.model attributes:", [a for a in dir(model.model) if not a.startswith('_')])

# Try to find layers
if hasattr(model.model, 'layers'):
    print("\n✓ model.model.layers exists!")
    print(f"  Number of layers: {len(model.model.layers)}")
elif hasattr(model.model, 'text_model'):
    print("\n✓ model.model.text_model exists!")
    if hasattr(model.model.text_model, 'layers'):
        print(f"  Number of layers: {len(model.model.text_model.layers)}")
    else:
        print("  text_model attributes:", [a for a in dir(model.model.text_model) if not a.startswith('_')])
else:
    print("\n⚠ Need to find where layers are...")
    print("Checking common paths:")
    for path in ['encoder', 'decoder', 'transformer', 'language_model']:
        if hasattr(model.model, path):
            print(f"  ✓ model.model.{path} exists")
            obj = getattr(model.model, path)
            if hasattr(obj, 'layers'):
                print(f"    → model.model.{path}.layers found!")
                print(f"    → Number of layers: {len(obj.layers)}")
            if hasattr(obj, 'model') and hasattr(obj.model, 'layers'):
                print(f"    → model.model.{path}.model.layers found!")
                print(f"    → Number of layers: {len(obj.model.layers)}")
            
            # Show attributes of language_model
            if path == 'language_model':
                print(f"    → language_model attributes: {[a for a in dir(obj) if not a.startswith('_')][:20]}")