from ollama import Client
import datetime
import json

client = Client(
    host="http://192.168.11.102:11500",
)

models = client.list()

# Pretty-print the models
model_list = models["models"]
print(f"Found {len(model_list)} models")
for model in model_list:
    print(f"{model.model}")
    print(f"  Model: {model.model}")
    print(f"  Size: {model.size / (1024**3):.2f} GB")  # Convert bytes to GB
    print(f"  Format: {model.details.format}")
    print(f"  Family: {model.details.family}")
    print(f"  Parameter Size: {model.details.parameter_size}")
    print(f"  Quantization: {model.details.quantization_level}")
    print("-" * 40)