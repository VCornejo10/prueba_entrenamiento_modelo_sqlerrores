import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

modelo_id = "Qwen/Qwen3-14B"
ruta_adaptador = "./titin-adaptador-qwen14b"
ruta_salida_fusion = "./titin-qwen14b-fused"

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)

print("Cargando modelo base en precisión completa (bfloat16 en CPU/RAM)...")
# Se carga sin cuantización de 4 bits para permitir la fusión algebraica de pesos
modelo_base = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Ejecútalo en la RAM del sistema para no colapsar la VRAM de 12GB
    trust_remote_code=True
)

print("Cargando adaptador LoRA...")
modelo_lora = PeftModel.from_pretrained(modelo_base, ruta_adaptador)

print("Fusionando pesos (merge_and_unload)...")
modelo_fusionado = modelo_lora.merge_and_unload()

print(f"Guardando modelo fusionado en: {ruta_salida_fusion}")
modelo_fusionado.save_pretrained(ruta_salida_fusion, safe_serialization=True)
tokenizer.save_pretrained(ruta_salida_fusion)
print("Fusión completada con éxito.")