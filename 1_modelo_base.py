import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Definir el modelo base que quieres ejecutar
modelo_id = "Qwen/Qwen3-14B"  # Reemplaza por el modelo exacto que vayas a usar

print("Cargando el tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)

# 2. Configurar la cuantización a 4-bits para que quepa en tus 12GB de VRAM
print("Configurando cuantización a 4-bits...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Cargar el modelo en la tarjeta gráfica (VRAM)
print("Cargando el modelo en la VRAM (esto puede tomar unos minutos)...")
modelo_base = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=bnb_config,
    device_map="auto",  # Asigna automáticamente el modelo a la GPU disponible
    trust_remote_code=True
)
print("¡Modelo cargado exitosamente!")

# 4. Preparar tu prompt de prueba (emulando la inyección del esquema DDL)
ddl_prueba = "CREATE TABLE cliente (id INT PRIMARY KEY, nombre VARCHAR(50));"
sql_erroneo = "SELECT apellido FROM cliente WHERE nombre = 'Juan';"

prompt = f"""<|im_start|>system
Eres un asistente experto en SQL. Revisa la siguiente consulta en base al DDL.
DDL: {ddl_prueba}<|im_end|>
<|im_start|>user
SQL a evaluar: {sql_erroneo}<|im_end|>
<|im_start|>assistant
"""

# 5. Tokenizar la entrada y enviarla a la GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 6. Generar la respuesta del modelo
print("\nGenerando respuesta...")
outputs = modelo_base.generate(
    **inputs,
    max_new_tokens=200,      # Límite de palabras en la respuesta
    temperature=0.1,         # Temperatura baja (0.1) para que sea más determinista y lógico
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 7. Decodificar y mostrar el resultado
respuesta = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n--- RESPUESTA DEL MODELO BASE ---")
print(respuesta)
print("---------------------------------")