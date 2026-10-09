import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# Identificador del modelo en Hugging Face (Qwen 3 14B)
modelo_id = "Qwen/Qwen3-14B" # Reemplaza con el path exacto de la versión base o Coder

# ==========================================
# 1. Aprovisionamiento del Entorno
# ==========================================
# (Se asume la instalación previa de: pip install torch transformers peft bitsandbytes trl datasets)

# Cargar el Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen a veces requiere definir el pad_token explícitamente

# ==========================================
# 2. Carga Cuantizada del Modelo Base
# ==========================================
# Configuración QLoRA para comprimir el modelo a 4-bits y ahorrar VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", # NormalFloat4 (Estándar de QLoRA)
    bnb_4bit_compute_dtype=torch.bfloat16 # Mantener cálculos en 16-bits
)

modelo_base = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=bnb_config,
    device_map="auto", # Asigna automáticamente a la GPU disponible
    trust_remote_code=True
)

# Preparar la arquitectura para entrenamiento en 4-bits
modelo_base = prepare_model_for_kbit_training(modelo_base)

# ==========================================
# 3. Configuración de los Hiperparámetros (Adaptador LoRA)
# ==========================================
peft_config = LoraConfig(
    r=8, # Rango del adaptador (Bajo para ahorrar RAM, 8 o 16 es estándar)
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Capas de atención de Qwen a entrenar
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

modelo_peft = get_peft_model(modelo_base, peft_config)
modelo_peft.print_trainable_parameters() # Verifica qué porcentaje se entrenará (usualmente < 1%)

# ==========================================
# 4. Mapeo del Dataset Sintético
# ==========================================
# Cargar el dataset en formato JSONL generado en tu Bootstrapping
dataset = load_dataset("json", data_files="dataset_titin_sintetico.jsonl", split="train")

# Función para formatear el dataset inyectando el esquema en el prompt (Context Window Packing)
def formatear_prompt(ejemplo):
    prompt = f"""<|im_start|>system
Eres Titin, un experto en diagnóstico SQL. Evalúa la consulta proporcionada utilizando el siguiente esquema DDL.
DDL:
{ejemplo['entrada_esquema_ddl']}<|im_end|>
<|im_start|>user
Pregunta: {ejemplo['entrada_lenguaje_natural']}
SQL a evaluar: {ejemplo['entrada_sql_evaluar']}<|im_end|>
<|im_start|>assistant
Validación: {ejemplo['salida_validacion']}
Tipo de Error: {', '.join(ejemplo['salida_tipo_error'])}
Diagnóstico: {ejemplo['salida_feedback']}
Consulta Correcta: {ejemplo['entrada_sql_correcto']}<|im_end|>"""
    return {"text": prompt}

dataset_formateado = dataset.map(formatear_prompt)

# ==========================================
# 5. Configuración y Ejecución del Entrenamiento[cite: 59]
# ==========================================
# Argumentos optimizados ESTRICTAMENTE para tarjetas de 12GB VRAM
argumentos_entrenamiento = TrainingArguments(
    output_dir="./resultados_titin_qwen14b",
    per_device_train_batch_size=1, # CRÍTICO: 1 para no desbordar los 12GB de VRAM
    gradient_accumulation_steps=8, # Acumula gradientes para simular un batch size de 8
    optim="paged_adamw_32bit", # Optimizado paginado transfiere picos de memoria a la RAM de la CPU
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True, # Usa precisión mixta fp16
    max_grad_norm=0.3,
    max_steps=200, # Ajusta según el tamaño de tu dataset
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

trainer = SFTTrainer(
    model=modelo_peft,
    train_dataset=dataset_formateado,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024, # CRÍTICO: Limita la longitud del contexto para ahorrar VRAM
    tokenizer=tokenizer,
    args=argumentos_entrenamiento
)

# Iniciar el entrenamiento
print("Iniciando ajuste fino de Titin...")
trainer.train()

# ==========================================
# 6. Exportación del Adaptador
# ==========================================
ruta_adaptador = "./titin-adaptador-qwen14b"
trainer.model.save_pretrained(ruta_adaptador)
tokenizer.save_pretrained(ruta_adaptador)
print(f"Adaptador LoRA guardado exitosamente en: {ruta_adaptador}")