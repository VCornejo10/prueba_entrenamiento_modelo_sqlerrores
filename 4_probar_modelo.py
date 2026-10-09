import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
# Carpeta donde guardaste el modelo tras ejecutar 'fusionar_modelo.py'
RUTA_MODELO_FUSIONADO = "./titin-qwen14b-fused"

# Archivo con los ejercicios de prueba (puedes cambiarlo a 'dataset_titin_sintetico.jsonl')
ARCHIVO_TEST = "prueba.jsonl"

# Número de casos a probar (None para evaluar todo el archivo)
MAX_EJEMPLOS = 5 

# ==============================================================================
# 1. CARGA DEL MODELO Y TOKENIZER
# ==============================================================================
print(f"Cargando Tokenizer desde: {RUTA_MODELO_FUSIONADO} ...")
tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO_FUSIONADO, trust_remote_code=True)

# Cuantización de 4 bits para que un modelo 14B entre sin problemas en 12 GB de VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Cargando modelo fusionado (4-bits en GPU)...")
modelo = AutoModelForCausalLM.from_pretrained(
    RUTA_MODELO_FUSIONADO,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
modelo.eval()
print("¡Modelo cargado y listo para inferencia!\n")

# ==============================================================================
# 2. FUNCIÓN DE EXTRACCIÓN ROBUSTA DE CAMPOS
# ==============================================================================
def extraer_datos(ejemplo):
    """
    Soporta tanto formato anidado:
      {"entrada": {"esquema_bd": ...}, "salida": {...}}
    como formato plano:
      {"entrada_esquema_ddl": ..., "salida_validacion": ...}
    """
    if "entrada" in ejemplo:
        ent = ejemplo["entrada"]
        sal = ejemplo.get("salida", {})
        ddl = ent.get("esquema_bd", ent.get("esquema_ddl", ""))
        pregunta = ent.get("consulta_lenguaje_natural", "")
        sql_evaluar = ent.get("consulta_sql_a_evaluar", "")
        sql_correcto = ent.get("consulta_sql_correcta", "")
        validacion = sal.get("validacion", "")
        tipo_error = sal.get("tipo_de_error", [])
        feedback = sal.get("feedback", "")
    else:
        ddl = ejemplo.get("entrada_esquema_ddl", "")
        pregunta = ejemplo.get("entrada_lenguaje_natural", "")
        sql_evaluar = ejemplo.get("entrada_sql_evaluar", "")
        sql_correcto = ejemplo.get("entrada_sql_correcto", "")
        validacion = ejemplo.get("salida_validacion", "")
        tipo_error = ejemplo.get("salida_tipo_error", [])
        feedback = ejemplo.get("salida_feedback", "")
        
    if isinstance(tipo_error, list):
        tipo_error_str = ", ".join(tipo_error)
    else:
        tipo_error_str = str(tipo_error)
        
    return {
        "ddl": ddl,
        "pregunta": pregunta,
        "sql_evaluar": sql_evaluar,
        "gt_validacion": validacion,
        "gt_tipo_error": tipo_error_str,
        "gt_feedback": feedback,
        "gt_sql_correcto": sql_correcto
    }

# ==============================================================================
# 3. LECTURA Y EVALUACIÓN
# ==============================================================================
if not os.path.exists(ARCHIVO_TEST):
    raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO_TEST}")

with open(ARCHIVO_TEST, "r", encoding="utf-8") as f:
    lineas = [json.loads(line.strip()) for line in f if line.strip()]

if MAX_EJEMPLOS:
    lineas = lineas[:MAX_EJEMPLOS]

print(f"============================================================")
print(f"   INICIANDO PRUEBAS DE TITIN ({len(lineas)} EJEMPLOS)")
print(f"============================================================\n")

for i, ej in enumerate(lineas, start=1):
    datos = extraer_datos(ej)

    # Prompt idéntico al formato de entrenamiento con los tokens especiales de Qwen
    prompt = f"""<|im_start|>system
Eres Titin, un experto en diagnóstico SQL. Evalúa la consulta proporcionada utilizando el siguiente esquema DDL.
DDL:
{datos['ddl']}<|im_end|>
<|im_start|>user
Pregunta: {datos['pregunta']}
SQL a evaluar: {datos['sql_evaluar']}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    with torch.no_grad():
        output_tokens = modelo.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,         # Temperatura baja para diagnósticos deterministas y precisos
            do_sample=False,
            eos_token_id=tokenizer.encode("<|im_end|>")[-1],
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodificar solo la respuesta generada por el asistente
    tokens_generados = output_tokens[0][inputs["input_ids"].shape[1]:]
    respuesta_generada = tokenizer.decode(tokens_generados, skip_special_tokens=True).strip()

    # Visualización clara en terminal
    print(f"------------------------------------------------------------")
    print(f" CASO #{i}")
    print(f"------------------------------------------------------------")
    print(f"📌 PREGUNTA:       {datos['pregunta']}")
    print(f"🔍 SQL A EVALUAR:  {datos['sql_evaluar']}")
    print(f"\n🎯 ESPERADO (Ground Truth):")
    print(f"   Validación:    {datos['gt_validacion']}")
    print(f"   Tipo Error:    {datos['gt_tipo_error']}")
    print(f"   Diagnóstico:   {datos['gt_feedback']}")
    print(f"\n🤖 PREDICCIÓN DE TITIN:")
    print(respuesta_generada)
    print(f"------------------------------------------------------------\n")