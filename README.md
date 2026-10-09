# Guía Paso a Paso: Implementación y Pruebas del Proyecto Titin

**Modelo de Lenguaje Especializado para la Detección y Diagnóstico Automático de Errores en Consultas SQL**  

---

## 1. Preparativos Iniciales (Abrir la Terminal)

Antes de ejecutar cualquier instrucción, debes abrir la terminal directamente en la carpeta donde tienes tus archivos (`Pruebas de entrenamiento`):

* **En Windows:** Abre la carpeta en el Explorador de Archivos, haz clic en la barra de dirección superior (donde dice la ruta), escribe `powershell` o `cmd` y presiona **Enter**.
* **En VS Code:** Presiona las teclas `Ctrl + Ñ` (o `Ctrl + ~`) para abrir la terminal integrada en la raíz del proyecto.

---

## 2. Instalación de Dependencias (Paso a Paso)

Ejecuta los siguientes comandos uno por uno en tu terminal.

### 2.1 Crear el Entorno Virtual
Aísla las librerías del proyecto para evitar conflictos de versiones:

```bash
python -m venv venv_titin
```

### 2.2 Activar el Entorno Virtual

* **Si estás en Windows (PowerShell):**
  ```powershell
  .\venv_titin\Scripts\Activate.ps1
  ```
  *(Si aparece un error de permisos en PowerShell, ejecuta primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y luego vuelve a activarlo).*

* **Si estás en Windows (CMD / Símbolo del Sistema):**
  ```cmd
  venv_titin\Scripts\activate.bat
  ```

* **Si estás en Linux o WSL:**
  ```bash
  source venv_titin/bin/activate
  ```

> **Verificación:** Sabrás que está activado porque verás `(venv_titin)` al principio de la línea de comandos.

---

### 2.3 Instalar PyTorch con aceleración GPU (CUDA)
Copia y pega la siguiente línea para instalar PyTorch compatible con tarjetas NVIDIA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verificar que detecta tu tarjeta gráfica:**
```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU detectada:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguna')"
```
*Debe responder `CUDA disponible: True` junto con el nombre de tu tarjeta NVIDIA.*

---

### 2.4 Instalar librerías de Inteligencia Artificial (Hugging Face y QLoRA)
Copia y pega este comando para instalar las herramientas de cuantización, entrenamiento y evaluación:

```bash
pip install transformers peft bitsandbytes trl datasets accelerate scipy sentencepiece protobuf
```

---

## 3. PASO 0: Prueba del Modelo Base (Línea Base sin Entrenar)

Antes de hacer el ajuste fino, es fundamental ejecutar el archivo `modelo_base.py` para verificar:
1. Que el modelo descarga y se carga correctamente en los 12 GB de VRAM en 4 bits.
2. Cómo responde el modelo virgen (sin especialización) frente a un error SQL, lo que servirá como punto de comparación (*baseline*).

### 3.1 Contenido de `modelo_base.py`
Asegúrate de que tu archivo `modelo_base.py` contenga el siguiente código optimizado:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELO_ID = "Qwen/Qwen3-14B"

print("==================================================")
print("  PASO 0: PROBANDO MODELO BASE (SIN ENTRENAR)")
print("==================================================")

print(f"\n[1/3] Descargando/Cargando Tokenizer ({MODELO_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_ID, trust_remote_code=True)

# Cuantización de 4 bits para no sobrepasar los 12 GB de VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("[2/3] Cargando modelo base en 4-bits en GPU...")
modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
modelo.eval()

# Caso de prueba: consulta con error sintáctico evidente
esquema_prueba = "CREATE TABLE empleado (id INT PRIMARY KEY, nombre VARCHAR(50), salario INT);"
pregunta_prueba = "Mostrar el nombre y salario de todos los empleados"
sql_erroneo = "SELET nombre, salario FROM empleado"

prompt = f"""<|im_start|>system
Eres Titin, un experto en diagnóstico SQL. Evalúa la consulta proporcionada utilizando el siguiente esquema DDL.
DDL:
{esquema_prueba}<|im_end|>
<|im_start|>user
Pregunta: {pregunta_prueba}
SQL a evaluar: {sql_erroneo}<|im_end|>
<|im_start|>assistant
"""

print("[3/3] Generando diagnóstico con el modelo base...")
inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

with torch.no_grad():
    output_tokens = modelo.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False,
        eos_token_id=tokenizer.encode("<|im_end|>")[-1],
        pad_token_id=tokenizer.eos_token_id
    )

tokens_generados = output_tokens[0][inputs["input_ids"].shape[1]:]
respuesta = tokenizer.decode(tokens_generados, skip_special_tokens=True).strip()

print("\n---------------- RESULTADO DEL MODELO BASE ----------------")
print(respuesta)
print("-----------------------------------------------------------\n")
```

### 3.2 Comando para ejecutar el Paso 0:
```bash
python modelo_base.py
```
*(La primera vez tardará unos minutos mientras descarga los pesos del modelo).*

---

## 4. PASO 1: Verificación de Datos (`dataset_titin_sintetico.jsonl`)

Asegúrate de que en la misma carpeta tengas el archivo generado `dataset_titin_sintetico.jsonl`. 

Puedes comprobar que el archivo tiene contenido y cuántos ejemplos contiene con este comando:

* **En Windows (PowerShell):**
  ```powershell
  Get-Content dataset_titin_sintetico.jsonl | Measure-Object -Line
  ```
* **En Linux / WSL / Git Bash:**
  ```bash
  wc -l dataset_titin_sintetico.jsonl
  ```

---

## 5. PASO 2: Entrenamiento con QLoRA (`entrenamiento.py`)

Este paso ejecuta el ajuste fino de parámetros eficientes. Se cargará el modelo en 4 bits y se entrenará un adaptador LoRA sobre las capas de atención sin desbordar los 12 GB de memoria.

### 5.1 Comando para iniciar el entrenamiento:
```bash
python entrenamiento.py
```

### 5.2 ¿Qué observarás durante la ejecución?
1. Mensajes de carga del modelo en 4-bits.
2. Estadísticas de parámetros: confirmará que menos del $1\%$ de los parámetros totales se están entrenando.
3. Una barra de progreso con pasos (`Steps`), pérdida (`Loss`) y tiempo estimado.
4. Al finalizar, verás el mensaje:
   ```text
   Adaptador LoRA guardado exitosamente en: ./titin-adaptador-qwen14b
   ```

---

## 6. PASO 3: Fusión del Adaptador (`fusionar_modelo.py`)

Para poder utilizar el modelo como un único bloque sin tener que cargar adaptadores dinámicos, unimos los pesos de LoRA al modelo base.

### 6.1 Contenido de `fusionar_modelo.py`:
Verifica que tu archivo tenga el siguiente código (nota que usa `device_map="cpu"` para no colapsar la VRAM):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELO_BASE_ID = "Qwen/Qwen3-14B"
RUTA_ADAPTADOR = "./titin-adaptador-qwen14b"
RUTA_SALIDA = "./titin-qwen14b-fused"

print(">> [1/4] Cargando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE_ID, trust_remote_code=True)

print(">> [2/4] Cargando modelo base en CPU (bfloat16)...")
modelo_base = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

print(">> [3/4] Acoplando adaptador LoRA...")
modelo_con_lora = PeftModel.from_pretrained(modelo_base, RUTA_ADAPTADOR)

print(">> [4/4] Fusionando pesos (merge_and_unload)...")
modelo_fusionado = modelo_con_lora.merge_and_unload()

print(f">> Guardando modelo fusionado completo en: {RUTA_SALIDA}")
modelo_fusionado.save_pretrained(RUTA_SALIDA, safe_serialization=True)
tokenizer.save_pretrained(RUTA_SALIDA)
print(">> Proceso finalizado exitosamente.")
```

### 6.2 Comando para ejecutar la fusión:
```bash
python fusionar_modelo.py
```
*(Este proceso utiliza la memoria RAM del sistema y tardará entre 3 y 8 minutos en guardar la carpeta `./titin-qwen14b-fused`).*

---

## 7. PASO 4: Pruebas del Modelo Entrenado (`probar_modelo.py`)

Este paso toma un archivo de casos de prueba (como `casos_prueba.jsonl`), le envía únicamente la pregunta, la consulta a evaluar y el DDL al modelo, y muestra en pantalla la predicción generada junto con la respuesta esperada.

### 7.1 Contenido de `probar_modelo.py`:

```python
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RUTA_MODELO_FUSIONADO = "./titin-qwen14b-fused"
ARCHIVO_TEST = "casos_prueba.jsonl"
MAX_EJEMPLOS = 3

print(f">> Cargando Tokenizer desde: {RUTA_MODELO_FUSIONADO}")
tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO_FUSIONADO, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(">> Cargando modelo entrenado y fusionado en GPU (4-bit)...")
modelo = AutoModelForCausalLM.from_pretrained(
    RUTA_MODELO_FUSIONADO,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
modelo.eval()

def extraer_datos(ejemplo):
    if "entrada" in ejemplo:
        ent = ejemplo["entrada"]
        sal = ejemplo.get("salida", {})
        ddl = ent.get("esquema_bd", ent.get("esquema_ddl", ""))
        pregunta = ent.get("consulta_lenguaje_natural", "")
        sql_evaluar = ent.get("consulta_sql_a_evaluar", "")
        validacion = sal.get("validacion", "")
        tipo_error = sal.get("tipo_de_error", [])
        feedback = sal.get("feedback", "")
    else:
        ddl = ejemplo.get("entrada_esquema_ddl", "")
        pregunta = ejemplo.get("entrada_lenguaje_natural", "")
        sql_evaluar = ejemplo.get("entrada_sql_evaluar", "")
        validacion = ejemplo.get("salida_validacion", "")
        tipo_error = ejemplo.get("salida_tipo_error", [])
        feedback = ejemplo.get("salida_feedback", "")

    tipo_error_str = ", ".join(tipo_error) if isinstance(tipo_error, list) else str(tipo_error)

    return {
        "ddl": ddl,
        "pregunta": pregunta,
        "sql_evaluar": sql_evaluar,
        "gt_validacion": validacion,
        "gt_tipo_error": tipo_error_str,
        "gt_feedback": feedback
    }

if not os.path.exists(ARCHIVO_TEST):
    raise FileNotFoundError(f"No existe el archivo {ARCHIVO_TEST} en la carpeta actual.")

with open(ARCHIVO_TEST, "r", encoding="utf-8") as f:
    lineas = [json.loads(line.strip()) for line in f if line.strip()]

if MAX_EJEMPLOS:
    lineas = lineas[:MAX_EJEMPLOS]

print(f"\n============================================================")
print(f"      EVALUACIÓN DE TITIN ({len(lineas)} CASOS DE PRUEBA)")
print(f"============================================================\n")

for i, ej in enumerate(lineas, start=1):
    datos = extraer_datos(ej)

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
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.encode("<|im_end|>")[-1],
            pad_token_id=tokenizer.eos_token_id
        )

    tokens_generados = output_tokens[0][inputs["input_ids"].shape[1]:]
    respuesta = tokenizer.decode(tokens_generados, skip_special_tokens=True).strip()

    print(f"================ CASO DE PRUEBA #{i} ================")
    print(f"📌 REQUERIMIENTO: {datos['pregunta']}")
    print(f"🔍 SQL A EVALUAR: {datos['sql_evaluar']}")
    print(f"\n🎯 ESPERADO (Ground Truth del Dataset):")
    print(f"   Validación:    {datos['gt_validacion']}")
    print(f"   Tipo de Error: {datos['gt_tipo_error']}")
    print(f"   Feedback:      {datos['gt_feedback']}")
    print(f"\n🤖 PREDICCIÓN DE TITIN (Generada por el modelo):")
    print(respuesta)
    print("====================================================\n")
```

### 7.2 Comando para ejecutar la prueba:
```bash
python probar_modelo.py
```

---

## 8. Resumen de Comandos en Orden de Ejecución

Para una referencia rápida, estos son los comandos que debes ejecutar en orden:

```bash
# 1. Activar el entorno virtual (PowerShell)
.\venv_titin\Scripts\Activate.ps1

# 2. Paso 0: Probar el modelo base virgen
python modelo_base.py

# 3. Paso 2: Entrenar el modelo con QLoRA
python entrenamiento.py

# 4. Paso 3: Fusionar el adaptador con el modelo base
python fusionar_modelo.py

# 5. Paso 4: Probar el modelo entrenado
python probar_modelo.py
```