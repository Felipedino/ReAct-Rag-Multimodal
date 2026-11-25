"""Carga del modelo (vLLM) y generación en lote.

Uso:
    import llm
    modelo = llm.cargar("1.5B")
    p = llm.armar_prompt(modelo, "Eres...", "Pregunta: ...", prefijo_asistente="Respuesta:")
    llm.generar(modelo, [p], max_tokens=32, stop=["\n"])
"""

import os

# Debe fijarse ANTES de importar vllm: sin esto, el motor corre en un
# subproceso que falla al iniciar en Colab (DECISIONES 3.1).
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402

MODELOS = {"1.5B": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
           "3B":   "Qwen/Qwen2.5-3B-Instruct-AWQ",
           "7B":   "Qwen/Qwen2.5-7B-Instruct-AWQ",
           "14B":  "Qwen/Qwen2.5-14B-Instruct-AWQ"}  # extra: pesa ~9,5 GB, casi llena la T4

MAX_MODEL_LEN = 8192   # largo máximo de contexto (prompt + salida), en tokens. 4096 no alcanza para 7 pasos de ReAct (DECISIONES 3.2)

# Total de tokens generados en este proceso (para reportar el cómputo).
tokens_generados = 0


def cargar(nombre: str) -> LLM:
    """Carga un modelo AWQ con la configuración probada en T4 (DECISIONES 3.1).

    dtype="half" porque la T4 no soporta bfloat16. Sin enforce_eager, que es
    más lento (los CUDA graphs aceleran la generación).
    """
    return LLM(MODELOS[nombre], quantization="awq", dtype="half",
               max_model_len=MAX_MODEL_LEN,
               # El 14B deja poca memoria para la caché KV: se usa más de la GPU.
               gpu_memory_utilization=0.95 if nombre == "14B" else 0.85, seed=0)


def armar_prompt(llm: LLM, sistema: str, usuario: str,
                 prefijo_asistente: str = "") -> str:
    """Prompt en el formato de chat del modelo.

    Si hay `prefijo_asistente`, el turno del asistente queda abierto con ese
    texto para que el modelo lo continúe (ej. "Pensamiento 3:"). Así ReAct
    agrega pasos a la trayectoria sin romper el formato de chat.
    """
    mensajes = [{"role": "system", "content": sistema},
                {"role": "user", "content": usuario}]
    tokenizer = llm.get_tokenizer()
    if not prefijo_asistente:
        return tokenizer.apply_chat_template(mensajes, tokenize=False,
                                             add_generation_prompt=True)
    mensajes.append({"role": "assistant", "content": prefijo_asistente})
    return tokenizer.apply_chat_template(mensajes, tokenize=False,
                                         add_generation_prompt=False,
                                         continue_final_message=True)


def contar_tokens(llm: LLM, texto: str) -> int:
    """Cantidad de tokens de un prompt ya armado (para no pasar MAX_MODEL_LEN)."""
    return len(llm.get_tokenizer().encode(texto, add_special_tokens=False))


def generar(llm: LLM, prompts: list[str], max_tokens: int,
            stop: list[str] | None = None) -> list[str]:
    """Genera para todos los prompts en una sola llamada (lote), sin muestreo.

    temperature=0 hace la salida determinista (greedy). Devuelve un texto por
    prompt, en el mismo orden, y suma los tokens generados al contador global.
    """
    global tokens_generados
    if not prompts:
        return []
    params = SamplingParams(temperature=0, max_tokens=max_tokens, stop=stop, seed=0)
    salidas = llm.generate(prompts, params, use_tqdm=False)
    tokens_generados += sum(len(s.outputs[0].token_ids) for s in salidas)
    return [s.outputs[0].text for s in salidas]
