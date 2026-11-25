# Especificación: llm.py, agents.py, correr.py, evaluate.py

Alcance recortado por tiempo (ver DECISIONES.md 3.2): 3 condiciones
(directa, rag, react) × 3 modelos (1.5B, 3B, 7B) × dataset (dev + test).
Código simple: funciones, comentarios en español, sin dependencias nuevas
(vllm, transformers, matplotlib y pandas ya están en Colab).

---------------------------------------------------------------------------
## llm.py — carga del modelo y generación

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # ANTES de importar vllm (DECISIONES 3.1)

MODELOS = {"1.5B": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
           "3B":   "Qwen/Qwen2.5-3B-Instruct-AWQ",
           "7B":   "Qwen/Qwen2.5-7B-Instruct-AWQ"}
```

- `cargar(nombre) -> LLM`: `LLM(MODELOS[nombre], quantization="awq", dtype="half",
  max_model_len=4096, gpu_memory_utilization=0.85, seed=0)`. Sin enforce_eager.
- `armar_prompt(llm, sistema, usuario, prefijo_asistente="") -> str`: usa
  `llm.get_tokenizer().apply_chat_template(...)`. Si hay `prefijo_asistente`, el
  turno del asistente queda abierto con ese texto (`continue_final_message=True`,
  `add_generation_prompt=False`) para que el modelo lo continúe. Así ReAct puede
  ir agregando pasos a la trayectoria sin romper el formato de chat.
- `generar(llm, prompts, max_tokens, stop=None) -> list[str]`: una sola llamada
  en lote a `llm.generate`, `SamplingParams(temperature=0, max_tokens=..., stop=..., seed=0)`.
  Devuelve los textos y suma los tokens generados en un contador global
  `tokens_generados` (para reportar cómputo).

---------------------------------------------------------------------------
## agents.py — las 3 condiciones

Todas reciben una lista de preguntas (dicts del dataset) y devuelven una traza
por pregunta. Todas procesan en lote (todas las preguntas a la vez).

### Formato de respuesta común
El modelo termina con `Respuesta: <texto>` (directa, rag) o `Terminar[<texto>]`
(react). `extraer_respuesta(texto)`: lo que sigue a "Respuesta:" hasta el fin de
línea; si no aparece, la primera línea no vacía (y la traza marca `formato_ok=False`).

### Ejemplos few-shot (los mismos 2 temas en las 3 condiciones)
- 2 preguntas puente escritas a mano, sobre entidades que NO están en
  `data/pairs.jsonl` (comprobarlo), con observaciones reales sacadas de `wiki.py`.
- Constantes en agents.py: `EJEMPLOS_DIRECTA`, `EJEMPLOS_RAG`, `EJEMPLOS_REACT`.
  Usar los mismos 2 temas en las tres para que la comparación sea justa.
- En español. Las trayectorias ReAct de ejemplo usan 2 búsquedas y terminan.

### 1. `directa(llm, preguntas)`
Instrucción: responder con una respuesta corta (nombre, lugar, número o año),
sin explicación, en el formato `Respuesta: ...`. `max_tokens=32`, stop `["\n"]`
después de la respuesta (o recortar a la primera línea).

### 2. `rag(llm, preguntas)`
Una sola búsqueda con el texto de la pregunta: `wiki.search(pregunta, k=3)`; para
cada título, `observacion(titulo)` (abajo). Se ponen los 3 textos en el prompt
como contexto y se pide `Respuesta: ...`. `max_tokens=32`.
Es el baseline de "buscar sin razonar": una búsqueda fija, sin decidir qué buscar.

### 3. `react(llm, preguntas, max_pasos=7)`
Formato (como el paper, en español):
```
Pensamiento 1: ...
Acción 1: Buscar[<entidad>]
Observación 1: ...
...
Acción k: Terminar[<respuesta corta>]
```
Acciones (solo dos; sin lookup, por simplicidad):
- `Buscar[x]`: si `wiki.get_page(x)` existe → `observacion(titulo)`. Si no existe →
  `"No se encontró [x]. Similares: " + títulos de wiki.search(x, k=5)` (como el paper).
- `Terminar[x]`: fin; x es la respuesta.

Bucle en lote:
1. Cada pregunta activa tiene su trayectoria (texto). El prompt es
   `armar_prompt(sistema, instrucciones + ejemplos + pregunta, prefijo_asistente=trayectoria + "Pensamiento i:")`.
2. Generar para todas las activas: `max_tokens=150`, `stop=["\nObservación"]`.
3. Parsear con regex `Acción \d+:\s*(Buscar|Terminar)\[(.*?)\]`.
   - Terminar → guardar respuesta, sacar de activas (`termino="ok"`).
   - Buscar → ejecutar, agregar `Observación i: ...` a la trayectoria.
   - Sin acción válida → observación `"Acción inválida. Usa Buscar[...] o Terminar[...]."`
     (como el paper) y contar `acciones_invalidas`.
4. Al llegar a `max_pasos` sin Terminar: respuesta vacía, `termino="max_pasos"`.

Si un prompt se acerca a 4096 tokens, cortar la trayectoria: `termino="contexto"`.

### `observacion(titulo) -> str`
Intro del artículo (`introduccion()` de pairs.py) recortada a `MAX_CHARS_OBS = 1500`
caracteres. Misma función para rag y react (y questions.py filtra que la respuesta
y el puente estén dentro de esos 1500 caracteres; ver specs/questions.md).

### Traza (una por pregunta × condición × modelo)
```
{"id", "celda", "modelo", "condicion", "pregunta", "respuesta_gold", "alias",
 "respuesta_modelo", "texto_modelo" (salida cruda; en react, la trayectoria completa),
 "pasos": [{"pensamiento", "accion", "argumento", "titulo_observado", "observacion"}],
 "titulos_vistos" (lista de títulos de páginas observadas: rag y react),
 "termino": "ok" | "max_pasos" | "contexto" | "formato",
 "acciones_invalidas", "formato_ok", "segundos"}
```

---------------------------------------------------------------------------
## correr.py — un modelo por proceso

`python correr.py <modelo> <split> [condiciones...]`, ej. `python correr.py 7B test`.
- Carga el modelo, corre las condiciones pedidas (por defecto las 3) sobre las
  preguntas del split en `data/dataset.jsonl`, y escribe
  `resultados/trazas_<modelo>_<split>.jsonl`.
- Un proceso por modelo: al terminar, el proceso libera la GPU (vLLM no la libera
  bien dentro de un mismo notebook). El notebook lo llama con `!python correr.py ...`.
- Imprime al final: tiempo por condición, tokens generados, llamadas reales a la API
  de Wikipedia (`wiki.llamadas_api`).
- Fija `random.seed(0)` no hace falta (temperature=0); no usar `random` global.

---------------------------------------------------------------------------
## evaluate.py — métricas, intervalos y errores

- `normalizar` (importar de questions.py) + quitar artículos (el, la, los, las,
  un, una, unos, unas, lo, del→de) para EM/F1.
- Por traza: `em` (máximo sobre alias), `f1` (F1 de tokens, máximo sobre alias),
  `contiene` (algún alias normalizado está contenido en la respuesta normalizada;
  métrica secundaria, porque los modelos chicos a veces agregan palabras).
- `tabla(trazas)`: por modelo × condición × celda: n, EM, F1, contiene, con IC 95 %
  por bootstrap (1000 remuestreos, `random.Random(0)`). También por origen y por
  popularidad (juntando celdas).
- `ganancia(trazas, a="react", b="directa")`: diferencia de EM por celda y modelo,
  con IC por bootstrap pareado (se remuestrean preguntas y se usa la misma muestra
  para ambas condiciones). Es la cifra central de la pregunta de investigación.
- `clasificar_error(traza, par)` para react y rag, automático y simple:
  - `formato`: termino == "formato" o acciones_invalidas ≥ 3
  - `max_pasos` / `contexto`: según termino
  - `leyo_mal`: la respuesta correcta (o un alias) aparece, normalizada, en alguna
    observación, pero respondió mal → falla de lectura o razonamiento.
  - Si la respuesta nunca apareció en lo que leyó → falla de búsqueda:
    - `no_encontro_A`: ninguna observación menciona el nombre de B (no llegó al dato
      de A que identifica el puente)
    - `no_encontro_B`: alguna observación menciona a B, pero nunca leyó la página de B
      (ej. vio "Jørn Utzon" en la página de la Ópera y respondió de memoria)
  - Por qué con observaciones y no con títulos: el agente puede llegar a B por una
    desambiguación o una redirección ("Ramón Carnicer" es una desambiguación que
    igual trae la respuesta), y comparar títulos lo contaría como no encontrado.
  Los tipos "alucinación" y "error del dataset" se asignan a mano en el análisis.
- `grafico(tabla)`: matplotlib, EM por condición (barras agrupadas por celda), un
  panel por modelo. Guardar en `resultados/`.
- Escribe `resultados/tabla.csv` y `resultados/errores.csv`.

---------------------------------------------------------------------------
## Orden de prueba
1. Con 5 preguntas escritas a mano (o las primeras de dev), correr las 3 condiciones
   con el 1.5B y revisar las trazas a ojo: formato, parseo, observaciones.
2. Correr dev completo con 1.5B y 7B. Ajustar prompts SOLO con dev.
3. Congelar prompts y correr test con los 3 modelos.
