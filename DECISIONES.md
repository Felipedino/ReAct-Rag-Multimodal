# Decisiones de diseño — Proyecto CC5219 (ReAct sobre conocimiento chileno)

Registro de las decisiones del proyecto y sus razones, para escribir el informe
y poder explicarlas en la presentación. Cada decisión dice qué se eligió, por qué
y qué alternativas se descartaron.

---

## 1. Pregunta de investigación

**Pregunta:** ¿La ventaja de ReAct sobre responder de memoria crece cuando el
conocimiento es poco conocido y/o local (Chile), y cuando el modelo es pequeño?

**Conexión con el paper:** en la Figura 3 de ReAct (Yao et al., 2022), ReAct por
prompting rinde peor que las otras condiciones con modelos chicos. Este proyecto
lo pone a prueba en otro idioma, con conocimiento local y con modelos abiertos.

**Reformulación tras el diseño 2×2:** la pregunta separa dos causas posibles:
¿ReAct ayuda más por ser conocimiento *local* o por ser conocimiento *poco conocido*?

---

## 2. Dataset

### 2.1 Diseño 2×2: origen × popularidad
- **Decisión:** cuatro celdas: (chile / no_chile) × (pocos / muchos idiomas).
- **Por qué:** "pocos idiomas" mide qué tan conocido es algo, no qué tan chileno
  es. Comparar solo Chile poco conocido contra global muy conocido mezclaría dos
  variables. Con el 2×2 se separan:
  - Si todo lo poco popular cae igual, chileno o no → efecto de popularidad.
  - Si lo chileno cae más que lo no chileno con igual popularidad → efecto local.
- **Costo:** casi nulo, porque el total de preguntas es el mismo; solo se agrega
  un filtro automático.
- **Plan B:** si falta tiempo, se juntan celdas (Chile vs. no Chile) y los idiomas
  quedan como variable de análisis. El 2×2 contiene a la versión simple.
- **Limitación:** con ~60 preguntas por celda, la interacción tiene poca potencia
  estadística. Los efectos principales juntan ~120 preguntas cada uno.

### 2.2 Popularidad = número de ediciones de Wikipedia en otros idiomas
- **Decisión:** usar la cantidad de *langlinks* del artículo como proxy de
  popularidad. Umbrales con zona intermedia descartada (ej. pocos ≤ 10,
  muchos ≥ 40; se ajustan con el diagnóstico de `pairs.py`).
- **Por qué:** es objetivo, se obtiene con la misma API y se justifica en una línea.
- **Respaldo en la literatura** (verificado el 24-11-2026):
  - Mallen, Asai, Zhong, Das, Khashabi y Hajishirzi (2023). *When Not to Trust
    Language Models: Investigating Effectiveness of Parametric and Non-Parametric
    Memories*. ACL 2023, pp. 9802–9822. doi:10.18653/v1/2023.acl-long.546.
    Crean PopQA (14k preguntas). Los LMs fallan más en conocimiento poco popular;
    escalar casi no mejora la memorización en la cola larga; los LMs con
    recuperación superan a LMs órdenes de magnitud más grandes, mientras que sin
    recuperación siguen siendo competitivos en entidades populares. Proponen
    recuperar solo cuando hace falta (adaptive retrieval). Miden popularidad con
    visitas mensuales a Wikipedia.
  - Kandpal, Deng, Roberts, Wallace y Raffel (2023). *Large Language Models
    Struggle to Learn Long-Tail Knowledge*. ICML 2023, PMLR 202:15696–15707.
    La exactitud en QA depende de cuántos documentos relevantes a la pregunta
    había en el preentrenamiento (relación correlacional y causal). Modelos más
    grandes aprenden algo más de la cola larga, pero harían falta órdenes de
    magnitud más de escala. La recuperación reduce esa dependencia.
  - **Cómo se usan:** justifican que la popularidad prediga el desempeño de memoria
    y que la recuperación (aquí, ReAct y RAG) ayude más en lo poco popular. Ninguno
    usa langlinks: es un proxy distinto, que se justifica igual (más idiomas ≈ más
    texto sobre la entidad en la web).
  - **Hipótesis que sugiere Mallen et al.:** en las celdas "muchos", buscar podría
    no ayudar o incluso empeorar frente a responder de memoria, sobre todo con el 7B.
- **Alternativa descartada:** visitas (pageviews). Requieren otra API, y las
  visitas de artículos chilenos en es.wikipedia vienen sobre todo de Chile.

### 2.2.1 La popularidad se mide solo en A (cambio tras la prueba de pairs.py)
- **Decisión:** la celda la define la popularidad de A. La de B se guarda
  (`idiomas_B`) como covariable para el análisis, pero no filtra.
- **Antes:** se exigía que A y B cayeran en el mismo grupo (pocos o muchos).
- **Por qué se cambió:**
  - En las celdas "muchos", los B con ≥ 40 idiomas que aparecen en una intro
    suelen ser conceptos genéricos ("Historia de Chile → Historia",
    "Historia de Polonia → Estado"). La pregunta deja de ser sobre Chile y la
    etiqueta de origen pierde sentido.
  - La doble condición vaciaba chile_muchos (pocos A chilenos populares, y
    además cada uno necesitaba un B popular no genérico).
  - A es el tema de la pregunta, y el dato que el modelo debe saber de memoria
    (el salto A → B) está en el artículo de A. Su popularidad es el mejor
    proxy de cuánto texto sobre el tema vio el modelo (argumento de Kandpal et al.).
- **Alternativa descartada:** mínimo de idiomas entre A y B: mismo problema de escasez.
- **Limitación:** en "muchos" puede haber B poco conocidos; se analiza con `idiomas_B`.
- **Origen:** también lo define A (su categoría). B puede no ser chileno
  (ej. Himno Nacional de Chile → Ramón Carnicer, español) y la pregunta igual
  es sobre conocimiento chileno.

### 2.2.2 Emparejamiento de popularidad por tramos (tras el diagnóstico de pairs.py)
- **Problema:** dentro de "pocos", lo chileno era mucho menos popular que el control.
  Mediana de idiomas de A: chile_pocos 0 vs. no_chile_pocos 4; con 0 idiomas,
  60 % vs. 17 %. Si el modelo falla más en chile_pocos, no se sabría si es por ser
  chileno o por ser aún menos conocido: justo la confusión que el 2×2 debía evitar.
- **Decisión:** los idiomas de A se dividen en tramos (pocos: 0 / 1–5 / 6–10;
  muchos: 21–39 / 40–79 / 80+; 11–20 se descarta). Al muestrear, chile toma sus pares
  con su distribución natural por tramo, y no_chile toma de cada tramo la misma
  cantidad que chile. Así ambos orígenes tienen la misma distribución de popularidad
  en cada celda, y la diferencia que quede se atribuye al origen.
- Es emparejamiento por frecuencia (*frequency matching*), estándar en estudios
  observacionales. Se empareja hacia chile porque es el grupo escaso.
- El tope de pares pasa a ser por (celda, tramo) en vez de por celda, para juntar
  suficientes A de control en los tramos donde abunda chile (ej. 0 idiomas).

### 2.2.3 Umbral de "muchos": regla fijada antes de ver los datos
- Se recolecta desde ≥ 21 idiomas. Si chile junta ≥ 90 pares solo con los tramos
  40–79 y 80+, "muchos" = ≥ 40 (contraste más claro con "pocos"); si no, "muchos" = ≥ 21.
- **Por qué una regla previa:** la elección no depende de los resultados de los
  modelos (que aún no existen), y queda escrita antes de ver los pares. Una sola
  corrida sirve para ambas opciones: la elección se aplica al muestrear.
- Histograma chileno (profundidad 1): 30 semillas con ≥ 40 idiomas, 58 con 21–39,
  73 con 11–20.

### 2.3 Preguntas puente a partir de pares A → B
- **Decisión:** cada pregunta se genera a partir de un par de artículos donde A
  enlaza a B en su introducción. La pregunta describe a B con información de A
  sin nombrarlo, y la respuesta sale de B. Es el mismo método que HotpotQA.
- **Ejemplo:** A = "Himno Nacional de Chile", B = "Ramón Carnicer".
  "¿De qué país era el compositor de la música del himno nacional de Chile?" → España.
- **Por qué:**
  - Garantiza que la pregunta es multi-hop por construcción.
  - Garantiza que la evidencia existe en Wikipedia, así que la herramienta puede encontrarla.
  - Se conocen los artículos correctos (A y B), lo que permite clasificar los
    errores en las trazas: no encontró el artículo (búsqueda) vs. lo encontró y
    respondió mal (razonamiento o lectura).
- **Semillas:** artículos A tomados de categorías temáticas (historia, geografía,
  política, cultura) de Chile y de 8 países de control (Francia, Japón, Polonia,
  Perú, Egipto, Australia, Canadá, Italia: distintos continentes y tamaños; Kenia
  se descartó por tener muy pocos artículos), con subcategorías hasta
  **profundidad 2**, igual para ambos grupos.
  - **Por qué 2 y no 1:** con profundidad 1, los A chilenos populares y realmente
    chilenos eran ~15, casi todos volcanes y cordillera: chile_muchos habría medido
    "geografía chilena". Con profundidad 2 entran presidentes, escritores y ciudades
    (93 semillas chilenas con ≥ 40 idiomas antes de limpiar). Se aplica a ambos
    grupos para que la forma de elegir semillas sea la misma. Mismas categorías temáticas en ambos grupos para que sean comparables.
  - Un artículo que aparece en ambos grupos se descarta de los dos (origen ambiguo).
  - **A debe mencionar su país en la introducción** (nombre o gentilicio:
    "Chile"/"chileno", "Francia"/"francés", etc.). Motivo: las subcategorías traen
    artículos que no son del país. En la primera corrida, 11 de las 30 semillas
    chilenas con ≥ 40 idiomas eran extranjeros condecorados por Chile (García Márquez,
    Paul McCartney, Dario Fo, Bono...; subcategorías "Condecorados con la Orden al
    Mérito…"), y varias eran conceptos genéricos (Taxidermia, Embalsamamiento,
    Agujero de la capa de ozono). El filtro es automático y simétrico: se aplica
    igual a Chile y a los países de control.
    - **Alternativa descartada:** excluir las subcategorías "Condecorados…". Solo
      arregla ese caso; no saca los conceptos genéricos, y con profundidad 2
      aparecerían otras subcategorías parecidas.
    - **Costo:** puede perder temas chilenos cuya intro no nombra a Chile (ej. Moái
      podría decir solo "Isla de Pascua"). El diagnóstico lista los descartados para
      revisar esos falsos negativos.
  - Aun con A del país, el fragmento usado puede hablar de otro país ("Cigarro de
    chala → Río Grande del Sur", fragmento sobre Uruguay y Brasil). Eso lo revisa la
    verificación con LLM (regla 7 de `specs/questions.md`), que recibe el país de A.
  - Se descartan los artículos panorámicos, como A y como B: títulos de la forma
    "<una palabra> de/del <país del estudio>" ("Historia de Chile", "Economía de
    Polonia", "Clima de Japón"). Generan preguntas vagas y produjeron los pares
    malos de la prueba. Se generalizó de los 4 temas a cualquier palabra porque
    con D2 relajado (ver abajo) "economía de Chile" pasaría. Costo: se pierden
    algunos B específicos como "Bandera de Chile".
- **Filtros de B:** sin años ni fechas, anexos, desambiguaciones, artículos cortos
  ni hubs (ej. "Chile" → "Santiago" es un mal par: ambos son hubs y la pregunta
  sería trivial o ambigua). B debe aparecer en la introducción de A.
  - B no puede ser semilla del otro grupo de origen.
  - **B debe contener un nombre propio:** su primera mención en la intro de A
    tiene al menos una palabra con mayúscula. Descarta conceptos genéricos
    ("historia", "general", "yidis") con un cambio de una línea.
    - Primero se propuso "la mención empieza con mayúscula", pero en español
      los eventos se escriben con minúscula inicial ("batalla de Rancagua",
      "guerra de la Independencia de Chile"), y son B típicos de las semillas
      de historia. Se relajó a "alguna palabra con mayúscula".
    - No atrapa todo ("el Estado"); lo que se escape lo detecta la verificación con LLM.
  - El filtro de hub (> 800 links) no atrapa los conceptos genéricos, por eso
    se agregó el de nombre propio.
  - Desambiguación por heurística ("(desambiguación)" en el título o "puede
    referirse a" en la intro), porque `get_page` no trae categorías.
  - Después de traer B se revisa su título final, porque un link puede ser una
    redirección (así se detectó un B que redirigía a un "Anexo:").
- **Elección de B:** candidatos = links cuyo nombre aparece en la intro de A,
  ordenados por primera mención (los primeros suelen ser los más relacionados);
  se toma el primero que pasa todos los filtros.
- **Diversidad:** máximo un par por A, y cada B se usa una sola vez.
- **Orden y tope:** las semillas se recorren en orden aleatorio (semilla fija),
  mezclando ambos orígenes para que ninguno tenga prioridad al disputarse un
  mismo B. Cada celda se llena hasta 300 pares y luego se muestrean 100 por
  celda. Como el orden es aleatorio, equivale a muestrear del total sin
  procesar todas las semillas de las celdas abundantes.

### 2.4 Tamaño y split
- **Decisión:** ~260 preguntas: 20 de dev (5 por celda) y 60 de test por celda (240).
- **Por qué:** con ~80 preguntas el intervalo de confianza de un accuracy es de
  ±10 puntos aprox.; con menos no se distinguen diferencias moderadas.
- **Regla:** el split se define antes de experimentar. Dev se usa para ajustar
  prompts; test no se usa para ajustar nada.
- **Recorte por tiempo (24-11):** test de 40 por celda (160) + 5 de dev por celda,
  y auditoría manual de 20 preguntas. Intervalos más anchos (≈ ±15 puntos por
  celda); los efectos principales juntan 80 preguntas por grupo.
- **Profundidad asimétrica por tiempo:** profundidad 2 solo para Chile y 1 para
  no_chile (con 2 para ambos la corrida tomaba 3–5 h; las celdas de control ya se
  llenan con profundidad 1). Limitación declarada: la forma de elegir semillas
  difiere entre grupos; el emparejamiento por tramos controla la popularidad.

### 2.5 Generación y verificación
- **Generación:** un LLM (Claude, en una sesión interactiva de Claude Code o de
  Cowork; sin API key) procesa `pairs.jsonl` por lotes de 20. `questions.py` no llama
  a ningún LLM: escribe archivos de lote con el prompt y los pares, y lee las salidas.
  Así se guardan los prompts y las salidas (reproducibilidad) y el flujo no depende
  de la herramienta. Especificación en `specs/questions.md`.
  - El generador recibe solo `fragmento_A` e `intro_B`, no los artículos completos:
    la respuesta queda verificable contra lo guardado, y el dato está al inicio del
    artículo, donde el agente lo puede encontrar.
  - El generador no ve la celda ni los idiomas, para no sesgar la dificultad por celda.
  - Una pregunta por par, para que sean variadas. Respuestas cortas (entidad, lugar,
    número o año, ≤ 5 palabras) porque EM/F1 lo necesitan.
  - Puede descartar un par si no da una buena pregunta, en vez de forzarla.
  - Cada pregunta guarda `evidencia_A` y `evidencia_B` (frases textuales) para el
    análisis de errores: permiten ver si el agente leyó la frase correcta.
  - Generación y verificación en sesiones distintas, cada una sin el contexto de la
    otra ni el del diseño. Se anota qué herramienta y modelo se usaron.
- **Verificación en tres capas** (en vez de revisar todo a mano, ~6 h):
  1. Filtros automáticos: la respuesta aparece en B; la respuesta no aparece en la
     pregunta; el puente (B) no aparece en la pregunta; respuesta de ≤ 5 palabras;
     la respuesta no aparece en la intro de A (si aparece, se responde con un solo
     salto); las evidencias son copias textuales.
  2. Verificación con LLM, con un prompt adversarial ("busca razones por las que la
     pregunta sea mala"): ¿se puede responder con A y B?, ¿la respuesta es única?,
     ¿requiere ambos artículos? Además propone alias.
  3. Auditoría manual de una muestra aleatoria de ~40 preguntas para estimar la
     tasa de error del dataset, que se reporta en el informe.
- **Dev:** las 20 preguntas de dev se revisan completas a mano.
- **Limitación:** el mismo modelo genera y verifica, así que sus errores pueden
  estar correlacionados. La auditoría manual mide justamente eso.
- **Lo que NO se hace:** revisar solo las preguntas que los modelos fallaron y
  borrar las malas, porque sesga los resultados. Esas se clasifican como
  "error del dataset" en el análisis.

### 2.6 Fuente y licencia
- Wikipedia en español, CC BY-SA 4.0. Se declara el uso de LLMs en la generación
  y verificación.

---

## 3. Modelos y condiciones

### 3.1 Modelos
- **Decisión:** Qwen2.5-Instruct 1.5B, 3B y 7B, **todos en AWQ (4 bits)**.
- **Por qué:** el 7B en fp16 casi no cabe en una T4 (16 GB). Cuantizar los tres
  de la misma forma evita que la cuantización contamine la comparación entre tamaños.
- **Limitación:** los resultados son con modelos cuantizados.
- **Inferencia con vLLM en T4 (probado en Colab, 24-11):** vllm 0.34.0, torch 2.14.0+cu130,
  Python 3.13. Configuración que funciona:
  `LLM(modelo, quantization="awq", dtype="half", max_model_len=4096, gpu_memory_utilization=0.85)`
  con `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
  - Sin esa variable, el motor (que corre en un subproceso) falla al iniciar en Colab.
    Con ella corre en el mismo proceso del notebook.
  - `dtype="half"`: la T4 no soporta bfloat16.
  - vLLM elige solo los kernels compatibles con la T4: Marlin para AWQ y Triton para
    la atención (FlashAttention 2 requiere capacidad de cómputo ≥ 8).
  - Sin `enforce_eager` (con torch.compile y CUDA graphs) es más rápido:
    7B, 32 prompts, 64 tokens: 2,6 s (674 tok/s de salida) vs. 4,1 s (427 tok/s) en modo eager.
  - 7B: 5,3 GiB de pesos, ~91k tokens de caché KV, ~1 min de arranque.
  - Al correr en el mismo proceso, vLLM fija la semilla global de `random`. Por eso el
    código propio usa instancias `random.Random(SEMILLA)` y no el `random` global.
- **Anécdota para la presentación:** en la prueba, el 7B respondió de memoria que el
  primer presidente de Chile fue Pedro Aguirre Cerda, en 1920. Es una alucinación sobre
  conocimiento local, justo el fenómeno que se estudia.

### 3.2 Condiciones
- **Recorte por tiempo (24-11, la presentación es el 25-11):** 3 condiciones:
  respuesta directa, RAG fijo (una búsqueda, 3 resultados) y ReAct (máximo 7 pasos).
  - Directa vs. ReAct es la pregunta central (memoria vs. agente).
  - RAG fijo separa "buscar" de "buscar razonando": si ReAct supera a RAG, la
    ventaja viene de decidir qué buscar, no solo de tener acceso a Wikipedia.
  - CoT y Act eran ablaciones (razonar sin buscar; buscar sin razonar). Quedan
    como trabajo futuro.
- ReAct usa solo `Buscar[x]` y `Terminar[x]` (sin `lookup` del paper), por
  simplicidad. La observación es la intro del artículo, recortada a 1500 caracteres;
  questions.py filtra que la respuesta y el puente estén dentro de ese recorte.
- Los mismos 2 temas de ejemplo (few-shot) en las 3 condiciones, sobre entidades
  que no están en el dataset.
- Cada modelo corre en un proceso aparte (`correr.py`), porque vLLM no libera bien
  la GPU dentro de un mismo notebook.
- Especificación: `specs/experimento.md`.
- **Contexto de 8192 tokens** (en vez de 4096): el prompt de ReAct con ejemplos usa
  ~1.300 tokens y cada paso agrega ~450 (observación de 1500 caracteres + pensamiento),
  así que 7 pasos llegan a ~4.500–5.000. Con 4096 se cortaría cerca del paso 6 y eso
  castigaría a ReAct por una razón técnica. La memoria de la T4 alcanza: vLLM reserva
  la caché KV por bloques según el uso real, no por el máximo.
- El generador de preguntas recibe `intro_B` recortada a 1500 caracteres, lo mismo
  que ve el agente, para que las respuestas sean siempre observables.

---

## 4. Evaluación

- **Métricas:** Exact Match y F1 a nivel de tokens.
- **Normalización:** minúsculas, sin tildes, sin artículos ni puntuación.
- **Alias:** cada pregunta tiene una lista de respuestas aceptadas
  (ej. ["Pedro Aguirre Cerda", "Aguirre Cerda"]); se toma el mejor puntaje.
  Los alias se completan durante la verificación.
- **Incertidumbre:** intervalos de confianza con bootstrap.
- **Errores:** cada traza fallida se clasifica: formato roto, búsqueda mala,
  lectura mala, alucinación, se acabaron los pasos, error del dataset.

---

## 5. Reproducibilidad y cómputo

- **Caché de Wikipedia:** todas las consultas a la API se guardan en `cache/wiki/`
  y se sube al repo. Modo `OFFLINE` para verificar que una corrida sale 100% de la caché.
- **Trazas:** cada corrida (pregunta × condición × modelo) se guarda en JSONL con el
  prompt, cada Thought/Action/Observation, la respuesta final, el tiempo y los tokens.
  Permite recalcular métricas sin re-ejecutar modelos y es la base del análisis de errores.
- **Límite de una hora:** la pauta pide menos de una hora en T4, pero acepta más si se
  declara el cómputo. Se apunta a cumplirlo con vLLM y ReAct en batch (todas las
  preguntas avanzan un paso a la vez). El notebook puede reproducir la tabla desde
  las trazas guardadas y volver a correr un subconjunto en vivo.
- **Semillas fijas** en muestreo y generación.

---

## 6. Flujo de trabajo

- Implementación por etapas con Claude Code; pruebas en Colab.
- `CLAUDE.md` en la raíz del repo con el contexto del proyecto y las reglas de código
  (simple, comentado en español, sin adelantar etapas, explicar al terminar).
- Cada módulo parte de una especificación escrita antes del código.
- Uso de asistentes de IA declarado en el informe (lo exige la pauta).

---

## 7. Estado por etapa

### wiki.py — listo
API de Wikipedia en español con caché en disco. Funciones: `search`, `get_page`,
`category_members`. Decisiones de implementación:
- Caché: sha1 de los parámetros ordenados → `cache/wiki/<sha1>.json`. Ruta relativa
  a `wiki.py`, así que funciona igual en local y en Colab.
- Escritura atómica (`.tmp` + `os.replace`) para no dejar JSON corruptos.
- `formatversion=2` para respuestas más simples de procesar.
- `get_page` hace dos consultas: texto (con redirects) y links (sobre el título final),
  para no mezclar paginaciones.
- Errores de la API en el cuerpo (HTTP 200 con "error") no se guardan ni se reintentan.
- Reintentos ante fallas de red: 3, con esperas de 1, 2 y 4 s.
- Dato para la etapa siguiente: "Categoría:Historia de Chile" tiene solo 71 artículos
  directos, por eso se recorren subcategorías y se juntan varias categorías.

Agregado con pairs.py:
- `langlinks_count(title)` y parámetro `namespace` en `category_members` (14 = subcategorías).
- `langlinks_counts(titles)`: 50 títulos por llamada, sigue normalizaciones y
  redirecciones hasta el título final. Probado contra la versión de a uno
  (124 títulos: mismos conteos, 5 llamadas en vez de 124).
- Límites de Wikimedia: User-Agent con contacto y la palabra "bot" (sin contacto
  el límite es 10 llamadas/min; con contacto, 200/min). Pausa de 0,35 s entre
  llamadas. Ante un 429 se espera lo que indique `Retry-After` (pedía 26 s;
  las esperas de 1-2-4 s no alcanzaban).

### pairs.py — implementado; pendiente aplicar cambios D1–D3 y fijar umbrales
Ver 2.2.1 y 2.3. Imprime un diagnóstico (semillas por grupo, histograma de
idiomas, pares por celda) para fijar los umbrales.
- Prueba chica (Historia de Chile y de Polonia): solo 1 de 71 semillas chilenas
  directas tiene ≥ 40 idiomas → chile_muchos será la celda escasa.
- Cambios acordados tras la prueba: popularidad solo en A (2.2.1), B nombre
  propio y sin artículos panorámicos (2.3).
- Si chile_muchos no alcanza (~90 pares, para tener ~65 preguntas tras perder
  ~25 % en la verificación), en este orden: (1) bajar el umbral de "muchos",
  igual para ambos grupos; (2) agregar temas a ambos grupos por igual;
  (3) PROFUNDIDAD = 2. Umbrales distintos por grupo rompen la comparación.

Implementación de D1–D3 y del diagnóstico (corrida del 24-11-2026):
- **Idiomas de B (D1):** se consultan solo cuando ya se encontró B, con un lote
  de todos los candidatos que pasaron los filtros de título. El lote no depende
  del orden de recorrido ni de qué B ya están usados, así que la consulta es
  siempre la misma y sale de la caché en una segunda corrida.
- **Nombre propio (D2):** la mención es el texto que calza en la intro de A; se
  descarta si ninguna de sus palabras empieza con mayúscula. Se cuenta como
  filtro de título (`sin_mayuscula`), junto a fecha, anexo, panorámico y otro grupo.
- **Panorámicos (D3):** regex sobre el título completo, `\w+ del? <país>`, con
  <país> en Chile + `PAISES_NO_CHILE` (si se cambian los países, el filtro se
  ajusta solo). Como A se aplica al recolectar semillas; como B, en los filtros
  de título y otra vez sobre el título final (redirecciones).
  - Falsos positivos conocidos: marca 166 de 12.500 semillas brutas (1,3 %), entre
    ellas personas ("Faruq de Egipto", "Yolanda de Polonia") e instituciones
    ("Senado de Chile", "Reino de Italia"). Como B también descarta, p. ej.,
    "Universidad de Chile". Se aceptó por simplicidad; si molesta, la alternativa
    es una lista cerrada de palabras panorámicas (Historia, Economía, Clima...).
- **Motivos por semilla A** (diagnóstico, uno por A revisada): `ok`;
  `sin_candidatos` (ningún link de A aparece en su intro); `descartados_titulo`
  (todos los candidatos caen en filtros de título); `b_usado` (los que pasan
  los filtros de título ya fueron usados por otra A); `filtros_pagina` (alguno
  llegó a traer su página, pero ninguno pasó largo, hub, desambiguación o
  título final); `sin_pagina_A`; `celda_llena` (A no se procesó por el tope).
  Aparte se cuentan los candidatos descartados por cada filtro de título.
- **Diagnóstico de profundidad:** cuenta las semillas chilenas con ≥ 40 idiomas
  con profundidad 1 y 2 sin construir pares. Excluye las semillas no chilenas y
  las comunes de la corrida (a PROFUNDIDAD = 1); no recorre los 8 países a
  profundidad 2 porque costaría mucho más. (Primera versión con un bug: no
  excluía las 98 comunes y daba 51 en vez de 30.)

Resultados del diagnóstico (umbrales 10 / 40, PROFUNDIDAD = 1):
- 87,9 min y 7.721 llamadas a la API (la caché ya tenía parte de la corrida anterior).
- Semillas: chile 1.888 (pocos 1.727, muchos 30, intermedias 131); no_chile
  10.348 (pocos 6.059, muchos 827, intermedias 3.462); 98 comunes descartadas.
- Pares antes del muestreo: chile_pocos 300 (tope), **chile_muchos 25**,
  no_chile_pocos 300 (tope), no_chile_muchos 300 (tope).
- El filtro de título que más descarta es `sin_mayuscula` (~700–1.400 candidatos
  por celda); panorámico, 23–55.
- Semillas chilenas con ≥ 40 idiomas: 30 con profundidad 1, 93 con profundidad 2.
- **Contaminación de chile_muchos:** de las 30 semillas, 11 son extranjeros
  condecorados por Chile (García Márquez, Paul McCartney, Dario Fo, Bono...;
  subcategorías "Condecorados con la Orden al Mérito ... Gabriela Mistral /
  Pablo Neruda") y varias son conceptos genéricos (Taxidermia, Embalsamamiento,
  Agujero de la capa de ozono, Poncho). Los temas realmente chilenos y populares
  son ~15 (Cordillera de los Andes, Desierto de Atacama, Moái, Rongo rongo...).
- Países en la muestra no chilena: Perú es 34 % de no_chile_pocos; Japón es
  29 % de no_chile_muchos.

Implementación de D4–D7 (corrida final del 24-11-2026):
- **Profundidad por grupo:** `PROFUNDIDAD_CHILE = 2`, `PROFUNDIDAD_NO_CHILE = 1`
  (cambio por tiempo: no_chile a profundidad 2 eran ~4.000 listados más y ~4-5 h
  en total; sus celdas se llenan igual con profundidad 1).
- **D4 (A menciona su país):** el texto de la intro se pasa a minúsculas y se le
  quitan las tildes (Unicode NFD, borrando las marcas diacríticas); luego se
  buscan las raíces de `RAICES_PAIS` como subcadenas. El país sale de
  `categoria_origen` (lo que va después del último " de "). Se revisa después de
  traer la página de A (necesita la intro) y antes de buscar B; si falla, el
  motivo es `sin_pais`. Se guardan las semillas chilenas con ≥ 21 idiomas que
  descarta, para revisar falsos negativos.
- **D6 (tramos y muestreo emparejado):**
  - `TRAMOS` define los 6 tramos; 11-20 no está en ninguno y se salta sin
    traer la página. El tope es por (celda, tramo): `MAX_PARES_TRAMO = 100`
    (bajado de 150 por tiempo).
  - "Distribución natural" de chile: el tope deforma la proporción entre tramos
    (en pocos, los tramos 0 y 1-5 se llenan y el 6-10 no), así que la cuota de
    cada tramo es proporcional a los *pares disponibles estimados* =
    pares juntados / semillas revisadas × semillas del tramo. Como el orden es
    aleatorio, la estimación es insesgada; en tramos que no se llenaron es exacta.
  - Las cuotas se redondean por restos mayores para sumar exactamente
    N_CANDIDATOS. Si chile tiene ≤ N_CANDIDATOS pares en el nivel, los toma todos.
  - no_chile toma de cada tramo la misma cantidad que chile (si alcanza).
  - La selección se baraja antes de asignar ids, para que no queden ordenados
    por tramo. Cada par lleva el campo `tramo`.
- **D7:** el tramo 21-39 se junta siempre; la regla (`MIN_PARES_MUCHOS_ESTRICTO
  = 90` pares chilenos con ≥ 40) se aplica recién al muestrear.
- **Ritmo de llamadas (wiki.py):** la pausa se mide desde el inicio de la
  llamada anterior (como máximo una cada 0,32 s, ≤ 187/min) en vez de sumar
  0,35 s a la latencia. Casi duplica el ritmo sin pasar el límite de 200/min.

Resultados de la corrida final (36,3 min, 5.881 llamadas; salida completa en
`diagnostico_pairs.txt`):
- Semillas: chile 9.720, no_chile 10.320, 126 comunes descartadas.
- **Regla D7 aplicada: muchos = ≥ 21** (chile junta solo 48 pares con ≥ 40).
- chile_muchos: 148 pares antes del muestreo (21-39: 100 con tope, 40-79: 34,
  80+: 14). Las demás celdas llenan todos sus tramos, salvo no_chile 80+ (91).
- Muestra final: 100 pares por celda (400). Pocos: 50/45/5 por tramo (0, 1-5,
  6-10), mediana 0,5 idiomas en ambas celdas. Muchos: 72/20/8 (21-39, 40-79,
  80+), mediana 28 en chile y 32 en no_chile.
- D4 descarta 77 semillas chilenas con ≥ 21 idiomas: saca a los extranjeros
  condecorados, los feriados religiosos, los videojuegos y "Idioma español".
  Falsos negativos claros (~10): Moái, Rongo rongo, Rano Raraku, Bandera de
  Isla de Pascua (la intro dice "Isla de Pascua", no "Chile"), Provincia de
  Maipo, Quilicura, Tejuela, Lenguas chon.
- Pendientes detectados al revisar los pares:
  - **B cuyo nombre está dentro del título de A** (~10 de 100 en chile_muchos):
    "Ñuñoa → Ñuñoa (estación)", "Roberto Matta → Matta (estación)",
    "Volcán Calbuco → Calbuco", "José Manuel Balmaceda → Balmaceda". La mención
    que calza es el propio nombre de A, no un link real al puente.
  - Siguen pasando B genéricos con mayúscula: "Tierra", "Territorio",
    "Estadio", "Unesco".
  - **Perú es el 58 % de no_chile_pocos** (Japón 24 % de no_chile_muchos).

### Decisiones tras revisar la corrida de D4–D7 (24-11, noche)
- **B cuyo nombre está en el título de A → se descarta** (todos los grupos). Ej.
  "Ñuñoa → Ñuñoa (estación)", "Volcán Calbuco → Calbuco": la mención es el nombre
  del propio A, no un puente real (~10 % de chile_muchos).
- **Falsos negativos de D4 (Moái, Rongo rongo, Rano Raraku: su intro dice "Isla de
  Pascua" y no "Chile") → no se parchan.** Agregar "pascua" o "rapa nui" solo a Chile
  rompería la simetría del filtro (los países de control también pierden artículos
  que nombran una región y no el país, y eso no se puede corregir esta noche).
  chile_muchos ya tiene 148 pares, así que no hacen falta. Se declara como limitación.
- **B genéricos con mayúscula ("Tierra", "Territorio", "Estadio", "Unesco") → los
  revisa la verificación**, con una regla explícita en el prompt del verificador.
- **Perú es el 58 % de no_chile_pocos → se mantiene y se reporta.** Al emparejar
  hacia el tramo de 0 idiomas, el control se llena de artículos de países
  hispanohablantes: en es.wikipedia, casi no hay artículos sin traducciones sobre
  Japón o Francia (suelen ser traducciones). Poner un tope por país dejaría vacío
  ese tramo y rompería el emparejamiento, que es más importante. Consecuencia para
  el análisis: en "pocos", el contraste es en gran parte Chile vs. Perú (conocimiento
  local latinoamericano en ambos lados). Se agrega un corte exploratorio
  no_chile = Perú vs. resto.

Implementación (corrida del 24-11, noche: 7,7 min, 1.195 llamadas):
- **B en el título de A:** filtro de título `en_titulo_A`. Se usa la misma
  búsqueda de mención que en la intro (palabra completa, sin importar
  mayúsculas) del nombre de B sin paréntesis dentro del título de A. Se revisa
  sobre el título del link y otra vez sobre el título final de B. Descarta
  121–215 candidatos por celda; en la salida no queda ningún par así.
- La regla 8 (puente genérico) se copió a `PROMPT_VERIFICACION` en questions.py.
- Resultado: la regla D7 sigue en "muchos" = ≥ 21 (47 pares chilenos con
  ≥ 40); chile_muchos junta 147 pares. Muestra: pocos 51/44/5, muchos 71/21/8,
  igual en chile y no_chile. Perú es el 62 % de no_chile_pocos.

### questions.py — especificación lista (`specs/questions.md`)
Subcomandos: lotes → [LLM] → filtrar → lotes_verif → [LLM, sesión nueva] → split → auditoria.
Split después de la verificación y antes de cualquier experimento: por celda,
5 dev + 60 test con semilla fija; el resto queda como reserva y no se usa para
reemplazar preguntas después de ver resultados.

### questions.py — implementado (24-11)
Probado sobre una copia de `data/` con salidas de LLM inventadas (lotes, filtrar,
lotes_verif, split, auditoria). Decisiones de implementación:
- `N_TEST = 40`, auditoría de 5 por celda (20), según el recorte de 2.4.
- **intro_B recortada a 1500 caracteres en los lotes** (generación y verificación):
  es lo mismo que verá el agente, así el generador elige respuestas observables y
  el filtro `fuera_de_observacion` queda como red de seguridad.
- `MAX_CHARS_OBS = 1500` vive en questions.py y agents.py lo importa: filtro y
  experimento usan el mismo recorte por construcción.
- `normalizar`: puntuación y símbolos → espacio ("Tárrega, 1789" → "tarrega 1789"),
  se borran tildes y caracteres invisibles (Wikipedia trae espacios de ancho cero).
- "Aparece en" = coincidencia de palabras completas sobre texto normalizado
  (" frase " dentro de " texto "), para que "Chile" no calce con "chilena".
- `fuera_de_observacion` para A: la mención de B se busca por el título de B sin
  paréntesis; como ese título puede venir de una redirección y no calzar, también
  vale que `evidencia_A` esté en los primeros 1500 caracteres de la intro de A.
- Las evidencias se comparan normalizadas (tolera espacios y caracteres invisibles).
- `split` usa un `random.Random(SEMILLA)` nuevo por celda: el split de una celda no
  depende de cuántas preguntas tengan las otras.
- dataset.jsonl agrega `tramo` (si viene en pairs.jsonl) como covariable.
- Los lotes se ordenan por `id`, así que mezclan celdas solo en los bordes
  (alfabético: chile_muchos, chile_pocos, ...). El generador no ve la celda igual.

Corrida real (24-11, noche):
- **Herramienta y modelo:** Claude Code. Un subagente Opus nuevo por lote,
  5 en paralelo, con la instrucción "Lee data/lotes_gen/lote_XX.md y sigue las
  instrucciones" (20 lotes). La verificación usó otros 16 subagentes Opus, también
  nuevos (sin el contexto de la generación), con la misma instrucción sobre
  `data/lotes_verif/`.
- Los generadores escribieron un script propio para copiar las evidencias
  carácter por carácter (con los espacios de ancho cero) y revisar las reglas
  mecánicas antes de escribir. Por eso `formato`, `respuesta_no_en_B` y
  `B_en_pregunta` descartaron 0.
- filtrar: 400 → 305 (74 descartadas por el propio LLM, 17 `respuesta_en_A`).
- Verificación: 75 malas de 305. Las reglas más citadas son la 3 (no requiere
  ambos textos), la 2 (respuesta no única), la 6 (redacción) y la 7 (no trata del país).
- split: las 4 celdas completan 5 dev + 40 test. La reserva queda en 18 / 13 /
  16 / 3 (chile_pocos / chile_muchos / no_chile_pocos / no_chile_muchos).

### llm.py, agents.py, correr.py, evaluate.py — implementados (24-11)
Probados localmente con un `llm` falso (sin GPU): parseo de acciones, Buscar con y
sin página, Terminar, acción inválida, max_pasos, trazas, métricas, CSV y gráfico.
Falta la prueba real en Colab (`prueba_colab.ipynb`, 5 preguntas en `data/prueba.jsonl`).
- **Few-shot:** "¿En qué localidad nació el compositor que musicalizó el himno
  nacional de México?" (Himno nacional mexicano → Jaime Nunó → San Juan de las
  Abadesas) y "¿En qué río desemboca el río en el que se encuentran las cataratas
  del Iguazú?" (Cataratas del Iguazú → Río Iguazú → río Paraná). México, Argentina
  y Brasil no son países del estudio, así que no pueden ser A; ninguno está en
  pairs.jsonl. Observaciones reales de wiki.py, **recortadas a las primeras
  oraciones** para ahorrar contexto (con 1500 caracteres completos, los ejemplos
  ocuparían ~1.700 tokens de los 4.096).
- **RAG, ejemplos:** los textos de ejemplo traen las 2 páginas útiles y una que no
  sirve (resultado real de `wiki.search` para la pregunta). Con la pregunta completa
  como consulta, la búsqueda real casi nunca trae las páginas útiles: es esperable,
  RAG fijo es el baseline de "buscar sin decidir qué buscar".
- **Turno del asistente abierto en "Respuesta:"** en directa y rag (igual que ReAct
  abre en "Pensamiento i:"). Evita que el formato castigue a los modelos chicos en
  las condiciones sin agente; corta en el primer salto de línea.
- **Observaciones en una línea:** se recorta a 1500 caracteres y después se quitan
  saltos de línea y espacios de ancho cero (el recorte es el mismo que filtra
  questions.py).
- **Buscar[x]:** se le quitan comillas a x. Si x no es un título válido para la API
  se trata como "no encontrado".
- **ReAct, parseo:** se toma la primera acción válida del paso y se descarta lo que
  el modelo escriba después (p. ej. una observación inventada en la misma línea).
  `formato_ok` = sin acciones inválidas.
- **Contexto:** antes de cada paso se cuentan los tokens del prompt; si prompt + 150
  pasa de `MAX_MODEL_LEN` (4096), `termino="contexto"`. Estimación: instrucciones y
  ejemplos ≈ 900 tokens y cada paso ≈ 450, así que una trayectoria de 7 pasos no
  cabe (corta cerca del paso 6). Si en dev aparecen muchos "contexto", subir
  `MAX_MODEL_LEN` a 8192 (la caché KV alcanza: el 7B tenía ~91k tokens).
- **Tiempo por traza:** las preguntas van en lote, así que `segundos` es el tiempo
  del lote dividido por el número de preguntas.
- **correr.py:** split `prueba` lee `data/prueba.jsonl`. Si el archivo de trazas ya
  existe, solo se reemplazan las condiciones que se vuelven a correr, y se guarda
  después de cada condición (por si Colab se corta).
- **evaluate.py:** archivos con sufijo de split (`tabla_test.csv`, `tabla_dev.csv`...)
  para no pisar dev con test. IC por bootstrap de percentiles, `random.Random(0)`
  por grupo. Ganancia pareada para react − directa, react − rag y rag − directa.
  Colores del gráfico: paleta categórica validada (azul, naranja, aqua; separación
  para daltonismo verificada con script).
- **clasificar_error por observaciones (tras la prueba en Colab):** se clasifica
  según si la respuesta correcta (o un alias) aparece en lo que el modelo leyó, no
  según los títulos vistos. `Buscar[Ramón Carnicer]` cae en una desambiguación que
  igual trae "Tárrega"; comparar títulos lo contaba como `no_encontro_B`. Orden:
  formato → max_pasos/contexto → `leyo_mal` (vio la respuesta) → `no_encontro_B`
  (alguna observación menciona a B) → `no_encontro_A`. En rag las trazas guardan
  solo títulos, así que evaluate.py reconstruye el texto desde la caché con el
  mismo recorte de 1500 caracteres (sin importar agents.py, que importa vLLM).

### Siguientes
Prueba en Colab → dev con 1.5B y 7B → congelar prompts → test → notebook.

---

## 8. Pendientes

- Preguntarle al profe si el dataset de su paper sobre RAG sirve o cuenta como nuevo
  (probablemente no tenga el contraste Chile vs. no Chile).
- Confirmar el trabajo individual: la pauta dice "en dupla".
- Aplicar la regla de 2.2.3 para el umbral de "muchos" con el diagnóstico de la
  próxima corrida; revisar si chile_muchos alcanza (ver 7, pairs.py).
- Revisar a ojo una muestra de pares por celda antes de generar preguntas.
- Revisar la mezcla de países en las celdas no_chile después del emparejamiento
  (antes: Perú 34 % de no_chile_pocos, Japón 29 % de no_chile_muchos). Si sigue
  desbalanceada, poner un tope por país. Por ahora solo se reporta.
- Revisar los falsos negativos del filtro de país (semillas chilenas populares
  descartadas porque su intro no nombra a Chile).

---

## 9. Trabajo relacionado (búsqueda rápida del 24-11-2026; leer antes de citar)

**Separar popularidad de origen (lo más cercano al 2×2):**
- Iferroudjene et al. (2026). *A Wikidata-Based Framework to Measure Cross-Lingual
  Bias in Multilingual LLMs*. KG-LLM @ LREC 2026. Benchmark WILA-PopQA, 9 idiomas,
  con perfiles de popularidad emparejados; separan idioma de la pregunta, contexto
  cultural de la entidad y popularidad. 12 modelos abiertos. Solo memoria, un salto,
  sin herramientas.
- Lalai et al. (2025). *The World According to LLMs: How Geographic Origin Influences
  LLMs' Entity Deduction Capabilities*. COLM 2025. Los LLMs deducen mejor entidades
  del Norte Global; las visitas a Wikipedia y la frecuencia en el preentrenamiento
  solo correlacionan levemente con el desempeño → la brecha geográfica no se explica
  solo por popularidad. Motiva buscar un "efecto local" más allá de la popularidad.

**Latinoamérica y Chile:**
- Karmim, Pino, Contreras et al. (2026). *Leveraging Wikidata for Geographically
  Informed Sociocultural Bias Dataset Creation: Application to Latin America*.
  arXiv 2603.10001 (Inria Chile + U. de Chile). LatamQA: 23k preguntas de alternativas
  desde Wikidata; la cultura de España se conoce mejor que la latinoamericana.
- CHOCLO (CENIA, esfuerzo LatamGPT): ~105k preguntas de conocimiento cultural
  latinoamericano. Visto en un blog; buscar la fuente oficial antes de citar.
- Kawasaki (2026). *Evaluating Geographic Lexical Knowledge in LLMs Across 21
  Spanish-Speaking Countries*. arXiv 2602.09346. Chile tiene el puntaje más bajo en
  el formato de alternativas para todos los modelos.
- BLEnD (Myung et al., NeurIPS 2024 D&B): conocimiento cotidiano de 16 países; los
  LLMs rinden mejor en culturas más presentes en línea (no separa popularidad de origen).

**Recuperación y popularidad:**
- Mallen et al. (2023) y Kandpal et al. (2023): ver 2.2.
- Maekawa et al. (2024). *Retrieval Helps or Hurts? A Deeper Dive into the Efficacy
  of Retrieval Augmentation to Language Models*. arXiv 2402.13492 (WiTQA). La
  recuperación no siempre ayuda; depende de la popularidad del hecho.

**Agentes ReAct con modelos chicos:**
- Pelosi et al. (2026). *Beyond Accuracy: A Qualitative Error Analysis of SLM Agents
  in Multi-Hop QA*. CEUR-WS vol. 4307. Modelos abiertos de 4–9B como agentes ReAct
  con la API de Wikipedia en HotpotQA; taxonomía de errores (página equivocada, falla
  multi-hop, alucinación, respuesta vacía...). Útil como referencia para la
  clasificación de errores de la sección 4.

**Proxy de popularidad:**
- Un trabajo de 2026 (arXiv 2604.24827) reporta que la cantidad de sitelinks (≈ idiomas)
  correlaciona con el reconocimiento de entidades (r ≈ 0,50 en log), pero menos que
  las visitas. Va en limitaciones: langlinks es un proxy razonable pero imperfecto.

**Qué parece nuevo en este proyecto** (según esta búsqueda, no exhaustiva): combinar
(a) recuperación agéntica (ReAct) vs. memoria, (b) origen local vs. popularidad,
separados con emparejamiento, (c) escala con modelos abiertos chicos y (d) preguntas
multi-hop en español sobre Chile. Los trabajos anteriores cubren uno o dos de estos ejes.


---

## 10. Resultados de dev y congelamiento de prompts (24-11, noche)

- **Prompts congelados después de dev.** Dev (20 preguntas) con 1.5B y 7B: sin errores
  de formato ni cortes por contexto. 7B ReAct: EM 0,6–0,8 por celda vs. ~0 de directa.
  Con 5 preguntas por celda no se interpreta ningún efecto; dev solo revisa que todo funcione.
- **1.5B ReAct entra en bucles (11/20 llegan a 7 pasos) → no se corrige.** Es
  comportamiento del modelo, no un bug: repite la misma búsqueda o alterna entre dos.
  Coincide con el paper de ReAct, que reporta pasos repetitivos como error de
  razonamiento, y con su Figura 3 (ReAct por prompting rinde mal en modelos chicos).
  Ajustar el prompt para "salvar" al 1.5B sería acomodar el método. Tres tipos vistos:
  - Alterna entre dos entidades inventadas: para "estación de la línea 5 del Metro" busca
    Metro de Barcelona / Madrid, nunca Santiago (sesgo hacia España, como LatamQA).
  - Repite una entidad inventada ("Cabo de la Vizcaya" en vez de Cabo de Hornos)
    ignorando los títulos similares.
  - Encuentra A (Pedro Aguirre Cerda) pero no extrae el puente y repite.
- **RAG fijo casi no acierta → se mantiene, está bien implementado.** Es esperable en
  preguntas puente: la respuesta está en B y B no se nombra en la pregunta, así que una
  sola búsqueda con la pregunta, en el mejor caso, trae A. Además, el buscador de
  Wikipedia (por palabras clave) funciona mal con preguntas largas: en 18/20 ni siquiera
  trae A. Que una sola recuperación no basta en multi-hop es conocido (Trivedi et al.,
  2023, IRCoT). La comparación ReAct vs. RAG muestra el valor de decidir qué buscar.
- **~15 % de las preguntas no nombran a A** (lo describen indirectamente, ej. "el cabo
  chileno en el extremo opuesto del pasaje de Drake" = Cabo de Hornos), lo que agrega un
  salto. Está balanceado entre celdas (83–88 % nombran A), así que no confunde la
  comparación; se puede usar como covariable en el análisis.

---

## 11. Interpretación de resultados de test (25-11, madrugada)

Detalle en `resultados/analisis_errores.md`; figura en `resultados/figura_principal.png`.

- **Dos predicciones en competencia** para "¿la ventaja de ReAct crece cuando el modelo es pequeño?":
  (A) conocimiento (Mallen, Kandpal): el modelo chico sabe menos → más que ganar al buscar;
  (B) capacidad (ReAct, Figura 3): usar la herramienta exige un modelo capaz → menos ganancia en
  chicos. La pregunta original (sección 1) ya conectaba con la Figura 3. **Resultado: gana B**
  (ganancia 0,17 → 0,29 → 0,56). No "replica" la Figura 3 (allí se compara con CoT, aquí con
  Directa); es consistente con ella.
- Directa está en el piso (≤ 8 %) en todas las celdas, así que las diferencias de ganancia entre celdas
  reflejan cuánto rinde ReAct, no cuánto sabe el modelo de memoria.
- Efecto local: dirección consistente (Chile +4–7 puntos en los 3 modelos), no significativo.
- La caída de ReAct en "muchos" está confundida con el tipo de pregunta (más "ciudad natal" en "pocos").
- 45–65 % de las búsquedas de ReAct no encuentran página por exigir título exacto: buena parte del
  error es de interfaz con la herramienta.
