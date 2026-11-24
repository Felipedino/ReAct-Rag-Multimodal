# Proyecto CC5219: ReAct sobre conocimiento chileno

## Qué es
Proyecto final del curso CC5219 (IA multimodal, U. de Chile). Pregunta:
¿la ventaja de ReAct sobre responder de memoria crece cuando el conocimiento
es local (Chile), cuando es poco conocido, y cuando el modelo es pequeño?
El diseño 2×2 separa "local" de "poco conocido". Se conecta con la Figura 3 del paper
ReAct (Yao et al., 2022), donde ReAct por prompting rinde peor con modelos chicos.

## Diseño (resumen)
- Dataset propio: preguntas multi-hop puente, generadas desde pares de
  artículos enlazados A -> B de Wikipedia en español (CC BY-SA 4.0).
- Diseño 2×2: origen de A (chile / no_chile) × popularidad de A (pocos /
  muchos idiomas en Wikipedia). Celdas: chile_pocos, chile_muchos,
  no_chile_pocos, no_chile_muchos.
- Split: 20 dev (5 por celda) y 240 test (60 por celda). El test no se usa
  para ajustar nada.
- Modelos: Qwen2.5-Instruct 1.5B, 3B y 7B, todos en AWQ (4 bits).
- Condiciones: respuesta directa, CoT, RAG fijo (una búsqueda), Act, ReAct.
  Máximo 7 pasos.
- Métrica: EM y F1 con normalización y lista de alias por respuesta.
- Se guardan trazas completas en JSONL para evaluación y análisis de errores.
- Las decisiones de diseño se registran en DECISIONES.md. Al terminar cada
  etapa, agregar ahí las decisiones de implementación tomadas y sus razones.

## Estructura planeada
- wiki.py         API de Wikipedia + caché en disco
- pairs.py        semillas, popularidad y pares A -> B (data/pairs.jsonl)
- questions.py    generación y verificación de preguntas (con Claude Code, por lotes)
- llm.py          carga del modelo e inferencia
- agents.py       las 5 condiciones
- evaluate.py     métricas y bootstrap
- notebook.ipynb  corre todo de punta a punta en Colab

## Reglas para el código
- Simple y legible: funciones, no clases, salvo que sea claramente necesario.
- Comentarios y docstrings en español.
- No agregar dependencias sin preguntar.
- Implementar solo lo que pide la especificación de cada etapa; no
  adelantar etapas ni modificar otros módulos sin preguntar.
- Reproducibilidad: semillas fijas, caché de todo lo externo.
- Debe correr en Colab con GPU T4.
- Al terminar cada módulo, explicar brevemente cómo funciona y qué
  decisiones se tomaron. Debo poder explicar todo el código en la presentación.


