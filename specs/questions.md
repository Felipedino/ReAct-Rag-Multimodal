# Especificación: questions.py

## Objetivo
Convertir cada par A → B de `data/pairs.jsonl` en una pregunta puente con su
respuesta, filtrarla, verificarla y separar dev/test. Ver DECISIONES.md 2.3–2.5.

La pregunta describe a B con información de A, sin nombrarlo, y la respuesta
es un dato de B. Ejemplo:
- A = Himno Nacional de Chile. fragmento_A: "La música fue compuesta por Ramón Carnicer…"
- B = Ramón Carnicer. intro_B: "Ramón Carnicer (Tárrega, 1789 – Madrid, 1855) fue un compositor español…"
- Pregunta: "¿En qué ciudad nació el compositor de la música del himno nacional de Chile?"
- Respuesta: "Tárrega"

## Quién genera y verifica
Un LLM en una sesión interactiva (Claude, en Claude Code o en un chat de Cowork
con acceso a la carpeta; no hay API key). El script NO llama a ningún LLM: solo
escribe archivos de lote con las instrucciones y lee las salidas. Así los prompts
y las salidas quedan guardados (reproducibilidad), y el flujo no depende de la
herramienta usada. Se anota en DECISIONES.md qué herramienta y modelo se usaron.

## Flujo (subcomandos: `python questions.py <subcomando>`)

```
pairs.jsonl
  └─ lotes            → data/lotes_gen/lote_XX.md          (prompt + 20 pares)
       [LLM]          → data/generadas/lote_XX.jsonl
  └─ filtrar          → data/filtradas.jsonl + data/descartadas.jsonl
  └─ lotes_verif      → data/lotes_verif/lote_XX.md        (prompt + 20 preguntas)
       [LLM, sesión nueva] → data/verificadas/lote_XX.jsonl
  └─ split            → data/dataset.jsonl                 (campo split: dev/test/reserva)
  └─ auditoria        → data/auditoria.md                  (40 preguntas de test al azar)
```

Usar `sys.argv[1]` para elegir el subcomando (sin argparse; es más simple).
Cada subcomando es una función. Todas las rutas son relativas a questions.py.

## 1. `lotes`
- Lee `data/pairs.jsonl`, lo ordena por `id` y lo divide en lotes de
  `TAM_LOTE = 20`, mezclando celdas en el orden del archivo (no hace falta separar).
- Cada `lote_XX.md` contiene, en este orden:
  1. `PROMPT_GENERACION` (constante en el script, texto abajo).
  2. La ruta de salida exacta: `data/generadas/lote_XX.jsonl`.
  3. Los pares, uno por línea en JSON, solo con: `id`, `titulo_A`,
     `fragmento_A`, `titulo_B`, `intro_B`. (Sin celda ni idiomas: el
     generador no debe saber a qué celda pertenece cada par.)
- No sobrescribe un lote que ya tenga su salida en `data/generadas/`.

### PROMPT_GENERACION (borrador; se puede pulir con las 5 primeras salidas)
```
Vas a escribir preguntas de trivia de dos saltos ("puente") en español.
Para cada par de artículos de Wikipedia (A y B) recibes:
- fragmento_A: oraciones del artículo A que mencionan a B.
- intro_B: la introducción del artículo B.

Escribe UNA pregunta por par que cumpla TODO esto:
1. Describe a B usando un dato de fragmento_A, sin nombrar a B (ni su nombre
   completo ni una parte reconocible de él). La descripción debe identificar a B
   de forma única.
2. La respuesta es un dato de intro_B: una entidad, lugar, número o año, de
   1 a 5 palabras, copiado tal como aparece en intro_B.
3. La respuesta NO aparece en fragmento_A ni en la pregunta.
4. Para responder hacen falta los dos artículos: el dato de A para identificar
   a B y el dato de B para responder.
5. El dato no cambia con el tiempo (nada de "actual", poblaciones, cargos vigentes).
6. Una sola oración, natural, que termina en "?".

Si el par no permite una buena pregunta, descártalo en vez de forzarla.

Salida: un archivo JSONL en la ruta indicada, una línea por par, en el mismo orden:
{"id": "...", "pregunta": "...", "respuesta": "...",
 "evidencia_A": "<oración copiada textual de fragmento_A>",
 "evidencia_B": "<oración copiada textual de intro_B que contiene la respuesta>"}
o, si se descarta:
{"id": "...", "descartar": true, "motivo": "..."}

Ejemplo:
fragmento_A: "La música fue compuesta por Ramón Carnicer..."
intro_B: "Ramón Carnicer (Tárrega, 1789 – Madrid, 1855) fue un compositor español..."
{"id": "ej", "pregunta": "¿En qué ciudad nació el compositor de la música del himno nacional de Chile?",
 "respuesta": "Tárrega", "evidencia_A": "La música fue compuesta por Ramón Carnicer...",
 "evidencia_B": "Ramón Carnicer (Tárrega, 1789 – Madrid, 1855) fue un compositor español..."}
```

## 2. `filtrar`
- Lee todas las salidas de `data/generadas/` y las une con su par (por `id`).
- Avisa (sin fallar) si un par de un lote no tiene salida o si una línea no es JSON válido.
- Función `normalizar(texto)`: minúsculas, sin tildes, sin puntuación, espacios
  colapsados. (evaluate.py la reutilizará; agregar ahí la eliminación de artículos.)
- Filtros, todos sobre texto normalizado. Una pregunta pasa si cumple todos:

| Motivo de descarte        | Regla |
|---------------------------|-------|
| `descartada_por_llm`      | la salida trae `"descartar": true` |
| `formato`                 | faltan campos o la pregunta no termina en "?" |
| `respuesta_larga`         | la respuesta tiene más de 5 palabras |
| `respuesta_no_en_B`       | la respuesta no aparece en `intro_B` |
| `respuesta_en_pregunta`   | la respuesta aparece en la pregunta |
| `B_en_pregunta`           | el nombre de B (título sin paréntesis) aparece en la pregunta |
| `respuesta_en_A`          | la respuesta aparece en la intro de A (traerla con `wiki.get_page(titulo_A)`, sale de la caché) |
| `fuera_de_observacion`    | la respuesta no está en los primeros 1500 caracteres de `intro_B`, o la mención de B no está en los primeros 1500 caracteres de la intro de A (el agente solo ve eso; ver specs/experimento.md) |
| `evidencia_no_textual`    | `evidencia_A` no está en `fragmento_A`, o `evidencia_B` no está en `intro_B`, o `evidencia_B` no contiene la respuesta |

- `respuesta_en_A` mira la intro completa de A, no solo el fragmento: si A ya
  contiene la respuesta, la pregunta se responde con un solo salto.
- Salidas: `data/filtradas.jsonl` (par + pregunta) y `data/descartadas.jsonl`
  (lo mismo + `motivo`). Se registra el primer motivo que falla.
- Imprime una tabla: por celda, generadas / descartadas por motivo / filtradas.

## 3. `lotes_verif`
- Igual que `lotes`, pero sobre `data/filtradas.jsonl`, con `PROMPT_VERIFICACION`
  y salida en `data/verificadas/lote_XX.jsonl`.
- Cada ítem lleva: `id`, `pregunta`, `respuesta`, `pais`, `fragmento_A`, `intro_B`.
  Sin celda, ni idiomas, ni evidencias, ni títulos. `pais` se obtiene de
  `categoria_origen` ("Categoría:Historia de Chile" → "Chile").
- Dar el país al verificador no sesga la dificultad: solo juzga, no escribe, y la
  regla es la misma para Chile y para los países de control.
- Se hace en una sesión nueva del LLM (sin el contexto de la generación), para
  que no defienda sus propias preguntas.

### PROMPT_VERIFICACION (borrador)
```
Eres un revisor estricto de un dataset de preguntas de dos saltos. Tu trabajo es
encontrar razones por las que cada pregunta sea MALA. Recibes la pregunta, la
respuesta propuesta y los dos textos de los que salió.

Marca la pregunta como "mala" si ocurre cualquiera de estas cosas:
1. No se puede responder con fragmento_A + intro_B, o la respuesta es incorrecta.
2. La respuesta no es única: hay otra respuesta razonable igual de válida.
3. No requiere ambos textos (se responde solo con la pregunta, solo con A, o
   es conocimiento trivial).
4. La descripción de la pregunta es ambigua: podría referirse a otra entidad que no es B.
5. El dato puede cambiar con el tiempo.
6. La pregunta es confusa o mal redactada.
7. El primer salto (el dato de fragmento_A que usa la pregunta para describir a B)
   no se refiere a algo de <pais>. B sí puede ser de otro país: "el compositor del
   himno nacional de Chile" es válido aunque el compositor sea español; pero una
   pregunta que solo habla del tabaco de un estado de Brasil no trata sobre Chile.

Además, propone alias: otras formas válidas de escribir LA MISMA respuesta
(nombre completo, sin nombre de pila, abreviatura, con o sin "de"). No pongas
respuestas distintas como alias.

Salida: JSONL en la ruta indicada, una línea por pregunta, en el mismo orden:
{"id": "...", "veredicto": "ok" | "mala", "problemas": ["<número>: <explicación>", ...],
 "alias": ["...", ...]}
```

## 4. `split`
- Une `filtradas.jsonl` con las verificaciones. Se quedan solo las de veredicto
  `ok`. `alias` = respuesta + alias propuestos, sin duplicados tras `normalizar`.
- Por celda: ordenar por `id`, mezclar con `random.Random(SEMILLA)` (SEMILLA = 0),
  las primeras `N_DEV = 5` → dev, las siguientes `N_TEST = 40` → test, el resto → reserva.
  (Se bajó de 60 a 40 por tiempo; ver DECISIONES.md 2.4.)
- Si una celda tiene menos de 45 válidas: avisar; dev se llena primero y test
  queda más chico.
- Escribe `data/dataset.jsonl`. Campos de cada pregunta:
  `id`, `split`, `celda`, `pregunta`, `respuesta`, `alias`, `titulo_A`, `titulo_B`,
  `evidencia_A`, `evidencia_B`, `idiomas_A`, `idiomas_B`, `categoria_origen`.
- Imprime una tabla: por celda, filtradas / malas en verificación (y problemas
  más frecuentes) / dev / test / reserva.
- La reserva NO se usa para reemplazar preguntas después de ver resultados.

## 5. `auditoria`
- Toma 20 preguntas de test al azar (`random.Random(SEMILLA)`), estratificadas
  por celda (5 por celda).
- Escribe `data/auditoria.md`: una sección por pregunta con pregunta, respuesta,
  alias, evidencia_A, evidencia_B y una línea `Veredicto: ` para llenar a mano.
- Sirve para estimar la tasa de error del dataset, que se reporta en el informe.
  No se corrige ni se borra nada a partir de esta auditoría.

## Fuera de alcance
- No llamar a ninguna API de LLM.
- No modificar pairs.py ni wiki.py.
- No usar el `random` global (vLLM lo modifica): solo instancias `random.Random`.

## Prueba sugerida
1. `lotes` y revisar `lote_00.md` a mano.
2. Generar solo `lote_00` con el LLM y correr `filtrar`. Revisar las descartadas:
   si un filtro descarta preguntas buenas (o al revés), ajustar el prompt antes de
   generar el resto.
