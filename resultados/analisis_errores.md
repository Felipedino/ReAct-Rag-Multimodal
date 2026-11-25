# Análisis de resultados y errores (test, n = 160; 40 por celda)

Figura principal: `resultados/figura_principal.png`. Tablas completas: `tabla_test.csv`,
`ganancia_test.csv`, `errores_test.csv`. Análisis exploratorio hecho con las trazas guardadas.

## 1. Resultados principales

**Exact Match total (IC 95 % bootstrap):**

| Modelo | Directa | RAG fijo | ReAct |
|---|---|---|---|
| 1,5B | 0,02 [0,00–0,04] | 0,07 [0,03–0,11] | 0,19 [0,13–0,25] |
| 3B   | 0,03 [0,01–0,05] | 0,06 [0,03–0,10] | 0,31 [0,24–0,38] |
| 7B   | 0,05 [0,02–0,09] | 0,09 [0,05–0,13] | 0,61 [0,53–0,68] |

**Ganancia ReAct − Directa (EM, bootstrap pareado):**

| Modelo | Total | Chile | No Chile | Pocos idiomas | Muchos idiomas |
|---|---|---|---|---|---|
| 1,5B | 0,17 [0,11–0,24] | 0,19 | 0,15 | 0,22 | 0,11 |
| 3B   | 0,29 [0,22–0,36] | 0,31 | 0,26 | 0,28 | 0,30 |
| 7B   | 0,56 [0,47–0,63] | 0,59 [0,46–0,70] | 0,52 [0,41–0,64] | 0,64 [0,52–0,75] | 0,48 [0,35–0,59] |

**Lectura:**
1. **Efecto piso de la memoria.** Directa acierta ≤ 8 % en todas las celdas y tamaños. Preguntas de dos saltos sobre temas con ≥ 21 idiomas siguen siendo cola larga para modelos ≤ 7B.
   Consistente con la "brecha de composición" (Press et al., 2023): saber cada hecho no basta para encadenarlos.
2. **La ventaja de ReAct crece con el tamaño** (0,17 → 0,29 → 0,56; IC de 1,5B y 7B no se tocan).
   De las dos predicciones en competencia (DECISIONES 1 y 11), gana la de capacidad: consistente con
   la Figura 3 de ReAct. En modelos chicos el cuello de botella es operar la herramienta, no lo que saben.
3. **Efecto local no detectable.** En la ganancia, Chile saca +4, +5 y +6 puntos sobre no Chile (1,5B, 3B, 7B; +6 con el 14B; dirección
   consistente), pero los IC se superponen por completo. Con 80 preguntas por grupo no se puede distinguir.
4. **Popularidad: ReAct (7B) rinde peor en "muchos"**, no porque la memoria ayude (Directa está en el piso).
   Ver sección 3.
5. **RAG fijo falla en recuperar**: en ~80 % de las preguntas ni siquiera llega a A (sección 2).

## 2. Hallazgos cuantitativos de los errores

| Patrón | Dato |
|---|---|
| Búsquedas de ReAct que no encuentran página (título no exacto) | 1,5B: 60 % · 3B: 65 % · 7B: 45 % |
| 1,5B: bucles hasta el máximo de pasos | 66 / 160 |
| 7B `no_encontro_B` | 36; en 20 de ellas el modelo **identificó bien a B** pero la búsqueda no encontró la página (ej. "Comuna de Ñuñoa", "Taltal, Chile", "Provincia Última Esperanza") |
| Responde el puente B en vez de la respuesta | 1,5B: 15 · 3B: 7 · 7B: 4 errores de ReAct |
| RAG `no_encontro_A` | ~128 / 160 en los 3 modelos |
| Respuesta por defecto en Directa | 1,5B responde "Santiago" 9 veces en Chile y "Lima" 5 en no Chile |

**Hipótesis central de errores:** la herramienta `Buscar[x]` exige el título exacto (igual que en el
paper). Los modelos escriben descripciones ("Carretera Panamericana en Quellón") o variantes del título,
reciben "No se encontró" y casi nunca usan la lista de títulos similares. Una parte importante del
error de ReAct es de **interfaz con la herramienta**, no de conocimiento ni de razonamiento.

## 3. ¿Por qué ReAct (7B) rinde peor en chile_muchos (0,45) que en chile_pocos (0,80)?

Exploratorio, n chico. El tipo de pregunta difiere por popularidad:
- Las A poco populares son sobre todo personas (políticos locales) cuya intro empieza con
  "(Ciudad, fecha)": el puente es "ciudad natal", explícito y fácil. Las A populares son lugares,
  conceptos o eventos, con descripciones relacionales más difíciles ("el poblado a 70 km al noroeste del
  volcán Láscar", "la comuna cuya ciudad está 80 km al sur del cerro Paranal").
- Preguntas de "ciudad natal": 45 % en chile_pocos vs. 22 % en chile_muchos.
- 7B ReAct acierta 0,81 en preguntas de "ciudad natal" vs. 0,52 en el resto.
- Dentro de "ciudad natal", chile_muchos (0,78) ≈ chile_pocos (0,83): la brecha casi desaparece.
  Fuera de ese tipo persiste (0,35 vs. 0,77), así que el tipo de pregunta explica solo una parte.
- Además, chile_muchos tuvo la mayor tasa de preguntas malas en la verificación (30 %), sobre todo
  por "respuesta no única".

**Conclusión:** la diferencia por popularidad está confundida con el tipo de pregunta que genera cada
tipo de artículo. No es evidencia de que ReAct funcione peor con temas populares en sí.

## 4. Ejemplos concretos (con hipótesis)

1. **Bucle por confusión local (1,5B, chile_pocos_011).** "¿En qué ciudad nació el militar que lideró
   el golpe de Estado tras el cual se creó el Campo de Prisioneros Chacabuco?" (Valparaíso). Busca la
   pregunta literal, falla, y luego repite 6 veces "Guerra de la Independencia de Chile".
   *Hipótesis:* confunde el campo de prisioneros de 1973 con la batalla de Chacabuco (1817); sin
   conocimiento local no tiene de dónde salir, y el modelo chico no cambia de estrategia.
2. **Bucle con la pregunta como búsqueda (1,5B, chile_pocos_040).** Repite 7 veces
   "Coalición de izquierda que gobernó Chile en 1971" (la Unidad Popular).
   *Hipótesis:* no sabe el nombre de la entidad y la herramienta exige un título; el modelo no usa
   los títulos similares que devuelve la observación.
3. **Identifica B pero la búsqueda falla por el título (7B, chile_muchos_019).** Encuentra en
   "Cerro Paranal" que la ciudad es Taltal, pero busca "Taltal, Chile" (no existe con ese título) varias
   veces y termina respondiendo "Sierra Gorda".
   *Hipótesis:* fragilidad de la interfaz de búsqueda por título exacto; el razonamiento era correcto.
4. **Inventa la respuesta tras una búsqueda fallida (7B, chile_pocos_010).** Llega a Ñuñoa, busca
   "Comuna de Ñuñoa" (sin página) y responde "Ñuñoa Sur" (no existe; la respuesta es Macul).
   *Hipótesis:* cuando la herramienta falla, el modelo completa con una respuesta plausible en vez
   de reintentar: alucinación inducida por la falla de la herramienta.
5. **Responde el puente en vez de la respuesta (7B, chile_muchos_046 y chile_muchos_081).**
   "¿En qué provincia se ubica el poblado a 70 km del volcán Láscar?" → "San Pedro de Atacama"
   (es el poblado; la respuesta es El Loa). En el aeropuerto: responde "Pudahuel" en vez de
   "Las Barrancas". *Hipótesis:* el modelo pierde de vista cuál era la pregunta final y se detiene
   en el primer salto; más frecuente cuanto más chico el modelo (15 casos en 1,5B).
6. **Elige el puente equivocado (7B, no_chile_muchos_089).** "Corte que controla la acción de la
   Administración" = Consejo de Estado; el modelo busca la Corte de Casación y responde "Palais de
   Justice". *Hipótesis:* error de comprensión del dato de A, no de búsqueda.
7. **Error de evaluación, no del modelo (7B, no_chile_pocos_048).** Responde "río Ródano y Saona";
   el gold es "Ródano y Saona" y el EM lo cuenta como error. *Hipótesis:* EM es estricto con palabras
   extra; F1 (0,67 vs. 0,61 de EM en 7B ReAct) muestra que hay aciertos parciales de este tipo.
8. **Se abstiene teniendo la respuesta (7B, no_chile_muchos_095).** Lee la página del teclado QWERTY
   (que menciona a Christopher Sholes) y responde "Información insuficiente…".
   *Hipótesis:* observación larga y dato no destacado; el modelo no conecta el diseño del teclado con
   la pregunta.
9. **RAG con resultados irrelevantes (prueba de formato, 1,5B).** Para "¿En qué ciudad nació el compositor del himno de Chile?"
   la búsqueda con la pregunta trae "Música de Chile", "Blas Parera", "Nacionalismo musical" y el
   modelo responde "Murcia". *Hipótesis:* el buscador de Wikipedia funciona por palabras clave y una
   pregunta larga trae ruido; además, en preguntas puente B no se nombra en la pregunta.
10. **Respuesta por defecto en Directa.** El 1,5B responde "Santiago" 9 veces en preguntas chilenas
    ("¿En qué ciudad nació…?"). *Hipótesis:* sin conocimiento, el modelo devuelve la entidad más
    frecuente asociada al contexto (Chile → Santiago, Perú → Lima).

## 5. Limitaciones

- **Tamaño:** 40 preguntas por celda (IC ≈ ±15 puntos); el efecto local y la interacción no tienen potencia.
- **Calidad del dataset:** generado y verificado por un LLM (Claude); auditoría manual de 20 preguntas:
  1 mala (no multi-hop) y 1 dudosa → 5–10 % de error (IC amplio).
- **Confusión tipo de pregunta × popularidad** (sección 3).
- **Herramienta por título exacto:** fiel al paper, pero penaliza a modelos que no reformulan bien; una
  búsqueda más tolerante probablemente subiría ReAct.
- **Selección de semillas asimétrica** (profundidad 2 en Chile, 1 en no Chile) y control dominado por
  Perú en "pocos" (58 %), por cómo es la Wikipedia en español.
- **Solo Qwen2.5 cuantizado (AWQ)**; un prompt few-shot fijo; sin CoT ni Act (ablaciones por tiempo).
- **EM estricto:** subestima aciertos con palabras extra (ver F1).
- **Cómputo:** Colab T4. Test 7B: ~11 min (incluye descarga); las 3 corridas de test < 1 h.

## 6. Extra: Qwen2.5-14B-Instruct-AWQ

Configuración distinta: solo directa y ReAct (sin RAG fijo, por tiempo), `max_model_len=6144`
(lo que cabe en la T4 con `gpu_memory_utilization=0.95`). Ninguna trayectoria se cortó por contexto.
Cómputo: ~4 min de carga, 17 s de directa, ~25 min de ReAct.

| | Directa | ReAct | Ganancia ReAct − Directa |
|---|---|---|---|
| EM total | 0,06 [0,03–0,11] | 0,69 [0,62–0,76] | 0,63 [0,54–0,70] |
| F1 total | 0,16 | 0,73 | |

Ganancia por grupo: Chile 0,66 [0,56–0,76] · No Chile 0,60 [0,47–0,71] · Pocos 0,65 [0,55–0,76] ·
Muchos 0,61 [0,47–0,72]. ReAct por celda: chile_pocos 0,68 · chile_muchos 0,75 · no_chile_pocos 0,73 ·
no_chile_muchos 0,63.

**Lectura:**
1. **La ganancia sigue creciendo, pero menos:** 0,17 → 0,29 → 0,56 → 0,63. Del 7B al 14B los IC se
   superponen; parece acercarse a un techo, en parte por la herramienta (punto 4).
2. **Efecto local:** Chile +6 puntos sobre no Chile; la misma dirección en los 4 modelos, no significativo.
3. **La brecha de popularidad del 7B desaparece:** chile_muchos 0,75 vs. chile_pocos 0,68 (en el 7B era
   0,45 vs. 0,80). Refuerza la sección 3: no es que los temas populares sean peores para ReAct.
4. **Errores (49 fallos):** no_encontro_B 21, max_pasos 11, no_encontro_A 10, leyo_mal 7, formato 0.
   El 42 % de sus búsquedas no encuentra página (1,5B: 60 %, 3B: 65 %, 7B: 45 %).
   **Sus bucles son distintos a los del 1,5B:** no repite, sino que reformula búsquedas descriptivas, como
   si `Buscar` fuera un buscador ("Patricio Aylwin ciudad nacimiento", "Fundación de la comuna de Renca",
   "Península del Sinaí en la época romana"); casi todas fallan por no ser títulos. Le falta una acción
   para buscar dentro de una página (`lookup` en el paper) o un buscador tolerante.
5. Directa en el piso también con 14B (≤ 10 % por celda): más parámetros no resuelven el multi-hop de memoria.

## 7. Alucinaciones (comparación con el paper de ReAct)

El paper de ReAct (Tabla 2, revisión manual de fallos de PaLM-540B en HotpotQA) reporta 0 % de
alucinación entre los fallos de ReAct. Aquí sí hay. Criterio automático (calculado en la parte B de
`notebook.ipynb`): un fallo de ReAct es alucinación si da una respuesta concreta que no aparece en
ninguna observación que el modelo leyó. Se excluyen los fallos sin respuesta (bucles), las
abstenciones ("No se encontró información", "?", "Desconocida"…) y los aciertos parciales (la respuesta
correcta está dentro de la del modelo, ej. "río Ródano y Saona").

| ReAct | Fallos | Sin respuesta | Se abstiene | Acierto parcial | Responde algo leído | Alucinación | % de los fallos |
|---|---|---|---|---|---|---|---|
| 1,5B | 130 | 69 | 1 | 0 | 46 | 14 | 11 % |
| 3B   | 110 | 5 | 20 | 3 | 54 | 28 | 25 % |
| 7B   | 63 | 3 | 1 | 1 | 36 | 22 | 35 % |
| 14B  | 49 | 11 | 3 | 2 | 25 | 8 | 16 % |

- Ejemplos (7B): "Ñuñoa Sur" (Macul), "Comuna de Ñuñoa" (Puente Alto), Taltal → "Sierra Gorda". Ocurren
  sobre todo cuando la herramienta falla: en vez de decir que no sabe, el modelo rellena.
- El 1,5B "alucina poco" solo en apariencia: más de la mitad de sus fallos son bucles sin respuesta.
- El 3B se abstiene más que los demás (20 fallos), lo que baja su tasa de alucinación.
- **Por qué difiere del paper:** modelos chicos y cuantizados (vs. PaLM-540B), y una herramienta que
  falla en el 42–65 % de las búsquedas con estos datos.
- **Es una cota aproximada:** cuenta como alucinación respuestas correctas sin alias (14B: "St. Malo"
  por "Saint-Malo") o calculadas a partir de lo leído (distancias, fechas). El paper usó revisión manual.
- Una versión anterior de esta cifra (sin excluir abstenciones y aciertos parciales) daba 3B 26 % y
  14B 20 %; la diferencia es solo de definición.
