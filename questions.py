"""Generación, filtrado, verificación y split de las preguntas puente.

Este script NO llama a ningún LLM. Escribe archivos de lote (prompt + datos)
que un LLM procesa en una sesión interactiva, y después lee sus salidas.
Así los prompts y las salidas quedan guardados (reproducibilidad).

Flujo (ver specs/questions.md):
    pairs.jsonl
      lotes        -> data/lotes_gen/lote_XX.md
        [LLM]      -> data/generadas/lote_XX.jsonl
      filtrar      -> data/filtradas.jsonl + data/descartadas.jsonl
      lotes_verif  -> data/lotes_verif/lote_XX.md
        [LLM, sesión nueva] -> data/verificadas/lote_XX.jsonl
      split        -> data/dataset.jsonl (campo split: dev/test/reserva)
      auditoria    -> data/auditoria.md

Uso: python questions.py <lotes|filtrar|lotes_verif|split|auditoria>
"""

import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import wiki
from pairs import CELDAS, introduccion, pais_de_categoria, quitar_parentesis

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parent / "data"

TAM_LOTE = 20        # pares (o preguntas) por archivo de lote
MAX_PALABRAS = 5     # largo máximo de la respuesta
SEMILLA = 0
N_DEV = 5            # preguntas de dev por celda
N_TEST = 40          # preguntas de test por celda (bajó de 60; DECISIONES 2.4)
N_AUDITORIA = 5      # preguntas de test por celda en la auditoría manual

# Caracteres de la intro que ve el agente en cada observación (rag y react).
# agents.py importa esta constante, para que el filtro y el experimento usen
# exactamente el mismo recorte.
MAX_CHARS_OBS = 1500

MOTIVOS = ["descartada_por_llm", "formato", "respuesta_larga", "respuesta_no_en_B",
           "respuesta_en_pregunta", "B_en_pregunta", "respuesta_en_A",
           "fuera_de_observacion", "evidencia_no_textual"]

PROMPT_GENERACION = """\
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
"""

PROMPT_VERIFICACION = """\
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
   no se refiere a algo del país indicado en el campo "pais". B sí puede ser de
   otro país: "el compositor del himno nacional de Chile" es válido aunque el
   compositor sea español; pero una pregunta que solo habla del tabaco de un
   estado de Brasil no trata sobre Chile.
8. El puente (la entidad que describe la pregunta) es un concepto genérico
   ("la Tierra", "un estadio", "el territorio", "la Unesco") y no una entidad específica.

Además, propone alias: otras formas válidas de escribir LA MISMA respuesta
(nombre completo, sin nombre de pila, abreviatura, con o sin "de"). No pongas
respuestas distintas como alias.

Salida: JSONL en la ruta indicada, una línea por pregunta, en el mismo orden:
{"id": "...", "veredicto": "ok" | "mala", "problemas": ["<número>: <explicación>", ...],
 "alias": ["...", ...]}
"""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación y con espacios colapsados.

    La puntuación y los símbolos se cambian por espacio ("Tárrega, 1789" ->
    "tarrega 1789"). Se borran los caracteres invisibles de formato (ej. el
    espacio de ancho cero que trae el texto de Wikipedia junto a las notas).
    evaluate.py la reutiliza y además quita artículos.
    """
    salida = []
    for c in unicodedata.normalize("NFD", texto.lower()):
        categoria = unicodedata.category(c)
        if categoria in ("Mn", "Cf"):      # tildes y caracteres invisibles
            continue
        salida.append(" " if categoria[0] in "PS" else c)
    return " ".join("".join(salida).split())


def contiene(texto: str, frase: str) -> bool:
    """True si `frase` aparece en `texto` como palabras completas (normalizados)."""
    frase = normalizar(frase)
    return bool(frase) and f" {frase} " in f" {normalizar(texto)} "


def leer_jsonl(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def guardar_jsonl(filas: list[dict], ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def leer_salidas_llm(carpeta: Path) -> dict[str, dict]:
    """Une las salidas JSONL de una carpeta en {id: salida}.

    Avisa (sin fallar) si una línea no es JSON válido o no trae id.
    """
    salidas = {}
    for ruta in sorted(carpeta.glob("lote_*.jsonl")):
        with open(ruta, encoding="utf-8") as f:
            for n, linea in enumerate(f, start=1):
                if not linea.strip():
                    continue
                try:
                    fila = json.loads(linea)
                except json.JSONDecodeError:
                    print(f"[aviso] {ruta.name}:{n} no es JSON válido; se ignora")
                    continue
                if "id" not in fila:
                    print(f"[aviso] {ruta.name}:{n} no tiene 'id'; se ignora")
                    continue
                salidas[fila["id"]] = fila
    return salidas


def en_lotes(filas: list[dict]) -> list[list[dict]]:
    """Divide `filas` (ya ordenadas por id) en lotes de TAM_LOTE."""
    return [filas[i:i + TAM_LOTE] for i in range(0, len(filas), TAM_LOTE)]


def escribir_lotes(filas: list[dict], prompt: str, carpeta_md: Path,
                   carpeta_salida: Path, campos):
    """Escribe un .md por lote: prompt, ruta de salida y un ítem JSON por línea.

    `campos(fila)` arma el dict que ve el LLM. No sobrescribe un lote que ya
    tenga su salida (para no cambiar el prompt de un lote ya procesado).
    """
    carpeta_md.mkdir(parents=True, exist_ok=True)
    nuevos = saltados = 0
    for i, lote in enumerate(en_lotes(filas)):
        nombre = f"lote_{i:02d}"
        salida = carpeta_salida / f"{nombre}.jsonl"
        if salida.exists():
            saltados += 1
            continue
        ruta_rel = salida.relative_to(DATA.parent).as_posix()
        lineas = [prompt, f"Ruta de salida: {ruta_rel}", "",
                  f"Ítems ({len(lote)}):", ""]
        lineas += [json.dumps(campos(f), ensure_ascii=False) for f in lote]
        (carpeta_md / f"{nombre}.md").write_text("\n".join(lineas) + "\n",
                                                 encoding="utf-8")
        nuevos += 1
    print(f"{nuevos} lotes escritos en {carpeta_md}; "
          f"{saltados} saltados porque ya tienen salida.")


# ---------------------------------------------------------------------------
# 1. lotes
# ---------------------------------------------------------------------------

def lotes():
    """Lotes de generación a partir de data/pairs.jsonl.

    El generador no ve la celda ni los idiomas. intro_B se recorta a
    MAX_CHARS_OBS, lo mismo que verá el agente: así la respuesta sale de la
    parte observable y el filtro fuera_de_observacion casi no descarta.
    """
    pares = sorted(leer_jsonl(DATA / "pairs.jsonl"), key=lambda p: p["id"])
    escribir_lotes(
        pares, PROMPT_GENERACION, DATA / "lotes_gen", DATA / "generadas",
        lambda p: {"id": p["id"], "titulo_A": p["titulo_A"],
                   "fragmento_A": p["fragmento_A"], "titulo_B": p["titulo_B"],
                   "intro_B": p["intro_B"][:MAX_CHARS_OBS]})


# ---------------------------------------------------------------------------
# 2. filtrar
# ---------------------------------------------------------------------------

def motivo_descarte(par: dict, gen: dict) -> str | None:
    """Primer filtro que falla (en el orden de MOTIVOS), o None si pasa todos."""
    if gen.get("descartar"):
        return "descartada_por_llm"
    campos = ("pregunta", "respuesta", "evidencia_A", "evidencia_B")
    if (not all(isinstance(gen.get(c), str) and gen[c].strip() for c in campos)
            or not gen["pregunta"].strip().endswith("?")
            or not normalizar(gen["respuesta"])):
        return "formato"

    pregunta, respuesta = gen["pregunta"], gen["respuesta"]
    if len(normalizar(respuesta).split()) > MAX_PALABRAS:
        return "respuesta_larga"
    if not contiene(par["intro_B"], respuesta):
        return "respuesta_no_en_B"
    if contiene(pregunta, respuesta):
        return "respuesta_en_pregunta"
    nombre_B = quitar_parentesis(par["titulo_B"])
    if contiene(pregunta, nombre_B):
        return "B_en_pregunta"

    # Intro completa de A (sale de la caché de wiki.py). Si ya contiene la
    # respuesta, la pregunta se responde con un solo salto.
    pagina_A = wiki.get_page(par["titulo_A"])
    intro_A = introduccion(pagina_A["text"]) if pagina_A else par["fragmento_A"]
    if contiene(intro_A, respuesta):
        return "respuesta_en_A"

    # El agente solo ve los primeros MAX_CHARS_OBS caracteres de cada intro.
    # La mención de B se busca por su nombre; como el título de B puede venir
    # de una redirección (y no calzar), también vale que la evidencia de A
    # esté dentro del recorte.
    visible_A = intro_A[:MAX_CHARS_OBS]
    mencion_visible = contiene(visible_A, nombre_B) or contiene(visible_A, gen["evidencia_A"])
    if not contiene(par["intro_B"][:MAX_CHARS_OBS], respuesta) or not mencion_visible:
        return "fuera_de_observacion"

    if (not contiene(par["fragmento_A"], gen["evidencia_A"])
            or not contiene(par["intro_B"], gen["evidencia_B"])
            or not contiene(gen["evidencia_B"], respuesta)):
        return "evidencia_no_textual"
    return None


def filtrar():
    pares = sorted(leer_jsonl(DATA / "pairs.jsonl"), key=lambda p: p["id"])
    salidas = leer_salidas_llm(DATA / "generadas")
    ids_pares = {p["id"] for p in pares}
    for id_ in salidas:
        if id_ not in ids_pares:
            print(f"[aviso] la salida {id_} no corresponde a ningún par")

    # Pares de los lotes ya generados que quedaron sin salida.
    for i, lote in enumerate(en_lotes(pares)):
        if (DATA / "generadas" / f"lote_{i:02d}.jsonl").exists():
            for p in lote:
                if p["id"] not in salidas:
                    print(f"[aviso] lote_{i:02d}: el par {p['id']} no tiene salida")

    filtradas, descartadas = [], []
    conteo = {c: Counter() for c in CELDAS}
    for par in pares:
        gen = salidas.get(par["id"])
        if gen is None:
            continue
        conteo[par["celda"]]["generadas"] += 1
        motivo = motivo_descarte(par, gen)
        if motivo:
            descartadas.append({**par, **gen, "motivo": motivo})
            conteo[par["celda"]][motivo] += 1
        else:
            campos = ("pregunta", "respuesta", "evidencia_A", "evidencia_B")
            filtradas.append({**par, **{c: gen[c].strip() for c in campos}})
            conteo[par["celda"]]["filtradas"] += 1

    guardar_jsonl(filtradas, DATA / "filtradas.jsonl")
    guardar_jsonl(descartadas, DATA / "descartadas.jsonl")

    # Tabla: una fila por motivo, una columna por celda.
    filas = ["generadas"] + MOTIVOS + ["filtradas"]
    print(f"\n{'':24}" + "".join(f"{c:>17}" for c in CELDAS) + f"{'total':>8}")
    for fila in filas:
        valores = [conteo[c][fila] for c in CELDAS]
        print(f"{fila:24}" + "".join(f"{v:>17}" for v in valores) + f"{sum(valores):>8}")
    print(f"\n{len(filtradas)} filtradas, {len(descartadas)} descartadas.")


# ---------------------------------------------------------------------------
# 3. lotes_verif
# ---------------------------------------------------------------------------

def lotes_verif():
    """Lotes de verificación. Se procesan en una sesión NUEVA del LLM.

    El verificador no ve celda, idiomas, evidencias ni títulos. Sí ve el país
    de A, para la regla 7 (igual para Chile y para los países de control).
    """
    filtradas = sorted(leer_jsonl(DATA / "filtradas.jsonl"), key=lambda p: p["id"])
    escribir_lotes(
        filtradas, PROMPT_VERIFICACION, DATA / "lotes_verif", DATA / "verificadas",
        lambda p: {"id": p["id"], "pregunta": p["pregunta"],
                   "respuesta": p["respuesta"],
                   "pais": pais_de_categoria(p["categoria_origen"]),
                   "fragmento_A": p["fragmento_A"],
                   "intro_B": p["intro_B"][:MAX_CHARS_OBS]})


# ---------------------------------------------------------------------------
# 4. split
# ---------------------------------------------------------------------------

CAMPOS_DATASET = ["id", "split", "celda", "tramo", "pregunta", "respuesta", "alias",
                  "titulo_A", "titulo_B", "evidencia_A", "evidencia_B",
                  "idiomas_A", "idiomas_B", "categoria_origen"]


def unir_alias(respuesta: str, alias: list) -> list[str]:
    """Respuesta + alias propuestos, sin duplicados tras normalizar."""
    vistos, salida = set(), []
    for a in [respuesta] + [a for a in alias if isinstance(a, str)]:
        clave = normalizar(a)
        if clave and clave not in vistos:
            vistos.add(clave)
            salida.append(a.strip())
    return salida


def split():
    filtradas = sorted(leer_jsonl(DATA / "filtradas.jsonl"), key=lambda p: p["id"])
    verif = leer_salidas_llm(DATA / "verificadas")

    validas = {c: [] for c in CELDAS}
    conteo = {c: Counter() for c in CELDAS}
    problemas = {c: Counter() for c in CELDAS}
    for q in filtradas:
        c = q["celda"]
        conteo[c]["filtradas"] += 1
        v = verif.get(q["id"])
        if v is None:
            print(f"[aviso] {q['id']} no tiene verificación; queda fuera")
            conteo[c]["sin_verif"] += 1
        elif v.get("veredicto") != "ok":
            conteo[c]["malas"] += 1
            for p in v.get("problemas", []):
                # "3: no requiere ambos textos" -> regla "3"
                problemas[c][str(p).split(":")[0].strip()] += 1
        else:
            validas[c].append({**q, "alias": unir_alias(q["respuesta"], v.get("alias", []))})

    dataset = []
    for c in CELDAS:
        preguntas = validas[c]   # ya ordenadas por id
        random.Random(SEMILLA).shuffle(preguntas)
        if len(preguntas) < N_DEV + N_TEST:
            print(f"[aviso] {c}: solo {len(preguntas)} válidas "
                  f"(< {N_DEV + N_TEST}); test queda más chico")
        for i, q in enumerate(preguntas):
            nombre = "dev" if i < N_DEV else "test" if i < N_DEV + N_TEST else "reserva"
            conteo[c][nombre] += 1
            q = {**q, "split": nombre}
            dataset.append({k: q.get(k) for k in CAMPOS_DATASET})
    guardar_jsonl(dataset, DATA / "dataset.jsonl")

    print(f"\n{'':12}" + "".join(f"{c:>17}" for c in CELDAS))
    for fila in ["filtradas", "sin_verif", "malas", "dev", "test", "reserva"]:
        print(f"{fila:12}" + "".join(f"{conteo[c][fila]:>17}" for c in CELDAS))
    print("\nProblemas más frecuentes en las malas (regla: veces):")
    for c in CELDAS:
        top = ", ".join(f"{r}: {n}" for r, n in problemas[c].most_common(4))
        print(f"  {c}: {top or '-'}")
    print(f"\n{len(dataset)} preguntas en {DATA / 'dataset.jsonl'}")


# ---------------------------------------------------------------------------
# 5. auditoria
# ---------------------------------------------------------------------------

def auditoria():
    """N_AUDITORIA preguntas de test por celda, al azar, para revisar a mano."""
    dataset = leer_jsonl(DATA / "dataset.jsonl")
    rng = random.Random(SEMILLA)
    secciones = ["# Auditoría manual del dataset\n",
                 "Llenar cada `Veredicto:` con ok / mala (+ motivo). "
                 "Sirve para estimar la tasa de error; no se corrige nada a partir de aquí.\n"]
    for c in CELDAS:
        test = sorted((q for q in dataset if q["celda"] == c and q["split"] == "test"),
                      key=lambda q: q["id"])
        for q in rng.sample(test, min(N_AUDITORIA, len(test))):
            secciones.append(
                f"## {q['id']} ({c})\n\n"
                f"- **Pregunta:** {q['pregunta']}\n"
                f"- **Respuesta:** {q['respuesta']}\n"
                f"- **Alias:** {', '.join(q['alias'])}\n"
                f"- **A → B:** {q['titulo_A']} → {q['titulo_B']}\n"
                f"- **evidencia_A:** {q['evidencia_A']}\n"
                f"- **evidencia_B:** {q['evidencia_B']}\n\n"
                f"Veredicto: \n")
    (DATA / "auditoria.md").write_text("\n".join(secciones), encoding="utf-8")
    print(f"Auditoría escrita en {DATA / 'auditoria.md'}")


# ---------------------------------------------------------------------------

SUBCOMANDOS = {"lotes": lotes, "filtrar": filtrar, "lotes_verif": lotes_verif,
               "split": split, "auditoria": auditoria}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SUBCOMANDOS:
        print(f"Uso: python questions.py <{'|'.join(SUBCOMANDOS)}>")
        sys.exit(1)
    SUBCOMANDOS[sys.argv[1]]()
