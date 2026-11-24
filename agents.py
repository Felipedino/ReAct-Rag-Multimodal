"""Las 3 condiciones del experimento: respuesta directa, RAG fijo y ReAct.

Todas reciben el modelo cargado y una lista de preguntas (dicts del dataset),
procesan todas las preguntas en lote y devuelven una traza por pregunta.
Ver specs/experimento.md.

- directa: responde de memoria, sin buscar.
- rag:     una sola búsqueda con el texto de la pregunta (3 resultados) y
           responde con esos textos. Es "buscar sin razonar".
- react:   alterna Pensamiento / Acción / Observación (Yao et al., 2022), con
           dos acciones: Buscar[x] y Terminar[x]. Máximo 7 pasos.

Los ejemplos few-shot son los mismos 2 temas en las 3 condiciones, sobre
entidades que no están en data/pairs.jsonl (México y Argentina/Brasil no son
países del estudio). Sus observaciones son texto real de wiki.py (24-11-2026),
recortado a las primeras oraciones para que el prompt no pase el contexto.
"""

import re
import time

import llm
import wiki
from pairs import introduccion
from questions import MAX_CHARS_OBS

MAX_PASOS = 7          # pasos máximos de ReAct
MAX_TOKENS_PASO = 150  # tokens por paso de ReAct (pensamiento + acción)
MAX_TOKENS_RESP = 32   # tokens para la respuesta en directa y rag
K_RAG = 3              # resultados de búsqueda que usa rag
K_SIMILARES = 5        # títulos sugeridos cuando Buscar no encuentra la página

SISTEMA = "Eres un asistente que responde preguntas de conocimiento general en español."

# Acción válida de ReAct, ej. "Acción 2: Buscar[Jaime Nunó]".
PATRON_ACCION = re.compile(r"Acción \d+:\s*(Buscar|Terminar)\[(.*?)\]")

# ---------------------------------------------------------------------------
# Ejemplos few-shot (mismos 2 temas en las 3 condiciones)
# ---------------------------------------------------------------------------

PREGUNTA_EJ1 = "¿En qué localidad nació el compositor que musicalizó el himno nacional de México?"
RESPUESTA_EJ1 = "San Juan de las Abadesas"
PREGUNTA_EJ2 = "¿En qué río desemboca el río en el que se encuentran las cataratas del Iguazú?"
RESPUESTA_EJ2 = "río Paraná"

# Observaciones reales (wiki.get_page + introduccion), recortadas.
OBS_HIMNO_MX = (
    "El Himno nacional mexicano es uno de los tres símbolos patrios establecidos por "
    "la ley en dicho país, junto con el escudo y la bandera. Compuesto por el poeta "
    "Francisco González Bocanegra y musicalizado por el compositor español Jaime Nunó, "
    "fue estrenado en una ceremonia festiva el 15 de septiembre de 1854; ...")
OBS_NUNO = (
    "Jaime Nunó Roca (en catalán Jaume Nunó i Roca; San Juan de las Abadesas, Gerona, "
    "Cataluña, España, 8 de septiembre de 1824-Nueva York, Estados Unidos, 18 de julio "
    "de 1908), fue un compositor, concertista, director de orquesta y director de "
    "óperas español, célebre por haber musicalizado las estrofas escritas por "
    "Francisco González Bocanegra para dar origen al Himno Nacional de México.")
OBS_SERRAT = (
    "Joan Manuel Serrat Teresa (Barcelona; 27 de diciembre de 1943) es un cantante, "
    "compositor, actor, escritor, poeta y guitarrista retirado español, ...")
OBS_CATARATAS = (
    "Las cataratas del Iguazú (en portugués: cataratas do Iguaçu) son un conjunto de "
    "cataratas que se localizan en el río Iguazú, entre la ciudad de Puerto Iguazú, "
    "provincia de Misiones, Argentina, y Foz do Iguaçu, estado de Paraná, Brasil. ...")
OBS_RIO_IGUAZU = (
    "El río Iguazú (en portugués: Rio Iguaçu) es un río del sur de Brasil y del "
    "nordeste de la Argentina, en el centro-este de América del Sur. Pertenece a la "
    "cuenca del Plata. Nace de la unión de los rios Atuba e Iraí, en la region "
    "metropolitana de la ciudad de Curitiba, en el estado de Paraná, y luego de un "
    "recorrido de aproximadamente 1300 kilómetros desemboca en el río Paraná. ...")
OBS_CUENCA_PLATA = (
    "La cuenca del Plata, también conocida como la región platina, es la cuenca "
    "hidrográfica de América del Sur drenada por el propio Río de la Plata y sus "
    "afluentes. ...")

EJEMPLOS_DIRECTA = f"""\
Pregunta: {PREGUNTA_EJ1}
Respuesta: {RESPUESTA_EJ1}

Pregunta: {PREGUNTA_EJ2}
Respuesta: {RESPUESTA_EJ2}"""

# En rag los textos de ejemplo incluyen las dos páginas útiles y una que no
# sirve (resultado real de wiki.search para la pregunta), para mostrar el formato.
EJEMPLOS_RAG = f"""\
Textos:
[1] Himno nacional mexicano: {OBS_HIMNO_MX}
[2] Joan Manuel Serrat: {OBS_SERRAT}
[3] Jaime Nunó: {OBS_NUNO}
Pregunta: {PREGUNTA_EJ1}
Respuesta: {RESPUESTA_EJ1}

Textos:
[1] Cuenca del Plata: {OBS_CUENCA_PLATA}
[2] Cataratas del Iguazú: {OBS_CATARATAS}
[3] Río Iguazú: {OBS_RIO_IGUAZU}
Pregunta: {PREGUNTA_EJ2}
Respuesta: {RESPUESTA_EJ2}"""

EJEMPLOS_REACT = f"""\
Pregunta: {PREGUNTA_EJ1}
Pensamiento 1: Tengo que buscar el himno nacional de México para saber quién compuso su música, y luego buscar dónde nació esa persona.
Acción 1: Buscar[Himno nacional mexicano]
Observación 1: {OBS_HIMNO_MX}
Pensamiento 2: La música la compuso el español Jaime Nunó. Ahora tengo que buscar dónde nació Jaime Nunó.
Acción 2: Buscar[Jaime Nunó]
Observación 2: {OBS_NUNO}
Pensamiento 3: Jaime Nunó nació en San Juan de las Abadesas. Esa es la respuesta.
Acción 3: Terminar[{RESPUESTA_EJ1}]

Pregunta: {PREGUNTA_EJ2}
Pensamiento 1: Tengo que buscar las cataratas del Iguazú para saber en qué río están, y luego buscar dónde desemboca ese río.
Acción 1: Buscar[Cataratas del Iguazú]
Observación 1: {OBS_CATARATAS}
Pensamiento 2: Las cataratas están en el río Iguazú. Ahora tengo que buscar el río Iguazú para saber dónde desemboca.
Acción 2: Buscar[Río Iguazú]
Observación 2: {OBS_RIO_IGUAZU}
Pensamiento 3: El río Iguazú desemboca en el río Paraná. Esa es la respuesta.
Acción 3: Terminar[{RESPUESTA_EJ2}]"""

INSTR_DIRECTA = (
    "Responde la pregunta con una respuesta corta (un nombre, lugar, número o año), "
    "sin explicación, en el formato:\nRespuesta: <respuesta corta>")

INSTR_RAG = (
    "Responde la pregunta usando los textos de Wikipedia que se entregan. Da una "
    "respuesta corta (un nombre, lugar, número o año), sin explicación, en el "
    "formato:\nRespuesta: <respuesta corta>")

INSTR_REACT = (
    "Responde la pregunta alternando pasos de Pensamiento, Acción y Observación. "
    "En el Pensamiento razonas sobre lo que sabes y lo que falta averiguar. "
    "La Acción puede ser de dos tipos:\n"
    "(1) Buscar[entidad]: busca la entidad en Wikipedia y devuelve el comienzo de su "
    "artículo. Si no existe, devuelve títulos parecidos para que intentes otra búsqueda.\n"
    "(2) Terminar[respuesta]: entrega la respuesta corta (un nombre, lugar, número o "
    "año) y termina.\n"
    f"Tienes como máximo {MAX_PASOS} pasos.")


# ---------------------------------------------------------------------------
# Herramienta: observaciones de Wikipedia
# ---------------------------------------------------------------------------

def _texto_observacion(pagina: dict) -> str:
    """Intro del artículo recortada a MAX_CHARS_OBS, en una sola línea.

    Primero se recorta (el mismo recorte que filtra questions.py) y después se
    limpian saltos de línea y espacios de ancho cero, que solo gastan tokens.
    """
    texto = introduccion(pagina["text"])[:MAX_CHARS_OBS]
    texto = texto.replace("​", "")
    return " ".join(texto.split())


def observacion(titulo: str) -> str:
    """Observación de un artículo: su intro recortada (rag y react)."""
    pagina = wiki.get_page(titulo)
    return _texto_observacion(pagina) if pagina else ""


def buscar(x: str) -> tuple[str | None, str]:
    """Acción Buscar[x]: (título observado o None, texto de la observación).

    Si la página existe (siguiendo redirecciones) se devuelve su intro. Si no,
    se sugieren títulos parecidos, como en el paper.
    """
    pagina = None
    if x:
        try:
            pagina = wiki.get_page(x)
        except ValueError:      # título inválido para la API
            pagina = None
    if pagina:
        return pagina["title"], _texto_observacion(pagina)
    try:
        similares = wiki.search(x, k=K_SIMILARES) if x else []
    except ValueError:
        similares = []
    return None, f"No se encontró [{x}]. Similares: [{', '.join(similares)}]."


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def extraer_respuesta(texto: str) -> tuple[str, bool]:
    """(respuesta, formato_ok). Lo que sigue a "Respuesta:" hasta el fin de línea.

    Si no aparece "Respuesta:", se usa la primera línea no vacía y el formato
    se marca como roto.
    """
    m = re.search(r"Respuesta:[ \t]*(.*)", texto)
    if m:
        return m.group(1).strip(), True
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    return (lineas[0] if lineas else ""), False


def _traza_base(q: dict, condicion: str) -> dict:
    """Traza vacía; correr.py completa el campo "modelo"."""
    return {"id": q["id"], "celda": q.get("celda"), "modelo": None,
            "condicion": condicion, "pregunta": q["pregunta"],
            "respuesta_gold": q.get("respuesta"), "alias": q.get("alias", []),
            "respuesta_modelo": "", "texto_modelo": "", "pasos": [],
            "titulos_vistos": [], "termino": None, "acciones_invalidas": 0,
            "formato_ok": True, "segundos": 0.0}


def _cerrar(trazas: list[dict], inicio: float):
    """Tiempo por pregunta: el lote completo dividido por la cantidad de preguntas
    (las preguntas se procesan juntas, así que no hay un tiempo individual)."""
    por_pregunta = (time.time() - inicio) / max(len(trazas), 1)
    for t in trazas:
        t["segundos"] = round(por_pregunta, 3)
    return trazas


def _responder_corto(modelo, preguntas: list[dict], condicion: str,
                     usuarios: list[str]) -> list[dict]:
    """Directa y rag: un solo llamado en lote con el turno del asistente abierto
    en "Respuesta:", y se corta en el primer salto de línea."""
    prompts = [llm.armar_prompt(modelo, SISTEMA, u, prefijo_asistente="Respuesta:")
               for u in usuarios]
    salidas = llm.generar(modelo, prompts, MAX_TOKENS_RESP, stop=["\n"])
    trazas = []
    for q, salida in zip(preguntas, salidas):
        t = _traza_base(q, condicion)
        t["texto_modelo"] = "Respuesta:" + salida
        t["respuesta_modelo"], t["formato_ok"] = extraer_respuesta(t["texto_modelo"])
        t["termino"] = "ok" if t["formato_ok"] and t["respuesta_modelo"] else "formato"
        trazas.append(t)
    return trazas


# ---------------------------------------------------------------------------
# 1. Respuesta directa
# ---------------------------------------------------------------------------

def directa(modelo, preguntas: list[dict]) -> list[dict]:
    """Responde de memoria, sin herramientas."""
    inicio = time.time()
    usuarios = [f"{INSTR_DIRECTA}\n\nEjemplos:\n\n{EJEMPLOS_DIRECTA}\n\n"
                f"Pregunta: {q['pregunta']}" for q in preguntas]
    return _cerrar(_responder_corto(modelo, preguntas, "directa", usuarios), inicio)


# ---------------------------------------------------------------------------
# 2. RAG fijo
# ---------------------------------------------------------------------------

def rag(modelo, preguntas: list[dict]) -> list[dict]:
    """Una búsqueda con el texto de la pregunta; responde con los K_RAG textos."""
    inicio = time.time()
    usuarios, titulos = [], []
    for q in preguntas:
        encontrados = wiki.search(q["pregunta"], k=K_RAG)
        textos = "\n".join(f"[{i}] {t}: {observacion(t)}"
                           for i, t in enumerate(encontrados, start=1))
        titulos.append(encontrados)
        usuarios.append(f"{INSTR_RAG}\n\nEjemplos:\n\n{EJEMPLOS_RAG}\n\n"
                        f"Textos:\n{textos}\nPregunta: {q['pregunta']}")
    trazas = _responder_corto(modelo, preguntas, "rag", usuarios)
    for t, encontrados in zip(trazas, titulos):
        t["titulos_vistos"] = encontrados
    return _cerrar(trazas, inicio)


# ---------------------------------------------------------------------------
# 3. ReAct
# ---------------------------------------------------------------------------

def react(modelo, preguntas: list[dict], max_pasos: int = MAX_PASOS) -> list[dict]:
    """Bucle Pensamiento / Acción / Observación, en lote.

    En cada paso se genera para todas las preguntas activas a la vez. El
    turno del asistente queda abierto con la trayectoria hasta ahora más
    "Pensamiento i:", y se corta antes de que el modelo invente la observación.
    """
    inicio = time.time()
    trazas = [_traza_base(q, "react") for q in preguntas]
    usuarios = [f"{INSTR_REACT}\n\nEjemplos:\n\n{EJEMPLOS_REACT}\n\n"
                f"Pregunta: {q['pregunta']}" for q in preguntas]
    trayectorias = [""] * len(preguntas)
    activas = list(range(len(preguntas)))

    for paso in range(1, max_pasos + 1):
        # Prompts de las activas; si alguna ya no cabe en el contexto, se corta.
        prompts, indices = [], []
        for i in activas:
            p = llm.armar_prompt(modelo, SISTEMA, usuarios[i],
                                 prefijo_asistente=trayectorias[i] + f"Pensamiento {paso}:")
            if llm.contar_tokens(modelo, p) + MAX_TOKENS_PASO > llm.MAX_MODEL_LEN:
                trazas[i]["termino"] = "contexto"
                continue
            prompts.append(p)
            indices.append(i)

        salidas = llm.generar(modelo, prompts, MAX_TOKENS_PASO, stop=["\nObservación"])

        activas = []
        for i, salida in zip(indices, salidas):
            t = trazas[i]
            m = PATRON_ACCION.search(salida)
            if m is None:
                # Sin acción válida: se avisa en la observación (como el paper).
                texto = salida.rstrip()
                accion = argumento = titulo = None
                obs = "Acción inválida. Usa Buscar[...] o Terminar[...]."
                t["acciones_invalidas"] += 1
                pensamiento = texto.strip()
            else:
                # Se descarta lo que el modelo haya escrito después de la acción.
                texto = salida[:m.end()]
                pensamiento = salida[:m.start()].strip()
                accion = m.group(1)
                argumento = m.group(2).strip().strip("\"'«»").strip()
                if accion == "Terminar":
                    trayectorias[i] += f"Pensamiento {paso}:{texto}"
                    t["pasos"].append({"pensamiento": pensamiento, "accion": accion,
                                       "argumento": argumento, "titulo_observado": None,
                                       "observacion": None})
                    t["respuesta_modelo"] = argumento
                    t["termino"] = "ok"
                    continue
                titulo, obs = buscar(argumento)
                if titulo:
                    t["titulos_vistos"].append(titulo)

            trayectorias[i] += f"Pensamiento {paso}:{texto}\nObservación {paso}: {obs}\n"
            t["pasos"].append({"pensamiento": pensamiento, "accion": accion,
                               "argumento": argumento, "titulo_observado": titulo,
                               "observacion": obs})
            activas.append(i)

        if not activas:
            break

    for i in activas:            # llegaron al máximo de pasos sin Terminar
        trazas[i]["termino"] = "max_pasos"
    for t, tray in zip(trazas, trayectorias):
        t["texto_modelo"] = tray
        t["formato_ok"] = t["acciones_invalidas"] == 0
    return _cerrar(trazas, inicio)


CONDICIONES = {"directa": directa, "rag": rag, "react": react}
