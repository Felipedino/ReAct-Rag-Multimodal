"""Construcción de pares de artículos A -> B para preguntas multi-hop puente.

A enlaza a B en su introducción. Luego (en la etapa siguiente) la pregunta
describirá a B usando información de A, y la respuesta saldrá de B.

Diseño 2x2: origen de A (chile / no_chile) x popularidad de A (pocos / muchos
idiomas). La popularidad es la cantidad de ediciones de Wikipedia en otros
idiomas que tienen el artículo, agrupada en tramos. La de B se guarda como
covariable, no filtra.

Flujo:
  1. Semillas A: artículos de categorías temáticas de Chile y de otros países,
     incluyendo subcategorías (profundidad 2 en Chile, 1 en el resto) y sin
     artículos panorámicos.
  2. Idiomas de cada semilla (en lotes) -> tramo; el tramo 11-20 se descarta.
  3. Se recorren las semillas en orden aleatorio (semilla fija). Cada A debe
     mencionar su país en la intro; luego se busca un B en esa intro que pase
     todos los filtros. Cada (celda, tramo) se llena hasta MAX_PARES_TRAMO.
  4. Se elige el umbral de "muchos" con la regla fijada de antemano (D7) y se
     hace un muestreo emparejado por tramo entre chile y no_chile.
  5. Se guardan los pares en data/pairs.jsonl y se imprime un diagnóstico.

Uso: python pairs.py
"""

import json
import random
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

import wiki

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

TEMAS = ["Historia", "Geografía", "Política", "Cultura"]

# Países de control: distintos continentes y tamaños. En esta Wikipedia todas
# sus categorías siguen el patrón "Categoría:<Tema> de <País>".
PAISES_NO_CHILE = ["Francia", "Japón", "Polonia", "Perú",
                   "Egipto", "Australia", "Canadá", "Italia"]

CATEGORIAS_CHILE = [f"Categoría:{t} de Chile" for t in TEMAS]
CATEGORIAS_NO_CHILE = [f"Categoría:{t} de {p}"
                       for p in PAISES_NO_CHILE for t in TEMAS]

# Raíces para reconocer que un texto menciona al país (D4). Se buscan como
# subcadenas en el texto en minúsculas y sin tildes: "chilen" calza con
# "chileno" y "chilena"; "frances" con "francés" y "francesa".
RAICES_PAIS = {
    "Chile": ["chile", "chilen"],
    "Francia": ["francia", "frances"],
    "Japón": ["japon"],
    "Polonia": ["polonia", "polac"],
    "Perú": ["peru"],
    "Egipto": ["egipt", "egipc"],
    "Australia": ["australia"],
    "Canadá": ["canad"],
    "Italia": ["itali"],
}

# Niveles de subcategorías a recorrer. Chile usa 2 para tener más semillas
# populares; no_chile usa 1 porque sus celdas se llenan igual (y así la
# corrida cabe en el tiempo disponible).
PROFUNDIDAD_CHILE = 2
PROFUNDIDAD_NO_CHILE = 1
LIMITE_CATEGORIA = 5000  # máximo de miembros que se leen por categoría

MIN_CHARS_B = 1500       # B debe tener al menos este largo de texto
MAX_LINKS_B = 800        # B con más links que esto se considera un "hub"

# Tramos de idiomas de A: (nombre, mínimo, máximo). El tramo 11-20 no está en
# ninguna lista, así que esas semillas se descartan (zona intermedia).
TRAMOS = {
    "pocos": [("0", 0, 0), ("1-5", 1, 5), ("6-10", 6, 10)],
    "muchos": [("21-39", 21, 39), ("40-79", 40, 79), ("80+", 80, 10**6)],
}

# Regla D7, fijada antes de ver los pares: si chile junta al menos
# MIN_PARES_MUCHOS_ESTRICTO pares solo con los tramos TRAMOS_MUCHOS_ESTRICTO,
# "muchos" = ≥ 40 idiomas; si no, "muchos" = ≥ 21. Los pares del tramo 21-39
# se juntan siempre, y la elección se aplica recién al muestrear.
TRAMOS_MUCHOS_ESTRICTO = ["40-79", "80+"]
MIN_PARES_MUCHOS_ESTRICTO = 90

# Tope de pares por (celda, tramo). Como las semillas se recorren en orden
# aleatorio, los pares juntados son una muestra al azar de los disponibles.
MAX_PARES_TRAMO = 100
N_CANDIDATOS = 100       # pares por celda en la salida (como máximo)
SEMILLA = 0

SALIDA = Path(__file__).resolve().parent / "data" / "pairs.jsonl"

CELDAS = ["chile_pocos", "chile_muchos", "no_chile_pocos", "no_chile_muchos"]

MESES = ("enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
         "septiembre|octubre|noviembre|diciembre")

# Títulos que son años o fechas (se comparan con el título completo).
PATRONES_FECHA = [
    re.compile(r"\d+"),                                         # 1818
    re.compile(r"\d+ a\. ?C\."),                                # 44 a. C.
    re.compile(rf"\d{{1,2}} de ({MESES})", re.IGNORECASE),      # 5 de abril
    re.compile(r"años \d+", re.IGNORECASE),                     # Años 1810
    re.compile(r"siglo [IVXLC]+( a\. ?C\.)?", re.IGNORECASE),   # Siglo XIX
]

# Artículos panorámicos: "<una palabra> de/del <país del estudio>", ej.
# "Historia de Chile", "Economía de Polonia", "Clima de Japón".
PAISES_ESTUDIO = ["Chile"] + PAISES_NO_CHILE
PATRON_PANORAMICO = re.compile(rf"\w+ del? ({'|'.join(PAISES_ESTUDIO)})")

# Motivos por los que una semilla A no produce par (para el diagnóstico).
MOTIVOS = ["ok", "sin_pais", "sin_candidatos", "descartados_titulo", "b_usado",
           "filtros_pagina", "sin_pagina_A", "tramo_lleno"]


# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------

def es_panoramico(titulo: str) -> bool:
    """True si el título completo es '<una palabra> de/del <país del estudio>'."""
    return PATRON_PANORAMICO.fullmatch(titulo) is not None


def articulos_de_categoria(categoria: str, profundidad: int) -> list[str]:
    """Artículos de una categoría y de sus subcategorías hasta `profundidad`."""
    articulos = wiki.category_members(categoria, limit=LIMITE_CATEGORIA)
    if profundidad > 0:
        subcategorias = wiki.category_members(
            categoria, limit=LIMITE_CATEGORIA, namespace=14)
        for sub in subcategorias:
            articulos += articulos_de_categoria(sub, profundidad - 1)
    return articulos


def recolectar_semillas(categorias: list[str], profundidad: int) -> dict[str, str]:
    """Devuelve {título: categoría de origen} para un grupo de categorías.

    Si un artículo aparece en varias categorías del grupo, se queda con la
    primera (la categoría de primer nivel, no la subcategoría). Los artículos
    panorámicos no se incluyen.
    """
    semillas = {}
    for categoria in categorias:
        for titulo in articulos_de_categoria(categoria, profundidad):
            if not es_panoramico(titulo):
                semillas.setdefault(titulo, categoria)
    return semillas


def quitar_comunes(semillas: dict) -> set:
    """Borra de ambos grupos los artículos que están en los dos (origen ambiguo)."""
    comunes = semillas["chile"].keys() & semillas["no_chile"].keys()
    for grupo in semillas:
        for titulo in comunes:
            del semillas[grupo][titulo]
    return comunes


def tramo_de(idiomas: int) -> tuple[str, str] | None:
    """(nivel, tramo) de una cantidad de idiomas, o None si cae en 11-20."""
    for nivel, tramos in TRAMOS.items():
        for nombre, minimo, maximo in tramos:
            if minimo <= idiomas <= maximo:
                return nivel, nombre
    return None


def pais_de_categoria(categoria: str) -> str:
    """'Categoría:Historia de Chile' -> 'Chile'."""
    return categoria.rsplit(" de ", 1)[-1]


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def introduccion(texto: str) -> str:
    """Texto antes de la primera línea de sección ('== ... ==')."""
    m = re.search(r"^==.*==\s*$", texto, flags=re.MULTILINE)
    return (texto[:m.start()] if m else texto).strip()


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: 'Japón' -> 'japon'."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def menciona_pais(texto: str, pais: str) -> bool:
    """True si `texto` contiene alguna raíz del país (D4)."""
    texto = normalizar(texto)
    return any(raiz in texto for raiz in RAICES_PAIS[pais])


def quitar_parentesis(titulo: str) -> str:
    """'Mercurio (planeta)' -> 'Mercurio'."""
    return re.sub(r"\s*\([^)]*\)$", "", titulo)


def buscar_mencion(nombre: str, texto: str):
    """Primera mención de `nombre` como palabra completa, sin importar mayúsculas."""
    patron = r"\b" + re.escape(nombre) + r"\b"
    return re.search(patron, texto, flags=re.IGNORECASE)


def oraciones_que_mencionan(texto: str, nombre: str) -> str:
    """Oraciones de `texto` que mencionan a `nombre` (se divide por puntuación)."""
    oraciones = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return " ".join(o.strip() for o in oraciones if buscar_mencion(nombre, o))


# ---------------------------------------------------------------------------
# Filtros sobre B
# ---------------------------------------------------------------------------

def es_fecha(titulo: str) -> bool:
    return any(p.fullmatch(titulo) for p in PATRONES_FECHA)


def es_desambiguacion(titulo: str, texto: str) -> bool:
    """Heurística: get_page no trae categorías, así que se mira título e intro."""
    intro = introduccion(texto).lower()
    return ("(desambiguación)" in titulo.lower()
            or "puede referirse a" in intro
            or "puede hacer referencia a" in intro)


def motivo_descarte_titulo(titulo: str, mencion: str | None,
                           semillas_otro: dict) -> str | None:
    """Motivo para descartar un candidato B sin llamar a la API, o None si pasa.

    `mencion` es el texto de su primera mención en la intro de A. Si no tiene
    ninguna palabra con mayúscula ("historia", "general"), B no es nombre
    propio. No se exige mayúscula inicial porque en español los eventos van
    con minúscula ("batalla de Rancagua").
    Con mencion=None se revisa solo el título (para el título final de B).
    """
    if titulo.startswith("Anexo:"):
        return "anexo"
    if es_fecha(titulo):
        return "fecha"
    if es_panoramico(titulo):
        return "panoramico"
    if titulo in semillas_otro:
        return "otro_grupo"
    if mencion is not None and not any(p[0].isupper() for p in mencion.split()):
        return "sin_mayuscula"
    return None


def candidatos_B(pagina_A: dict) -> list[tuple[str, str]]:
    """Links de A cuyo nombre (sin paréntesis) aparece en la intro de A.

    Devuelve pares (título, texto de la primera mención), ordenados por la
    posición de esa mención: los primeros suelen ser los más relacionados
    con A. No hace llamadas a la API.
    """
    intro = introduccion(pagina_A["text"])
    encontrados = []
    for titulo in pagina_A["links"]:
        if titulo == pagina_A["title"]:
            continue
        m = buscar_mencion(quitar_parentesis(titulo), intro)
        if m:
            encontrados.append((m.start(), titulo, m.group(0)))
    return [(titulo, mencion) for _, titulo, mencion in sorted(encontrados)]


def buscar_B(pagina_A: dict, semillas_otro: dict, usados: set,
             descartes: Counter) -> tuple[dict | None, str]:
    """Primer candidato B que pasa todos los filtros.

    Devuelve (datos de B o None, motivo). El motivo es "ok" o la razón por la
    que A no produjo par. `descartes` acumula por qué se descartó cada
    candidato por título (fecha, anexo, panorámico, otro grupo, sin mayúscula).

    Los filtros van de más barato a más caro: primero los que solo miran el
    título y luego los que necesitan la página de B (≥2 llamadas cada una).
    """
    candidatos = candidatos_B(pagina_A)
    if not candidatos:
        return None, "sin_candidatos"

    validos = []
    for titulo, mencion in candidatos:
        motivo = motivo_descarte_titulo(titulo, mencion, semillas_otro)
        if motivo:
            descartes[motivo] += 1
        else:
            validos.append(titulo)
    if not validos:
        return None, "descartados_titulo"

    libres = [t for t in validos if t not in usados]
    if not libres:
        return None, "b_usado"

    for titulo in libres:
        pagina_B = wiki.get_page(titulo)
        if pagina_B is None:
            continue
        final = pagina_B["title"]  # puede diferir si el link era una redirección
        if final == pagina_A["title"] or final in usados:
            continue
        if (motivo_descarte_titulo(final, None, semillas_otro)
                or es_desambiguacion(final, pagina_B["text"])):
            continue
        if (len(pagina_B["text"]) < MIN_CHARS_B
                or len(pagina_B["links"]) > MAX_LINKS_B):
            continue

        # Idiomas de B: solo covariable para el análisis, no filtra.
        # Se consulta el lote de candidatos válidos (no depende del orden
        # de recorrido, así que la consulta queda igual en la caché).
        idiomas = wiki.langlinks_counts(validos)
        return {"titulo_B": final, "pagina_B": pagina_B,
                "idiomas_B": idiomas[titulo],
                "nombre_B": quitar_parentesis(titulo)}, "ok"
    return None, "filtros_pagina"


# ---------------------------------------------------------------------------
# Construcción de pares
# ---------------------------------------------------------------------------

def construir_pares(semillas: dict, idiomas: dict) -> dict:
    """Recorre las semillas en orden aleatorio y arma a lo más un par por A.

    Devuelve un diccionario con:
      pares[(celda, tramo)]       lista de pares juntados
      semillas_tramo[(c, t)]      semillas de ese (celda, tramo)
      procesadas[(c, t)]          semillas revisadas (no saltadas por el tope)
      motivos[celda]              resultado de cada semilla A
      descartes[celda]            candidatos B descartados por título
      sin_pais_chile              (título, idiomas) de semillas chilenas con
                                  ≥ 21 idiomas descartadas por D4
    Cada B se usa una sola vez en todo el dataset. Los dos orígenes se
    mezclan en el mismo orden aleatorio para que ninguno tenga prioridad al
    disputarse un mismo B.
    """
    orden = sorted((origen, t) for origen in semillas for t in semillas[origen])
    random.Random(SEMILLA).shuffle(orden)

    claves = [(f"{origen}_{nivel}", nombre)
              for origen in ("chile", "no_chile")
              for nivel, tramos in TRAMOS.items() for nombre, _, _ in tramos]
    res = {
        "pares": {clave: [] for clave in claves},
        "semillas_tramo": Counter(),
        "procesadas": Counter(),
        "motivos": {celda: Counter() for celda in CELDAS},
        "descartes": {celda: Counter() for celda in CELDAS},
        "sin_pais_chile": [],
    }
    for origen, titulo in orden:
        nivel_tramo = tramo_de(idiomas[titulo])
        if nivel_tramo:
            celda = f"{origen}_{nivel_tramo[0]}"
            res["semillas_tramo"][(celda, nivel_tramo[1])] += 1

    pares = res["pares"]
    usados = set()
    for i, (origen, titulo_A) in enumerate(orden, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(orden)} semillas revisadas, "
                  f"{sum(len(p) for p in pares.values())} pares, "
                  f"llamadas a la API: {wiki.llamadas_api}")
        if all(len(p) >= MAX_PARES_TRAMO for p in pares.values()):
            break

        nivel_tramo = tramo_de(idiomas[titulo_A])
        if nivel_tramo is None:
            continue  # tramo 11-20: ni siquiera se trae la página
        nivel, tramo = nivel_tramo
        celda = f"{origen}_{nivel}"
        if len(pares[(celda, tramo)]) >= MAX_PARES_TRAMO:
            res["motivos"][celda]["tramo_lleno"] += 1
            continue
        res["procesadas"][(celda, tramo)] += 1

        pagina_A = wiki.get_page(titulo_A)
        if pagina_A is None:
            res["motivos"][celda]["sin_pagina_A"] += 1
            continue

        # D4: A debe mencionar su país en la introducción.
        intro_A = introduccion(pagina_A["text"])
        categoria = semillas[origen][titulo_A]
        if not menciona_pais(intro_A, pais_de_categoria(categoria)):
            res["motivos"][celda]["sin_pais"] += 1
            if origen == "chile" and nivel == "muchos":
                res["sin_pais_chile"].append((titulo_A, idiomas[titulo_A]))
            continue

        otro = "no_chile" if origen == "chile" else "chile"
        b, motivo = buscar_B(pagina_A, semillas[otro], usados,
                             res["descartes"][celda])
        res["motivos"][celda][motivo] += 1
        if b is None:
            continue
        usados.add(b["titulo_B"])
        pares[(celda, tramo)].append({
            "celda": celda,
            "tramo": tramo,
            "titulo_A": pagina_A["title"],
            "titulo_B": b["titulo_B"],
            "fragmento_A": oraciones_que_mencionan(intro_A, b["nombre_B"]),
            "intro_B": introduccion(b["pagina_B"]["text"]),
            "idiomas_A": idiomas[titulo_A],
            "idiomas_B": b["idiomas_B"],
            "categoria_origen": categoria,
        })
    return res


# ---------------------------------------------------------------------------
# Umbral de "muchos" (D7) y muestreo emparejado (D6)
# ---------------------------------------------------------------------------

def elegir_tramos(pares: dict) -> tuple[dict, str]:
    """Aplica la regla D7. Devuelve ({nivel: tramos usados}, regla aplicada)."""
    n_estricto = sum(len(pares[("chile_muchos", t)]) for t in TRAMOS_MUCHOS_ESTRICTO)
    todos_muchos = [nombre for nombre, _, _ in TRAMOS["muchos"]]
    tramos_pocos = [nombre for nombre, _, _ in TRAMOS["pocos"]]
    if n_estricto >= MIN_PARES_MUCHOS_ESTRICTO:
        regla = f"muchos = ≥ 40 (chile junta {n_estricto} pares con ≥ 40)"
        return {"pocos": tramos_pocos, "muchos": TRAMOS_MUCHOS_ESTRICTO}, regla
    regla = (f"muchos = ≥ 21 (chile junta solo {n_estricto} pares con ≥ 40, "
             f"menos de {MIN_PARES_MUCHOS_ESTRICTO})")
    return {"pocos": tramos_pocos, "muchos": todos_muchos}, regla


def disponibles_estimados(res: dict, clave: tuple) -> float:
    """Pares que habría en (celda, tramo) si se hubieran revisado todas sus semillas.

    Si el tramo se llenó, se extrapola con la tasa de éxito observada
    (pares / semillas revisadas); como el orden es aleatorio, la estimación
    es insesgada. Si no se llenó, es el conteo exacto.
    """
    if res["procesadas"][clave] == 0:
        return 0.0
    tasa = len(res["pares"][clave]) / res["procesadas"][clave]
    return tasa * res["semillas_tramo"][clave]


def repartir(pesos: dict, total: int) -> dict:
    """Reparte `total` entre tramos en proporción a `pesos` (restos mayores)."""
    suma = sum(pesos.values())
    if suma == 0:
        return {t: 0 for t in pesos}
    exactas = {t: total * w / suma for t, w in pesos.items()}
    cuotas = {t: int(x) for t, x in exactas.items()}
    faltan = total - sum(cuotas.values())
    por_resto = sorted(exactas, key=lambda t: exactas[t] - cuotas[t], reverse=True)
    for t in por_resto[:faltan]:
        cuotas[t] += 1
    return cuotas


def muestrear(res: dict, tramos_por_nivel: dict) -> list[dict]:
    """Muestreo emparejado por tramo, con semilla fija.

    Para cada nivel, chile toma hasta N_CANDIDATOS pares con su distribución
    natural por tramo (proporcional a los pares disponibles estimados), y
    no_chile toma de cada tramo la misma cantidad que chile, si alcanza.
    Así las dos celdas de un nivel tienen la misma distribución de idiomas.
    """
    rng = random.Random(SEMILLA)
    pares = res["pares"]
    filas = []
    for nivel, tramos in tramos_por_nivel.items():
        chile = f"chile_{nivel}"
        total_chile = sum(len(pares[(chile, t)]) for t in tramos)
        if total_chile <= N_CANDIDATOS:
            cuotas = {t: len(pares[(chile, t)]) for t in tramos}
        else:
            pesos = {t: disponibles_estimados(res, (chile, t)) for t in tramos}
            cuotas = repartir(pesos, N_CANDIDATOS)
            cuotas = {t: min(c, len(pares[(chile, t)])) for t, c in cuotas.items()}

        for celda in (chile, f"no_chile_{nivel}"):
            seleccion = []
            for t in tramos:
                disponibles = pares[(celda, t)]
                seleccion += rng.sample(disponibles, min(cuotas[t], len(disponibles)))
            rng.shuffle(seleccion)  # para que los ids no queden ordenados por tramo
            for i, par in enumerate(seleccion):
                filas.append({"id": f"{celda}_{i:03d}", **par})
    return filas


def guardar_jsonl(filas: list[dict], ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------

TRAMOS_HISTOGRAMA = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 39), (40, 79),
                     (80, 10**6)]


def etiqueta_tramo(lo: int, hi: int) -> str:
    if lo == hi:
        return f"{lo}"
    return f"{lo}+" if hi == 10**6 else f"{lo}-{hi}"


def imprimir_histograma(nombre: str, valores: list[int]):
    print(f"\nIdiomas de las semillas: {nombre} ({len(valores)} artículos)")
    conteos = [sum(lo <= v <= hi for v in valores) for lo, hi in TRAMOS_HISTOGRAMA]
    maximo = max(conteos) or 1
    for (lo, hi), n in zip(TRAMOS_HISTOGRAMA, conteos):
        barra = "#" * round(40 * n / maximo)
        print(f"  {etiqueta_tramo(lo, hi):>6} | {barra:<40} {n}")


def imprimir_diagnostico_semillas(semillas: dict, idiomas: dict, comunes: set):
    print("\n=== Diagnóstico de semillas ===")
    print(f"Descartadas por estar en ambos grupos: {len(comunes)}")
    for grupo in semillas:
        print(f"{grupo}: {len(semillas[grupo])} semillas")
    for grupo in semillas:
        imprimir_histograma(grupo, [idiomas[t] for t in semillas[grupo]])


def imprimir_pares_por_tramo(res: dict):
    print("\n=== Pares por (celda, tramo), antes del muestreo ===")
    print(f"  {'celda':<16} {'tramo':>6} {'semillas':>9} {'revisadas':>10} "
          f"{'pares':>6} {'disp. estimados':>16}")
    for (celda, tramo), lista in res["pares"].items():
        clave = (celda, tramo)
        lleno = " (tope)" if len(lista) >= MAX_PARES_TRAMO else ""
        print(f"  {celda:<16} {tramo:>6} {res['semillas_tramo'][clave]:>9} "
              f"{res['procesadas'][clave]:>10} {len(lista):>6} "
              f"{disponibles_estimados(res, clave):>16.0f}{lleno}")


def imprimir_motivos(res: dict):
    print("\n=== Resultado de cada semilla A, por celda ===")
    for celda in CELDAS:
        motivos = res["motivos"][celda]
        partes = [f"{m} {motivos[m]}" for m in MOTIVOS if motivos[m]]
        print(f"  {celda:<16} " + ", ".join(partes))
    print("\nCandidatos B descartados por título (todas las A revisadas):")
    for celda in CELDAS:
        partes = [f"{m} {n}" for m, n in res["descartes"][celda].most_common()]
        print(f"  {celda:<16} " + ", ".join(partes))


def imprimir_sin_pais(res: dict):
    lista = sorted(res["sin_pais_chile"], key=lambda x: -x[1])
    print(f"\n=== Semillas chilenas con ≥ 21 idiomas descartadas por D4 "
          f"({len(lista)}) ===")
    for titulo, n in lista:
        print(f"  {n:>4}  {titulo}")


def imprimir_muestra(filas: list[dict]):
    print("\n=== Muestra final: idiomas de A por celda ===")
    for celda in CELDAS:
        de_celda = [f for f in filas if f["celda"] == celda]
        if not de_celda:
            print(f"  {celda:<16} 0 pares")
            continue
        por_tramo = Counter(f["tramo"] for f in de_celda)
        mediana = statistics.median(f["idiomas_A"] for f in de_celda)
        tramos = ", ".join(f"{t}: {n}" for t, n in sorted(
            por_tramo.items(), key=lambda x: tramo_orden(x[0])))
        print(f"  {celda:<16} {len(de_celda)} pares, mediana {mediana:g} | {tramos}")

    print("\n=== Muestra final: países en las celdas no_chile ===")
    for celda in ("no_chile_pocos", "no_chile_muchos"):
        de_celda = [f for f in filas if f["celda"] == celda]
        paises = Counter(pais_de_categoria(f["categoria_origen"]) for f in de_celda)
        partes = [f"{p} {100 * n / len(de_celda):.0f}%" for p, n in paises.most_common()]
        print(f"  {celda:<16} " + ", ".join(partes))

    print("\n=== Muestra final: títulos A de chile_muchos ===")
    for f in sorted((f for f in filas if f["celda"] == "chile_muchos"),
                    key=lambda f: -f["idiomas_A"]):
        print(f"  {f['idiomas_A']:>4}  {f['titulo_A']} -> {f['titulo_B']}")


def tramo_orden(tramo: str) -> int:
    """Posición de un tramo en el orden pocos -> muchos (para imprimir)."""
    nombres = [n for tramos in TRAMOS.values() for n, _, _ in tramos]
    return nombres.index(tramo)


def imprimir_ejemplos(filas: list[dict], n: int = 5):
    """Primeros n pares de cada celda en la muestra (ya está en orden aleatorio)."""
    print(f"\n=== {n} pares al azar por celda ===")
    for celda in CELDAS:
        print(f"\n[{celda}]")
        for fila in [f for f in filas if f["celda"] == celda][:n]:
            print(f"  {fila['titulo_A']} ({fila['idiomas_A']}) -> "
                  f"{fila['titulo_B']} ({fila['idiomas_B']})")
            print(f"      {fila['fragmento_A'][:300]}")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    inicio = time.time()

    print("Recolectando semillas...")
    semillas = {"chile": recolectar_semillas(CATEGORIAS_CHILE, PROFUNDIDAD_CHILE),
                "no_chile": recolectar_semillas(CATEGORIAS_NO_CHILE,
                                                PROFUNDIDAD_NO_CHILE)}
    comunes = quitar_comunes(semillas)

    print("Consultando idiomas de las semillas (en lotes de 50)...")
    todas = [t for grupo in semillas for t in semillas[grupo]]
    idiomas = wiki.langlinks_counts(todas)
    imprimir_diagnostico_semillas(semillas, idiomas, comunes)

    print("\nConstruyendo pares...")
    res = construir_pares(semillas, idiomas)
    imprimir_pares_por_tramo(res)
    imprimir_motivos(res)
    imprimir_sin_pais(res)

    tramos_por_nivel, regla = elegir_tramos(res["pares"])
    print(f"\n=== Regla D7 aplicada: {regla} ===")

    filas = muestrear(res, tramos_por_nivel)
    guardar_jsonl(filas, SALIDA)
    print(f"\n{len(filas)} pares guardados en {SALIDA}")
    imprimir_muestra(filas)
    imprimir_ejemplos(filas)

    minutos = (time.time() - inicio) / 60
    print(f"\nTiempo total: {minutos:.1f} min. "
          f"Llamadas reales a la API: {wiki.llamadas_api}")


if __name__ == "__main__":
    main()
