"""Acceso mínimo a la API de Wikipedia en español, con caché en disco.

Toda consulta pasa por `_consultar`, que primero busca la respuesta en
`cache/wiki/`. Así, una vez descargado, todo el pipeline (dataset y
herramienta de búsqueda de ReAct) es reproducible sin red.

Uso típico:
    import wiki
    wiki.search("Batalla de Maipú")
    wiki.get_page("Batalla de Maipú")
    wiki.OFFLINE = True   # desde aquí, falla si algo no está en caché
"""

import hashlib
import json
import os
import time
from pathlib import Path

import requests

API_URL = "https://es.wikipedia.org/w/api.php"

# La política de Wikimedia exige un User-Agent que identifique al cliente,
# con contacto y la palabra "bot". Sin contacto, la API limita a 10 llamadas
# por minuto; con contacto, a 200 por minuto.
USER_AGENT = (
    "CC5219-ReAct-Chile-bot/0.1 (proyecto universitario; felipe.cardenas@ug.uchile.cl) "
    f"python-requests/{requests.__version__}"
)

# Carpeta de caché, relativa a este archivo (funciona igual en local y Colab).
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "wiki"

# Si es True, una consulta que no esté en caché lanza error en vez de
# llamar a la API. Sirve para verificar que una corrida sale 100% de la caché.
OFFLINE = False

# Separación mínima entre el inicio de dos llamadas reales. 0,32 s da
# ≤ 187 llamadas por minuto, bajo el límite de 200 de Wikimedia.
PAUSA = 0.32
MAX_REINTENTOS = 3   # reintentos ante fallas de red o errores HTTP
TIMEOUT = 30         # segundos de espera máxima por respuesta

# Contador de llamadas reales a la API (útil para comprobar el uso de caché).
llamadas_api = 0

# Momento en que empezó la última llamada real (para respetar PAUSA).
_ultima_llamada = 0.0

_sesion = requests.Session()
_sesion.headers.update({"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# Caché y llamada a la API
# ---------------------------------------------------------------------------

def _ruta_cache(params: dict) -> Path:
    """Ruta del archivo de caché: sha1 de los parámetros ordenados."""
    clave = json.dumps(params, sort_keys=True, ensure_ascii=False)
    nombre = hashlib.sha1(clave.encode("utf-8")).hexdigest() + ".json"
    return CACHE_DIR / nombre


def _segundos_retry_after(resp, por_defecto: int) -> int:
    """Segundos del header Retry-After de un 429, o `por_defecto` si no hay."""
    try:
        return int(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return por_defecto


def _llamar_api(params: dict) -> dict:
    """Hace la petición HTTP real, con pausa y reintentos con espera creciente.

    Si la API responde 429 (demasiadas peticiones), se espera lo que indique
    el header Retry-After antes de reintentar, como pide Wikimedia.
    """
    global llamadas_api, _ultima_llamada
    for intento in range(MAX_REINTENTOS + 1):
        # Se espera solo lo que falta para completar PAUSA desde el inicio de
        # la llamada anterior: la latencia de la red ya cuenta como pausa.
        falta = PAUSA - (time.monotonic() - _ultima_llamada)
        if falta > 0:
            time.sleep(falta)
        _ultima_llamada = time.monotonic()
        try:
            resp = _sesion.get(API_URL, params=params, timeout=TIMEOUT)
            if resp.status_code == 429 and intento < MAX_REINTENTOS:
                espera = _segundos_retry_after(resp, por_defecto=5 * 2 ** intento)
                print(f"[wiki] límite de peticiones (429); reintento en {espera} s")
                time.sleep(espera)
                continue
            resp.raise_for_status()
            datos = resp.json()
            llamadas_api += 1
            # La API puede responder 200 pero con un error en el cuerpo
            # (parámetro inválido, etc.). Eso no se arregla reintentando.
            if "error" in datos:
                raise ValueError(f"Error de la API de Wikipedia: {datos['error']}")
            return datos
        except requests.RequestException as e:
            if intento == MAX_REINTENTOS:
                raise
            espera = 2 ** intento  # 1, 2, 4 segundos
            print(f"[wiki] falla de red ({e}); reintento en {espera} s")
            time.sleep(espera)


def _consultar(params: dict) -> dict:
    """Devuelve la respuesta cruda de la API para `params`, usando la caché.

    Si la respuesta está en disco se lee de ahí y no se toca la red.
    Si no, se llama a la API (salvo en modo OFFLINE) y se guarda tal cual.
    """
    # Parámetros comunes a todas las consultas. formatversion=2 da un JSON
    # más simple (páginas como lista, "missing" como booleano).
    params = {"format": "json", "formatversion": 2, **params}
    ruta = _ruta_cache(params)

    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)

    if OFFLINE:
        raise RuntimeError(f"Modo OFFLINE: la consulta no está en caché: {params}")

    datos = _llamar_api(params)

    # Escritura atómica: si el proceso muere a mitad, no queda un JSON roto.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False)
    os.replace(tmp, ruta)
    return datos


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def search(query: str, k: int = 5) -> list[str]:
    """Títulos de los k primeros resultados de búsqueda (solo artículos)."""
    datos = _consultar({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srlimit": k,
    })
    return [r["title"] for r in datos["query"]["search"]]


def get_page(title: str) -> dict | None:
    """Texto plano completo y links de un artículo, siguiendo redirecciones.

    Devuelve {"title", "text", "links"} o None si la página no existe.
    Se hacen dos tipos de consulta: una para el texto y otra (paginada)
    para los links, para no mezclar la paginación de ambas propiedades.
    """
    # 1) Texto plano. redirects=1 resuelve, p. ej., "Maipú (batalla)".
    datos = _consultar({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    })
    pagina = datos["query"]["pages"][0]
    if pagina.get("missing") or pagina.get("invalid"):
        return None
    titulo_final = pagina["title"]
    texto = pagina.get("extract", "")

    # 2) Links a otros artículos (namespace 0), recorriendo todas las páginas
    # de resultados con el mecanismo "continue" de la API.
    links = []
    params = {
        "action": "query",
        "prop": "links",
        "plnamespace": 0,
        "pllimit": "max",
        "titles": titulo_final,
    }
    while True:
        datos = _consultar(params)
        for p in datos["query"]["pages"]:
            links.extend(l["title"] for l in p.get("links", []))
        if "continue" not in datos:
            break
        # La API indica qué parámetros agregar para pedir la siguiente tanda.
        params = {**params, **datos["continue"]}

    return {"title": titulo_final, "text": texto, "links": links}


def langlinks_count(title: str) -> int:
    """Cantidad de ediciones de Wikipedia en otros idiomas que tienen el artículo.

    Se usa como medida de popularidad. Sigue redirecciones; si la página
    no existe devuelve 0.
    """
    total = 0
    params = {
        "action": "query",
        "prop": "langlinks",
        "lllimit": "max",
        "redirects": 1,
        "titles": title,
    }
    while True:
        datos = _consultar(params)
        for p in datos["query"]["pages"]:
            total += len(p.get("langlinks", []))
        if "continue" not in datos:
            break
        params = {**params, **datos["continue"]}
    return total


def langlinks_counts(titles: list[str]) -> dict[str, int]:
    """Como langlinks_count, pero para muchos títulos: {título pedido: cantidad}.

    Pide hasta 50 títulos por llamada (titles=A|B|C), como recomienda
    Wikimedia, lo que reduce mucho la cantidad de llamadas. La API puede
    normalizar o redirigir los títulos, así que se sigue la pista de cada
    título pedido hasta su título final para devolver el conteo correcto.
    """
    unicos = list(dict.fromkeys(titles))  # sin duplicados, en el mismo orden
    resultado = {}
    for i in range(0, len(unicos), 50):
        lote = unicos[i:i + 50]
        destino = {t: t for t in lote}  # título pedido -> título final
        conteo = {}                     # título final -> cantidad de idiomas
        params = {
            "action": "query",
            "prop": "langlinks",
            "lllimit": "max",
            "redirects": 1,
            "titles": "|".join(lote),
        }
        while True:
            datos = _consultar(params)
            q = datos["query"]
            # Primero normalizaciones ("batalla" -> "Batalla"), luego redirecciones.
            for paso in q.get("normalized", []) + q.get("redirects", []):
                for pedido, actual in destino.items():
                    if actual == paso["from"]:
                        destino[pedido] = paso["to"]
            # El límite de 500 langlinks es para todo el lote, así que un mismo
            # artículo puede venir repartido en varias respuestas: se suma.
            for p in q.get("pages", []):
                n = len(p.get("langlinks", []))
                conteo[p["title"]] = conteo.get(p["title"], 0) + n
            if "continue" not in datos:
                break
            params = {**params, **datos["continue"]}
        for pedido in lote:
            resultado[pedido] = conteo.get(destino[pedido], 0)
    return resultado


def category_members(category: str, limit: int = 500, namespace: int = 0) -> list[str]:
    """Títulos de los miembros de una categoría en `namespace`, hasta `limit`.

    namespace=0 son artículos; namespace=14 son subcategorías (sus títulos
    vienen con el prefijo "Categoría:", listos para volver a consultarlos).
    Ej.: category_members("Categoría:Historia de Chile").
    """
    if not category.startswith("Categoría:"):
        category = "Categoría:" + category

    titulos = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": namespace,
        "cmlimit": min(limit, 500),  # 500 es el máximo que permite la API
    }
    while len(titulos) < limit:
        datos = _consultar(params)
        titulos.extend(m["title"] for m in datos["query"]["categorymembers"])
        if "continue" not in datos:
            break
        params = {**params, **datos["continue"]}
    return titulos[:limit]


# ---------------------------------------------------------------------------
# Prueba rápida
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for ronda in (1, 2):
        antes = llamadas_api
        print(f"=== Ronda {ronda} ===")

        titulos = search("Batalla de Maipú")
        print("Resultados:", titulos)

        pagina = get_page(titulos[0])
        print("Título:", pagina["title"])
        print("Texto:", pagina["text"][:300].replace("\n", " "), "...")
        print("Cantidad de links:", len(pagina["links"]))

        nuevas = llamadas_api - antes
        if nuevas == 0:
            print("-> Todo salió de la caché (0 llamadas a la API).\n")
        else:
            print(f"-> {nuevas} llamadas reales a la API.\n")
