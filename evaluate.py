"""Métricas, intervalos de confianza, ganancia de ReAct y clasificación de errores.

Uso: python evaluate.py [split]        (por defecto "test")
Lee resultados/trazas_*_<split>.jsonl y escribe en resultados/:
  tabla_<split>.csv     EM / F1 / contiene con IC 95 % por modelo × condición × grupo
  ganancia_<split>.csv  diferencia de EM (react - directa, react - rag) con IC pareado
  errores_<split>.csv   tipo de error de cada traza fallida de rag y react
  em_<split>.png        EM por celda y condición, un panel por modelo
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import wiki
from pairs import introduccion, quitar_parentesis
from questions import MAX_CHARS_OBS, normalizar

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"

N_BOOTSTRAP = 1000
ORDEN_MODELOS = ["1.5B", "3B", "7B", "14B"]
ORDEN_CONDICIONES = ["directa", "rag", "react"]
ORDEN_CELDAS = ["chile_pocos", "chile_muchos", "no_chile_pocos", "no_chile_muchos"]

ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo"}


# ---------------------------------------------------------------------------
# Métricas por traza
# ---------------------------------------------------------------------------

def tokens(texto: str) -> list[str]:
    """normalizar() de questions.py, sin artículos y con "del" -> "de"."""
    return ["de" if w == "del" else w
            for w in normalizar(texto).split() if w not in ARTICULOS]


def exact_match(pred: str, gold: str) -> float:
    return float(bool(tokens(gold)) and tokens(pred) == tokens(gold))


def f1(pred: str, gold: str) -> float:
    """F1 de tokens (como SQuAD/HotpotQA): solapamiento entre bolsas de palabras."""
    p, g = tokens(pred), tokens(gold)
    comunes = sum((Counter(p) & Counter(g)).values())
    if comunes == 0:
        return 0.0
    precision, recall = comunes / len(p), comunes / len(g)
    return 2 * precision * recall / (precision + recall)


def contiene(pred: str, gold: str) -> float:
    """1 si la respuesta correcta aparece dentro de la del modelo (palabras completas).

    Métrica secundaria: los modelos chicos a veces agregan palabras
    ("la ciudad de Tárrega")."""
    p, g = " ".join(tokens(pred)), " ".join(tokens(gold))
    return float(bool(g) and f" {g} " in f" {p} ")


def puntuar(traza: dict) -> dict:
    """Agrega em, f1 y contiene a la traza: el máximo sobre todos los alias."""
    golds = traza["alias"] or [traza["respuesta_gold"]]
    pred = traza["respuesta_modelo"] or ""
    traza["em"] = max(exact_match(pred, g) for g in golds)
    traza["f1"] = max(f1(pred, g) for g in golds)
    traza["contiene"] = max(contiene(pred, g) for g in golds)
    return traza


# ---------------------------------------------------------------------------
# Agrupaciones e intervalos
# ---------------------------------------------------------------------------

def grupos_de(celda: str) -> list[tuple[str, str]]:
    """Grupos a los que pertenece una pregunta: su celda, su origen, su
    popularidad y el total. Ej. chile_pocos -> celda, chile, pocos, total."""
    origen, popularidad = celda.rsplit("_", 1)
    return [("celda", celda), ("origen", origen),
            ("popularidad", popularidad), ("total", "total")]


def intervalo(valores: list[float], semilla: int = 0) -> tuple[float, float]:
    """IC 95 % por bootstrap de percentiles: se remuestrea con reemplazo
    N_BOOTSTRAP veces y se toman los percentiles 2,5 y 97,5 de la media."""
    if not valores:
        return float("nan"), float("nan")
    rng = random.Random(semilla)
    n = len(valores)
    medias = sorted(sum(rng.choices(valores, k=n)) / n for _ in range(N_BOOTSTRAP))
    return medias[int(0.025 * N_BOOTSTRAP)], medias[int(0.975 * N_BOOTSTRAP) - 1]


def _agrupar(trazas: list[dict]) -> dict:
    """{(modelo, condicion, agrupacion, grupo): [trazas]}"""
    grupos = {}
    for t in trazas:
        for agrupacion, grupo in grupos_de(t["celda"]):
            grupos.setdefault((t["modelo"], t["condicion"], agrupacion, grupo), []).append(t)
    return grupos


def tabla(trazas: list[dict]) -> pd.DataFrame:
    """n, EM, F1 y contiene (con IC 95 %) por modelo × condición × grupo."""
    filas = []
    for (modelo, condicion, agrupacion, grupo), ts in _agrupar(trazas).items():
        fila = {"modelo": modelo, "condicion": condicion,
                "agrupacion": agrupacion, "grupo": grupo, "n": len(ts)}
        for m in ("em", "f1", "contiene"):
            valores = [t[m] for t in ts]
            lo, hi = intervalo(valores)
            fila[m], fila[f"{m}_lo"], fila[f"{m}_hi"] = sum(valores) / len(valores), lo, hi
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(["modelo", "agrupacion", "grupo", "condicion"],
                                           ignore_index=True)


def ganancia(trazas: list[dict], a: str = "react", b: str = "directa") -> pd.DataFrame:
    """EM(a) - EM(b) por modelo y grupo, con IC por bootstrap pareado.

    Pareado: se remuestrean preguntas y cada pregunta remuestreada aporta su
    diferencia a - b, así ambas condiciones usan la misma muestra. Es la cifra
    central de la pregunta de investigación.
    """
    em = {}   # (modelo, condicion, id) -> em
    celda = {}
    for t in trazas:
        em[(t["modelo"], t["condicion"], t["id"])] = t["em"]
        celda[t["id"]] = t["celda"]
    diferencias = {}  # (modelo, agrupacion, grupo) -> [em_a - em_b]
    for (modelo, condicion, id_), valor in em.items():
        if condicion != a or (modelo, b, id_) not in em:
            continue
        for agrupacion, grupo in grupos_de(celda[id_]):
            diferencias.setdefault((modelo, agrupacion, grupo), []).append(
                valor - em[(modelo, b, id_)])
    filas = []
    for (modelo, agrupacion, grupo), d in diferencias.items():
        lo, hi = intervalo(d)
        filas.append({"modelo": modelo, "comparacion": f"{a} - {b}",
                      "agrupacion": agrupacion, "grupo": grupo, "n": len(d),
                      "ganancia_em": sum(d) / len(d), "ic_lo": lo, "ic_hi": hi})
    return pd.DataFrame(filas).sort_values(["modelo", "agrupacion", "grupo"],
                                           ignore_index=True)


# ---------------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------------

def observaciones(traza: dict) -> list[str]:
    """Textos que leyó el modelo en una traza de rag o react.

    react los guarda en sus pasos. rag guarda solo los títulos, así que su texto
    se reconstruye desde la caché de wiki.py con el mismo recorte que usa
    agents.py (no se importa agents porque importa vLLM). Los saltos de línea y
    espacios de ancho cero que agents.py limpia no importan: se normaliza igual.
    """
    if traza["condicion"] == "react":
        return [p["observacion"] for p in traza["pasos"] if p.get("observacion")]
    textos = []
    for titulo in traza["titulos_vistos"]:
        pagina = wiki.get_page(titulo)
        if pagina:
            textos.append(introduccion(pagina["text"])[:MAX_CHARS_OBS])
    return textos


def clasificar_error(traza: dict, par: dict) -> str | None:
    """Tipo de error de una traza de rag o react (None si acertó).

    Se mira lo que el modelo leyó, no los títulos: puede llegar a B por una
    desambiguación o una redirección ("Ramón Carnicer" es una desambiguación
    que igual trae la respuesta), y comparar títulos lo contaría como no
    encontrado.
    - leyo_mal: la respuesta correcta (o un alias) aparece en alguna
      observación, pero respondió mal (lectura o razonamiento).
    - Si nunca la vio, es falla de búsqueda: no_encontro_B si alguna
      observación menciona a B (llegó al puente pero no a su página), y
      no_encontro_A si ninguna lo menciona.
    "Alucinación" y "error del dataset" se asignan a mano en el análisis.
    `par` es la pregunta del dataset (trae titulo_B).
    """
    if traza["em"] == 1:
        return None
    if traza["termino"] == "formato" or traza["acciones_invalidas"] >= 3:
        return "formato"
    if traza["termino"] in ("max_pasos", "contexto"):
        return traza["termino"]
    textos = observaciones(traza)
    golds = traza["alias"] or [traza["respuesta_gold"]]
    if any(contiene(texto, g) for texto in textos for g in golds):
        return "leyo_mal"
    nombre_B = quitar_parentesis(par["titulo_B"])
    if any(contiene(texto, nombre_B) for texto in textos):
        return "no_encontro_B"
    return "no_encontro_A"


def errores(trazas: list[dict], preguntas: dict) -> pd.DataFrame:
    filas = []
    for t in trazas:
        if t["condicion"] not in ("rag", "react"):
            continue
        tipo = clasificar_error(t, preguntas[t["id"]])
        if tipo:
            filas.append({"id": t["id"], "modelo": t["modelo"], "condicion": t["condicion"],
                          "celda": t["celda"], "tipo": tipo,
                          "respuesta_modelo": t["respuesta_modelo"],
                          "respuesta_gold": t["respuesta_gold"]})
    return pd.DataFrame(filas, columns=["id", "modelo", "condicion", "celda", "tipo",
                                        "respuesta_modelo", "respuesta_gold"])


# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

# Paleta categórica validada (azul, naranja, aqua), una por condición.
COLORES = {"directa": "#2a78d6", "rag": "#eb6834", "react": "#1baf7a"}


def grafico(tab: pd.DataFrame, ruta: Path):
    """EM por celda: barras agrupadas por condición, con IC 95 %; un panel por modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = tab[tab["agrupacion"] == "celda"]
    modelos = [m for m in ORDEN_MODELOS if m in set(datos["modelo"])]
    condiciones = [c for c in ORDEN_CONDICIONES if c in set(datos["condicion"])]
    if not modelos:
        return
    fig, ejes = plt.subplots(1, len(modelos), figsize=(4.5 * len(modelos), 3.8),
                             sharey=True, squeeze=False)
    ancho = 0.8 / len(condiciones)
    for eje, modelo in zip(ejes[0], modelos):
        for k, cond in enumerate(condiciones):
            fila = (datos[(datos["modelo"] == modelo) & (datos["condicion"] == cond)]
                    .set_index("grupo").reindex(ORDEN_CELDAS))
            xs = [i + (k - (len(condiciones) - 1) / 2) * ancho for i in range(len(ORDEN_CELDAS))]
            eje.bar(xs, fila["em"], width=ancho * 0.9, color=COLORES.get(cond), label=cond,
                    yerr=[fila["em"] - fila["em_lo"], fila["em_hi"] - fila["em"]],
                    error_kw={"elinewidth": 1, "ecolor": "#555555", "capsize": 0})
        eje.set_title(f"Qwen2.5 {modelo}", fontsize=11)
        eje.set_xticks(range(len(ORDEN_CELDAS)))
        eje.set_xticklabels([c.replace("no_chile", "no chile").replace("_", "\n")
                             for c in ORDEN_CELDAS], fontsize=9)
        eje.set_ylim(0, 1)
        eje.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
    ejes[0][0].set_ylabel("Exact Match")
    ejes[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def cargar_trazas(split: str) -> list[dict]:
    trazas = []
    for ruta in sorted(RESULTADOS.glob(f"trazas_*_{split}.jsonl")):
        with open(ruta, encoding="utf-8") as f:
            trazas += [json.loads(l) for l in f if l.strip()]
    return [puntuar(t) for t in trazas]


def cargar_preguntas(split: str) -> dict:
    ruta = RAIZ / "data" / ("prueba.jsonl" if split == "prueba" else "dataset.jsonl")
    with open(ruta, encoding="utf-8") as f:
        return {q["id"]: q for q in map(json.loads, filter(str.strip, f))}


def main(split: str = "test"):
    trazas = cargar_trazas(split)
    if not trazas:
        sys.exit(f"No hay trazas en {RESULTADOS} para el split {split}")
    preguntas = cargar_preguntas(split)

    tab = tabla(trazas)
    gan = pd.concat([ganancia(trazas, "react", "directa"),
                     ganancia(trazas, "react", "rag"),
                     ganancia(trazas, "rag", "directa")], ignore_index=True)
    err = errores(trazas, preguntas)

    tab.to_csv(RESULTADOS / f"tabla_{split}.csv", index=False)
    gan.to_csv(RESULTADOS / f"ganancia_{split}.csv", index=False)
    err.to_csv(RESULTADOS / f"errores_{split}.csv", index=False)
    grafico(tab, RESULTADOS / f"em_{split}.png")

    pd.set_option("display.width", 160)
    print("EM por celda (filas: modelo, condición; columnas: celda):")
    celdas = tab[tab["agrupacion"] == "celda"]
    print(celdas.pivot_table(index=["modelo", "condicion"], columns="grupo",
                             values="em").reindex(columns=ORDEN_CELDAS).round(2))
    print("\nGanancia de EM por origen y popularidad:")
    print(gan[gan["agrupacion"] != "celda"].round(2).to_string(index=False))
    if len(err):
        print("\nErrores (rag y react):")
        print(err.groupby(["modelo", "condicion", "tipo"]).size().unstack(fill_value=0))
    print(f"\nArchivos escritos en {RESULTADOS}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
