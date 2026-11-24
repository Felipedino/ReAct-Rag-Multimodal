"""Corre las condiciones con UN modelo sobre un split y guarda las trazas.

Uso: python correr.py <modelo> <split> [condiciones...]
    python correr.py 7B test                 # las 3 condiciones
    python correr.py 1.5B dev directa react  # solo algunas
    python correr.py 1.5B prueba             # 5 preguntas de data/prueba.jsonl

Un proceso por modelo: al terminar el proceso se libera la GPU (vLLM no la
libera bien dentro de un mismo notebook). El notebook lo llama con
`!python correr.py ...`.

Salida: resultados/trazas_<modelo>_<split>.jsonl. Si el archivo ya existe,
se reemplazan solo las trazas de las condiciones que se vuelven a correr.
"""

import json
import sys
import time
from pathlib import Path

import agents
import llm
import wiki

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"


def cargar_preguntas(split: str) -> list[dict]:
    """Preguntas del split en data/dataset.jsonl ("prueba" usa data/prueba.jsonl)."""
    if split == "prueba":
        ruta = RAIZ / "data" / "prueba.jsonl"
    else:
        ruta = RAIZ / "data" / "dataset.jsonl"
    with open(ruta, encoding="utf-8") as f:
        filas = [json.loads(l) for l in f if l.strip()]
    return [q for q in filas if split == "prueba" or q["split"] == split]


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in llm.MODELOS:
        print(f"Uso: python correr.py <{'|'.join(llm.MODELOS)}> <split> [condiciones...]")
        sys.exit(1)
    nombre, split = sys.argv[1], sys.argv[2]
    condiciones = sys.argv[3:] or list(agents.CONDICIONES)
    for c in condiciones:
        if c not in agents.CONDICIONES:
            sys.exit(f"Condición desconocida: {c}. Opciones: {list(agents.CONDICIONES)}")

    preguntas = cargar_preguntas(split)
    print(f"{len(preguntas)} preguntas ({split}); modelo {nombre}; condiciones {condiciones}")

    inicio = time.time()
    modelo = llm.cargar(nombre)
    print(f"Modelo cargado en {time.time() - inicio:.0f} s")

    # Se conservan las trazas de otras condiciones de una corrida anterior.
    RESULTADOS.mkdir(exist_ok=True)
    salida = RESULTADOS / f"trazas_{nombre}_{split}.jsonl"
    trazas = []
    if salida.exists():
        with open(salida, encoding="utf-8") as f:
            trazas = [t for t in map(json.loads, filter(str.strip, f))
                      if t["condicion"] not in condiciones]

    resumen = []
    for c in condiciones:
        t0, tok0, api0 = time.time(), llm.tokens_generados, wiki.llamadas_api
        nuevas = agents.CONDICIONES[c](modelo, preguntas)
        for t in nuevas:
            t["modelo"] = nombre
        trazas += nuevas
        resumen.append((c, time.time() - t0, llm.tokens_generados - tok0,
                        wiki.llamadas_api - api0))
        # Se guarda después de cada condición, por si la sesión se corta.
        with open(salida, "w", encoding="utf-8") as f:
            for t in trazas:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nTrazas en {salida}")
    print(f"{'condición':10} {'segundos':>9} {'tokens gen.':>12} {'llamadas API':>13}")
    for c, seg, tok, api in resumen:
        print(f"{c:10} {seg:9.1f} {tok:12d} {api:13d}")
    print(f"Total: {time.time() - inicio:.0f} s, {llm.tokens_generados} tokens generados, "
          f"{wiki.llamadas_api} llamadas reales a Wikipedia")


if __name__ == "__main__":
    main()
