"""Ejecuta los tres algoritmos de la Tarea 2 sobre las instancias del curso."""

import csv
import math
import random
from pathlib import Path

from algoritmos_tsp import SEMILLA, costo, dos_opt, mst_tour, recocido

BASE = Path(__file__).resolve().parent
DATOS = BASE / "data"
RESULTADOS = BASE / "resultados"
ALGORITMOS = ("2opt_hill_climbing", "sa_2opt", "mst_2opt")
COLUMNAS_COSTO = tuple(f"cost_{nombre}" for nombre in ALGORITMOS)
ESCALA = 100.0
TOLERANCIA = 1e-10


def medir(fila):
    instance_id = fila["instance_id"]
    with (DATOS / "instances" / f"{instance_id}.csv").open(newline="", encoding="utf-8") as archivo:
        puntos = [(float(p["x"]) * ESCALA, float(p["y"]) * ESCALA)
                  for p in csv.DictReader(archivo)]
    n = len(puntos)
    if n != int(fila["n"]) or n < 4 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in puntos):
        raise ValueError(f"Coordenadas inválidas en {instance_id}")
    d = [[math.dist(a, b) for b in puntos] for a in puntos]
    semilla = SEMILLA + int(instance_id.split("_")[-1])
    inicial = random.Random(semilla).sample(range(n), n)

    local, _ = dos_opt(inicial, d)
    sa, _, _ = recocido(inicial, d, "Exponencial", semilla + 1000)
    mst, _ = dos_opt(mst_tour(d), d)
    costos = [costo(ruta, d) / ESCALA for ruta in (local, sa, mst)]
    minimo = min(costos)
    ganador = next(nombre for nombre, valor in zip(ALGORITMOS, costos)
                   if valor - minimo <= TOLERANCIA)
    return {**fila, **dict(zip(COLUMNAS_COSTO, costos)), "best_algorithm": ganador}


def main():
    RESULTADOS.mkdir(exist_ok=True)
    for grupo in ("train", "test"):
        with (DATOS / f"{grupo}.csv").open(newline="", encoding="utf-8") as archivo:
            lector = csv.DictReader(archivo)
            columnas = [*lector.fieldnames, *COLUMNAS_COSTO, "best_algorithm"]
            filas = [medir(fila) for fila in lector]
        with (RESULTADOS / f"{grupo}_etiquetado.csv").open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"{grupo}: {len(filas)} instancias etiquetadas", flush=True)


if __name__ == "__main__":
    main()
