"""Funciones de TSP adaptadas de codigo/experimentos.py de la Tarea 2."""

import math
import random
import time

SEMILLA = 2043042
T_INICIAL = 30.0
T_FINAL = 0.01
ITERACIONES = 3000


def costo(ruta, d):
    return sum(d[ruta[i]][ruta[(i+1)%len(ruta)]] for i in range(len(ruta)))


def pares(n):
    # Mantener fija la primera ciudad elimina movimientos redundantes.
    return [(i,j) for i in range(1,n-1) for j in range(i+1,n)
            if not (i==1 and j==n-1)]


def cambio(ruta, i, j):
    return ruta[:i] + ruta[i:j+1][::-1] + ruta[j+1:]


def recocido(inicial, d, esquema, semilla, iteraciones=ITERACIONES):
    rng = random.Random(semilla)
    ruta = inicial[:]; actual = costo(ruta,d)
    mejor, mejor_costo = ruta[:], actual
    movimientos = pares(len(ruta))
    historia = [mejor_costo]
    # Mismos extremos y presupuesto para comparar solo el enfriamiento.
    t0, tf = T_INICIAL, T_FINAL
    inicio = time.perf_counter()
    for k in range(iteraciones):
        progreso = k/(iteraciones-1)
        if esquema == 'Lineal':
            temperatura = t0+(tf-t0)*progreso
        else:
            temperatura = t0*(tf/t0)**progreso
        i,j = rng.choice(movimientos)
        nueva = cambio(ruta,i,j)
        valor = costo(nueva,d)
        delta = valor-actual
        if delta <= 0 or rng.random() < math.exp(-delta/temperatura):
            ruta, actual = nueva, valor
            if actual < mejor_costo:
                mejor, mejor_costo = ruta[:], actual
        historia.append(mejor_costo)
    return mejor, historia, time.perf_counter()-inicio


def mst_tour(d):
    # Prim sencillo; después recorrido en profundidad y retorno al inicio.
    n = len(d); visitados = {0}; arbol = [[] for _ in range(n)]
    while len(visitados) < n:
        _,a,b = min((d[a][b],a,b) for a in visitados for b in range(n) if b not in visitados)
        arbol[a].append(b); arbol[b].append(a); visitados.add(b)
    ruta = []
    def recorrer(nodo, padre):
        ruta.append(nodo)
        for hijo in arbol[nodo]:
            if hijo != padre: recorrer(hijo,nodo)
    recorrer(0,-1)
    return ruta



def dos_opt(ruta, d):
    """Primera mejora, con descarte rápido mediante las dos aristas que cambian."""
    ruta = ruta[:]
    actual = costo(ruta, d)
    movimientos = pares(len(ruta))
    mejoras = 0
    while True:
        for i, j in movimientos:
            a, b = ruta[i - 1], ruta[i]
            c, e = ruta[j], ruta[(j + 1) % len(ruta)]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            # Conserva el criterio original cuando el delta está cerca del umbral.
            if delta >= 1e-8:
                continue
            nueva = cambio(ruta, i, j)
            valor = costo(nueva, d)
            if valor < actual - 1e-10:
                ruta, actual = nueva, valor
                mejoras += 1
                break
        else:
            return ruta, mejoras
