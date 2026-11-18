# PIA · Hiper-heurística de selección para TSP

Osvaldo González Martínez · 2043042 · LCC, FCFM, UANL
Oliver Alexander Vega Casas 1996963
Carlos Eduardo Barrios Barron 2065883
Luis Antonio Diaz Chavez 2086172


El programa ejecuta tres métodos de TSP sobre las 500 instancias provistas y entrena un árbol para recomendar un método a partir de las características de la instancia.

## Ejecutar

Con Python 3.12, abre una terminal dentro de la carpeta `pia` y ejecuta:

```powershell
py -m pip install -r requirements.txt
py generar_etiquetas.py
py entrenar_evaluar.py
```

En Linux o macOS, usa `python3` en lugar de `py`. El dataset de entrada está en `data/`. Las 351 filas de entrenamiento y las 149 de prueba conservan la partición del curso.

## Archivos

| Archivo | Contenido |
|---|---|
| `algoritmos_tsp.py` | Funciones derivadas de la Tarea 2 del [repositorio original](https://github.com/OsvaldoGzz/experimentos-algoritmos/blob/eb64d81ed7405e5da7b515ef16e2d11930dbaf65/codigo/experimentos.py). |
| `generar_etiquetas.py` | Corre los tres métodos y guarda los costos y el ganador. |
| `entrenar_evaluar.py` | Entrena con validación cruzada sobre train y evalúa en test. |
| `resultados/train_etiquetado.csv` y `resultados/test_etiquetado.csv` | Copias etiquetadas de los conjuntos del curso. |
| `resultados/predicciones_test.csv` | Predicciones y regret de cada instancia, para el árbol y la línea base. |
| `resultados/metricas.json` | Balance, hiperparámetros, matrices y comparación de desempeño. |
| `resultados/importancia.csv` | Importancia por permutación de las seis características. |
| `resultados/ganadores_por_grupo.csv` | Conteo por tamaño y distribución. |

## Cómo se compara

- **2-opt** usa primera mejora desde un tour aleatorio. **SA + 2-opt** comparte el tour inicial, propone movimientos 2-opt durante el recocido y devuelve la mejor ruta visitada. **MST + 2-opt** construye el árbol con Prim, genera un tour por recorrido en profundidad y aplica 2-opt.
- Se conserva el recocido exponencial de la Tarea 2 con 3000 movimientos, temperatura inicial 30 y final 0.01. Para conservar la escala de esa tarea se multiplican las coordenadas por 100 durante la corrida y se dividen los costos finales por 100. Las temperaturas equivalentes en la escala de entrada son 0.30 y 0.0001.
- Cada instancia tiene una corrida por algoritmo. La semilla del tour inicial es 2043042 + número de instancia. SA suma 1000 a esa semilla. Costos a una distancia de hasta 10⁻¹⁰ del mínimo se tratan como empate. Se asigna la etiqueta en el orden 2-opt, SA + 2-opt, MST + 2-opt.
- El baseline predice la clase más frecuente de train. El regret es el costo del algoritmo elegido menos el menor de los tres costos medidos para la instancia. Una predicción puede diferir de la etiqueta y obtener regret cero por empate.
