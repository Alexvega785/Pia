"""Entrena el selector y evalúa las predicciones sobre la prueba del curso."""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from generar_etiquetas import ALGORITMOS, COLUMNAS_COSTO, RESULTADOS, SEMILLA, TOLERANCIA

OBJETIVO = "best_algorithm"
NUMERICAS = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
CARACTERISTICAS = ["n", "distribution", *NUMERICAS[1:]]
PARAMETROS = {
    "arbol__max_depth": [2, 3, 4, 5, 7, None],
    "arbol__min_samples_leaf": [3, 5, 10, 15],
    "arbol__class_weight": [None, "balanced"],
}


def balance(filas, grupos=None):
    etiquetas = filas[OBJETIVO]
    if grupos:
        return pd.crosstab([filas[c] for c in grupos], etiquetas).reindex(
            columns=ALGORITMOS, fill_value=0)
    return etiquetas.value_counts().reindex(ALGORITMOS, fill_value=0)


def resultados(y_real, y_pred, regret):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "aciertos": int(np.sum(np.asarray(y_real) == y_pred)),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "matriz": confusion_matrix(y_real, y_pred, labels=ALGORITMOS).tolist(),
        "regret_promedio": float(regret.mean()),
        "regret_mediano": float(np.median(regret)),
        "regret_maximo": float(regret.max()),
        "costo_minimo": int(np.sum(regret <= TOLERANCIA)),
    }


def main():
    train = pd.read_csv(RESULTADOS / "train_etiquetado.csv")
    test = pd.read_csv(RESULTADOS / "test_etiquetado.csv")
    todos = pd.concat([train, test], ignore_index=True)
    if todos["instance_id"].duplicated().any():
        raise ValueError("Hay identificadores repetidos entre las instancias etiquetadas.")
    x_train, y_train = train[CARACTERISTICAS], train[OBJETIVO]
    x_test, y_test = test[CARACTERISTICAS], test[OBJETIVO]

    arbol = Pipeline([
        ("preparar", ColumnTransformer([
            ("distribucion", OneHotEncoder(handle_unknown="ignore"), ["distribution"]),
            ("numericas", "passthrough", NUMERICAS),
        ])),
        ("arbol", DecisionTreeClassifier(criterion="gini", random_state=SEMILLA)),
    ])
    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(arbol, PARAMETROS, cv=validacion, scoring="accuracy")
    busqueda.fit(x_train, y_train)
    pred = busqueda.predict(x_test)

    conteos = pd.DataFrame({
        "total": balance(todos), "train": balance(train), "test": balance(test),
    })
    frecuente = conteos["train"].idxmax()
    pred_base = np.repeat(frecuente, len(test))
    costos = test[list(COLUMNAS_COSTO)].to_numpy()
    mejor_costo = costos.min(axis=1)
    renglon = np.arange(len(test))
    indice_pred = [ALGORITMOS.index(etiqueta) for etiqueta in pred]
    costo_pred = costos[renglon, indice_pred]
    costo_baseline = costos[:, ALGORITMOS.index(frecuente)]
    regret = costo_pred - mejor_costo
    regret_base = costo_baseline - mejor_costo
    empates = (costos - mejor_costo[:, None] <= TOLERANCIA).sum(axis=1)

    importancia = permutation_importance(
        busqueda, x_test, y_test, scoring="accuracy", n_repeats=50, random_state=SEMILLA,
    )
    importancias = pd.DataFrame({
        "caracteristica": CARACTERISTICAS,
        "media": importancia.importances_mean,
        "desviacion": importancia.importances_std,
    }).sort_values("media", ascending=False)
    importancias.to_csv(RESULTADOS / "importancia.csv", index=False)

    balance(todos, ["n", "distribution"]).to_csv(RESULTADOS / "ganadores_por_grupo.csv")
    predicciones = test[["instance_id", "n", "distribution", OBJETIVO, *COLUMNAS_COSTO]].copy()
    predicciones = predicciones.assign(
        prediccion=pred, baseline=pred_base, costo_minimo=mejor_costo,
        costo_predicho=costo_pred, costo_baseline=costo_baseline, regret=regret,
        regret_baseline=regret_base, algoritmos_en_minimo=empates,
    )
    predicciones.to_csv(RESULTADOS / "predicciones_test.csv", index=False, float_format="%.16g")

    costos_totales = todos[list(COLUMNAS_COSTO)].to_numpy()
    minimos_totales = costos_totales.min(axis=1)
    cercanos = costos_totales - minimos_totales[:, None] <= TOLERANCIA
    local = costos[:, 0]
    supera_a_local = {}
    for index, nombre in enumerate(ALGORITMOS[1:], start=1):
        mejora = costos[:, index] < local - TOLERANCIA
        supera_a_local[nombre] = {
            "casos": int(mejora.sum()),
            "modelo_elige": int((mejora & (pred == nombre)).sum()),
        }

    metricas = {
        "algoritmos": ALGORITMOS,
        "balance": conteos.to_dict(),
        "total_empates": int((cercanos.sum(axis=1) > 1).sum()),
        "empates_test": int((empates > 1).sum()),
        "minimo_unico": dict(zip(ALGORITMOS, cercanos[cercanos.sum(axis=1) == 1].sum(axis=0).tolist())),
        "baseline_clase": frecuente,
        "hiperparametros": {k.split("__")[1]: v for k, v in busqueda.best_params_.items()},
        "accuracy_cv": busqueda.best_score_,
        "profundidad": int(busqueda.best_estimator_.named_steps["arbol"].get_depth()),
        "hojas": int(busqueda.best_estimator_.named_steps["arbol"].get_n_leaves()),
        "modelo": resultados(y_test, pred, regret),
        "baseline": resultados(y_test, pred_base, regret_base),
        "supera_a_local": supera_a_local,
    }
    salida = json.dumps(metricas, indent=2, ensure_ascii=False)
    (RESULTADOS / "metricas.json").write_text(salida + "\n", encoding="utf-8")
    print(salida)


if __name__ == "__main__":
    main()
