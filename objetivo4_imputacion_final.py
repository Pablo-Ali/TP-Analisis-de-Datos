"""OBJETIVO 4 - IMPUTACIÓN FINAL DE P47T.

Utiliza la regresión histórica con IPC ya validada.

IMPORTANTE:
- No modifica los archivos EPH originales.
- Conserva P47T original.
- Agrega:
    P47T_predicho
    P47T_imputado
    P47T_completo
    fue_imputado
    prediccion_negativa_original
- Las predicciones negativas se reemplazan por 0 SOLO
  al momento de generar la imputación final.
- La evaluación del modelo realizada anteriormente se hizo
  sobre las predicciones originales, sin corregirlas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analisis_eph_objetivo2_etapa1 import (
    RESULTADOS,
    GRAFICOS,
)

from objetivo4_regresion_historica_ipc import (
    cargar_historia,
    preparar_variables,
    construir_matrices,
    ajustar,
    predecir,
)


# ============================================================
# AJUSTE FINAL
# ============================================================

def ajustar_modelo_final(datos):

    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    faltantes = datos.loc[
        datos["ingreso_no_responde"]
    ].copy()

    # La función construye las mismas variables usadas
    # durante la validación cruzada.
    x_train, x_faltantes = (
        construir_matrices(
            conocidos,
            faltantes,
        )
    )

    y_train = (
        conocidos["P47T"]
        .astype(float)
    )

    coeficientes, rango = ajustar(
        x_train,
        y_train,
    )

    if rango != x_train.shape[1]:

        raise ValueError(
            "La matriz final no tiene rango completo."
        )

    predicciones = predecir(
        x_faltantes,
        coeficientes,
    )

    return (
        conocidos,
        faltantes,
        predicciones,
        x_train,
        coeficientes,
    )


# ============================================================
# IMPUTACIÓN
# ============================================================

def generar_imputaciones(
    datos,
    faltantes,
    predicciones,
):

    imputados = faltantes[
        [
            "CODUSU",
            "NRO_HOGAR",
            "COMPONENTE",
            "periodo",
            "AGLOMERADO",
            "ESTADO",
            "P47T",
        ]
    ].copy()

    imputados = imputados.rename(
        columns={
            "P47T":
                "P47T_original"
        }
    )

    imputados[
        "P47T_predicho"
    ] = predicciones

    imputados[
        "prediccion_negativa_original"
    ] = (
        imputados[
            "P47T_predicho"
        ] < 0
    )

    # Una renta negativa no tiene sentido
    # para P47T.
    #
    # Esto NO se utilizó para calcular R².
    imputados[
        "P47T_imputado"
    ] = (
        imputados[
            "P47T_predicho"
        ]
        .clip(lower=0)
    )

    imputados[
        "fue_imputado"
    ] = True

    # --------------------------------------------------------
    # Base completa
    # --------------------------------------------------------

    completa = datos[
        [
            "CODUSU",
            "NRO_HOGAR",
            "COMPONENTE",
            "periodo",
            "AGLOMERADO",
            "ESTADO",
            "P47T",
        ]
    ].copy()

    completa = completa.rename(
        columns={
            "P47T":
                "P47T_original"
        }
    )

    claves = [
        "CODUSU",
        "NRO_HOGAR",
        "COMPONENTE",
        "periodo",
    ]

    completa = completa.merge(
        imputados[
            claves
            + [
                "P47T_predicho",
                "P47T_imputado",
                "prediccion_negativa_original",
            ]
        ],
        on=claves,
        how="left",
        validate="one_to_one",
    )

    completa[
        "fue_imputado"
    ] = (
        completa[
            "P47T_original"
        ]
        .isna()
        | completa[
            "P47T_original"
        ]
        .eq(-9)
    )

    completa[
        "P47T_completo"
    ] = (
        completa[
            "P47T_original"
        ]
    )

    mascara = (
        completa[
            "fue_imputado"
        ]
    )

    completa.loc[
        mascara,
        "P47T_completo",
    ] = (
        completa.loc[
            mascara,
            "P47T_imputado",
        ]
    )

    return (
        imputados,
        completa,
    )


# ============================================================
# RESUMEN
# ============================================================

def resumir_imputacion(
    imputados,
):

    filas = []

    for (
        periodo,
        parte,
    ) in imputados.groupby(
        "periodo"
    ):

        filas.append({
            "periodo":
                periodo,

            "n_imputados":
                len(parte),

            "n_predicciones_negativas":
                int(
                    parte[
                        "prediccion_negativa_original"
                    ]
                    .sum()
                ),

            "predicciones_negativas_pct":
                (
                    100
                    * parte[
                        "prediccion_negativa_original"
                    ]
                    .mean()
                ),

            "media_imputada":
                parte[
                    "P47T_imputado"
                ]
                .mean(),

            "mediana_imputada":
                parte[
                    "P47T_imputado"
                ]
                .median(),
        })

    return pd.DataFrame(
        filas
    )


# ============================================================
# GRÁFICOS DE EVALUACIÓN
# ============================================================

def graficar_evaluacion():

    ruta = (
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_folds.csv"
    )

    folds = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # R² por fold
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        folds["fold"].astype(str),
        folds["r2"],
    )

    ax.axhline(
        0.30,
        linestyle="--",
        linewidth=1.5,
        label="Umbral de la consigna (R² = 0,30)",
    )

    ax.axhline(
        0.4812,
        linestyle=":",
        linewidth=1.5,
        label="R² OOF conjunto = 0,481",
    )

    ax.set_xlabel(
        "Fold de validación"
    )

    ax.set_ylabel(
        "R²"
    )

    ax.set_title(
        "Rendimiento fuera de muestra de la regresión"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
        linestyle="--",
    )

    fig.tight_layout()

    destino = (
        GRAFICOS
        / "objetivo4_r2_validacion.png"
    )

    fig.savefig(
        destino,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico guardado: {destino}"
    )


def graficar_observado_predicho():

    ruta = (
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_predicciones.csv"
    )

    tabla = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # Para que el gráfico sea legible:
    # mostramos hasta percentil 99.
    #
    # Esto afecta SOLO la figura.
    # Las métricas se calcularon con todos los casos.
    limite = float(
        tabla["P47T"]
        .quantile(0.99)
    )

    visual = tabla.loc[
        (tabla["P47T"] >= 0)
        & (tabla["P47T"] <= limite)
        & (tabla["prediccion"] >= 0)
        & (tabla["prediccion"] <= limite)
    ].copy()

    # Evitamos dibujar cientos de miles de puntos.
    if len(visual) > 12000:

        visual = visual.sample(
            12000,
            random_state=42,
        )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.scatter(
        visual["P47T"],
        visual["prediccion"],
        alpha=0.20,
        s=10,
    )

    ax.plot(
        [0, limite],
        [0, limite],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "P47T observado"
    )

    ax.set_ylabel(
        "P47T predicho"
    )

    ax.set_title(
        "Ingresos observados y predichos"
    )

    ax.grid(
        alpha=0.25,
        linestyle="--",
    )

    fig.text(
        0.5,
        0.01,
        (
            "Visualización limitada al percentil 99. "
            "Las métricas utilizan todas las observaciones."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0.035, 1, 1]
    )

    destino = (
        GRAFICOS
        / "objetivo4_observado_predicho.png"
    )

    fig.savefig(
        destino,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Gráfico guardado: {destino}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTADOS.mkdir(
        exist_ok=True
    )

    GRAFICOS.mkdir(
        exist_ok=True
    )

    datos = cargar_historia()

    datos = preparar_variables(
        datos
    )

    (
        conocidos,
        faltantes,
        predicciones,
        x_train,
        coeficientes,
    ) = ajustar_modelo_final(
        datos
    )

    (
        imputados,
        completa,
    ) = generar_imputaciones(
        datos,
        faltantes,
        predicciones,
    )

    resumen = resumir_imputacion(
        imputados
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    imputados.to_csv(
        RESULTADOS
        / "objetivo4_p47t_imputados.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    completa.to_csv(
        RESULTADOS
        / "objetivo4_p47t_base_completa.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    resumen.to_csv(
        RESULTADOS
        / "objetivo4_resumen_imputacion.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------

    graficar_evaluacion()
    graficar_observado_predicho()

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "IMPUTACIÓN FINAL"
    )

    print(
        "=" * 70
    )

    print(
        "Ingresos conocidos:",
        f"{len(conocidos):,}",
    )

    print(
        "Ingresos imputados:",
        f"{len(imputados):,}",
    )

    negativos = int(
        imputados[
            "prediccion_negativa_original"
        ]
        .sum()
    )

    print(
        "Predicciones negativas antes de corregir:",
        f"{negativos:,}",
        (
            f"({100 * negativos / len(imputados):.2f}%)"
        ),
    )

    print(
        "\nMediana imputada general:",
        f"${imputados['P47T_imputado'].median():,.0f}",
    )

    print(
        "Media imputada general:",
        f"${imputados['P47T_imputado'].mean():,.0f}",
    )

    print(
        "\nArchivos generados:"
    )

    print(
        " - objetivo4_p47t_imputados.csv"
    )

    print(
        " - objetivo4_p47t_base_completa.csv"
    )

    print(
        " - objetivo4_resumen_imputacion.csv"
    )

    print(
        " - objetivo4_r2_validacion.png"
    )

    print(
        " - objetivo4_observado_predicho.png"
    )


if __name__ == "__main__":
    main()