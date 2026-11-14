"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 4 - REGRESIÓN LINEAL MÚLTIPLE HISTÓRICA

Utiliza las 41 bases:
2016 T2 -> 2026 T2.

Variable objetivo:
- P47T original, SIN logaritmos ni transformación.

Predictores:
- período
- aglomerado
- sexo
- edad
- nivel educativo
- estado laboral
- categoría ocupacional
- rama de actividad
- horas trabajadas

Validación:
- 5 folds
- hogares completos quedan dentro del mismo fold
- si un hogar aparece en distintos trimestres, no se reparte
  entre entrenamiento y evaluación.

Se informan:
- R² global fuera de muestra
- MAE
- RMSE
- R² de cada trimestre
- cantidad de trimestres con R² > 0,30

Todavía NO imputa ingresos faltantes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa1 import (
    RESULTADOS,
    buscar_archivos,
)

from analisis_eph_objetivo2_etapa3_corregido import (
    clasificar_rama,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_FOLDS = 5
SEMILLA = 42


COLUMNAS = [
    "CODUSU",
    "NRO_HOGAR",
    "COMPONENTE",

    "AGLOMERADO",
    "H15",

    "CH04",
    "CH06",
    "NIVEL_ED",

    "ESTADO",
    "CAT_OCUP",

    "PP04B_COD",
    "PP3E_TOT",

    "P47T",
]


MAPA_EDUCACION = {
    7: "Sin instrucción",

    1: "Primaria incompleta",

    2: (
        "Primario completo o "
        "secundario incompleto"
    ),

    3: (
        "Primario completo o "
        "secundario incompleto"
    ),

    4: (
        "Secundario completo o "
        "superior incompleto"
    ),

    5: (
        "Secundario completo o "
        "superior incompleto"
    ),

    6: "Superior completo",

    9: "No sabe / no responde",
}


# ============================================================
# 1. LECTURA DE LAS 41 BASES
# ============================================================

def leer_base(
    ruta,
    anio,
    trimestre,
):

    df = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=COLUMNAS,
        dtype={
            "CODUSU": "string",
            "PP04B_COD": "string",
        },
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    for columna in df.columns:

        if columna not in {
            "CODUSU",
            "PP04B_COD",
        }:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    # Dos aglomerados + entrevista realizada.
    df = df.loc[
        df["AGLOMERADO"].isin(
            [32, 33]
        )
        & df["H15"].eq(1)
    ].copy()

    df["anio"] = anio
    df["trimestre"] = trimestre

    df["periodo"] = (
        str(anio)
        + "-T"
        + str(trimestre)
    )

    return df


def cargar_historia():

    partes = []

    archivos = buscar_archivos()

    print(
        f"Se encontraron "
        f"{len(archivos)} bases."
    )

    for (
        anio,
        trimestre,
        ruta,
    ) in archivos:

        print(
            f"Procesando "
            f"{anio} T{trimestre}: "
            f"{ruta.name}"
        )

        parte = leer_base(
            ruta,
            anio,
            trimestre,
        )

        partes.append(
            parte
        )

    datos = pd.concat(
        partes,
        ignore_index=True,
    )

    print(
        f"\nRegistros acumulados "
        f"en los dos aglomerados: "
        f"{len(datos):,}"
    )

    return datos


# ============================================================
# 2. PREPARACIÓN
# ============================================================

def preparar_variables(
    df,
):

    datos = df.copy()

    ocupados = (
        datos["ESTADO"] == 1
    )

    # --------------------------------------------------------
    # Edad
    # --------------------------------------------------------

    datos["edad_modelo"] = (
        datos["CH06"]
        .replace(-1, 0)
        .astype(float)
    )

    # --------------------------------------------------------
    # Educación
    # --------------------------------------------------------

    datos[
        "educacion_modelo"
    ] = (
        datos["NIVEL_ED"]
        .map(
            MAPA_EDUCACION
        )
    )

    if (
        datos[
            "educacion_modelo"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Hay códigos NIVEL_ED "
            "no contemplados."
        )

    # --------------------------------------------------------
    # Categoría ocupacional
    # --------------------------------------------------------

    # Referencia: categoría 3.
    # Para no ocupados usamos la referencia:
    # ESTADO ya identifica que no trabajan.
    datos[
        "cat_ocup_modelo"
    ] = "3"

    datos.loc[
        ocupados,
        "cat_ocup_modelo",
    ] = (
        datos.loc[
            ocupados,
            "CAT_OCUP",
        ]
        .astype("Int64")
        .astype("string")
    )

    # --------------------------------------------------------
    # Rama
    # --------------------------------------------------------

    # Comercio = referencia.
    datos[
        "rama_modelo"
    ] = "Comercio"

    datos.loc[
        ocupados,
        "rama_modelo",
    ] = (
        datos.loc[
            ocupados,
            "PP04B_COD",
        ]
        .map(
            clasificar_rama
        )
    )

    # --------------------------------------------------------
    # Horas
    # --------------------------------------------------------

    horas = (
        datos["PP3E_TOT"]
    )

    horas_validas = (
        ocupados
        & horas.notna()
        & horas.between(
            0,
            168,
        )
    )

    horas_especiales = (
        ocupados
        & (
            horas.isna()
            | horas.gt(168)
        )
    )

    datos[
        "horas_modelo"
    ] = 0.0

    datos.loc[
        horas_validas,
        "horas_modelo",
    ] = (
        horas.loc[
            horas_validas
        ]
        .astype(float)
    )

    datos[
        "horas_especiales"
    ] = (
        horas_especiales
        .astype(float)
    )

    # --------------------------------------------------------
    # P47T
    # --------------------------------------------------------

    ingreso = datos["P47T"]

    invalidos = (
        ingreso.notna()
        & ingreso.lt(0)
        & ingreso.ne(-9)
    )

    if invalidos.any():

        raise ValueError(
            "P47T tiene códigos "
            "negativos inesperados."
        )

    datos[
        "ingreso_conocido"
    ] = (
        ingreso.notna()
        & ingreso.ge(0)
    )

    datos[
        "ingreso_no_responde"
    ] = (
        ingreso.isna()
        | ingreso.eq(-9)
    )

    return datos


# ============================================================
# 3. IDENTIFICADOR DEL HOGAR
# ============================================================

def construir_id_hogar(
    df,
):

    return (
        df["CODUSU"]
        .astype("string")
        + "_"
        + df["NRO_HOGAR"]
        .astype("Int64")
        .astype("string")
    )


# ============================================================
# 4. FOLDS
# ============================================================

def crear_folds(
    conocidos,
):

    hogares = (
        construir_id_hogar(
            conocidos
        )
        .drop_duplicates()
        .to_numpy()
    )

    rng = (
        np.random.default_rng(
            SEMILLA
        )
    )

    rng.shuffle(
        hogares
    )

    grupos = (
        np.array_split(
            hogares,
            N_FOLDS,
        )
    )

    mapa = {}

    for fold, grupo in enumerate(
        grupos,
        start=1,
    ):

        for hogar in grupo:

            mapa[hogar] = fold

    return mapa


# ============================================================
# 5. CODIFICACIÓN
# ============================================================

def texto(
    serie,
):

    return (
        serie
        .astype("string")
        .fillna(
            "Sin dato"
        )
    )


def agregar_dummies(
    x_train,
    x_test,
    train,
    test,
    variable,
    referencia,
):

    train = texto(
        train
    )

    test = texto(
        test
    )

    categorias = sorted(
        train.unique()
    )

    referencia = str(
        referencia
    )

    if referencia not in categorias:

        referencia = (
            train
            .value_counts()
            .index[0]
        )

    for categoria in categorias:

        if (
            categoria
            == referencia
        ):
            continue

        nombre = (
            f"{variable}="
            f"{categoria}"
        )

        x_train[
            nombre
        ] = (
            train.eq(
                categoria
            )
            .astype(float)
        )

        x_test[
            nombre
        ] = (
            test.eq(
                categoria
            )
            .astype(float)
        )


def construir_matrices(
    train,
    test,
):

    x_train = pd.DataFrame(
        index=train.index
    )

    x_test = pd.DataFrame(
        index=test.index
    )

    # --------------------------------------------------------
    # Numéricas
    # --------------------------------------------------------

    numericas = [
        "edad_modelo",
        "horas_modelo",
        "horas_especiales",
    ]

    for variable in numericas:

        tr = pd.to_numeric(
            train[variable],
            errors="coerce",
        )

        te = pd.to_numeric(
            test[variable],
            errors="coerce",
        )

        if tr.isna().any():

            mediana = (
                tr.median()
            )

            tr = tr.fillna(
                mediana
            )

            te = te.fillna(
                mediana
            )

        x_train[
            variable
        ] = (
            tr.astype(float)
        )

        x_test[
            variable
        ] = (
            te.astype(float)
        )

    # --------------------------------------------------------
    # Categóricas
    # --------------------------------------------------------

    categoricas = {
        # Período de referencia:
        # segundo trimestre 2026.
        "periodo":
            "2026-T2",

        "AGLOMERADO":
            "32",

        "CH04":
            "1",

        "educacion_modelo":
            "Superior completo",

        "ESTADO":
            "1",

        "cat_ocup_modelo":
            "3",

        "rama_modelo":
            "Comercio",
    }

    for (
        variable,
        referencia,
    ) in categoricas.items():

        agregar_dummies(
            x_train=x_train,
            x_test=x_test,

            train=train[
                variable
            ],

            test=test[
                variable
            ],

            variable=variable,
            referencia=referencia,
        )

    x_test = x_test.reindex(
        columns=x_train.columns,
        fill_value=0.0,
    )

    return (
        x_train,
        x_test,
    )


# ============================================================
# 6. REGRESIÓN
# ============================================================

def matriz_intercepto(
    x,
):

    return np.column_stack([
        np.ones(
            len(x)
        ),

        x.to_numpy(
            dtype=float
        ),
    ])


def ajustar(
    x,
    y,
):

    matriz = (
        matriz_intercepto(
            x
        )
    )

    (
        coeficientes,
        _,
        rango,
        _,
    ) = np.linalg.lstsq(
        matriz,
        y.to_numpy(
            dtype=float
        ),
        rcond=None,
    )

    return (
        coeficientes,
        rango,
    )


def predecir(
    x,
    coeficientes,
):

    return (
        matriz_intercepto(
            x
        )
        @ coeficientes
    )


# ============================================================
# 7. MÉTRICAS
# ============================================================

def r2(
    y,
    pred,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    if len(y) < 2:
        return np.nan

    total = np.sum(
        (
            y
            - y.mean()
        ) ** 2
    )

    if total <= 0:
        return np.nan

    residuos = np.sum(
        (
            y
            - pred
        ) ** 2
    )

    return float(
        1
        - residuos / total
    )


def metricas(
    y,
    pred,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    error = (
        y - pred
    )

    mse = np.mean(
        error ** 2
    )

    return {
        "r2":
            r2(
                y,
                pred,
            ),

        "mae":
            float(
                np.mean(
                    np.abs(
                        error
                    )
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mse
                )
            ),

        "pred_negativas_pct":
            float(
                100
                * np.mean(
                    pred < 0
                )
            ),
    }


# ============================================================
# 8. VALIDACIÓN CRUZADA
# ============================================================

def validar(
    datos,
):

    conocidos = datos.loc[
        datos[
            "ingreso_conocido"
        ]
    ].copy()

    conocidos[
        "_hogar"
    ] = construir_id_hogar(
        conocidos
    )

    mapa = crear_folds(
        conocidos
    )

    conocidos[
        "_fold"
    ] = (
        conocidos[
            "_hogar"
        ]
        .map(
            mapa
        )
    )

    resultados = []
    predicciones = []

    for fold in range(
        1,
        N_FOLDS + 1,
    ):

        print(
            "\n"
            f"Procesando fold "
            f"{fold}/{N_FOLDS}"
        )

        train = conocidos.loc[
            conocidos[
                "_fold"
            ] != fold
        ].copy()

        test = conocidos.loc[
            conocidos[
                "_fold"
            ] == fold
        ].copy()

        (
            x_train,
            x_test,
        ) = construir_matrices(
            train,
            test,
        )

        y_train = (
            train["P47T"]
            .astype(float)
        )

        y_test = (
            test["P47T"]
            .astype(float)
        )

        coef, rango = ajustar(
            x_train,
            y_train,
        )

        pred = predecir(
            x_test,
            coef,
        )

        met = metricas(
            y_test,
            pred,
        )

        resultados.append({
            "fold":
                fold,

            "n_train":
                len(train),

            "n_test":
                len(test),

            "n_predictores":
                x_train.shape[1],

            "rango_completo":
                (
                    rango
                    == x_train.shape[1]
                    + 1
                ),

            "r2":
                met["r2"],

            "mae":
                met["mae"],

            "rmse":
                met["rmse"],

            "pred_negativas_pct":
                met[
                    "pred_negativas_pct"
                ],
        })

        salida = test[
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

        salida[
            "fold"
        ] = fold

        salida[
            "prediccion"
        ] = pred

        predicciones.append(
            salida
        )

        print(
            f"R²={met['r2']:.4f} | "
            f"MAE=${met['mae']:,.0f} | "
            f"RMSE=${met['rmse']:,.0f}"
        )

    return (
        pd.DataFrame(
            resultados
        ),

        pd.concat(
            predicciones,
            ignore_index=True,
        ),
    )


# ============================================================
# 9. MÉTRICAS POR TRIMESTRE
# ============================================================

def calcular_por_periodo(
    predicciones,
):

    filas = []

    for (
        periodo,
        parte,
    ) in predicciones.groupby(
        "periodo"
    ):

        met = metricas(
            parte[
                "P47T"
            ].to_numpy(
                dtype=float
            ),

            parte[
                "prediccion"
            ].to_numpy(
                dtype=float
            ),
        )

        filas.append({
            "periodo":
                periodo,

            "n":
                len(parte),

            "r2":
                met["r2"],

            "mae":
                met["mae"],

            "rmse":
                met["rmse"],
        })

    return pd.DataFrame(
        filas
    )


# ============================================================
# 10. AJUSTE SOBRE TODA LA BASE
# ============================================================

def ajuste_completo(
    datos,
):

    conocidos = datos.loc[
        datos[
            "ingreso_conocido"
        ]
    ].copy()

    (
        x,
        _,
    ) = construir_matrices(
        conocidos,
        conocidos,
    )

    y = (
        conocidos[
            "P47T"
        ]
        .astype(float)
    )

    coef, rango = ajustar(
        x,
        y,
    )

    pred = predecir(
        x,
        coef,
    )

    r2_completo = r2(
        y,
        pred,
    )

    n = len(y)
    p = x.shape[1]

    r2_ajustado = (
        1
        - (
            (1 - r2_completo)
            * (n - 1)
            / (n - p - 1)
        )
    )

    coeficientes = pd.DataFrame({
        "variable":
            ["Intercepto"]
            + list(
                x.columns
            ),

        "coeficiente":
            coef,
    })

    return {
        "n":
            n,

        "n_predictores":
            p,

        "rango_completo":
            (
                rango
                == p + 1
            ),

        "r2_ajuste_completo":
            r2_completo,

        "r2_ajustado":
            r2_ajustado,
    }, coeficientes


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTADOS.mkdir(
        exist_ok=True
    )

    datos = cargar_historia()

    datos = preparar_variables(
        datos
    )

    print(
        "\n"
        "Ingresos conocidos:",
        f"{datos['ingreso_conocido'].sum():,}",
    )

    print(
        "No respuestas:",
        f"{datos['ingreso_no_responde'].sum():,}",
    )

    # --------------------------------------------------------
    # Validación cruzada
    # --------------------------------------------------------

    (
        folds,
        predicciones,
    ) = validar(
        datos
    )

    # --------------------------------------------------------
    # R² conjunto fuera de muestra
    # --------------------------------------------------------

    met_oof = metricas(
        predicciones[
            "P47T"
        ].to_numpy(
            dtype=float
        ),

        predicciones[
            "prediccion"
        ].to_numpy(
            dtype=float
        ),
    )

    # --------------------------------------------------------
    # Por trimestre
    # --------------------------------------------------------

    por_periodo = (
        calcular_por_periodo(
            predicciones
        )
    )

    # --------------------------------------------------------
    # Ajuste completo
    # --------------------------------------------------------

    (
        completo,
        coeficientes,
    ) = ajuste_completo(
        datos
    )

    resumen = pd.DataFrame([
        {
            **completo,

            "r2_cv_promedio":
                folds[
                    "r2"
                ].mean(),

            "r2_cv_min":
                folds[
                    "r2"
                ].min(),

            "r2_cv_max":
                folds[
                    "r2"
                ].max(),

            "r2_oof_conjunto":
                met_oof[
                    "r2"
                ],

            "mae_oof":
                met_oof[
                    "mae"
                ],

            "rmse_oof":
                met_oof[
                    "rmse"
                ],

            "pred_negativas_pct":
                met_oof[
                    "pred_negativas_pct"
                ],

            "r2_periodos_promedio":
                por_periodo[
                    "r2"
                ].mean(),

            "r2_periodos_mediana":
                por_periodo[
                    "r2"
                ].median(),

            "r2_periodos_min":
                por_periodo[
                    "r2"
                ].min(),

            "r2_periodos_max":
                por_periodo[
                    "r2"
                ].max(),

            "periodos_r2_mayor_030":
                int(
                    (
                        por_periodo[
                            "r2"
                        ] > 0.30
                    )
                    .sum()
                ),

            "cantidad_periodos":
                len(
                    por_periodo
                ),
        }
    ])

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    resumen.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_resumen.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    folds.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_folds.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    por_periodo.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_por_periodo.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    coeficientes.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_coeficientes.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    predicciones.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_predicciones.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RESULTADO HISTÓRICO "
        "2016-2026"
    )

    print(
        "=" * 75
    )

    print(
        resumen.to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.4f}",
        )
    )

    print(
        "\nR² por trimestre:"
    )

    print(
        por_periodo[
            [
                "periodo",
                "n",
                "r2",
                "mae",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.4f}",
        )
    )

    print(
        "\nCRITERIO PRINCIPAL:"
    )

    valor = float(
        resumen[
            "r2_oof_conjunto"
        ].iloc[0]
    )

    print(
        f"R² OOF conjunto = "
        f"{valor:.4f}"
    )

    if valor > 0.30:

        print(
            "SUPERA R² = 0,30."
        )

    else:

        print(
            "NO SUPERA R² = 0,30."
        )


if __name__ == "__main__":
    main()