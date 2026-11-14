"""OBJETIVO 4 - Regresión histórica con escala IPC.

- 41 trimestres: 2016 T2 - 2026 T2.
- P47T permanece ORIGINAL, nominal y sin transformar.
- El IPC se usa para escalar los efectos de los predictores.
- Validación cruzada de 5 folds por hogar.
- Métrica principal: R² conjunto fuera de muestra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS,
    RESULTADOS,
    buscar_archivos,
)

from analisis_eph_objetivo2_etapa3_corregido import (
    clasificar_rama,
    cargar_factores,
)


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

    2: "Primario completo o secundario incompleto",
    3: "Primario completo o secundario incompleto",

    4: "Secundario completo o superior incompleto",
    5: "Secundario completo o superior incompleto",

    6: "Superior completo",

    9: "No sabe / no responde",
}


# ============================================================
# LECTURA
# ============================================================

def leer_base(
    ruta,
    anio,
    trimestre,
    factores,
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

    df = df.loc[
        df["AGLOMERADO"].isin(
            AGLOMERADOS
        )
        & df["H15"].eq(1)
    ].copy()

    periodo = (
        f"{anio}-T{trimestre}"
    )

    df["periodo"] = periodo

    # --------------------------------------------------------
    # Factor IPC validado previamente.
    #
    # ingreso_real_2026 =
    # ingreso_nominal * factor
    #
    # Por lo tanto:
    #
    # ingreso_nominal =
    # ingreso_real / factor
    #
    # escala_precios = 1 / factor
    # --------------------------------------------------------

    factores_fila = []

    for codigo in df[
        "AGLOMERADO"
    ]:

        nombre = (
            AGLOMERADOS[
                int(codigo)
            ]
        )

        clave = (
            periodo,
            nombre,
        )

        if clave not in factores:
            raise ValueError(
                f"Falta factor IPC para {clave}"
            )

        factores_fila.append(
            factores[clave]
        )

    df["factor_ipc"] = (
        factores_fila
    )

    df["escala_precios"] = (
        1.0
        / df["factor_ipc"]
    )

    return df


def cargar_historia():

    factores = cargar_factores()

    archivos = buscar_archivos()

    print(
        f"Se encontraron "
        f"{len(archivos)} bases."
    )

    partes = []

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

        partes.append(
            leer_base(
                ruta,
                anio,
                trimestre,
                factores,
            )
        )

    return pd.concat(
        partes,
        ignore_index=True,
    )


# ============================================================
# VARIABLES
# ============================================================

def preparar_variables(df):

    datos = df.copy()

    ocupados = (
        datos["ESTADO"] == 1
    )

    # Edad.
    datos["edad_modelo"] = (
        datos["CH06"]
        .replace(-1, 0)
        .astype(float)
    )

    # Educación.
    datos["educacion_modelo"] = (
        datos["NIVEL_ED"]
        .map(MAPA_EDUCACION)
    )

    # Categoría ocupacional.
    # Referencia = 3.
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

    # Rama.
    # Referencia = Comercio.
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

    # Horas.
    horas = datos[
        "PP3E_TOT"
    ]

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

    # Ingreso conocido.
    ingreso = datos["P47T"]

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
# HOGARES / FOLDS
# ============================================================

def id_hogar(df):

    return (
        df["CODUSU"]
        .astype("string")
        + "_"
        + df["NRO_HOGAR"]
        .astype("Int64")
        .astype("string")
    )


def crear_folds(df):

    hogares = (
        id_hogar(df)
        .drop_duplicates()
        .to_numpy()
    )

    rng = np.random.default_rng(
        SEMILLA
    )

    rng.shuffle(hogares)

    grupos = np.array_split(
        hogares,
        N_FOLDS,
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
# MATRICES
# ============================================================

def texto(serie):

    return (
        serie
        .astype("string")
        .fillna("Sin dato")
    )


def matrices_base(
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

    for variable in [
        "edad_modelo",
        "horas_modelo",
        "horas_especiales",
    ]:

        x_train[variable] = (
            pd.to_numeric(
                train[variable],
                errors="coerce",
            )
            .fillna(0)
            .astype(float)
        )

        x_test[variable] = (
            pd.to_numeric(
                test[variable],
                errors="coerce",
            )
            .fillna(0)
            .astype(float)
        )

    # --------------------------------------------------------
    # Categóricas
    # --------------------------------------------------------

    categoricas = {
        "periodo": "2026-T2",
        "AGLOMERADO": "32",
        "CH04": "1",

        "educacion_modelo":
            "Superior completo",

        "ESTADO": "1",

        "cat_ocup_modelo":
            "3",

        "rama_modelo":
            "Comercio",
    }

    for (
        variable,
        referencia,
    ) in categoricas.items():

        tr = texto(
            train[variable]
        )

        te = texto(
            test[variable]
        )

        categorias = sorted(
            tr.unique()
        )

        referencia = str(
            referencia
        )

        if referencia not in categorias:
            referencia = (
                tr.value_counts()
                .index[0]
            )

        for categoria in categorias:

            if categoria == referencia:
                continue

            nombre = (
                f"{variable}="
                f"{categoria}"
            )

            x_train[nombre] = (
                tr.eq(categoria)
                .astype(float)
            )

            x_test[nombre] = (
                te.eq(categoria)
                .astype(float)
            )

    x_test = x_test.reindex(
        columns=x_train.columns,
        fill_value=0.0,
    )

    return x_train, x_test


def aplicar_escala_ipc(
    x_train,
    x_test,
    train,
    test,
):
    """Escala los efectos monetarios según el nivel de precios.

    P47T NO se modifica.
    """

    escala_train = (
        train[
            "escala_precios"
        ]
        .astype(float)
    )

    escala_test = (
        test[
            "escala_precios"
        ]
        .astype(float)
    )

    final_train = pd.DataFrame(
        index=train.index
    )

    final_test = pd.DataFrame(
        index=test.index
    )

    # Funciona como intercepto
    # expresado en escala nominal.
    final_train[
        "escala_precios"
    ] = escala_train

    final_test[
        "escala_precios"
    ] = escala_test

    # Todos los efectos monetarios
    # se expresan en la escala de
    # precios de cada trimestre.
    for columna in x_train.columns:

        nombre = (
            f"{columna}"
            "_x_escala"
        )

        final_train[nombre] = (
            x_train[columna]
            * escala_train
        )

        final_test[nombre] = (
            x_test[columna]
            * escala_test
        )

    return (
        final_train,
        final_test,
    )


def construir_matrices(
    train,
    test,
):

    base_train, base_test = (
        matrices_base(
            train,
            test,
        )
    )

    return aplicar_escala_ipc(
        base_train,
        base_test,
        train,
        test,
    )


# ============================================================
# REGRESIÓN
# ============================================================

def ajustar(x, y):

    matriz = x.to_numpy(
        dtype=float
    )

    coef, _, rango, _ = (
        np.linalg.lstsq(
            matriz,
            y.to_numpy(
                dtype=float
            ),
            rcond=None,
        )
    )

    return coef, rango


def predecir(x, coef):

    return (
        x.to_numpy(
            dtype=float
        )
        @ coef
    )


# ============================================================
# MÉTRICAS
# ============================================================

def r2(y, pred):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    total = np.sum(
        (
            y - y.mean()
        ) ** 2
    )

    residuos = np.sum(
        (
            y - pred
        ) ** 2
    )

    if total <= 0:
        return np.nan

    return float(
        1
        - residuos / total
    )


def metricas(y, pred):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    error = y - pred

    mse = np.mean(
        error ** 2
    )

    return {
        "r2":
            r2(y, pred),

        "mae":
            float(
                np.mean(
                    np.abs(error)
                )
            ),

        "rmse":
            float(
                np.sqrt(mse)
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
# VALIDACIÓN CRUZADA
# ============================================================

def validar(datos):

    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    conocidos["_hogar"] = (
        id_hogar(conocidos)
    )

    mapa = crear_folds(
        conocidos
    )

    conocidos["_fold"] = (
        conocidos["_hogar"]
        .map(mapa)
    )

    folds = []
    predicciones = []

    for fold in range(
        1,
        N_FOLDS + 1,
    ):

        print(
            f"\nFold {fold}/{N_FOLDS}"
        )

        train = conocidos.loc[
            conocidos["_fold"]
            != fold
        ].copy()

        test = conocidos.loc[
            conocidos["_fold"]
            == fold
        ].copy()

        x_train, x_test = (
            construir_matrices(
                train,
                test,
            )
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

        folds.append({
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

        salida["fold"] = fold
        salida["prediccion"] = pred

        predicciones.append(
            salida
        )

        print(
            f"R²={met['r2']:.4f} | "
            f"MAE=${met['mae']:,.0f} | "
            f"RMSE=${met['rmse']:,.0f}"
        )

    return (
        pd.DataFrame(folds),

        pd.concat(
            predicciones,
            ignore_index=True,
        ),
    )


# ============================================================
# POR PERÍODO
# ============================================================

def por_periodo(pred):

    filas = []

    for periodo, parte in (
        pred.groupby(
            "periodo"
        )
    ):

        met = metricas(
            parte["P47T"],
            parte["prediccion"],
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
# AJUSTE COMPLETO
# ============================================================

def ajuste_completo(datos):

    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    x, _ = construir_matrices(
        conocidos,
        conocidos,
    )

    y = (
        conocidos["P47T"]
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

    coeficientes = pd.DataFrame({
        "variable":
            list(x.columns),

        "coeficiente":
            coef,
    })

    return {
        "n":
            len(y),

        "n_predictores":
            x.shape[1],

        "rango_completo":
            (
                rango
                == x.shape[1]
            ),

        "r2_ajuste_completo":
            r2(
                y,
                pred,
            ),
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
        "\nIngresos conocidos:",
        int(
            datos[
                "ingreso_conocido"
            ].sum()
        ),
    )

    print(
        "No respuestas:",
        int(
            datos[
                "ingreso_no_responde"
            ].sum()
        ),
    )

    folds, pred = validar(
        datos
    )

    periodo = por_periodo(
        pred
    )

    ajuste, coef = (
        ajuste_completo(
            datos
        )
    )

    met_global = metricas(
        pred["P47T"],
        pred["prediccion"],
    )

    resumen = pd.DataFrame([
        {
            **ajuste,

            "r2_cv_promedio":
                folds["r2"].mean(),

            "r2_cv_min":
                folds["r2"].min(),

            "r2_cv_max":
                folds["r2"].max(),

            "r2_oof_conjunto":
                met_global["r2"],

            "mae_oof":
                met_global["mae"],

            "rmse_oof":
                met_global["rmse"],

            "pred_negativas_pct":
                met_global[
                    "pred_negativas_pct"
                ],

            "r2_periodos_promedio":
                periodo[
                    "r2"
                ].mean(),

            "r2_periodos_mediana":
                periodo[
                    "r2"
                ].median(),

            "r2_periodos_min":
                periodo[
                    "r2"
                ].min(),

            "r2_periodos_max":
                periodo[
                    "r2"
                ].max(),

            "periodos_r2_mayor_030":
                int(
                    (
                        periodo["r2"]
                        > 0.30
                    )
                    .sum()
                ),

            "cantidad_periodos":
                len(periodo),
        }
    ])

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    resumen.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_resumen.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    folds.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_folds.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    periodo.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_periodos.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    coef.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_coeficientes.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    pred.to_csv(
        RESULTADOS
        / "objetivo4_regresion_historica_ipc_predicciones.csv",
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
        "RESULTADO REGRESIÓN "
        "HISTÓRICA CON IPC"
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
        "\nR² POR TRIMESTRE"
    )

    print(
        periodo[
            [
                "periodo",
                "n",
                "r2",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    valor = float(
        resumen[
            "r2_oof_conjunto"
        ].iloc[0]
    )

    print(
        "\nR² OOF CONJUNTO:",
        f"{valor:.4f}",
    )

    print(
        "RESULTADO:",
        (
            "SUPERA 0,30"
            if valor > 0.30
            else "NO SUPERA 0,30"
        ),
    )


if __name__ == "__main__":
    main()