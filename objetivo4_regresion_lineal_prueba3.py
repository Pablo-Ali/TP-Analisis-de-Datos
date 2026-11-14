from pathlib import Path

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa3_corregido import clasificar_rama


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RESULTADOS = BASE / "resultados"

ARCHIVO = DATOS / "usu_individual_T226.txt"

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
    "PP04D_COD",
    "PP04C99",
    "PP04G",

    "PP3E_TOT",

    "PP07H",
    "PP07J",
    "PP07K",

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

def cargar_base():

    df = pd.read_csv(
        ARCHIVO,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=COLUMNAS,
        dtype={
            "CODUSU": "string",
            "PP04B_COD": "string",
            "PP04D_COD": "string",
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
            "PP04D_COD",
        }:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    # CABA y Partidos del GBA.
    df = df.loc[
        df["AGLOMERADO"].isin([32, 33])
        & df["H15"].eq(1)
    ].copy()

    return df


# ============================================================
# PREPARACIÓN
# ============================================================

def preparar_variables(df):

    datos = df.copy()

    ocupados = datos["ESTADO"].eq(1)

    # --------------------------------------------------------
    # Edad
    # --------------------------------------------------------

    datos["edad_modelo"] = (
        datos["CH06"]
        .replace(-1, 0)
        .astype(float)
    )

    # --------------------------------------------------------
    # Educación agrupada
    # --------------------------------------------------------

    datos["educacion_modelo"] = (
        datos["NIVEL_ED"]
        .map(MAPA_EDUCACION)
    )

    # --------------------------------------------------------
    # Categoría ocupacional
    # --------------------------------------------------------

    # Referencia = 3.
    # Para no ocupados también usamos la referencia.
    datos["cat_ocup_modelo"] = "3"

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

    # Referencia = Comercio.
    datos["rama_modelo"] = "Comercio"

    datos.loc[
        ocupados,
        "rama_modelo",
    ] = (
        datos.loc[
            ocupados,
            "PP04B_COD",
        ]
        .map(clasificar_rama)
    )

    # --------------------------------------------------------
    # Grupo ocupacional CNO
    # --------------------------------------------------------

    # Referencia = grupo 3.
    datos["grupo_cno"] = "3"

    codigos_cno = (
        datos.loc[
            ocupados,
            "PP04D_COD",
        ]
        .astype("string")
        .str.strip()
        .str.zfill(5)
    )

    datos.loc[
        ocupados,
        "grupo_cno",
    ] = codigos_cno.str[0]

    # --------------------------------------------------------
    # Tamaño / características laborales
    # --------------------------------------------------------

    # Para los no ocupados asignamos la categoría de referencia.
    # ESTADO ya informa que no están ocupados.

    referencias = {
        "PP04C99": "0",
        "PP04G": "11",
        "PP07H": "1",
        "PP07J": "1",
        "PP07K": "1",
    }

    for variable, referencia in referencias.items():

        nueva = f"{variable}_modelo"

        datos[nueva] = referencia

        datos.loc[
            ocupados,
            nueva,
        ] = (
            datos.loc[
                ocupados,
                variable,
            ]
            .astype("Int64")
            .astype("string")
        )

    # --------------------------------------------------------
    # Horas
    # --------------------------------------------------------

    horas = datos["PP3E_TOT"]

    horas_validas = (
        ocupados
        & horas.notna()
        & horas.between(0, 168)
    )

    horas_especiales = (
        ocupados
        & (
            horas.isna()
            | horas.gt(168)
        )
    )

    datos["horas_modelo"] = 0.0

    datos.loc[
        horas_validas,
        "horas_modelo",
    ] = (
        horas.loc[
            horas_validas
        ]
        .astype(float)
    )

    datos["horas_especiales"] = (
        horas_especiales
        .astype(float)
    )

    # --------------------------------------------------------
    # Ingreso conocido
    # --------------------------------------------------------

    ingreso = datos["P47T"]

    datos["ingreso_conocido"] = (
        ingreso.notna()
        & ingreso.ge(0)
    )

    datos["ingreso_no_responde"] = (
        ingreso.isna()
        | ingreso.eq(-9)
    )

    return datos


# ============================================================
# HOGAR Y FOLDS
# ============================================================

def id_hogar(df):

    return (
        df["CODUSU"].astype("string")
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

    for numero, grupo in enumerate(
        grupos,
        start=1,
    ):

        for hogar in grupo:
            mapa[hogar] = numero

    return mapa


# ============================================================
# CODIFICACIÓN
# ============================================================

def texto(serie):

    return (
        serie
        .astype("string")
        .fillna("Sin dato")
    )


def construir_matrices(
    train,
    test,
    numericas,
    categoricas,
):

    x_train = pd.DataFrame(
        index=train.index
    )

    x_test = pd.DataFrame(
        index=test.index
    )

    # Numéricas.
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

            mediana = tr.median()

            tr = tr.fillna(
                mediana
            )

            te = te.fillna(
                mediana
            )

        x_train[variable] = tr.astype(float)
        x_test[variable] = te.astype(float)

    # Categóricas.
    for variable, referencia in categoricas.items():

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
                f"{variable}={categoria}"
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


# ============================================================
# REGRESIÓN
# ============================================================

def matriz_intercepto(x):

    return np.column_stack([
        np.ones(len(x)),
        x.to_numpy(dtype=float),
    ])


def ajustar(x, y):

    matriz = matriz_intercepto(x)

    coeficientes, _, rango, _ = (
        np.linalg.lstsq(
            matriz,
            y.to_numpy(dtype=float),
            rcond=None,
        )
    )

    return coeficientes, rango


def predecir(x, coeficientes):

    return (
        matriz_intercepto(x)
        @ coeficientes
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_r2(y, pred):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    ss_total = np.sum(
        (y - y.mean()) ** 2
    )

    ss_res = np.sum(
        (y - pred) ** 2
    )

    return (
        1 - ss_res / ss_total
        if ss_total > 0
        else np.nan
    )


def metricas(y, pred):

    errores = y - pred

    mse = np.mean(
        errores ** 2
    )

    return {
        "r2": calcular_r2(
            y,
            pred,
        ),

        "mae": np.mean(
            np.abs(errores)
        ),

        "rmse": np.sqrt(
            mse
        ),

        "pred_negativas_pct":
            100 * np.mean(
                pred < 0
            ),
    }


# ============================================================
# MODELOS
# ============================================================

def obtener_modelos():

    base = {
        "AGLOMERADO": "32",
        "CH04": "1",

        "educacion_modelo":
            "Superior completo",

        "ESTADO": "1",
        "cat_ocup_modelo": "3",
        "rama_modelo": "Comercio",
    }

    extendidas = {
        **base,

        "grupo_cno": "3",

        "PP04C99_modelo": "0",
        "PP04G_modelo": "11",
        "PP07H_modelo": "1",
    }

    completo = {
        **extendidas,

        "PP07J_modelo": "1",
        "PP07K_modelo": "1",
    }

    return [
        {
            "nombre":
                "Laboral extendido",

            "numericas": [
                "edad_modelo",
                "horas_modelo",
                "horas_especiales",
            ],

            "categoricas":
                extendidas,
        },

        {
            "nombre":
                "Laboral completo",

            "numericas": [
                "edad_modelo",
                "horas_modelo",
                "horas_especiales",
            ],

            "categoricas":
                completo,
        },
    ]


# ============================================================
# VALIDACIÓN CRUZADA
# ============================================================

def evaluar_modelo(
    datos,
    modelo,
    mapa_folds,
):

    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    conocidos["_hogar"] = (
        id_hogar(conocidos)
    )

    conocidos["_fold"] = (
        conocidos["_hogar"]
        .map(mapa_folds)
    )

    resultados = []
    predicciones = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        modelo["nombre"].upper()
    )

    print(
        "=" * 75
    )

    for fold in range(
        1,
        N_FOLDS + 1,
    ):

        train = conocidos.loc[
            conocidos["_fold"] != fold
        ].copy()

        test = conocidos.loc[
            conocidos["_fold"] == fold
        ].copy()

        x_train, x_test = (
            construir_matrices(
                train,
                test,
                modelo[
                    "numericas"
                ],
                modelo[
                    "categoricas"
                ],
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
            y_test.to_numpy(),
            pred,
        )

        resultados.append({
            "modelo":
                modelo["nombre"],

            "fold":
                fold,

            "n":
                len(test),

            "n_predictores":
                x_train.shape[1],

            "rango_completo":
                rango
                == x_train.shape[1] + 1,

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
                "ESTADO",
                "P47T",
            ]
        ].copy()

        salida["modelo"] = (
            modelo["nombre"]
        )

        salida["fold"] = fold

        salida["prediccion"] = pred

        predicciones.append(
            salida
        )

        print(
            f"Fold {fold}: "
            f"R²={met['r2']:.4f} | "
            f"MAE=${met['mae']:,.0f} | "
            f"RMSE=${met['rmse']:,.0f}"
        )

    return resultados, predicciones


# ============================================================
# AJUSTE COMPLETO
# ============================================================

def ajustar_completo(
    datos,
    modelo,
):

    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    x, _ = construir_matrices(
        conocidos,
        conocidos,
        modelo["numericas"],
        modelo["categoricas"],
    )

    y = conocidos[
        "P47T"
    ].astype(float)

    coef, rango = ajustar(
        x,
        y,
    )

    pred = predecir(
        x,
        coef,
    )

    r2 = calcular_r2(
        y.to_numpy(),
        pred,
    )

    n = len(y)
    p = x.shape[1]

    r2_ajustado = (
        1
        - (1 - r2)
        * (n - 1)
        / (n - p - 1)
    )

    tabla_coef = pd.DataFrame({
        "modelo":
            modelo["nombre"],

        "variable":
            ["Intercepto"]
            + list(x.columns),

        "coeficiente":
            coef,
    })

    return {
        "modelo":
            modelo["nombre"],

        "r2_ajuste_completo":
            r2,

        "r2_ajustado":
            r2_ajustado,

        "n_predictores":
            p,

        "rango_completo":
            rango == p + 1,
    }, tabla_coef


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTADOS.mkdir(
        exist_ok=True
    )

    df = cargar_base()

    df = preparar_variables(
        df
    )

    conocidos = df.loc[
        df["ingreso_conocido"]
    ].copy()

    mapa_folds = crear_folds(
        conocidos
    )

    modelos = obtener_modelos()

    detalle = []
    predicciones = []
    ajustes = []
    coeficientes = []

    for modelo in modelos:

        res, pred = evaluar_modelo(
            df,
            modelo,
            mapa_folds,
        )

        detalle.extend(
            res
        )

        predicciones.extend(
            pred
        )

        ajuste, coef = (
            ajustar_completo(
                df,
                modelo,
            )
        )

        ajustes.append(
            ajuste
        )

        coeficientes.append(
            coef
        )

    detalle = pd.DataFrame(
        detalle
    )

    predicciones = pd.concat(
        predicciones,
        ignore_index=True,
    )

    ajustes = pd.DataFrame(
        ajustes
    )

    coeficientes = pd.concat(
        coeficientes,
        ignore_index=True,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    filas = []

    for modelo in modelos:

        nombre = modelo["nombre"]

        parte = detalle.loc[
            detalle["modelo"]
            == nombre
        ]

        pred = predicciones.loc[
            predicciones["modelo"]
            == nombre
        ]

        y = pred["P47T"].to_numpy(
            dtype=float
        )

        estimado = (
            pred["prediccion"]
            .to_numpy(
                dtype=float
            )
        )

        met_global = metricas(
            y,
            estimado,
        )

        ajuste = ajustes.loc[
            ajustes["modelo"]
            == nombre
        ].iloc[0]

        filas.append({
            "modelo":
                nombre,

            "r2_ajuste_completo":
                ajuste[
                    "r2_ajuste_completo"
                ],

            "r2_ajustado":
                ajuste[
                    "r2_ajustado"
                ],

            "r2_cv_promedio":
                parte["r2"].mean(),

            "r2_cv_min":
                parte["r2"].min(),

            "r2_cv_max":
                parte["r2"].max(),

            # ESTE ES EL PRINCIPAL.
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

            "todos_rangos_completos":
                bool(
                    parte[
                        "rango_completo"
                    ].all()
                ),
        })

    resumen = pd.DataFrame(
        filas
    ).sort_values(
        "r2_oof_conjunto",
        ascending=False,
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    resumen.to_csv(
        RESULTADOS
        / "objetivo4_resumen_regresion_prueba3.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    detalle.to_csv(
        RESULTADOS
        / "objetivo4_folds_regresion_prueba3.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    coeficientes.to_csv(
        RESULTADOS
        / "objetivo4_coeficientes_regresion_prueba3.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    predicciones.to_csv(
        RESULTADOS
        / "objetivo4_predicciones_regresion_prueba3.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RESULTADO FINAL PRUEBA 3"
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
        "\nIMPORTANTE:"
    )

    print(
        "El criterio principal será "
        "r2_oof_conjunto."
    )

    for _, fila in (
        resumen.iterrows()
    ):

        cumple = (
            fila[
                "r2_oof_conjunto"
            ]
            > 0.30
        )

        print(
            f"{fila['modelo']}: "
            f"R² OOF="
            f"{fila['r2_oof_conjunto']:.4f} "
            f"=> "
            f"{'SUPERA' if cumple else 'NO SUPERA'} 0,30"
        )


if __name__ == "__main__":
    main()