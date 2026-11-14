"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 4 - ETAPA 3
Validación cruzada y depuración de la regresión lineal múltiple.

OBJETIVOS:
1. Comprobar si el R² > 0,30 se mantiene al cambiar la muestra.
2. Utilizar validación cruzada de 5 partes por hogar.
3. Eliminar redundancias del modelo ampliado anterior.
4. Utilizar referencias categóricas interpretables.
5. Evaluar por separado el rendimiento general y entre ocupados.
6. Obtener coeficientes más aptos para interpretar posteriormente.

IMPORTANTE:
- P47T permanece en su escala original.
- P47T = 0 es una respuesta válida.
- P47T = -9 / ausente no participa del entrenamiento.
- Todavía NO se imputan los ingresos faltantes.
- No se utiliza P21.
- Los ponderadores no se utilizan como predictores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa3_corregido import (
    clasificar_rama,
)


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
    "PP3E_TOT",

    "P47T",
]


# ============================================================
# GRUPOS EDUCATIVOS
# ============================================================

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
# 1. LECTURA
# ============================================================

def cargar_base() -> pd.DataFrame:

    if not ARCHIVO.is_file():
        raise FileNotFoundError(
            f"No se encontró:\n{ARCHIVO}"
        )

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

    # Solo CABA y Partidos del GBA.
    df = df.loc[
        df["AGLOMERADO"].isin(
            [32, 33]
        )
    ].copy()

    # Solo entrevistas individuales realizadas.
    df = df.loc[
        df["H15"] == 1
    ].copy()

    return df


# ============================================================
# 2. PREPARACIÓN DE VARIABLES
# ============================================================

def preparar_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:

    datos = df.copy()

    # --------------------------------------------------------
    # Edad
    # --------------------------------------------------------

    # CH06 = -1 significa menor de un año.
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
        .map(MAPA_EDUCACION)
    )

    if (
        datos[
            "educacion_modelo"
        ].isna().any()
    ):

        valores = sorted(
            datos.loc[
                datos[
                    "educacion_modelo"
                ].isna(),
                "NIVEL_ED",
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            "NIVEL_ED contiene códigos "
            f"no previstos: {valores}"
        )

    # --------------------------------------------------------
    # Ocupación
    # --------------------------------------------------------

    ocupados = (
        datos["ESTADO"] == 1
    )

    # En el modelo general queremos que los no ocupados
    # no generen una categoría adicional redundante.
    #
    # CAT_OCUP=3 será la categoría de referencia.
    # A los no ocupados también les asignamos esa referencia,
    # porque ESTADO ya informa que no están ocupados.
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

    # Comercio será referencia.
    #
    # Los no ocupados reciben la referencia,
    # ya que ESTADO indica que no corresponde rama laboral.
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

    horas = datos["PP3E_TOT"]

    # Las horas entre 0 y 168 se conservan.
    horas_validas = (
        ocupados
        & horas.notna()
        & horas.between(
            0,
            168,
        )
    )

    # Código especial: por ejemplo, 999.
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
    # Ingreso
    # --------------------------------------------------------

    ingreso = datos["P47T"]

    negativos_invalidos = (
        ingreso.notna()
        & ingreso.lt(0)
        & ingreso.ne(-9)
    )

    if negativos_invalidos.any():

        valores = sorted(
            ingreso.loc[
                negativos_invalidos
            ].unique()
        )

        raise ValueError(
            "P47T contiene valores "
            f"negativos inesperados: {valores}"
        )

    datos[
        "ingreso_conocido"
    ] = (
        ingreso.notna()
        & ingreso.ge(0)
    )

    return datos


# ============================================================
# 3. IDENTIFICADOR DE HOGAR
# ============================================================

def construir_id_hogar(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["CODUSU"]
        .astype("string")
        + "_"
        + df["NRO_HOGAR"]
        .astype("Int64")
        .astype("string")
    )


# ============================================================
# 4. CREACIÓN DE LOS 5 FOLDS
# ============================================================

def asignar_folds(
    conocidos: pd.DataFrame,
) -> dict:

    datos = conocidos.copy()

    datos[
        "_id_hogar"
    ] = construir_id_hogar(
        datos
    )

    hogares = (
        datos["_id_hogar"]
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

def a_texto(
    serie: pd.Series,
) -> pd.Series:

    return (
        serie
        .astype("string")
        .fillna("Sin dato")
    )


def agregar_categoria(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    variable: str,
    referencia: str,
) -> None:
    """Agrega variables dummy dejando una categoría de referencia."""

    train = a_texto(
        train
    )

    test = a_texto(
        test
    )

    categorias = sorted(
        train.unique()
    )

    if referencia not in categorias:

        # Si por alguna razón la referencia no aparece
        # en ese fold, usamos la categoría más frecuente.
        referencia = (
            train
            .value_counts()
            .index[0]
        )

    for categoria in categorias:

        if categoria == referencia:
            continue

        nombre = (
            f"{variable}={categoria}"
        )

        x_train[nombre] = (
            train.eq(
                categoria
            )
            .astype(float)
        )

        x_test[nombre] = (
            test.eq(
                categoria
            )
            .astype(float)
        )


def construir_matrices(
    entrenamiento: pd.DataFrame,
    evaluacion: pd.DataFrame,
    numericas: list[str],
    categoricas: dict[str, str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    x_train = pd.DataFrame(
        index=entrenamiento.index
    )

    x_test = pd.DataFrame(
        index=evaluacion.index
    )

    # --------------------------------------------------------
    # Numéricas
    # --------------------------------------------------------

    for variable in numericas:

        train = pd.to_numeric(
            entrenamiento[
                variable
            ],
            errors="coerce",
        )

        test = pd.to_numeric(
            evaluacion[
                variable
            ],
            errors="coerce",
        )

        if train.isna().any():

            mediana = float(
                train.median()
            )

            train = (
                train.fillna(
                    mediana
                )
            )

            test = (
                test.fillna(
                    mediana
                )
            )

        x_train[
            variable
        ] = train.astype(float)

        x_test[
            variable
        ] = test.astype(float)

    # --------------------------------------------------------
    # Categóricas
    # --------------------------------------------------------

    for (
        variable,
        referencia,
    ) in categoricas.items():

        agregar_categoria(
            x_train=x_train,
            x_test=x_test,
            train=entrenamiento[
                variable
            ],
            test=evaluacion[
                variable
            ],
            variable=variable,
            referencia=str(
                referencia
            ),
        )

    # Mismo orden.
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

def matriz_con_intercepto(
    x: pd.DataFrame,
) -> np.ndarray:

    matriz = x.to_numpy(
        dtype=float
    )

    return np.column_stack([
        np.ones(
            len(matriz)
        ),
        matriz,
    ])


def ajustar(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    np.ndarray,
    int,
]:

    matriz = matriz_con_intercepto(
        x
    )

    objetivo = y.to_numpy(
        dtype=float
    )

    (
        coeficientes,
        _,
        rango,
        _,
    ) = np.linalg.lstsq(
        matriz,
        objetivo,
        rcond=None,
    )

    return (
        coeficientes,
        rango,
    )


def predecir(
    x: pd.DataFrame,
    coeficientes: np.ndarray,
) -> np.ndarray:

    return (
        matriz_con_intercepto(
            x
        )
        @ coeficientes
    )


# ============================================================
# 7. MÉTRICAS
# ============================================================

def r2(
    reales: np.ndarray,
    predichos: np.ndarray,
) -> float:

    reales = np.asarray(
        reales,
        dtype=float,
    )

    predichos = np.asarray(
        predichos,
        dtype=float,
    )

    if len(reales) < 2:
        return np.nan

    ss_total = np.sum(
        (
            reales
            - reales.mean()
        ) ** 2
    )

    if ss_total <= 0:
        return np.nan

    ss_res = np.sum(
        (
            reales
            - predichos
        ) ** 2
    )

    return float(
        1
        - ss_res / ss_total
    )


def metricas(
    reales: np.ndarray,
    predichos: np.ndarray,
) -> dict:

    reales = np.asarray(
        reales,
        dtype=float,
    )

    predichos = np.asarray(
        predichos,
        dtype=float,
    )

    errores = (
        reales
        - predichos
    )

    mse = float(
        np.mean(
            errores ** 2
        )
    )

    return {
        "r2":
            r2(
                reales,
                predichos,
            ),

        "mae":
            float(
                np.mean(
                    np.abs(
                        errores
                    )
                )
            ),

        "mse":
            mse,

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
                    predichos < 0
                )
            ),
    }


def r2_subgrupo(
    datos: pd.DataFrame,
    predichos: np.ndarray,
    mascara: pd.Series,
) -> float:

    mascara = np.asarray(
        mascara,
        dtype=bool,
    )

    if mascara.sum() < 2:
        return np.nan

    return r2(
        datos.loc[
            mascara,
            "P47T",
        ].to_numpy(
            dtype=float
        ),
        predichos[
            mascara
        ],
    )


# ============================================================
# 8. ESPECIFICACIONES
# ============================================================

def obtener_modelos(
    datos: pd.DataFrame,
) -> list[dict]:

    ocupados = datos.loc[
        datos["ESTADO"] == 1
    ].copy()

    return [
        {
            "nombre":
                "General básico limpio",

            "datos":
                datos,

            "numericas": [
                "edad_modelo",
            ],

            "categoricas": {
                "AGLOMERADO": "32",
                "CH04": "1",
                "educacion_modelo":
                    "Superior completo",
                "ESTADO": "1",
            },
        },

        {
            "nombre":
                "General laboral limpio",

            "datos":
                datos,

            "numericas": [
                "edad_modelo",
                "horas_modelo",
                "horas_especiales",
            ],

            "categoricas": {
                "AGLOMERADO": "32",
                "CH04": "1",
                "educacion_modelo":
                    "Superior completo",
                "ESTADO": "1",
                "cat_ocup_modelo": "3",
                "rama_modelo":
                    "Comercio",
            },
        },

        {
            "nombre":
                "Ocupados laboral limpio",

            "datos":
                ocupados,

            "numericas": [
                "edad_modelo",
                "horas_modelo",
                "horas_especiales",
            ],

            "categoricas": {
                "AGLOMERADO": "32",
                "CH04": "1",
                "educacion_modelo":
                    "Superior completo",
                "cat_ocup_modelo": "3",
                "rama_modelo":
                    "Comercio",
            },
        },
    ]


# ============================================================
# 9. VALIDACIÓN CRUZADA
# ============================================================

def evaluar_modelo_cv(
    especificacion: dict,
    mapa_folds: dict,
) -> tuple[
    list[dict],
    list[pd.DataFrame],
]:

    nombre = especificacion[
        "nombre"
    ]

    datos = (
        especificacion[
            "datos"
        ]
        .loc[
            especificacion[
                "datos"
            ][
                "ingreso_conocido"
            ]
        ]
        .copy()
    )

    datos[
        "_id_hogar"
    ] = construir_id_hogar(
        datos
    )

    datos[
        "_fold"
    ] = (
        datos[
            "_id_hogar"
        ]
        .map(
            mapa_folds
        )
    )

    resultados = []
    predicciones_folds = []

    print(
        "\n"
        + "=" * 80
    )
    print(
        nombre.upper()
    )
    print(
        "=" * 80
    )

    for fold in range(
        1,
        N_FOLDS + 1,
    ):

        entrenamiento = (
            datos.loc[
                datos[
                    "_fold"
                ] != fold
            ]
            .copy()
        )

        evaluacion = (
            datos.loc[
                datos[
                    "_fold"
                ] == fold
            ]
            .copy()
        )

        (
            x_train,
            x_test,
        ) = construir_matrices(
            entrenamiento,
            evaluacion,
            especificacion[
                "numericas"
            ],
            especificacion[
                "categoricas"
            ],
        )

        y_train = (
            entrenamiento[
                "P47T"
            ]
            .astype(float)
        )

        y_test = (
            evaluacion[
                "P47T"
            ]
            .astype(float)
        )

        (
            coeficientes,
            rango,
        ) = ajustar(
            x_train,
            y_train,
        )

        pred = predecir(
            x_test,
            coeficientes,
        )

        met = metricas(
            y_test.to_numpy(),
            pred,
        )

        columnas_matriz = (
            x_train.shape[1]
            + 1
        )

        rango_completo = (
            rango
            == columnas_matriz
        )

        # Diagnósticos específicos.
        positivos = (
            evaluacion[
                "P47T"
            ] > 0
        )

        ocupados = (
            evaluacion[
                "ESTADO"
            ] == 1
        )

        ocupados_positivos = (
            ocupados
            & positivos
        )

        fila = {
            "modelo":
                nombre,

            "fold":
                fold,

            "n_entrenamiento":
                len(
                    entrenamiento
                ),

            "n_evaluacion":
                len(
                    evaluacion
                ),

            "n_predictores":
                x_train.shape[1],

            "rango_completo":
                rango_completo,

            "r2":
                met["r2"],

            "mae":
                met["mae"],

            "mse":
                met["mse"],

            "rmse":
                met["rmse"],

            "pred_negativas_pct":
                met[
                    "pred_negativas_pct"
                ],

            "r2_ingresos_positivos":
                r2_subgrupo(
                    evaluacion,
                    pred,
                    positivos,
                ),

            "r2_ocupados":
                r2_subgrupo(
                    evaluacion,
                    pred,
                    ocupados,
                ),

            "r2_ocupados_positivos":
                r2_subgrupo(
                    evaluacion,
                    pred,
                    ocupados_positivos,
                ),
        }

        resultados.append(
            fila
        )

        tabla_pred = (
            evaluacion[
                [
                    "CODUSU",
                    "NRO_HOGAR",
                    "COMPONENTE",
                    "AGLOMERADO",
                    "ESTADO",
                    "P47T",
                ]
            ]
            .copy()
        )

        tabla_pred[
            "modelo"
        ] = nombre

        tabla_pred[
            "fold"
        ] = fold

        tabla_pred[
            "prediccion"
        ] = pred

        tabla_pred[
            "error"
        ] = (
            tabla_pred[
                "P47T"
            ]
            - tabla_pred[
                "prediccion"
            ]
        )

        predicciones_folds.append(
            tabla_pred
        )

        print(
            f"Fold {fold}: "
            f"R²={met['r2']:.4f} | "
            f"MAE=${met['mae']:,.0f} | "
            f"RMSE=${met['rmse']:,.0f} | "
            f"R² ocupados="
            f"{fila['r2_ocupados']:.4f}"
        )

    return (
        resultados,
        predicciones_folds,
    )


# ============================================================
# 10. AJUSTE FINAL PARA COEFICIENTES
# ============================================================

def ajustar_modelo_completo(
    especificacion: dict,
) -> tuple[
    dict,
    pd.DataFrame,
]:

    nombre = (
        especificacion[
            "nombre"
        ]
    )

    datos = (
        especificacion[
            "datos"
        ]
        .loc[
            especificacion[
                "datos"
            ][
                "ingreso_conocido"
            ]
        ]
        .copy()
    )

    # Para obtener los coeficientes finales
    # construimos matrices usando el mismo conjunto
    # como entrenamiento y evaluación.
    (
        x,
        _,
    ) = construir_matrices(
        datos,
        datos,
        especificacion[
            "numericas"
        ],
        especificacion[
            "categoricas"
        ],
    )

    y = (
        datos[
            "P47T"
        ]
        .astype(float)
    )

    (
        coeficientes,
        rango,
    ) = ajustar(
        x,
        y,
    )

    pred = predecir(
        x,
        coeficientes,
    )

    r2_entrenamiento = r2(
        y.to_numpy(),
        pred,
    )

    p = x.shape[1]
    n = len(y)

    r2_ajustado = (
        1
        - (
            (1 - r2_entrenamiento)
            * (n - 1)
            / (n - p - 1)
        )
        if n > p + 1
        else np.nan
    )

    nombres = (
        ["Intercepto"]
        + list(
            x.columns
        )
    )

    coef = pd.DataFrame({
        "modelo":
            nombre,

        "variable":
            nombres,

        "coeficiente":
            coeficientes,
    })

    resumen = {
        "modelo":
            nombre,

        "n":
            n,

        "n_predictores":
            p,

        "rango_matriz":
            rango,

        "rango_completo":
            (
                rango
                == p + 1
            ),

        "r2_entrenamiento_completo":
            r2_entrenamiento,

        "r2_ajustado_completo":
            r2_ajustado,
    }

    return (
        resumen,
        coef,
    )


# ============================================================
# 11. RESUMEN DE VALIDACIÓN
# ============================================================

def resumir_cv(
    detalle: pd.DataFrame,
) -> pd.DataFrame:

    columnas_metricas = [
        "r2",
        "mae",
        "mse",
        "rmse",
        "pred_negativas_pct",
        "r2_ingresos_positivos",
        "r2_ocupados",
        "r2_ocupados_positivos",
    ]

    filas = []

    for modelo, parte in (
        detalle.groupby(
            "modelo"
        )
    ):

        fila = {
            "modelo":
                modelo,
        }

        for columna in (
            columnas_metricas
        ):

            serie = (
                parte[columna]
                .dropna()
            )

            fila[
                f"{columna}_promedio"
            ] = (
                serie.mean()
                if len(serie)
                else np.nan
            )

            fila[
                f"{columna}_desvio"
            ] = (
                serie.std(
                    ddof=1
                )
                if len(serie) > 1
                else np.nan
            )

            fila[
                f"{columna}_min"
            ] = (
                serie.min()
                if len(serie)
                else np.nan
            )

            fila[
                f"{columna}_max"
            ] = (
                serie.max()
                if len(serie)
                else np.nan
            )

        fila[
            "todos_los_folds_rango_completo"
        ] = bool(
            parte[
                "rango_completo"
            ].all()
        )

        filas.append(
            fila
        )

    return (
        pd.DataFrame(
            filas
        )
        .sort_values(
            "r2_promedio",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:

    RESULTADOS.mkdir(
        exist_ok=True
    )

    df = cargar_base()

    df = preparar_variables(
        df
    )

    conocidos_generales = (
        df.loc[
            df[
                "ingreso_conocido"
            ]
        ]
        .copy()
    )

    mapa_folds = asignar_folds(
        conocidos_generales
    )

    modelos = obtener_modelos(
        df
    )

    resultados_cv = []
    predicciones = []

    resumenes_completos = []
    coeficientes = []

    for modelo in modelos:

        (
            resultado,
            pred,
        ) = evaluar_modelo_cv(
            modelo,
            mapa_folds,
        )

        resultados_cv.extend(
            resultado
        )

        predicciones.extend(
            pred
        )

        (
            resumen_completo,
            coef,
        ) = ajustar_modelo_completo(
            modelo
        )

        resumenes_completos.append(
            resumen_completo
        )

        coeficientes.append(
            coef
        )

    detalle = pd.DataFrame(
        resultados_cv
    )

    resumen = resumir_cv(
        detalle
    )

    completos = pd.DataFrame(
        resumenes_completos
    )

    resumen = resumen.merge(
        completos,
        on="modelo",
        how="left",
        validate="one_to_one",
    )

    tabla_coef = pd.concat(
        coeficientes,
        ignore_index=True,
    )

    tabla_pred = pd.concat(
        predicciones,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    detalle.to_csv(
        RESULTADOS
        / "objetivo4_validacion_cruzada_prueba2.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    resumen.to_csv(
        RESULTADOS
        / "objetivo4_resumen_validacion_prueba2.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_coef.to_csv(
        RESULTADOS
        / "objetivo4_coeficientes_limpios_prueba2.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_pred.to_csv(
        RESULTADOS
        / "objetivo4_predicciones_cv_prueba2.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resultado en terminal
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "RESUMEN DE VALIDACIÓN CRUZADA"
    )
    print(
        "=" * 80
    )

    columnas = [
        "modelo",
        "r2_promedio",
        "r2_desvio",
        "r2_min",
        "r2_max",
        "mae_promedio",
        "rmse_promedio",
        "pred_negativas_pct_promedio",
        "r2_ingresos_positivos_promedio",
        "r2_ocupados_promedio",
        "r2_ocupados_positivos_promedio",
        "todos_los_folds_rango_completo",
    ]

    print(
        resumen[
            columnas
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.4f}",
        )
    )

    print(
        "\nCriterio formal de la consigna:"
    )

    for _, fila in (
        resumen.iterrows()
    ):

        cumple = (
            fila[
                "r2_promedio"
            ] > 0.30
        )

        print(
            f" - {fila['modelo']}: "
            f"R² medio="
            f"{fila['r2_promedio']:.4f} "
            f"=> "
            f"{'SUPERA' if cumple else 'NO SUPERA'} 0,30"
        )

    print(
        "\nTodavía NO se imputaron "
        "los ingresos faltantes."
    )


if __name__ == "__main__":
    main()