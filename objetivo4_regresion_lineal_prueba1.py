"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 4 - ETAPA 2
Primera comparación de modelos de regresión lineal múltiple.

IMPORTANTE:
- Se trabaja únicamente con el segundo trimestre de 2026.
- P47T se mantiene en su escala original.
- P47T = 0 es una respuesta válida.
- P47T = -9 / ausente es no respuesta y NO participa del ajuste.
- En esta etapa NO se imputan todavía los ingresos faltantes.
- No se utiliza P21 como predictor.
- No se utiliza ningún ponderador como variable explicativa.

Se comparan:
1. Modelo general básico.
2. Modelo general ampliado.
3. Modelo de ocupados básico.
4. Modelo de ocupados ampliado.

La regresión se ajusta por mínimos cuadrados con NumPy.
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

AGLOMERADOS = {
    32: "CABA",
    33: "Partidos del GBA",
}

PROPORCION_EVALUACION = 0.20
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
# 1. LECTURA
# ============================================================

def cargar_base() -> pd.DataFrame:
    """Carga las variables necesarias del segundo trimestre de 2026."""

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

    # Solo los dos aglomerados elegidos.
    df = df.loc[
        df["AGLOMERADO"].isin(
            AGLOMERADOS
        )
    ].copy()

    # Universo de entrevistas individuales realizadas.
    df = df.loc[
        df["H15"] == 1
    ].copy()

    return df


# ============================================================
# 2. VARIABLES PARA EL MODELO
# ============================================================

def preparar_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye variables auxiliares sin modificar los datos originales."""

    datos = df.copy()

    # --------------------------------------------------------
    # Edad
    # --------------------------------------------------------

    # En CH06, -1 significa menor de un año.
    datos["edad_modelo"] = (
        datos["CH06"]
        .replace(-1, 0)
    )

    # --------------------------------------------------------
    # Rama de actividad
    # --------------------------------------------------------

    # Solo corresponde a ocupados.
    datos["rama_modelo"] = (
        "No corresponde"
    )

    ocupados = datos["ESTADO"].eq(1)

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
    # Horas
    # --------------------------------------------------------

    horas = datos["PP3E_TOT"]

    # Distinguimos tres situaciones:
    # - horas válidas;
    # - código especial / no informado;
    # - no corresponde por no estar ocupado.
    datos["situacion_horas"] = (
        "No corresponde"
    )

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

    datos.loc[
        horas_validas,
        "situacion_horas",
    ] = "Horas válidas"

    datos.loc[
        horas_especiales,
        "situacion_horas",
    ] = "Código especial"

    # Para poder utilizar una columna numérica:
    # solo conservamos las horas válidas.
    # En las demás situaciones usamos 0, pero la variable
    # situacion_horas permite distinguir esos casos.
    datos["horas_modelo"] = 0.0

    datos.loc[
        horas_validas,
        "horas_modelo",
    ] = horas.loc[
        horas_validas
    ]

    # --------------------------------------------------------
    # P47T
    # --------------------------------------------------------

    ingresos = datos["P47T"]

    negativos_inesperados = (
        ingresos.notna()
        & ingresos.lt(0)
        & ingresos.ne(-9)
    )

    if negativos_inesperados.any():

        valores = sorted(
            ingresos.loc[
                negativos_inesperados
            ].unique()
        )

        raise ValueError(
            f"Códigos negativos inesperados en P47T: {valores}"
        )

    datos["ingreso_conocido"] = (
        ingresos.notna()
        & ingresos.ge(0)
    )

    datos["ingreso_no_responde"] = (
        ingresos.isna()
        | ingresos.eq(-9)
    )

    return datos


# ============================================================
# 3. DIVISIÓN ENTRENAMIENTO / EVALUACIÓN
# ============================================================

def construir_id_hogar(
    df: pd.DataFrame,
) -> pd.Series:
    """Construye un identificador de hogar dentro del trimestre."""

    return (
        df["CODUSU"].astype("string")
        + "_"
        + df["NRO_HOGAR"]
        .astype("Int64")
        .astype("string")
    )


def dividir_por_hogares(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa hogares completos entre entrenamiento y evaluación.

    Así evitamos que integrantes del mismo hogar queden repartidos
    entre ambos conjuntos.
    """

    datos = df.copy()

    datos["_id_hogar"] = (
        construir_id_hogar(datos)
    )

    hogares = (
        datos["_id_hogar"]
        .drop_duplicates()
        .to_numpy()
    )

    rng = np.random.default_rng(
        SEMILLA
    )

    rng.shuffle(hogares)

    cantidad_evaluacion = max(
        1,
        int(
            round(
                len(hogares)
                * PROPORCION_EVALUACION
            )
        ),
    )

    hogares_evaluacion = set(
        hogares[
            :cantidad_evaluacion
        ]
    )

    mascara_evaluacion = (
        datos["_id_hogar"]
        .isin(
            hogares_evaluacion
        )
    )

    entrenamiento = (
        datos.loc[
            ~mascara_evaluacion
        ]
        .drop(
            columns="_id_hogar"
        )
        .copy()
    )

    evaluacion = (
        datos.loc[
            mascara_evaluacion
        ]
        .drop(
            columns="_id_hogar"
        )
        .copy()
    )

    return (
        entrenamiento,
        evaluacion,
    )


# ============================================================
# 4. CODIFICACIÓN DE VARIABLES
# ============================================================

def convertir_a_texto(
    serie: pd.Series,
) -> pd.Series:
    """Convierte categorías a texto y preserva faltantes."""

    return (
        serie
        .astype("string")
        .fillna("Sin dato")
    )


def construir_matrices(
    entrenamiento: pd.DataFrame,
    evaluacion: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte las variables categóricas en variables indicadoras.

    Las categorías se definen utilizando únicamente el conjunto de
    entrenamiento. La primera categoría queda como referencia.
    """

    x_train = pd.DataFrame(
        index=entrenamiento.index
    )

    x_test = pd.DataFrame(
        index=evaluacion.index
    )

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    for variable in numericas:

        train = pd.to_numeric(
            entrenamiento[variable],
            errors="coerce",
        )

        test = pd.to_numeric(
            evaluacion[variable],
            errors="coerce",
        )

        if train.isna().any():

            mediana = float(
                train.median()
            )

            train = train.fillna(
                mediana
            )

            test = test.fillna(
                mediana
            )

        x_train[variable] = (
            train.astype(float)
        )

        x_test[variable] = (
            test.astype(float)
        )

    # --------------------------------------------------------
    # Variables categóricas
    # --------------------------------------------------------

    for variable in categoricas:

        train = convertir_a_texto(
            entrenamiento[variable]
        )

        test = convertir_a_texto(
            evaluacion[variable]
        )

        categorias = sorted(
            train.unique()
        )

        if len(categorias) <= 1:
            # Si no varía, no aporta al modelo.
            continue

        # La primera funciona como categoría de referencia.
        referencia = categorias[0]

        for categoria in categorias[1:]:

            nombre = (
                f"{variable}={categoria}"
            )

            x_train[nombre] = (
                train.eq(categoria)
                .astype(float)
            )

            x_test[nombre] = (
                test.eq(categoria)
                .astype(float)
            )

        print(
            f"    {variable}: "
            f"{len(categorias)} categorías "
            f"(referencia: {referencia})"
        )

    # Aseguramos mismo orden de columnas.
    x_test = x_test.reindex(
        columns=x_train.columns,
        fill_value=0.0,
    )

    return (
        x_train,
        x_test,
    )


# ============================================================
# 5. REGRESIÓN LINEAL
# ============================================================

def agregar_intercepto(
    x: pd.DataFrame,
) -> np.ndarray:
    """Agrega una columna de unos para estimar beta_0."""

    matriz = x.to_numpy(
        dtype=float
    )

    return np.column_stack(
        [
            np.ones(
                len(matriz)
            ),
            matriz,
        ]
    )


def ajustar_regresion(
    x: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """Ajusta una regresión lineal por mínimos cuadrados."""

    matriz = agregar_intercepto(
        x
    )

    objetivo = y.to_numpy(
        dtype=float
    )

    coeficientes, _, _, _ = (
        np.linalg.lstsq(
            matriz,
            objetivo,
            rcond=None,
        )
    )

    return coeficientes


def predecir(
    x: pd.DataFrame,
    coeficientes: np.ndarray,
) -> np.ndarray:
    """Calcula las predicciones del modelo."""

    matriz = agregar_intercepto(
        x
    )

    return (
        matriz
        @ coeficientes
    )


# ============================================================
# 6. MÉTRICAS
# ============================================================

def calcular_metricas(
    reales: np.ndarray,
    predichos: np.ndarray,
) -> dict:
    """Calcula R², MAE, MSE y RMSE."""

    reales = np.asarray(
        reales,
        dtype=float,
    )

    predichos = np.asarray(
        predichos,
        dtype=float,
    )

    residuos = (
        reales
        - predichos
    )

    ss_res = float(
        np.sum(
            residuos ** 2
        )
    )

    ss_total = float(
        np.sum(
            (
                reales
                - reales.mean()
            ) ** 2
        )
    )

    r2 = (
        1
        - ss_res / ss_total
        if ss_total > 0
        else np.nan
    )

    mae = float(
        np.mean(
            np.abs(
                residuos
            )
        )
    )

    mse = float(
        np.mean(
            residuos ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "predicciones_negativas":
            int(
                np.sum(
                    predichos < 0
                )
            ),
        "predicciones_negativas_pct":
            (
                100
                * np.mean(
                    predichos < 0
                )
            ),
    }


def r2_ajustado(
    r2: float,
    n: int,
    p: int,
) -> float:
    """Calcula R² ajustado.

    n = observaciones.
    p = predictores, sin contar el intercepto.
    """

    if (
        not np.isfinite(r2)
        or n <= p + 1
    ):
        return np.nan

    return (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )


# ============================================================
# 7. EVALUACIÓN DE UN MODELO
# ============================================================

def evaluar_modelo(
    nombre: str,
    datos: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Ajusta y evalúa una especificación."""

    print(
        "\n"
        + "=" * 75
    )
    print(nombre.upper())
    print("=" * 75)

    # Solo valores de P47T conocidos.
    conocidos = datos.loc[
        datos["ingreso_conocido"]
    ].copy()

    entrenamiento, evaluacion = (
        dividir_por_hogares(
            conocidos
        )
    )

    print(
        f"Observaciones conocidas: {len(conocidos):,}"
    )
    print(
        f"Entrenamiento: {len(entrenamiento):,}"
    )
    print(
        f"Evaluación: {len(evaluacion):,}"
    )

    print("\nCodificación:")

    x_train, x_test = (
        construir_matrices(
            entrenamiento,
            evaluacion,
            numericas,
            categoricas,
        )
    )

    y_train = (
        entrenamiento["P47T"]
        .astype(float)
    )

    y_test = (
        evaluacion["P47T"]
        .astype(float)
    )

    coeficientes = (
        ajustar_regresion(
            x_train,
            y_train,
        )
    )

    pred_train = predecir(
        x_train,
        coeficientes,
    )

    pred_test = predecir(
        x_test,
        coeficientes,
    )

    metricas_train = (
        calcular_metricas(
            y_train.to_numpy(),
            pred_train,
        )
    )

    metricas_test = (
        calcular_metricas(
            y_test.to_numpy(),
            pred_test,
        )
    )

    cantidad_predictores = (
        x_train.shape[1]
    )

    ajustado = r2_ajustado(
        metricas_train["r2"],
        len(y_train),
        cantidad_predictores,
    )

    cumple = (
        metricas_test["r2"]
        > 0.30
    )

    print(
        f"\nPredictores después de codificar: "
        f"{cantidad_predictores}"
    )

    print(
        "\nRendimiento:"
    )

    print(
        f" - R² entrenamiento: "
        f"{metricas_train['r2']:.4f}"
    )

    print(
        f" - R² ajustado entrenamiento: "
        f"{ajustado:.4f}"
    )

    print(
        f" - R² evaluación: "
        f"{metricas_test['r2']:.4f}"
    )

    print(
        f" - MAE evaluación: "
        f"${metricas_test['mae']:,.0f}"
    )

    print(
        f" - MSE evaluación: "
        f"{metricas_test['mse']:,.0f}"
    )

    print(
        f" - RMSE evaluación: "
        f"${metricas_test['rmse']:,.0f}"
    )

    print(
        f" - Predicciones negativas: "
        f"{metricas_test['predicciones_negativas']} "
        f"({metricas_test['predicciones_negativas_pct']:.2f}%)"
    )

    print(
        " - Supera R² = 0,30:",
        "SÍ" if cumple else "NO",
    )

    # --------------------------------------------------------
    # Tabla resumen
    # --------------------------------------------------------

    resumen = {
        "modelo": nombre,

        "n_conocidos":
            len(conocidos),

        "n_entrenamiento":
            len(entrenamiento),

        "n_evaluacion":
            len(evaluacion),

        "n_predictores":
            cantidad_predictores,

        "r2_entrenamiento":
            metricas_train["r2"],

        "r2_ajustado_entrenamiento":
            ajustado,

        "r2_evaluacion":
            metricas_test["r2"],

        "mae_evaluacion":
            metricas_test["mae"],

        "mse_evaluacion":
            metricas_test["mse"],

        "rmse_evaluacion":
            metricas_test["rmse"],

        "predicciones_negativas":
            metricas_test[
                "predicciones_negativas"
            ],

        "predicciones_negativas_pct":
            metricas_test[
                "predicciones_negativas_pct"
            ],

        "supera_r2_030":
            cumple,
    }

    # --------------------------------------------------------
    # Coeficientes
    # --------------------------------------------------------

    nombres_coeficientes = (
        ["Intercepto"]
        + list(
            x_train.columns
        )
    )

    tabla_coeficientes = pd.DataFrame({
        "modelo": nombre,
        "variable":
            nombres_coeficientes,
        "coeficiente":
            coeficientes,
    })

    # --------------------------------------------------------
    # Predicciones de evaluación
    # --------------------------------------------------------

    tabla_predicciones = (
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

    tabla_predicciones[
        "modelo"
    ] = nombre

    tabla_predicciones[
        "prediccion"
    ] = pred_test

    tabla_predicciones[
        "error"
    ] = (
        tabla_predicciones["P47T"]
        - tabla_predicciones["prediccion"]
    )

    tabla_predicciones[
        "error_absoluto"
    ] = (
        tabla_predicciones[
            "error"
        ]
        .abs()
    )

    return (
        resumen,
        tabla_coeficientes,
        tabla_predicciones,
    )


# ============================================================
# 8. PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    RESULTADOS.mkdir(
        exist_ok=True
    )

    df = cargar_base()

    df = preparar_variables(
        df
    )

    # ========================================================
    # MODELO 1
    # General básico
    # ========================================================

    modelo_1 = {
        "nombre":
            "General básico",

        "datos":
            df,

        "numericas": [
            "edad_modelo",
        ],

        "categoricas": [
            "AGLOMERADO",
            "CH04",
            "NIVEL_ED",
            "ESTADO",
        ],
    }

    # ========================================================
    # MODELO 2
    # General ampliado
    # ========================================================

    modelo_2 = {
        "nombre":
            "General ampliado",

        "datos":
            df,

        "numericas": [
            "edad_modelo",
            "horas_modelo",
        ],

        "categoricas": [
            "AGLOMERADO",
            "CH04",
            "NIVEL_ED",
            "ESTADO",
            "CAT_OCUP",
            "rama_modelo",
            "situacion_horas",
        ],
    }

    # ========================================================
    # MODELO 3
    # Ocupados básico
    # ========================================================

    ocupados = df.loc[
        df["ESTADO"] == 1
    ].copy()

    modelo_3 = {
        "nombre":
            "Ocupados básico",

        "datos":
            ocupados,

        "numericas": [
            "edad_modelo",
        ],

        "categoricas": [
            "AGLOMERADO",
            "CH04",
            "NIVEL_ED",
            "CAT_OCUP",
        ],
    }

    # ========================================================
    # MODELO 4
    # Ocupados ampliado
    # ========================================================

    modelo_4 = {
        "nombre":
            "Ocupados ampliado",

        "datos":
            ocupados,

        "numericas": [
            "edad_modelo",
            "horas_modelo",
        ],

        "categoricas": [
            "AGLOMERADO",
            "CH04",
            "NIVEL_ED",
            "CAT_OCUP",
            "rama_modelo",
            "situacion_horas",
        ],
    }

    modelos = [
        modelo_1,
        modelo_2,
        modelo_3,
        modelo_4,
    ]

    resumenes = []
    coeficientes = []
    predicciones = []

    for modelo in modelos:

        (
            resumen,
            tabla_coef,
            tabla_pred,
        ) = evaluar_modelo(
            nombre=modelo["nombre"],
            datos=modelo["datos"],
            numericas=modelo["numericas"],
            categoricas=modelo["categoricas"],
        )

        resumenes.append(
            resumen
        )

        coeficientes.append(
            tabla_coef
        )

        predicciones.append(
            tabla_pred
        )

    tabla_resumen = (
        pd.DataFrame(
            resumenes
        )
        .sort_values(
            "r2_evaluacion",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    tabla_coeficientes = (
        pd.concat(
            coeficientes,
            ignore_index=True,
        )
    )

    tabla_predicciones = (
        pd.concat(
            predicciones,
            ignore_index=True,
        )
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    tabla_resumen.to_csv(
        RESULTADOS
        / "objetivo4_comparacion_regresiones.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_coeficientes.to_csv(
        RESULTADOS
        / "objetivo4_coeficientes_prueba1.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_predicciones.to_csv(
        RESULTADOS
        / "objetivo4_predicciones_evaluacion_prueba1.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 75
    )
    print(
        "COMPARACIÓN FINAL"
    )
    print(
        "=" * 75
    )

    columnas = [
        "modelo",
        "n_entrenamiento",
        "n_evaluacion",
        "n_predictores",
        "r2_entrenamiento",
        "r2_ajustado_entrenamiento",
        "r2_evaluacion",
        "mae_evaluacion",
        "rmse_evaluacion",
        "predicciones_negativas_pct",
        "supera_r2_030",
    ]

    print(
        tabla_resumen[
            columnas
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.4f}",
        )
    )

    print(
        "\nArchivos generados:"
    )

    print(
        " - objetivo4_comparacion_regresiones.csv"
    )

    print(
        " - objetivo4_coeficientes_prueba1.csv"
    )

    print(
        " - objetivo4_predicciones_evaluacion_prueba1.csv"
    )

    print(
        "\nTodavía NO se imputaron "
        "los 1.509 ingresos faltantes."
    )


if __name__ == "__main__":
    main()