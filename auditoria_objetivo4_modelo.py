"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 4 - ETAPA 1
Auditoría previa al modelo de imputación de ingresos.

Esta etapa NO ajusta una regresión y NO imputa ningún ingreso.

Objetivos:
1. Revisar el universo del segundo trimestre de 2026.
2. Separar P47T conocido, cero, positivo y no respuesta.
3. Analizar en qué situaciones laborales aparece la no respuesta.
4. Revisar la disponibilidad de posibles variables explicativas.
5. Auditar horas y códigos laborales antes de utilizarlos en un modelo.

Se trabaja inicialmente con 2026 T2 para modelar P47T en su escala
original, evitando que la inflación entre años distorsione esta primera
evaluación del modelo.
"""

from pathlib import Path

import numpy as np
import pandas as pd


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

# Variables que nos interesa revisar.
# El programa comprobará cuáles existen realmente antes de leerlas.
CANDIDATAS = [
    "CODUSU",
    "NRO_HOGAR",
    "COMPONENTE",

    "AGLOMERADO",
    "H15",

    # Características personales
    "CH04",       # sexo
    "CH06",       # edad
    "NIVEL_ED",   # nivel educativo

    # Situación laboral
    "ESTADO",
    "CAT_OCUP",

    # Características del empleo
    "PP04B_COD",
    "PP04D_COD",
    "PP3E_TOT",

    # Variable objetivo
    "P47T",

    # Ponderadores, solo para auditoría.
    "PONDERA",
    "PONDII",
]


# ============================================================
# LECTURA
# ============================================================

def obtener_columnas() -> list[str]:
    """Lee solo el encabezado para conocer las variables disponibles."""

    if not ARCHIVO.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ARCHIVO}"
        )

    encabezado = pd.read_csv(
        ARCHIVO,
        sep=";",
        encoding="latin-1",
        nrows=0,
    )

    return [
        columna.strip().upper()
        for columna in encabezado.columns
    ]


def cargar_base() -> tuple[pd.DataFrame, list[str]]:
    """Carga únicamente las variables candidatas que existen."""

    disponibles = obtener_columnas()

    presentes = [
        variable
        for variable in CANDIDATAS
        if variable in disponibles
    ]

    ausentes = [
        variable
        for variable in CANDIDATAS
        if variable not in disponibles
    ]

    print("\nVariables candidatas presentes:")
    for variable in presentes:
        print(" -", variable)

    if ausentes:
        print("\nVariables candidatas que NO aparecen en la base:")
        for variable in ausentes:
            print(" -", variable)

    # Conservamos códigos laborales como texto para no perder ceros.
    tipos = {}

    for variable in [
        "CODUSU",
        "PP04B_COD",
        "PP04D_COD",
    ]:
        if variable in presentes:
            tipos[variable] = "string"

    df = pd.read_csv(
        ARCHIVO,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=presentes,
        dtype=tipos,
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    # Convertimos las variables no textuales a números.
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

    return df, ausentes


# ============================================================
# CLASIFICACIÓN DEL INGRESO
# ============================================================

def clasificar_p47t(df: pd.DataFrame) -> pd.DataFrame:
    """Distingue respuestas válidas de no respuesta."""

    if "P47T" not in df.columns:
        raise ValueError(
            "La base no contiene P47T."
        )

    datos = df.copy()

    ingresos = datos["P47T"]

    # La única codificación negativa esperada es -9.
    negativos_inesperados = (
        ingresos.notna()
        & (ingresos < 0)
        & ingresos.ne(-9)
    )

    if negativos_inesperados.any():
        valores = sorted(
            ingresos.loc[
                negativos_inesperados
            ].unique()
        )

        raise ValueError(
            "Se encontraron códigos negativos "
            f"inesperados en P47T: {valores}"
        )

    condiciones = [
        ingresos.eq(-9) | ingresos.isna(),
        ingresos.eq(0),
        ingresos.gt(0),
    ]

    etiquetas = [
        "No respuesta",
        "Cero",
        "Positivo",
    ]

    datos["situacion_p47t"] = np.select(
        condiciones,
        etiquetas,
        default="Revisar",
    )

    datos["p47t_conocido"] = (
        ingresos.notna()
        & ingresos.ge(0)
    )

    return datos


# ============================================================
# RESÚMENES
# ============================================================

def mostrar_resumen_general(
    df: pd.DataFrame,
) -> None:
    """Muestra el tamaño del universo y la situación de P47T."""

    print("\n" + "=" * 70)
    print("UNIVERSO DEL SEGUNDO TRIMESTRE DE 2026")
    print("=" * 70)

    print(
        f"Registros de CABA y Partidos del GBA: {len(df):,}"
    )

    if "H15" in df.columns:

        print("\nH15:")
        print(
            df["H15"]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        entrevistas = df.loc[
            df["H15"] == 1
        ].copy()

    else:
        entrevistas = df.copy()

    print(
        "\nRegistros en el universo inicial "
        f"de entrevistas realizadas: {len(entrevistas):,}"
    )

    print("\nSituación de P47T:")

    resumen = (
        entrevistas[
            "situacion_p47t"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(resumen)

    porcentajes = (
        resumen
        / len(entrevistas)
        * 100
    )

    print("\nPorcentajes muestrales:")

    for categoria, valor in porcentajes.items():
        print(
            f" - {categoria}: {valor:.2f}%"
        )


def resumen_por_aglomerado(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Resume conocidos y faltantes por aglomerado."""

    universo = df.copy()

    if "H15" in universo.columns:
        universo = universo.loc[
            universo["H15"] == 1
        ]

    filas = []

    for codigo, nombre in AGLOMERADOS.items():

        parte = universo.loc[
            universo["AGLOMERADO"]
            == codigo
        ]

        filas.append({
            "aglomerado": nombre,
            "n_entrevistas": len(parte),

            "n_p47t_conocido":
                int(
                    parte[
                        "p47t_conocido"
                    ].sum()
                ),

            "n_p47t_cero":
                int(
                    parte[
                        "situacion_p47t"
                    ].eq("Cero")
                    .sum()
                ),

            "n_p47t_positivo":
                int(
                    parte[
                        "situacion_p47t"
                    ].eq("Positivo")
                    .sum()
                ),

            "n_p47t_no_respuesta":
                int(
                    parte[
                        "situacion_p47t"
                    ].eq("No respuesta")
                    .sum()
                ),
        })

    return pd.DataFrame(filas)


def resumen_por_estado(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analiza dónde aparece la no respuesta según ESTADO."""

    if "ESTADO" not in df.columns:
        return pd.DataFrame()

    universo = df.copy()

    if "H15" in universo.columns:
        universo = universo.loc[
            universo["H15"] == 1
        ]

    filas = []

    for estado, parte in universo.groupby(
        "ESTADO",
        dropna=False,
    ):

        n = len(parte)

        no_respuesta = int(
            parte[
                "situacion_p47t"
            ]
            .eq("No respuesta")
            .sum()
        )

        filas.append({
            "estado": estado,
            "n": n,
            "n_no_respuesta": no_respuesta,
            "no_respuesta_pct":
                100 * no_respuesta / n
                if n > 0
                else np.nan,
            "n_cero":
                int(
                    parte[
                        "situacion_p47t"
                    ]
                    .eq("Cero")
                    .sum()
                ),
            "n_positivo":
                int(
                    parte[
                        "situacion_p47t"
                    ]
                    .eq("Positivo")
                    .sum()
                ),
        })

    return pd.DataFrame(filas)


# ============================================================
# DISPONIBILIDAD DE PREDICTORES
# ============================================================

def auditar_predictores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compara cobertura entre ingresos conocidos y no respondidos."""

    universo = df.copy()

    if "H15" in universo.columns:
        universo = universo.loc[
            universo["H15"] == 1
        ]

    predictores = [
        "AGLOMERADO",
        "CH04",
        "CH06",
        "NIVEL_ED",
        "ESTADO",
        "CAT_OCUP",
        "PP04B_COD",
        "PP04D_COD",
        "PP3E_TOT",
    ]

    filas = []

    for variable in predictores:

        if variable not in universo.columns:
            continue

        for grupo, parte in [
            (
                "Ingreso conocido",
                universo.loc[
                    universo[
                        "p47t_conocido"
                    ]
                ],
            ),
            (
                "No respuesta P47T",
                universo.loc[
                    universo[
                        "situacion_p47t"
                    ].eq("No respuesta")
                ],
            ),
        ]:

            total = len(parte)

            faltantes = int(
                parte[variable]
                .isna()
                .sum()
            )

            filas.append({
                "variable": variable,
                "grupo": grupo,
                "n": total,
                "n_ausentes": faltantes,
                "ausentes_pct":
                    100 * faltantes / total
                    if total > 0
                    else np.nan,
                "n_valores_distintos":
                    int(
                        parte[variable]
                        .nunique(
                            dropna=True
                        )
                    ),
            })

    return pd.DataFrame(filas)


# ============================================================
# HORAS Y CÓDIGOS LABORALES
# ============================================================

def auditar_variables_laborales(
    df: pd.DataFrame,
) -> None:
    """Revisa variables que solo tienen sentido para personas ocupadas."""

    if "ESTADO" not in df.columns:
        return

    ocupados = df.loc[
        df["ESTADO"] == 1
    ].copy()

    print("\n" + "=" * 70)
    print("AUDITORÍA DE VARIABLES LABORALES ENTRE OCUPADOS")
    print("=" * 70)

    print(
        f"Ocupados en los dos aglomerados: {len(ocupados):,}"
    )

    if "PP3E_TOT" in ocupados.columns:

        horas = ocupados["PP3E_TOT"]

        print("\nPP3E_TOT:")
        print(
            " - Ausentes:",
            int(horas.isna().sum())
        )
        print(
            " - Iguales a 0:",
            int(horas.eq(0).sum())
        )
        print(
            " - Entre 1 y 168:",
            int(
                horas.between(
                    1, 168
                ).sum()
            )
        )
        print(
            " - Mayores a 168:",
            int(
                horas.gt(168).sum()
            )
        )
        print(
            " - Iguales a 999:",
            int(
                horas.eq(999).sum()
            )
        )

    for variable in [
        "PP04B_COD",
        "PP04D_COD",
        "CAT_OCUP",
    ]:

        if variable not in ocupados.columns:
            continue

        serie = ocupados[variable]

        print(f"\n{variable}:")
        print(
            " - Ausentes:",
            int(
                serie.isna().sum()
            )
        )
        print(
            " - Valores distintos:",
            int(
                serie.nunique(
                    dropna=True
                )
            )
        )

        if variable in {
            "PP04B_COD",
            "PP04D_COD",
        }:

            texto = (
                serie
                .astype("string")
                .str.strip()
            )

            print(
                " - Código 9999:",
                int(
                    texto.eq("9999")
                    .sum()
                )
            )


# ============================================================
# DUPLICADOS DEL TRIMESTRE
# ============================================================

def auditar_identificadores(
    df: pd.DataFrame,
) -> None:
    """Comprueba duplicados por persona dentro del trimestre."""

    claves = [
        "CODUSU",
        "NRO_HOGAR",
        "COMPONENTE",
    ]

    if not all(
        clave in df.columns
        for clave in claves
    ):
        return

    duplicados = df.duplicated(
        claves,
        keep=False,
    )

    print("\n" + "=" * 70)
    print("IDENTIFICADORES")
    print("=" * 70)

    print(
        "Filas involucradas en claves duplicadas "
        f"dentro del trimestre: {int(duplicados.sum())}"
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    RESULTADOS.mkdir(
        exist_ok=True
    )

    df, ausentes = cargar_base()

    # Nos quedamos únicamente con los dos aglomerados elegidos.
    df = df.loc[
        df["AGLOMERADO"].isin(
            AGLOMERADOS
        )
    ].copy()

    df = clasificar_p47t(df)

    mostrar_resumen_general(df)

    tabla_aglomerados = (
        resumen_por_aglomerado(df)
    )

    tabla_estados = (
        resumen_por_estado(df)
    )

    tabla_predictores = (
        auditar_predictores(df)
    )

    auditar_variables_laborales(df)
    auditar_identificadores(df)

    print("\n" + "=" * 70)
    print("P47T POR AGLOMERADO")
    print("=" * 70)

    print(
        tabla_aglomerados.to_string(
            index=False
        )
    )

    if not tabla_estados.empty:

        print("\n" + "=" * 70)
        print("P47T POR ESTADO LABORAL")
        print("=" * 70)

        print(
            tabla_estados.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

    print("\n" + "=" * 70)
    print("COBERTURA DE VARIABLES CANDIDATAS")
    print("=" * 70)

    print(
        tabla_predictores.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # Guardamos auditorías sin tocar resultados anteriores.
    tabla_aglomerados.to_csv(
        RESULTADOS
        / "objetivo4_auditoria_p47t_aglomerado.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_estados.to_csv(
        RESULTADOS
        / "objetivo4_auditoria_p47t_estado.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    tabla_predictores.to_csv(
        RESULTADOS
        / "objetivo4_auditoria_predictores.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nLISTO: auditoría previa al modelo terminada."
    )

    print(
        "\nTodavía no se ajustó ninguna regresión "
        "ni se imputó ningún ingreso."
    )


if __name__ == "__main__":
    main()