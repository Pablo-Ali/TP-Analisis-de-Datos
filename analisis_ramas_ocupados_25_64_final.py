"""TP Introducción al Análisis de Datos - EPH 2016-2026.

ANÁLISIS FINAL DE RAMAS DE ACTIVIDAD.

Compara CABA y Partidos del GBA para:
- 2016 T2
- 2026 T2

Universo:
- Personas ocupadas (ESTADO=1)
- Entre 25 y 64 años

Analiza:
1. Composición del empleo por rama.
2. Ingreso real mensual de la ocupación principal (P21).
3. Brecha de ingresos entre CABA y GBA.
4. Evolución real de ingresos entre 2016 y 2026.
5. Horas semanales observadas.
6. Ingreso horario real aproximado.

No modifica scripts ni TXT anteriores.
No imputa ingresos.
No realiza regresión.

Reutiliza la clasificación de ramas ya validada en:
analisis_eph_objetivo2_etapa3_corregido.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS,
    RESULTADOS,
    buscar_archivos,
    guardar_csv,
)

from analisis_eph_objetivo2_etapa3_corregido import (
    RAMAS,
    clasificar_rama,
    cargar_factores,
    cuantil_ponderado,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PERIODOS = {
    (2016, 2),
    (2026, 2),
}

MINIMO_MUESTRAL = 30

SEMANAS_POR_MES = 52 / 12
MAX_HORAS_SEMANA = 168

RAMAS_RESIDUALES = {
    "Otras ramas",
    "Sin clasificar",
}

REQUERIDAS = {
    "ANO4",
    "TRIMESTRE",
    "AGLOMERADO",
    "CH06",
    "ESTADO",
    "PP04B_COD",
    "P21",
    "PONDERA",
    "PONDIIO",
    "PP3E_TOT",
}


# ============================================================
# 1. LECTURA DE LOS MICRODATOS
# ============================================================

def leer_base(
    ruta,
    anio: int,
    trimestre: int,
) -> pd.DataFrame:
    """Lee solamente las variables necesarias.

    PP04B_COD se conserva como texto para no perder posibles
    ceros iniciales de los códigos de actividad.
    """

    df = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        dtype={"PP04B_COD": "string"},
        usecols=lambda col:
            col.strip().upper() in REQUERIDAS,
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    faltantes = REQUERIDAS - set(df.columns)

    if faltantes:
        raise ValueError(
            f"{ruta.name}: faltan columnas "
            f"{sorted(faltantes)}"
        )

    for columna in REQUERIDAS - {"PP04B_COD"}:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    df["PP04B_COD"] = (
        df["PP04B_COD"]
        .astype("string")
        .str.strip()
    )

    if set(df["ANO4"].dropna()) != {anio}:
        raise ValueError(
            f"{ruta.name}: año interno incorrecto"
        )

    if set(df["TRIMESTRE"].dropna()) != {trimestre}:
        raise ValueError(
            f"{ruta.name}: trimestre interno incorrecto"
        )

    df = df.loc[
        df["AGLOMERADO"].isin(AGLOMERADOS)
    ].copy()

    if df.empty:
        raise ValueError(
            f"{ruta.name}: no contiene los aglomerados elegidos"
        )

    if (
        df["PONDERA"].isna()
        | (df["PONDERA"] <= 0)
    ).any():
        raise ValueError(
            f"{ruta.name}: PONDERA inválido"
        )

    if (
        df["PONDIIO"].notna()
        & (df["PONDIIO"] < 0)
    ).any():
        raise ValueError(
            f"{ruta.name}: PONDIIO negativo"
        )

    return df


# ============================================================
# 2. CÁLCULO DE UNA RAMA
# ============================================================

def analizar_rama(
    parte: pd.DataFrame,
    periodo: str,
    codigo_aglomerado: int,
    rama: str,
    factor_ipc: float,
    total_poblacional: float,
) -> dict:
    """Calcula composición, ingresos y horas de una rama."""

    nombre_aglomerado = AGLOMERADOS[
        codigo_aglomerado
    ]

    # --------------------------------------------------------
    # Participación de la rama
    # --------------------------------------------------------

    peso_rama = float(
        parte["PONDERA"].sum()
    )

    porcentaje_rama = (
        100 * peso_rama / total_poblacional
        if total_poblacional > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Ingreso mensual P21
    # --------------------------------------------------------

    ingreso = parte["P21"]

    negativos_inesperados = (
        ingreso.notna()
        & ingreso.lt(0)
        & ingreso.ne(-9)
    )

    if negativos_inesperados.any():
        raise ValueError(
            f"P21 negativo inesperado: "
            f"{periodo}, {nombre_aglomerado}, {rama}"
        )

    no_respuesta = (
        ingreso.eq(-9)
        | ingreso.isna()
    )

    positivos = ingreso.gt(0)

    ingreso_valido = (
        positivos
        & parte["PONDIIO"].notna()
        & parte["PONDIIO"].gt(0)
    )

    n_ingresos_validos = int(
        ingreso_valido.sum()
    )

    ingreso_real = (
        ingreso.loc[ingreso_valido]
        * factor_ipc
    )

    pesos_ingreso = parte.loc[
        ingreso_valido,
        "PONDIIO",
    ]

    mediana_ingreso = cuantil_ponderado(
        ingreso_real,
        pesos_ingreso,
        0.50,
    )

    # No respuesta descriptiva utilizando PONDERA.
    porcentaje_no_respuesta = (
        100
        * float(
            parte.loc[
                no_respuesta,
                "PONDERA",
            ].sum()
        )
        / peso_rama
        if peso_rama > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Horas semanales
    # --------------------------------------------------------

    horas = parte["PP3E_TOT"]

    horas_validas = (
        horas.gt(0)
        & horas.le(MAX_HORAS_SEMANA)
    )

    mediana_horas = cuantil_ponderado(
        horas.loc[horas_validas],
        parte.loc[
            horas_validas,
            "PONDERA",
        ],
        0.50,
    )

    # --------------------------------------------------------
    # Ingreso horario aproximado
    # --------------------------------------------------------

    pares_validos = (
        ingreso_valido
        & horas_validas
    )

    n_pares_validos = int(
        pares_validos.sum()
    )

    ingreso_horario = (
        ingreso.loc[pares_validos]
        * factor_ipc
        / (
            horas.loc[pares_validos]
            * SEMANAS_POR_MES
        )
    )

    pesos_pares = parte.loc[
        pares_validos,
        "PONDIIO",
    ]

    mediana_horaria = cuantil_ponderado(
        ingreso_horario,
        pesos_pares,
        0.50,
    )

    peso_ingresos = float(
        parte.loc[
            ingreso_valido,
            "PONDIIO",
        ].sum()
    )

    peso_pares = float(
        pesos_pares.sum()
    )

    cobertura_horas_ponderada = (
        100 * peso_pares / peso_ingresos
        if peso_ingresos > 0
        else np.nan
    )

    return {
        "periodo": periodo,
        "codigo_aglomerado": codigo_aglomerado,
        "aglomerado": nombre_aglomerado,
        "rama": rama,

        # Composición
        "n_ocupados_muestrales":
            int(len(parte)),

        "poblacion_ocupada_ponderada":
            peso_rama,

        "porcentaje_ocupados_ponderado":
            porcentaje_rama,

        # P21
        "n_p21_cero":
            int(ingreso.eq(0).sum()),

        "n_p21_positivo":
            int(positivos.sum()),

        "n_p21_no_responde":
            int(no_respuesta.sum()),

        "n_ingresos_positivos_validos":
            n_ingresos_validos,

        "n_p21_positivo_sin_pondiio":
            int(
                positivos.sum()
                - n_ingresos_validos
            ),

        "no_respuesta_p21_pct_pondera":
            porcentaje_no_respuesta,

        "ingreso_real_mediana_2026_t2":
            mediana_ingreso,

        # Horas
        "n_horas_validas":
            int(horas_validas.sum()),

        "n_horas_cero":
            int(horas.eq(0).sum()),

        "n_horas_mayores_168":
            int(
                horas.gt(
                    MAX_HORAS_SEMANA
                ).sum()
            ),

        "n_horas_ausentes":
            int(horas.isna().sum()),

        "horas_semanales_mediana_pondera":
            mediana_horas,

        # Ingreso horario
        "n_pares_horas_ingreso_validos":
            n_pares_validos,

        "cobertura_horas_ponderada_pct":
            cobertura_horas_ponderada,

        "ingreso_horario_real_mediana_estimada":
            mediana_horaria,

        # Advertencias
        "advertencia_ingreso_muestra_pequena":
            n_ingresos_validos
            < MINIMO_MUESTRAL,

        "advertencia_horario_muestra_pequena":
            n_pares_validos
            < MINIMO_MUESTRAL,
    }


# ============================================================
# 3. AUDITORÍA DEL UNIVERSO
# ============================================================

def auditar_universo(
    ocupados: pd.DataFrame,
    periodo: str,
    codigo_aglomerado: int,
) -> dict:
    """Resume el universo de ocupados 25-64."""

    sin_clasificar = (
        ocupados["rama"]
        == "Sin clasificar"
    )

    total = float(
        ocupados["PONDERA"].sum()
    )

    return {
        "periodo": periodo,
        "aglomerado":
            AGLOMERADOS[codigo_aglomerado],

        "n_ocupados_25_64":
            int(len(ocupados)),

        "poblacion_ocupada_25_64_ponderada":
            total,

        "n_sin_clasificar":
            int(sin_clasificar.sum()),

        "porcentaje_sin_clasificar":
            (
                100
                * float(
                    ocupados.loc[
                        sin_clasificar,
                        "PONDERA",
                    ].sum()
                )
                / total
                if total > 0
                else np.nan
            ),

        "n_horas_mayores_168":
            int(
                ocupados[
                    "PP3E_TOT"
                ].gt(
                    MAX_HORAS_SEMANA
                ).sum()
            ),
    }


# ============================================================
# 4. BRECHA CABA - GBA
# ============================================================

def calcular_brechas(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """Compara la misma rama entre CABA y GBA."""

    campos = [
        "periodo",
        "rama",
        "porcentaje_ocupados_ponderado",
        "n_ingresos_positivos_validos",
        "ingreso_real_mediana_2026_t2",
        "horas_semanales_mediana_pondera",
        "n_pares_horas_ingreso_validos",
        "ingreso_horario_real_mediana_estimada",
    ]

    caba = tabla.loc[
        tabla["aglomerado"] == "CABA",
        campos,
    ].rename(columns={
        "porcentaje_ocupados_ponderado":
            "participacion_caba",

        "n_ingresos_positivos_validos":
            "n_ingresos_caba",

        "ingreso_real_mediana_2026_t2":
            "ingreso_caba",

        "horas_semanales_mediana_pondera":
            "horas_caba",

        "n_pares_horas_ingreso_validos":
            "n_pares_caba",

        "ingreso_horario_real_mediana_estimada":
            "ingreso_horario_caba",
    })

    gba = tabla.loc[
        tabla["aglomerado"]
        == "Partidos del GBA",
        campos,
    ].rename(columns={
        "porcentaje_ocupados_ponderado":
            "participacion_gba",

        "n_ingresos_positivos_validos":
            "n_ingresos_gba",

        "ingreso_real_mediana_2026_t2":
            "ingreso_gba",

        "horas_semanales_mediana_pondera":
            "horas_gba",

        "n_pares_horas_ingreso_validos":
            "n_pares_gba",

        "ingreso_horario_real_mediana_estimada":
            "ingreso_horario_gba",
    })

    resultado = caba.merge(
        gba,
        on=["periodo", "rama"],
        validate="one_to_one",
    )

    resultado[
        "diferencia_participacion_pp"
    ] = (
        resultado["participacion_caba"]
        - resultado["participacion_gba"]
    )

    resultado[
        "brecha_ingreso_caba_gba_pct"
    ] = np.where(
        resultado["ingreso_gba"] > 0,

        (
            resultado["ingreso_caba"]
            / resultado["ingreso_gba"]
            - 1
        ) * 100,

        np.nan,
    )

    resultado[
        "brecha_ingreso_horario_caba_gba_pct"
    ] = np.where(
        resultado["ingreso_horario_gba"] > 0,

        (
            resultado[
                "ingreso_horario_caba"
            ]
            / resultado[
                "ingreso_horario_gba"
            ]
            - 1
        ) * 100,

        np.nan,
    )

    resultado[
        "advertencia_ingreso_muestra_pequena"
    ] = (
        (resultado["n_ingresos_caba"]
         < MINIMO_MUESTRAL)
        |
        (resultado["n_ingresos_gba"]
         < MINIMO_MUESTRAL)
    )

    resultado[
        "advertencia_horario_muestra_pequena"
    ] = (
        (resultado["n_pares_caba"]
         < MINIMO_MUESTRAL)
        |
        (resultado["n_pares_gba"]
         < MINIMO_MUESTRAL)
    )

    return resultado.sort_values(
        ["periodo", "rama"]
    )


# ============================================================
# 5. EVOLUCIÓN 2016 - 2026
# ============================================================

def calcular_evolucion(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """Compara cada rama consigo misma entre los extremos."""

    campos = [
        "aglomerado",
        "rama",
        "porcentaje_ocupados_ponderado",
        "n_ingresos_positivos_validos",
        "ingreso_real_mediana_2026_t2",
        "horas_semanales_mediana_pondera",
        "n_pares_horas_ingreso_validos",
        "ingreso_horario_real_mediana_estimada",
    ]

    inicial = tabla.loc[
        tabla["periodo"] == "2016-T2",
        campos,
    ].rename(columns={
        "porcentaje_ocupados_ponderado":
            "participacion_2016",

        "n_ingresos_positivos_validos":
            "n_ingresos_2016",

        "ingreso_real_mediana_2026_t2":
            "ingreso_2016",

        "horas_semanales_mediana_pondera":
            "horas_2016",

        "n_pares_horas_ingreso_validos":
            "n_pares_2016",

        "ingreso_horario_real_mediana_estimada":
            "ingreso_horario_2016",
    })

    final = tabla.loc[
        tabla["periodo"] == "2026-T2",
        campos,
    ].rename(columns={
        "porcentaje_ocupados_ponderado":
            "participacion_2026",

        "n_ingresos_positivos_validos":
            "n_ingresos_2026",

        "ingreso_real_mediana_2026_t2":
            "ingreso_2026",

        "horas_semanales_mediana_pondera":
            "horas_2026",

        "n_pares_horas_ingreso_validos":
            "n_pares_2026",

        "ingreso_horario_real_mediana_estimada":
            "ingreso_horario_2026",
    })

    resultado = inicial.merge(
        final,
        on=["aglomerado", "rama"],
        validate="one_to_one",
    )

    resultado[
        "cambio_participacion_pp"
    ] = (
        resultado["participacion_2026"]
        - resultado["participacion_2016"]
    )

    resultado[
        "variacion_ingreso_real_pct"
    ] = np.where(
        resultado["ingreso_2016"] > 0,

        (
            resultado["ingreso_2026"]
            / resultado["ingreso_2016"]
            - 1
        ) * 100,

        np.nan,
    )

    resultado[
        "variacion_horas_pct"
    ] = np.where(
        resultado["horas_2016"] > 0,

        (
            resultado["horas_2026"]
            / resultado["horas_2016"]
            - 1
        ) * 100,

        np.nan,
    )

    resultado[
        "variacion_ingreso_horario_real_pct"
    ] = np.where(
        resultado["ingreso_horario_2016"] > 0,

        (
            resultado["ingreso_horario_2026"]
            / resultado["ingreso_horario_2016"]
            - 1
        ) * 100,

        np.nan,
    )

    resultado[
        "advertencia_ingreso_muestra_pequena"
    ] = (
        (resultado["n_ingresos_2016"]
         < MINIMO_MUESTRAL)
        |
        (resultado["n_ingresos_2026"]
         < MINIMO_MUESTRAL)
    )

    resultado[
        "advertencia_horario_muestra_pequena"
    ] = (
        (resultado["n_pares_2016"]
         < MINIMO_MUESTRAL)
        |
        (resultado["n_pares_2026"]
         < MINIMO_MUESTRAL)
    )

    return resultado.sort_values(
        ["aglomerado", "rama"]
    )


# ============================================================
# 6. SELECCIÓN TRANSPARENTE PARA GRÁFICOS
# ============================================================

def preparar_ramas_para_graficos(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """Clasifica qué ramas tienen muestras suficientes.

    No elimina información del análisis.
    Solo prepara una selección para evitar gráficos
    basados en celdas demasiado pequeñas.
    """

    base = tabla.loc[
        ~tabla["rama"].isin(
            RAMAS_RESIDUALES
        )
    ].copy()

    resumen = (
        base.groupby(
            "rama",
            as_index=False,
        )
        .agg(
            min_n_ocupados=(
                "n_ocupados_muestrales",
                "min",
            ),

            min_n_ingresos_validos=(
                "n_ingresos_positivos_validos",
                "min",
            ),

            min_n_pares_horarios=(
                "n_pares_horas_ingreso_validos",
                "min",
            ),
        )
    )

    poblacion_2026 = (
        base.loc[
            base["periodo"]
            == "2026-T2"
        ]
        .groupby(
            "rama",
            as_index=False,
        )[
            "poblacion_ocupada_ponderada"
        ]
        .sum()
        .rename(columns={
            "poblacion_ocupada_ponderada":
                "poblacion_ponderada_2026_ambos_aglomerados"
        })
    )

    resumen = resumen.merge(
        poblacion_2026,
        on="rama",
        how="left",
        validate="one_to_one",
    )

    resumen[
        "apta_ingreso_mensual"
    ] = (
        resumen[
            "min_n_ingresos_validos"
        ] >= MINIMO_MUESTRAL
    )

    resumen[
        "apta_ingreso_horario"
    ] = (
        resumen[
            "min_n_pares_horarios"
        ] >= MINIMO_MUESTRAL
    )

    resumen = resumen.sort_values(
        "poblacion_ponderada_2026_ambos_aglomerados",
        ascending=False,
    ).reset_index(drop=True)

    resumen[
        "orden_2026"
    ] = np.arange(
        1,
        len(resumen) + 1,
    )

    return resumen


# ============================================================
# 7. PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    RESULTADOS.mkdir(exist_ok=True)

    factores = cargar_factores()

    archivos = [
        (anio, trimestre, ruta)
        for anio, trimestre, ruta
        in buscar_archivos()
        if (anio, trimestre)
        in PERIODOS
    ]

    if len(archivos) != 2:
        raise ValueError(
            "Se esperaban exactamente las bases "
            "2016 T2 y 2026 T2."
        )

    filas = []
    auditorias = []

    for anio, trimestre, ruta in archivos:

        periodo = (
            f"{anio}-T{trimestre}"
        )

        print(
            f"\nProcesando {periodo}: "
            f"{ruta.name}"
        )

        df = leer_base(
            ruta,
            anio,
            trimestre,
        )

        # Universo común con el análisis educativo.
        df = df.loc[
            df["ESTADO"].eq(1)
            & df["CH06"].between(25, 64)
        ].copy()

        df["rama"] = (
            df["PP04B_COD"]
            .map(clasificar_rama)
        )

        for codigo, nombre in AGLOMERADOS.items():

            ocupados = df.loc[
                df["AGLOMERADO"]
                == codigo
            ].copy()

            if ocupados.empty:
                raise ValueError(
                    f"Sin ocupados de 25-64 "
                    f"en {periodo}, {nombre}"
                )

            total = float(
                ocupados[
                    "PONDERA"
                ].sum()
            )

            factor = factores.get(
                (periodo, nombre)
            )

            if factor is None:
                raise ValueError(
                    f"Falta factor IPC: "
                    f"{periodo}, {nombre}"
                )

            auditorias.append(
                auditar_universo(
                    ocupados,
                    periodo,
                    codigo,
                )
            )

            for rama in RAMAS:

                parte = ocupados.loc[
                    ocupados["rama"]
                    == rama
                ]

                filas.append(
                    analizar_rama(
                        parte=parte,
                        periodo=periodo,
                        codigo_aglomerado=codigo,
                        rama=rama,
                        factor_ipc=factor,
                        total_poblacional=total,
                    )
                )

    tabla = pd.DataFrame(filas)

    esperado = (
        len(PERIODOS)
        * len(AGLOMERADOS)
        * len(RAMAS)
    )

    if len(tabla) != esperado:
        raise ValueError(
            f"Se esperaban {esperado} filas "
            f"y se obtuvieron {len(tabla)}"
        )

    if tabla.duplicated(
        [
            "periodo",
            "aglomerado",
            "rama",
        ]
    ).any():
        raise ValueError(
            "Hay filas duplicadas"
        )

    # Las ramas, incluidas las residuales,
    # deben sumar 100%.
    suma = tabla.groupby(
        [
            "periodo",
            "aglomerado",
        ]
    )[
        "porcentaje_ocupados_ponderado"
    ].sum()

    if not np.allclose(
        suma.to_numpy(),
        100,
    ):
        raise ValueError(
            "Las ramas no suman 100%"
        )

    brechas = calcular_brechas(
        tabla
    )

    evolucion = calcular_evolucion(
        tabla
    )

    ramas_graficos = (
        preparar_ramas_para_graficos(
            tabla
        )
    )

    auditoria = pd.DataFrame(
        auditorias
    )

    guardar_csv(
        tabla,
        "ramas_ocupados_25_64_extremos.csv",
    )

    guardar_csv(
        brechas,
        "brechas_ramas_ocupados_25_64_extremos.csv",
    )

    guardar_csv(
        evolucion,
        "evolucion_ramas_ocupados_25_64_extremos.csv",
    )

    guardar_csv(
        auditoria,
        "auditoria_ramas_ocupados_25_64_extremos.csv",
    )

    guardar_csv(
        ramas_graficos,
        "seleccion_ramas_para_graficos.csv",
    )

    # --------------------------------------------------------
    # RESUMEN EN TERMINAL
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )
    print(
        "RAMAS CON MUESTRAS SUFICIENTES"
    )
    print(
        "=========================================="
    )

    print(
        ramas_graficos[
            [
                "rama",
                "min_n_ingresos_validos",
                "min_n_pares_horarios",
                "apta_ingreso_mensual",
                "apta_ingreso_horario",
                "orden_2026",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=========================================="
    )
    print(
        "AUDITORÍA DEL UNIVERSO"
    )
    print(
        "=========================================="
    )

    print(
        auditoria.to_string(
            index=False,
            float_format=lambda x:
                f"{x:,.2f}",
        )
    )

    print(
        "\nLISTO: análisis final de ramas "
        "para ocupados de 25 a 64 años."
    )


if __name__ == "__main__":
    main()