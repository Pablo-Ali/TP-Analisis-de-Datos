"""TP Introducción al Análisis de Datos - EPH 2016-2026.

AMPLIACIÓN DEL ANÁLISIS DE EDUCACIÓN E INGRESOS.

Objetivos:
1. Calcular ingresos totales individuales positivos en personas de 25+.
2. Repetir el análisis para ocupados de 25 a 64 años.
3. Comparar CABA y Partidos del GBA.
4. Comparar 2016 T2 y 2026 T2.
5. Analizar las medianas generales y por nivel educativo.
6. Auditar tamaños muestrales, ceros y no respuesta.

No modifica archivos ni resultados anteriores.
No realiza regresiones ni imputaciones.

Reutiliza:
- analisis_eph_objetivo2_etapa2.py
- analisis_ponderado_ipc_listo.py

La etapa 2 también requiere su dependencia:
- analisis_eph_objetivo2_etapa1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analisis_eph_objetivo2_etapa2 import (
    GRUPOS_EDUCACION,
    MAPA_EDUCACION,
)

from analisis_ponderado_ipc_listo import (
    AGLOMERADOS,
    RESULTADOS,
    archivos_eph,
    leer_eph,
    cargar_ipc,
    cuantil_ponderado,
    guardar_csv,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PERIODOS = [(2016, 2), (2026, 2)]

UNIVERSO_GENERAL = "Personas de 25 años o más"
UNIVERSO_OCUPADOS = "Ocupados de 25 a 64 años"

GRUPO_TOTAL = "TOTAL"

# Este umbral sirve como advertencia exploratoria.
# No constituye un criterio oficial de precisión estadística.
MINIMO_MUESTRAL = 30


# ============================================================
# 1. FACTORES DE INFLACIÓN
# ============================================================

def obtener_factores_ipc() -> dict:
    """Calcula factores para expresar ingresos en pesos de 2026 T2."""

    ipc = cargar_ipc()

    ipc_trimestral = ipc.groupby(
        ["anio", "trimestre"]
    )["ipc"].mean()

    base = float(ipc_trimestral.loc[(2026, 2)])

    factores = {}

    for anio, trimestre in PERIODOS:
        indice = float(
            ipc_trimestral.loc[(anio, trimestre)]
        )

        if indice <= 0:
            raise ValueError(
                f"IPC inválido: {anio} T{trimestre}"
            )

        factores[(anio, trimestre)] = base / indice

    return factores


# ============================================================
# 2. DEFINICIÓN DE UNIVERSOS
# ============================================================

def preparar_universos(datos: pd.DataFrame) -> dict:
    """Define las dos poblaciones que vamos a comparar."""

    mayores = datos.loc[
        datos["CH06"] >= 25
    ].copy()

    ocupados = datos.loc[
        datos["CH06"].between(25, 64)
        & datos["ESTADO"].eq(1)
    ].copy()

    return {
        UNIVERSO_GENERAL: mayores,
        UNIVERSO_OCUPADOS: ocupados,
    }


# ============================================================
# 3. VALIDACIÓN DE EDUCACIÓN
# ============================================================

def validar_educacion(
    datos: pd.DataFrame,
    periodo: str,
    aglomerado: str,
) -> None:
    """Verifica los códigos educativos oficiales ya utilizados."""

    codigos_validos = set(MAPA_EDUCACION) | {9}

    invalidos = (
        datos["NIVEL_ED"].isna()
        | ~datos["NIVEL_ED"].isin(codigos_validos)
    )

    if invalidos.any():
        raise ValueError(
            f"Educación inválida: {periodo}, {aglomerado}"
        )


# ============================================================
# 4. ESTADÍSTICAS DE INGRESO
# ============================================================

def calcular_estadisticas(
    datos: pd.DataFrame,
    universo: str,
    grupo: str,
    periodo: str,
    aglomerado: str,
    factor_ipc: float,
    porcentaje_grupo: float,
    orden_grupo: int,
) -> dict:
    """Calcula ingresos y auditoría para un subgrupo.

    P47T = -9: no respuesta.
    P47T = 0: ingreso declarado igual a cero.
    P47T > 0: ingreso positivo.

    Las medianas se calculan únicamente sobre P47T positivo
    con PONDII válido y se expresan en pesos de 2026 T2.
    """

    entrevistados = datos.loc[
        datos["H15"] == 1
    ].copy()

    ingresos = entrevistados["P47T"]

    # Rechazamos códigos negativos inesperados.
    invalidos = (
        ingresos.notna()
        & (ingresos < 0)
        & ingresos.ne(-9)
    )

    if invalidos.any():
        raise ValueError(
            f"P47T inválido: {periodo}, {aglomerado}, {grupo}"
        )

    no_respuesta = (
        ingresos.eq(-9) | ingresos.isna()
    )

    ceros = ingresos.eq(0)

    positivos = entrevistados.loc[
        ingresos > 0
    ].copy()

    # PONDII es el ponderador del ingreso total individual.
    ponderador_valido = (
        positivos["PONDII"].notna()
        & (positivos["PONDII"] > 0)
    )

    positivos_validos = positivos.loc[
        ponderador_valido
    ]

    n_validos = int(len(positivos_validos))

    # No reemplazamos ponderadores ausentes o iguales a cero.
    n_sin_ponderador = (
        len(positivos) - n_validos
    )

    # No respuesta ponderada con PONDERA entre entrevistados.
    peso_entrevistados = float(
        entrevistados["PONDERA"].sum()
    )

    peso_no_respuesta = float(
        entrevistados.loc[
            no_respuesta, "PONDERA"
        ].sum()
    )

    porcentaje_no_respuesta = (
        peso_no_respuesta / peso_entrevistados * 100
        if peso_entrevistados > 0
        else np.nan
    )

    # Mediana ponderada de ingresos positivos.
    mediana_nominal = cuantil_ponderado(
        positivos_validos["P47T"],
        positivos_validos["PONDII"],
        0.50,
    )

    mediana_real = (
        mediana_nominal * factor_ipc
    )

    return {
        "periodo": periodo,
        "aglomerado": aglomerado,
        "universo": universo,
        "grupo_educacion": grupo,
        "orden_grupo": orden_grupo,

        "n_muestra": int(len(datos)),
        "poblacion_ponderada_pondera": float(
            datos["PONDERA"].sum()
        ),
        "porcentaje_dentro_universo": porcentaje_grupo,

        "n_entrevistados": int(len(entrevistados)),
        "n_ingreso_cero": int(ceros.sum()),
        "n_ingreso_positivo": int(len(positivos)),
        "n_ingreso_positivo_pondii_valido": n_validos,
        "n_positivo_sin_pondii_valido": n_sin_ponderador,

        "n_no_respuesta": int(no_respuesta.sum()),
        "no_respuesta_pct_pondera":
            porcentaje_no_respuesta,

        "mediana_nominal": mediana_nominal,
        "mediana_real_2026_t2": mediana_real,

        "advertencia_muestra_pequena":
            n_validos < MINIMO_MUESTRAL,
    }


# ============================================================
# 5. ANÁLISIS DE CADA UNIVERSO
# ============================================================

def analizar_universo(
    datos: pd.DataFrame,
    universo: str,
    periodo: str,
    aglomerado: str,
    factor_ipc: float,
) -> list[dict]:
    """Calcula la mediana total y las cinco medianas educativas."""

    if datos.empty:
        raise ValueError(
            f"Universo vacío: {periodo}, {aglomerado}, {universo}"
        )

    validar_educacion(
        datos, periodo, aglomerado
    )

    datos = datos.copy()

    datos["grupo_educacion"] = (
        datos["NIVEL_ED"].map(MAPA_EDUCACION)
    )

    peso_total = float(
        datos["PONDERA"].sum()
    )

    if peso_total <= 0:
        raise ValueError("Población ponderada inválida")

    resultados = []

    # Primero calculamos la mediana del universo completo.
    # No es el promedio de las medianas educativas.
    resultados.append(
        calcular_estadisticas(
            datos=datos,
            universo=universo,
            grupo=GRUPO_TOTAL,
            periodo=periodo,
            aglomerado=aglomerado,
            factor_ipc=factor_ipc,
            porcentaje_grupo=100.0,
            orden_grupo=0,
        )
    )

    # Luego analizamos los niveles educativos por separado.
    for indice, grupo in enumerate(
        GRUPOS_EDUCACION, start=1
    ):
        subgrupo = datos.loc[
            datos["grupo_educacion"] == grupo
        ]

        peso_grupo = float(
            subgrupo["PONDERA"].sum()
        )

        porcentaje = (
            peso_grupo / peso_total * 100
        )

        resultados.append(
            calcular_estadisticas(
                datos=subgrupo,
                universo=universo,
                grupo=grupo,
                periodo=periodo,
                aglomerado=aglomerado,
                factor_ipc=factor_ipc,
                porcentaje_grupo=porcentaje,
                orden_grupo=indice,
            )
        )

    # Auditoría de clasificación educativa.
    no_clasificados = int(
        datos["grupo_educacion"].isna().sum()
    )

    if no_clasificados:
        print(
            f"AVISO: {no_clasificados} registros sin "
            f"grupo educativo en {periodo}, {aglomerado}, "
            f"{universo}. Se incluyen en TOTAL."
        )

    return resultados


# ============================================================
# 6. BRECHA ENTRE CABA Y GBA
# ============================================================

def calcular_brechas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Compara medianas reales del mismo período y universo."""

    claves = [
        "periodo",
        "universo",
        "grupo_educacion",
        "orden_grupo",
    ]

    campos = claves + [
        "mediana_real_2026_t2",
        "n_ingreso_positivo_pondii_valido",
    ]

    caba = tabla.loc[
        tabla["aglomerado"] == AGLOMERADOS[32],
        campos,
    ].rename(columns={
        "mediana_real_2026_t2": "mediana_caba",
        "n_ingreso_positivo_pondii_valido": "n_caba",
    })

    gba = tabla.loc[
        tabla["aglomerado"] == AGLOMERADOS[33],
        campos,
    ].rename(columns={
        "mediana_real_2026_t2": "mediana_gba",
        "n_ingreso_positivo_pondii_valido": "n_gba",
    })

    comparacion = caba.merge(
        gba,
        on=claves,
        validate="one_to_one",
    )

    # Brecha porcentual relativa a GBA.
    comparacion["brecha_caba_gba_pct"] = np.where(
        comparacion["mediana_gba"] > 0,
        (
            comparacion["mediana_caba"]
            / comparacion["mediana_gba"]
            - 1
        ) * 100,
        np.nan,
    )

    comparacion["advertencia_muestra_pequena"] = (
        (comparacion["n_caba"] < MINIMO_MUESTRAL)
        | (comparacion["n_gba"] < MINIMO_MUESTRAL)
    )

    return comparacion.sort_values(
        ["universo", "periodo", "orden_grupo"]
    )


# ============================================================
# 7. VARIACIÓN REAL ENTRE 2016 Y 2026
# ============================================================

def calcular_evolucion(tabla: pd.DataFrame) -> pd.DataFrame:
    """Calcula cambios porcentuales reales dentro del mismo universo."""

    claves = [
        "aglomerado",
        "universo",
        "grupo_educacion",
        "orden_grupo",
    ]

    campos = claves + [
        "mediana_real_2026_t2",
        "n_ingreso_positivo_pondii_valido",
    ]

    inicial = tabla.loc[
        tabla["periodo"] == "2016-T2",
        campos,
    ].rename(columns={
        "mediana_real_2026_t2": "mediana_2016",
        "n_ingreso_positivo_pondii_valido": "n_2016",
    })

    final = tabla.loc[
        tabla["periodo"] == "2026-T2",
        campos,
    ].rename(columns={
        "mediana_real_2026_t2": "mediana_2026",
        "n_ingreso_positivo_pondii_valido": "n_2026",
    })

    evolucion = inicial.merge(
        final,
        on=claves,
        validate="one_to_one",
    )

    evolucion["variacion_real_pct"] = np.where(
        evolucion["mediana_2016"] > 0,
        (
            evolucion["mediana_2026"]
            / evolucion["mediana_2016"]
            - 1
        ) * 100,
        np.nan,
    )

    evolucion["advertencia_muestra_pequena"] = (
        (evolucion["n_2016"] < MINIMO_MUESTRAL)
        | (evolucion["n_2026"] < MINIMO_MUESTRAL)
    )

    return evolucion.sort_values(
        ["universo", "aglomerado", "orden_grupo"]
    )


# ============================================================
# 8. PRESENTACIÓN DE RESULTADOS
# ============================================================

def mostrar_resultados(
    tabla: pd.DataFrame,
    brechas: pd.DataFrame,
    evolucion: pd.DataFrame,
) -> None:
    """Imprime las comparaciones principales de manera legible."""

    print("\n" + "=" * 65)
    print("MEDIANAS GENERALES POR UNIVERSO")
    print("=" * 65)

    totales = tabla.loc[
        tabla["grupo_educacion"] == GRUPO_TOTAL
    ]

    columnas = [
        "periodo",
        "aglomerado",
        "universo",
        "n_muestra",
        "n_ingreso_positivo_pondii_valido",
        "mediana_real_2026_t2",
        "no_respuesta_pct_pondera",
    ]

    print(
        totales[columnas].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\n" + "=" * 65)
    print("EVOLUCIÓN REAL DE LAS MEDIANAS")
    print("=" * 65)

    columnas_evolucion = [
        "aglomerado",
        "universo",
        "grupo_educacion",
        "mediana_2016",
        "mediana_2026",
        "variacion_real_pct",
        "advertencia_muestra_pequena",
    ]

    print(
        evolucion[columnas_evolucion].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\n" + "=" * 65)
    print("BRECHAS DE INGRESOS ENTRE CABA Y GBA")
    print("=" * 65)

    columnas_brechas = [
        "periodo",
        "universo",
        "grupo_educacion",
        "mediana_caba",
        "mediana_gba",
        "brecha_caba_gba_pct",
        "advertencia_muestra_pequena",
    ]

    print(
        brechas[columnas_brechas].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\nNOTAS:")
    print("- Las medianas corresponden a P47T positivo.")
    print("- Los ingresos se expresan en pesos de 2026 T2.")
    print("- No se incluyen P47T=-9 ni ceros en las medianas.")
    print("- No se modificaron ni imputaron valores originales.")
    print("- Las celdas pequeñas requieren cautela.")
    print("- El grupo TOTAL incluye todos los niveles educativos.")
    print("- Las medianas generales no son promedios de medianas.")


# ============================================================
# 9. PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    """Ejecuta las comparaciones y guarda los tres CSV."""

    RESULTADOS.mkdir(exist_ok=True)

    factores = obtener_factores_ipc()

    # Localizamos las bases pero solo leemos los dos trimestres.
    rutas = {
        (anio, trimestre): ruta
        for anio, trimestre, ruta in archivos_eph()
    }

    filas = []

    for anio, trimestre in PERIODOS:
        periodo = f"{anio}-T{trimestre}"

        ruta = rutas[(anio, trimestre)]

        print(f"\nProcesando {periodo}: {ruta.name}")

        datos = leer_eph(
            ruta,
            anio,
            trimestre,
        )

        universos = preparar_universos(datos)

        for codigo, nombre in AGLOMERADOS.items():

            for nombre_universo, universo in universos.items():

                seleccion = universo.loc[
                    universo["AGLOMERADO"] == codigo
                ]

                resultados = analizar_universo(
                    datos=seleccion,
                    universo=nombre_universo,
                    periodo=periodo,
                    aglomerado=nombre,
                    factor_ipc=factores[(anio, trimestre)],
                )

                filas.extend(resultados)

    tabla = pd.DataFrame(filas)

    # 2 períodos x 2 áreas x 2 universos x (total + 5 grupos).
    esperadas = (
        len(PERIODOS)
        * len(AGLOMERADOS)
        * 2
        * (len(GRUPOS_EDUCACION) + 1)
    )

    if len(tabla) != esperadas:
        raise ValueError(
            f"Se esperaban {esperadas} filas y "
            f"se obtuvieron {len(tabla)}"
        )

    claves = [
        "periodo",
        "aglomerado",
        "universo",
        "grupo_educacion",
    ]

    if tabla.duplicated(claves).any():
        raise ValueError(
            "Se encontraron combinaciones duplicadas"
        )

    tabla = tabla.sort_values(
        ["periodo", "aglomerado", "universo", "orden_grupo"]
    )

    brechas = calcular_brechas(tabla)
    evolucion = calcular_evolucion(tabla)

    # Los nombres son nuevos para no sobrescribir resultados.
    guardar_csv(
        tabla,
        "ingresos_educacion_dos_universos.csv",
    )

    guardar_csv(
        brechas,
        "brechas_educacion_dos_universos.csv",
    )

    guardar_csv(
        evolucion,
        "evolucion_educacion_dos_universos.csv",
    )

    mostrar_resultados(
        tabla,
        brechas,
        evolucion,
    )

    print("\nLISTO: ampliación descriptiva terminada.")


if __name__ == "__main__":
    main()
