"""TP EPH 2016-2026: educación, composición e ingresos individuales.

Etapa descriptiva previa al objetivo 4 (regresión).
Compara CABA y Partidos del GBA en 2016 T2 y 2026 T2.
No modifica microdatos ni archivos anteriores y no imputa ingresos.

Requiere los scripts existentes:
- analisis_eph_objetivo2_etapa1.py (dependencia de la etapa 2)
- analisis_eph_objetivo2_etapa2.py
- analisis_ponderado_ipc_listo.py
"""

import numpy as np
import pandas as pd

# Reutilizamos las agrupaciones que ya utilizamos en el objetivo 2.
from analisis_eph_objetivo2_etapa2 import GRUPOS_EDUCACION, MAPA_EDUCACION

# Reutilizamos la lectura auditada, los ponderadores y el IPC del objetivo 1.
from analisis_ponderado_ipc_listo import (
    AGLOMERADOS,
    RESULTADOS,
    archivos_eph,
    leer_eph,
    cargar_ipc,
    cuantil_ponderado,
    guardar_csv,
)

PERIODOS = [(2016, 2), (2026, 2)]
MINIMO_EXPLORATORIO = 30  # Advertencia, no umbral oficial de confiabilidad.


def factores_de_inflacion():
    """Devuelve factores para expresar ingresos en pesos de 2026 T2."""
    ipc_mensual = cargar_ipc()
    ipc_trimestral = ipc_mensual.groupby(
        ["anio", "trimestre"]
    )["ipc"].mean()

    ipc_base = float(ipc_trimestral.loc[(2026, 2)])
    factores = {}

    for periodo in PERIODOS:
        if periodo not in ipc_trimestral.index:
            raise ValueError(f"No se encontró IPC para {periodo}")

        factores[periodo] = (
            ipc_base / float(ipc_trimestral.loc[periodo])
        )

    return factores


def analizar_aglomerado(df, anio, trimestre, codigo, factor):
    """Cruza educación con composición, no respuesta e ingresos positivos."""

    # Conservamos el universo educativo anterior: 25 años o más.
    mayores = df.loc[
        (df["AGLOMERADO"] == codigo) & (df["CH06"] >= 25)
    ].copy()

    if mayores.empty:
        raise ValueError(
            f"No hay personas de 25+ en {anio} T{trimestre}, {codigo}"
        )

    # Validación de códigos educativos.
    codigos_esperados = set(MAPA_EDUCACION) | {9}
    invalidos = (
        mayores["NIVEL_ED"].isna()
        | ~mayores["NIVEL_ED"].isin(codigos_esperados)
    )

    if invalidos.any():
        raise ValueError(
            f"Códigos educativos inesperados en "
            f"{anio} T{trimestre}, {codigo}"
        )

    mayores["grupo_educacion"] = (
        mayores["NIVEL_ED"].map(MAPA_EDUCACION)
    )

    # El código 9 queda fuera de la distribución clasificada.
    clasificados = mayores.loc[
        mayores["grupo_educacion"].notna()
    ]

    peso_total_clasificado = float(
        clasificados["PONDERA"].sum()
    )

    if peso_total_clasificado <= 0:
        raise ValueError(
            "Sin población ponderada con educación clasificada"
        )

    periodo = f"{anio}-T{trimestre}"

    # Auditoría del universo educativo.
    auditoria = {
        "periodo": periodo,
        "aglomerado": AGLOMERADOS[codigo],
        "n_personas_25_mas": int(len(mayores)),
        "n_educacion_clasificada": int(len(clasificados)),
        "n_educacion_codigo_9": int(
            (mayores["NIVEL_ED"] == 9).sum()
        ),
        "poblacion_25_mas_clasificada_pondera":
            peso_total_clasificado,
        "factor_ipc_a_2026_t2": factor,
    }

    resultados = []

    for grupo in GRUPOS_EDUCACION:
        subgrupo = clasificados.loc[
            clasificados["grupo_educacion"] == grupo
        ]

        peso_grupo = float(subgrupo["PONDERA"].sum())

        # La no respuesta se estudia entre entrevistas realizadas.
        entrevistados = subgrupo.loc[
            subgrupo["H15"] == 1
        ]

        p47t = entrevistados["P47T"]

        ingreso_invalido = (
            p47t.notna()
            & (p47t < 0)
            & (p47t != -9)
        )

        if ingreso_invalido.any():
            raise ValueError(
                f"Valores de P47T inesperados: {periodo}, {grupo}"
            )

        no_respuesta = p47t.eq(-9) | p47t.isna()

        # No confundimos ingresos positivos, ceros y no respuestas.
        positivos = entrevistados.loc[p47t > 0]

        # PONDII es el ponderador específico de ingreso total.
        positivos_validos = positivos.loc[
            positivos["PONDII"].notna()
            & (positivos["PONDII"] > 0)
        ]

        peso_entrevistados = float(
            entrevistados["PONDERA"].sum()
        )

        porcentaje_no_respuesta = (
            100 * float(
                entrevistados.loc[
                    no_respuesta, "PONDERA"
                ].sum()
            ) / peso_entrevistados
            if peso_entrevistados > 0
            else np.nan
        )

        # Calculamos la mediana ponderada de ingresos positivos.
        mediana_nominal = cuantil_ponderado(
            positivos_validos["P47T"],
            positivos_validos["PONDII"],
            0.50,
        )

        # Convertimos a pesos constantes de 2026 T2.
        mediana_real = mediana_nominal * factor

        n_positivos_validos = int(len(positivos_validos))

        resultados.append({
            "periodo": periodo,
            "aglomerado": AGLOMERADOS[codigo],
            "grupo_educacion": grupo,
            "n_muestra_25_mas": int(len(subgrupo)),
            "poblacion_ponderada_pondera": peso_grupo,
            "porcentaje_educativo_ponderado":
                100 * peso_grupo / peso_total_clasificado,
            "n_entrevistados_h15_1": int(len(entrevistados)),
            "n_ingreso_cero": int((p47t == 0).sum()),
            "n_ingreso_positivo": int((p47t > 0).sum()),
            "n_ingreso_positivo_pondii_valido":
                n_positivos_validos,
            "n_no_respuesta_p47t": int(no_respuesta.sum()),
            "no_respuesta_pct_pondera":
                porcentaje_no_respuesta,
            "mediana_p47t_positivo_nominal":
                mediana_nominal,
            "mediana_p47t_positivo_real_2026_t2":
                mediana_real,
            "advertencia_menos_30_positivos_validos":
                n_positivos_validos < MINIMO_EXPLORATORIO,
        })

    # Comprobamos que la distribución clasificada sume 100%.
    porcentaje_total = sum(
        r["porcentaje_educativo_ponderado"]
        for r in resultados
    )

    if not np.isclose(porcentaje_total, 100.0):
        raise ValueError(
            f"La composición educativa no suma 100%: "
            f"{porcentaje_total}"
        )

    return resultados, auditoria


def construir_brechas(tabla):
    """Compara la mediana real de CABA y GBA dentro de cada grupo."""

    campos = [
        "periodo",
        "grupo_educacion",
        "mediana_p47t_positivo_real_2026_t2",
        "n_ingreso_positivo_pondii_valido",
    ]

    caba = tabla.loc[
        tabla["aglomerado"] == AGLOMERADOS[32],
        campos
    ].rename(columns={
        "mediana_p47t_positivo_real_2026_t2":
            "mediana_caba",
        "n_ingreso_positivo_pondii_valido":
            "n_caba",
    })

    gba = tabla.loc[
        tabla["aglomerado"] == AGLOMERADOS[33],
        campos
    ].rename(columns={
        "mediana_p47t_positivo_real_2026_t2":
            "mediana_gba",
        "n_ingreso_positivo_pondii_valido":
            "n_gba",
    })

    resultado = caba.merge(
        gba,
        on=["periodo", "grupo_educacion"],
        validate="one_to_one",
    )

    # Brecha porcentual: cuánto supera CABA a GBA.
    resultado["brecha_caba_sobre_gba_pct"] = np.where(
        resultado["mediana_gba"] > 0,
        (
            resultado["mediana_caba"]
            / resultado["mediana_gba"]
            - 1
        ) * 100,
        np.nan,
    )

    # Las celdas pequeñas se conservan, pero se advierten.
    resultado["advertencia_muestra_pequena"] = (
        (resultado["n_caba"] < MINIMO_EXPLORATORIO)
        | (resultado["n_gba"] < MINIMO_EXPLORATORIO)
    )

    return resultado


def main():
    RESULTADOS.mkdir(exist_ok=True)

    factores = factores_de_inflacion()

    # Localizamos los 41 archivos, pero solo cargamos los dos extremos.
    rutas = {
        (a, t): ruta
        for a, t, ruta in archivos_eph()
    }

    filas = []
    auditorias = []

    for anio, trimestre in PERIODOS:
        ruta = rutas[(anio, trimestre)]
        print(
            f"Procesando {anio} T{trimestre}: {ruta.name}"
        )

        datos = leer_eph(ruta, anio, trimestre)

        for codigo in AGLOMERADOS:
            resultados, auditoria = analizar_aglomerado(
                datos,
                anio,
                trimestre,
                codigo,
                factores[(anio, trimestre)],
            )

            filas.extend(resultados)
            auditorias.append(auditoria)

    tabla = pd.DataFrame(filas)

    # Se esperan 2 trimestres x 2 aglomerados x 5 grupos.
    esperadas = (
        len(PERIODOS)
        * len(AGLOMERADOS)
        * len(GRUPOS_EDUCACION)
    )

    if len(tabla) != esperadas:
        raise ValueError(
            "Cantidad inesperada de filas: revisar los grupos"
        )

    if tabla.duplicated([
        "periodo", "aglomerado", "grupo_educacion"
    ]).any():
        raise ValueError("Hay filas duplicadas")

    brechas = construir_brechas(tabla)

    # Guardamos los resultados sin modificar archivos anteriores.
    guardar_csv(
        tabla,
        "educacion_ingresos_extremos_2016_2026.csv"
    )

    guardar_csv(
        pd.DataFrame(auditorias),
        "auditoria_educacion_ingresos_extremos.csv"
    )

    guardar_csv(
        brechas,
        "brechas_ingresos_educacion_extremos.csv"
    )

    # Primera vista de resultados.
    vista = tabla[[
        "periodo",
        "aglomerado",
        "grupo_educacion",
        "n_muestra_25_mas",
        "porcentaje_educativo_ponderado",
        "n_ingreso_positivo_pondii_valido",
        "mediana_p47t_positivo_real_2026_t2",
        "advertencia_menos_30_positivos_validos",
    ]].copy()

    print(
        "\nCRUCE DE EDUCACIÓN E INGRESOS "
        "(medianas en pesos de 2026 T2)"
    )

    print(
        vista.to_string(
            index=False,
            float_format=lambda x: f"{x:,.1f}"
        )
    )

    print(
        "\nADVERTENCIA: menos de 30 positivos ponderables "
        "señala cautela; no se borraron casos."
    )

    print(
        "No hay imputaciones ni modificaciones "
        "de los TXT originales."
    )


if __name__ == "__main__":
    main()
