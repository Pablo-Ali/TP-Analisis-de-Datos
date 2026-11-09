"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 2, ETAPA 2: indicadores laborales por grupos de edad y educación.

- Reutiliza la lectura y los controles de ``analisis_eph_objetivo2_etapa1.py``.
- Trabaja con 41 trimestres (2016 T2 a 2026 T2) y dos aglomerados.
- Calcula tasas de actividad, empleo y desocupación ponderadas con PONDERA.
- No modifica las bases TXT, el objetivo 1 ni los resultados de etapa 1.
- No imputa ingresos ni calcula salarios: la etapa de ingresos será posterior.
- Los gráficos se generan con microdatos de la computadora del usuario.

DECISIONES EXPLÍCITAS DE AGRUPACIÓN (adaptables si se decide cambiarlas):
  Edad: 14-29, 30-64 y 65 años o más. Son franjas interpretables para
        estadísticas laborales y corresponden a cortes utilizados por INDEC.
        Las personas menores de 14 NO ingresan en estas tasas específicas,
        aunque SÍ están incluidas en las tasas generales de la etapa 1.
  Educación: personas de 25 años o más, por la decisión del objetivo 1.
        7       -> Sin instrucción.
        1       -> Primaria incompleta (grupo propio).
        2,3     -> Primario completo o secundario incompleto.
        4,5     -> Secundario completo o superior incompleto.
        6       -> Superior completo.
        9       -> No sabe/no responde, fuera de la comparación agrupada.
        Los cortes son descriptivos, no equivalen a años exactos de escolaridad.
        Se controla la cantidad de casos por grupo antes de interpretar
        tendencias, especialmente para el grupo de personas sin instrucción.

REQUISITOS: pandas, matplotlib y el script ``analisis_eph_objetivo2_etapa1.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reutilizamos las mismas rutinas de acceso y los mismos controles del paso 1.
# Esto evita mantener dos lectores divergentes de los mismos microdatos.
from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS, GRAFICOS, RESULTADOS,
    buscar_archivos, leer_trimestre, guardar_csv,
)

GRUPOS_EDAD = ["14 a 29 años", "30 a 64 años", "65 años o más"]
GRUPOS_EDUCACION = [
    "Sin instrucción",
    "Primaria incompleta",
    "Primario completo o secundario incompleto",
    "Secundario completo o superior incompleto",
    "Superior completo",
]

# Agrupaciones propuestas a partir de los códigos oficiales NIVEL_ED.
# Importante: primaria incompleta (1) NO se mezcla con primaria completa (2).
# Sin instrucción (7) queda identificada de forma independiente.
MAPA_EDUCACION = {
    7: GRUPOS_EDUCACION[0],
    1: GRUPOS_EDUCACION[1],
    2: GRUPOS_EDUCACION[2], 3: GRUPOS_EDUCACION[2],
    4: GRUPOS_EDUCACION[3], 5: GRUPOS_EDUCACION[3],
    6: GRUPOS_EDUCACION[4],
}


def identificar_grupo_edad(edad: pd.Series) -> pd.Series:
    """Asigna grupos de edad; <14 y edades inválidas quedan sin grupo.

    CH06=-1 significa menor de un año, como ya se contempló en el objetivo 1.
    Ese valor queda fuera de los grupos por ser menor de 14 años.
    """
    edad_ajustada = edad.replace(-1, 0)
    return pd.cut(
        edad_ajustada,
        bins=[13, 29, 64, np.inf],
        labels=GRUPOS_EDAD,
        right=True,
    )


def calcular_tasas_ponderadas(subgrupo: pd.DataFrame) -> dict:
    """Calcula tasas dentro de un grupo con sus denominadores respectivos.

    Actividad = 100*(ocupados + desocupados)/población del grupo.
    Empleo = 100*ocupados/población del grupo.
    Desocupación = 100*desocupados/(ocupados + desocupados).

    Todas las sumas poblacionales utilizan PONDERA, no conteos de filas.
    Un denominador nulo devuelve NaN en vez de inventar un cero.
    """
    est = subgrupo["ESTADO"]
    pesos = subgrupo["PONDERA"]
    poblacion = float(pesos.sum())
    ocupados = float(pesos[est == 1].sum())
    desocupados = float(pesos[est == 2].sum())
    pea = ocupados + desocupados

    def porcentaje(num: float, den: float) -> float:
        return float(num / den * 100) if den > 0 else np.nan

    return {
        "n_registros_muestrales": int(len(subgrupo)),
        "n_ocupados_muestrales": int((est == 1).sum()),
        "n_desocupados_muestrales": int((est == 2).sum()),
        "n_pea_muestral": int(est.isin([1, 2]).sum()),
        "poblacion_grupo_ponderada": poblacion,
        "poblacion_ocupada_ponderada": ocupados,
        "poblacion_desocupada_ponderada": desocupados,
        "poblacion_pea_ponderada": pea,
        "tasa_actividad": porcentaje(pea, poblacion),
        "tasa_empleo": porcentaje(ocupados, poblacion),
        "tasa_desocupacion": porcentaje(desocupados, pea),
    }


def analizar_variable(
    datos: pd.DataFrame,
    campo_grupo: str,
    categorias: list[str],
    nombre_variable: str,
    anio: int,
    trimestre: int,
    codigo: int,
) -> list[dict]:
    """Calcula una fila por categoría, preservando grupos sin observaciones.

    Guardar la cantidad muestral de cada celda permite evaluar estabilidad.
    No se ocultan ni borran automáticamente las celdas con pocos registros.
    """
    filas = []
    for categoria in categorias:
        subgrupo = datos[datos[campo_grupo] == categoria]
        fila = {
            "anio": anio,
            "trimestre": trimestre,
            "periodo": f"{anio}-T{trimestre}",
            "codigo_aglomerado": codigo,
            "aglomerado": AGLOMERADOS[codigo],
            "variable": nombre_variable,
            "grupo": categoria,
        }
        fila.update(calcular_tasas_ponderadas(subgrupo))
        filas.append(fila)
    return filas


def auditar_grupos(datos: pd.DataFrame, anio: int,
                   trimestre: int, codigo: int) -> dict:
    """Comprueba que ninguna observación quede fuera sin conocer el motivo."""
    edad = datos["CH06"]
    grupo_edad = datos["grupo_edad"]
    mayores = datos[edad >= 25]
    educacion = mayores["NIVEL_ED"]
    codigos_validos = set(MAPA_EDUCACION) | {9}

    # Por política conservadora, los valores inesperados interrumpen el cálculo.
    anomalias_edad = edad.isna() | (edad < -1) | (edad > 120)
    anomalias_educacion = educacion.isna() | (~educacion.isin(codigos_validos))
    if anomalias_edad.any() or anomalias_educacion.any():
        raise ValueError(
            f"{anio} T{trimestre}, {AGLOMERADOS[codigo]}: "
            "se detectaron edades o códigos educativos no esperados."
        )

    return {
        "periodo": f"{anio}-T{trimestre}",
        "aglomerado": AGLOMERADOS[codigo],
        "n_total": int(len(datos)),
        "n_menores_14_excluidos_estudio_edad": int((edad < 14).sum()),
        "n_edad_14_mas_con_grupo": int(grupo_edad.notna().sum()),
        "n_25_mas_estudio_educacion": int(len(mayores)),
        "n_educacion_grupo_asignado": int(mayores["grupo_educacion"].notna().sum()),
        "n_educacion_9_no_asignado": int((educacion == 9).sum()),
    }


def graficar_por_grupo(tabla: pd.DataFrame, variable: str,
                       categorias: list[str], indicador: str,
                       titulo: str, archivo: str) -> None:
    """Grafica una línea por aglomerado, un panel por subgrupo real.

    Al guardar PNG no se modifica ningún cálculo. Los 41 trimestres quedan
    graficados, y los rótulos se espacian para mejorar la legibilidad.
    """
    fuente = tabla.loc[tabla["variable"] == variable].copy()
    periodos = fuente["periodo"].drop_duplicates().tolist()
    mapa_posiciones = {p: i for i, p in enumerate(periodos)}

    # Ajustamos la altura al número de categorías: ahora educación tiene 5.
    # La figura debe seguir siendo legible sin aplastar las series.
    alto = max(9, 2.6 * len(categorias))
    fig, ejes = plt.subplots(len(categorias), 1, figsize=(12, alto), sharex=True)
    for eje, grupo in zip(ejes, categorias):
        for codigo, nombre in AGLOMERADOS.items():
            seleccion = fuente.loc[
                (fuente["grupo"] == grupo) &
                (fuente["codigo_aglomerado"] == codigo)
            ].sort_values(["anio", "trimestre"])
            eje.plot(
                [mapa_posiciones[p] for p in seleccion["periodo"]],
                seleccion[indicador],
                label=nombre, linewidth=1.6, marker="o", markersize=2.5,
            )
        eje.set_title(grupo, fontsize=11)
        eje.set_ylabel("Porcentaje (%)")
        eje.grid(axis="y", alpha=0.30, linestyle="--")
        eje.legend(fontsize=9)

    marcas = list(range(0, len(periodos), 4))
    if periodos and len(periodos) - 1 not in marcas:
        marcas.append(len(periodos) - 1)
    ejes[-1].set_xticks(marcas, [periodos[i] for i in marcas], rotation=45, ha="right")
    ejes[-1].set_xlabel("Trimestre")
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    ruta = GRAFICOS / archivo
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {ruta}")


def controlar_resultados(tabla: pd.DataFrame) -> None:
    """Audita cobertura y posibles celdas inestables, sin descartarlas."""
    esperadas = 82 * (len(GRUPOS_EDAD) + len(GRUPOS_EDUCACION))
    if len(tabla) != esperadas:
        raise ValueError(f"Se esperaban {esperadas} filas y se obtuvieron {len(tabla)}")
    if tabla.duplicated(["periodo", "aglomerado", "variable", "grupo"]).any():
        raise ValueError("Se encontraron filas duplicadas en el análisis por grupos")
    for indicador in ("tasa_actividad", "tasa_empleo", "tasa_desocupacion"):
        valores = tabla[indicador].dropna()
        if ((valores < 0) | (valores > 100)).any():
            raise ValueError(f"{indicador}: se detectaron porcentajes imposibles")

    # Menos de 30 observaciones es una SEÑAL EXPLORATORIA de prudencia,
    # no un umbral oficial de significación estadística ni de confiabilidad.
    pocas = tabla[tabla["n_registros_muestrales"] < 30]
    poca_pea = tabla[tabla["n_pea_muestral"] < 30]
    print(f"Celdas con menos de 30 observaciones: {len(pocas)} / {len(tabla)}")
    print(f"Celdas con menos de 30 personas económicamente activas: "
          f"{len(poca_pea)} / {len(tabla)}")
    # El grupo 'Sin instrucción' puede ser pequeño en algunos trimestres.
    # No se suprime automáticamente: se informa su tamaño para decidir
    # si conviene incluir su curva en el informe definitivo.
    sin_instruccion = tabla.loc[
        (tabla["variable"] == "Nivel educativo (25+)") &
        (tabla["grupo"] == "Sin instrucción")
    ]
    if not sin_instruccion.empty:
        print("Sin instrucción: menor número muestral por trimestre/aglomerado = "
              f"{int(sin_instruccion['n_registros_muestrales'].min())}")
    print("AVISO: esas celdas permanecen en el CSV; interpretar con cautela.")


def main() -> None:
    """Procesa 41 trimestres y produce dos series multivariadas y sus gráficos."""
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)
    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} archivos para el objetivo 2, etapa 2.")
    filas = []
    auditorias = []

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        df = leer_trimestre(ruta, anio, trimestre)
        # Los campos auditados en la etapa 1 ahora sí son imprescindibles.
        if not {"CH06", "NIVEL_ED"}.issubset(df.columns):
            raise ValueError(f"{ruta.name}: faltan CH06 o NIVEL_ED")

        for codigo in AGLOMERADOS:
            grupo = df.loc[df["AGLOMERADO"] == codigo].copy()
            if grupo.empty:
                raise ValueError(f"Sin datos del aglomerado {codigo} en {anio} T{trimestre}")

            grupo["grupo_edad"] = identificar_grupo_edad(grupo["CH06"])
            grupo["grupo_educacion"] = grupo["NIVEL_ED"].map(MAPA_EDUCACION)
            auditorias.append(auditar_grupos(grupo, anio, trimestre, codigo))

            # El análisis por edad se hace sobre personas >=14 años.
            grupo_edad = grupo.loc[grupo["CH06"] >= 14]
            filas.extend(analizar_variable(
                grupo_edad, "grupo_edad", GRUPOS_EDAD, "Edad", anio, trimestre, codigo,
            ))

            # Educación alcanzada: selección >=25 años, acordada en objetivo 1.
            grupo_educacion = grupo.loc[
                (grupo["CH06"] >= 25) & (grupo["grupo_educacion"].notna())
            ]
            filas.extend(analizar_variable(
                grupo_educacion, "grupo_educacion", GRUPOS_EDUCACION,
                "Nivel educativo (25+)", anio, trimestre, codigo,
            ))

    tabla = pd.DataFrame(filas)
    auditoria = pd.DataFrame(auditorias)
    controlar_resultados(tabla)
    guardar_csv(tabla, "tasas_laborales_edad_educacion_2016_2026.csv")
    guardar_csv(auditoria, "auditoria_grupos_edad_educacion_2016_2026.csv")

    graficar_por_grupo(
        tabla, "Edad", GRUPOS_EDAD, "tasa_empleo",
        "EPH: tasa de empleo según grupo de edad (2016-2026)",
        "empleo_por_edad_2016_2026.png",
    )
    graficar_por_grupo(
        tabla, "Nivel educativo (25+)", GRUPOS_EDUCACION, "tasa_empleo",
        "EPH: tasa de empleo según educación (25+; 2016-2026)",
        "empleo_por_educacion_2016_2026.png",
    )

    print("\nLISTO: resultados de edad y educación para los 41 trimestres.")
    print("Ramas de actividad e ingresos laborales quedan para la etapa siguiente.")


if __name__ == "__main__":
    main()
