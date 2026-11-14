"""TP Introducción al Análisis de Datos - EPH 2016-2026.

GRÁFICOS FINALES CANDIDATOS:
RAMAS DE ACTIVIDAD, INGRESOS Y HORAS.

Utiliza exclusivamente los CSV generados por:
analisis_ramas_ocupados_25_64_final.py

No recalcula microdatos.
"""

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE = Path(__file__).resolve().parent

RESULTADOS = BASE / "resultados"
GRAFICOS = BASE / "graficos"

ARCHIVO_DATOS = (
    RESULTADOS
    / "ramas_ocupados_25_64_extremos.csv"
)

ARCHIVO_SELECCION = (
    RESULTADOS
    / "seleccion_ramas_para_graficos.csv"
)

PERIODOS = [
    "2016-T2",
    "2026-T2",
]

AGLOMERADOS = [
    "CABA",
    "Partidos del GBA",
]

MAX_RAMAS = 6


# ============================================================
# 1. LECTURA
# ============================================================

def cargar_datos():
    """Carga resultados y tabla de selección."""

    if not ARCHIVO_DATOS.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_DATOS}"
        )

    if not ARCHIVO_SELECCION.is_file():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_SELECCION}"
        )

    datos = pd.read_csv(
        ARCHIVO_DATOS,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    seleccion = pd.read_csv(
        ARCHIVO_SELECCION,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # Pandas suele interpretar True/False automáticamente,
    # pero dejamos el script preparado por si llegaran como texto.
    for columna in [
        "apta_ingreso_mensual",
        "apta_ingreso_horario",
    ]:

        if seleccion[columna].dtype != bool:
            seleccion[columna] = (
                seleccion[columna]
                .astype(str)
                .str.lower()
                .eq("true")
            )

    seleccion = seleccion.sort_values(
        "orden_2026"
    )

    return datos, seleccion


# ============================================================
# 2. FORMATO
# ============================================================

def nombre_corto(texto):
    """Corta nombres largos en dos líneas."""

    return fill(
        texto,
        width=23,
    )


def millones(valor, _):
    """Formato del eje monetario."""

    return (
        f"${valor / 1_000_000:.1f} M"
        .replace(".", ",")
    )


def pesos_hora(valor, _):
    """Formato de valores horarios."""

    return (
        f"${valor:,.0f}"
        .replace(",", ".")
    )


def porcentaje(valor, _):
    return (
        f"{valor:.0f}%"
        .replace(".", ",")
    )


# ============================================================
# 3. SELECCIÓN DE RAMAS
# ============================================================

def seleccionar_ramas(
    seleccion: pd.DataFrame,
):
    """Obtiene ramas comparables para cada gráfico."""

    # Para composición no necesitamos P21:
    # usamos simplemente las seis ramas con mayor
    # población ocupada combinada en 2026 T2.
    ramas_composicion = (
        seleccion
        .head(MAX_RAMAS)["rama"]
        .tolist()
    )

    # Para ingresos exigimos >=30 ingresos válidos
    # en las cuatro celdas:
    # 2016/2026 × CABA/GBA.
    ramas_ingresos = (
        seleccion.loc[
            seleccion[
                "apta_ingreso_mensual"
            ]
        ]
        .head(MAX_RAMAS)["rama"]
        .tolist()
    )

    # Para ingreso horario hacemos lo mismo
    # utilizando pares ingreso-horas.
    ramas_horarias = (
        seleccion.loc[
            seleccion[
                "apta_ingreso_horario"
            ]
        ]
        .head(MAX_RAMAS)["rama"]
        .tolist()
    )

    return (
        ramas_composicion,
        ramas_ingresos,
        ramas_horarias,
    )


# ============================================================
# 4. GRÁFICO: COMPOSICIÓN DEL EMPLEO
# ============================================================

def grafico_composicion(
    datos,
    ramas,
):
    """Compara el peso de las principales ramas."""

    fig, ejes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        sharex=True,
    )

    y = np.arange(
        len(ramas)
    )

    ancho = 0.36

    for eje, aglomerado in zip(
        ejes,
        AGLOMERADOS,
    ):

        for desplazamiento, periodo in [
            (-ancho / 2, "2016-T2"),
            (ancho / 2, "2026-T2"),
        ]:

            parte = (
                datos.loc[
                    (datos["aglomerado"]
                     == aglomerado)
                    &
                    (datos["periodo"]
                     == periodo)
                ]
                .set_index("rama")
                .reindex(ramas)
            )

            valores = (
                parte[
                    "porcentaje_ocupados_ponderado"
                ]
                .to_numpy(dtype=float)
            )

            barras = eje.barh(
                y + desplazamiento,
                valores,
                height=ancho,
                label=periodo.replace("-", " "),
            )

            for barra, valor in zip(
                barras,
                valores,
            ):
                if np.isnan(valor):
                    continue

                eje.text(
                    barra.get_width(),
                    barra.get_y()
                    + barra.get_height() / 2,
                    (
                        f"{valor:.1f}%"
                        .replace(".", ",")
                    ),
                    va="center",
                    ha="left",
                    fontsize=8,
                )

        eje.set_title(
            aglomerado
        )

        eje.set_yticks(
            y,
            [
                nombre_corto(r)
                for r in ramas
            ],
        )

        eje.invert_yaxis()

        eje.xaxis.set_major_formatter(
            FuncFormatter(porcentaje)
        )

        eje.grid(
            axis="x",
            linestyle="--",
            alpha=0.25,
        )

        eje.legend()

    ejes[0].set_xlabel(
        "Porcentaje de ocupados"
    )

    ejes[1].set_xlabel(
        "Porcentaje de ocupados"
    )

    fig.suptitle(
        "Composición del empleo por rama\n"
        "Ocupados de 25 a 64 años",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        "Porcentajes ponderados con PONDERA. "
        "Se muestran las seis ramas con mayor "
        "población ocupada combinada en 2026 T2.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 0.94)
    )

    destino = (
        GRAFICOS
        / "composicion_ramas_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Gráfico guardado:",
        destino,
    )


# ============================================================
# 5. GRÁFICO: INGRESO MENSUAL
# ============================================================

def grafico_ingreso_mensual(
    datos,
    ramas,
):
    """Compara P21 real dentro de las mismas ramas."""

    if not ramas:
        print(
            "AVISO: no hay ramas con muestra "
            "suficiente para ingreso mensual."
        )
        return

    fig, ejes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        sharex=True,
    )

    y = np.arange(
        len(ramas)
    )

    ancho = 0.36

    for eje, periodo in zip(
        ejes,
        PERIODOS,
    ):

        for desplazamiento, aglomerado in [
            (-ancho / 2, "CABA"),
            (ancho / 2, "Partidos del GBA"),
        ]:

            parte = (
                datos.loc[
                    (datos["periodo"]
                     == periodo)
                    &
                    (datos["aglomerado"]
                     == aglomerado)
                ]
                .set_index("rama")
                .reindex(ramas)
            )

            valores = (
                parte[
                    "ingreso_real_mediana_2026_t2"
                ]
                .to_numpy(dtype=float)
            )

            barras = eje.barh(
                y + desplazamiento,
                valores,
                height=ancho,
                label=aglomerado,
            )

            for barra, valor in zip(
                barras,
                valores,
            ):

                if np.isnan(valor):
                    continue

                eje.text(
                    barra.get_width(),
                    barra.get_y()
                    + barra.get_height() / 2,
                    (
                        f"${valor / 1_000_000:.2f} M"
                        .replace(".", ",")
                    ),
                    va="center",
                    ha="left",
                    fontsize=8,
                )

        eje.set_title(
            periodo.replace("-", " ")
        )

        eje.set_yticks(
            y,
            [
                nombre_corto(r)
                for r in ramas
            ],
        )

        eje.invert_yaxis()

        eje.xaxis.set_major_formatter(
            FuncFormatter(millones)
        )

        eje.grid(
            axis="x",
            linestyle="--",
            alpha=0.25,
        )

        eje.legend()

    fig.suptitle(
        "Ingreso real de la ocupación principal por rama\n"
        "Ocupados de 25 a 64 años",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        "Mediana de P21 positivo, ponderada con PONDIIO "
        "y expresada en pesos de 2026 T2. "
        "Solo se muestran ramas con al menos 30 respuestas "
        "válidas en cada comparación.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.06, 1, 0.94)
    )

    destino = (
        GRAFICOS
        / "ingreso_real_ramas_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Gráfico guardado:",
        destino,
    )


# ============================================================
# 6. GRÁFICO: BRECHA CABA - GBA
# ============================================================

def grafico_brecha(
    datos,
    ramas,
):
    """Muestra cuánto supera CABA a GBA por rama."""

    if not ramas:
        return

    filas = []

    for periodo in PERIODOS:

        for rama in ramas:

            parte = datos.loc[
                (datos["periodo"]
                 == periodo)
                &
                (datos["rama"]
                 == rama)
            ]

            caba = parte.loc[
                parte["aglomerado"]
                == "CABA",
                "ingreso_real_mediana_2026_t2",
            ]

            gba = parte.loc[
                parte["aglomerado"]
                == "Partidos del GBA",
                "ingreso_real_mediana_2026_t2",
            ]

            if len(caba) != 1 or len(gba) != 1:
                raise ValueError(
                    f"Datos incompletos: "
                    f"{periodo}, {rama}"
                )

            valor_caba = float(
                caba.iloc[0]
            )

            valor_gba = float(
                gba.iloc[0]
            )

            brecha = (
                (
                    valor_caba
                    / valor_gba
                    - 1
                )
                * 100
                if valor_gba > 0
                else np.nan
            )

            filas.append({
                "periodo": periodo,
                "rama": rama,
                "brecha": brecha,
            })

    tabla = pd.DataFrame(
        filas
    )

    y = np.arange(
        len(ramas)
    )

    ancho = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for desplazamiento, periodo in [
        (-ancho / 2, "2016-T2"),
        (ancho / 2, "2026-T2"),
    ]:

        parte = (
            tabla.loc[
                tabla["periodo"]
                == periodo
            ]
            .set_index("rama")
            .reindex(ramas)
        )

        valores = (
            parte["brecha"]
            .to_numpy(dtype=float)
        )

        barras = ax.barh(
            y + desplazamiento,
            valores,
            height=ancho,
            label=periodo.replace("-", " "),
        )

        for barra, valor in zip(
            barras,
            valores,
        ):

            if np.isnan(valor):
                continue

            alineacion = (
                "left"
                if valor >= 0
                else "right"
            )

            ax.text(
                valor,
                barra.get_y()
                + barra.get_height() / 2,
                (
                    f"{valor:.1f}%"
                    .replace(".", ",")
                ),
                va="center",
                ha=alineacion,
                fontsize=8,
            )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(
        y,
        [
            nombre_corto(r)
            for r in ramas
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Diferencia de la mediana de CABA respecto del GBA (%)"
    )

    ax.set_title(
        "Brecha de ingresos dentro de las mismas ramas\n"
        "Ocupados de 25 a 64 años"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.legend()

    fig.text(
        0.5,
        0.01,
        "Un valor positivo indica una mediana mayor en CABA; "
        "un valor negativo, una mediana mayor en Partidos del GBA.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 1)
    )

    destino = (
        GRAFICOS
        / "brecha_ingresos_ramas_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Gráfico guardado:",
        destino,
    )


# ============================================================
# 7. GRÁFICO: INGRESO HORARIO
# ============================================================

def grafico_ingreso_horario(
    datos,
    ramas,
):
    """Compara el ingreso horario aproximado."""

    if not ramas:
        print(
            "AVISO: no hay suficientes ramas "
            "para comparar ingreso horario."
        )
        return

    fig, ejes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        sharex=True,
    )

    y = np.arange(
        len(ramas)
    )

    ancho = 0.36

    for eje, periodo in zip(
        ejes,
        PERIODOS,
    ):

        for desplazamiento, aglomerado in [
            (-ancho / 2, "CABA"),
            (ancho / 2, "Partidos del GBA"),
        ]:

            parte = (
                datos.loc[
                    (datos["periodo"]
                     == periodo)
                    &
                    (datos["aglomerado"]
                     == aglomerado)
                ]
                .set_index("rama")
                .reindex(ramas)
            )

            valores = (
                parte[
                    "ingreso_horario_real_mediana_estimada"
                ]
                .to_numpy(dtype=float)
            )

            barras = eje.barh(
                y + desplazamiento,
                valores,
                height=ancho,
                label=aglomerado,
            )

            for barra, valor in zip(
                barras,
                valores,
            ):

                if np.isnan(valor):
                    continue

                eje.text(
                    barra.get_width(),
                    barra.get_y()
                    + barra.get_height() / 2,
                    (
                        f"${valor:,.0f}"
                        .replace(",", ".")
                    ),
                    va="center",
                    ha="left",
                    fontsize=8,
                )

        eje.set_title(
            periodo.replace("-", " ")
        )

        eje.set_yticks(
            y,
            [
                nombre_corto(r)
                for r in ramas
            ],
        )

        eje.invert_yaxis()

        eje.xaxis.set_major_formatter(
            FuncFormatter(
                pesos_hora
            )
        )

        eje.grid(
            axis="x",
            linestyle="--",
            alpha=0.25,
        )

        eje.legend()

    fig.suptitle(
        "Ingreso horario real aproximado por rama\n"
        "Ocupados de 25 a 64 años",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        "Estimación: ingreso mensual P21 / "
        "(horas de la semana de referencia × 52/12). "
        "Solo se muestran ramas con al menos 30 pares válidos "
        "en cada comparación.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.06, 1, 0.94)
    )

    destino = (
        GRAFICOS
        / "ingreso_horario_ramas_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Gráfico guardado:",
        destino,
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    GRAFICOS.mkdir(
        exist_ok=True
    )

    datos, seleccion = (
        cargar_datos()
    )

    (
        ramas_composicion,
        ramas_ingresos,
        ramas_horarias,
    ) = seleccionar_ramas(
        seleccion
    )

    print(
        "\nRamas para composición:"
    )
    for rama in ramas_composicion:
        print(" -", rama)

    print(
        "\nRamas para ingreso mensual:"
    )
    for rama in ramas_ingresos:
        print(" -", rama)

    print(
        "\nRamas para ingreso horario:"
    )
    for rama in ramas_horarias:
        print(" -", rama)

    grafico_composicion(
        datos,
        ramas_composicion,
    )

    grafico_ingreso_mensual(
        datos,
        ramas_ingresos,
    )

    grafico_brecha(
        datos,
        ramas_ingresos,
    )

    grafico_ingreso_horario(
        datos,
        ramas_horarias,
    )

    print(
        "\nLISTO: gráficos candidatos "
        "de ramas generados."
    )


if __name__ == "__main__":
    main()