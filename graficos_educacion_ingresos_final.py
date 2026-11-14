"""TP Introducción al Análisis de Datos - EPH 2016-2026.

GRÁFICOS COMPLEMENTARIOS: EDUCACIÓN E INGRESOS.

Utiliza resultados ya calculados en:
    resultados/ingresos_educacion_dos_universos.csv

Genera tres gráficos:

1. Composición educativa de los ocupados de 25 a 64 años.
2. Mediana real de ingresos por nivel educativo.
3. Brecha de ingresos entre CABA y Partidos del GBA.

No modifica microdatos ni realiza cálculos de regresión.
"""

from pathlib import Path

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

ARCHIVO = RESULTADOS / "ingresos_educacion_dos_universos.csv"

UNIVERSO = "Ocupados de 25 a 64 años"

AGLOMERADOS = [
    "CABA",
    "Partidos del GBA",
]

PERIODOS = [
    "2016-T2",
    "2026-T2",
]

GRUPOS_COMPLETOS = [
    "Sin instrucción",
    "Primaria incompleta",
    "Primario completo o secundario incompleto",
    "Secundario completo o superior incompleto",
    "Superior completo",
]

# Para los gráficos de ingresos solamente mostramos grupos con
# tamaños muestrales suficientemente razonables en ambos períodos.
GRUPOS_INGRESOS = [
    "Primario completo o secundario incompleto",
    "Secundario completo o superior incompleto",
    "Superior completo",
]

NOMBRES_CORTOS = {
    "Sin instrucción": "Sin instrucción",
    "Primaria incompleta": "Primaria incompleta",
    "Primario completo o secundario incompleto":
        "Primario comp. /\nsecundario incomp.",
    "Secundario completo o superior incompleto":
        "Secundario comp. /\nsuperior incomp.",
    "Superior completo": "Superior completo",
}


# ============================================================
# LECTURA
# ============================================================

def cargar_resultados() -> pd.DataFrame:
    """Lee el CSV generado en la etapa anterior."""

    if not ARCHIVO.is_file():
        raise FileNotFoundError(
            f"No se encontró:\n{ARCHIVO}"
        )

    datos = pd.read_csv(
        ARCHIVO,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    columnas_necesarias = {
        "periodo",
        "aglomerado",
        "universo",
        "grupo_educacion",
        "porcentaje_dentro_universo",
        "n_ingreso_positivo_pondii_valido",
        "mediana_real_2026_t2",
    }

    faltantes = columnas_necesarias - set(datos.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            f"{sorted(faltantes)}"
        )

    return datos


# ============================================================
# FORMATO
# ============================================================

def formato_millones(valor, _):
    """Ejemplo: 1500000 -> $1,5 M."""

    return f"${valor / 1_000_000:.1f} M".replace(".", ",")


def etiqueta_millones(valor):
    """Formato compacto para etiquetas sobre barras."""

    return f"${valor / 1_000_000:.2f} M".replace(".", ",")


# ============================================================
# GRÁFICO 1
# COMPOSICIÓN EDUCATIVA
# ============================================================

def graficar_composicion_educativa(datos: pd.DataFrame) -> None:
    """Compara la distribución educativa de ocupados 25-64."""

    seleccion = datos.loc[
        (datos["universo"] == UNIVERSO)
        & (datos["grupo_educacion"].isin(GRUPOS_COMPLETOS))
    ].copy()

    filas = []

    for periodo in PERIODOS:
        for aglomerado in AGLOMERADOS:
            fila = {
                "grupo": (
                    f"{aglomerado}\n"
                    f"{periodo.replace('-', ' ')}"
                )
            }

            parte = seleccion.loc[
                (seleccion["periodo"] == periodo)
                & (seleccion["aglomerado"] == aglomerado)
            ]

            for nivel in GRUPOS_COMPLETOS:
                valor = parte.loc[
                    parte["grupo_educacion"] == nivel,
                    "porcentaje_dentro_universo",
                ]

                if len(valor) != 1:
                    raise ValueError(
                        f"No se encontró una única fila para "
                        f"{periodo}, {aglomerado}, {nivel}"
                    )

                fila[nivel] = float(valor.iloc[0])

            filas.append(fila)

    tabla = pd.DataFrame(filas)

    fig, ax = plt.subplots(figsize=(12, 6))

    acumulado = np.zeros(len(tabla))

    for nivel in GRUPOS_COMPLETOS:
        valores = tabla[nivel].to_numpy()

        ax.barh(
            tabla["grupo"],
            valores,
            left=acumulado,
            label=nivel,
        )

        acumulado += valores

    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje de los ocupados de 25 a 64 años")
    ax.set_ylabel("")

    ax.set_title(
        "Composición educativa de los ocupados de 25 a 64 años\n"
        "CABA y Partidos del GBA, 2016 T2 y 2026 T2"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    ax.legend(
        title="Nivel educativo",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
    )

    fig.text(
        0.01,
        0.01,
        "Porcentajes ponderados con PONDERA. "
        "Fuente: elaboración propia con microdatos EPH-INDEC.",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    ruta = (
        GRAFICOS
        / "composicion_educativa_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        ruta,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Gráfico guardado:", ruta)


# ============================================================
# GRÁFICO 2
# INGRESOS POR EDUCACIÓN
# ============================================================

def graficar_ingresos_por_educacion(
    datos: pd.DataFrame,
) -> None:
    """Compara medianas reales dentro de niveles educativos."""

    seleccion = datos.loc[
        (datos["universo"] == UNIVERSO)
        & (datos["grupo_educacion"].isin(GRUPOS_INGRESOS))
    ].copy()

    fig, ejes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharey=True,
    )

    for eje, periodo in zip(ejes, PERIODOS):

        parte = seleccion.loc[
            seleccion["periodo"] == periodo
        ]

        posiciones = np.arange(
            len(GRUPOS_INGRESOS)
        )

        ancho = 0.36

        for desplazamiento, aglomerado in [
            (-ancho / 2, "CABA"),
            (ancho / 2, "Partidos del GBA"),
        ]:

            valores = []

            for nivel in GRUPOS_INGRESOS:
                fila = parte.loc[
                    (parte["aglomerado"] == aglomerado)
                    & (parte["grupo_educacion"] == nivel)
                ]

                if len(fila) != 1:
                    raise ValueError(
                        f"Resultado inesperado para "
                        f"{periodo}, {aglomerado}, {nivel}"
                    )

                n = int(
                    fila[
                        "n_ingreso_positivo_pondii_valido"
                    ].iloc[0]
                )

                # No dibujamos celdas pequeñas.
                if n < 30:
                    valores.append(np.nan)
                else:
                    valores.append(
                        float(
                            fila[
                                "mediana_real_2026_t2"
                            ].iloc[0]
                        )
                    )

            barras = eje.bar(
                posiciones + desplazamiento,
                valores,
                width=ancho,
                label=aglomerado,
            )

            for barra, valor in zip(
                barras,
                valores,
            ):
                if np.isnan(valor):
                    continue

                eje.text(
                    barra.get_x()
                    + barra.get_width() / 2,
                    barra.get_height(),
                    etiqueta_millones(valor),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        eje.set_xticks(
            posiciones,
            [
                NOMBRES_CORTOS[nivel]
                for nivel in GRUPOS_INGRESOS
            ],
        )

        eje.set_title(
            periodo.replace("-", " ")
        )

        eje.grid(
            axis="y",
            alpha=0.25,
            linestyle="--",
        )

        eje.yaxis.set_major_formatter(
            FuncFormatter(formato_millones)
        )

        eje.legend()

    ejes[0].set_ylabel(
        "Mediana de ingreso individual positivo\n"
        "(pesos constantes de 2026 T2)"
    )

    fig.suptitle(
        "Ingresos reales según nivel educativo\n"
        "Ocupados de 25 a 64 años",
        fontsize=14,
    )

    fig.text(
        0.01,
        0.01,
        "P47T positivo, ponderado con PONDII. "
        "Solo se muestran grupos con al menos 30 observaciones "
        "válidas. Fuente: elaboración propia con EPH-INDEC.",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.06, 1, 0.94)
    )

    ruta = (
        GRAFICOS
        / "ingresos_reales_educacion_ocupados_25_64_extremos.png"
    )

    fig.savefig(
        ruta,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Gráfico guardado:", ruta)


# ============================================================
# GRÁFICO 3
# EVOLUCIÓN DE LA BRECHA
# ============================================================

def graficar_brecha_educativa(
    datos: pd.DataFrame,
) -> None:
    """Muestra cuánto supera la mediana de CABA a la de GBA."""

    seleccion = datos.loc[
        (datos["universo"] == UNIVERSO)
        & (datos["grupo_educacion"].isin(GRUPOS_INGRESOS))
    ].copy()

    filas = []

    for periodo in PERIODOS:
        for nivel in GRUPOS_INGRESOS:

            parte = seleccion.loc[
                (seleccion["periodo"] == periodo)
                & (seleccion["grupo_educacion"] == nivel)
            ]

            caba = parte.loc[
                parte["aglomerado"] == "CABA"
            ]

            gba = parte.loc[
                parte["aglomerado"]
                == "Partidos del GBA"
            ]

            if len(caba) != 1 or len(gba) != 1:
                raise ValueError(
                    f"Datos incompletos: {periodo}, {nivel}"
                )

            n_caba = int(
                caba[
                    "n_ingreso_positivo_pondii_valido"
                ].iloc[0]
            )

            n_gba = int(
                gba[
                    "n_ingreso_positivo_pondii_valido"
                ].iloc[0]
            )

            if n_caba < 30 or n_gba < 30:
                brecha = np.nan

            else:
                mediana_caba = float(
                    caba[
                        "mediana_real_2026_t2"
                    ].iloc[0]
                )

                mediana_gba = float(
                    gba[
                        "mediana_real_2026_t2"
                    ].iloc[0]
                )

                brecha = (
                    mediana_caba
                    / mediana_gba
                    - 1
                ) * 100

            filas.append({
                "periodo": periodo,
                "grupo": nivel,
                "brecha": brecha,
            })

    tabla = pd.DataFrame(filas)

    posiciones = np.arange(
        len(GRUPOS_INGRESOS)
    )

    ancho = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for desplazamiento, periodo in [
        (-ancho / 2, "2016-T2"),
        (ancho / 2, "2026-T2"),
    ]:

        parte = tabla.loc[
            tabla["periodo"] == periodo
        ].set_index("grupo")

        valores = [
            parte.loc[nivel, "brecha"]
            for nivel in GRUPOS_INGRESOS
        ]

        barras = ax.bar(
            posiciones + desplazamiento,
            valores,
            width=ancho,
            label=periodo.replace("-", " "),
        )

        for barra, valor in zip(
            barras,
            valores,
        ):
            if np.isnan(valor):
                continue

            ax.text(
                barra.get_x()
                + barra.get_width() / 2,
                barra.get_height(),
                f"{valor:.1f}%".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(
        posiciones,
        [
            NOMBRES_CORTOS[nivel]
            for nivel in GRUPOS_INGRESOS
        ],
    )

    ax.set_ylabel(
        "Ingreso mediano de CABA por encima del GBA (%)"
    )

    ax.set_title(
        "Brecha de ingresos entre CABA y Partidos del GBA\n"
        "dentro de niveles educativos comparables"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    ax.legend()

    fig.text(
        0.01,
        0.01,
        "Ocupados de 25 a 64 años con P47T positivo. "
        "Valores en pesos constantes de 2026 T2. "
        "Fuente: elaboración propia con EPH-INDEC.",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 1)
    )

    ruta = (
        GRAFICOS
        / "brecha_ingresos_educacion_ocupados_25_64.png"
    )

    fig.savefig(
        ruta,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Gráfico guardado:", ruta)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    GRAFICOS.mkdir(exist_ok=True)

    datos = cargar_resultados()

    graficar_composicion_educativa(datos)

    graficar_ingresos_por_educacion(datos)

    graficar_brecha_educativa(datos)

    print(
        "\nLISTO: se generaron los tres gráficos "
        "de educación e ingresos."
    )


if __name__ == "__main__":
    main()