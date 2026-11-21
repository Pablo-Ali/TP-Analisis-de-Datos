"""Objetivo 1 - Exploración univariada y calidad de datos EPH 2016 T2-2026 T2.

Aglomerados:
32 = CABA
33 = Partidos del Gran Buenos Aires

El script:
- recorre las 41 bases individuales;
- revisa P47T, no respuesta y valores atípicos;
- calcula la no respuesta ponderada con PONDERA;
- conserva los valores extremos, sin eliminarlos ni imputarlos;
- genera los dos gráficos usados en el informe para este objetivo.

Requisitos: pandas, numpy y matplotlib.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RESULTADOS = BASE / "resultados"
GRAFICOS = BASE / "graficos"

AGLOMERADOS = {32: "CABA", 33: "Partidos del GBA"}
PATRON = re.compile(r"usu_individual_T([1-4])(\d{2})\.txt$", re.I)

COLUMNAS = {
    "CODUSU", "NRO_HOGAR", "COMPONENTE",
    "ANO4", "TRIMESTRE", "AGLOMERADO",
    "H15", "CH06", "P47T", "PONDERA",
}


def periodos():
    return [
        (anio, trimestre)
        for anio in range(2016, 2027)
        for trimestre in range(1, 5)
        if (2016, 2) <= (anio, trimestre) <= (2026, 2)
    ]


def buscar_archivos():
    encontrados = {}
    for ruta in DATOS.glob("*.txt"):
        m = PATRON.fullmatch(ruta.name)
        if not m:
            continue
        clave = (2000 + int(m.group(2)), int(m.group(1)))
        if clave in encontrados:
            raise ValueError(f"Dos archivos para {clave}")
        encontrados[clave] = ruta

    faltantes = sorted(set(periodos()) - set(encontrados))
    if faltantes:
        raise FileNotFoundError(f"Faltan bases: {faltantes}")

    return [(a, t, encontrados[(a, t)]) for a, t in periodos()]


def leer_base(ruta, anio, trimestre):
    df = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=lambda c: c.strip().upper() in COLUMNAS,
    )
    df.columns = df.columns.str.strip().str.upper()

    faltantes = COLUMNAS - set(df.columns)
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan columnas {sorted(faltantes)}")

    for col in COLUMNAS - {"CODUSU"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if set(df["ANO4"].dropna()) != {anio}:
        raise ValueError(f"{ruta.name}: año interno incorrecto")
    if set(df["TRIMESTRE"].dropna()) != {trimestre}:
        raise ValueError(f"{ruta.name}: trimestre interno incorrecto")

    df = df.loc[df["AGLOMERADO"].isin(AGLOMERADOS)].copy()

    if df["PONDERA"].isna().any() or (df["PONDERA"] <= 0).any():
        raise ValueError(f"{ruta.name}: PONDERA inválido")

    return df


def detectar_atipicos(serie):
    """IQR y Z-score sobre ingresos positivos, sin borrar observaciones."""
    serie = serie.dropna()

    if serie.empty:
        return {
            "n_ingresos_positivos": 0,
            "mediana_ingreso_positivo": np.nan,
            "n_atipicos_iqr": 0,
            "pct_atipicos_iqr": np.nan,
            "n_atipicos_z": 0,
            "pct_atipicos_z": np.nan,
            "limite_superior_iqr": np.nan,
        }

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    superior = q3 + 1.5 * iqr
    inferior = q1 - 1.5 * iqr

    marca_iqr = (serie < inferior) | (serie > superior)

    if len(serie) > 1:
        media = float(serie.mean())
        desvio = float(serie.std(ddof=0))

        if desvio > 0:
            z = (serie - media) / desvio
            marca_z = z.abs() > 2
        else:
            marca_z = pd.Series(False, index=serie.index)
    else:
        marca_z = pd.Series(False, index=serie.index)

    return {
        "n_ingresos_positivos": int(len(serie)),
        "mediana_ingreso_positivo": float(serie.median()),
        "n_atipicos_iqr": int(marca_iqr.sum()),
        "pct_atipicos_iqr": float(marca_iqr.mean() * 100),
        "n_atipicos_z": int(marca_z.sum()),
        "pct_atipicos_z": float(marca_z.mean() * 100),
        "limite_superior_iqr": float(superior),
    }


def analizar(grupo, anio, trimestre, codigo):
    entrevistados = grupo.loc[grupo["H15"] == 1].copy()
    ingreso = entrevistados["P47T"]

    no_respuesta = ingreso.eq(-9) | ingreso.isna()
    positivos = ingreso[ingreso > 0]

    total_ponderado = float(entrevistados["PONDERA"].sum())
    no_respuesta_ponderada = float(
        entrevistados.loc[no_respuesta, "PONDERA"].sum()
    )

    salida = {
        "anio": anio,
        "trimestre": trimestre,
        "periodo": f"{anio}-T{trimestre}",
        "codigo_aglomerado": codigo,
        "aglomerado": AGLOMERADOS[codigo],
        "n_registros": int(len(grupo)),
        "n_entrevistados": int(len(entrevistados)),
        "n_no_respuesta": int(no_respuesta.sum()),
        "n_ingreso_cero": int(ingreso.eq(0).sum()),
        "no_respuesta_pct_muestral": (
            float(no_respuesta.mean() * 100) if len(entrevistados) else np.nan
        ),
        "no_respuesta_pct_ponderado": (
            100 * no_respuesta_ponderada / total_ponderado
            if total_ponderado > 0 else np.nan
        ),
        "id_duplicados": int(
            grupo.duplicated(["CODUSU", "NRO_HOGAR", "COMPONENTE"], keep=False).sum()
        ),
    }
    salida.update(detectar_atipicos(positivos))
    return salida


def graficar_no_respuesta(tabla):
    fig, ax = plt.subplots(figsize=(11, 5))
    periodos_orden = tabla["periodo"].drop_duplicates().tolist()
    mapa = {p: i for i, p in enumerate(periodos_orden)}

    for nombre, grupo in tabla.groupby("aglomerado", sort=False):
        ax.plot(
            [mapa[p] for p in grupo["periodo"]],
            grupo["no_respuesta_pct_ponderado"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=nombre,
        )

    ax.set_xticks(
        range(0, len(periodos_orden), 4),
        periodos_orden[::4],
        rotation=45,
        ha="right",
    )
    ax.set_title("No respuesta de P47T ponderada con PONDERA")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Porcentaje (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / "no_respuesta_ponderada.png", dpi=150)
    plt.close(fig)


def graficar_boxplot_2026_t2(df):
    datos = df.loc[
        (df["H15"] == 1) & (df["P47T"] > 0),
        ["AGLOMERADO", "P47T"],
    ].copy()

    grupos = [
        datos.loc[datos["AGLOMERADO"] == codigo, "P47T"].to_numpy()
        for codigo in AGLOMERADOS
    ]

    limites = []
    for valores in grupos:
        q1 = np.quantile(valores, 0.25)
        q3 = np.quantile(valores, 0.75)
        limites.append(q3 + 1.5 * (q3 - q1))

    zoom = max(limites) * 1.15

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        grupos,
        tick_labels=list(AGLOMERADOS.values()),
        widths=0.45,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
    )
    ax.set_ylim(0, zoom)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"${v / 1_000_000:.1f} M")
    )
    ax.set_title("Distribución de ingresos positivos - segundo trimestre de 2026")
    ax.set_xlabel("Aglomerado")
    ax.set_ylabel("Ingreso nominal total individual")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.5, 0.02,
        "Escala ampliada; los valores mayores siguen incluidos en los cálculos.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(GRAFICOS / "boxplot_2026_T2_zoom.png", dpi=150)
    plt.close(fig)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)

    filas = []
    ultimo = None

    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} bases.")

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        df = leer_base(ruta, anio, trimestre)

        if (anio, trimestre) == (2026, 2):
            ultimo = df.copy()

        for codigo in AGLOMERADOS:
            grupo = df.loc[df["AGLOMERADO"] == codigo]
            filas.append(analizar(grupo, anio, trimestre, codigo))

    tabla = pd.DataFrame(filas).sort_values(
        ["anio", "trimestre", "codigo_aglomerado"]
    )

    if len(tabla) != 82:
        raise ValueError(f"Se esperaban 82 filas y se obtuvieron {len(tabla)}")

    tabla.to_csv(
        RESULTADOS / "objetivo1_resumen.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )

    graficar_no_respuesta(tabla)

    if ultimo is None:
        raise ValueError("No se encontró 2026 T2")
    graficar_boxplot_2026_t2(ultimo)

    print("\nControl de extremos usado en el informe:")
    control = tabla.loc[
        tabla["periodo"].isin(["2016-T2", "2026-T2"]),
        ["periodo", "aglomerado", "no_respuesta_pct_ponderado",
         "pct_atipicos_iqr", "pct_atipicos_z"],
    ]
    print(control.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nLISTO: objetivo 1 reproducido.")


if __name__ == "__main__":
    main()
