"""TP Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 2, ETAPA 4: horas trabajadas e ingreso horario estimado por rama.

- Estudia personas OCUPADAS (ESTADO=1) de CABA y Partidos del GBA.
- PP3E_TOT: horas efectivamente trabajadas en la SEMANA DE REFERENCIA
  en la ocupación principal. No representa necesariamente horas habituales.
- P21: ingreso de la ocupación principal percibido en el mes de referencia.
- Mediana de horas semanales: PONDERA.
- Ingreso horario real ESTIMADO: PONDIIO, ajustado a pesos de 2026 T2.
- No modifica TXT, no imputa ingresos ni elimina registros originales.
- Registra por separado valores ausentes, cero e incompatibles.

Fuentes:
https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_registro_1T2025.pdf
https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/EPH_registro_t117.pdf

REQUISITOS: pandas, numpy, matplotlib y los scripts de etapas 1 y 3 corregida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS, RESULTADOS, GRAFICOS, buscar_archivos, guardar_csv,
)
from analisis_eph_objetivo2_etapa3_corregido import (
    RAMAS, clasificar_rama, cargar_factores, cuantil_ponderado,
)

# El año promedio tiene 52 semanas / 12 meses, usado solo como aproximación.
SEMANAS_POR_MES = 52 / 12
# El máximo físico es 24 horas * 7 días. Los valores mayores se auditan.
MAX_HORAS_SEMANA = 168
MIN_MUESTRA_GRAFICO = 30  # Regla exploratoria, NO umbral oficial INDEC.

REQUERIDAS = {
    "ANO4", "TRIMESTRE", "AGLOMERADO", "ESTADO", "PP04B_COD",
    "P21", "PONDERA", "PONDIIO", "PP3E_TOT",
}
OPCIONALES = {"PP3F_TOT", "INTENSI"}


def leer_horas(ruta, anio: int, trimestre: int) -> pd.DataFrame:
    """Lee solo las columnas necesarias, conservando los códigos de rama.

    No reutilizamos leer_trimestre() de etapa 1 porque no lee PP3E_TOT.
    """
    df = pd.read_csv(
        ruta, sep=";", decimal=",", encoding="latin-1",
        low_memory=False, dtype={"PP04B_COD": "string"},
        usecols=lambda col: col.strip().upper() in REQUERIDAS | OPCIONALES,
    )
    df.columns = df.columns.str.strip().str.upper()
    faltantes = REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan columnas {sorted(faltantes)}")
    for col in OPCIONALES - set(df.columns):
        df[col] = np.nan  # Solo para auditoría, no intervienen en los cálculos.
    for col in (REQUERIDAS | OPCIONALES) - {"PP04B_COD"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["PP04B_COD"] = df["PP04B_COD"].str.strip()
    if set(df["ANO4"].dropna()) != {anio} or set(df["TRIMESTRE"].dropna()) != {trimestre}:
        raise ValueError(f"{ruta.name}: el período interno no coincide con su nombre")
    df = df.loc[df["AGLOMERADO"].isin(AGLOMERADOS)].copy()
    if df.empty:
        raise ValueError(f"{ruta.name}: no hay registros de los aglomerados elegidos")
    if df["PONDERA"].isna().any() or (df["PONDERA"] <= 0).any():
        raise ValueError(f"{ruta.name}: PONDERA ausente o no positivo")
    if (df["PONDIIO"].dropna() < 0).any():
        raise ValueError(f"{ruta.name}: PONDIIO negativo")
    return df


def analizar_rama(parte: pd.DataFrame, datos_base: dict, factor_ipc: float) -> dict:
    """Calcula resultados para una sola rama, trimestre y aglomerado.

    Nunca divide medianas entre sí: calcula el cociente POR PERSONA y
    después su mediana ponderada entre quienes tienen datos utilizables.
    """
    horas = parte["PP3E_TOT"]
    ingreso = parte["P21"]
    peso_poblacion = parte["PONDERA"]
    peso_ingreso = parte["PONDIIO"]

    horas_validas = horas.gt(0) & horas.le(MAX_HORAS_SEMANA)
    ingresos_validos = ingreso.gt(0) & peso_ingreso.gt(0)
    pares_validos = horas_validas & ingresos_validos

    n_ingresos_validos = int(ingresos_validos.sum())
    n_pares = int(pares_validos.sum())

    # Horas: distribuciones de ocupados que sí trabajaron la semana relevada.
    horas_mediana = cuantil_ponderado(
        horas.loc[horas_validas], peso_poblacion.loc[horas_validas], 0.50
    )
    horas_media = (
        float(np.average(horas.loc[horas_validas],
                         weights=peso_poblacion.loc[horas_validas]))
        if horas_validas.any() else np.nan
    )

    # Ingreso mensual declarado / horas mensuales aproximadas de esa ocupación.
    ingreso_hora = (
        ingreso.loc[pares_validos] * factor_ipc
        / (horas.loc[pares_validos] * SEMANAS_POR_MES)
    )
    pesos_pares = peso_ingreso.loc[pares_validos]
    hora_media = (
        float(np.average(ingreso_hora, weights=pesos_pares))
        if n_pares else np.nan
    )

    # Cobertura entre quienes declaran P21 positivo y tienen peso válido.
    # Se informa tanto muestral como ponderada para detectar sesgos.
    peso_positivos = float(peso_ingreso.loc[ingresos_validos].sum())
    peso_pares = float(pesos_pares.sum())

    return {
        **datos_base,
        "n_ocupados_muestrales": int(len(parte)),
        "poblacion_ocupada_ponderada": float(peso_poblacion.sum()),
        "n_horas_ausentes": int(horas.isna().sum()),
        "n_horas_cero": int(horas.eq(0).sum()),
        "n_horas_negativas": int(horas.lt(0).sum()),
        "n_horas_mayores_168": int(horas.gt(MAX_HORAS_SEMANA).sum()),
        "n_horas_validas": int(horas_validas.sum()),
        "n_intensi_4_no_trabajo_semana": int(parte["INTENSI"].eq(4).sum()),
        "n_p21_positivo": int(ingreso.gt(0).sum()),
        "n_p21_cero": int(ingreso.eq(0).sum()),
        "n_p21_no_responde": int(ingreso.eq(-9).sum() + ingreso.isna().sum()),
        "n_p21_otros_negativos": int((ingreso.lt(0) & ingreso.ne(-9)).sum()),
        "n_p21_positivo_con_pondiio": n_ingresos_validos,
        "n_pares_horas_ingreso_validos": n_pares,
        "pct_pares_sobre_p21_positivos": (
            100 * n_pares / n_ingresos_validos if n_ingresos_validos else np.nan
        ),
        "pct_pares_ponderado_sobre_p21_positivos": (
            100 * peso_pares / peso_positivos if peso_positivos else np.nan
        ),
        "horas_semanales_media_pondera": horas_media,
        "horas_semanales_mediana_pondera": horas_mediana,
        "ingreso_horario_real_media_estimada": hora_media,
        "ingreso_horario_real_q1_estimado": cuantil_ponderado(ingreso_hora, pesos_pares, .25),
        "ingreso_horario_real_mediana_estimada": cuantil_ponderado(ingreso_hora, pesos_pares, .50),
        "ingreso_horario_real_q3_estimado": cuantil_ponderado(ingreso_hora, pesos_pares, .75),
    }


def crear_grafico(tabla: pd.DataFrame) -> None:
    """Grafica medianas horarias 2026 T2 para seis ramas relevantes.

    La selección usa el tamaño poblacional de los ocupados, no sus ingresos.
    Las celdas con menos de 30 pares quedan sin barra (pero en el CSV).
    """
    ultimo = tabla.loc[
        (tabla["periodo"] == "2026-T2")
        & (~tabla["rama"].isin(["Otras ramas", "Sin clasificar"]))
    ].copy()
    principales = (
        ultimo.groupby("rama")["poblacion_ocupada_ponderada"]
        .sum().nlargest(6).index.tolist()
    )
    if not principales:
        print("AVISO: no hay ramas para graficar ingreso horario")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(principales))
    ancho = .38
    for indice, (codigo, nombre) in enumerate(AGLOMERADOS.items()):
        parte = (
            ultimo.loc[ultimo["codigo_aglomerado"] == codigo]
            .set_index("rama").reindex(principales)
        )
        mediana = parte["ingreso_horario_real_mediana_estimada"].to_numpy(
            dtype=float, copy=True
        )
        muestras = parte["n_pares_horas_ingreso_validos"].fillna(0).to_numpy(dtype=int)
        mediana[muestras < MIN_MUESTRA_GRAFICO] = np.nan
        ax.barh(y + (indice - .5) * ancho, mediana,
                height=ancho, label=nombre)
    ax.set_yticks(y, principales)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Ingreso horario real estimado por rama - 2026 T2")
    ax.set_xlabel("Mediana ponderada, pesos de 2026 T2 por hora aproximada")
    ax.grid(axis="x", linestyle="--", alpha=.3)
    ax.legend()
    fig.text(.5, .005,
             "P21 mensual / (horas de la semana de referencia × 52/12). "
             "Omitidas barras con menos de 30 pares.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    destino = GRAFICOS / "ingreso_horario_estimado_por_rama_2026_T2.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {destino}")


def main() -> None:
    """Audita horas y estima ingresos horarios de 41 trimestres y 2 aglomerados."""
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)
    factores = cargar_factores()  # Se reutiliza IPC validado de la etapa 3.
    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} períodos para analizar horas.")
    filas = []

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        df = leer_horas(ruta, anio, trimestre)
        for codigo, nombre in AGLOMERADOS.items():
            ocupados = df.loc[
                (df["AGLOMERADO"] == codigo) & (df["ESTADO"] == 1)
            ].copy()
            if ocupados.empty:
                raise ValueError(f"{anio} T{trimestre}: sin ocupados en {nombre}")
            ocupados["rama"] = ocupados["PP04B_COD"].map(clasificar_rama)
            periodo = f"{anio}-T{trimestre}"
            factor = factores.get((periodo, nombre))
            if factor is None:
                raise ValueError(f"Falta factor IPC para {periodo}, {nombre}")

            for rama in RAMAS:
                parte = ocupados.loc[ocupados["rama"] == rama]
                base = {
                    "anio": anio, "trimestre": trimestre,
                    "periodo": periodo, "codigo_aglomerado": codigo,
                    "aglomerado": nombre, "rama": rama,
                }
                filas.append(analizar_rama(parte, base, factor))

    tabla = pd.DataFrame(filas)
    esperado = 82 * len(RAMAS)
    if len(tabla) != esperado or tabla.duplicated(
        ["periodo", "aglomerado", "rama"]
    ).any():
        raise ValueError(f"Se esperaban {esperado} filas únicas; hay {len(tabla)}")
    for col in ["pct_pares_sobre_p21_positivos", "pct_pares_ponderado_sobre_p21_positivos"]:
        serie = tabla[col].dropna()
        if ((serie < 0) | (serie > 100.001)).any():
            raise ValueError(f"Porcentaje de cobertura imposible en {col}")

    guardar_csv(tabla, "horas_ingresos_por_rama_2016_2026.csv")
    # Resumen de auditoría por trimestre/aglomerado. Las categorías son
    # excluyentes: sumamos los conteos sin duplicar registros dentro de un período.
    campos = [
        "n_ocupados_muestrales", "n_horas_ausentes", "n_horas_cero",
        "n_horas_negativas", "n_horas_mayores_168", "n_horas_validas",
        "n_intensi_4_no_trabajo_semana", "n_p21_positivo",
        "n_p21_no_responde", "n_p21_positivo_con_pondiio",
        "n_pares_horas_ingreso_validos",
    ]
    auditoria = tabla.groupby(["anio", "trimestre", "periodo", "aglomerado"], as_index=False)[campos].sum()
    auditoria["pct_pares_sobre_p21_positivos"] = np.where(
        auditoria["n_p21_positivo_con_pondiio"] > 0,
        100 * auditoria["n_pares_horas_ingreso_validos"] /
        auditoria["n_p21_positivo_con_pondiio"], np.nan,
    )
    guardar_csv(auditoria, "auditoria_horas_ingresos_2016_2026.csv")
    print("\nAUDITORÍA ACUMULADA (no son personas únicas):")
    for campo in ["n_ocupados_muestrales", "n_horas_ausentes", "n_horas_cero",
                  "n_horas_negativas", "n_horas_mayores_168", "n_pares_horas_ingreso_validos"]:
        print(f"  {campo}: {int(auditoria[campo].sum())}")
    print("Celdas de rama con menos de 30 pares:",
          int((tabla["n_pares_horas_ingreso_validos"] < MIN_MUESTRA_GRAFICO).sum()),
          "/", len(tabla))
    crear_grafico(tabla)
    print("\nLISTO: horas e ingreso horario estimado. Revisar cobertura antes de interpretar.")


if __name__ == "__main__":
    main()
