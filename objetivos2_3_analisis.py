"""Objetivos 2 y 3 - Análisis multivariado y visualización EPH 2016 T2-2026 T2.

Incluye:
- tasas de actividad, empleo y desocupación;
- evolución del ingreso individual real (P47T);
- empleo por edad;
- educación e ingresos en los extremos 2016 T2 y 2026 T2;
- rama de actividad, ingreso mensual, horas e ingreso horario aproximado;
- gráficos utilizados en el informe.

No imputa ingresos ni modifica las bases originales.

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

IPC_2016_GBA = {
    "2016-04": 85.5, "2016-05": 89.1, "2016-06": 91.9,
    "2016-07": 93.7, "2016-08": 93.9, "2016-09": 95.0,
    "2016-10": 97.2, "2016-11": 98.8, "2016-12": 100.0,
}

EDUCACION = {
    7: "Sin instrucción",
    1: "Primaria incompleta",
    2: "Primario completo o secundario incompleto",
    3: "Primario completo o secundario incompleto",
    4: "Secundario completo o superior incompleto",
    5: "Secundario completo o superior incompleto",
    6: "Superior completo",
}

GRUPOS_EDUCACION = [
    "Sin instrucción",
    "Primaria incompleta",
    "Primario completo o secundario incompleto",
    "Secundario completo o superior incompleto",
    "Superior completo",
]

GRUPOS_EDAD = [
    ("14 a 29 años", 14, 29),
    ("30 a 64 años", 30, 64),
    ("65 años o más", 65, 200),
]

RAMAS = [
    "Actividades primarias",
    "Industria manufacturera",
    "Construcción",
    "Comercio",
    "Transporte y almacenamiento",
    "Alojamiento y comidas",
    "Información y comunicación",
    "Finanzas, seguros e inmobiliarias",
    "Actividades profesionales y técnicas",
    "Servicios administrativos y de apoyo",
    "Administración pública",
    "Enseñanza",
    "Salud y servicios sociales",
    "Servicio doméstico",
    "Otras ramas",
    "Sin clasificar",
]

CLASES_CERO_INICIAL = {
    "0101", "0102", "0103", "0104", "0105",
    "0200", "0300", "0500", "0600", "0700", "0800", "0900",
}

SEMANAS_POR_MES = 52 / 12
MAX_HORAS = 168
MIN_MUESTRA = 30

COLUMNAS = {
    "ANO4", "TRIMESTRE", "AGLOMERADO", "H15",
    "CH06", "NIVEL_ED", "ESTADO",
    "P47T", "PONDERA", "PONDII",
    "P21", "PONDIIO", "PP04B_COD", "PP3E_TOT",
}


# ============================================================
# UTILIDADES DE LECTURA
# ============================================================

def periodos():
    return [
        (a, t)
        for a in range(2016, 2027)
        for t in range(1, 5)
        if (2016, 2) <= (a, t) <= (2026, 2)
    ]


def buscar_archivos():
    encontrados = {}
    for ruta in DATOS.glob("*.txt"):
        m = PATRON.fullmatch(ruta.name)
        if not m:
            continue
        clave = (2000 + int(m.group(2)), int(m.group(1)))
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
        dtype={"PP04B_COD": "string"},
        usecols=lambda c: c.strip().upper() in COLUMNAS,
    )
    df.columns = df.columns.str.strip().str.upper()

    faltantes = COLUMNAS - set(df.columns)
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan {sorted(faltantes)}")

    for col in COLUMNAS - {"PP04B_COD"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PP04B_COD"] = df["PP04B_COD"].astype("string").str.strip()

    if set(df["ANO4"].dropna()) != {anio}:
        raise ValueError(f"{ruta.name}: año incorrecto")
    if set(df["TRIMESTRE"].dropna()) != {trimestre}:
        raise ValueError(f"{ruta.name}: trimestre incorrecto")

    df = df.loc[df["AGLOMERADO"].isin(AGLOMERADOS)].copy()

    if df["PONDERA"].isna().any() or (df["PONDERA"] <= 0).any():
        raise ValueError(f"{ruta.name}: PONDERA inválido")

    return df


# ============================================================
# IPC Y ESTADÍSTICAS PONDERADAS
# ============================================================

def _convertir_numeros(serie):
    texto = serie.astype(str).str.strip().str.replace(" ", "", regex=False)
    con_coma = texto.str.contains(",", regex=False)
    texto = texto.where(~con_coma, texto.str.replace(".", "", regex=False))
    return pd.to_numeric(
        texto.str.replace(",", ".", regex=False),
        errors="coerce",
    )


def cargar_ipc():
    """IPC-GBA mensual empalmado y reexpresado con dic-2016=100."""
    ruta = DATOS / "serie_ipc_divisiones.csv"
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta {ruta}")

    datos = None
    for sep in (";", ","):
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                candidato = pd.read_csv(
                    ruta, sep=sep, encoding=encoding,
                    dtype=str, keep_default_na=False,
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            candidato.columns = candidato.columns.str.strip().str.lower()
            if len(candidato.columns) > 1:
                datos = candidato
                break
        if datos is not None:
            break

    if datos is None:
        raise ValueError("No se pudo leer el IPC")

    requeridas = {
        "codigo", "descripcion", "clasificador",
        "periodo", "indice_ipc", "region",
    }
    if not requeridas.issubset(datos.columns):
        raise ValueError("serie_ipc_divisiones.csv no tiene el formato esperado")

    seleccion = datos.loc[
        datos["codigo"].str.strip().eq("0")
        & datos["descripcion"].str.strip().str.upper().eq("NIVEL GENERAL")
        & datos["region"].str.strip().str.upper().eq("GBA")
        & datos["clasificador"].str.contains(
            "Nivel general", case=False, regex=False
        )
    ].copy()

    fechas = pd.to_datetime(
        seleccion["periodo"], format="%Y%m", errors="coerce"
    )

    nuevo = pd.DataFrame({
        "mes": fechas.dt.strftime("%Y-%m"),
        "ipc": _convertir_numeros(seleccion["indice_ipc"]),
    }).dropna()

    nuevo = nuevo.loc[nuevo["mes"] >= "2016-12"].copy()

    ipc_dic = float(
        nuevo.loc[nuevo["mes"] == "2016-12", "ipc"].iloc[0]
    )
    nuevo["ipc"] = nuevo["ipc"] / ipc_dic * 100

    viejo = pd.DataFrame(
        [{"mes": mes, "ipc": valor} for mes, valor in IPC_2016_GBA.items()]
    )

    completo = pd.concat(
        [viejo.loc[viejo["mes"] < "2016-12"], nuevo],
        ignore_index=True,
    )

    esperado = pd.period_range(
        "2016-04", "2026-06", freq="M"
    ).astype(str)

    tabla = completo.set_index("mes").reindex(esperado)
    if tabla["ipc"].isna().any():
        faltan = tabla.index[tabla["ipc"].isna()].tolist()
        raise ValueError(f"IPC incompleto: {faltan}")

    tabla = tabla.reset_index(names="mes")
    tabla["anio"] = tabla["mes"].str[:4].astype(int)
    tabla["trimestre"] = (
        (tabla["mes"].str[5:7].astype(int) - 1) // 3 + 1
    )
    return tabla


def factores_ipc():
    ipc = cargar_ipc()
    trimestral = ipc.groupby(
        ["anio", "trimestre"], as_index=False
    )["ipc"].mean()

    base = float(
        trimestral.loc[
            (trimestral["anio"] == 2026)
            & (trimestral["trimestre"] == 2),
            "ipc",
        ].iloc[0]
    )

    return {
        (int(f.anio), int(f.trimestre)): base / float(f.ipc)
        for f in trimestral.itertuples()
        if (2016, 2) <= (int(f.anio), int(f.trimestre)) <= (2026, 2)
    }


def cuantil_ponderado(valores, pesos, p=0.5):
    ok = valores.notna() & pesos.notna() & (pesos > 0)
    if not ok.any():
        return np.nan

    x = valores.loc[ok].to_numpy(dtype=float)
    w = pesos.loc[ok].to_numpy(dtype=float)

    orden = np.argsort(x, kind="stable")
    x = x[orden]
    w = w[orden]

    posicion = np.searchsorted(
        np.cumsum(w),
        p * w.sum(),
        side="left",
    )
    return float(x[min(posicion, len(x) - 1)])


# ============================================================
# TASAS LABORALES E INGRESO REAL
# ============================================================

def tasas_laborales(grupo, periodo, codigo):
    estado = grupo["ESTADO"]
    pesos = grupo["PONDERA"]

    poblacion = float(pesos.sum())
    ocupados = float(pesos.loc[estado == 1].sum())
    desocupados = float(pesos.loc[estado == 2].sum())
    pea = ocupados + desocupados

    return {
        "periodo": periodo,
        "aglomerado": AGLOMERADOS[codigo],
        "tasa_actividad": 100 * pea / poblacion,
        "tasa_empleo": 100 * ocupados / poblacion,
        "tasa_desocupacion": 100 * desocupados / pea,
    }


def ingreso_real(grupo, periodo, codigo, factor):
    entrevistados = grupo.loc[grupo["H15"] == 1]
    positivos = entrevistados.loc[
        entrevistados["P47T"].gt(0)
        & entrevistados["PONDII"].gt(0)
    ]

    # Medidas de posición ponderadas sobre P47T positivo.
    q1_nominal = cuantil_ponderado(
        positivos["P47T"], positivos["PONDII"], p=0.25
    )
    mediana_nominal = cuantil_ponderado(
        positivos["P47T"], positivos["PONDII"], p=0.50
    )
    q3_nominal = cuantil_ponderado(
        positivos["P47T"], positivos["PONDII"], p=0.75
    )

    return {
        "periodo": periodo,
        "aglomerado": AGLOMERADOS[codigo],
        "n_ingresos_positivos": int(len(positivos)),
        "q1_nominal": q1_nominal,
        "mediana_nominal": mediana_nominal,
        "q3_nominal": q3_nominal,
        "q1_real_2026_t2": q1_nominal * factor,
        "mediana_real_2026_t2": mediana_nominal * factor,
        "q3_real_2026_t2": q3_nominal * factor,
    }


def empleo_por_edad(grupo, periodo, codigo):
    filas = []
    for nombre, minimo, maximo in GRUPOS_EDAD:
        parte = grupo.loc[grupo["CH06"].between(minimo, maximo)]
        total = float(parte["PONDERA"].sum())
        ocupados = float(
            parte.loc[parte["ESTADO"] == 1, "PONDERA"].sum()
        )
        filas.append({
            "periodo": periodo,
            "aglomerado": AGLOMERADOS[codigo],
            "grupo_edad": nombre,
            "tasa_empleo": 100 * ocupados / total if total > 0 else np.nan,
        })
    return filas


# ============================================================
# EDUCACIÓN E INGRESOS
# ============================================================

def analizar_educacion(df, periodo, codigo, factor):
    """Ocupados de 25 a 64 años; universo usado en el informe."""
    ocupados = df.loc[
        (df["AGLOMERADO"] == codigo)
        & df["ESTADO"].eq(1)
        & df["CH06"].between(25, 64)
    ].copy()

    ocupados["grupo_educacion"] = ocupados["NIVEL_ED"].map(EDUCACION)
    total_peso = float(ocupados["PONDERA"].sum())

    filas = []

    for grupo in GRUPOS_EDUCACION:
        parte = ocupados.loc[ocupados["grupo_educacion"] == grupo].copy()
        entrevistados = parte.loc[parte["H15"] == 1].copy()

        positivos = entrevistados.loc[
            entrevistados["P47T"].gt(0)
            & entrevistados["PONDII"].gt(0)
        ]

        mediana = cuantil_ponderado(
            positivos["P47T"], positivos["PONDII"]
        )

        peso_grupo = float(parte["PONDERA"].sum())

        filas.append({
            "periodo": periodo,
            "aglomerado": AGLOMERADOS[codigo],
            "grupo_educacion": grupo,
            "n_muestra": int(len(parte)),
            "porcentaje_dentro_universo": (
                100 * peso_grupo / total_peso if total_peso > 0 else np.nan
            ),
            "n_ingreso_positivo_pondii_valido": int(len(positivos)),
            "mediana_real_2026_t2": mediana * factor,
            "advertencia_muestra_pequena": len(positivos) < MIN_MUESTRA,
        })

    return filas


# ============================================================
# RAMAS, HORAS E INGRESO HORARIO
# ============================================================

def clasificar_rama(codigo):
    if pd.isna(codigo):
        return "Sin clasificar"

    codigo = str(codigo).strip()
    if not codigo.isdecimal() or codigo == "9999":
        return "Sin clasificar"

    if len(codigo) == 3:
        candidato = codigo.zfill(4)
        if candidato not in CLASES_CERO_INICIAL:
            return "Sin clasificar"
        codigo = candidato

    if len(codigo) == 4:
        division = int(codigo[:2])
    elif len(codigo) == 2:
        division = int(codigo)
    else:
        return "Sin clasificar"

    if division in (1, 2, 3, 5, 6, 7, 8, 9):
        return "Actividades primarias"
    if 10 <= division <= 33:
        return "Industria manufacturera"
    if division == 40:
        return "Construcción"
    if division in (45, 48):
        return "Comercio"
    if 49 <= division <= 53:
        return "Transporte y almacenamiento"
    if division in (55, 56):
        return "Alojamiento y comidas"
    if 58 <= division <= 63:
        return "Información y comunicación"
    if division in (64, 65, 66, 68):
        return "Finanzas, seguros e inmobiliarias"
    if 69 <= division <= 75:
        return "Actividades profesionales y técnicas"
    if 77 <= division <= 82:
        return "Servicios administrativos y de apoyo"
    if division in (83, 84):
        return "Administración pública"
    if division == 85:
        return "Enseñanza"
    if division in (86, 87, 88):
        return "Salud y servicios sociales"
    if division == 97:
        return "Servicio doméstico"
    if division in (35, 36, 37, 38, 39, 90, 91, 92, 93, 94, 95, 96, 98, 99):
        return "Otras ramas"
    return "Sin clasificar"


def analizar_ramas(df, periodo, codigo, factor):
    ocupados = df.loc[
        (df["AGLOMERADO"] == codigo)
        & df["ESTADO"].eq(1)
        & df["CH06"].between(25, 64)
    ].copy()

    ocupados["rama"] = ocupados["PP04B_COD"].map(clasificar_rama)
    total_peso = float(ocupados["PONDERA"].sum())

    filas = []

    for rama in RAMAS:
        parte = ocupados.loc[ocupados["rama"] == rama].copy()

        peso_rama = float(parte["PONDERA"].sum())
        ingreso = parte["P21"]
        horas = parte["PP3E_TOT"]

        ingreso_valido = (
            ingreso.gt(0)
            & parte["PONDIIO"].notna()
            & parte["PONDIIO"].gt(0)
        )

        horas_validas = horas.gt(0) & horas.le(MAX_HORAS)
        pares_validos = ingreso_valido & horas_validas

        mediana_ingreso = cuantil_ponderado(
            ingreso.loc[ingreso_valido] * factor,
            parte.loc[ingreso_valido, "PONDIIO"],
        )

        mediana_horas = cuantil_ponderado(
            horas.loc[horas_validas],
            parte.loc[horas_validas, "PONDERA"],
        )

        ingreso_horario = (
            ingreso.loc[pares_validos] * factor
            / (horas.loc[pares_validos] * SEMANAS_POR_MES)
        )

        mediana_horaria = cuantil_ponderado(
            ingreso_horario,
            parte.loc[pares_validos, "PONDIIO"],
        )

        no_respuesta = ingreso.eq(-9) | ingreso.isna()

        filas.append({
            "periodo": periodo,
            "aglomerado": AGLOMERADOS[codigo],
            "rama": rama,
            "n_ocupados_muestrales": int(len(parte)),
            "porcentaje_ocupados_ponderado": (
                100 * peso_rama / total_peso if total_peso > 0 else np.nan
            ),
            "n_ingresos_positivos_validos": int(ingreso_valido.sum()),
            "ingreso_real_mediana_2026_t2": mediana_ingreso,
            "no_respuesta_p21_pct_pondera": (
                100 * float(parte.loc[no_respuesta, "PONDERA"].sum()) / peso_rama
                if peso_rama > 0 else np.nan
            ),
            "n_pares_horas_ingreso_validos": int(pares_validos.sum()),
            "horas_semanales_mediana_pondera": mediana_horas,
            "ingreso_horario_real_mediana_estimada": mediana_horaria,
        })

    return filas


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_tasas(tabla):
    periodos_orden = tabla["periodo"].drop_duplicates().tolist()
    xmap = {p: i for i, p in enumerate(periodos_orden)}

    fig, ax = plt.subplots(figsize=(12, 6))

    estilos = {
        "tasa_actividad": "-",
        "tasa_empleo": "--",
        "tasa_desocupacion": ":",
    }
    nombres = {
        "tasa_actividad": "Actividad",
        "tasa_empleo": "Empleo",
        "tasa_desocupacion": "Desocupación",
    }

    for aglomerado, grupo in tabla.groupby("aglomerado", sort=False):
        for columna in estilos:
            ax.plot(
                [xmap[p] for p in grupo["periodo"]],
                grupo[columna],
                linestyle=estilos[columna],
                linewidth=1.5,
                label=f"{aglomerado} - {nombres[columna]}",
            )

    ax.set_xticks(
        range(0, len(periodos_orden), 4),
        periodos_orden[::4],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Porcentaje (%)")
    ax.set_xlabel("Trimestre")
    ax.set_title("Tasas laborales 2016-2026")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "tasas_laborales_2016_2026.png", dpi=150)
    plt.close(fig)


def grafico_ingreso_real(tabla):
    fig, ax = plt.subplots(figsize=(11, 5))

    periodos_orden = tabla["periodo"].drop_duplicates().tolist()
    posiciones = {periodo: i for i, periodo in enumerate(periodos_orden)}

    for nombre, grupo in tabla.groupby("aglomerado", sort=False):
        grupo = grupo.copy()
        x = np.array([posiciones[p] for p in grupo["periodo"]], dtype=float)

        linea, = ax.plot(
            x,
            grupo["mediana_real_2026_t2"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=f"{nombre} - mediana",
        )

        # Banda Q1-Q3: contiene el 50 % central de los ingresos positivos.
        ax.fill_between(
            x,
            grupo["q1_real_2026_t2"].to_numpy(dtype=float),
            grupo["q3_real_2026_t2"].to_numpy(dtype=float),
            alpha=0.15,
            color=linea.get_color(),
            label=f"{nombre} - Q1 a Q3",
        )

    marcas = list(range(0, len(periodos_orden), 4))
    ax.set_xticks(
        marcas,
        [periodos_orden[i] for i in marcas],
        rotation=45,
        ha="right",
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"${v / 1_000_000:.1f} M")
    )
    ax.set_title(
        "Ingresos individuales positivos: mediana y rango intercuartílico"
    )
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Pesos constantes del segundo trimestre de 2026")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        GRAFICOS / "ingresos_reales_mediana_ponderada.png",
        dpi=150,
    )
    plt.close(fig)


def grafico_educacion(tabla):
    """Gráfico comparativo de ingresos reales por nivel educativo.

    Se muestran por separado 2016-T2 y 2026-T2 para evitar mezclar
    los cuatro grupos en un único eje. Dentro de cada período se
    comparan CABA y Partidos del GBA con colores diferentes.
    """

    confiables = [
        "Primario completo o secundario incompleto",
        "Secundario completo o superior incompleto",
        "Superior completo",
    ]

    etiquetas_cortas = [
        "Primario completo /\nsecundario incompleto",
        "Secundario completo /\nsuperior incompleto",
        "Superior completo",
    ]

    datos = tabla.loc[
        tabla["grupo_educacion"].isin(confiables)
    ].copy()

    fig, ejes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        sharey=True,
    )

    ancho = 0.36
    x = np.arange(len(confiables))

    colores = {
        "CABA": "tab:blue",
        "Partidos del GBA": "tab:orange",
    }

    for ax, periodo in zip(ejes, ("2016-T2", "2026-T2")):
        for desplazamiento, aglomerado in zip(
            (-ancho / 2, ancho / 2),
            AGLOMERADOS.values(),
        ):
            parte = (
                datos.loc[
                    (datos["periodo"] == periodo)
                    & (datos["aglomerado"] == aglomerado)
                ]
                .set_index("grupo_educacion")
                .reindex(confiables)
            )

            valores = (
                parte["mediana_real_2026_t2"]
                .to_numpy(dtype=float)
            )

            barras = ax.bar(
                x + desplazamiento,
                valores,
                width=ancho,
                label=aglomerado,
                color=colores[aglomerado],
            )

            # Etiquetas de valor sobre las barras.
            for barra, valor in zip(barras, valores):
                if np.isnan(valor):
                    continue

                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height(),
                    f"${valor / 1_000_000:.2f} M",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(
            "2016 T2" if periodo == "2016-T2" else "2026 T2"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            etiquetas_cortas,
            fontsize=8,
        )
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )
        ax.set_axisbelow(True)

    ejes[0].set_ylabel(
        "Mediana de ingreso real, pesos del segundo trimestre de 2026"
    )

    ejes[0].yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"${v / 1_000_000:.1f} M"
        )
    )

    # Una sola leyenda para toda la figura.
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )

    fig.suptitle(
        "Ingresos reales por nivel educativo - ocupados de 25 a 64 años",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.88]
    )

    fig.savefig(
        GRAFICOS
        / "ingresos_reales_educacion_ocupados_25_64_extremos.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def grafico_ramas(tabla):
    base = tabla.loc[
        ~tabla["rama"].isin(["Otras ramas", "Sin clasificar"])
    ].copy()

    confiables = (
        base.groupby("rama")
        .agg(
            min_ingresos=("n_ingresos_positivos_validos", "min"),
            peso_2026=(
                "porcentaje_ocupados_ponderado",
                lambda s: float(s.iloc[-1]) if len(s) else 0,
            ),
        )
    )
    confiables = confiables.loc[
        confiables["min_ingresos"] >= MIN_MUESTRA
    ].index.tolist()

    datos = base.loc[base["rama"].isin(confiables)].copy()

    orden = (
        datos.loc[datos["periodo"] == "2026-T2"]
        .groupby("rama")["porcentaje_ocupados_ponderado"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    figuras = []
    for periodo in ("2016-T2", "2026-T2"):
        for aglomerado in AGLOMERADOS.values():
            parte = datos.loc[
                (datos["periodo"] == periodo)
                & (datos["aglomerado"] == aglomerado)
            ].set_index("rama")
            figuras.append((
                f"{aglomerado}\n{periodo}",
                [parte.loc[r, "ingreso_real_mediana_2026_t2"] for r in orden],
            ))

    x = np.arange(len(orden))
    ancho = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (nombre, valores) in enumerate(figuras):
        ax.bar(x + (i - 1.5) * ancho, valores, width=ancho, label=nombre)

    ax.set_xticks(x, [r.replace(" y ", "\ny ") for r in orden], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"${v / 1_000_000:.1f} M")
    )
    ax.set_title("Ingreso real por rama - ocupados de 25 a 64 años")
    ax.set_ylabel("Mediana de P21, pesos de 2026 T2")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        GRAFICOS / "ingreso_real_ramas_ocupados_25_64_extremos.png",
        dpi=150,
    )
    plt.close(fig)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)

    factores = factores_ipc()

    tasas = []
    ingresos = []
    edades = []
    educacion = []
    ramas = []

    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} bases.")

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        df = leer_base(ruta, anio, trimestre)
        periodo = f"{anio}-T{trimestre}"
        factor = factores[(anio, trimestre)]

        for codigo in AGLOMERADOS:
            grupo = df.loc[df["AGLOMERADO"] == codigo]
            tasas.append(tasas_laborales(grupo, periodo, codigo))
            ingresos.append(ingreso_real(grupo, periodo, codigo, factor))
            edades.extend(empleo_por_edad(grupo, periodo, codigo))

            if (anio, trimestre) in {(2016, 2), (2026, 2)}:
                educacion.extend(
                    analizar_educacion(df, periodo, codigo, factor)
                )
                ramas.extend(
                    analizar_ramas(df, periodo, codigo, factor)
                )

    tasas = pd.DataFrame(tasas)
    ingresos = pd.DataFrame(ingresos)
    edades = pd.DataFrame(edades)
    educacion = pd.DataFrame(educacion)
    ramas = pd.DataFrame(ramas)

    tasas.to_csv(
        RESULTADOS / "tasas_laborales_2016_2026.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    ingresos.to_csv(
        RESULTADOS / "ingresos_reales_2016_2026.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    edades.to_csv(
        RESULTADOS / "empleo_por_edad_2016_2026.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    educacion.to_csv(
        RESULTADOS / "educacion_ingresos_extremos.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    ramas.to_csv(
        RESULTADOS / "ramas_ocupados_25_64_extremos.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )

    grafico_tasas(tasas)
    grafico_ingreso_real(ingresos)
    grafico_educacion(educacion)
    grafico_ramas(ramas)

    print("\nControles principales:")
    print(
        tasas.loc[
            tasas["periodo"].isin(["2016-T2", "2026-T2"])
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )
    print(
        ingresos.loc[
            ingresos["periodo"].isin(["2016-T2", "2026-T2"])
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )

    print("\nLISTO: objetivos 2 y 3 reproducidos.")


if __name__ == "__main__":
    main()
