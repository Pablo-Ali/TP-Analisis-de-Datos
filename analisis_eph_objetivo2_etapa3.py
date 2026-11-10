"""TP de Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 2, ETAPA 3: rama económica e ingreso de la ocupación principal.

- Procesa 41 trimestres de CABA y Partidos del GBA.
- Analiza únicamente personas ocupadas.
- PP04B_COD identifica la actividad del establecimiento donde trabajan.
- Agrupa las ramas según la clasificación CAES-Mercosur 1.0.
- Conserva los códigos no clasificables en una categoría separada.
- Calcula la participación de cada rama utilizando PONDERA.
- Calcula media, mediana, Q1 y Q3 del ingreso P21 utilizando PONDIIO.
- Expresa los ingresos en pesos constantes de 2026 T2.
- No modifica las bases originales ni imputa valores faltantes.

FUENTES:
INDEC, CAES-Mercosur 1.0:
https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/caes_mercosur_1.0.pdf

INDEC, ENGHo 2017-2018, Informe de ingresos, cuadro 16:
https://www.indec.gob.ar/ftp/cuadros/sociedad/engho_2017_2018_informe_ingresos.pdf

REQUISITOS:
- pandas, numpy y matplotlib.
- analisis_eph_objetivo2_etapa1.py en la misma carpeta.
- Carpeta datos con los 41 archivos TXT.
- CSV de ingresos reales generado en el objetivo 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Reutilizamos las funciones de lectura de la primera etapa.
from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS,
    RESULTADOS,
    GRAFICOS,
    buscar_archivos,
    leer_trimestre,
    guardar_csv,
)


# ============================================================
# 1. CLASIFICACIÓN DE RAMAS ECONÓMICAS
# ============================================================

# Agrupación adaptada del cuadro 16 de la ENGHo 2017-2018.
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


def clasificar_rama(codigo: object) -> str:
    """Clasifica una actividad mediante su código CAES-Mercosur.

    Los dos primeros dígitos indican la división económica.

    Se exige un código de cuatro dígitos porque el objetivo es
    identificar actividades codificadas completamente.

    No completamos códigos con ceros ni inventamos sectores.
    """

    if (
        not isinstance(codigo, str)
        or len(codigo) != 4
        or not codigo.isdecimal()
    ):
        return "Sin clasificar"

    # Actividad no especificada o descripción insuficiente.
    if codigo == "9999":
        return "Sin clasificar"

    division = int(codigo[:2])

    # A y B: agricultura, pesca, minería, etc.
    if division in (1, 2, 3, 5, 6, 7, 8, 9):
        return "Actividades primarias"

    # C: industria manufacturera.
    if 10 <= division <= 33:
        return "Industria manufacturera"

    # F: construcción.
    if division == 40:
        return "Construcción"

    # G: comercio y reparación de vehículos.
    if division in (45, 48):
        return "Comercio"

    # H: transporte y almacenamiento.
    if 49 <= division <= 53:
        return "Transporte y almacenamiento"

    # I: alojamiento y gastronomía.
    if division in (55, 56):
        return "Alojamiento y comidas"

    # J: información y comunicación.
    if 58 <= division <= 63:
        return "Información y comunicación"

    # K y L: finanzas, seguros y actividades inmobiliarias.
    if division in (64, 65, 66, 68):
        return "Finanzas, seguros e inmobiliarias"

    # M: actividades profesionales, científicas y técnicas.
    if 69 <= division <= 75:
        return "Actividades profesionales y técnicas"

    # N: actividades administrativas y de apoyo.
    if 77 <= division <= 82:
        return "Servicios administrativos y de apoyo"

    # O: administración pública, defensa y planes sociales.
    if division in (83, 84):
        return "Administración pública"

    # P: enseñanza.
    if division == 85:
        return "Enseñanza"

    # Q: salud humana y servicios sociales.
    if division in (86, 87, 88):
        return "Salud y servicios sociales"

    # T: actividades de los hogares.
    if division in (97, 98):
        return "Servicio doméstico"

    # D, E, R, S y U: agrupadas como otras ramas.
    if division in (
        35, 36, 37, 38, 39,
        90, 91, 92, 93, 94, 95, 96, 99
    ):
        return "Otras ramas"

    return "Sin clasificar"


# ============================================================
# 2. FACTORES DE INFLACIÓN
# ============================================================

def cargar_factores() -> dict[tuple[str, str], float]:
    """Recupera los factores de inflación del objetivo 1.

    No vuelve a descargar ni calcular el IPC.

    Cada factor permite expresar un ingreso nominal en pesos
    constantes del segundo trimestre de 2026.
    """

    ruta = (
        RESULTADOS
        / "resumen_ponderado_ingresos_reales_2016_2026.csv"
    )

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró {ruta}. "
            "Es necesario el CSV de ingresos reales del objetivo 1."
        )

    tabla = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    campos = {
        "periodo",
        "aglomerado",
        "factor_a_pesos_2026_t2",
    }

    faltantes = campos - set(tabla.columns)

    if faltantes:
        raise ValueError(
            f"Faltan campos en el CSV de IPC: {sorted(faltantes)}"
        )

    # 41 trimestres por dos aglomerados.
    if (
        len(tabla) != 82
        or tabla.duplicated(
            ["periodo", "aglomerado"]
        ).any()
    ):
        raise ValueError(
            "Se esperaban 82 factores sin duplicados."
        )

    factores = pd.to_numeric(
        tabla["factor_a_pesos_2026_t2"],
        errors="coerce",
    )

    if factores.isna().any() or (factores <= 0).any():
        raise ValueError(
            "Se encontraron factores de inflación inválidos."
        )

    tabla["factor"] = factores

    return {
        (fila.periodo, fila.aglomerado): float(fila.factor)
        for fila in tabla.itertuples()
    }


# ============================================================
# 3. ESTADÍSTICAS PONDERADAS DE INGRESOS
# ============================================================

def cuantil_ponderado(
    serie: pd.Series,
    pesos: pd.Series,
    p: float,
) -> float:
    """Calcula un cuantil ponderado.

    Ordena los ingresos de menor a mayor y acumula los pesos.

    Q1 corresponde a p=0.25.
    Mediana corresponde a p=0.50.
    Q3 corresponde a p=0.75.

    Se utiliza el mismo criterio que en el objetivo 1.
    """

    validos = (
        serie.notna()
        & pesos.notna()
        & (pesos > 0)
    )

    if not validos.any():
        return np.nan

    valores = serie[validos].to_numpy(dtype=float)
    ponderadores = pesos[validos].to_numpy(dtype=float)

    orden = np.argsort(valores, kind="stable")

    valores = valores[orden]
    ponderadores = ponderadores[orden]

    acumulados = np.cumsum(ponderadores)

    posicion = np.searchsorted(
        acumulados,
        p * ponderadores.sum(),
        side="left",
    )

    return float(
        valores[min(posicion, len(valores) - 1)]
    )


# ============================================================
# 4. ANÁLISIS POR AGLOMERADO Y TRIMESTRE
# ============================================================

def analizar_aglomerado(
    grupo: pd.DataFrame,
    anio: int,
    trimestre: int,
    codigo_aglomerado: int,
    factor_ipc: float,
) -> tuple[list[dict], list[dict]]:
    """Analiza la distribución del empleo y los ingresos por rama.

    Universo de referencia: personas ocupadas, ESTADO=1.

    Distribución del empleo: PONDERA.
    Ingreso de ocupación principal P21: PONDIIO.

    Conserva por separado P21=0, P21=-9 e ingresos positivos.
    """

    ocupados = grupo.loc[
        grupo["ESTADO"] == 1
    ].copy()

    columnas = {
        "PP04B_COD",
        "P21",
        "PONDIIO",
        "PONDERA",
    }

    faltantes = columnas - set(ocupados.columns)

    if faltantes:
        raise ValueError(
            f"Faltan variables necesarias: {sorted(faltantes)}"
        )

    ocupados["rama"] = (
        ocupados["PP04B_COD"].map(clasificar_rama)
    )

    total_ocupados_ponderado = float(
        ocupados["PONDERA"].sum()
    )

    if total_ocupados_ponderado <= 0:
        raise ValueError(
            f"{anio} T{trimestre}: no hay población ocupada válida."
        )

    # Nunca utilizamos ponderadores de ingresos negativos.
    if (
        ocupados["PONDIIO"].notna()
        & (ocupados["PONDIIO"] < 0)
    ).any():
        raise ValueError(
            f"{anio} T{trimestre}: PONDIIO negativo."
        )

    # Se admite -9 como no respuesta, pero no otros negativos.
    otros_negativos = (
        ocupados["P21"].lt(0)
        & ocupados["P21"].ne(-9)
    )

    if otros_negativos.any():
        raise ValueError(
            f"{anio} T{trimestre}: P21 contiene negativos inesperados."
        )

    periodo = f"{anio}-T{trimestre}"
    nombre = AGLOMERADOS[codigo_aglomerado]

    datos_base = {
        "anio": anio,
        "trimestre": trimestre,
        "periodo": periodo,
        "codigo_aglomerado": codigo_aglomerado,
        "aglomerado": nombre,
    }

    filas = []

    for rama in RAMAS:

        parte = ocupados.loc[
            ocupados["rama"] == rama
        ]

        # Ingresos positivos declarados.
        positivos = parte.loc[
            parte["P21"] > 0
        ]

        # Solo los ingresos con ponderador válido contribuyen
        # a las estadísticas ponderadas.
        validos = positivos.loc[
            positivos["PONDIIO"].notna()
            & (positivos["PONDIIO"] > 0)
        ]

        ingreso_real = validos["P21"] * factor_ipc
        peso_ingreso = validos["PONDIIO"]

        poblacion_rama = float(
            parte["PONDERA"].sum()
        )

        # Promedio ponderado de ingresos positivos reales.
        if len(validos) > 0:
            media = float(
                np.average(
                    ingreso_real,
                    weights=peso_ingreso,
                )
            )
        else:
            media = np.nan

        fila = {
            **datos_base,
            "rama": rama,

            "n_ocupados_muestrales": int(len(parte)),
            "poblacion_ocupada_ponderada": poblacion_rama,

            "porcentaje_ocupados_ponderado": (
                100 * poblacion_rama
                / total_ocupados_ponderado
            ),

            "n_p21_positivo": int(len(positivos)),
            "n_p21_cero": int(
                (parte["P21"] == 0).sum()
            ),

            "n_p21_no_responde": int(
                (parte["P21"] == -9).sum()
                + parte["P21"].isna().sum()
            ),

            "n_p21_positivo_sin_pondiio_valido": int(
                len(positivos) - len(validos)
            ),

            "n_ingresos_positivos_validos": int(
                len(validos)
            ),

            "ingreso_real_media_2026_t2": media,

            "ingreso_real_q1_2026_t2": cuantil_ponderado(
                ingreso_real, peso_ingreso, 0.25
            ),

            "ingreso_real_mediana_2026_t2": cuantil_ponderado(
                ingreso_real, peso_ingreso, 0.50
            ),

            "ingreso_real_q3_2026_t2": cuantil_ponderado(
                ingreso_real, peso_ingreso, 0.75
            ),
        }

        filas.append(fila)

    # Auditoría: conservamos los códigos originales para poder
    # revisar las clasificaciones realizadas.
    auditoria_codigos = []

    for codigo_original, parte in ocupados.groupby(
        "PP04B_COD",
        dropna=False,
    ):

        codigo_texto = (
            "[ausente]"
            if pd.isna(codigo_original)
            else str(codigo_original)
        )

        auditoria_codigos.append({
            **datos_base,
            "codigo_rama_original": codigo_texto,
            "rama_asignada": clasificar_rama(
                codigo_original
            ),
            "n_ocupados_muestrales": int(len(parte)),
            "poblacion_ocupada_ponderada": float(
                parte["PONDERA"].sum()
            ),
        })

    return filas, auditoria_codigos


# ============================================================
# 5. GRÁFICOS
# ============================================================

def graficar_ramas(tabla: pd.DataFrame) -> None:
    """Genera gráficos con los resultados reales del procesamiento.

    Gráfico 1:
        Evolución trimestral de las cuatro ramas con mayor
        población ocupada ponderada en 2026 T2.

    Gráfico 2:
        Medianas reales de P21 en las ocho ramas principales
        del último trimestre.

    No se inventan valores para completar categorías vacías.
    """

    ultimo = tabla.loc[
        (tabla["periodo"] == "2026-T2")
        & (tabla["rama"] != "Sin clasificar")
    ]

    # Elegimos las principales ramas según la población
    # ocupada ponderada de los dos aglomerados en 2026 T2.
    totales = ultimo.groupby(
        "rama"
    )["poblacion_ocupada_ponderada"].sum()

    principales_4 = (
        totales.nlargest(4).index.tolist()
    )

    principales_8 = (
        totales.nlargest(8).index.tolist()
    )

    # --------------------------------------------------------
    # Gráfico 1: evolución de ramas principales
    # --------------------------------------------------------

    fig, ejes = plt.subplots(
        2, 1,
        figsize=(12, 9),
        sharex=True,
    )

    periodos = tabla["periodo"].drop_duplicates().tolist()
    posiciones = {
        periodo: i
        for i, periodo in enumerate(periodos)
    }

    for eje, (codigo, nombre) in zip(
        ejes, AGLOMERADOS.items()
    ):

        for rama in principales_4:

            seleccion = tabla.loc[
                (tabla["codigo_aglomerado"] == codigo)
                & (tabla["rama"] == rama)
            ].sort_values(["anio", "trimestre"])

            eje.plot(
                [
                    posiciones[p]
                    for p in seleccion["periodo"]
                ],
                seleccion["porcentaje_ocupados_ponderado"],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=rama,
            )

        eje.set_title(nombre)
        eje.set_ylabel("% de ocupados")
        eje.grid(
            axis="y",
            alpha=0.3,
            linestyle="--",
        )
        eje.legend(fontsize=8, ncol=2)

    marcas = list(range(0, len(periodos), 4))

    if periodos and len(periodos) - 1 not in marcas:
        marcas.append(len(periodos) - 1)

    ejes[-1].set_xticks(
        marcas,
        [periodos[i] for i in marcas],
        rotation=45,
        ha="right",
    )

    ejes[-1].set_xlabel("Trimestre")

    fig.suptitle(
        "Participación de cuatro ramas principales "
        "en el empleo (2016-2026)"
    )

    fig.tight_layout()

    ruta = (
        GRAFICOS
        / "evolucion_ramas_principales_2016_2026.png"
    )

    fig.savefig(
        ruta,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Gráfico guardado: {ruta}")

    # --------------------------------------------------------
    # Gráfico 2: ingresos laborales por rama
    # --------------------------------------------------------

    seleccion = ultimo.loc[
        ultimo["rama"].isin(principales_8)
    ]

    fig, eje = plt.subplots(
        figsize=(11, 6)
    )

    ancho = 0.38
    posiciones_y = np.arange(len(principales_8))

    for indice, (codigo, nombre) in enumerate(
        AGLOMERADOS.items()
    ):

        parte = seleccion.loc[
            seleccion["codigo_aglomerado"] == codigo
        ].set_index("rama")

        parte = parte.reindex(principales_8)

        medianas = parte[
            "ingreso_real_mediana_2026_t2"
        ].to_numpy(dtype=float, copy=True)

        muestras = parte[
            "n_ingresos_positivos_validos"
        ].to_numpy(dtype=int)

        # Menos de 30 respuestas es una advertencia exploratoria.
        # Los valores no se eliminan del CSV.
        # Solo evitamos dibujar barras demasiado inestables.
        medianas[muestras < 30] = np.nan

        eje.barh(
            posiciones_y + (indice - 0.5) * ancho,
            medianas,
            height=ancho,
            label=nombre,
        )

    eje.set_yticks(
        posiciones_y,
        principales_8,
    )

    eje.invert_yaxis()

    eje.xaxis.set_major_formatter(
        FuncFormatter(
            lambda valor, _: f"${valor / 1e6:.1f} M"
        )
    )

    eje.set_xlabel(
        "Mediana de P21 positivo, "
        "pesos constantes de 2026 T2"
    )

    eje.set_title(
        "Ingreso de la ocupación principal "
        "por rama - 2026 T2"
    )

    eje.legend()

    eje.grid(
        axis="x",
        alpha=0.3,
        linestyle="--",
    )

    fig.text(
        0.5,
        0.005,
        "Las barras con menos de 30 respuestas "
        "válidas no se representan; consultar el CSV.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0.025, 1, 1]
    )

    ruta = (
        GRAFICOS
        / "ingreso_real_por_rama_2026_T2.png"
    )

    fig.savefig(
        ruta,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Gráfico guardado: {ruta}")


# ============================================================
# 6. PROCESAMIENTO PRINCIPAL
# ============================================================

def main() -> None:
    """Procesa los 41 trimestres y audita los resultados."""

    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)

    factores = cargar_factores()
    archivos = buscar_archivos()

    filas = []
    codigos = []

    for anio, trimestre, ruta in archivos:

        print(
            f"Procesando ramas e ingresos "
            f"{anio} T{trimestre}: {ruta.name}"
        )

        datos = leer_trimestre(
            ruta,
            anio,
            trimestre,
        )

        for codigo, nombre in AGLOMERADOS.items():

            grupo = datos.loc[
                datos["AGLOMERADO"] == codigo
            ]

            if grupo.empty:
                raise ValueError(
                    f"Sin datos de {nombre} en {anio} T{trimestre}"
                )

            clave = (
                f"{anio}-T{trimestre}",
                nombre,
            )

            if clave not in factores:
                raise ValueError(
                    f"Falta factor IPC para {clave}"
                )

            resumen, auditoria = analizar_aglomerado(
                grupo,
                anio,
                trimestre,
                codigo,
                factores[clave],
            )

            filas.extend(resumen)
            codigos.extend(auditoria)

    tabla = pd.DataFrame(filas)
    auditoria = pd.DataFrame(codigos)

    # Verificamos que exista una fila por trimestre,
    # aglomerado y categoría económica.
    esperado = 82 * len(RAMAS)

    if (
        len(tabla) != esperado
        or tabla.duplicated(
            ["periodo", "aglomerado", "rama"]
        ).any()
    ):
        raise ValueError(
            f"Se esperaban {esperado} filas sin duplicados."
        )

    # La suma de participaciones de todas las ramas,
    # incluida Sin clasificar, debe dar aproximadamente 100%.
    porcentajes = tabla.groupby(
        ["periodo", "aglomerado"]
    )["porcentaje_ocupados_ponderado"].sum()

    if not np.allclose(
        porcentajes.to_numpy(),
        100,
    ):
        raise ValueError(
            "Las ramas no suman 100% de los ocupados."
        )

    # Auditoría de códigos no clasificables.
    desconocidos = tabla.loc[
        tabla["rama"] == "Sin clasificar"
    ]

    print("\nAUDITORÍA DE RAMAS NO CLASIFICADAS")

    print(
        "Mayor porcentaje de ocupados sin rama "
        f"clasificable: "
        f"{desconocidos['porcentaje_ocupados_ponderado'].max():.2f}%"
    )

    print(
        "Casos muestrales sin clasificar acumulados: "
        f"{desconocidos['n_ocupados_muestrales'].sum()}"
    )

    print(
        "Nota: los registros acumulados entre trimestres "
        "no representan personas únicas."
    )

    # Guardamos los resultados completos para auditoría.
    guardar_csv(
        tabla,
        "ramas_actividad_ingresos_2016_2026.csv",
    )

    guardar_csv(
        auditoria,
        "auditoria_codigos_ramas_2016_2026.csv",
    )

    # Los gráficos se elaboran con los resultados reales.
    graficar_ramas(tabla)

    print(
        "\nLISTO: ramas de actividad e ingresos reales "
        "calculados y auditados."
    )


if __name__ == "__main__":
    main()
