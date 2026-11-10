"""TP de Introducción al Análisis de Datos - EPH 2016-2026.

OBJETIVO 2, ETAPA 3: rama económica e ingreso de la ocupación principal.

- 41 trimestres, CABA y Partidos del GBA; analiza únicamente ocupados.
- PP04B_COD: clasifica la actividad del establecimiento, NO la ocupación.
- Reagrupamiento basado en el cuadro 16 del informe de ingresos de la
  ENGHo 2017-2018 (INDEC), que reúne secciones del CAES-Mercosur 1.0.
- Se admiten divisiones válidas de 2 dígitos y algunas clases con cero
  inicial recuperable; 9999 y códigos ambiguos quedan «Sin clasificar».
- Participación laboral por rama: PONDERA y total de ocupados como denominador.
- Ingreso P21 mensual positivo: mediana, media, Q1 y Q3 con PONDIIO válido.
- Deflactación: factor de 2026 T2 ya calculado por el segundo script del TP.
- No elimina valores originales, no imputa ingresos y no crea datos ficticios.

FUENTES DE LAS AGRUPACIONES:
INDEC, ENGHo 2017-2018, Informe de ingresos, cuadro 16 (p. 134):
https://www.indec.gob.ar/ftp/cuadros/sociedad/engho_2017_2018_informe_ingresos.pdf
INDEC, clasificador CAES-Mercosur 1.0 (incluye división 83):
https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/caes_mercosur_1.0.pdf

Los cambios en clasificadores se interpretan al nivel agregado de división.
Se excluye «Otras ramas» solo del ranking gráfico por ser una mezcla de sectores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from analisis_eph_objetivo2_etapa1 import (
    AGLOMERADOS, RESULTADOS, GRAFICOS,
    buscar_archivos, leer_trimestre, guardar_csv,
)

# Nombres adaptados (abreviados) del reagrupamiento del cuadro 16 del INDEC.
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


# Clases oficiales de CAES cuyo código de cuatro dígitos comienza con 0.
# Los códigos como 101 o 600 pueden aparecer sin el cero inicial en los TXT.
# Solo normalizamos cuando la equivalencia está comprobada en el clasificador.
CLASES_CERO_INICIAL = {
    "0101", "0102", "0103", "0104", "0105",
    "0200", "0300", "0500", "0600", "0700", "0800", "0900",
}


def clasificar_rama(codigo: object) -> str:
    """Clasifica la sección económica usando divisiones CAES válidas.

    Acepta códigos de división de 2 dígitos, códigos de clase de 4 dígitos
    y los 3 dígitos que pueden normalizarse a una clase oficial con cero
    inicial. Conserva sin clasificar códigos de 1 dígito o ambiguos.
    """
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


def cargar_factores() -> dict[tuple[str, str], float]:
    """Reutiliza el IPC validado anteriormente; nunca lo vuelve a estimar."""
    ruta = RESULTADOS / "resumen_ponderado_ingresos_reales_2016_2026.csv"
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Es el CSV del objetivo 1 con el IPC validado."
        )
    tabla = pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
    campos = {"periodo", "aglomerado", "factor_a_pesos_2026_t2"}
    if not campos.issubset(tabla.columns):
        raise ValueError(f"Faltan campos del IPC en {ruta.name}: {sorted(campos - set(tabla))}")
    if len(tabla) != 82 or tabla.duplicated(["periodo", "aglomerado"]).any():
        raise ValueError("Se esperaban 82 factores de deflactación sin duplicados")
    pesos = pd.to_numeric(tabla["factor_a_pesos_2026_t2"], errors="coerce")
    if pesos.isna().any() or (pesos <= 0).any():
        raise ValueError("Factor IPC ausente, inválido o no positivo")
    tabla["factor"] = pesos
    return {(f.periodo, f.aglomerado): float(f.factor) for f in tabla.itertuples()}


def cuantil_ponderado(serie: pd.Series, pesos: pd.Series, p: float) -> float:
    """Calcula cuantil empírico ponderado, como en objetivo 1."""
    ok = serie.notna() & pesos.notna() & (pesos > 0)
    if not ok.any():
        return np.nan
    x = serie[ok].to_numpy(dtype=float)
    w = pesos[ok].to_numpy(dtype=float)
    orden = np.argsort(x, kind="stable")
    x, w = x[orden], w[orden]
    indice = np.searchsorted(np.cumsum(w), p * w.sum(), side="left")
    return float(x[min(indice, len(x) - 1)])


def analizar_aglomerado(grupo: pd.DataFrame, anio: int, trimestre: int,
                        codigo_aglomerado: int, factor_ipc: float) -> tuple[list[dict], list[dict]]:
    """Resume distribución del empleo y P21 real por rama económica."""
    ocupados = grupo.loc[grupo["ESTADO"] == 1].copy()
    columnas = {"PP04B_COD", "P21", "PONDIIO", "PONDERA"}
    if not columnas.issubset(ocupados):
        raise ValueError(f"Faltan variables: {sorted(columnas - set(ocupados))}")
    ocupados["rama"] = ocupados["PP04B_COD"].map(clasificar_rama)
    total_poblacional = float(ocupados["PONDERA"].sum())
    if total_poblacional <= 0:
        raise ValueError(f"{anio} T{trimestre}: sin personas ocupadas")
    if (ocupados["PONDIIO"].notna() & (ocupados["PONDIIO"] < 0)).any():
        raise ValueError(f"{anio} T{trimestre}: PONDIIO negativo")
    otros_negativos = ocupados["P21"].lt(0) & ocupados["P21"].ne(-9)
    if otros_negativos.any():
        raise ValueError(f"{anio} T{trimestre}: hay códigos negativos P21 inesperados")

    periodo = f"{anio}-T{trimestre}"
    nombre_aglomerado = AGLOMERADOS[codigo_aglomerado]
    base = {
        "anio": anio, "trimestre": trimestre, "periodo": periodo,
        "codigo_aglomerado": codigo_aglomerado,
        "aglomerado": nombre_aglomerado,
    }
    filas = []
    for rama in RAMAS:
        parte = ocupados.loc[ocupados["rama"] == rama]
        positivos = parte.loc[parte["P21"] > 0]
        validos = positivos.loc[
            positivos["PONDIIO"].notna() & (positivos["PONDIIO"] > 0)
        ]
        ingreso_real = validos["P21"] * factor_ipc
        peso_ingreso = validos["PONDIIO"]
        fila = dict(base, rama=rama,
                    n_ocupados_muestrales=int(len(parte)),
                    poblacion_ocupada_ponderada=float(parte["PONDERA"].sum()),
                    porcentaje_ocupados_ponderado=(
                        100 * float(parte["PONDERA"].sum()) / total_poblacional
                    ),
                    n_p21_positivo=int(len(positivos)),
                    n_p21_cero=int((parte["P21"] == 0).sum()),
                    n_p21_no_responde=int((parte["P21"] == -9).sum() + parte["P21"].isna().sum()),
                    n_p21_positivo_sin_pondiio_valido=int(len(positivos) - len(validos)),
                    n_ingresos_positivos_validos=int(len(validos)),
                    ingreso_real_media_2026_t2=(
                        float(np.average(ingreso_real, weights=peso_ingreso))
                        if len(validos) else np.nan
                    ),
                    ingreso_real_q1_2026_t2=cuantil_ponderado(ingreso_real, peso_ingreso, .25),
                    ingreso_real_mediana_2026_t2=cuantil_ponderado(ingreso_real, peso_ingreso, .50),
                    ingreso_real_q3_2026_t2=cuantil_ponderado(ingreso_real, peso_ingreso, .75),
                    )
        filas.append(fila)

    # Auditoría por código ORIGINAL; conserva códigos parciales no clasificados.
    codigos = []
    agrupados = ocupados.groupby("PP04B_COD", dropna=False)
    for valor, parte in agrupados:
        codigo_txt = "[ausente]" if pd.isna(valor) else str(valor)
        codigos.append(dict(base, codigo_rama_original=codigo_txt,
                            rama_asignada=clasificar_rama(valor),
                            n_ocupados_muestrales=int(len(parte)),
                            poblacion_ocupada_ponderada=float(parte["PONDERA"].sum())))
    return filas, codigos


def graficar_ramas(tabla: pd.DataFrame) -> None:
    """Crea dos gráficos reales: evolución de empleo e ingresos del trimestre final."""
    ultimos = tabla.loc[
        (tabla["periodo"] == "2026-T2") & (~tabla["rama"].isin(["Sin clasificar", "Otras ramas"]))
    ]
    # Selección transparente: cuatro u ocho ramas con más población ocupada
    # expandida en 2026 T2, sumando los dos aglomerados.
    totales = ultimos.groupby("rama")["poblacion_ocupada_ponderada"].sum()
    principales4 = totales.nlargest(4).index.tolist()
    principales8 = totales.nlargest(8).index.tolist()

    fig, ejes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    periodos = tabla["periodo"].drop_duplicates().tolist()
    posicion = {p: i for i, p in enumerate(periodos)}
    for eje, (codigo, nombre) in zip(ejes, AGLOMERADOS.items()):
        for rama in principales4:
            parte = tabla.loc[(tabla["codigo_aglomerado"] == codigo) &
                              (tabla["rama"] == rama)].sort_values(["anio", "trimestre"])
            eje.plot([posicion[p] for p in parte["periodo"]],
                     parte["porcentaje_ocupados_ponderado"],
                     marker="o", markersize=2.5, linewidth=1.5, label=rama)
        eje.set_title(nombre)
        eje.set_ylabel("% de ocupados")
        eje.grid(axis="y", alpha=.3, linestyle="--")
        eje.legend(fontsize=8, ncol=2)
    marcas = list(range(0, len(periodos), 4))
    if periodos and len(periodos) - 1 not in marcas:
        marcas.append(len(periodos) - 1)
    ejes[-1].set_xticks(marcas, [periodos[i] for i in marcas], rotation=45, ha="right")
    ejes[-1].set_xlabel("Trimestre")
    fig.suptitle("Participación de cuatro ramas principales en el empleo (2016-2026)")
    fig.tight_layout()
    destino = GRAFICOS / "evolucion_ramas_principales_2016_2026.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {destino}")

    seleccion = ultimos.loc[ultimos["rama"].isin(principales8)]
    fig, eje = plt.subplots(figsize=(11, 6))
    ancho = .38
    y = np.arange(len(principales8))
    for indice, (codigo, nombre) in enumerate(AGLOMERADOS.items()):
        parte = seleccion.loc[seleccion["codigo_aglomerado"] == codigo].set_index("rama")
        parte = parte.reindex(principales8)
        mediana = parte["ingreso_real_mediana_2026_t2"].to_numpy(dtype=float, copy=True)
        muestras = parte["n_ingresos_positivos_validos"].to_numpy(dtype=int)
        # Solo afecta la VISUALIZACIÓN; el CSV conserva todas las estimaciones.
        # No graficamos una barra cuando hay menos de 30 ingresos válidos.
        mediana[muestras < 30] = np.nan
        eje.barh(y + (indice - .5) * ancho, mediana,
                 height=ancho, label=nombre)
    eje.set_yticks(y, principales8)
    eje.invert_yaxis()
    eje.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1e6:.1f} M"))
    eje.set_xlabel("Mediana de P21 positivo, pesos constantes de 2026 T2")
    eje.set_title("Ingreso de la ocupación principal por rama - 2026 T2")
    eje.legend()
    eje.grid(axis="x", alpha=.3, linestyle="--")
    fig.text(.5, .005, "Para barras omitidas por menos de 30 respuestas válidas, consultar el CSV.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .025, 1, 1])
    destino = GRAFICOS / "ingreso_real_por_rama_2026_T2.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {destino}")


def main() -> None:
    """Procesa la serie y revisa integridad de resultados antes de graficar."""
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)
    factores = cargar_factores()
    archivos = buscar_archivos()
    filas, codigos = [], []
    for anio, trimestre, ruta in archivos:
        print(f"Procesando ramas e ingresos {anio} T{trimestre}: {ruta.name}")
        datos = leer_trimestre(ruta, anio, trimestre)
        for codigo, nombre in AGLOMERADOS.items():
            parte = datos.loc[datos["AGLOMERADO"] == codigo]
            if parte.empty:
                raise ValueError(f"Sin observaciones de {nombre} en {anio} T{trimestre}")
            clave = (f"{anio}-T{trimestre}", nombre)
            if clave not in factores:
                raise ValueError(f"Falta factor IPC para {clave}")
            f, c = analizar_aglomerado(parte, anio, trimestre, codigo, factores[clave])
            filas.extend(f)
            codigos.extend(c)

    tabla = pd.DataFrame(filas)
    auditoria = pd.DataFrame(codigos)
    esperado = 82 * len(RAMAS)
    if len(tabla) != esperado or tabla.duplicated(["periodo", "aglomerado", "rama"]).any():
        raise ValueError(f"Se esperaban {esperado} filas sin duplicados")
    suma_pct = tabla.groupby(["periodo", "aglomerado"])["porcentaje_ocupados_ponderado"].sum()
    if not np.allclose(suma_pct.to_numpy(), 100):
        raise ValueError("Las ramas no suman 100 % de la población ocupada")
    desconocidos = tabla.loc[tabla["rama"] == "Sin clasificar"]
    print("\nAUDITORÍA DE RAMAS NO CLASIFICADAS")
    print(f"Mayor porcentaje de ocupados sin rama clasificable: "
          f"{desconocidos['porcentaje_ocupados_ponderado'].max():.2f}%")
    print(f"Casos muestrales sin clasificar acumulados: "
          f"{desconocidos['n_ocupados_muestrales'].sum()}")
    print("Nota: la suma entre trimestres NO representa personas únicas.")

    guardar_csv(tabla, "ramas_actividad_ingresos_2016_2026.csv")
    guardar_csv(auditoria, "auditoria_codigos_ramas_2016_2026.csv")
    graficar_ramas(tabla)
    print("\nLISTO: ramas de actividad e ingresos reales calculados y auditados.")


if __name__ == "__main__":
    main()
