"""Objetivo 2 de Introducción al Análisis de Datos - EPH 2016-2026.

ETAPA 1: tasas generales de actividad, empleo y desocupación y auditoría
preliminar de las variables elegidas para el análisis multivariado.

Aglomerados: 32 (CABA) y 33 (Partidos del Gran Buenos Aires).
Períodos: 2016 T2 a 2026 T2, inclusive, 41 archivos trimestrales.

Importante:
- No modifica las bases originales ni los scripts del objetivo 1.
- Lee los TXT de la carpeta 'datos', situada junto a este programa.
- Calcula tasas poblacionales utilizando el factor de expansión PONDERA.
- Para las tasas generales de actividad y empleo, el denominador es la
  población TOTAL del aglomerado, incluidos los menores de 10 años.
- Para la desocupación, el denominador es la PEA (ocupados + desocupados).
- No fija todavía grupos de edad ni agrupa códigos de ramas de actividad:
  primero auditamos la cobertura de esas variables.
- El gráfico se genera con los microdatos reales, en la computadora donde
  se ejecute este programa. No utiliza cifras ficticias.

Requiere pandas y matplotlib, bibliotecas utilizadas en el trabajo previo.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN DE CARPETAS Y CAMPOS
# ============================================================

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RESULTADOS = BASE / "resultados"
GRAFICOS = BASE / "graficos"

AGLOMERADOS = {32: "CABA", 33: "Partidos del GBA"}
PATRON = re.compile(r"usu_individual_T([1-4])(\d{2})\.txt$", re.I)

# Estos campos siempre se necesitan para calcular las tasas.
OBLIGATORIOS = {"ANO4", "TRIMESTRE", "AGLOMERADO", "ESTADO", "PONDERA"}

# Se leen para determinar si están disponibles y completos, pero NO se
# utilizan todavía para producir estadísticas por subgrupos.
PARA_AUDITAR = {"CH06", "NIVEL_ED", "H15", "PP04B_COD", "P21", "PONDIIO"}
CAMPOS = OBLIGATORIOS | PARA_AUDITAR

# Referencia de validación externa (INDEC, cuadro 3.2, 2026 T2).
# No interviene en ningún cálculo ni reemplaza resultados de microdatos.
REFERENCIA_OFICIAL_2026_T2 = {
    32: {"tasa_actividad": 56.2, "tasa_empleo": 53.1, "tasa_desocupacion": 5.6},
    33: {"tasa_actividad": 47.8, "tasa_empleo": 43.6, "tasa_desocupacion": 8.8},
}


# ============================================================
# 2. LOCALIZACIÓN Y LECTURA DE LOS 41 ARCHIVOS
# ============================================================

def periodos_esperados() -> list[tuple[int, int]]:
    """Devuelve todos los trimestres desde 2016 T2 hasta 2026 T2."""
    return [
        (anio, trimestre)
        for anio in range(2016, 2027)
        for trimestre in range(1, 5)
        if (2016, 2) <= (anio, trimestre) <= (2026, 2)
    ]


def buscar_archivos() -> list[tuple[int, int, Path]]:
    """Identifica los períodos por nombre y comprueba la cobertura completa."""
    if not DATOS.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta {DATOS}. Colocá este script junto a 'datos'."
        )

    encontrados: dict[tuple[int, int], Path] = {}
    for ruta in DATOS.iterdir():
        if not ruta.is_file() or ruta.suffix.lower() != ".txt":
            continue
        coincidencia = PATRON.fullmatch(ruta.name)
        if coincidencia is None:
            continue
        trimestre = int(coincidencia.group(1))
        anio = 2000 + int(coincidencia.group(2))
        periodo = (anio, trimestre)
        if periodo in encontrados:
            raise ValueError(f"Dos TXT identifican el período {periodo}.")
        encontrados[periodo] = ruta

    esperados = set(periodos_esperados())
    faltantes = sorted(esperados - set(encontrados))
    if faltantes:
        raise FileNotFoundError(f"Faltan los siguientes períodos: {faltantes}")

    return [(anio, t, encontrados[(anio, t)]) for anio, t in periodos_esperados()]


def leer_trimestre(ruta: Path, anio: int, trimestre: int) -> pd.DataFrame:
    """Lee columnas relevantes y filtra CABA y Partidos del GBA.

    Se conserva PP04B_COD como texto para no eliminar posibles ceros
    iniciales de los códigos de rama de actividad.
    """
    datos = pd.read_csv(
        ruta,
        sep=";",
        encoding="latin-1",
        decimal=",",
        low_memory=False,
        dtype={"PP04B_COD": "string"},
        usecols=lambda nombre: nombre.strip().upper() in CAMPOS,
    )
    datos.columns = datos.columns.str.strip().str.upper()

    faltantes = sorted(OBLIGATORIOS - set(datos.columns))
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan columnas obligatorias {faltantes}")

    for campo in set(datos.columns) - {"PP04B_COD"}:
        datos[campo] = pd.to_numeric(datos[campo], errors="coerce")

    if "PP04B_COD" in datos.columns:
        datos["PP04B_COD"] = datos["PP04B_COD"].str.strip()

    if set(datos["ANO4"].dropna()) != {anio}:
        raise ValueError(f"{ruta.name}: ANO4 no coincide con el nombre del archivo")
    if set(datos["TRIMESTRE"].dropna()) != {trimestre}:
        raise ValueError(f"{ruta.name}: TRIMESTRE no coincide con el archivo")

    datos = datos[datos["AGLOMERADO"].isin(AGLOMERADOS)].copy()
    if datos.empty:
        raise ValueError(f"{ruta.name}: no contiene nuestros aglomerados")

    pesos = datos["PONDERA"]
    if pesos.isna().any() or (pesos <= 0).any():
        raise ValueError(f"{ruta.name}: contiene PONDERA nulos o no positivos")

    codigos = set(datos["ESTADO"].dropna().unique())
    if not codigos.issubset({0, 1, 2, 3, 4}) or datos["ESTADO"].isna().any():
        raise ValueError(
            f"{ruta.name}: ESTADO tiene códigos inesperados o faltantes: {codigos}"
        )
    return datos


# ============================================================
# 3. TASAS LABORALES - SIEMPRE PONDERADAS CON PONDERA
# ============================================================

def calcular_tasas(grupo: pd.DataFrame, anio: int,
                   trimestre: int, codigo: int) -> dict:
    """Calcula las tres tasas oficiales y guarda numeradores/denominadores.

    ESTADO: 0=sin entrevista individual, 1=ocupado, 2=desocupado,
    3=inactivo, 4=menor de diez años.

    Población total: suma de PONDERA en TODAS las filas del aglomerado;
    esto incluye a las personas menores de 10 años y no restringe el
    denominador a quienes respondieron el cuestionario individual.

    PEA = ocupados + desocupados. Las tasas de desocupación se calculan
    sobre PEA, mientras actividad y empleo usan población total.
    """
    w = grupo["PONDERA"]
    estado = grupo["ESTADO"]

    total = float(w.sum())
    ocupados = float(w[estado == 1].sum())
    desocupados = float(w[estado == 2].sum())
    activos = ocupados + desocupados

    if total <= 0 or activos <= 0:
        raise ValueError(
            f"{anio} T{trimestre}, {AGLOMERADOS[codigo]}: denominadores no válidos"
        )

    return {
        "anio": anio,
        "trimestre": trimestre,
        "periodo": f"{anio}-T{trimestre}",
        "codigo_aglomerado": codigo,
        "aglomerado": AGLOMERADOS[codigo],
        "registros_muestrales": len(grupo),
        "n_ocupados": int((estado == 1).sum()),
        "n_desocupados": int((estado == 2).sum()),
        "n_inactivos": int((estado == 3).sum()),
        "n_menores_10": int((estado == 4).sum()),
        "n_sin_entrevista_individual": int((estado == 0).sum()),
        "poblacion_total_ponderada": total,
        "poblacion_ocupada_ponderada": ocupados,
        "poblacion_desocupada_ponderada": desocupados,
        "poblacion_pea_ponderada": activos,
        "tasa_actividad": 100 * activos / total,
        "tasa_empleo": 100 * ocupados / total,
        "tasa_desocupacion": 100 * desocupados / activos,
    }


# ============================================================
# 4. AUDITORÍA PREVIA DE LAS TRES VARIABLES ELEGIDAS
# ============================================================

def auditar_variables(grupo: pd.DataFrame, anio: int,
                      trimestre: int, codigo: int) -> tuple[dict, list[dict]]:
    """Revisa disponibilidad de edad, educación y rama sin agruparlas.

    P21 y PONDIIO se inspeccionan para el futuro análisis del ingreso de
    la ocupación principal. Aquí NO se calculan todavía ingresos reales.
    """
    fila = {
        "periodo": f"{anio}-T{trimestre}",
        "aglomerado": AGLOMERADOS[codigo],
        "n_registros": len(grupo),
        "n_ocupados": int((grupo["ESTADO"] == 1).sum()),
    }
    ocupados = grupo[grupo["ESTADO"] == 1]
    conteos_rama = []

    if "CH06" in grupo:
        edad = grupo["CH06"]
        fila["n_edad_ausente"] = int(edad.isna().sum())
        fila["n_edad_fuera_de_rango"] = int(((edad < -1) | (edad > 120)).sum())
        mayores_25 = grupo[edad >= 25]
        fila["n_personas_25_mas"] = len(mayores_25)
        if "NIVEL_ED" in grupo:
            educacion = mayores_25["NIVEL_ED"]
            fila["n_educacion_25_mas_ausente"] = int(educacion.isna().sum())
            fila["n_educacion_25_mas_codigo_9"] = int((educacion == 9).sum())
            fila["n_educacion_25_mas_codigos_otros"] = int(
                (~educacion.isin([1, 2, 3, 4, 5, 6, 7, 9]) & educacion.notna()).sum()
            )
    else:
        fila["n_edad_ausente"] = None
        fila["n_edad_fuera_de_rango"] = None
        fila["n_personas_25_mas"] = None

    if "PP04B_COD" in ocupados:
        # Para la auditoría contamos todo código textual no vacío; el
        # clasificador oficial se verificará ANTES de asignar etiquetas.
        codigos = ocupados["PP04B_COD"].dropna()
        codigos = codigos[codigos.ne("")]
        fila["n_ocupados_con_rama_informada"] = len(codigos)
        fila["n_codigos_rama_distintos"] = codigos.nunique()
        for valor, frecuencia in codigos.value_counts().items():
            subconjunto = ocupados[ocupados["PP04B_COD"] == valor]
            conteos_rama.append({
                "periodo": f"{anio}-T{trimestre}",
                "aglomerado": AGLOMERADOS[codigo],
                "codigo_rama_original": valor,
                "frecuencia_muestral": int(frecuencia),
                "poblacion_ponderada": float(subconjunto["PONDERA"].sum()),
            })
    else:
        fila["n_ocupados_con_rama_informada"] = None
        fila["n_codigos_rama_distintos"] = None

    if "P21" in ocupados:
        ingreso = ocupados["P21"]
        fila["n_ocupados_p21_positivo"] = int((ingreso > 0).sum())
        fila["n_ocupados_p21_cero"] = int((ingreso == 0).sum())
        fila["n_ocupados_p21_menos9"] = int((ingreso == -9).sum())
        fila["n_ocupados_p21_otros_negativos"] = int(
            ((ingreso < 0) & (ingreso != -9)).sum()
        )
        fila["n_ocupados_p21_ausente"] = int(ingreso.isna().sum())
    else:
        fila["n_ocupados_p21_positivo"] = None

    if "PONDIIO" in ocupados:
        peso_ingreso = ocupados["PONDIIO"]
        fila["n_ocupados_pondiio_positivo"] = int((peso_ingreso > 0).sum())
        fila["n_ocupados_pondiio_cero"] = int((peso_ingreso == 0).sum())
        fila["n_ocupados_pondiio_ausente"] = int(peso_ingreso.isna().sum())
    else:
        fila["n_ocupados_pondiio_positivo"] = None

    return fila, conteos_rama


# ============================================================
# 5. EXPORTACIÓN Y GRÁFICOS CON RESULTADOS CALCULADOS
# ============================================================

def guardar_csv(datos: pd.DataFrame, nombre: str) -> None:
    """Exporta CSV compatible con la configuración regional argentina."""
    ruta = RESULTADOS / nombre
    datos.to_csv(ruta, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"CSV guardado: {ruta}")


def graficar_tasas(tabla: pd.DataFrame) -> None:
    """Crea un PNG con tres paneles usando las tasas recién calculadas.

    Este archivo se genera únicamente al ejecutar el script sobre los TXT.
    """
    fig, ejes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    periodos = tabla["periodo"].drop_duplicates().tolist()
    posiciones = {periodo: i for i, periodo in enumerate(periodos)}
    indicadores = [
        ("tasa_actividad", "Tasa de actividad"),
        ("tasa_empleo", "Tasa de empleo"),
        ("tasa_desocupacion", "Tasa de desocupación"),
    ]
    for eje, (campo, titulo) in zip(ejes, indicadores):
        for codigo, nombre in AGLOMERADOS.items():
            grupo = tabla[tabla["codigo_aglomerado"] == codigo]
            eje.plot(
                [posiciones[p] for p in grupo["periodo"]],
                grupo[campo], label=nombre, linewidth=1.7, marker="o", markersize=3,
            )
        eje.set_title(titulo)
        eje.set_ylabel("Porcentaje (%)")
        eje.grid(axis="y", alpha=0.35, linestyle="--")
        eje.legend(loc="best")

    marcas = list(range(0, len(periodos), 4))
    if len(periodos) - 1 not in marcas:
        marcas.append(len(periodos) - 1)
    ejes[-1].set_xticks(marcas, [periodos[i] for i in marcas], rotation=45, ha="right")
    ejes[-1].set_xlabel("Trimestre")
    fig.suptitle("EPH: tasas laborales en CABA y Partidos del GBA (2016-2026)")
    fig.tight_layout()
    ruta = GRAFICOS / "tasas_laborales_2016_2026.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {ruta}")


def comparar_con_indec(tabla: pd.DataFrame) -> None:
    """Muestra datos oficiales para verificar la coincidencia al primer decimal.

    Las cifras del INDEC se presentan como control, no como sustitutos
    de los resultados obtenidos con la EPH.
    """
    print("\nCONTROL EXTERNO: segundo trimestre de 2026")
    print("INDEC, Mercado de trabajo, 2026 T2, cuadro 3.2, pág. 18.")
    print("https://www.indec.gob.ar/uploads/informesdeprensa/mercado_trabajo_eph_2trim26433FCBC5A8.pdf")
    for codigo, nombre in AGLOMERADOS.items():
        datos = tabla[(tabla["periodo"] == "2026-T2") &
                      (tabla["codigo_aglomerado"] == codigo)]
        if len(datos) != 1:
            raise ValueError(f"No se encontró una fila para 2026 T2: {nombre}")
        fila = datos.iloc[0]
        print(f"\n{nombre}:")
        for campo, valor_oficial in REFERENCIA_OFICIAL_2026_T2[codigo].items():
            valor_calculado = float(fila[campo])
            diferencia = valor_calculado - valor_oficial
            print(f"  {campo}: calculado={valor_calculado:.2f}% | "
                  f"INDEC={valor_oficial:.1f}% | diferencia={diferencia:+.2f} pp")
            if abs(diferencia) >= 0.15:
                print("  AVISO: verificar denominadores, ponderadores y diseño de registro.")


# ============================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Procesa cada trimestre una sola vez y genera archivos de control."""
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)
    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} archivos válidos.")

    filas_tasas = []
    filas_auditoria = []
    filas_ramas = []

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        datos = leer_trimestre(ruta, anio, trimestre)
        for codigo, nombre in AGLOMERADOS.items():
            grupo = datos[datos["AGLOMERADO"] == codigo]
            if grupo.empty:
                raise ValueError(f"Sin observaciones para {nombre} en {anio} T{trimestre}")
            filas_tasas.append(calcular_tasas(grupo, anio, trimestre, codigo))
            auditoria, ramas = auditar_variables(grupo, anio, trimestre, codigo)
            filas_auditoria.append(auditoria)
            filas_ramas.extend(ramas)

    tabla = pd.DataFrame(filas_tasas)
    if len(tabla) != 82 or tabla.duplicated(["periodo", "aglomerado"]).any():
        raise ValueError("La tabla de tasas no contiene los 82 resultados esperados")

    guardar_csv(tabla, "tasas_laborales_2016_2026.csv")
    guardar_csv(pd.DataFrame(filas_auditoria), "auditoria_variables_objetivo2.csv")
    guardar_csv(pd.DataFrame(filas_ramas), "frecuencias_codigos_rama_objetivo2.csv")
    graficar_tasas(tabla)
    comparar_con_indec(tabla)

    print("\nLISTO: se calcularon tasas para los 41 trimestres y dos aglomerados.")
    print("Todavía no se agruparon edades, niveles educativos ni ramas de actividad.")


if __name__ == "__main__":
    main()
