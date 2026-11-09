"""TP de Introducción al Análisis de Datos: exploración univariada de la EPH.

Período esperado: 2016 T2 a 2026 T2 (41 bases individuales).
Aglomerados: 32 = CABA y 33 = Partidos del Gran Buenos Aires.

Este archivo implementa únicamente el OBJETIVO 1 de la consigna:
    - exploración de microdatos por trimestre y aglomerado;
    - revisión de identificadores, códigos e ingresos no respondidos;
    - estadísticas descriptivas y frecuencias univariadas;
    - detección exploratoria de atípicos por IQR y Z-score;
    - gráficos de distribución y evolución de la calidad de datos.

Todos los cálculos estadísticos son MUESTRALES (SIN PONDERAR).
Los ingresos son NOMINALES de cada trimestre (SIN AJUSTAR POR INFLACIÓN).
NO se eliminan duplicados, valores atípicos ni se imputan ingresos.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

BASE = Path(__file__).resolve().parent
CARPETA_DATOS = BASE / "datos"
CARPETA_RESULTADOS = BASE / "resultados"
CARPETA_GRAFICOS = BASE / "graficos"

# El gráfico de ejemplo usa automáticamente el último trimestre incluido.
# No se elige un período arbitrario.
TRIMESTRE_EJEMPLO = max((anio, t) for anio in range(2016, 2027)
                        for t in range(1, 5) if (2016, 2) <= (anio, t) <= (2026, 2))

# False evita crear 82 imágenes innecesarias. Si se cambia a True,
# además se guardan boxplots completo y ampliado de CADA trimestre.
GRAFICAR_CADA_TRIMESTRE = False

# True abre ventanas de Matplotlib. False solo guarda los archivos PNG.
MOSTRAR_GRAFICOS = False

AGLOMERADOS = {32: "CABA", 33: "Partidos del GBA"}
SEXOS = {1: "Varón", 2: "Mujer"}
EDUCACION = {
    1: "Primario incompleto",
    2: "Primario completo",
    3: "Secundario incompleto",
    4: "Secundario completo",
    5: "Superior/universitario incompleto",
    6: "Superior/universitario completo",
    7: "Sin instrucción",
    9: "No sabe / No responde",
}
ESTADOS = {1: "Ocupado", 2: "Desocupado", 3: "Inactivo"}

# El patrón identifica el trimestre y año desde el nombre del archivo.
# IGNORECASE tolera diferencias de mayúsculas y minúsculas.
PATRON_ARCHIVO = re.compile(r"usu_individual_T([1-4])(\d{2})\.txt$", re.IGNORECASE)

IDENTIFICADORES = ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
COLUMNAS_OBLIGATORIAS = {
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE",
    "AGLOMERADO", "H15", "CH04", "CH06", "NIVEL_ED", "ESTADO", "P47T",
}
# Leemos los ponderadores si están presentes, pero NO los utilizamos
# todavía. Serán relevantes para estimar indicadores poblacionales.
COLUMNAS_OPCIONALES = {"PONDERA", "PONDII"}
COLUMNAS_LECTURA = COLUMNAS_OBLIGATORIAS | COLUMNAS_OPCIONALES
COLUMNAS_NUMERICAS = (COLUMNAS_LECTURA - {"CODUSU"})

pd.set_option("display.float_format", "{:,.2f}".format)


# ============================================================
# 1. IDENTIFICAR LOS ARCHIVOS Y LEER LOS MICRODATOS
# ============================================================

def periodos_esperados() -> set[tuple[int, int]]:
    """Construye los 41 pares (año, trimestre) que pide la consigna.

    Incluye 2016 T2-T4, los cuatro trimestres de 2017-2025
    y 2026 T1-T2. No inventa un archivo de 2016 T1.
    """
    return {
        (anio, trimestre)
        for anio in range(2016, 2027)
        for trimestre in range(1, 5)
        if (2016, 2) <= (anio, trimestre) <= (2026, 2)
    }


def buscar_archivos() -> list[tuple[int, int, Path]]:
    """Localiza y ordena las bases individuales usando sus nombres.

    La expresión regular busca T225, T216, etc. Verifica faltantes,
    períodos inesperados y archivos repetidos. Si hay un problema,
    se detiene en lugar de entregar un informe histórico incompleto.
    """
    if not CARPETA_DATOS.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {CARPETA_DATOS}")

    encontrados = {}
    nombres_invalidos = []

    for ruta in CARPETA_DATOS.iterdir():
        if not ruta.is_file() or ruta.suffix.lower() != ".txt":
            continue
        coincidencia = PATRON_ARCHIVO.fullmatch(ruta.name)
        if coincidencia is None:
            nombres_invalidos.append(ruta.name)
            continue

        trimestre = int(coincidencia.group(1))
        anio = 2000 + int(coincidencia.group(2))
        periodo = (anio, trimestre)
        if periodo in encontrados:
            raise ValueError(
                f"Hay dos archivos para {anio} T{trimestre}: "
                f"{encontrados[periodo].name} y {ruta.name}"
            )
        encontrados[periodo] = ruta

    esperados = periodos_esperados()
    faltantes = sorted(esperados - encontrados.keys())
    adicionales = sorted(encontrados.keys() - esperados)

    if nombres_invalidos:
        print("AVISO: archivos TXT con nombres no reconocidos:", nombres_invalidos)
    if faltantes or adicionales:
        raise ValueError(
            f"La serie no coincide con el período esperado. "
            f"Faltantes: {faltantes}. Períodos adicionales: {adicionales}."
        )

    return [(anio, trim, encontrados[(anio, trim)]) for anio, trim in sorted(esperados)]


def cargar_trimestre(ruta: Path, anio: int, trimestre: int) -> pd.DataFrame:
    """Lee SOLO las columnas relevantes y verifica año y trimestre.

    read_csv lee el TXT delimitado por ';'. usecols disminuye memoria:
    no necesitamos importar las ~235 variables para este objetivo.
    to_numeric(errors='coerce') convierte valores no numéricos a NaN,
    que posteriormente contamos en los controles de calidad.
    """
    datos = pd.read_csv(
        ruta,
        sep=";",
        encoding="latin-1",
        decimal=",",
        low_memory=False,
        usecols=lambda nombre: nombre.strip().upper() in COLUMNAS_LECTURA,
    )
    datos.columns = datos.columns.str.strip().str.upper()

    faltantes = sorted(COLUMNAS_OBLIGATORIAS - set(datos.columns))
    if faltantes:
        raise ValueError(f"En {ruta.name} faltan columnas: {faltantes}")

    for columna in COLUMNAS_NUMERICAS & set(datos.columns):
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    anios_en_archivo = set(datos["ANO4"].dropna().unique())
    trimestres_en_archivo = set(datos["TRIMESTRE"].dropna().unique())
    if anios_en_archivo != {anio} or trimestres_en_archivo != {trimestre}:
        raise ValueError(
            f"{ruta.name}: el nombre indica {anio} T{trimestre}, "
            f"pero ANO4={anios_en_archivo}, TRIMESTRE={trimestres_en_archivo}"
        )

    return datos[datos["AGLOMERADO"].isin(AGLOMERADOS)].copy()


# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS Y CALIDAD DE DATOS
# ============================================================

def medidas(serie: pd.Series, prefijo: str) -> dict:
    """Devuelve medidas univariadas de una variable cuantitativa.

    count indica cantidad válida; mean es promedio; median, el centro;
    std mide dispersión; quantile obtiene Q1 y Q3; min/max, extremos.
    Se utilizan los valores presentes, sin ponderadores.
    """
    valores = serie.dropna()
    salida = {f"{prefijo}_n": len(valores)}
    for nombre, funcion in {
        "media": valores.mean, "mediana": valores.median,
        "desvio": valores.std, "q1": lambda: valores.quantile(0.25),
        "q3": lambda: valores.quantile(0.75),
        "minimo": valores.min, "maximo": valores.max,
    }.items():
        salida[f"{prefijo}_{nombre}"] = funcion() if not valores.empty else np.nan
    return salida


def frecuencias(datos: pd.DataFrame, anio: int, trim: int,
                codigo: int) -> list[dict]:
    """Cuenta categorías de sexo, educación y condición laboral.

    Son análisis UNIVARIADOS, aunque la población de referencia cambia:
    sexo: todas las personas; nivel educativo: personas >=25 años;
    condición laboral: entrevistas individuales H15=1.

    value_counts(dropna=False) conserva los faltantes en el recuento.
    """
    configuraciones = [
        ("Sexo", datos["CH04"], SEXOS, "Todas las edades"),
        ("Nivel educativo", datos.loc[datos["CH06"] >= 25, "NIVEL_ED"],
         EDUCACION, "25 años o más"),
        ("Condición laboral", datos.loc[datos["H15"] == 1, "ESTADO"],
         ESTADOS, "Entrevista individual realizada"),
    ]
    resultado = []
    for variable, serie, etiquetas, universo in configuraciones:
        conteos = serie.value_counts(dropna=False).sort_index()
        total = len(serie)
        for valor, cantidad in conteos.items():
            etiqueta = "NaN" if pd.isna(valor) else etiquetas.get(valor, f"Código {valor}")
            resultado.append({
                "anio": anio, "trimestre": trim, "periodo": f"{anio}-T{trim}",
                "codigo_aglomerado": codigo, "aglomerado": AGLOMERADOS[codigo],
                "variable": variable, "universo": universo,
                "categoria": etiqueta, "frecuencia_muestral": int(cantidad),
                "porcentaje_muestral": cantidad / total * 100 if total else np.nan,
            })
    return resultado


def evaluar_atipicos(serie: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Detecta ingresos positivos inusuales con IQR y Z-score.

    IQR: posibles extremos fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Z-score: observaciones a más de dos desviaciones estándar
    de la media, reproduciendo el criterio mostrado en clase.

    Los límites se recalculan para CADA trimestre y aglomerado.
    Los valores quedan intactos: un outlier no equivale a un error.
    """
    if serie.empty:
        vacio = {
            "iqr": np.nan, "limite_inferior_iqr": np.nan,
            "limite_superior_iqr": np.nan, "n_atipicos_iqr": 0,
            "pct_atipicos_iqr": np.nan, "n_atipicos_z": 0,
            "pct_atipicos_z": np.nan, "coincidencias_iqr_z": 0,
        }
        return vacio, pd.DataFrame()

    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    marca_iqr = (serie < inferior) | (serie > superior)

    # SciPy usa desviación estándar poblacional (ddof=0), igual
    # que stats.zscore en el código anterior. Evitamos dividir por 0.
    if len(serie) > 1 and serie.std(ddof=0) > 0:
        z = pd.Series(stats.zscore(serie), index=serie.index)
        marca_z = z.abs() > 2
    else:
        z = pd.Series(np.nan, index=serie.index)
        marca_z = pd.Series(False, index=serie.index)

    estadisticas = {
        "iqr": iqr,
        "limite_inferior_iqr": inferior,
        "limite_superior_iqr": superior,
        "n_atipicos_iqr": int(marca_iqr.sum()),
        "pct_atipicos_iqr": float(marca_iqr.mean() * 100),
        "n_atipicos_z": int(marca_z.sum()),
        "pct_atipicos_z": float(marca_z.mean() * 100),
        "coincidencias_iqr_z": int((marca_iqr & marca_z).sum()),
    }

    # La tabla de revisión preserva el índice del TXT. Sumamos 2
    # para expresar la línea de archivo (línea 1: encabezados).
    extremos = pd.DataFrame({
        "fila_txt_aproximada": serie.index + 2,
        "ingreso": serie,
        "z_score": z,
        "atipico_iqr": marca_iqr,
        "atipico_z": marca_z,
    }).sort_values("ingreso", ascending=False).head(5)
    return estadisticas, extremos


def analizar_aglomerado(grupo: pd.DataFrame, anio: int, trimestre: int,
                        codigo: int) -> tuple[dict, pd.DataFrame]:
    """Produce una fila de resumen para un aglomerado y trimestre.

    Separa la no entrevista (H15=2), los niños fuera del módulo
    individual (H15=0), la no respuesta del ingreso (P47T=-9),
    ingresos cero e ingresos positivos.

    Controla duplicados SOLO dentro del trimestre. En la EPH,
    una persona puede aparecer legítimamente en otros trimestres.
    """
    entrevistados = grupo[grupo["H15"] == 1]
    ingreso = entrevistados["P47T"]
    positivos = ingreso[ingreso > 0]
    conocidos = ingreso[ingreso >= 0]

    nr = int((ingreso == -9).sum())
    nulos = int(ingreso.isna().sum())
    ceros = int((ingreso == 0).sum())
    otros_negativos = int(((ingreso < 0) & (ingreso != -9)).sum())
    total_clasificado = nr + nulos + ceros + len(positivos) + otros_negativos

    edades = grupo["CH06"].replace(-1, 0)
    resumen = {
        "anio": anio,
        "trimestre": trimestre,
        "periodo": f"{anio}-T{trimestre}",
        "codigo_aglomerado": codigo,
        "aglomerado": AGLOMERADOS[codigo],
        "registros_muestrales": len(grupo),
        "id_incompletos": int(grupo[IDENTIFICADORES].isna().any(axis=1).sum()),
        "id_duplicados": int(grupo.duplicated(IDENTIFICADORES, keep=False).sum()),
        "entrevistas_h15_0": int((grupo["H15"] == 0).sum()),
        "entrevistas_h15_1": len(entrevistados),
        "entrevistas_h15_2": int((grupo["H15"] == 2).sum()),
        "h15_otros_o_nan": int((~grupo["H15"].isin([0, 1, 2])).sum()),
        "h15_0_con_edad_10_o_mas": int(((grupo["H15"] == 0) &
                                        (grupo["CH06"] >= 10)).sum()),
        "edad_valores_fuera_de_rango": int(((grupo["CH06"] < -1) |
                                             (grupo["CH06"] > 120)).sum()),
        "p47t_no_responde_menos9": nr,
        "p47t_nan": nulos,
        "p47t_cero": ceros,
        "p47t_positivo": len(positivos),
        "p47t_otros_negativos": otros_negativos,
        "p47t_recuento_consistente": total_clasificado == len(entrevistados),
        "p47t_pct_no_respuesta_codificada": (
            nr / len(entrevistados) * 100 if len(entrevistados) else np.nan
        ),
        "p47t_pct_faltante_total": (
            (nr + nulos) / len(entrevistados) * 100
            if len(entrevistados) else np.nan
        ),
    }

    resumen.update(medidas(edades, "edad"))
    resumen.update(medidas(conocidos, "ingreso_con_ceros"))
    resumen.update(medidas(positivos, "ingreso_positivo"))
    atipicos, extremos = evaluar_atipicos(positivos)
    resumen.update(atipicos)

    if not extremos.empty:
        extremos.insert(0, "aglomerado", AGLOMERADOS[codigo])
        extremos.insert(0, "codigo_aglomerado", codigo)
        extremos.insert(0, "periodo", f"{anio}-T{trimestre}")

    return resumen, extremos


# ============================================================
# 3. GRÁFICOS
# ============================================================

def guardar_figura(fig, ruta: Path) -> None:
    """Guarda un PNG y, opcionalmente, muestra el gráfico en pantalla."""
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


def graficar_boxplots(datos: pd.DataFrame, anio: int, trimestre: int) -> None:
    """Guarda un boxplot completo y otro ampliado para un trimestre.

    Los dos gráficos utilizan exactamente los mismos registros.
    set_ylim solo cambia la escala visible, sin descartar datos.
    Para otros trimestres se calcula automáticamente el zoom,
    porque $3,5 millones no sirve como umbral fijo para 2016-2026.
    """
    grafico = datos.loc[
        (datos["H15"] == 1) & (datos["P47T"] > 0),
        ["AGLOMERADO", "P47T"],
    ].copy()
    if grafico.empty:
        return
    grafico["AGLOMERADO"] = grafico["AGLOMERADO"].map(AGLOMERADOS)

    limites = []
    for nombre in AGLOMERADOS.values():
        valores = grafico.loc[grafico["AGLOMERADO"] == nombre, "P47T"]
        if valores.empty:
            continue
        q1, q3 = valores.quantile(0.25), valores.quantile(0.75)
        limites.append(q3 + 1.5 * (q3 - q1))

    if not limites:
        return
    zoom = max(limites) * 1.15
    for es_zoom in (False, True):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=grafico,
            x="AGLOMERADO", y="P47T",
            order=list(AGLOMERADOS.values()),
            color="steelblue", width=0.45,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            ax=ax,
        )
        ax.set_ylim(bottom=0, top=zoom if es_zoom else None)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda valor, _: f"${valor / 1_000_000:.1f} M")
        )
        ax.set_title(
            f"Distribución de ingresos positivos - {anio} T{trimestre}"
            + (" (ampliación)" if es_zoom else " (completa)")
        )
        ax.set_xlabel("Aglomerado")
        ax.set_ylabel("Ingreso nominal total individual (millones de pesos)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        nota = (f"Escala ampliada hasta ${zoom / 1_000_000:.2f} M; "
                "los valores mayores no se muestran." if es_zoom else
                "Escala completa; todos los ingresos positivos están incluidos.")
        fig.text(0.5, 0.02, nota, ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 1])

        sufijo = "zoom" if es_zoom else "completo"
        ruta = CARPETA_GRAFICOS / f"boxplot_{anio}_T{trimestre}_{sufijo}.png"
        guardar_figura(fig, ruta)


def graficar_evolucion(resumen: pd.DataFrame, columna: str,
                       titulo: str, eje_y: str, archivo: str) -> None:
    """Dibuja una línea por aglomerado a partir del resumen trimestral.

    Cada punto representa un trimestre. Como analizamos porcentajes
    del propio trimestre, no mezclamos montos monetarios afectados
    por la inflación. Todavía son porcentajes muestrales sin ponderar.
    """
    periodos = resumen[["anio", "trimestre", "periodo"]].drop_duplicates()
    periodos = periodos.sort_values(["anio", "trimestre"])["periodo"].tolist()
    posicion = {periodo: indice for indice, periodo in enumerate(periodos)}

    fig, ax = plt.subplots(figsize=(12, 5))
    for codigo, nombre in AGLOMERADOS.items():
        grupo = resumen[resumen["codigo_aglomerado"] == codigo]
        grupo = grupo.sort_values(["anio", "trimestre"])
        ax.plot([posicion[p] for p in grupo["periodo"]], grupo[columna],
                marker="o", markersize=3, linewidth=1.7, label=nombre)

    marcas = list(range(0, len(periodos), 4))
    if periodos and len(periodos) - 1 not in marcas:
        marcas.append(len(periodos) - 1)
    ax.set_xticks(marcas, [periodos[i] for i in marcas], rotation=45, ha="right")
    ax.set_title(titulo)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel(eje_y)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    guardar_figura(fig, CARPETA_GRAFICOS / archivo)


def guardar_csv(datos: pd.DataFrame, nombre: str) -> None:
    """Guarda tablas fáciles de abrir tanto con Python como con Excel.

    sep=';' y decimal=',' siguen la configuración habitual argentina.
    utf-8-sig facilita que Excel reconozca acentos correctamente.
    """
    ruta = CARPETA_RESULTADOS / nombre
    datos.to_csv(ruta, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"Tabla guardada: {ruta}")


# ============================================================
# 4. EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Procesa 41 archivos de uno en uno y conserva resúmenes compactos.

    No concatena los 41 DataFrames originales: reduce uso de RAM.
    Al terminar, guarda tablas por trimestre y gráficos históricos.
    """
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    CARPETA_GRAFICOS.mkdir(exist_ok=True)

    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} bases individuales válidas.")
    print("Período: 2016 T2 - 2026 T2. Aglomerados: 32 y 33.")

    resumenes = []
    tablas_frecuencias = []
    tablas_extremos = []

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        try:
            filtrados = cargar_trimestre(ruta, anio, trimestre)
        except Exception as error:
            raise RuntimeError(f"No se pudo procesar {ruta.name}: {error}") from error

        for codigo in AGLOMERADOS:
            grupo = filtrados[filtrados["AGLOMERADO"] == codigo].copy()
            if grupo.empty:
                raise ValueError(f"{ruta.name} no tiene registros del aglomerado {codigo}")

            resultado, extremos = analizar_aglomerado(grupo, anio, trimestre, codigo)
            resumenes.append(resultado)
            tablas_frecuencias.extend(frecuencias(grupo, anio, trimestre, codigo))
            if not extremos.empty:
                tablas_extremos.append(extremos)

            if (resultado["id_duplicados"] > 0 or resultado["id_incompletos"] > 0
                    or resultado["p47t_otros_negativos"] > 0
                    or not resultado["p47t_recuento_consistente"]):
                print(f"  AVISO: revisar calidad de datos en {resultado['aglomerado']}.")

        # Solo hacemos estos dos gráficos en un trimestre de referencia,
        # salvo que activemos GRAFICAR_CADA_TRIMESTRE.
        if GRAFICAR_CADA_TRIMESTRE or (anio, trimestre) == TRIMESTRE_EJEMPLO:
            graficar_boxplots(filtrados, anio, trimestre)

    resumen = pd.DataFrame(resumenes).sort_values(
        ["anio", "trimestre", "codigo_aglomerado"]
    )
    freq = pd.DataFrame(tablas_frecuencias)
    extremos = (pd.concat(tablas_extremos, ignore_index=True)
                if tablas_extremos else pd.DataFrame())

    guardar_csv(resumen, "resumen_univariado_2016_2026.csv")
    guardar_csv(freq, "frecuencias_univariadas_2016_2026.csv")
    guardar_csv(extremos, "revision_5_ingresos_mayores_por_trimestre.csv")

    graficar_evolucion(
        resumen, "p47t_pct_no_respuesta_codificada",
        "No respuesta de ingresos individuales por trimestre",
        "Porcentaje muestral (%)", "evolucion_no_respuesta_p47t.png",
    )
    graficar_evolucion(
        resumen, "pct_atipicos_iqr",
        "Ingresos positivos atípicos (IQR) por trimestre",
        "Porcentaje de ingresos positivos (%)", "evolucion_atipicos_iqr.png",
    )

    print("\n=== CONTROL FINAL ===")
    print("Trimestres procesados:", len(archivos))
    print("Filas de resumen (2 por trimestre):", len(resumen))
    print("Años:", resumen["anio"].min(), "a", resumen["anio"].max())
    print(f"\n=== CONTROL DEL ÚLTIMO TRIMESTRE: {TRIMESTRE_EJEMPLO[0]} T{TRIMESTRE_EJEMPLO[1]} ===")
    ejemplo = resumen[(resumen["anio"] == TRIMESTRE_EJEMPLO[0]) & (resumen["trimestre"] == TRIMESTRE_EJEMPLO[1])]
    columnas = ["aglomerado", "registros_muestrales", "id_duplicados",
                "p47t_no_responde_menos9", "p47t_pct_no_respuesta_codificada",
                "ingreso_positivo_mediana", "n_atipicos_iqr", "n_atipicos_z"]
    print(ejemplo[columnas].to_string(index=False))
    print("\nTerminado. Revisar las carpetas 'resultados' y 'graficos'.")


if __name__ == "__main__":
    main()
