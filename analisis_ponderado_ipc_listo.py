"""Complemento del punto 1: EPH 2016 T2-2026 T2, ponderadores e IPC GBA.

Este script NO modifica los TXT ni los resultados exploratorios previos.

Esta versión lee directamente el archivo oficial serie_ipc_divisiones.csv
que fue compartido por el grupo (Region=GBA, Codigo=0, NIVEL GENERAL).
La base del poder adquisitivo es el promedio trimestral del IPC en 2026 T2.


Versión corregida: admite PONDII=0 o ausente en registros con ingreso
conocido, los audita y no los utiliza en estadísticas ponderadas de ingreso.
No aplica ponderadores inventados ni borra observaciones.
Requiere pandas, numpy y matplotlib (ya usados en el proyecto).

IPC: colocar el archivo descargado en datos/serie_ipc_divisiones.csv:
https://www.indec.gob.ar/ftp/cuadros/economia/serie_ipc_divisiones.csv
El lector también acepta un CSV simple con mes,ipc o un archivo
indice_tiempo,ipc_ng_gba. Si faltan meses, no se generan ingresos reales.

Los índices abril-noviembre de 2016 en base dic-2016=100 proceden de:
https://datos.gob.ar/es/dataset/sspm_101/archivo/sspm_101.3
(columna ipc_2016_nivel_general). La serie posterior procede del INDEC.

Medidas descriptivas ponderadas: no incorporan el diseño complejo para
calcular errores estándar o intervalos de confianza.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
DATOS = BASE / 'datos'
RESULTADOS = BASE / 'resultados'
GRAFICOS = BASE / 'graficos'
AGLOMERADOS = {32: 'CABA', 33: 'Partidos del GBA'}

# Esta breve serie histórica está expresada en base diciembre 2016=100.
# Fuente primaria identificada en Datos Argentina: IPC-GBA, nivel general,
# serie base abril 2016 reexpresada con diciembre 2016=100.
# https://datos.gob.ar/dataset/sspm_101/archivo/sspm_101.3
# Redondeada a un decimal, como en la serie publicada.
IPC_2016_GBA = {
    '2016-04': 85.5, '2016-05': 89.1, '2016-06': 91.9,
    '2016-07': 93.7, '2016-08': 93.9, '2016-09': 95.0,
    '2016-10': 97.2, '2016-11': 98.8, '2016-12': 100.0,
}
PATRON = re.compile(r'usu_individual_T([1-4])(\d{2})\.txt$', re.I)
CAMPOS = {
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'CH04', 'CH06', 'NIVEL_ED',
    'H15', 'ESTADO', 'P47T', 'PONDERA', 'PONDII',
}


def periodos():
    """Devuelve los 41 pares (año, trimestre) que exige el trabajo."""
    return [(anio, trimestre) for anio in range(2016, 2027)
            for trimestre in range(1, 5)
            if (2016, 2) <= (anio, trimestre) <= (2026, 2)]


def archivos_eph():
    """Relaciona cada TXT con el período codificado en su nombre y valida cobertura."""
    encontrados = {}
    for ruta in DATOS.iterdir():
        m = PATRON.fullmatch(ruta.name)
        if not ruta.is_file() or not m:
            continue
        clave = (2000 + int(m.group(2)), int(m.group(1)))
        if clave in encontrados:
            raise ValueError(f'Dos archivos para el mismo trimestre: {clave}')
        encontrados[clave] = ruta
    faltantes = sorted(set(periodos()) - set(encontrados))
    if faltantes:
        raise FileNotFoundError(f'Faltan microdatos trimestrales: {faltantes}')
    return [(anio, t, encontrados[(anio, t)]) for anio, t in periodos()]


def leer_eph(ruta, anio, trimestre):
    """Carga solo campos necesarios y verifica el período antes de ponderar."""
    df = pd.read_csv(ruta, sep=';', encoding='latin-1', decimal=',',
                     low_memory=False,
                     usecols=lambda c: c.strip().upper() in CAMPOS)
    df.columns = df.columns.str.strip().str.upper()
    faltantes = CAMPOS - set(df.columns)
    if faltantes:
        raise ValueError(f'{ruta.name} no contiene: {sorted(faltantes)}')
    for campo in CAMPOS:
        df[campo] = pd.to_numeric(df[campo], errors='coerce')
    if set(df['ANO4'].dropna()) != {anio} or set(df['TRIMESTRE'].dropna()) != {trimestre}:
        raise ValueError(f'Período incorrecto en {ruta.name}')
    df = df[df['AGLOMERADO'].isin(AGLOMERADOS)].copy()
    if (df['PONDERA'].isna() | (df['PONDERA'] <= 0)).any():
        raise ValueError(f'PONDERA inválido en {ruta.name}')
    # PONDII es un peso específico de ingreso y puede valer 0 (los casos
    # con peso cero NO aportan a las estimaciones). Un peso faltante tampoco
    # debe reemplazarse automáticamente por PONDERA.
    #
    # La versión anterior exigía PONDII > 0 para TODO ingreso conocido.
    # Eso era demasiado estricto: por eso se detenía en 2016 T3.
    # Ahora registramos los casos en analizar_grupo() y los excluimos SOLO
    # del cálculo ponderado si PONDII <= 0 o está ausente.
    negativos = df['PONDII'].notna() & (df['PONDII'] < 0)
    if negativos.any():
        raise ValueError(
            f'{ruta.name}: hay {int(negativos.sum())} ponderadores PONDII negativos; '
            'revisar la fuente antes de continuar.'
        )
    return df


def media_ponderada(valores, pesos):
    """Calcula sum(valor * ponderador) / sum(ponderador) sobre filas válidas."""
    ok = valores.notna() & pesos.notna() & (pesos > 0)
    if not ok.any():
        return np.nan
    x = valores[ok].to_numpy(dtype=float)
    w = pesos[ok].to_numpy(dtype=float)
    return float(np.average(x, weights=w))


def cuantil_ponderado(valores, pesos, p):
    """Cuantil empírico ponderado: primer valor cuya masa acumulada alcanza p.

    A diferencia de pd.Series.quantile, considera la población representada por
    cada registro. Puede no coincidir con la mediana muestral interpolada.
    """
    ok = valores.notna() & pesos.notna() & (pesos > 0)
    if not ok.any():
        return np.nan
    x = valores[ok].to_numpy(dtype=float)
    w = pesos[ok].to_numpy(dtype=float)
    orden = np.argsort(x, kind='stable')
    x, w = x[orden], w[orden]
    pos = np.searchsorted(np.cumsum(w), p * w.sum(), side='left')
    return float(x[min(pos, len(x) - 1)])


def estadisticas_ponderadas(valores, pesos, nombre):
    """Devuelve media y percentiles 25, 50 y 75, además del peso total válido."""
    ok = valores.notna() & pesos.notna() & (pesos > 0)
    return {
        f'{nombre}_peso_total': float(pesos[ok].sum()),
        f'{nombre}_media': media_ponderada(valores, pesos),
        f'{nombre}_q1': cuantil_ponderado(valores, pesos, .25),
        f'{nombre}_mediana': cuantil_ponderado(valores, pesos, .50),
        f'{nombre}_q3': cuantil_ponderado(valores, pesos, .75),
    }


def frecuencias_ponderadas(grupo, anio, trimestre, codigo):
    """Estima proporciones de sexo, educación (25+) y estado laboral.

    PONDERA expande cada caso al número de personas representadas.
    Estas son frecuencias descriptivas, no pruebas de significación.
    """
    selecciones = [
        ('Sexo', grupo, 'CH04'),
        ('Nivel educativo (25+)', grupo[grupo['CH06'] >= 25], 'NIVEL_ED'),
        ('Condición laboral (H15=1)', grupo[grupo['H15'] == 1], 'ESTADO'),
    ]
    filas = []
    for nombre, datos, campo in selecciones:
        validos = datos[datos[campo].notna()]
        total = validos['PONDERA'].sum()
        for categoria, parte in validos.groupby(campo):
            n_ponderado = parte['PONDERA'].sum()
            filas.append({
                'anio': anio, 'trimestre': trimestre,
                'periodo': f'{anio}-T{trimestre}',
                'aglomerado': AGLOMERADOS[codigo], 'variable': nombre,
                'codigo_categoria': int(categoria),
                'frecuencia_muestral': len(parte),
                'poblacion_estimada': round(float(n_ponderado)),
                'porcentaje_ponderado': float(n_ponderado / total * 100),
            })
    return filas


def analizar_grupo(grupo, anio, trimestre, codigo):
    """Construye un resumen de población, no respuesta e ingresos ponderados."""
    resp = grupo[grupo['H15'] == 1]
    pesos_resp = resp['PONDERA']
    sin_respuesta = resp['P47T'].eq(-9) | resp['P47T'].isna()
    conocidos = resp[resp['P47T'] >= 0]
    positivos = conocidos[conocidos['P47T'] > 0]
    total_resp = pesos_resp.sum()

    # Auditoría de los pesos de ingreso. Los casos se CONSERVAN en la base.
    # Solo se excluyen de estadísticas que usan PONDII cuando su peso
    # es cero o falta. No se reemplaza PONDII por PONDERA.
    pesos_ingresos = conocidos['PONDII']
    pesos_positivos = pesos_ingresos.notna() & (pesos_ingresos > 0)
    n_ponderador_cero = int((pesos_ingresos == 0).sum())
    n_ponderador_faltante = int(pesos_ingresos.isna().sum())
    n_ponderador_invalido = n_ponderador_cero + n_ponderador_faltante
    if not pesos_positivos.any():
        raise ValueError(
            f'{anio} T{trimestre} - {AGLOMERADOS[codigo]}: '
            'no hay ingresos conocidos con PONDII positivo. Revisar la base.'
        )
    if n_ponderador_invalido:
        print(
            f'  AVISO {anio} T{trimestre}, {AGLOMERADOS[codigo]}: '
            f'{n_ponderador_cero} ingresos conocidos con PONDII=0 y '
            f'{n_ponderador_faltante} con PONDII ausente. '
            'No participan en cálculos ponderados de ingresos.'
        )

    # Otra comprobación: si una persona no declaró el ingreso, su peso
    # de ingreso corregido normalmente debería ser cero o no aplicable.
    no_respondentes = resp[sin_respuesta]
    pesos_nr_positivos = int((no_respondentes['PONDII'] > 0).sum())
    if pesos_nr_positivos:
        print(
            f'  AVISO {anio} T{trimestre}, {AGLOMERADOS[codigo]}: '
            f'{pesos_nr_positivos} no respondentes tienen PONDII positivo. '
            'Revisar la correspondencia entre P47T y PONDII.'
        )

    resultado = {
        'anio': anio, 'trimestre': trimestre,
        'periodo': f'{anio}-T{trimestre}',
        'codigo_aglomerado': codigo,
        'aglomerado': AGLOMERADOS[codigo],
        'n_registros': len(grupo),
        'poblacion_estimada': float(grupo['PONDERA'].sum()),
        'n_entrevistados': len(resp),
        'n_no_respuesta_p47t': int(sin_respuesta.sum()),
        'no_respuesta_pct_muestral': float(sin_respuesta.mean() * 100),
        'no_respuesta_pct_ponderado': float(
            pesos_resp[sin_respuesta].sum() / total_resp * 100
        ),
        'n_ingresos_conocidos': len(conocidos),
        'n_ingresos_positivos': len(positivos),
        # Estos campos permiten auditar casos excluidos de cada estadística.
        'n_ingresos_conocidos_pondii_positivo': int(pesos_positivos.sum()),
        'n_ingresos_conocidos_pondii_cero': n_ponderador_cero,
        'n_ingresos_conocidos_pondii_faltante': n_ponderador_faltante,
        'pct_conocidos_sin_peso_positivo': float(
            n_ponderador_invalido / len(conocidos) * 100
        ) if len(conocidos) else np.nan,
        'n_no_respondentes_pondii_positivo': pesos_nr_positivos,
    }
    edad = grupo['CH06'].replace(-1, 0)  # Menores de un año.
    resultado.update(estadisticas_ponderadas(edad, grupo['PONDERA'], 'edad'))
    # PONDII: ajuste oficial de no respuesta para ingreso TOTAL individual P47T.
    resultado.update(estadisticas_ponderadas(
        conocidos['P47T'], conocidos['PONDII'], 'ingreso_con_ceros'))
    resultado.update(estadisticas_ponderadas(
        positivos['P47T'], positivos['PONDII'], 'ingreso_positivo'))
    return resultado


def serie_ipc_2016():
    """Datos históricos del IPC-GBA con diciembre de 2016=100."""
    return pd.DataFrame([{'mes': mes, 'ipc': valor} for mes, valor in IPC_2016_GBA.items()])


def preparar_ipc_fuente(ruta):
    """Lee IPC-GBA mensual desde un CSV y devuelve columnas mes, ipc.

    Formatos reconocidos:
      1) INDEC ``serie_ipc_divisiones.csv`` (formato largo):
         Codigo=0, Descripcion=NIVEL GENERAL, Region=GBA,
         Periodo=AAAAMM e Indice_IPC con coma decimal.
      2) CSV simple: mes,ipc (mes AAAA-MM).
      3) Datos Argentina: indice_tiempo,ipc_ng_gba.

    Se valida que exista una sola observación por mes; seleccionar el
    nivel general es indispensable para no confundirlo con una división
    de consumo ni con el índice nacional.
    """
    # Probamos los separadores habituales y ambas codificaciones.
    datos = None
    for sep in (';', ','):
        for codificacion in ('utf-8-sig', 'latin-1'):
            try:
                candidato = pd.read_csv(
                    ruta, sep=sep, encoding=codificacion,
                    dtype=str, keep_default_na=False
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
        raise ValueError(f'No se pudo leer el CSV del IPC: {ruta.name}')

    def convertir_numeros(serie):
        """Transforma índices con coma o punto decimal en valores float."""
        texto = serie.astype(str).str.strip().str.replace(' ', '', regex=False)
        # Un posible punto de miles solo se elimina en presencia de coma.
        coma = texto.str.contains(',', regex=False)
        texto = texto.where(~coma, texto.str.replace('.', '', regex=False))
        return pd.to_numeric(texto.str.replace(',', '.', regex=False), errors='coerce')

    # Formato largo oficial incluido en los archivos que descargaste.
    oficiales = {'codigo', 'descripcion', 'clasificador', 'periodo',
                 'indice_ipc', 'region'}
    if oficiales.issubset(datos.columns):
        seleccion = datos.loc[
            datos['codigo'].str.strip().eq('0')
            & datos['descripcion'].str.strip().str.upper().eq('NIVEL GENERAL')
            & datos['region'].str.strip().str.upper().eq('GBA')
            & datos['clasificador'].str.strip().str.contains(
                'Nivel general', case=False, regex=False
            )
        ].copy()
        if seleccion.empty:
            raise ValueError('IPC: no hay registros Codigo=0, Nivel general, GBA')
        fechas = pd.to_datetime(seleccion['periodo'], format='%Y%m', errors='coerce')
        salida = pd.DataFrame({
            'mes': fechas.dt.strftime('%Y-%m'),
            'ipc': convertir_numeros(seleccion['indice_ipc']),
        })
    elif {'indice_tiempo', 'ipc_ng_gba'}.issubset(datos.columns):
        fechas = pd.to_datetime(datos['indice_tiempo'], errors='coerce')
        salida = pd.DataFrame({
            'mes': fechas.dt.strftime('%Y-%m'),
            'ipc': convertir_numeros(datos['ipc_ng_gba']),
        })
    elif {'mes', 'ipc'}.issubset(datos.columns):
        fechas = pd.to_datetime(datos['mes'], format='%Y-%m', errors='coerce')
        salida = pd.DataFrame({
            'mes': fechas.dt.strftime('%Y-%m'),
            'ipc': convertir_numeros(datos['ipc']),
        })
    else:
        raise ValueError(
            f'Formato IPC no reconocido: {ruta.name}. Columnas: {list(datos.columns)}'
        )

    # Un mes duplicado no se debe resolver arbitrariamente con un promedio.
    if salida['mes'].isna().any() or salida['ipc'].isna().any():
        raise ValueError('IPC: existen fechas o índices inválidos en la selección')
    if salida.duplicated('mes').any():
        meses = salida.loc[salida.duplicated('mes'), 'mes'].tolist()
        raise ValueError(f'IPC: meses duplicados {meses[:10]}')
    if (salida['ipc'] <= 0).any():
        raise ValueError('IPC: se detectaron valores no positivos')
    return salida.sort_values('mes').reset_index(drop=True)


def cargar_ipc():
    """Empalma 2016 con GBA posterior y exige todos los meses hasta 2026-06.

    El IPC promedio trimestral aproxima el nivel de precios correspondiente
    al trimestre de la encuesta. Se usa como base 2026 T2.
    """
    opciones = [DATOS / 'ipc_gba_mensual.csv', DATOS / 'ipc_ng_gba.csv',
                DATOS / 'ipc_nivel_general.csv', DATOS / 'serie_ipc_divisiones.csv',
                DATOS / 'serie_ipc_aperturas.csv']
    ruta = next((r for r in opciones if r.is_file()), None)
    if ruta is None:
        raise FileNotFoundError(
            'Falta el IPC: descargá la serie del nivel general REGIÓN GBA '\
            'y guardala como datos/ipc_gba_mensual.csv (columnas mes,ipc). '\
            'Fuente: https://www.indec.gob.ar/ftp/cuadros/economia/serie_ipc_divisiones.csv'
        )
    nuevo = preparar_ipc_fuente(ruta)
    # Mantenemos los datos posteriores a diciembre 2016; 2016 se agrega
    # siempre de la serie histórica oficial para tener un empalme explícito.
    nuevo = nuevo[nuevo['mes'] >= '2016-12']
    if nuevo.duplicated('mes').any():
        raise ValueError('IPC: meses duplicados en el archivo de entrada')
    if '2016-12' not in set(nuevo['mes']):
        raise ValueError('El IPC de origen debe contener diciembre 2016 para enlazar las series')
    ipc_dic = float(nuevo.loc[nuevo['mes'] == '2016-12', 'ipc'].iloc[0])
    if not (90 <= ipc_dic <= 110):
        raise ValueError('IPC: se esperaba diciembre 2016 cercano a 100. Revisar la base.')
    # Rebase exacto en caso de decimales o redondeos.
    nuevo['ipc'] = nuevo['ipc'] / ipc_dic * 100
    viejo = serie_ipc_2016()
    completo = pd.concat([viejo[viejo['mes'] < '2016-12'], nuevo], ignore_index=True)
    esperado = pd.period_range('2016-04', '2026-06', freq='M').astype(str)
    tabla = completo.set_index('mes').reindex(esperado)
    faltan = tabla.index[tabla['ipc'].isna()].tolist()
    if faltan:
        raise ValueError(f'La serie IPC regional GBA no está completa. Faltan meses: {faltan}')
    if (tabla['ipc'] <= 0).any():
        raise ValueError('IPC contiene índices no positivos')
    tabla = tabla.reset_index(names='mes')
    tabla['anio'] = tabla['mes'].str[:4].astype(int)
    tabla['trimestre'] = ((tabla['mes'].str[5:7].astype(int) - 1) // 3 + 1)
    return tabla


def guardar_csv(df, nombre):
    """Guarda CSV compatible con Excel en español (punto y coma, coma decimal)."""
    ruta = RESULTADOS / nombre
    df.to_csv(ruta, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print('Guardado:', ruta)


def graficar_serie(df, columna, archivo, titulo, etiqueta, moneda=False):
    """Dibuja serie trimestral por aglomerado, sin alterar observaciones."""
    fig, ax = plt.subplots(figsize=(12, 5))
    periodos = df['periodo'].drop_duplicates().tolist()
    mapa = {p: i for i, p in enumerate(periodos)}
    for nombre, grupo in df.groupby('aglomerado', sort=False):
        ax.plot([mapa[p] for p in grupo['periodo']], grupo[columna],
                label=nombre, linewidth=1.7, marker='o', markersize=3)
    ax.set_xticks(range(0, len(periodos), 4), periodos[::4], rotation=45, ha='right')
    ax.set_title(titulo)
    ax.set_xlabel('Trimestre')
    ax.set_ylabel(etiqueta)
    if moneda:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v / 1e6:.1f} M'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / archivo, dpi=150)
    plt.close(fig)


def main():
    """Procesa todos los TXT, guarda análisis ponderado y luego deflacta."""
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)
    filas = []
    frecuencias = []
    for anio, trim, ruta in archivos_eph():
        print(f'Ponderando {anio} T{trim}: {ruta.name}')
        datos = leer_eph(ruta, anio, trim)
        for codigo, nombre in AGLOMERADOS.items():
            grupo = datos[datos['AGLOMERADO'] == codigo]
            if grupo.empty:
                raise ValueError(f'Sin registros para {nombre}, {anio} T{trim}')
            filas.append(analizar_grupo(grupo, anio, trim, codigo))
            frecuencias.extend(frecuencias_ponderadas(grupo, anio, trim, codigo))
    tabla = pd.DataFrame(filas).sort_values(['anio', 'trimestre', 'codigo_aglomerado'])
    guardar_csv(tabla, 'resumen_ponderado_2016_2026.csv')
    guardar_csv(pd.DataFrame(frecuencias), 'frecuencias_ponderadas_2016_2026.csv')

    # Una tabla pequeña para revisar dónde hubo pesos de ingreso que
    # no pueden aportar al cálculo y si hay no respondentes con peso > 0.
    columnas_auditoria = [
        'periodo', 'aglomerado', 'n_ingresos_conocidos',
        'n_ingresos_conocidos_pondii_positivo',
        'n_ingresos_conocidos_pondii_cero',
        'n_ingresos_conocidos_pondii_faltante',
        'pct_conocidos_sin_peso_positivo',
        'n_no_respondentes_pondii_positivo',
    ]
    auditoria = tabla[columnas_auditoria].copy()
    guardar_csv(auditoria, 'auditoria_ponderadores_ingreso.csv')
    atencion = auditoria[
        (auditoria['pct_conocidos_sin_peso_positivo'] > 0) |
        (auditoria['n_no_respondentes_pondii_positivo'] > 0)
    ]
    print(f'\nAuditoría: {len(atencion)} de {len(auditoria)} filas tienen observaciones para revisar.')
    if not atencion.empty:
        print(atencion.to_string(index=False))
    graficar_serie(tabla, 'no_respuesta_pct_ponderado',
                   'no_respuesta_ponderada.png',
                   'No respuesta de P47T (ponderada con PONDERA)', 'Porcentaje (%)')

    try:
        ipc = cargar_ipc()
    except (FileNotFoundError, ValueError) as error:
        print('\nATENCIÓN: cálculos ponderados completos; IPC pendiente.')
        print(error)
        print('No se generaron ingresos reales: falta validar la fuente de IPC.')
        return

    trimestral = ipc.groupby(['anio', 'trimestre'], as_index=False)['ipc'].mean()
    base = float(trimestral.loc[(trimestral.anio == 2026) &
                                 (trimestral.trimestre == 2), 'ipc'].iloc[0])
    tabla = tabla.merge(trimestral, on=['anio', 'trimestre'], how='left', validate='many_to_one')
    if tabla['ipc'].isna().any():
        raise ValueError('Quedaron trimestres sin IPC después del empalme.')
    tabla['ipc_gba_promedio_trimestral'] = tabla.pop('ipc')
    tabla['factor_a_pesos_2026_t2'] = base / tabla['ipc_gba_promedio_trimestral']
    for columna in ('ingreso_con_ceros_media', 'ingreso_con_ceros_mediana',
                    'ingreso_positivo_media', 'ingreso_positivo_q1',
                    'ingreso_positivo_mediana', 'ingreso_positivo_q3'):
        tabla[columna + '_real_2026_t2'] = tabla[columna] * tabla['factor_a_pesos_2026_t2']
    guardar_csv(ipc, 'ipc_gba_mensual_validado.csv')
    guardar_csv(trimestral, 'ipc_gba_promedio_trimestral.csv')
    guardar_csv(tabla, 'resumen_ponderado_ingresos_reales_2016_2026.csv')
    graficar_serie(tabla, 'ingreso_positivo_mediana_real_2026_t2',
                   'ingresos_reales_mediana_ponderada.png',
                   'Mediana ponderada de ingresos individuales positivos, pesos de 2026 T2',
                   'Millones de pesos constantes (2026 T2)', moneda=True)
    print('\nLISTO: los 41 trimestres tienen ponderaciones e ingresos reales.')
    print('Control 2025-T2:')
    print(tabla[(tabla.anio == 2025) & (tabla.trimestre == 2)][[
        'aglomerado', 'no_respuesta_pct_ponderado', 'ingreso_positivo_mediana',
        'ingreso_positivo_mediana_real_2026_t2'
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
