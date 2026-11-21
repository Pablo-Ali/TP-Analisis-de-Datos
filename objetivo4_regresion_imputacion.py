"""Objetivo 4 - Regresión lineal múltiple e imputación de P47T.

Modelo final utilizado en el informe:
- 41 trimestres: 2016 T2 a 2026 T2;
- P47T permanece nominal y sin transformación;
- los efectos de los predictores se escalan con el IPC;
- validación cruzada de 5 folds, manteniendo hogares completos;
- R² principal: conjunto de predicciones fuera de muestra (OOF);
- luego se ajusta el modelo sobre todos los ingresos conocidos e imputa P47T=-9;
- las predicciones negativas solo se llevan a cero al generar la imputación final.

Este archivo sustituye las pruebas intermedias de regresión.

Requiere:
- objetivos2_3_analisis.py en la misma carpeta;
- datos/ con las 41 bases y serie_ipc_divisiones.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from objetivos2_3_analisis import (
    AGLOMERADOS,
    DATOS,
    RESULTADOS,
    GRAFICOS,
    buscar_archivos,
    factores_ipc,
    clasificar_rama,
)


N_FOLDS = 5
SEMILLA = 42

COLUMNAS = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE",
    "AGLOMERADO", "H15",
    "CH04", "CH06", "NIVEL_ED",
    "ESTADO", "CAT_OCUP",
    "PP04B_COD", "PP3E_TOT",
    "P47T",
]

MAPA_EDUCACION = {
    7: "Sin instrucción",
    1: "Primaria incompleta",
    2: "Primario completo o secundario incompleto",
    3: "Primario completo o secundario incompleto",
    4: "Secundario completo o superior incompleto",
    5: "Secundario completo o superior incompleto",
    6: "Superior completo",
    9: "No sabe / no responde",
}


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

def leer_base_modelo(ruta, anio, trimestre, factor):
    df = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=COLUMNAS,
        dtype={"CODUSU": "string", "PP04B_COD": "string"},
    )

    df.columns = df.columns.str.strip().str.upper()

    for col in df.columns:
        if col not in {"CODUSU", "PP04B_COD"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[
        df["AGLOMERADO"].isin(AGLOMERADOS)
        & df["H15"].eq(1)
    ].copy()

    df["periodo"] = f"{anio}-T{trimestre}"
    df["factor_ipc"] = float(factor)
    df["escala_precios"] = 1.0 / df["factor_ipc"]

    return df


def cargar_historia():
    factores = factores_ipc()
    partes = []

    archivos = buscar_archivos()
    print(f"Se encontraron {len(archivos)} bases.")

    for anio, trimestre, ruta in archivos:
        print(f"Procesando {anio} T{trimestre}: {ruta.name}")
        partes.append(
            leer_base_modelo(
                ruta,
                anio,
                trimestre,
                factores[(anio, trimestre)],
            )
        )

    return pd.concat(partes, ignore_index=True)


def preparar_variables(df):
    datos = df.copy()
    ocupados = datos["ESTADO"].eq(1)

    datos["edad_modelo"] = datos["CH06"].replace(-1, 0).astype(float)
    datos["educacion_modelo"] = datos["NIVEL_ED"].map(MAPA_EDUCACION)

    if datos["educacion_modelo"].isna().any():
        raise ValueError("Hay códigos NIVEL_ED no contemplados")

    # Referencias elegidas en el modelo final.
    datos["cat_ocup_modelo"] = "3"
    datos.loc[ocupados, "cat_ocup_modelo"] = (
        datos.loc[ocupados, "CAT_OCUP"]
        .astype("Int64")
        .astype("string")
    )

    datos["rama_modelo"] = "Comercio"
    datos.loc[ocupados, "rama_modelo"] = (
        datos.loc[ocupados, "PP04B_COD"].map(clasificar_rama)
    )

    horas = datos["PP3E_TOT"]
    horas_validas = ocupados & horas.notna() & horas.between(0, 168)
    horas_especiales = ocupados & (horas.isna() | horas.gt(168))

    datos["horas_modelo"] = 0.0
    datos.loc[horas_validas, "horas_modelo"] = (
        horas.loc[horas_validas].astype(float)
    )
    datos["horas_especiales"] = horas_especiales.astype(float)

    ingreso = datos["P47T"]
    invalidos = ingreso.notna() & ingreso.lt(0) & ingreso.ne(-9)
    if invalidos.any():
        raise ValueError("P47T contiene códigos negativos inesperados")

    datos["ingreso_conocido"] = ingreso.notna() & ingreso.ge(0)
    datos["ingreso_no_responde"] = ingreso.isna() | ingreso.eq(-9)

    return datos


# ============================================================
# MATRIZ DEL MODELO
# ============================================================

def id_hogar(df):
    return (
        df["CODUSU"].astype("string")
        + "_"
        + df["NRO_HOGAR"].astype("Int64").astype("string")
    )


def crear_folds(df):
    hogares = id_hogar(df).drop_duplicates().to_numpy()

    rng = np.random.default_rng(SEMILLA)
    rng.shuffle(hogares)

    grupos = np.array_split(hogares, N_FOLDS)

    return {
        hogar: fold
        for fold, grupo in enumerate(grupos, start=1)
        for hogar in grupo
    }


def _texto(serie):
    return serie.astype("string").fillna("Sin dato")


def matrices_base(train, test):
    x_train = pd.DataFrame(index=train.index)
    x_test = pd.DataFrame(index=test.index)

    for variable in ["edad_modelo", "horas_modelo", "horas_especiales"]:
        x_train[variable] = pd.to_numeric(
            train[variable], errors="coerce"
        ).fillna(0).astype(float)

        x_test[variable] = pd.to_numeric(
            test[variable], errors="coerce"
        ).fillna(0).astype(float)

    categoricas = {
        "periodo": "2026-T2",
        "AGLOMERADO": "32",
        "CH04": "1",
        "educacion_modelo": "Superior completo",
        "ESTADO": "1",
        "cat_ocup_modelo": "3",
        "rama_modelo": "Comercio",
    }

    for variable, referencia in categoricas.items():
        tr = _texto(train[variable])
        te = _texto(test[variable])

        categorias = sorted(tr.unique())
        referencia = str(referencia)

        if referencia not in categorias:
            referencia = tr.value_counts().index[0]

        for categoria in categorias:
            if categoria == referencia:
                continue

            nombre = f"{variable}={categoria}"
            x_train[nombre] = tr.eq(categoria).astype(float)
            x_test[nombre] = te.eq(categoria).astype(float)

    x_test = x_test.reindex(
        columns=x_train.columns,
        fill_value=0.0,
    )

    return x_train, x_test


def construir_matrices(train, test):
    """Replica la especificación final usada para obtener R² OOF=0,4812."""
    base_train, base_test = matrices_base(train, test)

    escala_train = train["escala_precios"].astype(float)
    escala_test = test["escala_precios"].astype(float)

    final_train = pd.DataFrame(index=train.index)
    final_test = pd.DataFrame(index=test.index)

    # Actúa como intercepto que cambia con el nivel general de precios.
    final_train["escala_precios"] = escala_train
    final_test["escala_precios"] = escala_test

    for columna in base_train.columns:
        nombre = f"{columna}_x_escala"
        final_train[nombre] = base_train[columna] * escala_train
        final_test[nombre] = base_test[columna] * escala_test

    return final_train, final_test


# ============================================================
# REGRESIÓN Y MÉTRICAS
# ============================================================

def ajustar(x, y):
    coef, _, rango, _ = np.linalg.lstsq(
        x.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        rcond=None,
    )
    return coef, rango


def predecir(x, coef):
    return x.to_numpy(dtype=float) @ coef


def r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    total = np.sum((y - y.mean()) ** 2)
    if total <= 0:
        return np.nan

    return float(
        1 - np.sum((y - pred) ** 2) / total
    )


def metricas(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = y - pred
    mse = float(np.mean(error ** 2))

    return {
        "r2": r2(y, pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "pred_negativas_pct": float(100 * np.mean(pred < 0)),
    }


# ============================================================
# VALIDACIÓN CRUZADA
# ============================================================

def validar(datos):
    conocidos = datos.loc[datos["ingreso_conocido"]].copy()

    conocidos["_hogar"] = id_hogar(conocidos)
    mapa_folds = crear_folds(conocidos)
    conocidos["_fold"] = conocidos["_hogar"].map(mapa_folds)

    filas_folds = []
    predicciones = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\nFold {fold}/{N_FOLDS}")

        train = conocidos.loc[conocidos["_fold"] != fold].copy()
        test = conocidos.loc[conocidos["_fold"] == fold].copy()

        x_train, x_test = construir_matrices(train, test)

        y_train = train["P47T"].astype(float)
        y_test = test["P47T"].astype(float)

        coef, rango = ajustar(x_train, y_train)
        pred = predecir(x_test, coef)
        met = metricas(y_test, pred)

        filas_folds.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "n_predictores": x_train.shape[1],
            "rango_completo": rango == x_train.shape[1],
            **met,
        })

        salida = test[
            [
                "CODUSU", "NRO_HOGAR", "COMPONENTE",
                "periodo", "AGLOMERADO", "ESTADO", "P47T",
            ]
        ].copy()
        salida["fold"] = fold
        salida["prediccion"] = pred
        predicciones.append(salida)

        print(
            f"R²={met['r2']:.4f} | "
            f"MAE=${met['mae']:,.0f} | "
            f"RMSE=${met['rmse']:,.0f}"
        )

    return (
        pd.DataFrame(filas_folds),
        pd.concat(predicciones, ignore_index=True),
    )


def rendimiento_por_periodo(predicciones):
    filas = []

    for periodo, parte in predicciones.groupby("periodo"):
        met = metricas(parte["P47T"], parte["prediccion"])
        filas.append({
            "periodo": periodo,
            "n": len(parte),
            "r2": met["r2"],
            "mae": met["mae"],
            "rmse": met["rmse"],
        })

    return pd.DataFrame(filas)


def ajustar_completo(datos):
    conocidos = datos.loc[datos["ingreso_conocido"]].copy()

    x, _ = construir_matrices(conocidos, conocidos)
    y = conocidos["P47T"].astype(float)

    coef, rango = ajustar(x, y)
    pred = predecir(x, coef)

    tabla_coeficientes = pd.DataFrame({
        "variable": x.columns,
        "coeficiente": coef,
    })

    return {
        "n": len(y),
        "n_predictores": x.shape[1],
        "rango_completo": rango == x.shape[1],
        "r2_ajuste_completo": r2(y, pred),
    }, tabla_coeficientes, x, coef


# ============================================================
# IMPUTACIÓN
# ============================================================

def imputar(datos):
    conocidos = datos.loc[datos["ingreso_conocido"]].copy()
    faltantes = datos.loc[datos["ingreso_no_responde"]].copy()

    x_train, x_faltantes = construir_matrices(conocidos, faltantes)
    coef, rango = ajustar(
        x_train,
        conocidos["P47T"].astype(float),
    )

    if rango != x_train.shape[1]:
        raise ValueError("La matriz final no tiene rango completo")

    pred = predecir(x_faltantes, coef)

    imputados = faltantes[
        [
            "CODUSU", "NRO_HOGAR", "COMPONENTE",
            "periodo", "AGLOMERADO", "ESTADO", "P47T",
        ]
    ].copy()

    imputados = imputados.rename(columns={"P47T": "P47T_original"})
    imputados["P47T_predicho"] = pred
    imputados["prediccion_negativa_original"] = (
        imputados["P47T_predicho"] < 0
    )

    # Esta corrección se aplica DESPUÉS de evaluar el modelo.
    imputados["P47T_imputado"] = (
        imputados["P47T_predicho"].clip(lower=0)
    )
    imputados["fue_imputado"] = True

    completa = datos[
        [
            "CODUSU", "NRO_HOGAR", "COMPONENTE",
            "periodo", "AGLOMERADO", "ESTADO", "P47T",
        ]
    ].copy().rename(columns={"P47T": "P47T_original"})

    claves = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "periodo"]

    completa = completa.merge(
        imputados[
            claves
            + [
                "P47T_predicho",
                "P47T_imputado",
                "prediccion_negativa_original",
            ]
        ],
        on=claves,
        how="left",
        validate="one_to_one",
    )

    completa["fue_imputado"] = (
        completa["P47T_original"].isna()
        | completa["P47T_original"].eq(-9)
    )

    completa["P47T_completo"] = completa["P47T_original"]
    mascara = completa["fue_imputado"]
    completa.loc[mascara, "P47T_completo"] = (
        completa.loc[mascara, "P47T_imputado"]
    )

    resumen = (
        imputados.groupby("periodo", as_index=False)
        .agg(
            n_imputados=("P47T_imputado", "size"),
            n_predicciones_negativas=(
                "prediccion_negativa_original", "sum"
            ),
            media_imputada=("P47T_imputado", "mean"),
            mediana_imputada=("P47T_imputado", "median"),
        )
    )
    resumen["predicciones_negativas_pct"] = (
        100
        * resumen["n_predicciones_negativas"]
        / resumen["n_imputados"]
    )

    return imputados, completa, resumen


# ============================================================
# GRÁFICOS
# ============================================================

def graficar_validacion(folds, r2_oof):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(folds["fold"].astype(str), folds["r2"])
    ax.axhline(
        0.30,
        linestyle="--",
        linewidth=1.5,
        label="Umbral de la consigna (R² = 0,30)",
    )
    ax.axhline(
        r2_oof,
        linestyle=":",
        linewidth=1.5,
        label=f"R² OOF conjunto = {r2_oof:.3f}",
    )

    ax.set_xlabel("Fold de validación")
    ax.set_ylabel("R²")
    ax.set_title("Rendimiento fuera de muestra de la regresión")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()

    fig.tight_layout()
    fig.savefig(GRAFICOS / "objetivo4_r2_validacion.png", dpi=150)
    plt.close(fig)


def graficar_observado_predicho(predicciones):
    limite = float(predicciones["P47T"].quantile(0.99))

    visual = predicciones.loc[
        predicciones["P47T"].between(0, limite)
        & predicciones["prediccion"].between(0, limite)
    ].copy()

    if len(visual) > 12000:
        visual = visual.sample(12000, random_state=42)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        visual["P47T"],
        visual["prediccion"],
        alpha=0.20,
        s=10,
    )
    ax.plot(
        [0, limite],
        [0, limite],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("P47T observado")
    ax.set_ylabel("P47T predicho")
    ax.set_title("Ingresos observados y predichos")
    ax.grid(alpha=0.25, linestyle="--")

    fig.text(
        0.5, 0.01,
        "Visualización limitada al percentil 99. "
        "Las métricas utilizan todas las observaciones.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(
        GRAFICOS / "objetivo4_observado_predicho.png",
        dpi=150,
    )
    plt.close(fig)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICOS.mkdir(exist_ok=True)

    datos = preparar_variables(cargar_historia())

    conocidos = int(datos["ingreso_conocido"].sum())
    faltantes = int(datos["ingreso_no_responde"].sum())

    print(f"\nIngresos conocidos: {conocidos:,}")
    print(f"No respuestas: {faltantes:,}")

    folds, pred_oof = validar(datos)
    por_periodo = rendimiento_por_periodo(pred_oof)

    ajuste, coeficientes, _, _ = ajustar_completo(datos)
    met_oof = metricas(pred_oof["P47T"], pred_oof["prediccion"])

    resumen_modelo = pd.DataFrame([{
        **ajuste,
        "r2_cv_promedio": folds["r2"].mean(),
        "r2_cv_min": folds["r2"].min(),
        "r2_cv_max": folds["r2"].max(),
        "r2_oof_conjunto": met_oof["r2"],
        "mae_oof": met_oof["mae"],
        "rmse_oof": met_oof["rmse"],
        "pred_negativas_pct": met_oof["pred_negativas_pct"],
        "r2_periodos_promedio": por_periodo["r2"].mean(),
        "r2_periodos_mediana": por_periodo["r2"].median(),
        "r2_periodos_min": por_periodo["r2"].min(),
        "r2_periodos_max": por_periodo["r2"].max(),
        "periodos_r2_mayor_030": int((por_periodo["r2"] > 0.30).sum()),
        "cantidad_periodos": len(por_periodo),
    }])

    # Resultados de validación.
    resumen_modelo.to_csv(
        RESULTADOS / "objetivo4_regresion_historica_ipc_resumen.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    folds.to_csv(
        RESULTADOS / "objetivo4_regresion_historica_ipc_folds.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    por_periodo.to_csv(
        RESULTADOS / "objetivo4_regresion_historica_ipc_periodos.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    coeficientes.to_csv(
        RESULTADOS / "objetivo4_regresion_historica_ipc_coeficientes.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    pred_oof.to_csv(
        RESULTADOS / "objetivo4_regresion_historica_ipc_predicciones.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )

    # Imputación final.
    imputados, completa, resumen_imputacion = imputar(datos)

    imputados.to_csv(
        RESULTADOS / "objetivo4_p47t_imputados.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    completa.to_csv(
        RESULTADOS / "objetivo4_p47t_base_completa.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )
    resumen_imputacion.to_csv(
        RESULTADOS / "objetivo4_resumen_imputacion.csv",
        sep=";", decimal=",", index=False, encoding="utf-8-sig",
    )

    graficar_validacion(folds, met_oof["r2"])
    graficar_observado_predicho(pred_oof)

    negativos = int(
        imputados["prediccion_negativa_original"].sum()
    )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"R² OOF conjunto: {met_oof['r2']:.7f}")
    print(f"MAE OOF: ${met_oof['mae']:,.0f}")
    print(f"RMSE OOF: ${met_oof['rmse']:,.0f}")
    print(
        f"Trimestres con R² > 0,30: "
        f"{int((por_periodo['r2'] > 0.30).sum())}/{len(por_periodo)}"
    )
    print(f"Ingresos imputados: {len(imputados):,}")
    print(
        f"Predicciones negativas antes de corregir: "
        f"{negativos:,} ({100 * negativos / len(imputados):.2f}%)"
    )
    print(
        f"Mediana imputada general: "
        f"${imputados['P47T_imputado'].median():,.0f}"
    )
    print(
        f"Media imputada general: "
        f"${imputados['P47T_imputado'].mean():,.0f}"
    )

    # Control contra los resultados del informe.
    esperado_r2 = 0.48122070976290743
    if not np.isclose(met_oof["r2"], esperado_r2, atol=1e-8):
        print(
            "\nAVISO: el R² no coincide exactamente con la ejecución "
            "usada para el informe. Revisar datos, IPC o versiones."
        )

    if conocidos != 231760 or len(imputados) != 55196:
        print(
            "\nAVISO: los conteos no coinciden con el informe "
            "(231.760 conocidos y 55.196 imputados)."
        )


if __name__ == "__main__":
    main()
