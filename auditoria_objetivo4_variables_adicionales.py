from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "datos" / "usu_individual_T226.txt"


VARIABLES = [
    "AGLOMERADO",
    "H15",
    "ESTADO",
    "CAT_OCUP",
    "P47T",

    # Variables nuevas que queremos evaluar.
    "PP04D_COD",
    "PP04C99",
    "PP04G",
    "PP07H",
    "PP07J",
    "PP07K",
]


def main():

    # --------------------------------------------------------
    # Primero miramos qué columnas existen realmente.
    # --------------------------------------------------------

    encabezado = pd.read_csv(
        ARCHIVO,
        sep=";",
        encoding="latin-1",
        nrows=0,
    )

    disponibles = {
        columna.strip().upper()
        for columna in encabezado.columns
    }

    presentes = [
        variable
        for variable in VARIABLES
        if variable in disponibles
    ]

    ausentes = [
        variable
        for variable in VARIABLES
        if variable not in disponibles
    ]

    print("=" * 70)
    print("VARIABLES PRESENTES")
    print("=" * 70)

    for variable in presentes:
        print(" -", variable)

    if ausentes:
        print("\nVariables ausentes:")
        for variable in ausentes:
            print(" -", variable)

    # --------------------------------------------------------
    # Lectura.
    # --------------------------------------------------------

    tipos = {}

    if "PP04D_COD" in presentes:
        tipos["PP04D_COD"] = "string"

    df = pd.read_csv(
        ARCHIVO,
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
        usecols=presentes,
        dtype=tipos,
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    # CABA + Partidos del GBA, entrevistas realizadas.
    df = df.loc[
        df["AGLOMERADO"].isin([32, 33])
        & df["H15"].eq(1)
    ].copy()

    # --------------------------------------------------------
    # Separamos ingreso conocido y no respuesta.
    # --------------------------------------------------------

    df["grupo_ingreso"] = "Ingreso conocido"

    df.loc[
        df["P47T"].isna()
        | df["P47T"].eq(-9),
        "grupo_ingreso",
    ] = "No respuesta P47T"

    # --------------------------------------------------------
    # Auditoría por variable.
    # --------------------------------------------------------

    filas = []

    nuevas = [
        "PP04D_COD",
        "PP04C99",
        "PP04G",
        "PP07H",
        "PP07J",
        "PP07K",
    ]

    for variable in nuevas:

        if variable not in df.columns:
            continue

        for grupo, parte in df.groupby(
            "grupo_ingreso"
        ):

            serie = parte[variable]

            filas.append({
                "variable": variable,
                "grupo": grupo,
                "n": len(parte),
                "n_con_dato": int(
                    serie.notna().sum()
                ),
                "n_sin_dato": int(
                    serie.isna().sum()
                ),
                "cobertura_pct": (
                    100
                    * serie.notna().mean()
                ),
                "valores_distintos": int(
                    serie.nunique(
                        dropna=True
                    )
                ),
            })

    resumen = pd.DataFrame(
        filas
    )

    print("\n" + "=" * 70)
    print("COBERTURA")
    print("=" * 70)

    print(
        resumen.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Auditoría específica entre ocupados.
    # --------------------------------------------------------

    ocupados = df.loc[
        df["ESTADO"] == 1
    ].copy()

    print("\n" + "=" * 70)
    print("ENTRE OCUPADOS")
    print("=" * 70)

    for variable in nuevas:

        if variable not in ocupados.columns:
            continue

        print(f"\n{variable}")

        print(
            ocupados[variable]
            .value_counts(
                dropna=False
            )
            .head(20)
        )

    # --------------------------------------------------------
    # Primer dígito del código de ocupación.
    # --------------------------------------------------------

    if "PP04D_COD" in ocupados.columns:

        codigos = (
            ocupados["PP04D_COD"]
            .astype("string")
            .str.strip()
        )

        # El CNO tiene cinco dígitos.
        # zfill conserva correctamente los códigos
        # cuyo primer dígito sea 0.
        codigos = codigos.str.zfill(5)

        ocupados[
            "grupo_cno"
        ] = codigos.str[0]

        print("\n" + "=" * 70)
        print("GRANDES GRUPOS CNO - PRIMER DÍGITO")
        print("=" * 70)

        print(
            ocupados[
                "grupo_cno"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

    # --------------------------------------------------------
    # Guardado.
    # --------------------------------------------------------

    salida = (
        BASE
        / "resultados"
        / "objetivo4_auditoria_variables_adicionales.csv"
    )

    resumen.to_csv(
        salida,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nCSV guardado: {salida}"
    )


if __name__ == "__main__":
    main()