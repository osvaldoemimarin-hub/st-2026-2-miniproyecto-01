"""Miniproyecto 1: una serie propia, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. El
informe, docs/informe.md, cita las cifras y las figuras que produce este
guion.

Serie: cabezas de ganado porcino sacrificadas por mes en Colombia, total
nacional (DANE, Encuesta de Sacrificio de Ganado), de 2008-10 a 2026-06.
"""
import itertools
import warnings

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)

warnings.filterwarnings("ignore")
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]

# ============================================================
# 1. La serie
# ============================================================
ARCHIVO = "datos/sacrificio_porcino.csv"
FECHA = "mes"
VALOR = "porcinos"
PERIODO = 12

# Los modelos se ajustan sobre serie / ESCALA (miles de cabezas) para que
# sigma2 quede del orden de los coeficientes; los pronósticos se devuelven
# en cabezas, así que el backtest y la tabla quedan en las unidades de la serie.
ESCALA = 1e3
PAIS = "CO"

# Choques de un solo mes, con causa conocida y que no se repiten. Se
# reemplazan por lo que la STL robusta esperaba en ese mes (tendencia +
# estacional), para que el SARIMA no aprenda de ellos.
CHOQUES = {"2020-04-01": "cuarentena nacional por el COVID-19",
           "2021-05-01": "paro nacional"}

DECISION = ("sin huecos; abr-2020 y may-2021 (cuarentena y paro) "
            "reemplazados por tendencia + estacional de la STL robusta")


def atipicos(serie, n=5):
    """Los n meses con el residuo más grande en una STL robusta de la serie."""
    stl = STL(serie, period=PERIODO, robust=True).fit()
    grandes = stl.resid.abs().nlargest(n).index
    tabla = pd.DataFrame({"observado": serie[grandes],
                          "esperado": stl.trend[grandes] + stl.seasonal[grandes],
                          "residuo": stl.resid[grandes],
                          "residuo / desv.": stl.resid[grandes] / stl.resid.std(),
                          "peso": stl.weights[grandes]})
    tabla.index = tabla.index.strftime("%Y-%m")
    return tabla.round(2)


def preparar(serie):
    """Interpola los huecos y reemplaza los choques de CHOQUES.

    El valor que entra en cada choque es tendencia + estacional de la STL
    robusta, que con robust=True no se deja arrastrar por el propio choque.
    Los dos meses quedan antes del primer origen del backtest (2024-06).
    """
    serie = serie.interpolate(method="linear")
    stl = STL(serie, period=PERIODO, robust=True).fit()
    corregida = serie.copy()
    for mes in CHOQUES:
        corregida[mes] = stl.trend[mes] + stl.seasonal[mes]
    return corregida


def graficar_serie(serie, cruda):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy() / 1e3, lw=1.2, label="serie preparada")
    meses = pd.to_datetime(list(CHOQUES))
    eje.scatter(meses, cruda[meses] / 1e3, color="tab:red", zorder=3,
                label="dato original del choque")
    for mes, causa in CHOQUES.items():
        eje.annotate(causa, (pd.Timestamp(mes), cruda[mes] / 1e3), xytext=(6, -14),
                     textcoords="offset points", fontsize=8, color="tab:red")
    eje.set_title("Sacrificio mensual de ganado porcino en Colombia (DANE, ESAG)")
    eje.set_ylabel("miles de cabezas")
    eje.legend(loc="upper left", frameon=False)
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. Descomposición
# ============================================================

def descomponer(serie):
    """STL robusta con el periodo de la serie; guarda figuras/stl.png."""
    stl = STL(serie, period=PERIODO, robust=True).fit()
    figura = stl.plot()
    figura.set_size_inches(9, 6.5)
    figura.suptitle("STL robusta, periodo 12 (cabezas por mes)", y=1.0)
    figura.tight_layout()
    figura.savefig(FIGURAS / "stl.png", dpi=120)
    plt.close(figura)

    tendencia = stl.trend
    inicio, fin = tendencia.iloc[0], tendencia.iloc[-1]
    anios = (len(tendencia) - 1) / 12
    print(f"  tendencia: {inicio:,.0f} en {tendencia.index[0]:%Y-%m} -> {fin:,.0f} en "
          f"{tendencia.index[-1]:%Y-%m} (x{fin / inicio:.2f}, "
          f"{((fin / inicio) ** (1 / anios) - 1) * 100:.1f} % anual)")
    por_mes = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    por_mes.index = MESES
    print("  estacional promedio por mes:",
          ", ".join(f"{m} {v:+,.0f}" for m, v in por_mes.items()))
    diciembre = stl.seasonal[stl.seasonal.index.month == 12]
    print(f"  estacional de diciembre: {diciembre.iloc[0]:,.0f} en {diciembre.index[0]:%Y} -> "
          f"{diciembre.iloc[-1]:,.0f} en {diciembre.index[-1]:%Y}")
    grandes = stl.resid.abs().nlargest(3).index
    print("  residuos más grandes:",
          ", ".join(f"{m:%Y-%m} {stl.resid[m]:+,.0f}" for m in grandes))
    return stl


# ============================================================
# 3. Estacionariedad e identificación
# ============================================================

def fuera_de_bandas(valores, intervalo):
    """Rezagos (sin el 0) cuyo valor sale de la banda del 95 % centrada en cero."""
    banda = intervalo[:, 1] - valores
    return {k: round(float(valores[k]), 2) for k in range(1, len(valores))
            if abs(valores[k]) > banda[k]}


def estacionariedad(serie):
    """ADF sobre la serie, su diferencia regular, la estacional y las dos.

    La serie original se prueba con regression="ct" porque crece sobre una
    recta; las diferenciadas, con "c", porque ya no tienen tendencia. Se
    elige la versión con menos diferencias que cumpla dos cosas: ADF rechaza
    la raíz unitaria (p < 0,05) y la ACF en el rezago 12 no queda persistente
    (r12 < 0,5). ADF no mira la raíz estacional; la ACF en 12 sí.
    """
    versiones = {(0, 0): serie, (1, 0): serie.diff(), (0, 1): serie.diff(PERIODO),
                 (1, 1): serie.diff(PERIODO).diff()}
    nombres = {(0, 0): "original", (1, 0): "diff()", (0, 1): "diff(12)",
               (1, 1): "diff(12).diff()"}
    filas = []
    for (d, D), x in versiones.items():
        x = x.dropna()
        regresion = "ct" if d + D == 0 else "c"
        estadistico, p, *_ = adfuller(x, regression=regresion, autolag="AIC")
        filas.append({"versión": nombres[(d, D)], "d": d, "D": D, "regression": regresion,
                      "estadístico": round(estadistico, 3), "valor p": round(p, 4),
                      "ACF(12)": round(acf(x, nlags=PERIODO)[PERIODO], 2)})
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    candidatas = tabla[(tabla["valor p"] < 0.05) & (tabla["ACF(12)"] < 0.5)]
    elegida = candidatas.assign(total=candidatas["d"] + candidatas["D"]).sort_values(
        ["total", "valor p"]).iloc[0]
    d, D = int(elegida["d"]), int(elegida["D"])
    print(f"  elegida: {elegida['versión']} -> d = {d}, D = {D}")

    x = versiones[(d, D)].dropna()
    r_acf, ic_acf = acf(x, nlags=36, alpha=0.05)
    r_pacf, ic_pacf = pacf(x, nlags=36, alpha=0.05, method="ywm")
    print(f"  banda aproximada: ±{1.96 / np.sqrt(len(x)):.3f} ({len(x)} datos)")
    print("  ACF fuera de bandas: ", fuera_de_bandas(r_acf, ic_acf))
    print("  PACF fuera de bandas:", fuera_de_bandas(r_pacf, ic_pacf))

    figura, ejes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    plot_acf(x, lags=36, ax=ejes[0], title=f"ACF, {elegida['versión']}")
    plot_pacf(x, lags=36, ax=ejes[1], method="ywm", title=f"PACF, {elegida['versión']}")
    for eje in ejes:
        eje.set_ylim(-1.05, 1.05)
        eje.set_xticks(range(0, 37, 6))
        eje.set_xlabel("rezago k (meses)")
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120)
    plt.close(figura)
    return d, D


# ============================================================
# 4. Modelos clásicos
# ============================================================
# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

def elegir_sarima(serie, d, D):
    """Rejilla de (p, q, P, Q) en {0, 1} comparada por AIC, con d y D fijos.

    d y D no compiten: cambian la serie que se modela y los AIC de series
    distintas no se comparan. Ljung-Box se aplica a los residuos sin los
    primeros d + 12 D meses, que absorben el arranque de las diferencias.
    """
    escalada = serie / ESCALA
    filas = []
    for p, q, P, Q in itertools.product([0, 1], repeat=4):
        ajuste = ARIMA(escalada, order=(p, d, q),
                       seasonal_order=(P, D, Q, PERIODO)).fit()
        filas.append({"order": (p, d, q), "seasonal_order": (P, D, Q, PERIODO),
                      "AIC": round(ajuste.aic, 1)})
    rejilla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    print(rejilla.to_string())

    orden = rejilla.loc[0, "order"]
    estacional = rejilla.loc[0, "seasonal_order"]
    mejor = ARIMA(escalada, order=orden, seasonal_order=estacional).fit()
    print(f"  elegido: SARIMA{orden}{estacional}, AIC {mejor.aic:.1f}")
    print("  coeficientes:", mejor.params.round(3).to_dict())
    print("  valores p:   ", mejor.pvalues.round(4).to_dict())
    residuos = mejor.resid.iloc[d + PERIODO * D:] * ESCALA
    print("  Ljung-Box de los residuos:")
    print(acorr_ljungbox(residuos, lags=[12, 24]).round(4).to_string())
    grandes = residuos.abs().nlargest(3).index
    print("  residuos más grandes:",
          ", ".join(f"{m:%Y-%m} {residuos[m]:+,.0f}" for m in grandes))
    graficar_residuos(residuos, f"SARIMA{orden}{estacional}")
    return orden, estacional


def graficar_residuos(residuos, titulo):
    figura = plt.figure(figsize=(10, 5.4))
    arriba = figura.add_subplot(2, 1, 1)
    arriba.plot(residuos.index, residuos / 1e3, lw=1)
    arriba.axhline(0, color="black", lw=0.8)
    arriba.set_title(f"Residuos del {titulo}")
    arriba.set_ylabel("miles de cabezas")
    plot_acf(residuos, lags=36, ax=figura.add_subplot(2, 2, 3),
             title="ACF de los residuos")
    histograma = figura.add_subplot(2, 2, 4)
    histograma.hist(residuos / 1e3, bins=25, color="tab:gray")
    histograma.set_title("Distribución de los residuos")
    histograma.set_xlabel("miles de cabezas")
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120)
    plt.close(figura)


def modelo_sarima(orden, orden_estacional):
    """modelo(entrenamiento, h) del SARIMA elegido, ajustado en miles de cabezas."""
    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, order=orden,
                       seasonal_order=orden_estacional).fit()
        return ajuste.forecast(h).to_numpy() * ESCALA
    return modelo


def calendario(serie):
    """Festivos y días hábiles de cada mes, de la serie y de los 12 meses que siguen.

    Salen del calendario oficial de Colombia (paquete holidays), que se
    conoce con años de anticipación: no hay que pronosticarlos.
    """
    indice = pd.date_range(serie.index[0], periods=len(serie) + 12, freq="MS")
    festivos = holidays.country_holidays(
        PAIS, years=range(indice[0].year, indice[-1].year + 1))
    filas = []
    for mes in indice:
        dias = pd.date_range(mes, mes + pd.offsets.MonthEnd(0))
        filas.append({"festivos": sum(dia in festivos for dia in dias),
                      "habiles": sum(dia.weekday() < 5 and dia not in festivos
                                     for dia in dias)})
    return pd.DataFrame(filas, index=indice, dtype=float)


def exogenas(serie):
    """Días hábiles de cada mes (lunes a viernes que no son festivo).

    Las plantas de beneficio sacrifican en días hábiles, así que un mes con
    más días hábiles debería sacrificar más. Cubre la serie y 12 meses más.
    """
    return calendario(serie)[["habiles"]]


def comparar_exogenas(serie, orden, estacional):
    """AIC del SARIMAX con cada candidata del calendario, sobre toda la serie."""
    escalada = serie / ESCALA
    candidatas = calendario(serie)
    for columnas in (["festivos"], ["habiles"]):
        x = candidatas.loc[escalada.index, columnas]
        ajuste = ARIMA(escalada, exog=x, order=orden, seasonal_order=estacional).fit()
        beta = ajuste.params[columnas[0]] * ESCALA
        ljung = acorr_ljungbox(ajuste.resid.iloc[orden[1] + PERIODO * estacional[1]:], lags=[12, 24])["lb_pvalue"]
        print(f"  {columnas[0]:>8}: AIC {ajuste.aic:.1f}, beta {beta:,.0f} cabezas "
              f"(valor p {ajuste.pvalues[columnas[0]]:.4f}), Ljung-Box p "
              f"{ljung.iloc[0]:.4f} (12) y {ljung.iloc[1]:.4f} (24)")


def modelo_sarimax(orden, orden_estacional, x):
    """modelo(entrenamiento, h) del SARIMAX con exógenas x.

    Ajusta con x.loc[entrenamiento.index] y pronostica con las x de las h
    fechas que siguen al origen; x es calendario, conocido de antemano.
    """
    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, exog=x.loc[entrenamiento.index],
                       order=orden, seasonal_order=orden_estacional).fit()
        futuro = x.loc[fechas_siguientes(entrenamiento, h)]
        return ajuste.forecast(h, exog=futuro).to_numpy() * ESCALA
    return modelo


# ============================================================
# 5. La tabla de métricas
# ============================================================

def fila(nombre, serie, modelo):
    _, promedio = backtest(serie, modelo)
    return {"modelo": nombre, "decision": DECISION,
            "RMSE": round(promedio["RMSE"], 2), "MAE": round(promedio["MAE"], 2),
            "MAPE": round(promedio["MAPE"], 2)}


def errores_del_backtest(serie, modelo, n_prueba=24, h=12):
    """Los pronósticos de cada origen, con los mismos cortes que comun.backtest."""
    filas = []
    for corte in range(len(serie) - n_prueba, len(serie) - h + 1):
        entrenamiento = serie.iloc[:corte]
        pronostico = np.asarray(modelo(entrenamiento, h))
        for paso, fecha in enumerate(fechas_siguientes(entrenamiento, h)):
            filas.append({"origen": entrenamiento.index[-1], "fecha": fecha,
                          "horizonte": paso + 1, "real": serie[fecha],
                          "pronostico": pronostico[paso]})
    errores = pd.DataFrame(filas)
    errores["error"] = errores["real"] - errores["pronostico"]
    return errores


def analizar_mejor(serie, nombre, modelo):
    """Dónde falla el mejor modelo: por horizonte y por mes del año."""
    errores = errores_del_backtest(serie, modelo)
    por_horizonte = errores.groupby("horizonte")["error"].agg(
        MAE=lambda e: e.abs().mean(), sesgo="mean")
    print(f"  {nombre}, error por horizonte (cabezas):")
    print(por_horizonte.round(0).astype(int).T.to_string())
    por_mes = errores.groupby(errores["fecha"].dt.month)["error"].agg(
        MAE=lambda e: e.abs().mean(), sesgo="mean")
    por_mes.index = [MESES[m - 1] for m in por_mes.index]
    print("  error por mes del año (cabezas):")
    print(por_mes.round(0).astype(int).T.to_string())
    print(f"  subestima en {(errores['error'] > 0).mean():.0%} de los pronósticos")

    figura, eje = plt.subplots(figsize=(9, 3.6))
    tramo = serie.iloc[-36:]
    eje.plot(tramo.index, tramo / 1e3, color="black", lw=1.6, label="real")
    for i, (_, grupo) in enumerate(errores.groupby("origen")):
        eje.plot(grupo["fecha"], grupo["pronostico"] / 1e3, color="tab:orange",
                 lw=0.8, alpha=0.6, label="pronóstico desde cada origen" if i == 0 else None)
    eje.set_title(f"Backtest del {nombre}: 13 orígenes, horizonte 1 a 12")
    eje.set_ylabel("miles de cabezas")
    eje.legend(loc="upper left", frameon=False)
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "backtest.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def titulo(texto):
    print(f"\n=== {texto} ===")


def main():
    titulo("1. La serie")
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    print("  meses más atípicos de la serie original (STL robusta):")
    print(atipicos(cruda.interpolate()).to_string())
    serie = preparar(cruda)
    for mes, causa in CHOQUES.items():
        print(f"  {mes[:7]} ({causa}): {cruda[mes]:,.0f} -> {serie[mes]:,.0f}")
    graficar_serie(serie, cruda)

    titulo("Líneas base")
    filas = [fila("ingenuo", serie, ingenuo),
             fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    por_origen, _ = backtest(serie, ingenuo)
    print(f"  backtest: {len(por_origen)} orígenes, de {por_origen['origen'].iloc[0]:%Y-%m} "
          f"a {por_origen['origen'].iloc[-1]:%Y-%m}; prueba de "
          f"{serie.index[-24]:%Y-%m} a {serie.index[-1]:%Y-%m}")
    print(pd.DataFrame(filas).drop(columns="decision").to_string(index=False))

    titulo("2. Descomposición")
    try:
        descomponer(serie)
    except NotImplementedError as error:
        pendiente("parte 2", error)

    titulo("3. Estacionariedad e identificación")
    try:
        d, D = estacionariedad(serie)
    except NotImplementedError as error:
        pendiente("parte 3", error)
        d = D = None

    if d is not None:
        try:
            titulo("4. Modelos clásicos")
            orden, estacional = elegir_sarima(serie, d, D)
            sarima = modelo_sarima(orden, estacional)
            filas.append(fila(f"SARIMA{orden}{estacional}", serie, sarima))

            print("  exógenas candidatas (SARIMAX sobre toda la serie):")
            comparar_exogenas(serie, orden, estacional)
            x = exogenas(serie)
            print("  días hábiles de los 12 meses siguientes:",
                  x.iloc[-12:, 0].astype(int).tolist())
            sarimax = modelo_sarimax(orden, estacional, x)
            filas.append(fila(f"SARIMAX{orden}{estacional} + exógenas", serie, sarimax))

            # El experimento de la parte 1: el mismo SARIMA sobre la serie sin
            # corregir. Los meses de prueba son iguales en las dos series, así
            # que las métricas se comparan.
            sin_corregir = fila("SARIMA, serie sin corregir", cruda.interpolate(), sarima)
            print("  experimento, SARIMA sin corregir los choques:",
                  {k: sin_corregir[k] for k in ("RMSE", "MAE", "MAPE")})
        except NotImplementedError as error:
            pendiente("parte 4", error)

    titulo("5. La tabla de métricas")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    mejor = tabla.iloc[0]
    print("Mejor modelo por RMSE:", mejor["modelo"])

    base = tabla[tabla["modelo"].isin(["ingenuo", "estacional ingenuo"])].iloc[0]
    for metrica in ("RMSE", "MAE", "MAPE"):
        print(f"  {metrica}: {mejor[metrica]:,.2f} frente a {base[metrica]:,.2f} del "
              f"{base['modelo']} ({(mejor[metrica] / base[metrica] - 1) * 100:+.1f} %)")
    if d is not None and mejor["modelo"].startswith("SARIMAX"):
        analizar_mejor(serie, mejor["modelo"], sarimax)
    elif d is not None and mejor["modelo"].startswith("SARIMA"):
        analizar_mejor(serie, mejor["modelo"], sarima)


if __name__ == "__main__":
    main()
