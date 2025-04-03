import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew ,norm,t
import yfinance as yf

#FUNCIONES 
# Funciones
def obtener_datos(stocks):
    '''
    El objetivo de esta funcion es descargar el precio
    de cierre de un o varios activos en una ventana de un año

    Input = Ticker del activo en string 
    Output = DataFrame del precio del activo
    '''
    df = yf.download(stocks, period="1y")['Close']
    return df

def calcular_rendimientos(df):
    '''
    Funcion de calcula los rendimientos de un activo

    Input = Data Frame de precios por activo

    Output = Data Frame de rendimientos
    '''
    return df.pct_change().dropna()

def calcular_rendimientos_log(df):
    '''
    Funcion que calcula los rendimientos logarítmicos de un activo.

    Input = Data Frame de precios por activo

    Output = Data Frame de rendimientos logarítmicos
    '''
    return np.log(df / df.shift(1)).dropna()
def calcular_rendimiento_ventana(returns, window):
    '''
    Calcula el rendimiento acumulado en una ventana de tiempo específica.

    Input:
        returns (Series): Rendimientos diarios de un activo.
        window (int): Número de días a considerar.
    
    Output:
        Rendimiento acumulado en la ventana (float) o NaN si la ventana es muy grande.
    '''
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

def calcular_rendimiento_ventana_log(returns, window):
    """
    Calcula el rendimiento acumulado en una ventana de tiempo usando rendimientos logarítmicos.

    Input:
        returns (Series): Rendimientos logarítmicos diarios de un activo.
        window (int): Número de días a considerar.

    Output:
        Rendimiento logarítmico acumulado en la ventana (float) o NaN si la ventana es mayor que la cantidad de datos disponibles.
    """
    if len(returns) < window:
        return np.nan
    # La suma de log returns es el rendimiento logarítmico acumulado
    log_return_acumulado = returns.iloc[-window:].sum()
    return log_return_acumulado

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_var(returns, alpha=0.95):
    """
    Calcula el VaR paramétrico basado en la distribución normal.
    """
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - alpha, mean, std_dev)
    return var

def calcular_cvar(returns, alpha=0.95):
    """
    Calcula el CVaR (Expected Shortfall) usando datos históricos.
    """
    var = np.percentile(returns, (1 - alpha) * 100)
    cvar = returns[returns <= var].mean()
    return cvar

def kupiec_test(violations, total_days, confidence_level):
    """
    Prueba de Kupiec para evaluar la precisión del VaR.
    
    :param violations: Número de veces que las pérdidas superaron el VaR.
    :param total_days: Total de días en la muestra.
    :param confidence_level: Nivel de confianza utilizado en el VaR.
    :return: Valor p de la prueba de Kupiec.
    """
    p_hat = violations / total_days
    p_var = 1 - confidence_level
    likelihood_ratio = -2 * np.log(((1 - p_var) ** (total_days - violations) * (p_var ** violations)) /
                                   ((1 - p_hat) ** (total_days - violations) * (p_hat ** violations)))
    p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
    return p_value
    
def christoffersen_test(violations_sequence):
    """
    Prueba de Christoffersen para evaluar la independencia de violaciones del VaR.
    
    :param violations_sequence: Lista de 0 (sin violación) y 1 (con violación).
    :return: Valor p de la prueba de Christoffersen.
    """
    n00 = sum(1 for i in range(1, len(violations_sequence)) if violations_sequence[i] == 0 and violations_sequence[i-1] == 0)
    n01 = sum(1 for i in range(1, len(violations_sequence)) if violations_sequence[i] == 1 and violations_sequence[i-1] == 0)
    n10 = sum(1 for i in range(1, len(violations_sequence)) if violations_sequence[i] == 0 and violations_sequence[i-1] == 1)
    n11 = sum(1 for i in range(1, len(violations_sequence)) if violations_sequence[i] == 1 and violations_sequence[i-1] == 1)

    p1 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p2 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    likelihood_ratio = -2 * np.log(((1 - p1) ** n00 * p1 ** n01 * (1 - p2) ** n10 * p2 ** n11) /
                                   (((n01 + n11) / (n00 + n01 + n10 + n11)) ** (n01 + n11) *
                                    ((n00 + n10) / (n00 + n01 + n10 + n11)) ** (n00 + n10)))
    p_value = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
    return p_value

def backtesting_var(returns, var_series, confidence_level=0.95):
    """
    Evalúa los métodos de VaR mediante backtesting con pruebas de Kupiec y Christoffersen.
    
    :param returns: Serie de rendimientos.
    :param var_series: Serie de valores de VaR estimados.
    :param confidence_level: Nivel de confianza usado en el VaR.
    """
    violations = returns < var_series
    num_violations = violations.sum()
    total_days = len(returns)
    violations_sequence = violations.astype(int).tolist()

    kupiec_p = kupiec_test(num_violations, total_days, confidence_level)
    christoffersen_p = christoffersen_test(violations_sequence)

    return {
        "Total Violations": num_violations,
        "Kupiec p-value": kupiec_p,
        "Christoffersen p-value": christoffersen_p
    }



##finfunciones
# Configuración de la página
    # Configuración de la página
st.set_page_config(page_title="Metricas de acciones", layout="wide")
# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs(["Metricas básicas y rendimientos", "Var & cVaR",  "Violaciones", "VaR Volatilidad Móvil"])
st.sidebar.title("Analizador de Métricas")
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de las acciones separados por comas (por ejemplo: AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT,AMZN,NVDA")
simbolos = [s.strip() for s in simbolos_input.split(',')]
# Selección del benchmark para Luis
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

#para las ventanas
# Selección de la ventana de tiempo


# Hoja 1
with tab1:
    st.header("Análisis del activo")
    selected_asset = st.selectbox(
        "Seleccione un activo para analizar:", 
        simbolos, 
        key="selected_asset"  # Almacenamos en session_state
    )
    
    # Obtener datos y calcular rendimientos solo para el activo seleccionado
    df_precios = obtener_datos([selected_asset])  # Se pasa la lista con un solo activo
    df_rendimientos = calcular_rendimientos(df_precios)
    df_rendimientos_log = calcular_rendimientos_log(df_precios)

    # Diccionario para almacenar los promedios de rendimiento
    promedios_rendi_diario = {stock: df_rendimientos[stock].mean() for stock in [selected_asset]}
    
    # Calcular el sesgo y la kurtosis para el activo seleccionado
    skew_rendi_diario = df_rendimientos[selected_asset].skew()  # Sesgo para el activo seleccionado
    kurtosis_rendi_diario = df_rendimientos[selected_asset].kurtosis()  # Kurtosis para el activo seleccionado
    
    # Crear columnas para mostrar métricas
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    # Mostrar métricas para el activo seleccionado
    if selected_asset:
        promedio_diario = promedios_rendi_diario[selected_asset]
        promedio_anualizado = (1 + promedio_diario) ** 252 - 1  # Convertir a rendimiento anualizado

        col1.metric("Rendimiento promedio diario", f"{promedio_diario:.5%}")
        col2.metric("Rendimiento anualizado", f"{promedio_anualizado:.2%}")
        col3.metric("Último precio en moneda de la acción correspondiente", f"${df_precios[selected_asset].iloc[-1]:.2f}")
        col4.metric(f"Sesgo de {selected_asset}", f"{skew_rendi_diario:.5f}")
        col5.metric(f"Kurtosis de {selected_asset}", f"{kurtosis_rendi_diario:.5f}")
    else:
        col1.metric("Rendimiento promedio diario", "N/A")
        col2.metric("Rendimiento anualizado", "N/A")
        col3.metric("Último precio", "N/A")
        col4.metric(f"Sesgo de {selected_asset}", "N/A")
        col5.metric(f"Kurtosis de {selected_asset}", "N/A")
    # Datos de países de inversión para ETFs
    etf_country_data = {
        'S&P 500': ['United States'],
        'Nasdaq': ['United States'],
        'Dow Jones': ['United States'],
        'Russell 2000': ['United States'],
        'ACWI': ['Global']
    }
    
    all_symbols = simbolos + [benchmark]
    df_stocks = obtener_datos(all_symbols)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    fig_asset = go.Figure()
    fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
    fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
    fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
    st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")
        
    # Mostrar los datos en la página para el activo seleccionado
    st.subheader("Últimos 5 Datos de Precios")
    st.dataframe(df_precios.tail(5))
    
    st.subheader("Últimos 5 Rendimientos")
    st.dataframe(df_rendimientos.tail(5))
    
    st.subheader("Últimos 5 Rendimientos Logarítmicos")
    st.dataframe(df_rendimientos_log.tail(5))

with tab2:
    with st.sidebar:
        st.header("Cálculo de VaR y CVaR")

        selected_asset = st.selectbox(
            "Seleccione el activo (debe ser el mismo que en Análisis del activo)", 
            simbolos, 
            index=simbolos.index(st.session_state["selected_asset"]),
            key="selected_asset_2"
        )

    # Mostrar sobre qué activo se está realizando el cálculo
    st.subheader(f"Análisis de VaR y CVaR para el activo: {selected_asset}")  

    # Selector de nivel de confianza
    alpha_options = {
        "95%": 0.95,
        "97.5%": 0.975,
        "99%": 0.99
    }
    selected_alpha_label = st.selectbox("Seleccione un nivel de confianza:", list(alpha_options.keys()))
    alpha = alpha_options[selected_alpha_label]
    
    # El percentil para el cálculo del VaR
    percentil = alpha * 100  

    if selected_asset:
        mean = np.mean(df_rendimientos[selected_asset])
        stdev = np.std(df_rendimientos[selected_asset])
        
        # Paramétrico (Normal) VaR
        VaR_param = -norm.ppf(alpha, mean, stdev) 
        
        # Historical VaR
        hVaR = -df_rendimientos[selected_asset].quantile(1 - alpha)  
        
        # Monte Carlo VaR
        n_sims = 100000
        np.random.seed(42)  # Para reproducibilidad
        sim_returns = np.random.normal(mean, stdev, n_sims)
        MCVaR = -np.percentile(sim_returns, 100 - percentil)  
        
        # CVaR (Expected Shortfall)
        CVaR = -df_rendimientos[selected_asset][df_rendimientos[selected_asset] <= -hVaR].mean()  
        
        # Mostrar métricas en Streamlit
        st.subheader("Métricas de riesgo")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{int(alpha*100)}% VaR (Paramétrico)", f"{VaR_param:.4%}")
        col2.metric(f"{int(alpha*100)}% VaR (Histórico)", f"{hVaR:.4%}")
        col3.metric(f"{int(alpha*100)}% VaR (Monte Carlo)", f"{MCVaR:.4%}")
        col4.metric(f"{int(alpha*100)}% CVaR", f"{CVaR:.4%}")
        
        # Verificación de cálculos
        st.write("### Verificación de Cálculos:")
        st.write(f"VaR Paramétrico: {VaR_param:.4%}")
        st.write(f"VaR Histórico: {hVaR:.4%}")
        st.write(f"VaR Monte Carlo: {MCVaR:.4%}")
        st.write(f"CVaR: {CVaR:.4%}")
        
        # Visualización gráfica
        st.subheader("Gráfica métricas de riesgo")
        
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(13, 5))
        
        # Generar histograma
        n, bins, patches = ax.hist(df_rendimientos[selected_asset], bins=50, color='blue', alpha=0.7, label='Returns')
        
        # Identificar y colorear de rojo las barras a la izquierda de -hVaR
        for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
            if bin_left < -hVaR:
                patch.set_facecolor('red')
        
        # Marcar las líneas de VaR y CVaR
        ax.axvline(x=-VaR_param, color='skyblue', linestyle='--', label=f'VaR {int(alpha*100)}% (Paramétrico)')
        ax.axvline(x=-MCVaR, color='grey', linestyle='--', label=f'VaR {int(alpha*100)}% (Monte Carlo)')
        ax.axvline(x=-hVaR, color='green', linestyle='--', label=f'VaR {int(alpha*100)}% (Histórico)')
        ax.axvline(x=-CVaR, color='purple', linestyle='-.', label=f'CVaR {int(alpha*100)}%')
        
        # Configurar etiquetas y leyenda
        ax.set_title(f"Histograma de Rendimientos con VaR y CVaR para {selected_asset}")
        ax.set_xlabel("Rendimiento Diario")
        ax.set_ylabel("Frecuencia")
        ax.legend()

        st.pyplot(fig)
        
        # explicación básica
        with st.expander("¿Qué significan estas métricas?"):
            st.write(f"""
            - **VaR {int(alpha*100)}%**: Con un nivel de confianza del {int(alpha*100)}%, se espera que la pérdida máxima diaria no exceda este valor.
            - **CVaR {int(alpha*100)}%**: Si se excede el VaR, esta es la pérdida promedio esperada.
            - **Métodos**:
                - Paramétrico: Asume distribución normal
                - Histórico: Basado en datos históricos reales
                - Monte Carlo: Simulación de {n_sims:,} escenarios
            """)

    else:
        st.write("Seleccione un activo para visualizar sus métricas de riesgo.")
        st.write("Seleccione un activo para visualizar sus métricas de riesgo.")


# Evaluación de eficiencia de VaR (Inciso d)
with tab3:
    st.header("Evaluación de eficiencia de VaR")
    alpha = st.selectbox("Seleccione nivel de confianza", [0.95, 0.975, 0.99], index=0)
    selected_asset = st.selectbox("Seleccione un activo", simbolos)
    df_rendimientos = calcular_rendimientos(df_precios[selected_asset])

    VaR = calcular_var(df_rendimientos, alpha)
    violaciones = df_rendimientos[df_rendimientos < VaR]
    porcentaje_violaciones = len(violaciones) / len(df_rendimientos) * 100

    st.metric("VaR estimado", f"{VaR:.4%}")
    st.metric("Número de violaciones", f"{len(violaciones)}")
    st.metric("Porcentaje de violaciones", f"{porcentaje_violaciones:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_rendimientos.index, df_rendimientos, label="Rendimientos", color='blue')
    ax.axhline(y=VaR, color='red', linestyle='--', label=f'VaR {int(alpha*100)}%')
    ax.scatter(violaciones.index, violaciones, color='red', label='Violaciones', zorder=3)
    ax.legend()
    st.pyplot(fig)

# Rolling Windows para VaR y CVaR (Inciso e)
with tab4:
    st.header("VaR y CVaR con Ventana Móvil")
    window_size = st.slider("Seleccione tamaño de la ventana", 20, 252, 60)
    df_rolling_var = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_var(x, alpha), raw=True)
    df_rolling_cvar = df_rendimientos.rolling(window=window_size).apply(lambda x: calcular_cvar(x, alpha), raw=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_rolling_var, label=f"Rolling VaR {int(alpha*100)}%", color='red')
    ax.plot(df_rolling_cvar, label=f"Rolling CVaR {int(alpha*100)}%", color='purple')
    ax.legend()
    st.pyplot(fig)



