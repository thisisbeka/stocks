import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

st.title("Оптимизация Портфеля и Прогнозирование Финансовых Данных")

# Шаг 1: Выбор финансового инструмента
instrument_type = st.selectbox(
    "Выберите тип финансового инструмента",
    ("Акции", "Облигации", "Фьючерсы", "Опционы")
)

if instrument_type == "Акции":
    tickers = st.text_input("Введите тикеры акций через запятую (например, AAPL, MSFT, GOOG)", "AAPL, MSFT, GOOG")
elif instrument_type == "Облигации":
    tickers = st.text_input("Введите тикеры облигаций через запятую (например, TLT, SHY, IEF)", "TLT, SHY, IEF")
elif instrument_type == "Фьючерсы":
    tickers = st.text_input("Введите тикеры фьючерсов через запятую (например, ES=F, NQ=F, YM=F)", "ES=F, NQ=F, YM=F")
else:
    tickers = st.text_input("Введите тикеры опционов через запятую (например, AAPL230318C00145000, MSFT230318P00260000)", "AAPL230318C00145000, MSFT230318P00260000")

tickers = [t.strip() for t in tickers.split(',')]
start_date = st.date_input("Дата начала", pd.to_datetime('2020-01-01'))
end_date = st.date_input("Дата окончания", pd.to_datetime('today'))

# Шаг 2: Выбор фильтров
sector = st.selectbox("Выберите сектор", ["Технологии", "Здравоохранение", "Финансы", "Энергетика", "Коммунальные услуги", "Потребительские товары"])
country = st.selectbox("Выберите страну", ["США", "Канада", "Великобритания", "Германия", "Япония"])
initial_capital = st.number_input("Начальный капитал", min_value=1000, value=10000, step=1000)
commission = st.number_input("Комиссия за сделку ($)", min_value=0.0, value=0.0, step=0.1)

# Шаг 3: Загрузка данных с Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

st.write("Исходные данные")
st.dataframe(data.head())

# Шаг 4: Реализация скользящего среднего
sma_window = st.slider("Выберите окно скользящего среднего", 5, 50, 20)
sma = data.rolling(window=sma_window).mean()
st.line_chart(sma)

# Шаг 5: Реализация Множественной Линейной Регрессии
st.write("Множественная Линейная Регрессия")
X = np.arange(len(data)).reshape(-1, 1)
y = data.values
model = LinearRegression().fit(X, y)
predictions = model.predict(X)
st.line_chart(predictions)

# Шаг 6: Оптимизация Портфеля (пример)
st.write("Оптимизация Портфеля")

# Расчет ожидаемой доходности и ковариационной матрицы
returns = data.pct_change().mean() * 252
cov_matrix = data.pct_change().cov() * 252

# Определение целевой функции (минимизация дисперсии портфеля)
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Ограничения и границы
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Начальное предположение (равномерное распределение)
initial_weights = np.array(len(tickers) * [1. / len(tickers)])

# Оптимизация портфеля
optimized_result = minimize(portfolio_variance, initial_weights, args=cov_matrix, method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized_result.x

st.write("Оптимальные Веса Портфеля:")
for ticker, weight in zip(tickers, optimized_weights):
    st.write(f"{ticker}: {weight:.2%}")

portfolio_return = np.sum(returns * optimized_weights)
portfolio_variance = portfolio_variance(optimized_weights, cov_matrix)
portfolio_sharpe_ratio = portfolio_return / np.sqrt(portfolio_variance)

st.write(f"Ожидаемая Доходность Портфеля: {portfolio_return:.2%}")
st.write(f"Дисперсия Портфеля: {portfolio_variance:.4f}")
st.write(f"Коэффициент Шарпа Портфеля: {portfolio_sharpe_ratio:.4f}")
