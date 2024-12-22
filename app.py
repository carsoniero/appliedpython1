import pandas as pd
import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Анализ данных с использованием Streamlit")
st.write("Интерактивное приложение для анализа данных")

uploadfile = st.file_uploader("Выберите CSV файл", type=['csv'])

def get_weather(city, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(url)
    if response.status_code == 200:  # Проверяем статус ответа
        data = response.json()  # Преобразуем ответ в JSON
        return data['main']['temp']  # Возвращаем температуру
    else:
        return None  # Возвращаем None, если произошла ошибка
def compute_stats(data):
    """Вычисление статистики для части DataFrame."""
    data['rolling_mean'] = data.groupby(['city', 'season'])['temperature'].transform(lambda x: x.rolling(window=30).mean())
    result = data.groupby(['city', 'season']).agg(
        average_temperature=('temperature', 'mean'),
        std_deviation=('temperature', 'std')
    ).reset_index()
    data = data.merge(result)
    data['lower_bound'] = data['average_temperature'] - 2 * data['std_deviation']
    data['upper_bound'] = data['average_temperature'] + 2 * data['std_deviation']

    # Выявляем аномалии
    anomalies = (data['temperature'] < data['lower_bound']) | (data['temperature'] > data['upper_bound'])
    return anomalies
if uploadfile is not None:
    data = pd.read_csv(uploadfile)
    st.dataframe(data.head())
else:
    st.write("Пожалуйста, загрузить csv-файл")

if uploadfile is not None:

    # Проверка наличия столбца с названиями городов
    if 'city' in data.columns:  # Замените 'Город' на название соответствующего столбца в ваших данных
        # Создание выпадающего списка для выбора города
        selected_city = st.selectbox("Выберите город", data['city'].unique())

        # Фильтрация данных по выбранному городу
        city_data = data[data['city'] == selected_city]

        # Отображение информации о выбранном городе
        st.write(f"Информация о городе: {selected_city}")
        st.dataframe(city_data)

        if not city_data.empty:
            # Описательная статистика
            st.subheader(f'Описательная статистика для города {selected_city}')
            st.write(city_data[['temperature', 'timestamp', 'season']].describe())

        else:
            st.write(f'Нет данных для города {selected_city}.')

    # Проверка наличия необходимых столбцов
    if 'temperature' in data.columns and 'timestamp' in data.columns:
        # Временной ряд температур
        st.subheader("Временной ряд температур")
        anomalies = compute_stats(data)

        plt.figure(figsize=(14, 7))
        plt.plot(data['timestamp'], data['temperature'], label='Температура', color='blue')
        plt.scatter(data['timestamp'][anomalies], data['temperature'][anomalies], color='red', label='Аномалии', s=30)
        plt.title('Временной ряд температур с аномалиями')
        plt.xlabel('Дата')
        plt.ylabel('Температура (°C)')
        plt.legend()
        st.pyplot(plt)

        # Сезонные профили
        st.subheader("Сезонный профиль температуры")
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data['month'] = data['timestamp'].dt.month
        seasonal_stats = data.groupby('month')['temperature'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=seasonal_stats, x='month', y='mean', marker='o', label='Среднее')
        plt.fill_between(seasonal_stats['month'],
                         seasonal_stats['mean'] - seasonal_stats['std'],
                         seasonal_stats['mean'] + seasonal_stats['std'],
                         color='lightblue', alpha=0.5, label='Стандартное отклонение')
        plt.title('Сезонный профиль температуры')
        plt.xlabel('Месяц')
        plt.ylabel('Температура (°C)')
        plt.xticks(ticks=np.arange(1, 13),
                   labels=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
        plt.legend()
        st.pyplot(plt)

        api = st.text_input('Введите API ключ от OpenWeatherMap')

        if st.button('Погода на данный момент в выбранном городе'):
            temperature = get_weather(selected_city, api)
            if temperature is not None:
                st.write(f'Температура в {selected_city}: {temperature}°C')
            else:
                st.write('Не удалось получить данные. Проверьте название города и API ключ.')

    else:
        st.error("В загруженных данных нет столбца 'city'. Пожалуйста, проверьте файл.")
else:
    st.info("Пожалуйста, загрузите файл с данными о городах.")


#import requests
#import time

#def sync_get_weather(city):
    #url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=52161177a6b184408df88a5098b19de4"
    #response = requests.get(url)
    #data = response.json()
    #if response.status_code == 200:
        #return data['main']['temp']
    #else:
        #return None

#if __name__ == "__main__":
    #cities = ["London", "Moscow", "New York", "Tokyo", "Berlin"]
    #start_time = time.time()

    #for city in cities:
        #temp = sync_get_weather(city)
        #print(f"Current temperature in {city}: {temp}°C")

    #print("Synchronous request took: ", time.time() - start_time, "seconds")'''

#Current temperature in London: 285.33°C
#Current temperature in Moscow: 272.39°C
#Current temperature in New York: 271.54°C
#Current temperature in Tokyo: 282.94°C
#Current temperature in Berlin: 279.37°C
#Synchronous request took:  0.11623477935791016 seconds'''

#'''import aiohttp
#import asyncio

#async def async_get_weather(session, city):
#  url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=52161177a6b184408df88a5098b19de4"
#  async with session.get(url) as response:
#    data = await response.json()
#    if response.status == 200:
#      return data['main']['temp']
#    else:
#      return None

#async def main():
#  cities = ["London", "Moscow", "New York", "Tokyo", "Berlin"]
#  async with aiohttp.ClientSession() as session:
#    tasks = [async_get_weather(session, city) for city in cities]
#    results = await asyncio.gather(*tasks)
#    for city, temp in zip(cities, results):
#      print(f"Current temperature in {city}: {temp}°C")

#if __name__ == "__main__":
#  start_time = time.time()
#  await main()
#  print("Asynchronous request took: ", time.time() - start_time, "seconds")'''

#Current temperature in London: 285.13°C
#Current temperature in Moscow: 270.39°C
#Current temperature in New York: 271.66°C
#Current temperature in Tokyo: 282.09°C
#Current temperature in Berlin: 279.33°C
#Asynchronous request took:  0.03430795669555664 seconds'''

#"""import time

#start_time = time.time()
#code = compute_stats(data)
#end_time = time.time()
#print("Время выполнения без распараллеливания: {:.2f} секунд".format(end_time - start_time))"""

#'''Время выполнения без распараллеливания: 0.08 секунд'''

#'''
#from concurrent.futures import ProcessPoolExecutor
#start_time_parallel = time.time()
#with ProcessPoolExecutor() as executor:
#    results = executor.map(compute_stats, df_chunks)
#stats_parallel = pd.concat(results)
#print(c)
#end_time_parallel = time.time()
#print("Время выполнения с распараллеливанием: {:.2f} секунд".format(end_time_parallel - start_time_parallel))
#'''

#'Время выполнения с распараллеливанием: 0.19 секунд'
