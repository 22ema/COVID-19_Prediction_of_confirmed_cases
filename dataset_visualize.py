import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
from pylab import rcParams

if __name__ == "__main__":
    covid_19_path = "./dataset/covid_19_data.csv"
    time_series = "./dataset/time_series_covid19_confirmed_global.csv"
    all = pd.read_csv(covid_19_path)
    confirmed = pd.read_csv(time_series)

    # 전세계 데이터 EDA
    group = all.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum()
    # iloc은 특정 행열을 추출하기 위해 사용 T는 행과 열을 switching 해주기 위해 사용
    korea = confirmed[confirmed['Country/Region'] == 'Korea, South'].iloc[:,4:].T
    korea.index = pd.to_datetime(korea.index)
    rcParams['figure.figsize'] = 12, 8
    sns.set(style ='whitegrid', palette='muted', font_scale=1.2)
    # diff()를 하게 되면 2020-01-22 날짜(첫번째 행)의 일단위 확진자를 알 수 없다. 그러므로 첫째날의 누적확진자수를 그대로 가져온다.
    daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
    print(daily_cases)
    plt.plot(daily_cases)
    plt.show()
