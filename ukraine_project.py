import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv("GDP_ANNUAL_GROWTH.csv", skiprows=4)


years = ['2018','2019', '2020', '2021']

#subset with all country GDP data for above years
df_subset = df[['Country Name'] + years]

#Ukraine GDP data right before the war
ukraine_gdp = df_subset[df_subset['Country Name'] == 'Ukraine'][years].values.flatten()

#find most similar country to Ukraine based on GDP
''' do 5'''
val = float('inf')
sim_country = "Ukraine"
for index, row in df_subset.iterrows():
    country_name = row['Country Name']
    if country_name != 'Ukraine':
        country_gdp = row[years].values.flatten()
        distance = euclidean(ukraine_gdp, country_gdp)
        if distance < val:
            sim_gdp = country_gdp
            sim_country = country_name
            val = distance

#plotting Ukraine GDP vs. most similar country
plt.plot(years, ukraine_gdp, label='Ukraine')
plt.plot(years, sim_gdp, label=sim_country)
print(f"Most similar country to Ukraine: {sim_country}")

# Preparing time series data for ARIMA
# Set the 'years' as the index for time series modeling
ukraine_data = df_subset[df_subset['Country Name'] == 'Ukraine'][years].T #makes years become rows
ukraine_data.columns = ['GDP'] #sets gdp as column
ukraine_data.index = ukraine_data.index.astype(int)  #makes sure years is an int
'''feed time series all years'''
#same process for sim country
sim_data = df_subset[df_subset['Country Name'] == sim_country][years].T
sim_data.columns = ['GDP']
sim_data.index = sim_data.index.astype(int)

#fits arima model for Ukraine
ukraine_series = ukraine_data.squeeze()  #df to series
model_ukraine = ARIMA(ukraine_series, order=(1, 1, 1))
model_ukraine_fit = model_ukraine.fit()

#fits arima model for sim country
sim_series = sim_data.squeeze()  #df to series
model_sim = ARIMA(sim_series, order=(1, 1, 1))
model_sim_fit = model_sim.fit()

#forecasts next 3 years
forecast_ukraine = model_ukraine_fit.forecast(steps=3)
forecast_sim = model_sim_fit.forecast(steps=3)

#creates a df to shjow forecast results
forecast_years = [2022, 2023, 2024]
forecast_df = pd.DataFrame({
    'Year': forecast_years,
    'Ukraine Forecast': forecast_ukraine,
    f'{sim_country} Forecast': forecast_sim
})

print(forecast_df)

#plot  historical data (2018-2021) + forecasted data (2022-2024)
plt.figure(figsize=(12, 6))


plt.plot(ukraine_series.index, ukraine_series.values, marker='o', label='Ukraine (Actual)')
plt.plot(sim_series.index, sim_series.values, marker='o', label=f'{sim_country} (Actual)')


plt.plot(forecast_years, forecast_ukraine, marker='o', linestyle='--', label='Ukraine (Forecast)')
plt.plot(forecast_years, forecast_sim, marker='o', linestyle='--', label=f'{sim_country} (Forecast)')


plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Forecast: Ukraine vs. Most Similar Country')
plt.legend()
plt.grid(True)
plt.show()


        
        
        
        
        
        
        
        
        