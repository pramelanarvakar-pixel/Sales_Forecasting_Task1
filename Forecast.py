import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Historical Sales Data (32 months)
sales = [52,55,64,63,58,53,47,60,79,76,73,74,65,69,75,86,89,95,91,88,86,88,89,104,106,109,112,107,94,99,106,108]
months = list(range(1, 33))

df = pd.DataFrame({'Month': months, 'Sales': sales})

# Model Training
model = LinearRegression()
model.fit(df[['Month']], df['Sales'])

# Future Forecast (Next 6 Months)
future_months = list(range(33, 39))
future_df = pd.DataFrame({'Month': future_months})
forecast = model.predict(future_df[['Month']])

print("Next 6 Months Forecast:")
for m, f in zip(future_months, forecast):
    print(f"Month {m}: {int(f)}k")

# Graph
plt.plot(df['Month'], df['Sales'], label='Historical')
plt.plot(future_df['Month'], forecast, label='Forecast', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Sales (k)')
plt.title('Sales & Demand Forecasting')
plt.legend()
plt.savefig('forecast_graph.png')
print("Graph saved!")
