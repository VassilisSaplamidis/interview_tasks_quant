import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_raw.csv', delimiter=';')
data.set_index('datetime_utc_from', inplace=True)
data.index = pd.to_datetime(data.index, utc=True)
data['datetime_local_from'] = data.index.tz_convert('Europe/Zurich')

# Sample DataFrame
np.random.seed(42)  # For reproducibility

# Function to calculate mp
def calculate_mp_pos(row):
    row['global_radiation_J'] = row['global_radiation_J']/1000000
    hour = row.name.hour
    weekday = row.name.weekday()  # Monday is 0 and Sunday is 6
    is_working_day = weekday < 5

    a = np.random.uniform(1.15, 2) + (hour%4)*0.1  # Randomness + dependency on hour
    if row['global_radiation_J']>1:
        b = np.random.normal(2, 3) + 0.005 * (row['global_radiation_J'] ** 2)  # Randomness + dependency on solar squared
    else:
        b = np.random.normal(1.4, 2) + np.random.uniform(-5,5)
    c = np.random.uniform(-0.2, 0.2) + hour * 0.01  # Randomness + dependency on hour
    mp_pos = a * row['spot_ch_eurpmwh'] + b * row['global_radiation_J'] + c

    spike_probability = 0.001
    if 10 <= hour <= 14 and is_working_day and row['spot_ch_eurpmwh'] > 150:
        spike_probability = 0.15
    
    if np.random.rand() < spike_probability:
        spike = np.random.uniform(5, 20)  # Random spike value
        mp_pos = np.min([spike*row['spot_ch_eurpmwh'], 14900])
    return mp_pos

def calculate_mp_neg(row):
    row['global_radiation_J'] = row['global_radiation_J']/1000000
    hour = row.name.hour
    weekday = row.name.weekday()  # Monday is 0 and Sunday is 6
    is_working_day = weekday < 5

    a = np.random.uniform(0.2, 0.8) - (hour%4)*0.5  # Randomness + dependency on hour
    if row['global_radiation_J']>1:
        b = np.random.normal(1, 3) - 0.05 * (row['global_radiation_J'] ** 2)  # Randomness + dependency on solar squared
    else:
        b = np.random.normal(1.4, 2) + np.random.uniform(-5,5)
    c = np.random.uniform(-5.2, 0.2) + hour * 0.01  # Randomness + dependency on hour
    mp_neg = a * row['spot_ch_eurpmwh'] + b * row['global_radiation_J'] + c
    
    spike_probability = 0.005
    if ((11 <= hour <= 13) or (7 > hour) or (hour > 22)) and not(is_working_day) and row['spot_ch_eurpmwh'] < 20:
        spike_probability = 0.06
    
    if np.random.rand() < spike_probability:
        spike = -np.random.uniform(2, 10)  # Random spike value
        mp_neg = np.max([spike*row['spot_ch_eurpmwh'],-14800])
    return mp_neg

# Apply function to DataFrame
data['activation_price_pos_eurpmwh'] = data.apply(calculate_mp_pos, axis=1)
#data['activation_price_neg_eurpmwh'] = data.apply(calculate_mp_neg, axis=1)

data.drop(columns=['consumption_MWh','temperature_celsius','datetime_local_from'], inplace=True)

# Save DataFrame to CSV
data.to_csv('data_raw_classification.csv')

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['spot_ch_eurpmwh'], label='Spot', color='blue')
plt.plot(data.index, data['activation_price_pos_eurpmwh'], label='MP', color='orange')
plt.plot(data.index, data['activation_price_neg_eurpmwh'], label='MP', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Spot and MP Prices Over Time')
plt.legend()
plt.grid(True)
plt.show()