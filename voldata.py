import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def calculate_garman_klass_volatility(open_price, close_price, high_price, low_price, num_trading_days):

    try:
        ln_ratio_open_close = math.log(close_price / open_price)
        ln_ratio_high_low = math.log(high_price / low_price)
        
        first_component = (0.5 * ln_ratio_open_close) ** 2
        second_component = (2 * ln_ratio_high_low - ln_ratio_open_close) ** 2
        
        average_daily_volatility = (first_component - second_component) / (num_trading_days * math.log(2))
        #print(f"average_daily_volatility: {average_daily_volatility}")
        realized_volatility = math.sqrt(abs(average_daily_volatility))
        
        return realized_volatility
    except:
        pass

df = pd.read_csv("optionsdata.csv")
#print(df.head())
#print(df.info())
#print(df.iloc[1])
#print(df.get(["OPEN","CLOSE","HIGH","LOW"]))
test = pd.DataFrame(df.get(["OPEN","CLOSE","HIGH","LOW"]))
l = test.values.tolist()
#l = test.iloc[1:3].values.tolist()
#print(l)
gks = []

for i in l:
    #print(f"i: {i}")
    garm = calculate_garman_klass_volatility(i[0],i[1],i[2],i[3],20)
    if garm != None:
        gks.append(garm)
    #print("Garman-Klass Vol: ", garm)



#time = np.arange(0, len(gks)*200, 200)

# Calculate the average of every 200 values
agks = [np.mean(gks[i:i+110]) for i in range(0, len(gks), 110)]


# Plotting the data
plt.plot(agks)

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Data')
plt.title('Data vs. Time')

# Display the plot
plt.show()


#Fourier Analysis on the signal

from numpy.fft import rfft, irfft, rfftfreq
from scipy import pi, signal, fftpack


gks_ft = np.abs(rfft(agks))

gks_freq = rfftfreq(len(agks))

plt.plot(gks_freq, gks_ft)
   
plt.xlabel('frequency (1/day)')
plt.show()


# Apply FFT
fft_values = np.fft.fft(agks)

# Calculate magnitude spectrum
magnitude_spectrum = np.abs(fft_values)

# Generate frequency axis
N = len(agks)
T = 1  # Assuming unit time interval between data points
frequencies = np.fft.fftfreq(N, T)



# Define the threshold value
threshold = 30  # Adjust as needed

# Filter out high-frequency components
fft_values_filtered = fft_values.copy()
fft_values_filtered[magnitude_spectrum < threshold] = 0
denoised_data = np.fft.ifft(fft_values_filtered).real
plt.plot(range(N), agks, label='Original Data')
plt.plot(range(N), denoised_data, label='Denoised Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Data vs Denoised Data')
plt.legend()







#ARIMA
#
#import pandas as pd
#import matplotlib.pyplot as plt
#from statsmodels.tsa.arima.model import ARIMA
#
## Example time series data
#
#data = agks
## Create the ARIMA model
#model = ARIMA(agks, order=(1, 0, 10))
#
## Fit the ARIMA model
#model_fit = model.fit()
#
## Generate forecasts
#forecast_values = model_fit.forecast(steps=5)
#
#
## Plot the original data and the forecasts
#plt.plot(range(len(data)), data, label='Original Data')
#plt.plot(range(len(data), len(data) + len(forecast_values)), forecast_values, label='Forecasted Data')
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.title('ARIMA Forecast')
#plt.legend()
#plt.show()
