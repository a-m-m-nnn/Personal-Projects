#!/usr/bin/env python
# coding: utf-8

# ## Python for Finance - TSLA Stock Analysis Project - A.M.M. ICL ICBS ESB '24
# 
# ###### ** Markdown cells contain subtitles of processes and sequenced steps underlying reasoning, # comments for code elaboration**
# 
# ## We begin by importing our prerequisites:
# 

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Data Cleaning
# 
# ### We load the raw data:

# In[82]:


###Save included data to own desired location
df = pd.read_csv('C:/Users/amjmo/Downloads/TSLA_10YR_raw_(2).csv')


# ### We try shift 'date' to "datetime" target format

# In[83]:


df['Date'] = pd.to_datetime(df['Date'])


# In[84]:


df.set_index('Date', inplace=True)


# ### Set the "Date" column index

# In[85]:


df.sort_index(inplace=True)


# ### Remove duplicates in data

# In[86]:


df.duplicated().sum()
df.drop_duplicates(inplace=True)


# ### Check the first and last few lines of the data  

# In[87]:


print(df.head())


# In[88]:


print(df.tail())


# ### Check the data overview

# In[89]:


print(df.info())


# ### Fill in the missing 'Open' price to complete Rule 1, and use the forward fill mode of the 'ffill' method:

# In[90]:


df['Open'] = df['Open'].ffill()


# ### This step fills in the missing “Close” price and uses the backfill mode of the “bfill” method:

# In[91]:


df['Close'] = df['Close'].bfill()


# ### This step fills in the missing values for the “Adj Close” column as Rule 3

# In[92]:


df['Adj Close'] = df['Adj Close'].fillna(df['Close'])


# ### This step uses “interpolate” for missing values

# In[93]:


df['High'] = df['High'].interpolate(method='linear')
df['Low'] = df['Low'].interpolate(method='linear')


# ### The missing values in the 'Volume' column are filled with the 'Close' price and the 'Open' price:

# In[94]:


close_equals_open = df['Close'] == df['Open']
median_volume = df['Volume'].median()


# ### "loc" is used to locate the position where the "Close" price equals the "Open" price and fill it in with "0"

# In[95]:


df.loc[close_equals_open, 'Volume'] = df.loc[close_equals_open, 'Volume'].fillna(0)


# In[96]:


df['Volume'] = df['Volume'].fillna(median_volume)


# ### Verify the results of data cleaning

# In[97]:


print(df.info())


# ### Extract data from 2019

# In[98]:


df = df.loc['2019-01-01':]


# ### Verify the results of data cleaning

# In[99]:


print(df.head())


# ### Save the cleaned data into new file

# In[100]:


###Save cleaned data to own desired location
df.to_csv('C:/Users/amjmo/Downloads/TSLA_10YR_clean_(11).csv')


# # Feature Engineering
# 
# ### We read in our cleaned data and summarily check for any error

# In[101]:


df = pd.read_csv('C:/Users/amjmo/Downloads/TSLA_10YR_clean_(11).csv')
df.info()


# ### "Boiler plate" date indexing and addition of 'Year' column 

# In[102]:


df['Date'] = pd.to_datetime(df['Date'])
#Year added now for later Key Dates operations
df['Year'] = df['Date'].dt.year
df = df.set_index('Date').sort_index().drop_duplicates()
df


# ### Daily Returns:

# In[103]:


returns = df['Daily Return'] = df['Adj Close'].pct_change()
returns


# ### Log Returns:

# In[104]:


#"Prev Close" column created to enable log return calculation
df['Prev Close'] = df['Adj Close'].shift(1)
log_returns = np.log(df['Adj Close'] / df['Prev Close'])
log_returns


# ### Surges
# 
# #### 1. We define mean return
# 
# #### 2. We define the surge threshold as 4 standard deviations above the mean
# 
# #### 3. We define the surge condition
# 
# #### 4. We subset the dataframe where daily returns are higher than the surge threshold

# In[105]:


mean_return = df['Daily Return'].mean()

return_threshold = mean_return + (df['Daily Return'].std() * 4)

condition = df['Daily Return'] > return_threshold

df[condition]


# ### Volume Spike
# 
# #### 1. We find the mean volume
# 
# #### 2. We define the spike threshold as 6 standard deviations above the mean
# 
# #### 3. We define the spike condition
# 
# #### 4. We subset the dataframe where volumes are higher than the spike threshold

# In[106]:


mean_volume = df['Volume'].mean()

return_threshold = mean_volume + (df['Volume'].std() * 6)

condition = df['Volume'] > return_threshold

df[condition]


# ### Simple Moving Average

# In[107]:


df['20-day MA'] = df['Adj Close'].rolling(window=20).mean()
df


# ### 20-Day Standard Deviation [Volatility]

# In[109]:


df['20-day SD'] = df['Adj Close'].rolling(window=20).std()
df


# ### Bollinger Bands
# 
# #### 1. We calculate the High and Low Bollinger Bands
# 
# #### 2. Creating plot with added SMA line
# 
# #### 3. We plot the High and Low Bollinger Bands
# 
# #### 4. Display

# In[110]:


df['Bollinger High'] = df['20-day MA'] + (2 * df['20-day SD'])
df['Bollinger Low'] = df['20-day MA']  - (2 * df['20-day SD'])

plt.figure(figsize=(10,5))
plt.plot(df['Adj Close'], label='Adj Close')
plt.plot(df["20-day MA"], label="20-day SMA")

plt.plot(df['Bollinger High'], label='Bollinger High', linestyle='--', color='grey')
plt.plot(df['Bollinger Low'], label='Bollinger Low', linestyle='--', color='grey')
plt.fill_between(df.index, df['Bollinger High'], df['Bollinger Low'], color='grey', alpha=0.1)
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')

plt.legend()
plt.show()


# # Key Dates
# 
# ### Date of Highest Volatility Day by Year:

# In[111]:


highest_volatility_day = df.loc[df.groupby('Year')['20-day SD'].idxmax(), ['Year', '20-day SD']]
highest_volatility_day


# ### Date of Largest Price Surge by Year:

# In[112]:


largest_price_surge_day = df.loc[df.groupby('Year')['Daily Return'].idxmax(), ['Year', 'Daily Return']]
largest_price_surge_day 


# ### Date of Largest Price Drop by Year:

# In[113]:


largest_price_drop_day = df.loc[df.groupby('Year')['Daily Return'].idxmin(), ['Year', 'Daily Return']]
largest_price_drop_day 


# ### Date of Highest Trading Volume by Year:

# In[114]:


highest_volume_day = df.loc[df.groupby('Year')['Volume'].idxmax(), ['Year', 'Volume']]
highest_volume_day 


# ### Date of Highest High by Year:

# In[115]:


highest_high = df.loc[df.groupby('Year')['High'].idxmax(), ['Year', 'High']]
highest_high


# ### Date of Lowest Low by Year:

# In[116]:


lowest_low = df.loc[df.groupby('Year')['Low'].idxmin(), ['Year', 'Low']]
lowest_low


# ### Open and Close Values by Year:

# In[117]:


annual_open_close = df.groupby('Year').agg({'Open': 'first', 'Close': 'last'})
annual_open_close


# # Data Visualisation
# 
# #### 1. Creating a main plot with Adj Close prices, Simple Moving Average, Bollinger Bands
# 
# ##### 1a. Adding subplots for Volume and Volatility 
# 
# #### 2. We plot the Adjusted Close Prices, SMA, Bollinger Bands on the main plot [top subplot]
# 
# #### 3. We plot Volume on the second subplot
# 
# #### 4. We plot Volatility on the third subplot
# 
# #### 5. Display with key/legend
# 

# In[118]:


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10, 5), gridspec_kw={'height_ratios': [5, 2, 2]})
fig.subplots_adjust(hspace=0.5)

ax1.plot(df.index, df['Adj Close'], label='Adj Close')
ax1.plot(df['20-day MA'], label= '20-day SMA')
ax1.plot(df['Bollinger High'], label='Bollinger High', linestyle='--', color='grey')
ax1.plot(df['Bollinger Low'], label='Bollinger Low', linestyle='--', color='grey')
ax1.set_title('Adjusted Close Price and Volume')
ax1.set_ylabel('Adjusted Close Price')
ax1.fill_between(df.index, df['Bollinger High'], df['Bollinger Low'], color='grey', alpha=0.1)

ax2.bar(df.index, df['Volume'], label='Volume', color='blue')
ax2.set_ylabel('Volume')
ax2.set_xlabel('Year')

ax3.plot(df.index, df['20-day SD'], label='Adj Close Volatility', color='red')
ax3.set_ylabel('Std. Dev.')
ax3.set_xlabel('Year')

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()


# # Histogram of Log Returns:

# In[119]:


plt.figure(figsize=(10, 5))
plt.hist(log_returns, bins=30)
plt.title('Histogram of Log Returns')
plt.ylabel('Frequency')
plt.xlabel('Log Returns')
plt.show()


# # Scatterplot of Volume vs Daily Returns:
# 
# 
# 

# In[120]:


plt.figure(figsize=(10, 7))
plt.scatter(returns, df['Volume'])
plt.title('Volume vs Daily Returns')
plt.ylabel('Trading Volume')
plt.xlabel('Daily Returns')
plt.show()


# # Report and Recommendation 
# 
# 
# Based on the analysis of Tesla's (TSLA) historical trading data from 2019 to 2023, it is evident that the stock has experienced periods of high volatility and significant price movements. The correlation between trading volume and price fluctuations suggests that market sentiment heavily influences Tesla's stock. Despite the volatility, the overall trend has been positive, with frequent instances of substantial gains. Therefore, a cautious yet optimistic approach is recommended for increasing the firm's stake in Tesla, considering the potential for growth while being mindful of the inherent risks.
# 

# 
# We believe that from 2019 to 2023, Tesla's stock has experienced significant volatility. 2020 was the year with the highest volatility, reaching 10.03%. In the same year, Tesla's stock rose 19.89% in a single day and fell -21.06% in a single day, indicating the wild volatility of its stock price. Investor interest was particularly strong in 2020, when the highest trading volume was recorded.
# 
# Tesla's share price reached a high of 11.80 in 2019, demonstrating its huge growth potential in the coming years. From 2019 to 2021, Tesla stock experienced significant growth, but there was a pullback in 2022. Overall, while Tesla's stock has strong growth potential, it also comes with high volatility. Therefore, we believe that a prudent investment strategy should be adopted. It is possible to modestly increase the proportion of investment in Tesla, but should ensure that the portfolio is diversified to balance the risk.
# 

# 
# Tesla's stock SMA is closely aligned with the closing price trend. And the price has repeatedly hit or broken through the Bollinger Bands which indicates high volatility. Considering Standard Deviation, volatility is relatively low until 2020. It soars in 2021 and reaches a higher degree in 2022. Most of the normal distribution of returns is very positive. Although there are fewer opportunities for extremely high returns, it is also difficult in that sense to lose money. The log returns distribution indicates that most daily returns are close to zero,  but there is potential for high gains. The presence of extreme values also highlights the risk of significant losses.
# As an investor, we recommend that the company consider the potential short-term volatility high risk, while at the same time bullish on Tesla's long-term growth prospects, so choose to increase holding the stock.

# 
# From the analysis we have conducted, it seems that Tesla is an ideal stock for investors with a high risk appetite who are willing to endure periods of high volatility in the expectation of substantial changes in stock price, which is often preferred by traders who seek to exploit volatility instead of directional trading. This would not have been a suitable stock for risk-averse investors looking to preserve capital, instead catering to risk-loving investors seeking quick capital accumulation in the recent past. Periods of massive stock price increase and high volatility in 2021-2022 coincide with the fiscal stimulus of the Covid-19 pandemic albeit with a slight lag, and stock prices have settled lower in recent years, likely coinciding with the end of Covid stimulus plus increased inflation and interest rates.
# 
# Tesla stock generally mirrors the NASDAQ and is buffeted by macro market conditions, though has lagged behind substantially in 2024, raising questions regarding further large 2021-style increases in stock price for the foreseeable future, instead demonstrating some indications of settling to lower volatility as indicated by log returns and 20-day rolling sigma. We would recommend a risk-neutral firm to cautiously increase TSLA positions, the value of which will likely increase to follow the NASDAQ, being wary that while volatility seems to be settling it remains high and TSLA hence entails greater tail risk, though this high volatility may be a desirable factor for risk-loving firms trading on volatility.
# 
