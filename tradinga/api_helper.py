import io
import time
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import yfinance



# Alpha Vantage API call to get list of stocks and ETFs
# Returns DataFrame
def get_nasdaq_symbols():
    import requests

    url = 'http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'

    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text), delimiter='|')
    # Remove empty values
    data.dropna(how='all', inplace=True)
    # Make sure all column manes are lowercase
    data.columns = map(str.lower, data.columns)

    return data


def yfinance_get_data(symbol: str, interval: str, from_date = None, max_tries=2):
    if from_date == None:
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
            from_date = datetime.datetime.now() - datetime.timedelta(days=730)
        else:
            from_date = datetime.datetime.now() - datetime.timedelta(days=3000)
    else:
        # Get difference from two dates
        delta = relativedelta(datetime.datetime.now(), from_date)
        years = delta.years
        months = delta.months
        days = delta.days
        if years == 0 and months == 0 and days < 1:
            print(f"{symbol} is already up to date")
            return
    print(f'Downloading data for {symbol}')
    tries = 0
    sleep_between_tries = 10
    while True:
        try:
            tries += 1
            data = yfinance.download(symbol, start=from_date, interval=interval, progress=False)
            break
        except:
            if tries >= max_tries:
                raise Exception(f'Failed to download {symbol}')
            print(f'Failed to download {symbol}. Retrying in {sleep_between_tries}s')
            time.sleep(sleep_between_tries)
            sleep_between_tries += 10
            continue

    # Removing timezone info
    idx = data.index
    idx = idx.tz_localize(None)
    data.index = idx
    data.reset_index(inplace=True)

    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
        data = data.sort_values(by='Datetime', ascending=False)
        data = data.rename(columns={'Datetime': 'time'})
        data['time'] = pd.to_datetime(data['time'])
    else:
        data = data.sort_values(by='Date', ascending=False)
        data = data.rename(columns={'Date': 'time'})
        data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d")

    
    if from_date:
        data = data[(data['time'] > from_date)]

    return data