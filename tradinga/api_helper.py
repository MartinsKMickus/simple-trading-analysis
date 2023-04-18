import io
import time
import requests
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import yfinance

import tradinga.constants as constants


# Alpha Vantage API call to get list of stocks and ETFs
# Returns DataFrame
def alpha_vantage_list():
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={constants.ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def yfinance_get_data(symbol: str, interval: str, from_date = None, max_tries=3):
    if from_date == None:
        from_date = datetime.datetime.now() - datetime.timedelta(days=730)
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
            data = yfinance.download(symbol, start=from_date, interval=interval, progress=False).sort_values(by='Datetime', ascending=False)
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
    if isinstance(data, pd.DataFrame):
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.rename(columns={'Datetime': 'time'})
        if from_date:
            data = data[(data['time'] > from_date)]

    return data