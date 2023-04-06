# simple-trading-analysis
Stock/Crypto AI Machine Learning

## Prerequisites:
- Install requirements.txt
- Add `constants.py` file to tradinga directory and set following varables:
  - `ALPHA_VANTAGE_API_KEY` API key for Alpha Vantage
  - `DATA_DIR` Directory where data will be stored
  - `STOCK_DIR` Directory where trading data will be stored
  - `AI_DIR` Directory where AI data will be stored
  - `INTERVAL` Default trading interval (1min, 5min, 15min, 30min, 60min)

## How to run:
- `python main.py` Right now this command doesn't do anything. Arguments have to be added.

### Arguments
- `update` Update local data
  - `-r` Update in random order
  - `-s/--single SIMBOL` Update data for single symbol

- `ai` NOT IMPLEMENTED
  - `-r` Learn in random order
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-e/--epochs EPOCHS` How many epochs (Default: 100)

- `train` Create or retrain model
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--single SIMBOL` Symbol to use as train data (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-e/--epochs EPOCHS` How many epochs (Default: 100)
  - `--test-symbol TEST_SIMBOL` Symbol to use as validation data

- `model` Check model precision metrics (plot)
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--single SIMBOL` Single symbol analysis (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `--date-from YYYY/MM/DD` Take data from date
  - `--date-to YYYY/MM/DD` Take data to date
  - `-c/--capital START_CAPITAL` NOT IMPLEMENTED

- `predict` Predict future from last date in data (plot)
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--single SIMBOL` Single symbol analysis (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-n/--next NEXT` How much next values to predict (REQUIRED)
  - `--date-from YYYY/MM/DD` Take data from date
  - `--date-to YYYY/MM/DD` Take data to date. Predict from.