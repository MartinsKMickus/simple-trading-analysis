# simple-trading-analysis
AI stock market recommendation generator.
- The tool target is to provide insight of stock market (NASDAQ for now) in form of a list that describes possible stock value change after specific time.
- User should provide time intervals that should be analyzed and time units in future to get prediction for. (Short-Term Swing Trading for now)
- Tool gives a list of analysed stocks, predicted price movement and probability of analysis being right or wrong.

<b>DISCLAIMER: Make trades at your own risk. Decisions should be your own!</b>


## Prerequisites:
- Install requirements.txt

## How to run:
- `python main.py` Right now this command doesn't do anything. Arguments have to be added.

### Arguments
- `update` Update local data
  - `-r` Update in random order
  - `-s/--symbol SIMBOL` Update data for single symbol

- `ai` NOT IMPLEMENTED
  - `-r` Learn in random order
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-e/--epochs EPOCHS` How many epochs (Default: 100)

- `train` Create or retrain model
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--symbol SIMBOL` Symbol to use as train data (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-e/--epochs EPOCHS` How many epochs (Default: 100)
  - `--test-symbol TEST_SIMBOL` Symbol to use as validation data

- `model_metrics` Get model metrics on local data
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)

- `model` Check model precision metrics (plot)
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `--date-from YYYY/MM/DD` Take data from date
  - `--date-to YYYY/MM/DD` Take data to date
  - `-c/--capital START_CAPITAL` NOT IMPLEMENTED

- `predict` Predict future from last date in data (plot)
  - `-p/--path MODEL_PATH` Model path (REQUIRED)
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `-w/--window INPUT_WINDOW` How much last data values to use for prediction (REQUIRED)
  - `-n/--next NEXT` How much next values to predict (REQUIRED)
  - `--date-from YYYY/MM/DD` Take data from date
  - `--date-to YYYY/MM/DD` Take data to date. Predict from.

## Code structure
Code consists of different modules from which main ones are `DataManager` and `AIManager`.
This graph should approximately represent data flow within:

![image](https://user-images.githubusercontent.com/64271878/234019715-df74c364-21db-4cd7-923a-4640626fb39c.png)


## Development
To make sure issues are closed automatically when iteration is closed, changes goes to main, follow this guide:

![Release_Diagram drawio](https://user-images.githubusercontent.com/64271878/234239562-07462dde-3b84-4e93-9243-073f17125104.png)
