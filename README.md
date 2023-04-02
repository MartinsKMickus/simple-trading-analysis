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
- `python tradinga` Right now this command doesn't do anything. Arguments have to be added.

### Arguments
- `update` Update local data
  - `-r` Update in random order
  - `-s/--single SIMBOL` Update data for single symbol
- `ai` AI mode
  - `-s/--single SIMBOL` Single symbol analysis