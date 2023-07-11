# simple-trading-analysis
AI stock market recommendation generator.
- The tool target is to provide insight of stock market (NASDAQ for now) in form of a list that describes possible stock value change after specific time.
- User should provide time intervals that should be analyzed and time units in future to get prediction for. (Short-Term Swing Trading for now)
- Tool gives a list of analysed stocks, predicted price movement and probability of analysis being right or wrong.

<b>DISCLAIMER: Make trades at your own risk. Decisions should be your own!</b>


## Prerequisites:
- Install requirements.txt

## How to run:
Use: `python tradinga.py` and add arguments.

1. Train model by using `train` argument.
2. Test model reliability by using `metrics` argument. Use different symbols to test. Tweak `settings.py` for best results.
3. Either use `predict` argument to predict specific symbol value afer provided date or use `predict_market` argument to get summarized report of all market of downloaded symbols.

<b>Note that `predict_market` is time consuming action!</b>


### Arguments
- `train` Create or retrain model (While training process can be stopped with Ctrl+C)
  - `--count COUNT` Count on how many symbols model has to be trained (Default: All symbols)
  - `-c` Clean settings and delete model before training
- `metrics` Get metrics for symbol and strategy
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `-p` Plot data of predicted values on top of actual values
- `predict` Predict symbol value
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `--last-date YYYY/MM/DD` Make prediction from this date
- `predict_market` Predict next market values and make csv with predictions
  - `--last-date YYYY/MM/DD` Make prediction from this date
  - `--count COUNT` How much symbols to analyze. Will go in random order (Default: All symbols)


## Code structure
Code consists of different classes.
This graph should approximately represent data flow within:

<b>GRAPH TO BE ADDED </b>


## Development
To make sure issues are closed automatically when iteration is closed, changes goes to main, follow this guide:

![Release_Diagram drawio](https://user-images.githubusercontent.com/64271878/234239562-07462dde-3b84-4e93-9243-073f17125104.png)
