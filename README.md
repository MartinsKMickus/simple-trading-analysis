# simple-trading-analysis
AI stock market recommendation generator.
- The tool target is to provide insight of stock market (NASDAQ for now) in form of a list that describes possible stock value change after specific time.
- User should provide time intervals that should be analyzed and time units in future to get prediction for. (Short-Term Swing Trading for now)
- Tool gives a list of analysed stocks, predicted price movement and probability of analysis being right or wrong.

<b>DISCLAIMER: Make trades at your own risk. Decisions should be your own!</b>


## Prerequisites:
- Install requirements.txt

## How to run:
- `python tradinga.py` Right now this command doesn't do anything. Arguments have to be added.

### Arguments
- `train` Create or retrain model (While training process can be stopped with Ctrl+C)
  - `--count COUNT` Count on how many symbols model has to be trained (Default: All symbols)
  - `-c` Clean settings and delete model before training
- `metrics` Predict symbol value
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `-p` Plot data of predicted values on top of actual values
- `predict` Predict symbol value
  - `-s/--symbol SIMBOL` Single symbol analysis (REQUIRED)
  - `--last-date YYYY/MM/DD` Make prediction from this date
- `predict_market` Predict symbol value
  - `--last-date YYYY/MM/DD` Make prediction from this date


## Code structure
Code consists of different classes.
This graph should approximately represent data flow within:

<b>GRAPH TO BE ADDED </b>


## Development
To make sure issues are closed automatically when iteration is closed, changes goes to main, follow this guide:

![Release_Diagram drawio](https://user-images.githubusercontent.com/64271878/234239562-07462dde-3b84-4e93-9243-073f17125104.png)
