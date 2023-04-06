import argparse
import datetime
import sys

import pandas as pd
from tradinga.ai_manager import draw_future, make_model, test_model_performance
import tradinga.constants as constants

from tradinga.data_helper import download_newest_data, get_data_interval, load_existing_data, save_data_to_csv


parser = argparse.ArgumentParser(description='Simple Trading Analysis')
action_parser = parser.add_subparsers(dest='action')
ap_update = action_parser.add_parser('update', help='Update local data')
ap_ai = action_parser.add_parser('ai', help='Start AI')
ap_train = action_parser.add_parser('train', help='Train model')
ap_check_model = action_parser.add_parser('model', help='Check model')
ap_predict = action_parser.add_parser('predict', help='Predict future')
# ap_list = action_parser.add_parser('list', help='List symbols')

ap_update.add_argument('-r', action='store_true',
                       help='Update in random order')
ap_update.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL",
                   help='Update single symbol')

ap_ai.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL",
                   help='Single symbol analysis')
ap_ai.add_argument('-l', '--last', dest="last", metavar="LAST_VALUES", type=int,
                   help='How much last values to use')

ap_train.add_argument('-p', '--path', dest="model_path", metavar="MODEL_PATH", required=True,
                   help='Model path')
ap_train.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL", required=True,
                   help='Single symbol analysis')
ap_train.add_argument('-w', '--window', dest="window", metavar="INPUT_WINDOW", type=int, required=True,
                   help='Input window for this model')
ap_train.add_argument('-e', '--epochs', dest="epochs", metavar="EPOCHS", type=int, default=100,
                   help='Training epochs. Default: 100')
ap_train.add_argument('--test-symbol', dest="test_symbol", metavar="TEST_SYMBOL", default=None,
                   help='Single symbol analysis')

ap_check_model.add_argument('-p', '--path', dest="model_path", metavar="MODEL_PATH", required=True,
                   help='Model path')
ap_check_model.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL", required=True,
                   help='Single symbol analysis')
ap_check_model.add_argument('-w', '--window', dest="window", metavar="INPUT_WINDOW", type=int, required=True,
                   help='Input window for this model')
ap_check_model.add_argument('--date-from', dest="date_from", metavar="DATE_FROM",
                   help='Date from YYYY/MM/DD')
ap_check_model.add_argument('--date-to', dest="date_to", metavar="DATE_TO",
                   help='Date to YYYY/MM/DD')
ap_check_model.add_argument('-c', '--capital', dest="start_capital", metavar="START_CAPITAL", type=int, default=1000,
                   help='Start capital. Default: 1000')

ap_predict.add_argument('-p', '--path', dest="model_path", metavar="MODEL_PATH", required=True,
                   help='Model path')
ap_predict.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL", required=True,
                   help='Single symbol analysis')
ap_predict.add_argument('-w', '--window', dest="window", metavar="INPUT_WINDOW", type=int, required=True,
                   help='Input window for this model')
ap_predict.add_argument('-n', '--next', dest="next", metavar="NEXT", type=int, required=True,
                   help='Predict next values')
ap_predict.add_argument('--date-from', dest="date_from", metavar="DATE_FROM",
                   help='Date from YYYY/MM/DD')
ap_predict.add_argument('--date-to', dest="date_to", metavar="DATE_TO",
                   help='Date to YYYY/MM/DD')
# ap_list.add_argument('-o', action='store_true', help='Get online data')

args = parser.parse_args()


if args.action == 'update':
    print("Update Started")
    if args.symbol:
        download_newest_data(args.symbol, constants.INTERVAL)
        sys.exit(0)
    symbols = load_existing_data(None, constants.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        from tradinga.api_helper import alpha_vantage_list
        print("Empty symbols file. Trying to download it now.")
        symbols = alpha_vantage_list()
        save_data_to_csv(symbol=None, data=symbols, interval=None)
    if args.r:
        symbols = symbols.sample(frac=1).reset_index(drop=True)
    for symbol in symbols['symbol']:
        download_newest_data(symbol, constants.INTERVAL)
elif args.action == 'ai':
    print("Not yet implemented")
    # if args.symbol:
    #     from tradinga.ai_helper import predict_simple_next_values
    #     from matplotlib import pyplot as plt
    #     print(f"Single mode for: {args.symbol}")
    #     data = load_existing_data(args.symbol, constants.INTERVAL)
    #     if not isinstance(data, pd.DataFrame):
    #         sys.exit(f'No data for {args.symbol}. Please update this data first!')
    #     data['time'] = pd.to_datetime(data['time'])
    #     data.sort_values('time', inplace=True)

    #     test_data = data.copy()
    #     test_data = test_data[(test_data['time'] < datetime.datetime.now()) & (test_data['time'] > datetime.datetime(2023, 3, 1))]
    #     data = data[(data['time'] < datetime.datetime(2023, 3, 1))]
        
        # make_few_models(data, test_data, 10, "LB10")
        # make_few_models(data, test_data, 30, "LB30")
        # make_few_models(data, test_data, 50, "LB50")
elif args.action == 'train':
    symbol = args.symbol
    model_path = args.model_path
    input_window = args.window
    epochs = args.epochs
    download_newest_data(symbol=symbol,interval=constants.INTERVAL)
    full_data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
    if args.test_symbol != None:
        download_newest_data(symbol=args.test_symbol,interval=constants.INTERVAL)
        test_data = load_existing_data(symbol=args.test_symbol, interval=constants.INTERVAL)
        make_model(data=full_data, look_back=input_window, epochs=epochs, model_path=model_path, test_data=test_data)
    else:
        make_model(data=full_data, look_back=input_window, epochs=epochs, model_path=model_path)
elif args.action == 'model':
    date_from = None
    date_to = None
    symbol = args.symbol
    capital = args.start_capital
    model_path = args.model_path
    input_window = args.window
    if args.date_from:
        date_from = datetime.datetime.strptime(args.date_from, '%Y/%m/%d')
    if args.date_to:
        date_to = datetime.datetime.strptime(args.date_to, '%Y/%m/%d')
    full_data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
    filtered_data = get_data_interval(data=full_data, date_from=date_from, date_to=date_to)
    test_model_performance(model_path=model_path,input_window=input_window,data=filtered_data,start_ammount=capital)
        # TODO: Implement money graph
elif args.action == 'predict':
    date_from = None
    date_to = None
    symbol = args.symbol
    model_path = args.model_path
    predict = args.next
    input_window = args.window
    download_newest_data(symbol=symbol,interval=constants.INTERVAL)
    if args.date_from:
        date_from = datetime.datetime.strptime(args.date_from, '%Y/%m/%d')
    if args.date_to:
        date_to = datetime.datetime.strptime(args.date_to, '%Y/%m/%d')
    full_data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
    filtered_data = get_data_interval(data=full_data, date_from=date_from, date_to=date_to)
    draw_future(model_path=model_path,input_window=input_window,data=filtered_data,predict=predict)