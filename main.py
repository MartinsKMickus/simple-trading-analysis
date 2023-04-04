import argparse
import datetime
import sys

import pandas as pd
from tradinga.ai_manager import make_few_models
import tradinga.constants as constants

from tradinga.data_helper import download_newest_data, load_existing_data, save_data_to_csv


parser = argparse.ArgumentParser(description='Simple Trading Analysis')
action_parser = parser.add_subparsers(dest='action')
ap_update = action_parser.add_parser('update', help='Update local data')
ap_ai = action_parser.add_parser('ai', help='Start AI')
# ap_list = action_parser.add_parser('list', help='List symbols')

ap_update.add_argument('-r', action='store_true',
                       help='Update in random order')
ap_update.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL",
                   help='Update single symbol')
ap_ai.add_argument('-s', '--single', dest="symbol", metavar="SYMBOL",
                   help='Single symbol analysis')
ap_ai.add_argument('-l', '--last', dest="last", metavar="LAST_VALUES", type=int,
                   help='How much last values to use')
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
    print("AI Activated")
    if args.symbol:
        from tradinga.ai_helper import predict_simple_next_values
        from matplotlib import pyplot as plt
        print(f"Single mode for: {args.symbol}")
        data = load_existing_data(args.symbol, constants.INTERVAL)
        if not isinstance(data, pd.DataFrame):
            sys.exit(f'No data for {args.symbol}. Please update this data first!')
        data['time'] = pd.to_datetime(data['time'])
        data.sort_values('time', inplace=True)

        test_data = data.copy()
        test_data = test_data[(test_data['time'] < datetime.datetime.now()) & (test_data['time'] > datetime.datetime(2023, 3, 1))]
        data = data[(data['time'] < datetime.datetime(2023, 3, 1))]
        
        make_few_models(data, test_data, "First_test")
# elif args.action == 'list':
#     if args.o:
#         print("Online mode")
#     else:
#         print("Offline mode")
