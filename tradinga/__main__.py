import argparse

import pandas as pd
import constants

from data_helper import download_newest_data, load_existing_data, save_data_to_csv
from api_helper import alpha_vantage_list


parser = argparse.ArgumentParser(description='Simple Trading Analysis')
action_parser = parser.add_subparsers(dest='action')
ap_update = action_parser.add_parser('update', help='Update local data')
#ap_list = action_parser.add_parser('list', help='List symbols')

ap_update.add_argument('-r', action='store_true',
                       help='Update in random order')
#ap_list.add_argument('-o', action='store_true', help='Get online data')

args = parser.parse_args()


if args.action == 'update':
    print("Update")
    symbols = load_existing_data(None, constants.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        print("Empty symbols file. Trying to download it now.")
        symbols = alpha_vantage_list()
        save_data_to_csv(symbol=None, data=symbols, interval=None)
    if args.r:
        symbols = symbols.sample(frac=1).reset_index(drop=True)
    for symbol in symbols['symbol']:
        download_newest_data(symbol, constants.INTERVAL)
# elif args.action == 'list':
#     if args.o:
#         print("Online mode")
#     else:
#         print("Offline mode")