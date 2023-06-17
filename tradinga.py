

import argparse
from tradinga.business_logic import BusinessLogic



parser = argparse.ArgumentParser(description='Simple Trading Analysis')
action_parser = parser.add_subparsers(dest='action')

ap_train = action_parser.add_parser('train', help='Train model')
ap_train.add_argument('--count', dest="count", metavar="COUNT",
                   help='Model path')


args = parser.parse_args()

if args.action == 'train':
    business_logic = BusinessLogic()
    if args.count:
        business_logic.improve_model(symbol_count=args.count)
    else:
        business_logic.improve_model()