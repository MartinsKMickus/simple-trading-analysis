

import argparse
import datetime
from tradinga.business_logic import BusinessLogic
from tradinga.data_analyzer import DataAnalyzer



parser = argparse.ArgumentParser(description='Simple Trading Analysis')
action_parser = parser.add_subparsers(dest='action')

ap_train = action_parser.add_parser('train', help='Train model')
ap_train.add_argument('--count', dest="count", metavar="COUNT",
                   help='Training symbol count')
ap_train.add_argument('-c',  dest="c", action='store_true',
                       help='Delete model and reset settings')

ap_metrics = action_parser.add_parser('metrics', help='Get metrics for symbol data')
ap_metrics.add_argument('-s', '--symbol', dest="symbol", metavar="SYMBOL", required=True,
                   help='Symbol')
ap_metrics.add_argument('-p', '--plot',  dest="p", action='store_true',
                       help='Plot prediction chart on real data')

ap_predict = action_parser.add_parser('predict', help='Predict symbol future')
ap_predict.add_argument('-s', '--symbol', dest="symbol", metavar="SYMBOL", required=True,
                   help='Symbol')
ap_predict.add_argument('--last-date', dest="last_date", metavar="LAST_DATE",
                   help='Last date to do analysis after YYYY/MM/DD')

ap_predict_market = action_parser.add_parser('predict_market', help='Predict market')
ap_predict_market.add_argument('--last-date', dest="last_date", metavar="LAST_DATE",
                   help='Last date to do analysis after YYYY/MM/DD')

args = parser.parse_args()

business_logic = BusinessLogic()
if args.action == 'train':
    if args.c:
        business_logic.data_analyzer.reset_model_and_settings()
    if args.count:
        business_logic.improve_model(symbol_count=args.count)
    else:
        business_logic.improve_model()
elif args.action == 'metrics':
    business_logic.print_metric_summary(symbol=args.symbol)
    business_logic.strategy_tester(symbol=args.symbol)
    if args.p:
        business_logic.show_prediction_chart(symbol=args.symbol)
elif args.action == 'predict':
    if args.last_date:
        business_logic.predict_next_close(symbol=args.symbol, last_date=datetime.datetime.strptime(args.last_date, '%Y/%m/%d'))
    else:
        business_logic.predict_next_close(symbol=args.symbol)
elif args.action == 'predict_market':
    if args.last_date:
        business_logic.predict_market(last_date=datetime.datetime.strptime(args.last_date, '%Y/%m/%d'))
    else:
        business_logic.predict_market()


# business_logic.show_prediction_chart(symbol='NIO')
# analyzer = DataAnalyzer(analyzer_name='ISITFINAL')
# print(analyzer.get_valuation(symbol='AAPL'))