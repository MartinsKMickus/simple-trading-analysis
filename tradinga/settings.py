DATA_DIR = 'data'
STOCK_DIR = 'market'
SYMBOL_FILE = 'data/symbols.csv'

TESTING_DIR = 'test_data'

INTERVAL = '60m'
MIN_DATA_CHECKS = 300
DATA_WINDOW = 2
PREDICT_AFTER = 10
SINGLE_DATA_EPOCHS = 64

EXCLUDE = ['ARYD', 'TBIL', 'TSLA', 'AAPL', 'NIO', 'ATOM', 'CRON', 'CVNA', 'AMD']

# Required precision defines how much model has to be correct to predict trend direction.
# Below given number warning/failure message will be given.
#REQUIRED_PRECISION = 0.6 # NOT USED

USE_MONTE_CARLO = True # NOT WORKING TODO: Implement use Monte Carlo switch
ACCURACY_CUTOFF = 0.6

# Strategy tester settings:
# Also interacts with model learning:
DIVERSITY_THRESHOLD = 0.01
MIN_CONFIDENCE = 0.6
# Only strategy settings:
REWARD_RISK_RATIO = 3
RISK_FROM_VALUE = 0.02
MIN_REQUIRED_PROFIT = 0.03
MAX_REQUIRED_PROFIT = 0.09
START_BALANCE = 500.00
# Only take long positions
CASH_ACCOUNT = True
# Take a look at only first hour of the day (Interval < 1d)
FIRST_DAY_HOUR = True