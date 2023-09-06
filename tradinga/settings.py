DATA_DIR = 'data'
STOCK_DIR = 'market'
SYMBOL_FILE = 'data/symbols.csv'

TESTING_DIR = 'test_data'

INTERVAL = '60m'
MIN_DATA_CHECKS = 300
DATA_WINDOW = 16
PREDICT_AFTER = 10
SINGLE_DATA_EPOCHS = 64

EXCLUDE = ['ARYD', 'TBIL', 'TSLA', 'AAPL', 'NIO', 'ATOM', 'CRON', 'CVNA', 'AMD']

TEST_STOCKS = ['TSLA', 'AAPL', 'NIO', 'ATOM', 'CRON', 'CVNA', 'AMD']

# Required precision defines how much model has to be correct to predict trend direction.
# Below given number warning/failure message will be given.
#REQUIRED_PRECISION = 0.6 # NOT USED

USE_MONTE_CARLO = False
ACCURACY_CUTOFF = 0.66

# Strategy tester settings:
# Also interacts with model learning:
DIVERSITY_THRESHOLD = 0.001
MIN_CONFIDENCE = 0.6
# Only strategy settings:
REWARD_RISK_RATIO = 2
# How much account balance there are will to risk
ACCOUNT_RISK = 0.01
MIN_REQUIRED_PROFIT = 0.03
MAX_REQUIRED_PROFIT = 0.2
START_BALANCE = 2000.00
# Wait for better entry as fraction to risk side (in percent)
BETTER_ENTRY = 0.1
WAIT_FOR_ENTRY = 2
# Time Units Extra To Wait In Trade:
TIME_UNITS_WAIT = 30
# Only take long positions
CASH_ACCOUNT = True
# Take a look at only first hour of the day (Interval < 1d)
FIRST_DAY_HOUR = True

# Training statistics:
RUNNING_AVG = 10