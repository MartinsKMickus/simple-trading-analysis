DATA_DIR = 'data'
STOCK_DIR = 'market'
SYMBOL_FILE = 'data/symbols.csv'

TESTING_DIR = 'test_data'

INTERVAL = '60m'
MIN_DATA_CHECKS = 300

EXCLUDE = []

# Required precision defines how much model has to be correct to predict trend direction.
# Below given number warning/failure message will be given.
REQUIRED_PRECISION = 0.6 # NOT USED

# Strategy tester settings:
MIN_CONFIDENCE = 0.4
REWARD_RISK_RATIO = 1
RISK_FROM_VALUE = 0.02
MIN_REQUIRED_PROFIT = 0.03
MAX_REQUIRED_PROFIT = 1
START_BALANCE = 2000.00
# Only take long positions
CASH_ACCOUNT = False
# Take a look at only first hour of the day
FIRST_DAY_HOUR = False