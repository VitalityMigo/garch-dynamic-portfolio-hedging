DEFAULT_TICKERS = ["MSFT", "JPM", "BBY", "F", "KO", "AMD", "UBER", "CMI", "DE", "SLB"]

# Hedge instrument
HEDGE_TICKER: str = "SPY"

# Default maturity
DEFAULT_START_DATE: str = "2020-01-01"
DEFAULT_END_DATE: str | None = None

# Year trading days
TRADING_DAYS: int = 252

# Windows
ROLLING_WINDOW: int = 126
GARCH_LOOKBACK: int = 506
REALIZED_VOL_WINDOW: int = 20

# Portfolio capital
DEFAULT_INITIAL_CAPITAL: float = 100_000.0

# Circuit breaker
MAX_HEDGE_RATIO: float = 3.0
MAX_SPY_POSITION: float = 500.0
MIN_CORRELATION: float = 0.10
MIN_ROLLING_OBS: int = 60

# target beta for backtest
DEFAULT_TARGET_BETA: float = 0.5
