"""splitsafe: check audio train/validation/test splits for leakage."""
__version__ = "0.1.0"
from .scan import check_splits, Result  # noqa: E402,F401
