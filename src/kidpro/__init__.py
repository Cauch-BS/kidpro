import os
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")

warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Error fetching version info.*",
    category=UserWarning,
)
