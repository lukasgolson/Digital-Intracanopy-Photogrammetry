from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelLimit:
    """
    A class used to represent the limits for a specific channel in the data.

    Attributes:
    ----------
    channel : int
        The channel index to which the limit is applied.
    min_val : Optional[float]
        The minimum value for the channel limit. Defaults to None.
    max_val : Optional[float]
        The maximum value for the channel limit. Defaults to None.
    mask_na : Optional[bool]
        If True, mask (exclude) NA values for this channel. Defaults to False.
    inclusive : Optional[bool]
        If True, the range is inclusive, and values within the range will be masked out.
        If False, the range is exclusive, and values outside the range will be masked out. Defaults to False.
    """

    channel: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mask_na: Optional[bool] = False
    inclusive: Optional[bool] = False


