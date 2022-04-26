# Greg Attra
# 04.26.22

'''
Config for plotting data
'''

from dataclasses import dataclass
from typing import Callable, List, Tuple

from lib.structs.experiment import Experiment


@dataclass
class DataFilter:
    label: str
    filter_func: Callable[[Experiment], bool]


@dataclass
class PlotComponent:
    label: str
    color: Tuple[float, float, float]
    filter_func: DataFilter
    datadirs: List[str]


@dataclass
class PlotConfig:
    title: str
    xaxis: str
    yaxis: str
    desc: str
    components: List[PlotComponent]
