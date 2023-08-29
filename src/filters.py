import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Self

import numpy as np
import seaborn as sns
import wfdb
from matplotlib import pyplot as plt
from numpy import typing as npt
from scipy.signal import cwt, find_peaks, ricker

PEAK_COLOR = "r"


class AnomalyType(str, Enum):
    max = "max"
    min = "min"
    line = "line"
    peak_count = "peaks"

    def __str__(self) -> str:
        return self.value


class SignalType(str, Enum):
    icp = "ICP"
    abp = "ABP"

    def __str__(self) -> str:
        return self.value

@dataclass()
class Anomaly(ABC):
    start: float
    end: float
    type: AnomalyType | list[AnomalyType]

    @abstractmethod
    def extend(self, other: 'SingleAnomaly | MergedAnomalies'):
        ...
    def overlap(self,other: 'SingleAnomaly | MergedAnomalies')-> float:
        """
        measuring overlapping area

        Args:
            other (Self): other measuret segment

        Returns:
            float: overlapped area relative to other  
        """
        if self.end < other.start or other.end < self.start:
            return 0 
        if self.start <= other.start:
            return 1 if self.end >= other.end else (self.end - other.start) / other.length()
        else:
            return self.length() / other.length() if self.end <= other.end else (other.end - self.start) / other.length()

    def length(self) -> float:
        return self.end - self.start
        
        



@dataclass
class SingleAnomaly(Anomaly):
    type: AnomalyType

    def extend(self, other: Self) -> None:
        if self.start > other.start:
            self.start = other.start
        if self.end < other.end:
            self.end = other.end

    def convert(self) -> 'MergedAnomalies':
        return MergedAnomalies(self.start,self.end,[self.type])

    def __str__(self) -> str:
        return f"{self.start}-{self.end}: {self.type}"

@dataclass    
class MergedAnomalies(Anomaly):
    type: list[AnomalyType]

    def extend(self,other:Self | SingleAnomaly):
        if self.start > other.start:
            self.start = other.start

        if self.end < other.end:
            self.end = other.end
            
        if isinstance(other,SingleAnomaly):
            tmp = set(self.type)
            tmp.add(other.type)
            self.type = list(tmp)
        else:
            self.type = list(set(self.type).union(other.type))
        
    def __str__(self) ->str:
        return f"{self.start}-{self.end}: {', '.join(self.type)}"
    
    @classmethod
    def join(cls,*arg: SingleAnomaly| Self):
        start = min([anomaly.start for anomaly in arg])
        end = max([anomaly.end for anomaly in arg])
        types = []
        for anomaly in arg:
           types.append(anomaly.type) if isinstance(anomaly, SingleAnomaly) else types.extend(anomaly.type)
        return cls(start,end,list(set(types)))
    


def window_cwt(values: npt.NDArray) -> tuple[npt.NDArray, float]:
    filtered_signal = cwt(values, ricker, [5])[0]
    peaks = find_peaks(filtered_signal)[0]
    return filtered_signal, len(peaks)


@dataclass
class Signal:
    signal: npt.NDArray
    signal_fs: int  # sampling frequency of the signal
    peaks_per_second: float
    path: Path
    window_indexes: npt.NDArray = field(init=False)

    @classmethod
    def load_signal(cls, path: Path, type: SignalType, cpu_count: int | None = os.cpu_count()):
        """
        load the signal file into memory

        Args:
            path (Path): path to the signal file
            type (SignalType): which signal from file load

        Returns:
            Signal: signal object with loaded signal
        """
        cpu_count = cpu_count if cpu_count else 1
        signals, fields = wfdb.rdsamp(path)

        signal_type = fields["sig_name"].index(type.value)
        signal: npt.NDArray = signals[:, signal_type].astype(np.float64)
        del signals        

        split = np.array_split(signal, cpu_count)
        preprocessed_signal = []

        # preprocess signal in parallel on splitted signal
        with multiprocessing.Pool(cpu_count) as pool:
            preprocessed_signal = pool.map(window_cwt, split)

        filtered_signal = np.concatenate([filtered_signal[0] for filtered_signal in preprocessed_signal])

        peaks_per_second = np.sum([peaks_window[1] for peaks_window in preprocessed_signal]) / (len(signal) / fields["fs"])

        return cls(
            signal=filtered_signal,
            signal_fs=fields["fs"],
            peaks_per_second=peaks_per_second.item(),
            path=path,
        )

    def look(self, start: float, end: float, peaks: bool = True):
        """
        Open graph window with time series of the signal in specified range. Unit are in the seconds

        Args:
            start (float): start of the range (in seconds)
            end (float): end of the range (in seconds)
            peaks (bool, optional): Whether to visualize peaks. Defaults to True.
        """
        _, axes = plt.subplots()
        values = self.signal[ceil(start * self.signal_fs) : floor(end * self.signal_fs)]
        sns.lineplot(y=values, x=np.arange(start, end, 1 / self.signal_fs), ax=axes)
        if peaks:
            found_peaks = find_peaks(values)[0]
            sns.scatterplot(
                x=(found_peaks / self.signal_fs) + start, y=[values[t] for t in found_peaks], ax=axes, color=PEAK_COLOR
            )
        plt.show(block=True)

    def check(
        self,
        anomaly: AnomalyType,
        th: float,
        window_size: float = 5,
        stride: float | None = None,
    ) -> list[SingleAnomaly]:
        anomalies: list[SingleAnomaly] = []
        detected = False
        start = 0
        window_size = floor(window_size * self.signal_fs)
        stride = floor(stride * self.signal_fs) if stride else 1
        self.window_indexes = np.arange(0, window_size, 1)

        for idx, window in enumerate(np.lib.stride_tricks.sliding_window_view(self.signal, window_size)[::stride, :]):
            anomaly_index = getattr(self, f"filter_{anomaly}")(window, th=th)
            if isinstance(anomaly_index, int):
                if not detected:
                    start = anomaly_index + idx * stride
                    detected = True
            elif detected:
                anomalies.append(SingleAnomaly(start / self.signal_fs, (idx * stride) / self.signal_fs,anomaly))
                detected = False

        if detected:
            anomalies.append(SingleAnomaly(start / self.signal_fs, len(self.signal) / self.signal_fs, anomaly))

        return anomalies

    # === filtering functions===
    def filter_max(
        self,
        window: npt.NDArray,
        th: int = 90,
    ) -> int | None:
        if window.max() >= th:
            return window.argmax().item()

    def filter_min(self, window: npt.NDArray, th: int = -30) -> int | None:
        if window.min() <= th:
            return window.argmin().item()

    def filter_line(self, window: npt.NDArray, th: float = 0.03) -> int | None:
        coef = np.polynomial.polynomial.Polynomial.fit(self.window_indexes, window, 1)
        if np.power(coef(window) - window,2).mean() < th:
            return 0

    def filter_peaks(self, window: npt.NDArray, th: float = 0.5):
        peaks = find_peaks(window)[0]
        seconds = len(window) / self.signal_fs
        peaks_per_second = len(peaks) / seconds

        if peaks_per_second > self.peaks_per_second * (1 + th) or peaks_per_second < self.peaks_per_second * (1 - th):
            return 0

    # ==========================
