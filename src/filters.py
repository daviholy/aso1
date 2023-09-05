from functools import partial
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Callable, Self

import numpy as np
import seaborn as sns
import wfdb
from matplotlib import pyplot as plt
from numpy import typing as npt
from scipy.signal import cwt, find_peaks, ricker
from src.config import AnomalyConfig, load_config

from copy import deepcopy

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

    @abstractmethod
    def extend(self, other: Self):
        ...

    def overlap(self, other: "SingleAnomaly | MergedAnomalies") -> float:
        """
        measuring overlapping area

        Args:
            other (Self): other measuret segment

        Returns:
            float: overlapped area relative to other
        """
        # if self.end < other.start or other.end < self.start:
        #     return 0
        if self.start < other.start or self.end > other.end:
            self,other = other, self # type: ignore
        if other.length() == 0:
            if self.start <= other.start and self.end >= other.end:
                return 1
            return 0
        if self.end <= other.start or self.start >= other.end : # check if the there is so overlap
            return 0
        return max((min(self.end,other.end) - max(other.start,self.start)) / other.length(),0)
    
    def distance(self, other: "SingleAnomaly | MergedAnomalies") -> float:
        if self.overlap(other) > 0:
            return 0
        return max(self.start, other.start) - min(self.end, other.end)

    def length(self) -> float:
        return max(self.end - self.start, 0) 


@dataclass
class SingleAnomaly(Anomaly):
    type: AnomalyType

    def extend(self, other: Self) -> None:
        if not self.type == other.type:
            return
        if self.start > other.start:
            self.start = other.start
        if self.end < other.end:
            self.end = other.end

    def convert(self) -> "MergedAnomalies":
        return MergedAnomalies(self.start, self.end, {self.type : [self]})

    def __str__(self) -> str:
        return f"{self.start}-{self.end}: {str(self.type)}"


@dataclass
class MergedAnomalies(Anomaly):
    type: dict[AnomalyType,list[SingleAnomaly]]

    def extend(self, other: SingleAnomaly| Self):
        if self.start > other.start:
            self.start = other.start

        if self.end < other.end:
            self.end = other.end

        if isinstance(other, SingleAnomaly):
            if other.type == AnomalyType.line:
                ...
            self._extend_anomaly(other)

        else:
            for key in other.type:
                if key == AnomalyType.line:
                    ...
                anomalies = other.type[key]
                for anomaly in anomalies:
                    self._extend_anomaly(anomaly)
                    
                           

    def __str__(self) -> str:
        return f"{self.start}-{self.end}: {', '.join(list(self.type))}"
    
    def length(self,type:AnomalyType | None = None) -> float:
        if type:
            if not self.type.get(type):
                return 0.
            return sum([anomaly.length() for anomaly in self.type[type]])
        return self.end - self.start
    
    def _extend_anomaly(self,other: SingleAnomaly):
        if other.type == AnomalyType.line:
            ...
        anomalies = self.type.get(other.type)
        if not anomalies:
            self.type[other.type] = [other]
        else:
            overlaps = []
            distances = []
            for anomaly in self.type[other.type]:
                overlaps.append(other.overlap(anomaly))
                distances.append(other.distance(anomaly))
            id = overlaps.index(max(overlaps))
            if overlaps[id] == 0:
                id_dist = distances.index(min(distances))
                if distances[id_dist] <= load_config().merge.distance:
                    anomalies[id_dist].extend(deepcopy(other))
                else:
                    anomalies.append(deepcopy(other))
            else:
                anomalies[id].extend(deepcopy(other))

    @classmethod
    def join(cls, *arg: SingleAnomaly | Self) -> Self:
        start = min([anomaly.start for anomaly in arg])
        end = max([anomaly.end for anomaly in arg])
        tmp = cls(start,end,{})

        for anomaly in arg:
            if isinstance(anomaly,MergedAnomalies):
                for key in anomaly.type.keys():
                    for single_anomaly in anomaly.type[key]:
                        tmp.extend(single_anomaly)
            else:
                tmp.extend(anomaly)

        return tmp



def window_average(values: npt.NDArray, window_size: int) -> tuple[npt.NDArray, float]:
    for idx, window in enumerate(np.lib.stride_tricks.sliding_window_view(values, window_size)[window_size:]):
        values[window_size + idx] = window.mean()

    peaks = find_peaks(values)
    return values, len(peaks)


def find_peaks_filtered(values: npt.NDArray, wavelet_len: float, wavelet: Callable):
    filtered = cwt(values, wavelet, [wavelet_len])
    return find_peaks(filtered[0])[0]


@dataclass
class Signal:
    signal: npt.NDArray
    fs: int  # sampling frequency of the signal
    peaks_per_second: float
    path: Path
    window_indexes: npt.NDArray = field(init=False)
    wavelet: Callable
    wavelet_len: float

    @classmethod
    def load_signal(
        cls,
        path: Path,
        type: SignalType,
        cpu_count: int | None = os.cpu_count(),
        wavelet_len: float = 5,
        wavelet: Callable = ricker,
    ):
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
        # filtered_signal, peaks = window_average(signal,average_window_size)

        # peaks_per_second = peaks / (len(signal)/fields["fs"])

        peak_finding = partial(find_peaks_filtered, wavelet_len=wavelet_len, wavelet=ricker)

        # preprocess signal in parallel on splitted signal
        with multiprocessing.Pool(cpu_count) as pool:
            preprocessed_signal = pool.map(peak_finding, split)

        peaks_per_second = np.sum([len(peaks_window) for peaks_window in preprocessed_signal]) / (len(signal) / fields["fs"])

        return cls(
            signal=signal,
            fs=fields["fs"],
            peaks_per_second=peaks_per_second,
            path=path,
            wavelet=wavelet,
            wavelet_len=wavelet_len,
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
        values = self.signal[ceil(start * self.fs) : floor(end * self.fs)]
        sns.lineplot(y=values, x=np.arange(start, end, 1 / self.fs), ax=axes)
        if peaks:
            found_peaks = find_peaks_filtered(values, self.wavelet_len, self.wavelet)
            sns.scatterplot(x=(found_peaks / self.fs) + start, y=[values[t] for t in found_peaks], ax=axes, color=PEAK_COLOR)
        plt.show(block=True)

    def check(
        self,
        anomaly: AnomalyType,
        th: float,
        config: AnomalyConfig,
        window_size: float = 5,
        stride: float | None = None,
    ) -> list[SingleAnomaly]:
        anomalies: list[SingleAnomaly] = []
        detected = False
        index = 0
        start = 0
        window_size = floor(window_size * self.fs)
        stride = floor(stride * self.fs) if stride else 1
        self.window_indexes = np.arange(0, window_size, 1)
        prev_anomaly: SingleAnomaly | None = None

        if anomaly == AnomalyType.line:
            ...

        for idx, window in enumerate(np.lib.stride_tricks.sliding_window_view(self.signal, window_size)[::stride, :]):
            anomaly_index = getattr(self, f"filter_{anomaly}")(window, th=th)
            if isinstance(anomaly_index,tuple):
                if not detected:
                    start = anomaly_index[0] + idx * stride
                    detected = True
                index =idx * stride + anomaly_index[1]
            elif detected:
                current_anomaly = SingleAnomaly(start / self.fs, index / self.fs, anomaly)
                if prev_anomaly and prev_anomaly.distance(current_anomaly) <= config.merge.distance:
                    prev_anomaly.extend(current_anomaly)
                    continue
                
                anomalies.append(current_anomaly)
                prev_anomaly = current_anomaly
                detected = False

        if detected:
            current_anomaly = SingleAnomaly(start / self.fs, index / self.fs, anomaly)
            if prev_anomaly and prev_anomaly.distance(current_anomaly) <= config.merge.distance:
                prev_anomaly.extend(current_anomaly)
            else:
                anomalies.append(current_anomaly)

        return anomalies

    # === filtering functions===
    def filter_max(
        self,
        window: npt.NDArray,
        th: int = 90,
    ) -> tuple[int,int]| None:
        idx = np.argwhere(window >= th)
        if idx.size > 0 :
            return idx[0][0], idx[0][-1]

    def filter_min(self, window: npt.NDArray, th: int = -30) -> tuple[int,int] | None:
        idx = np.argwhere(window <= th)
        if idx.size > 0 :
            return idx[0][0], idx[0][-1]

    def filter_line(self, window: npt.NDArray, th: float = 0.03) -> tuple[int,int] | None:
        coef = np.polynomial.polynomial.Polynomial.fit(self.window_indexes, window, 1)
        if np.sqrt(np.power(coef(window) - window, 2)).mean() < th:
            return 0, window.shape[0]

    def filter_peaks(self, window: npt.NDArray, th: float = 0.5) -> tuple[int,int]| None:
        peaks = find_peaks_filtered(window, self.wavelet_len, self.wavelet)
        seconds = len(window) / self.fs
        peaks_per_second = len(peaks) / seconds

        if peaks_per_second > self.peaks_per_second * (1 + th) or peaks_per_second < self.peaks_per_second * (1 - th):
            return 0, window.shape[0]

    # ==========================
