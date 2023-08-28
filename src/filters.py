import multiprocessing
import os
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from pathlib import Path

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


@dataclass
class Anomaly:
    type: AnomalyType
    start: float
    end: float

    def __str__(self) -> str:
        return f"{self.start}-{self.end}: {self.type}"

def window_cwt(values: npt.NDArray) -> tuple[npt.NDArray, float]:
    filtered_signal = cwt(values, ricker, [5])[0]
    peaks = find_peaks(filtered_signal)[0]
    return filtered_signal, len(peaks)


@dataclass
class Signal:
    signal: npt.NDArray
    signal_fs: int # sampling frequency of the signal
    peaks_per_second: float
    path: Path 
    window_indexes: npt.NDArray = field(init=False)

    @classmethod
    def load_signal(cls, path: Path, type: SignalType):
        """
        load the signal file into memory

        Args:
            path (Path): path to the signal file
            type (SignalType): which signal from file load

        Returns:
            Signal: signal object with loaded signal
        """
        signals, fields = wfdb.rdsamp(path)

        signal_type = fields["sig_name"].index(type.value)
        signal: npt.NDArray = signals[:, signal_type].astype(np.float32)

        cpu_count = os.cpu_count()
        split = np.array_split(signal, cpu_count if cpu_count else 1)

        #preprocess signal in parallel on splitted signal
        with multiprocessing.Pool() as pool:
            preprocessed_signal = pool.map(window_cwt, split)
        filtered_signal = np.concatenate(
            [filtered_signal[0] for filtered_signal in preprocessed_signal]
        )

        peaks_per_second = np.sum(
            [peaks_window[1] for peaks_window in preprocessed_signal]
        ) / (len(signal) / fields["fs"])

        return cls(
            signal=filtered_signal,
            signal_fs=fields["fs"],
            peaks_per_second=peaks_per_second.item(),
            path=path,
        )

    def look(
        self,
        start: float,
        end: float,
        peaks: bool = True
    ):
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
            sns.scatterplot(x=(found_peaks / self.signal_fs) + start,y =[values[t] for t in found_peaks], ax = axes, color = PEAK_COLOR)
        plt.show(block=True)

    def check(
        self,
        anomaly: AnomalyType,
        th: float,
        window_size: float = 5,
        stride: float | None = None,
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        detected = False
        start = 0
        window_size = floor(window_size * self.signal_fs)
        stride = floor(stride * self.signal_fs) if stride else 1
        self.window_indexes = np.arange(0, window_size, 1)

        for idx, window in enumerate(
            np.lib.stride_tricks.sliding_window_view(self.signal, window_size)[
                ::stride, :
            ]
        ):
            anomaly_index = getattr(self, f"filter_{anomaly}")(window, th=th)
            if isinstance(anomaly_index, int):
                if not detected:
                    start = anomaly_index + idx * stride
                    detected = True
            elif detected:
                anomalies.append(
                    Anomaly(
                        anomaly, start / self.signal_fs, (idx * stride) / self.signal_fs
                    )
                )
                detected = False


        if detected:
            anomalies.append(
                Anomaly(
                    anomaly, start / self.signal_fs, len(self.signal) / self.signal_fs
                )
            )

        return anomalies
    
    #=== filtering functions===
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
        _, diag = np.polynomial.Polynomial.fit(
            self.window_indexes, window, 1, full=True
        )
        if diag[0] / len(window) < th:
            return 0

    def filter_peaks(self, window: npt.NDArray, th: float = 0.5):
        peaks = find_peaks(window)[0]
        seconds = len(window) / self.signal_fs
        peaks_per_second = len(peaks) / seconds

        if  (peaks_per_second > self.peaks_per_second * (1 + th)
            or peaks_per_second < self.peaks_per_second * (1 - th)):
                return 0
    #==========================