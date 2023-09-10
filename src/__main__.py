import os
from concurrent.futures import Future, ProcessPoolExecutor
from math import ceil
from pathlib import Path
from typing import Annotated, Generator, Union, cast

import numpy as np
import typer
import wfdb
from rich.progress import Progress, TaskID
from copy import deepcopy

from .config import AnomalyConfig, load_config
from .filters import AnomalyType, MergedAnomalies, Signal, SignalType, SingleAnomaly

app = typer.Typer()


def dataset_files(path: Path | None = Path("data")) -> Generator:
    """
    generate list of signal files for given path. If folder is given, then include all files with .dat suffix. othervise when file is fiven return list of this single file.
    Args:
        path (Path | None): input path, can be folder or signal file without suffix
    """
    path = path if path else Path("data/")

    if path.is_dir():
        for file in path.rglob("*.dat"):
            yield file.with_suffix("")
    else:
        yield path.with_suffix("")


def process_signal(
    anomaly_type: AnomalyType,
    experiment: float,
    window_size: float,
    signal_type: SignalType,
    file: Path,
    config: AnomalyConfig,
    stride: float | None,
) -> list[SingleAnomaly]:
    signal = Signal.load_signal(file, signal_type)
    return signal.check(anomaly=anomaly_type, th=experiment, window_size=window_size, stride=stride, config=config)


def process_merging(
    input: list[list[SingleAnomaly | MergedAnomalies]], config: AnomalyConfig
) -> list[SingleAnomaly | MergedAnomalies]:
    input = deepcopy(input)
    ignore = [[False for _ in file] for file in input]
    if len(input) == 1:
        return cast(list[SingleAnomaly | MergedAnomalies], input[0])
    merged_segments: list[SingleAnomaly | MergedAnomalies] = []
    for index_anomaly, anomaly in enumerate(input):
        for segment in anomaly:
            merging: SingleAnomaly | MergedAnomalies = segment
            while True:
                try:
                    for index_anomaly_other, anomaly_other in enumerate(input[index_anomaly:]):
                        for idx_segment, segment_other in enumerate(anomaly_other):
                            if (
                                ignore[index_anomaly + index_anomaly_other][idx_segment]
                                and merging is not segment_other
                                and (
                                    merging.overlap(segment_other) > 0
                                    or merging.distance(segment_other) <= config.distance
                                )
                            ):
                                merging = MergedAnomalies.join(merging, segment_other)
                                ignore[index_anomaly + index_anomaly_other][idx_segment] = True
                                raise Exception()
                    break
                except Exception as e:
                    ...
            merged_segments.append(merging)
    return merged_segments

def merge_anomalies(
    anomalies: list[list[list[list[SingleAnomaly]]]],
    config: AnomalyConfig,
    workers: int | None = os.cpu_count(),
    progress: Progress | None = None,
) -> list[list[list[SingleAnomaly | MergedAnomalies]]]:
    total = 0
    task = TaskID(0)
    merged_file_anomalies: list[list[list[SingleAnomaly | MergedAnomalies]]] = []
    futures: list[Future] = []
    signals_len = len(anomalies[0])
    if progress:
        task: TaskID = progress.add_task("merging anomalies", total=total)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for file in anomalies:
            for signal in file:
                futures.append(
                    executor.submit(process_merging, cast(list[list[SingleAnomaly | MergedAnomalies]], signal), config)
                )
                total += 1

        while (n_finished := sum([future.done() for future in futures])) < total:
            if progress:
                progress.update(task, completed=n_finished, total=total)

        if progress:
            progress.update(task, visible=False)

        merged_signal = []
        for future in futures:
            merged_signal.append(future.result())
            if len(merged_signal) == signals_len:
                merged_file_anomalies.append(merged_signal)
                merged_signal = []

    return merged_file_anomalies


def merge_signals(
    files: list[list[list[SingleAnomaly | MergedAnomalies]]],
    config: AnomalyConfig,
    workers: int | None = os.cpu_count(),
    progress: Progress | None = None,
) -> list[list[SingleAnomaly | MergedAnomalies]]:
    total = 0
    task = TaskID(0)
    futures: list[Future] = []
    if progress:
        task: TaskID = progress.add_task("merging signals", total=total)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for file in files:
            futures.append(executor.submit(process_merging, file, config))
            total += 1

        while (n_finished := sum([future.done() for future in futures])) < total:
            if progress:
                progress.update(task, completed=n_finished, total=total)

        if progress:
            progress.update(task, visible=False)

    return [future.result() for future in futures]


@app.callback()
def main(debug: Annotated[bool, typer.Option(hidden=True)] = False):
    if debug:
        import debugpy

        debugpy.listen(5678)
        print("waiting for attaching debugger")
        debugpy.wait_for_client()


@app.command()
def check(
    signal: Annotated[
        Union[None, SignalType], typer.Option(help="which signal type to look at. Check all signals by default")
    ] = None,
    anomaly: Annotated[
        Union[None, list[AnomalyType]],
        typer.Option(help="which anomaly to check for. If not specified check for all anomalies"),
    ] = None,
    input: Annotated[
        Path,
        typer.Option(help="input signal file or folder, if folder specified, extract all files with .dat suffix"),
    ] = Path("data"),
    output: Annotated[
        Union[None, Path],
        typer.Option(help="path to the output file, print to console if not specified"),
    ] = None,
    workers: Annotated[Union[None, int], typer.Option(help="specify how many parallel workers use")] = os.cpu_count(),
) -> None | tuple[list[list[SingleAnomaly | MergedAnomalies]], list[str]]:
    """
    Check and return sectors of specified signal/s. Merge all signals and anomaly types if not specified
    """
    files = list(dataset_files(input))
    with Progress() as progress:
        merged_signals, types = _check([signal] if isinstance(signal,SignalType) else signal, anomaly, input, workers, progress)
    if output:
        if output.is_file():
            output.unlink()
        with open(output, "a") as opened_file:
            for index, file in enumerate(files):
                opened_file.write(f"file: {file}\n")
                for idx, segment in enumerate(merged_signals[index]):
                    opened_file.write(f"{str(segment)}\n")
                opened_file.write("\n")
    else:
        for idx, file in enumerate(files):
            print(f"file: {file}")
            for segment in merged_signals[idx]:
                print(str(segment))
            print()


@app.command()
def graph(
    start: Annotated[float, typer.Option(help="starting value of the testing threshold (inclusive)")],
    end: Annotated[float, typer.Option(help="ending value of the testing threshold (inclusive)")],
    step: Annotated[float, typer.Option(help="step size of the testing threshold")],
    x_axis_title: Annotated[str, typer.Option(help="title for the y axis")],
    y_axis_title_count: Annotated[str, typer.Option(help="title for the x axis of count plot")],
    y_axis_title_time: Annotated[str, typer.Option(help="title for the x axis of the time plot")],
    signal: Annotated[SignalType, typer.Argument(help="which signal type to look at")],
    anomaly: Annotated[AnomalyType, typer.Argument(help="which anomaly to observe")],
    output: Annotated[Path, typer.Option(help="file path where to store the result graph")],
    input: Annotated[
        Path,
        typer.Option(
            help="input path. for file without suffix, will test only this file. for folder, test all files with .dat suffix"
        ),
    ] = Path("data"),
    window_size: Annotated[float, typer.Option(help="set length of the window")] = 2,
    stride: Annotated[
        Union[None, float],
        typer.Option(help="stride (shift between sections) for rolling window. Shift 1 by default."),
    ] = None,
    workers: Annotated[Union[None, int], typer.Option(help="specify how many parallel workers use")] = os.cpu_count(),
):
    """
    Test the specified anomaly type in the given threshold range and create graph. Units are in the seconds
    """
    import plotly.express as px

    x = []
    y_count = []
    y_time = []

    config = load_config()
    config_anomaly = getattr(getattr(config, signal), anomaly)

    config_anomaly.stride = stride
    config_anomaly.window_size = window_size
    with Progress() as progress:
        experiment_id = progress.add_task("iterating over experiments", total=len(np.arange(start, end, step)))
        for experiment in np.arange(start, end, step):
            config_anomaly.th = experiment

            result, types = _check(signal=[signal], anomaly=[anomaly], input=input, workers=workers, progress=progress)

            x.append(float(experiment))
            y_count.append(sum([len(file) for file in result]))
            y_time.append(sum([sum(segment.length() for segment in file) for file in result]))

            progress.update(experiment_id, advance=1)
        progress.update(experiment_id, visible=False)

    # creating graph with counted segments
    fig = px.line(
        x=x,
        y=y_count,
        markers=True,
    )
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title_count,
        font={"size": 30},
    )
    fig.write_image(output.with_stem(f"{output.stem}_count"))

    # creating graph with summed total time
    fig = px.line(
        x=x,
        y=y_time,
        markers=True,
    )
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title_time,
        font={"size": 30},
    )
    fig.write_image(output.with_stem(f"{output.stem}_time"))


@app.command()
def look(
    signal_type: Annotated[SignalType, typer.Argument(help="which signal type to look at")],
    input: Annotated[Path, typer.Argument(help="file path for signal file")],
    start: Annotated[int, typer.Argument(help="start of the looking section (in seconds)")],
    end: Annotated[int, typer.Argument(help="end of the looking section (in seconds)")],
    peaks: Annotated[bool, typer.Option("--peaks/", help="whether to visualize finded peaks")] = False,
):
    """
    open graph window with time series graph of the specified signal
    """
    Signal.load_signal(input, signal_type).look(start, end, peaks)


def _check(
    signal: None | list[SignalType] = None,
    anomaly: None | list[AnomalyType] = None,
    input: Path | None = None,
    workers: None | int = os.cpu_count(),
    progress: None | Progress = None,
) -> tuple[list[list[SingleAnomaly | MergedAnomalies]], list[str]]:
    files = list(dataset_files(input))
    config = load_config()
    workers = workers if workers else 1
    progress = progress if progress else Progress()
    experiment_file = progress.add_task(
        "iterating over files",
        total=len(files),
    )
    futures: list[list[list[Future[list[SingleAnomaly]]]]] = []

    types = anomaly if anomaly else [anomaly for anomaly in AnomalyType]
    signal_types = signal if signal else [signal for signal in SignalType]
    # start processing files on separate processes
    with ProcessPoolExecutor(max_workers=workers) as executor:
        total = 0
        for file in files:
            file_futures = []
            for current_signal in signal_types:
                signal_futures = []
                config_type = getattr(config, str(current_signal))
                for segment in types:
                    signal_futures.append(
                        executor.submit(
                            process_signal,
                            segment,
                            getattr(config_type, segment).th,
                            getattr(config_type, segment).window_size,
                            current_signal,
                            file,
                            config,
                            getattr(config_type, segment).stride,
                        )
                    )
                total += len(signal_futures)
                file_futures.append(signal_futures)
            futures.append(file_futures)

        # monitor procceses if they are all finished and updating progress bar
        while (
            n_finished := sum([sum([sum([future.done() for future in signal]) for signal in file]) for file in futures])
        ) < total:
            progress.update(experiment_file, completed=n_finished, total=total)

    progress.update(experiment_file, visible=False)
    if not progress:
        progress.__exit__()

    # unpack all result segments from futures
    segments: list[list[list[list[SingleAnomaly]]]] = [
        [[future.result() for future in signal] for signal in file] for file in futures
    ]
    merged_anomalies = merge_anomalies(segments, config, workers, progress)  # merge overlapping segments
    if len(merged_anomalies[0]) == 1:  # check if there's single signal. Assuming that it's same for all files
        merged_signals = [signal[0] for signal in merged_anomalies]  # unpack the single signal
    else:
        merged_signals: list[list[SingleAnomaly | MergedAnomalies]] = merge_signals(
            merged_anomalies, config, workers, progress
        )

    for file in merged_signals:
        for segment in file:
            segment = cast(MergedAnomalies | SingleAnomaly, segment)
            segment = segment if isinstance(segment, MergedAnomalies) else segment.convert()

    signal_type_names = [str(type) for type in signal_types]
    return merged_signals, signal_type_names


if __name__ == "__main__":
    app()
