import os
from concurrent.futures import Future, ProcessPoolExecutor
from math import ceil
from pathlib import Path
from typing import Annotated, Generator, Union, cast

import numpy as np
import typer
from rich.progress import Progress

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
    stride: float | None,
) -> list[SingleAnomaly]:
    signal = Signal.load_signal(file, signal_type)
    return signal.check(anomaly=anomaly_type, th=experiment, window_size=window_size, stride=stride)


def merge_segments(
    futures: list[list[list[list[SingleAnomaly]]]], config: AnomalyConfig
) -> list[list[list[list[SingleAnomaly | MergedAnomalies]]]]:
    # merging close sections of same anomaly together
    anomalies: list[list[list[list[SingleAnomaly | MergedAnomalies]]]] = []
    for file in futures:
        file_anomalies: list[list[list[SingleAnomaly | MergedAnomalies]]] = []
        for signal in file:
            merged_anomaly: list[list[SingleAnomaly | MergedAnomalies]] = []
            for anomaly in signal:
                merged_segment: list[SingleAnomaly | MergedAnomalies] = []
                merging = None
                for segment in anomaly:
                    if merging is None:
                        merging = segment
                        continue
                    if segment.start - merging.end <= config.merge.max_seconds:
                        merging.extend(segment)
                    else:
                        merged_segment.append(merging)
                        merging = None
                if merging is not None:
                    merged_segment.append(merging)
                merged_anomaly.append(merged_segment)
            file_anomalies.append(merged_anomaly)
        anomalies.append(file_anomalies)
    return anomalies


def merge_anomalies(anomalies: list[list[list[list[SingleAnomaly | MergedAnomalies]]]], config: AnomalyConfig):
    merged_file_anomalies: list[list[list[MergedAnomalies | SingleAnomaly]]] = []
    for file in anomalies:
        merged_signal_anomalies: list[list[MergedAnomalies | SingleAnomaly]] = []
        for signal in file:
            if len(signal) == 1:  # if there's only single anomaly, consider it done and skip it
                merged_signal_anomalies.append(signal[0])
                continue
            merged_segments: list[MergedAnomalies | SingleAnomaly] = []
            for index_anomaly, anomaly in enumerate(signal[:-1]):
                for segment in anomaly:
                    merging = segment
                    while True:
                        try:
                            for anomaly_other in signal[index_anomaly + 1 :]:
                                for idx_segment, segment_other in enumerate(anomaly_other):
                                    if merging.overlap(segment_other) >= config.merge.overlap:
                                        if isinstance(merging, SingleAnomaly):
                                            merging = MergedAnomalies.join(merging, segment_other)
                                        else:
                                            merging.extend(segment_other)
                                        del anomaly_other[idx_segment]
                                        raise Exception()
                            break
                        except Exception as e:
                            ...
                    merged_segments.append(merging)
            merged_segments.extend(
                signal[-1]
            )  # consider the unmerged segments from last signals nonoverllaping, so they can be appended to the result.
            merged_signal_anomalies.append(merged_segments)
        merged_file_anomalies.append(merged_signal_anomalies)
    return merged_file_anomalies


@app.callback()
def main(debug: Annotated[bool, typer.Option(hidden=True)] = False):
    if debug:
        import debugpy

        debugpy.listen(5678)
        print("waiting for attaching debugger")
        debugpy.wait_for_client()


@app.command()
def check(
    signal_type: Annotated[
        Union[None, SignalType], typer.Argument(help="which signal type to look at. Check all signals by default")
    ] = None,
    anomaly_type: Annotated[
        Union[None, AnomalyType],
        typer.Argument(help="which anomaly to check for. If not specified check for all anomalies"),
    ] = None,
    window_size: Annotated[
        Union[None, float],
        typer.Option(help="Set length of the window.  Only works for single anomaly type."),
    ] = None,
    th: Annotated[
        Union[None, float],
        typer.Option(help="Threshold for looking anomaly. Only works for single anomaly type."),
    ] = None,
    stride: Annotated[
        Union[None, float],
        typer.Option(
            help="stride (shift between sections) for rolling window. Shift 1 by default. Only works for single anomaly."
        ),
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
    _return: bool = False,
) -> None | tuple:
    """
    Check and return sectors of given anomaly of specified signal/s
    """
    config = load_config()
    files = list(dataset_files(input))
    workers = workers if workers else 1
    with Progress() as progress:
        experiment_file = progress.add_task(
            "iterating over files",
            total=len(files),
        )
        futures: list[list[list[Future[list[SingleAnomaly]]]]] = []

        types = [anomaly_type] if anomaly_type else [anomaly for anomaly in AnomalyType]
        signal_types = [signal_type] if signal_type else [signal for signal in SignalType]
        # start processing files on separate processes
        with ProcessPoolExecutor(max_workers=workers) as executor:
            total = 0
            for file in files:
                file_futures = []
                for signal in signal_types:
                    signal_futures = []
                    config_type = getattr(config, str(signal))
                    for segment in types:
                        signal_futures.append(
                            executor.submit(
                                process_signal,
                                segment,
                                getattr(config_type, segment).th,
                                getattr(config_type, segment).window_size,
                                signal,
                                file,
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

        #unpack all result segmetns form fututres
        segments: list[list[list[list[SingleAnomaly]]]] = [
            [[future.result() for future in signal] for signal in file] for file in futures
        ]

        merged_segments = merge_segments(segments, config)# merge close segments of same anomaly type
        merged_anomalies = merge_anomalies(merged_segments, config)# merge overlapping segments

        for file in merged_anomalies:
            for signal in file:
                for segment in signal:
                    segment = segment if isinstance(segment,MergedAnomalies) else segment.convert()
        
        signal_type_names = [str(type) for type in signal_types]
        if _return:
            return merged_anomalies, signal_type_names

        if output:
            if output.is_file():
                output.unlink()
            with open(output, "a") as opened_file:
                for index, file in enumerate(files):
                    opened_file.write(f"file: {file}\n")
                    for idx, signal in enumerate(merged_anomalies[index]):
                        opened_file.write(f"{signal_type_names[idx]}\n")
                        signal = signal if isinstance(signal, list) else [signal]
                        for segment in signal:
                            opened_file.write(f"{str(segment)}\n")
                    opened_file.write("\n")
        else:
            for idx, file in enumerate(files):
                print(f"file: {file}")
                for segment in merged_anomalies[idx]:
                    print(str(segment))
                print()


@app.command()
def graph(
    start: Annotated[float, typer.Option(help="starting value of the testing threshold (inclusive)")],
    end: Annotated[float, typer.Option(help="ending value of the testing threshold (inclusive)")],
    step: Annotated[float, typer.Option(help="step size of the testing threshold")],
    x_axis_title: Annotated[str, typer.Option(help="title for the y axis")],
    y_axis_title: Annotated[str, typer.Option(help="title for the x axis")],
    title: Annotated[str, typer.Option(help="title for the graph")],
    signal_type: Annotated[SignalType, typer.Argument(help="which signal type to look at")],
    anomaly_type: Annotated[AnomalyType, typer.Argument(help="which anomaly to observe")],
    output: Annotated[Path, typer.Option(help="file path where to store the result graph")],
    input: Annotated[
        Union[None, Path],
        typer.Option(
            help="input path. for file without suffix, will test only this file. for folder, test all files with .dat suffix"
        ),
    ] = Path("data"),
    window_size: Annotated[float, typer.Option(help="set length of the window")] = 2,
    stride: Annotated[
        Union[None, float],
        typer.Option(help="stride (shift between sections) for rolling window. Shift 1 by default."),
    ] = None,
):
    """
    Test the specified anomaly type in the given threshold range and create graph. Units are in the seconds
    """
    import plotly.express as px

    x = []
    y = []
    threshold_range = list(np.linspace(start, end, ceil((end - start) / step) + 1))
    with Progress() as progress:
        experiments_task = progress.add_task("iterating over experiments", total=len(threshold_range))

        for experiment in threshold_range:
            futures: list[Future] = []
            # load the all neccesary files
            files = list(dataset_files(input))

            experiment_file = progress.add_task(
                "iterating over files",
                total=len(files),
            )
            # start processing files on separate processes
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                for file in files:
                    futures.append(
                        executor.submit(
                            process_signal,
                            anomaly_type,
                            experiment,
                            window_size,
                            signal_type,
                            file,
                            stride,
                        )
                    )

                # monitor procceses if they are all finished and updating progress bar
                while (n_finished := sum([future.done() for future in futures])) < len(futures):
                    progress.update(experiment_file, completed=n_finished, total=len(futures))

            progress.update(experiment_file, visible=False)

            # result gathering
            x.append(float(experiment))
            y.append(sum([len(future.result()) for future in futures]))

            progress.update(experiments_task, advance=1)

        progress.update(experiments_task, visible=False)

    # creating graph
    fig = px.line(
        x=x,
        y=y,
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        titlefont={"size": 30},
    )
    fig.write_image(output)


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


if __name__ == "__main__":
    app()
