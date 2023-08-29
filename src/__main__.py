import os
from concurrent.futures import Future, ProcessPoolExecutor
from math import ceil
from pathlib import Path
from typing import Annotated, Generator, Union, cast

import numpy as np
import typer
from rich.progress import Progress

from .config import load_config

from .filters import Anomaly, AnomalyType, Signal, SignalType

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
) -> list[Anomaly]:
    signal = Signal.load_signal(file, signal_type)
    return signal.check(
        anomaly=anomaly_type, th=experiment, window_size=window_size, stride=stride
    )


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
        SignalType, typer.Argument(help="which signal type to look at")
    ],
    window_size: Annotated[
        Union[None, float],
        typer.Option(help="Set length of the window.  Only works for single anomaly."),
    ] = None,
    th: Annotated[
        Union[None, float],
        typer.Option(
            help="Threshold for looking anomaly. Only works for single anomaly."
        ),
    ] = None,
    anomaly_type: Annotated[
        Union[None, AnomalyType],
        typer.Argument(
            help="which anomaly to check for. If not specified check for all anomalies. Only works for single anomaly."
        ),
    ] = None,
    stride: Annotated[
        Union[None, float],
        typer.Option(
            help="stride (shift between sections) for rolling window. Shift 1 by default. Only works for single anomaly."
        ),
    ] = None,
    input: Annotated[
        Path,
        typer.Option(
            help="input signal file or folder, if folder specified, extract all files with .dat suffix"
        ),
    ] = Path("data"),
    output: Annotated[
        Union[None, Path],
        typer.Option(help="path to the output file, print to console if not specified"),
    ] = None,
) -> None:
    """
    Check and return sectors of given anomaly of specified signal/s
    """
    config = load_config()
    files = list(dataset_files(input))
    with Progress() as progress:
        experiment_file = progress.add_task(
            "iterating over files",
            total=len(files),
        )
        futures = []
        anomalies: list[list[Anomaly]] = []
        # start processing files on separate processes
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            if anomaly_type:  # process single specified anomaly type
                futures = cast(list[Future], futures)
                for file in files:
                    futures.append(
                        executor.submit(
                            process_signal,
                            anomaly_type,
                            th if th else getattr(config, anomaly_type).th,
                            window_size
                            if window_size
                            else getattr(config, anomaly_type).window_size,
                            signal_type,
                            file,
                            stride if stride else getattr(config, anomaly_type).stride,
                        )
                    )

                # monitor procceses if they are all finished and updating progress bar
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        experiment_file, completed=n_finished, total=len(futures)
                    )

                anomalies = [future.result() for future in futures]

            else:  # process for all anomaly types
                futures = cast(list[list[Future]], futures)
                total = 0
                for file in files:
                    file_futures = []
                    for anomaly in AnomalyType:
                        file_futures.append(
                            executor.submit(
                                process_signal,
                                anomaly,
                                getattr(config, anomaly).th,
                                getattr(config, anomaly).window_size,
                                signal_type,
                                file,
                                getattr(config, anomaly).stride,
                            )
                        )
                    total += len(file_futures)
                    futures.append(file_futures)

                    # monitor procceses if they are all finished and updating progress bar
                    while (
                        n_finished := sum(
                            [
                                sum([future.done() for future in file])
                                for file in futures
                            ]
                        )
                    ) < total:
                        progress.update(
                            experiment_file, completed=n_finished, total=total
                        )

                progress.update(experiment_file, visible=False)

                anomalies: list[list[Anomaly]] = []
                for file_futures in futures:
                    file_anomalies = []
                    for future in file_futures:
                        file_anomalies.extend(future.result())
                    anomalies.append(file_anomalies)

        if output:
            if output.is_file():
                output.unlink()
            with open(output, "a") as opened_file:
                for index, file in enumerate(files):
                    opened_file.write(f"file: {file}\n")
                    for anomaly in anomalies[index]:
                        opened_file.write(f"{str(anomaly)}\n")
                    opened_file.write("\n")
        else:
            for idx, file in enumerate(files):
                print(f"file: {file}")
                for anomaly in anomalies[idx]:
                    print(str(anomaly))
                print()


@app.command()
def graph(
    start: Annotated[
        float, typer.Option(help="starting value of the testing threshold (inclusive)")
    ],
    end: Annotated[
        float, typer.Option(help="ending value of the testing threshold (inclusive)")
    ],
    step: Annotated[float, typer.Option(help="step size of the testing threshold")],
    x_axis_title: Annotated[str, typer.Option(help="title for the y axis")],
    y_axis_title: Annotated[str, typer.Option(help="title for the x axis")],
    title: Annotated[str, typer.Option(help="title for the graph")],
    signal_type: Annotated[
        SignalType, typer.Argument(help="which signal type to look at")
    ],
    anomaly_type: Annotated[
        AnomalyType, typer.Argument(help="which anomaly to observe")
    ],
    output: Annotated[
        Path, typer.Option(help="file path where to store the result graph")
    ],
    input: Annotated[
        Union[None, Path],
        typer.Option(
            help="input path. for file without suffix, will test only this file. for folder, test all files with .dat suffix"
        ),
    ] = Path("data"),
    window_size: Annotated[float, typer.Option(help="set length of the window")] = 2,
    stride: Annotated[
        Union[None, float],
        typer.Option(
            help="stride (shift between sections) for rolling window. Shift 1 by default."
        ),
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
        experiments_task = progress.add_task(
            "iterating over experiments", total=len(threshold_range)
        )

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
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        experiment_file, completed=n_finished, total=len(futures)
                    )

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
    signal_type: Annotated[
        SignalType, typer.Argument(help="which signal type to look at")
    ],
    input: Annotated[Path, typer.Argument(help="file path for signal file")],
    start: Annotated[
        int, typer.Argument(help="start of the looking section (in seconds)")
    ],
    end: Annotated[int, typer.Argument(help="end of the looking section (in seconds)")],
    peaks: Annotated[
        bool, typer.Option("--peaks/", help="whether to visualize finded peaks")
    ] = False,
):
    """
    open graph window with time series graph of the specified signal
    """
    Signal.load_signal(input, signal_type).look(start, end, peaks)


if __name__ == "__main__":
    app()
