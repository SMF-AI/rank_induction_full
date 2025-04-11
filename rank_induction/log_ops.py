import os
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler


import mlflow
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs")

TRACKING_URI = "http://XXX.XXX.XXX.XXX:5000/"
EXP_NAME = "camelyon"


def get_experiment(experiment_name: str = EXP_NAME):
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        client.create_experiment(experiment_name)
        return client.get_experiment_by_name(experiment_name)

    return experiment


def get_logger(
    module_name: str, log_path: str = None, interval: int = 7
) -> logging.Logger:
    """지정된 모듈에 대한 로깅 기능을 설정하고 로거를 반환
    Args
        module_name (str): 모듈 이름.
        log_path (str): 로그 파일의 경로 및 이름 (optional).
                     파일 이름은 '.log'로 끝나야함
                     만약 지정하지 않으면, 로그는 LOG_DIR 내에 모듈 이름과 현재 날짜로 생성
        interval (int): 로그를 백업할 기간 (optional=7)
                     지정하지 않으면 7일마다 새로운 log파일을 생성
    Returns:
        (logging.Logger)
    Example:
        >>> logger = get_logger("my_module", "my_logs.log")
        >>> logger.info("이것은 정보 로그 메시지입니다.")
        >>> logger.error("이것은 오류 로그 메시지입니다.")
        >>> another_logger = get_logger("another_module")
        >>> another_logger.debug("이것은 디버그 로그 메시지입니다.")
    """
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")

    if log_path and not log_path.endswith(".log"):
        msg = (
            f"passed log_path ({log_path}), "
            "expected log_path must be ended with '.log'"
        )
        raise ValueError(msg)

    elif not log_path:
        start_time = now.strftime("%H:%M:%S")
        log_module_dir = os.path.join(LOG_DIR, module_name, today)
        os.makedirs(log_module_dir, exist_ok=True)

        log_path = os.path.join(log_module_dir, f"{module_name}-{start_time}.log")
    else:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    logger_formatter = logging.Formatter(
        fmt="{asctime}\t{name}\t{filename}:{lineno}\t{levelname}\t{message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logger_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="D", interval=interval, backupCount=0
    )
    file_handler.setFormatter(logger_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_child_run_ids(parent_run_id):
    # Initialize an empty list to store child run IDs
    child_run_ids = []

    # Search for all runs in the experiment
    experiment_id = mlflow.get_run(parent_run_id).info.experiment_id
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
    )

    # Collect the child run IDs
    for run in all_runs.iterrows():
        child_run_ids.append(run[1].run_id)

    return child_run_ids


def save_and_log_figure(filename: str) -> None:
    """그려진 figure을 MLflow내에 figure을 등록함

    Note:
        mlflow내에 figure을 바로 넣는 메서드가 없음. disk에 저장 필요.

    Args:
        filename (str): filename

    Example:
        >>> from seedp.metrics import plot_auroc
        >>> plot_auroc(metrics.labels, metrics.probs)
        >>> save_and_log_figure("auroc.png")

    """
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
    plt.clf()

    return
