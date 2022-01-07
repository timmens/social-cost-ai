from project.config import SRC
from project.tracker_example import track_neural_network_fit


def task_run_tracker_example():
    logdir = SRC / "tracker_logdir"
    track_neural_network_fit(log_directory=logdir)
