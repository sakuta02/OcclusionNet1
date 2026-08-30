import os

REQUIRED_ENV = ("CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")


class NullTracker:
    """Без ключей ClearML обучение идёт как есть, метрики только в stdout."""

    def scalar(self, *args, **kwargs):
        pass

    def figure(self, *args, **kwargs):
        pass

    def connect(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ClearMLTracker:
    def __init__(self, project_name, task_name):
        from clearml import Task

        self.task = Task.init(project_name=project_name, task_name=task_name)
        self.logger = self.task.get_logger()

    def scalar(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value=value, iteration=iteration)

    def figure(self, title, series, figure, iteration):
        self.logger.report_matplotlib_figure(title=title, series=series, figure=figure, iteration=iteration)

    def connect(self, values):
        self.task.connect(values)

    def close(self):
        self.task.close()


def make_tracker(project_name, task_name):
    """Ключи берутся только из окружения (`CLEARML_*`) — в репозитории их нет.
    Создать: ClearML → Settings → Workspace → Create new credentials."""
    if not all(os.environ.get(key) for key in REQUIRED_ENV):
        return NullTracker()
    return ClearMLTracker(project_name, task_name)
