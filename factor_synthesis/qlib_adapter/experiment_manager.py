"""Keep Qlib file URI locks absolute, without patching installed dependencies."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from filelock import FileLock
from qlib.workflow.expm import ExpAlreadyExistError, MLflowExpManager


class LocalMLflowExpManager(MLflowExpManager):
    def _get_or_create_exp(self, experiment_id=None, experiment_name=None):
        uri = urlparse(self.uri)
        if uri.scheme != "file":
            return super()._get_or_create_exp(experiment_id, experiment_name)
        if uri.netloc not in {"", "localhost"}:
            raise ValueError("Local experiment tracking requires a local file URI")
        root = Path(url2pathname(uri.path))
        if not root.is_absolute():
            raise ValueError("Experiment tracking path must be absolute")
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / "filelock"):
            try:
                return self._get_exp(
                    experiment_id=experiment_id, experiment_name=experiment_name
                ), False
            except ValueError:
                name = experiment_name or self._default_exp_name
                try:
                    return self.create_exp(name), True
                except ExpAlreadyExistError:
                    return self._get_exp(
                        experiment_id=experiment_id, experiment_name=name
                    ), False
