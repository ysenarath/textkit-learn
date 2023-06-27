import os
from jinja2 import Template
from tklearn.core.config import Config

__all__ = [
    'Environment',
]


class Environment(object):
    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        config.setdefault(
            'DATASET_PATH',
            'dataset'
        )
        config.setdefault(
            'SQLALCHEMY_URI',
            os.path.join('sqlite:///${DATASET_PATH}', 'dataset.db')
        )
        self.config = Config(**config)

    def format(self, text: str) -> str:
        template = Template(
            str(text),
            variable_start_string='${', variable_end_string='}'
        )
        formatted = template.render(**self.config)
        if formatted != text:
            return self.format(formatted)
        return formatted

    def progress(self, iterable, total=None, desc=None, unit='it', ncols=80,
                 leave=True, file=None, **kwargs):
        try:
            from tqdm import tqdm
            return tqdm(
                iterable, total=total, desc=desc, unit=unit, ncols=ncols,
                leave=leave, file=file, **kwargs
            )
        except ImportError:
            return iterable
