from collections.abc import Mapping
import typing

from pydantic import BaseModel

__all__ = ['Document']


class Document(BaseModel, Mapping):
    """Document."""
    id: typing.Optional[typing.Text]
    props: typing.Dict[typing.Text, typing.Any]

    def __getattr__(self, attr) -> typing.Any:
        if attr not in self.props:
            raise AttributeError(f'attribute \'{attr}\' not found')
        return self.props[attr]

    def __getitem__(self, key) -> typing.Any:
        return self.props[key]

    def __len__(self) -> int:
        return len(self.props)
