from tklearn.datasets.array import DocumentArray
from tklearn.utils.observable import ObserverMixin

__all__ = [
    'Dataset',
]


class Dataset(ObserverMixin, DocumentArray):
    """
    A more or less complete user-defined wrapper around DocumentArray objects.
    """

    def __init__(self):
        super(Dataset, self).__init__()
        # for root group only attach it to sync the schema
        self._schema.observers.attach(self)
        self.notify()

    def notify(self, *args, **kwargs):
        # schema has been updated
        self._group.attrs['schema'] = self._schema.to_dict()
