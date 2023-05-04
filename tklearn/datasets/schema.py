import json
import typing
from collections.abc import MutableMapping

from tklearn.datasets import types
from tklearn.datasets.types import pytypes
from tklearn.exceptions import ValidationError
from tklearn.utils.observable import ObservableMixin, ObserverMixin

__all__ = [
    'Schema',
    'PropertyMapping',
]

type_ = type


class PropertyMapping(ObserverMixin, ObservableMixin, MutableMapping):
    def __init__(self):  # noqa
        super(PropertyMapping, self).__init__()
        super(ObservableMixin, self).__init__()
        self._props = {}

    def __getitem__(self, key: str) -> 'Schema':
        return self._props[key]

    def __setitem__(self, key: str, value: 'Schema'):
        # copies the object
        value = value.copy()
        value.observers.attach(self)
        self._props[key] = value
        self.observers.notify()

    def __delitem__(self, key) -> None:
        del self._props[key]
        self.observers.notify()

    def __iter__(self) -> typing.Generator[str, None, None]:
        for key in self._props:
            yield key

    def __len__(self) -> int:
        return len(self._props)

    def notify(self, *args, **kwargs):
        self.observers.notify(*args, **kwargs)

    def to_dict(self):
        return {key: value.to_dict() for key, value in self.items()}

    @classmethod
    def from_dict(cls, data):
        if data is None:
            data = {}
        self = cls()
        for key, value in data.items():
            prop = Schema.from_dict(data=value)
            self[key] = prop
        return self

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class Schema(ObserverMixin, ObservableMixin):
    def __init__(self, type=None, properties=None, items=None, required=None,  # noqa
                 min_items=None, max_items=None, pytype=None):
        super(Schema, self).__init__()
        self.type = type
        if properties is None:
            properties = {}
        self.properties.update(properties)
        self.items = items
        self.min_items = min_items
        self.max_items = max_items
        self.required = required
        self.pytype = pytype

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value
        self.observers.notify()

    @property
    def properties(self) -> 'PropertyMapping':
        if not hasattr(self, '_properties'):
            props = PropertyMapping()
            props.observers.attach(self)
            self._properties = props  # noqa
        return self._properties

    @property
    def items(self) -> typing.Optional['Schema']:
        return self._items

    @items.setter
    def items(self, items: 'Schema'):
        if items is None:
            self._items = None
            return
        items = items.copy()
        items.observers.attach(self)
        self._items = items
        self.observers.notify()

    @property
    def min_items(self):
        return self._min_items

    @min_items.setter
    def min_items(self, value):
        if value is not None:
            value = int(value)
        self._min_items = value
        self.observers.notify()

    @property
    def max_items(self):
        return self._max_items

    @max_items.setter
    def max_items(self, value):
        if value is not None:
            value = int(value)
        self._max_items = value
        self.observers.notify()

    @property
    def pytype(self) -> typing.Optional[types.PyType]:
        if self._pytype is None:
            return None
        return pytypes.get(self._pytype)

    @pytype.setter
    def pytype(self, value: typing.Union[str, types.PyType, type_]):
        if not isinstance(value, types.PyType):
            value = pytypes.get(value)
        if value is None:
            self._pytype = None
        else:
            self._pytype = value.name
        self.observers.notify()

    @property
    def dtype(self) -> str:
        if self.type == 'array':
            return self.items.dtype
        return self.type

    @property
    def shape(self) -> typing.Tuple[typing.Optional[int], ...]:
        # is shape does not have None => array can be converted to uniform multidimensional array
        if self.type != 'array':
            raise ValueError('can\'t calculate the shape of an \'{}\''.format(self.type))
        if self.items is None:
            # unknown length shape
            return None,
        if self.max_items is not None and \
                self.min_items is not None and \
                self.max_items != self.min_items:
            # variable length shape
            return None,
        try:
            return (self.max_items,) + self.items.shape
        except ValueError as _:
            return (self.max_items,)

    @property
    def required(self):
        # for non object types required is ignored
        # it will be None as long as type is not object
        if self.type != 'object':
            return None
        # for objects None => tuple
        if self._required is None:
            return tuple()
        return self._required

    @required.setter
    def required(self, value):
        if self.type != 'object' and value is not None:
            raise ValidationError(
                '\'required\' should be \'None\' for non \'object\' type, found {}'.format(value)
            )
        if self.type == 'object' and value is None:
            value = tuple()
        elif value is not None:
            value = tuple(value)
        self._required = value
        self.observers.notify()

    def __getitem__(self, key):
        if key is None:
            return self.copy()
        if self.type == 'object':
            suffix = None
            if '/' in key:
                key, suffix = key.split('/', maxsplit=1)
            return self.properties[key][suffix]
        elif self.type == 'array':
            schema = self.copy()
            schema.items = self.items[key]
            return schema
        raise KeyError(key)

    def __setitem__(self, key, value):
        self[key].update(value, override=True)

    def notify(self, *args, **kwargs):
        self.observers.notify(*args, **kwargs)

    def update(self, other, override=False, **kwargs) -> None:  # noqa
        if other is None:
            return
        # self.type can be None
        # - e.g., updating a (non initialized) schema from other
        if override:
            dtype = types.result_type(other)
            self.pytype = other.pytype
        else:
            try:
                dtype = types.result_type(self, other)
            except TypeError as ex:
                # failing type match stops adding the document
                raise ValidationError(ex.args[0]) from ex
        self.type = dtype
        if self.pytype is None and other.pytype is not None:
            # copies pytype from other if self.pytype is None
            self.pytype = other.pytype
        if self.type == 'object':
            if override:
                self.properties.clear()
            for key, other_prop in other.properties.items():
                if key not in self.properties:
                    self.properties[key] = Schema()
                self.properties[key].update(other_prop, override=override)
        elif self.type == 'array':
            if self.items is None or override:
                self.items = Schema()
            if other.items is not None:
                self.items.update(other.items, override=override)
            try:
                max_items = max(item for item in [self.max_items, other.max_items] if item is not None)
                self.max_items = max_items
            except ValueError as _:
                pass
            try:
                min_items = min(item for item in [self.min_items, other.min_items] if item is not None)
                self.min_items = min_items
            except ValueError as _:
                pass

    @classmethod
    def from_data(cls, data: typing.Any) -> 'Schema':
        self = Schema()
        self.pytype = type(data)
        if self.pytype is not None:
            data = self.pytype.encode(data)
        self.type = types.from_data(data)
        if self.type == 'object':
            for key, value in data.items():
                if key not in self.properties:
                    # create new schema for property
                    self.properties[key] = Schema()
                schema = cls.from_data(value)
                self.properties[key].update(schema)
        elif self.type == 'array':
            if self.items is None:
                # create new schema for items
                self.items = Schema()
            for item in data:
                if item is not None:
                    schema = cls.from_data(item)
                    self.items.update(schema)
            self.max_items = len(data)
            self.min_items = len(data)
        return self

    def validate(self, data: typing.Any) -> None:
        if self.pytype is not None:
            data = self.pytype.decode(data)
        dtype = types.from_data(data)
        if data is None:
            return
        if self.type == 'object':
            if dtype != 'object':
                raise ValidationError('expected \'object\', found \'{}\''.format(dtype))
            for key, schema in self.properties.items():
                if key not in data:
                    if self.required:
                        raise ValidationError('required attribute \'{}\' not found'.format(key))
                    else:
                        continue
                value = data[key]
                schema.validate(value)
        elif self.type == 'str':
            if dtype != 'str':
                raise ValidationError('expected \'str\', found \'{}\''.format(dtype))
        elif self.type == 'array':
            if dtype != 'array':
                raise ValidationError('expected \'array\', found \'{}\''.format(dtype))
            if self.max_items is not None and len(data) > self.max_items:
                raise ValidationError('number of items in array is higher than maximum number of values')
            if self.min_items is not None and len(data) < self.min_items:
                raise ValidationError('number of items in array is lower than minimum number of values')
            for item in data:
                self.items.validate(item)
        else:
            if dtype != self.type:
                raise ValidationError('expected \'{}\', found \'{}\''.format(self.type, dtype))

    def normalize(self, data: typing.Any, return_schema=False, validate=True) -> (typing.Any, 'Schema'):
        if validate:
            self.validate(data)
        normalize = True
        if self.pytype is not None:
            normalize = self.pytype.normalize
            # if pytype is defined stop normalization
            data = self.pytype.encode(data)
        if normalize and self.type == 'object':
            if data is None:
                data = {}
            schema = Schema()
            schema.type = 'object'
            result = {}
            for key, prop in self.properties.items():
                value = data.get(key)
                value, prop = prop.normalize(value, return_schema=True, validate=False)
                if prop.type == 'object':
                    for k, p in prop.properties.items():
                        kk = '{}/{}'.format(key, k)
                        result[kk] = value.pop(k)
                        schema.properties[kk] = p
                else:
                    result[key] = value
                    schema.properties[key] = prop.copy()
        elif normalize and self.type == 'array':
            if data is None:
                data = []
            items, dtype = [], None
            for item in data:
                item, schema = self.items.normalize(item, return_schema=True, validate=False)
                if dtype is None:
                    dtype = schema
                else:
                    dtype.update(schema)
                items.append(item)
            else:
                _, dtype = self.items.normalize(None, return_schema=True, validate=False)
            if dtype is not None and dtype.type == 'object':
                result = {}
                schema = Schema()
                schema.type = 'object'
                for key, prop in dtype.properties.items():
                    result[key] = []
                    for item in items:
                        result[key].append(item.get(key))
                    schema.properties[key] = Schema()
                    schema.properties[key].type = 'array'
                    schema.properties[key].items = prop
            else:
                result = items
                schema = Schema()
                schema.type = 'array'
                schema.items = dtype
        else:
            result, schema = data, self.copy()
        if return_schema:
            return result, schema
        return result

    def denormalize_level(self, nested_list, level):
        """
        Recursively denormalizes the specified level of a nested list.

        Parameters
        ----------
        nested_list : list
            The nested list to denormalize.
        level : int
            The level at which to denormalize.

        Returns
        -------
        list
            A denormalized version of the nested list.

        Notes
        -----
        This function creates a new nested list that contains the same values
        as the input nested list, but with the specified level denormalized. If
        `nested_list` is None, this function returns None.
        """
        if nested_list is None:
            return None
        if level == 0:
            result = [
                item for item in nested_list
            ]
        else:
            result = [
                self.denormalize_level(item, level - 1)
                for item in nested_list
            ]
        return result

    def denormalize(self, data, key=None, level=0):
        normalize = self.pytype.normalize if self.pytype is not None else True
        if normalize and self.type == 'object':
            result = {}
            base = key
            for key, prop in self.properties.items():
                if base is None:
                    result[key] = prop.denormalize(data, key, level=level)
                else:
                    result[key] = prop.denormalize(data, '{}/{}'.format(base, key), level=level)
        elif normalize and self.type == 'array':
            items = self.items.denormalize(data, key=key, level=level + 1)
            result = self.denormalize_level(items, level=level)
        else:
            try:
                result = data if key is None else data[key]
            except KeyError as ex:
                if self.required:
                    raise ex
                result = None
        if self.pytype is not None:
            result = self.pytype.decode(result)
        return result

    def to_dict(self) -> dict[str, typing.Union[str, dict]]:
        result = {
            'type': self.type,
        }
        if self.pytype is not None:
            result['pytype'] = self.pytype.name
        if self.type == 'object':
            result['properties'] = self.properties.to_dict()
            if self.required is not None:
                result['required'] = list(self.required)
        elif self.type == 'array':
            if self.items is not None:
                result['items'] = self.items.to_dict()
            if self.min_items is not None:
                result['minItems'] = self.min_items
            if self.max_items is not None:
                result['maxItems'] = self.max_items
        return result

    @classmethod
    def from_dict(cls, data):
        if data is None:
            data = {}
        self = cls()
        if 'type' in data:
            self.type = data['type']
        if 'pytype' in data:
            self.pytype = data['pytype']
        if self.type == 'object':
            if 'required' in data:
                self.required = data['required']
            if 'properties' in data:
                prop_data = data['properties']
                prop = PropertyMapping.from_dict(prop_data)
                self.properties.update(prop)
        elif self.type == 'array':
            if 'items' in data:
                items_data = data['items']
                self.items = Schema.from_dict(items_data)
            if 'minItems' in data:
                self.min_items = data['minItems']
            if 'maxItems' in data:
                self.max_items = data['maxItems']
        return self

    def copy(self):
        return self.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return json.dumps(dict(self.to_dict()), indent=2)
