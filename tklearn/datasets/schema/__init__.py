

class BaseSchema(ObserverMixin, ObservableMixin):
    def __init__(self, type=None):  # noqa
        super(BaseSchema, self).__init__()
        self.type = type

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value
        self.observers.notify()

    def update(self, other, **kwargs) -> None:  # noqa
        if other is None:
            return
        self.type = types.result_type(self, other)

    def validate(self, data: typing.Any) -> None:
        dtype = types.from_data(data)
        if self.type == dtype:
            return
        raise ValidationError('expected \'{}\', found \'{}\''.format(self.type, dtype))

    @classmethod
    def from_data(cls, data: typing.Any) -> typing.Any:
        self = cls()
        self.type = types.from_data(data)
        return self

    def to_dict(self):
        return {
            'type': self.type,
        }

    @classmethod
    def from_dict(cls, data) -> typing.Any:
        return cls(type=data['type'])

    def copy(self):
        cls = type(self)
        clone = cls()
        clone.type = self.type
        return clone

    def notify(self, *args, **kwargs):
        self.observers.notify(*args, **kwargs)

    def __repr__(self) -> str:
        return json.dumps(dict(self.to_dict()), indent=2)


class ObjectSchema(BaseSchema):
    def __init__(self, properties=None, required=False):
        super(ObjectSchema, self).__init__('object')
        if properties is None:
            properties = {}
        self.properties.update(properties)
        if required is None:
            required = tuple()
        self.required = required

    @property
    def properties(self) -> 'PropertyMapping':
        if not hasattr(self, '_properties'):
            props = PropertyMapping()
            props.observers.attach(self)
            self._properties = props  # noqa
        return self._properties

    @property
    def required(self):
        return self._required

    @required.setter
    def required(self, value):
        self._required = tuple(value)
        self.observers.notify()

    def update(self, other, **kwargs) -> None:  # noqa
        super(ObjectSchema, self).update(other, **kwargs)
        if other is None:
            return
        for key, value in other.properties.items():
            prop = self.properties.get(key, Schema())
            prop.update(value)

    @classmethod
    def from_data(cls, data: typing.Any) -> 'ObjectSchema':
        self = super(ObjectSchema, cls).from_data(data)  # type: ObjectSchema
        for key, value in data.items():
            if key not in self.properties:
                # create new schema for property
                self.properties[key] = Schema()
            schema = cls.from_data(value)
            self.properties[key].update(schema)
        return self

    def validate(self, data: typing.Any) -> None:
        super(ObjectSchema, self).validate(data)
        for key, schema in self.properties.items():
            if key not in data:
                if self.required:
                    raise ValidationError('required attribute \'{}\' not found'.format(key))
                else:
                    continue
            value = data[key]
            schema.validate(value)

    def to_dict(self):
        result = super(ObjectSchema, self).to_dict()
        result.update(dict(
            properties=self.properties.to_dict(),
            required=list(self.required)
        ))
        return result

    @classmethod
    def from_dict(cls, data):
        self = super(ObjectSchema, cls).from_dict(data)
        self.required = data.get('required')
        self.properties.update(PropertyMapping.from_dict(data.get('properties')))
        return self

    def copy(self):
        clone = super(ObjectSchema, self).copy() # type: ObjectSchema
        clone.required = tuple(list(self.required))
        clone.properties.update(self.properties)
        return clone


class ArraySchema(BaseSchema):
    def __init__(self, items=None, required=False, min_items=0, max_items=None):
        super(ArraySchema, self).__init__('array')
        self.items = items
        self.min_items = min_items
        self.max_items = max_items

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
        if value is None:
            value = 0
        self._min_items = int(value)
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

    def update(self, other, **kwargs) -> None:  # noqa
        super(ArraySchema, self).update(other, **kwargs)
        if other is None:
            return
        if self.items is None:
            self.items = Schema()
        if other.items is not None:
            self.items.update(other.items)

    @classmethod
    def from_data(cls, data: typing.Any) -> 'ArraySchema':
        self = super(ArraySchema, cls).from_data(data)  # type: ArraySchema
        if self.items is None:
            # create new schema for items
            self.items = Schema()
        for item in data:
            if item is not None:
                schema = cls.from_data(item)
                self.items.update(schema)
        return self

    def validate(self, data: typing.Any) -> None:
        super(ArraySchema, self).validate(data)
        for item in data:
            self.items.validate(item)