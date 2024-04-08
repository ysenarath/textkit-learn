import datetime as dt

import numpy as np
from tklearn.base.arrow.types import MonthDayNano, from_dtype, infer_type

if __name__ == "__main__":
    # none
    print(from_dtype(type(None)))
    print(infer_type(None))
    # bool
    print(from_dtype(np.dtype("bool")))
    print(infer_type(True))
    # int8
    print(from_dtype(np.dtype("int8")))
    print(infer_type(-128))
    # int16
    print(from_dtype(np.dtype("int16")))
    print(infer_type(-32767))
    # int32
    print(from_dtype(np.dtype("int32")))
    print(infer_type(-2147483648))
    # int64
    print(from_dtype(np.dtype("int64")))
    print(infer_type(-9223372036854775808))
    # uint8
    print(from_dtype(np.dtype("uint8")))
    print(infer_type(0))
    # uint16
    print(from_dtype(np.dtype("uint16")))
    print(infer_type(256))
    # uint32
    print(from_dtype(np.dtype("uint32")))
    print(infer_type(65536))
    # uint64
    print(from_dtype(np.dtype("uint64")))
    print(infer_type(4294967296))
    # float16
    print(from_dtype(np.dtype("float16")))
    print(infer_type(1.0))
    # float32
    print(from_dtype(np.dtype("float32")))
    print(infer_type(np.finfo("float32").max))
    # float64
    print(from_dtype(np.dtype("float64")))
    print(infer_type(np.finfo("float64").max))
    # time32
    print(infer_type(dt.time()))
    # time64
    print(from_dtype(dt.time))
    print(infer_type(dt.time(microsecond=1)))
    # timestamp
    print(from_dtype(dt.datetime))
    print(infer_type(dt.datetime(2021, 1, 1)))
    # date32
    print(from_dtype(dt.date))
    print(infer_type(dt.date(2021, 1, 1)))
    # duration
    print(from_dtype(dt.timedelta))
    print(infer_type(dt.timedelta(microseconds=1)))
    # month_day_nano_interval
    print(from_dtype(MonthDayNano))
    print(infer_type(MonthDayNano(1, 1, 1)))
    # binary
    print(from_dtype(np.dtype("S")))
    print(infer_type(b""))
    # string/utf8
    print(from_dtype(np.dtype("U")))
    print(infer_type(""))
    # list
    # from typing import Any, List, Optional, Union
    # print(from_pytype(List[int]))
    test = infer_type([])  # empty list
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([1, 2, 3])  # int list
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([1.0, 2.0, 3.0])  # float list
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([True, False, True])
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([None, None, None])
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([None, 1, 2, 3])
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    test = infer_type([1, np.finfo("float64").max, True])  # mixed list
    print(test)  # optional lists
    print("|-> nullable?", test.field(0).nullable)
    # struct
    test = infer_type({})
    print(test)
    test = infer_type({"a": 1, "b": 2})
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type({"a": 1, "b": 2.0})
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type({"a": 1, "b": True})
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type({"a": {"x": 1, "y": 2}, "b": {"x": 1.0, "y": 2.0}})
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
    # array of structs
    test = infer_type([
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ])
    print(test)
    for field in test.value_type:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type([
        {"a": 1, "b": 2},
        {"a": 3, "b": 4.0},
    ])
    print(test)
    for field in test.value_type:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type([
        {"b": 2},
        {"a": 3, "b": True},
    ])
    print(test)
    for field in test.value_type:
        print("|-> ", field.name, field.type, field.nullable)
    test = infer_type([
        {"a": 1, "b": 2},
        {"a": 3, "b": None},
    ])
    print(test)
    for field in test.value_type:
        print("|-> ", field.name, field.type, field.nullable)
    # deeply nested
    test = infer_type({
        "a": [
            {"b": 1, "c": 2},
            {"c": 4},
        ],
        "d": [
            {"e": 1, "f": 2},
            {"e": 3, "f": None},
        ],
    })
    print(test)
    for field in test:
        print("|-> ", field.name, field.type, field.nullable)
        for subfield in field.type.value_type:
            print("|    |--> ", subfield.name, subfield.type, subfield.nullable)
