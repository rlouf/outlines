import datetime
import pytest
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Tuple,
    TypedDict,
    Union
)

import interegular
from genson import SchemaBuilder
from pydantic import BaseModel

from outlines.types.utils import (
    is_bool,
    is_callable,
    is_date,
    is_dataclass,
    is_datetime,
    is_enum,
    is_float,
    is_float_instance,
    is_genson_schema_builder,
    is_int,
    is_int_instance,
    is_interegular_fsm,
    is_literal,
    is_native_dict,
    is_pydantic_model,
    is_str,
    is_str_instance,
    is_time,
    is_typed_dict,
    is_typing_dict,
    is_typing_list,
    is_typing_tuple,
    is_union
)


@pytest.fixture
def sample_enum():
    class SampleEnum(Enum):
        A = 1
        B = 2

    return SampleEnum

@pytest.fixture
def sample_class():
    class SampleClass:
        pass

    return SampleClass

@pytest.fixture
def sample_dataclass():
    @dataclass
    class SampleDataclass:
        field1: str
        field2: int

    return SampleDataclass

@pytest.fixture
def sample_typed_dict():
    class SampleTypedDict(TypedDict):
        name: str
        age: int

    return SampleTypedDict

@pytest.fixture
def sample_pydantic_model():
    class SamplePydanticModel(BaseModel):
        name: str
        age: int

    return SamplePydanticModel

@pytest.fixture
def sample_schema_builder():
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})
    return builder

@pytest.fixture
def sample_function():
    def sample_function():
        pass

    return sample_function

@pytest.fixture
def sample_interegular_fsm():
    pattern = interegular.parse_pattern(r"[a-z]{3}-[0-9]{2}")
    return pattern.to_fsm()


def test_is_int():
    assert is_int(int)
    assert not is_int(float)
    assert not is_int(1)
    assert not is_int(List[int])
    assert not is_int(Dict[int, int])
    assert is_int(Annotated[int, "some metadata"])
    assert not is_int(Annotated[str, "some metadata"])
    assert is_int(NewType("UserId", int))
    assert not is_int(NewType("UserId", str))


def test_is_int_instance():
    assert is_int_instance(1)
    assert not is_int_instance(1.0)
    assert not is_int_instance("1")
    assert not is_int_instance(int)


def test_is_float():
    assert is_float(float)
    assert not is_float(int)
    assert not is_float(1.0)
    assert not is_float(List[float])
    assert not is_float(Dict[float, float])
    assert is_float(Annotated[float, "some metadata"])
    assert not is_float(Annotated[int, "some metadata"])
    assert is_float(NewType("UserId", float))
    assert not is_float(NewType("UserId", int))


def test_is_float_instance():
    assert is_float_instance(1.0)
    assert not is_float_instance(1)
    assert not is_float_instance("1.0")
    assert not is_float_instance(float)


def test_is_str():
    assert is_str(str)
    assert not is_str(int)
    assert not is_str("hello")
    assert not is_str(List[str])
    assert not is_str(Dict[str, str])
    assert is_str(Annotated[str, "some metadata"])
    assert not is_str(Annotated[int, "some metadata"])
    assert is_str(NewType("UserId", str))
    assert not is_str(NewType("UserId", int))


def test_is_str_instance():
    assert is_str_instance("hello")
    assert is_str_instance("")
    assert is_str_instance("123")
    assert not is_str_instance(123)
    assert not is_str_instance(str)


def test_is_bool():
    assert is_bool(bool)
    assert not is_bool(int)
    assert not is_bool(True)
    assert is_bool(Annotated[bool, "some metadata"])
    assert not is_bool(Annotated[int, "some metadata"])
    assert is_bool(NewType("UserId", bool))
    assert not is_bool(NewType("UserId", int))


def test_is_datetime():
    assert is_datetime(datetime.datetime)
    assert not is_datetime(datetime.date)
    assert not is_datetime(datetime.time)
    assert not is_datetime(datetime.datetime.now())


def test_is_date():
    assert is_date(datetime.date)
    assert not is_date(datetime.datetime)
    assert not is_date(datetime.time)
    assert not is_date(datetime.date.today())


def test_is_time():
    assert is_time(datetime.time)
    assert not is_time(datetime.datetime)
    assert not is_time(datetime.date)
    assert not is_time(datetime.time(12, 30))


def test_is_native_dict():
    assert is_native_dict(dict)
    assert not is_native_dict({})
    assert not is_native_dict({"key": "value"})
    assert not is_native_dict(list)
    assert not is_native_dict(dict[str, int])


def test_is_typing_dict():
    assert is_typing_dict(dict[str, int])
    assert is_typing_dict(Dict[int, str])
    assert not is_typing_dict(dict)
    assert not is_typing_dict({})


def test_is_typing_list():
    assert is_typing_list(list[int])
    assert is_typing_list(List[int])
    assert not is_typing_list(list)
    assert not is_typing_list([])
    assert not is_typing_list(dict)


def test_is_typing_tuple():
    assert is_typing_tuple(tuple[int, str])
    assert is_typing_tuple(Tuple[int, str])
    assert not is_typing_tuple(tuple)
    assert not is_typing_tuple(())
    assert not is_typing_tuple(list)


def test_is_union():
    assert is_union(Union[int, str])
    assert is_union(Optional[int])
    assert not is_union(list)
    assert not is_union(["a", "b"])
    assert not is_union(Literal[int, str])


def test_is_literal():
    assert is_literal(Literal["a", "b"])
    assert not is_literal(str)
    assert not is_literal("a")
    assert not is_literal(["a", "b"])
    assert not is_literal(Union[str, int])


def test_is_dataclass(
    sample_dataclass,
    sample_class,
    sample_typed_dict,
    sample_pydantic_model
):
    assert is_dataclass(sample_dataclass)
    assert not is_dataclass(sample_dataclass(field1="test", field2=123))
    assert not is_dataclass(dict)
    assert not is_dataclass(sample_class)
    assert not is_dataclass(sample_typed_dict)
    assert not is_dataclass(sample_pydantic_model)


def test_is_typed_dict(
    sample_typed_dict,
    sample_class,
    sample_dataclass,
    sample_pydantic_model
):
    assert is_typed_dict(sample_typed_dict)
    assert not is_typed_dict(sample_typed_dict(name="test", age=30))
    assert not is_typed_dict(dict)
    assert not is_typed_dict(sample_class)
    assert not is_typed_dict(sample_dataclass)
    assert not is_typed_dict(sample_pydantic_model)


def test_is_pydantic_model(
    sample_pydantic_model,
    sample_class,
    sample_dataclass,
    sample_typed_dict
):
    assert is_pydantic_model(sample_pydantic_model)
    assert not is_pydantic_model(sample_pydantic_model(name="test", age=30))  # Instance
    assert not is_pydantic_model(dict)
    assert not is_pydantic_model(sample_class)
    assert not is_pydantic_model(sample_dataclass)
    assert not is_pydantic_model(sample_typed_dict)


def test_is_genson_schema_builder(
    sample_schema_builder,
    sample_class,
    sample_dataclass,
    sample_typed_dict,
    sample_pydantic_model
):
    assert is_genson_schema_builder(sample_schema_builder)
    assert not is_genson_schema_builder(dict)
    assert not is_genson_schema_builder(str)
    assert not is_genson_schema_builder({"type": 'object', "properties": {}})
    assert not is_genson_schema_builder('{"type": "object", "properties": {}}')
    assert not is_genson_schema_builder(sample_class)
    assert not is_genson_schema_builder(sample_dataclass)
    assert not is_genson_schema_builder(sample_typed_dict)
    assert not is_genson_schema_builder(sample_pydantic_model)


def test_is_enum(sample_enum):
    assert is_enum(sample_enum)
    assert not is_enum(sample_enum.A)
    assert not is_enum(dict)
    assert not is_enum(Literal["a", "b"])
    assert not is_enum(["a", "b"])


def test_is_callable(sample_function, sample_class, sample_dataclass, sample_typed_dict, sample_pydantic_model):
    assert is_callable(sample_function)
    assert is_callable(lambda x: x)
    assert not is_callable(dict)
    assert not is_callable(sample_class)
    assert not is_callable(sample_dataclass)
    assert not is_callable(sample_typed_dict)
    assert not is_callable(sample_pydantic_model)


def test_is_interegular_fsm(sample_interegular_fsm):
    assert is_interegular_fsm(sample_interegular_fsm)
    assert not is_interegular_fsm({})
    assert not is_interegular_fsm("")
