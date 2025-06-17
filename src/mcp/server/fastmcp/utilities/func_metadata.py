import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    Annotated,
    Any,
    ForwardRef,
)

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, create_model
from pydantic._internal._typing_extra import eval_type_backport
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from mcp.server.fastmcp.exceptions import InvalidSignature
from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


class ArgModelBase(BaseModel):
    """A model representing the arguments to a function."""

    def model_dump_one_level(self) -> dict[str, Any]:
        """Return a dict of the model's fields, one level deep.

        That is, sub-models etc are not dumped - they are kept as pydantic models.
        """
        kwargs: dict[str, Any] = {}
        for field_name in self.__class__.model_fields.keys():
            kwargs[field_name] = getattr(self, field_name)
        return kwargs

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class FuncMetadata(BaseModel):
    # Annotated 是 Python 3.9 引入的一个类型提示工具，定义在 typing 模块中。它用于为类型添加附加信息或元数据，而不影响类型本身。
    # 在 Python 中，Annotated 是一个类型提示工具（Type Hinting Tool），用于在类型注解中附加元数据或配置。结合 Pydantic 的 ArgModelBase 和自定义的 WithJsonSchema，这行代码的作用是：定义一个类型为 ArgModelBase 子类的变量，并附加 JSON Schema 的配置信息。

    # Annotated：用于在类型注解中附加元数据（如 WithJsonSchema）。
    # type[ArgModelBase]：表示变量的类型必须是 ArgModelBase 的子类。
    # WithJsonSchema(None)：可能用于控制 JSON Schema 的生成行为（如禁用或自定义配置）。
    arg_model: Annotated[type[ArgModelBase], WithJsonSchema(None)]
    # We can add things in the future like
    #  - Maybe some args are excluded from attempting to parse from JSON
    #  - Maybe some args are special (like context) for dependency injection

    async def call_fn_with_arg_validation(
        self,
        fn: Callable[..., Any] | Awaitable[Any],
        fn_is_async: bool,
        arguments_to_validate: dict[str, Any],
        arguments_to_pass_directly: dict[str, Any] | None,
    ) -> Any:
        """Call the given function with arguments validated and injected.

        Arguments are first attempted to be parsed from JSON, then validated against
        the argument model, before being passed to the function.
        """
        arguments_pre_parsed = self.pre_parse_json(arguments_to_validate)
        arguments_parsed_model = self.arg_model.model_validate(arguments_pre_parsed)
        arguments_parsed_dict = arguments_parsed_model.model_dump_one_level()

        arguments_parsed_dict |= arguments_to_pass_directly or {}

        if fn_is_async:
            if isinstance(fn, Awaitable):
                return await fn
            return await fn(**arguments_parsed_dict)
        if isinstance(fn, Callable):
            return fn(**arguments_parsed_dict)
        raise TypeError("fn must be either Callable or Awaitable")

    def pre_parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pre-parse data from JSON.

        Return a dict with same keys as input but with values parsed from JSON
        if appropriate.

        This is to handle cases like `["a", "b", "c"]` being passed in as JSON inside
        a string rather than an actual list. Claude desktop is prone to this - in fact
        it seems incapable of NOT doing this. For sub-models, it tends to pass
        dicts (JSON objects) as JSON strings, which can be pre-parsed here.
        """
        new_data = data.copy()  # Shallow copy
        for field_name in self.arg_model.model_fields.keys():
            if field_name not in data.keys():
                continue
            if isinstance(data[field_name], str):
                try:
                    pre_parsed = json.loads(data[field_name])
                except json.JSONDecodeError:
                    continue  # Not JSON - skip
                if isinstance(pre_parsed, str | int | float):
                    # This is likely that the raw value is e.g. `"hello"` which we
                    # Should really be parsed as '"hello"' in Python - but if we parse
                    # it as JSON it'll turn into just 'hello'. So we skip it.
                    continue
                new_data[field_name] = pre_parsed
        assert new_data.keys() == data.keys()
        return new_data

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


def func_metadata(func: Callable[..., Any], skip_names: Sequence[str] = ()) -> FuncMetadata:
    """Given a function, return metadata including a pydantic model representing its
    signature.

    The use case for this is
    ```
    meta = func_to_pyd(func)
    validated_args = meta.arg_model.model_validate(some_raw_data_dict)
    return func(**validated_args.model_dump_one_level())
    ```

    **critically** it also provides pre-parse helper to attempt to parse things from
    JSON.

    Args:
        func: The function to convert to a pydantic model
        skip_names: A list of parameter names to skip. These will not be included in
            the model.
    Returns:
        A pydantic model representing the function's signature.
    """
    sig = _get_typed_signature(func)
    params = sig.parameters
    dynamic_pydantic_model_params: dict[str, Any] = {}
    globalns = getattr(func, "__globals__", {})
    for param in params.values():
        if param.name.startswith("_"):
            raise InvalidSignature(f"Parameter {param.name} of {func.__name__} cannot start with '_'")
        if param.name in skip_names:
            continue
        annotation = param.annotation

        # `x: None` / `x: None = None`
        if annotation is None:
            annotation = Annotated[
                None,
                Field(default=param.default if param.default is not inspect.Parameter.empty else PydanticUndefined),
            ]

        # Untyped field
        if annotation is inspect.Parameter.empty:
            annotation = Annotated[
                Any,
                Field(),
                # 🤷
                WithJsonSchema({"title": param.name, "type": "string"}),
            ]

        field_info = FieldInfo.from_annotated_attribute(
            _get_typed_annotation(annotation, globalns),
            param.default if param.default is not inspect.Parameter.empty else PydanticUndefined,
        )
        dynamic_pydantic_model_params[param.name] = (field_info.annotation, field_info)
        continue

    arguments_model = create_model(
        f"{func.__name__}Arguments",
        **dynamic_pydantic_model_params,
        __base__=ArgModelBase,
    )
    resp = FuncMetadata(arg_model=arguments_model)
    return resp

# 描述: 解析类型注解，考虑可能的前向引用。
# 参数: annotation 是类型注解，globalns 是全局命名空间。
def _get_typed_annotation(annotation: Any, globalns: dict[str, Any]) -> Any:
    def try_eval_type(value: Any, globalns: dict[str, Any], localns: dict[str, Any]) -> tuple[Any, bool]:
        try:
            return eval_type_backport(value, globalns, localns), True
        except NameError:
            return value, False

    if isinstance(annotation, str):
        # 处理字符串类型注解: 如果类型注解是字符串（表示前向引用），将其转为 ForwardRef。
        annotation = ForwardRef(annotation)
        annotation, status = try_eval_type(annotation, globalns, globalns)

        # This check and raise could perhaps be skipped, and we (FastMCP) just call
        # model_rebuild right before using it 🤷
        if status is False:
            raise InvalidSignature(f"Unable to evaluate type annotation {annotation}")

    return annotation


# 前向引用类型注解是string串，标识具体的类类型
# demo：
# class User:
#     pass
# def create_user(user: 'User') -> bool:
#     return True
#
# # 使用 _get_typed_signature 解析签名
# signature = _get_typed_signature(create_user)
# print(signature)  # 打印解析后的签名
# 在这个示例中，create_user 函数的参数使用了前向引用类型注解 'User'。方法 _get_typed_signature 将解析注解，将 'User' 解析为实际的 User 类，返回的签名中包含了这个解析后的信息。

# 这个方法的作用是获取一个函数的签名，同时解析任何前向引用的类型注解。
def _get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    """Get function signature while evaluating forward references"""
    # 使用 inspect.signature(call) 获取函数的签名
    signature = inspect.signature(call)
    # globalns 用于存储函数的全局命名空间，通常是函数定义所在模块的全局变量。
    globalns = getattr(call, "__globals__", {})
    # 遍历签名中的参数，为每个参数创建一个新的 inspect.Parameter 对象。
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            # 使用 _get_typed_annotation 方法解析参数的类型注解
            annotation=_get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]
    # 使用解析后的参数创建一个新的 Signature 对象
    typed_signature = inspect.Signature(typed_params)
    return typed_signature
