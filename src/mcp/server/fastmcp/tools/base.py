from __future__ import annotations as _annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_origin

from pydantic import BaseModel, Field

from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata

if TYPE_CHECKING:
    from mcp.server.fastmcp.server import Context
    from mcp.server.session import ServerSessionT
    from mcp.shared.context import LifespanContextT


class Tool(BaseModel):
    """Internal tool registration info."""

    fn: Callable[..., Any] = Field(exclude=True)
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: dict[str, Any] = Field(description="JSON schema for tool parameters")
    fn_metadata: FuncMetadata = Field(
        description="Metadata about the function including a pydantic model for tool"
        " arguments"
    )
    is_async: bool = Field(description="Whether the tool is async")
    context_kwarg: str | None = Field(
        None, description="Name of the kwarg that should receive context"
    )

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        context_kwarg: str | None = None,
    ) -> Tool:
        """Create a Tool from a function."""
        from mcp.server.fastmcp import Context

        func_name = name or fn.__name__

        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        func_doc = description or fn.__doc__ or ""
        # inspect 模块是 Python 的一个标准库模块，用于获取关于活跃对象的信息，比如模块、类、方法、函数、回溯、帧对象、代码对象等。它提供了多种工具来分析 Python 代码的运行时状态。
        # inspect.isclass(obj): 检查对象是否是类。
        # inspect.ismethod(obj): 检查对象是否是方法。
        # inspect.iscoroutinefunction(): 用于检查 fn 是否为一个异步函数
        is_async = inspect.iscoroutinefunction(fn)

        if context_kwarg is None:
            # inspect.signature(fn) 可以获取函数的签名，包括参数名称和默认值等。
            # def example_func(x, y=10):
            #     pass
            # sig = inspect.signature(example_func)
            # print(sig)  # 输出: (x, y=10)
            sig = inspect.signature(fn)
            # 可以通过签名对象 sig.parameters 获取函数参数的详细信息。
            for param_name, param in sig.parameters.items():
                # param.annotation 返回的是函数参数的类型注解
                # import inspect
                # def example_func(x: int, y: str = "default") -> bool:
                #     return True
                # # 获取函数签名
                # sig = inspect.signature(example_func)
                # for param_name, param in sig.parameters.items():
                #     print(f"Parameter: {param_name}, Annotation: {param.annotation}")
                # Parameter: x, Annotation: <class 'int'>
                # Parameter: y, Annotation: <class 'str'>

                # get_origin 的主要作用是处理更复杂的类型注解，尤其是在涉及泛型类型（generic types）时。它可以帮助识别类型提示中的基础类型，而忽略具体的类型参数。这在处理像 List[int]、Dict[str, Any]、Union[int, str] 等复合类型时特别有用。
                # from typing import get_origin, List, Dict, Union
                # # 示例 1: 列表类型的处理
                # list_type = List[int]
                # origin_list = get_origin(list_type)
                # print(f"The origin of List[int] is: {origin_list}")  # 输出: The origin of List[int] is: <class 'list'>
                # # 示例 2: 字典类型的处理
                # dict_type = Dict[str, int]
                # origin_dict = get_origin(dict_type)
                # print(f"The origin of Dict[str, int] is: {origin_dict}")  # 输出: The origin of Dict[str, int] is: <class 'dict'>
                # # 示例 3: 联合类型
                # union_type = Union[int, str]
                # origin_union = get_origin(union_type)
                # print(f"The origin of Union[int, str] is: {origin_union}")  # 输出: The origin of Union[int, str] is: <class 'types.UnionType'>
                # # 示例 4: 非泛型类型处理
                # basic_type = int
                # origin_basic = get_origin(basic_type)
                # print(f"The origin of int is: {origin_basic}")  # 输出: The origin of int is: None
                if get_origin(param.annotation) is not None:
                    continue
                if issubclass(param.annotation, Context):
                    context_kwarg = param_name
                    break

        func_arg_metadata = func_metadata(
            fn,
            skip_names=[context_kwarg] if context_kwarg is not None else [],
        )
        parameters = func_arg_metadata.arg_model.model_json_schema()

        return cls(
            fn=fn,
            name=func_name,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            context_kwarg=context_kwarg,
        )

    async def run(
        self,
        arguments: dict[str, Any],
        context: Context[ServerSessionT, LifespanContextT] | None = None,
    ) -> Any:
        """Run the tool with arguments."""
        try:
            return await self.fn_metadata.call_fn_with_arg_validation(
                self.fn,
                self.is_async,
                arguments,
                {self.context_kwarg: context}
                if self.context_kwarg is not None
                else None,
            )
        except Exception as e:
            raise ToolError(f"Error executing tool {self.name}: {e}") from e
