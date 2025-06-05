"""FastMCP - A more ergonomic interface for MCP servers."""

from __future__ import annotations as _annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
)
from itertools import chain
from typing import Any, Generic, Literal

import anyio
import pydantic_core
from pydantic import BaseModel, Field
from pydantic.networks import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.prompts import Prompt, PromptManager
from mcp.server.fastmcp.resources import FunctionResource, Resource, ResourceManager
from mcp.server.fastmcp.tools import ToolManager
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger
from mcp.server.fastmcp.utilities.types import Image
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import LifespanResultT
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.lowlevel.server import lifespan as default_lifespan
from mcp.server.session import ServerSession, ServerSessionT
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import (
    AnyFunction,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    TextContent,
)
from mcp.types import Prompt as MCPPrompt
from mcp.types import PromptArgument as MCPPromptArgument
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from mcp.types import Tool as MCPTool

logger = get_logger(__name__)


# BaseSettings核心作用
# 配置管理:
#   BaseSettings 提供了一种结构化的方式来管理应用程序的配置。通过定义一个继承自 BaseSettings 的数据模型，开发者可以将所有相关的配置项集中管理。
# 环境变量加载:
#   通过 BaseSettings 定义的模型可以自动从环境变量中加载数据。当应用程序启动时，它会自动读取环境变量，并将其映射到模型的属性上。
# 类型验证:
#   BaseSettings 继承自 pydantic.BaseModel，因此可以利用 Pydantic 的数据验证功能。它会在加载配置的同时进行类型检查，确保配置数据符合预期的类型。
# 默认值和自定义解析:
#   可以为模型中的字段提供默认值。对于复杂的数据类型或自定义解析需求，可以使用 Pydantic 的字段验证器进行处理。

# Settings 类继承了 BaseSettings 和 Generic。BaseSettings 用于提供配置管理功能，Generic[LifespanResultT] 用于支持泛型，使得 Settings 可以适应不同的生命周期结果类型 LifespanResultT。
class Settings(BaseSettings, Generic[LifespanResultT]):
    """FastMCP server settings.

    All settings can be configured via environment variables with the prefix FASTMCP_.
    For example, FASTMCP_DEBUG=true will set debug=True.
    """

    # SettingsConfigDict 是用于配置 BaseSettings 行为的一个配置字典。
    # 这一配置允许你定义 BaseSettings 类如何从环境变量.env 文件等外部源加载配置。
    # 通过配置这个字典，你可以指定一些加载配置的规则和行为。
    model_config = SettingsConfigDict(
        # 这个选项指定了环境变量的前缀。在加载配置时，BaseSettings 会查找以 FASTMCP_ 开头的环境变量。
        # 如果有一个属性 debug，它会尝试从环境变量 FASTMCP_DEBUG 中加载值
        env_prefix="FASTMCP_",
        # 指定 .env 文件的名称，该文件用于本地开发环境中存储环境变量
        # BaseSettings 会自动从这个文件中加载变量，模拟环境变量的存在。
        env_file=".env",
        # 这个选项控制如何处理未在模型中定义的额外字段。
        # 设置为 "ignore" 时，BaseSettings 会忽略任何未在类中明确定义的环境变量或 .env 文件中的变量。
        # 这对于确保只加载和处理预期的配置项是有用的。
        extra="ignore",
    )

    # Server settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # HTTP settings
    host: str = "0.0.0.0"
    port: int = 8000
    sse_path: str = "/sse"
    message_path: str = "/messages/"

    # resource settings
    # 指示是否在检测到重复资源时发出警告，默认值为 True
    warn_on_duplicate_resources: bool = True

    # tool settings
    # 指示是否在检测到重复工具时发出警告，默认值为 True
    warn_on_duplicate_tools: bool = True

    # prompt settings
    # 指示是否在检测到重复提示词发出警告，默认值为 True
    warn_on_duplicate_prompts: bool = True

    # 一个字符串列表，表示需要安装在服务器环境中的依赖项。使用 default_factory 来初始化为空列表，并提供一个描述。
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of dependencies to install in the server environment",
    )

    # 一个可调用对象，接受 FastMCP 实例并返回 AbstractAsyncContextManager[LifespanResultT] 类型的上下文管理器。用于管理应用程序生命周期，默认值为 None，并提供了描述。
    lifespan: (
        Callable[[FastMCP], AbstractAsyncContextManager[LifespanResultT]] | None
    ) = Field(None, description="Lifespan context manager")


def lifespan_wrapper(
    # 这是一个应用程序的实例，代表你的应用程序上下文
    app: FastMCP,
    # 一个可调用对象，接受 FastMCP 实例并返回一个异步上下文管理器，这个上下文管理器可以管理应用程序的生命周期
    # AbstractAsyncContextManager这是一个抽象基类，定义了异步上下文管理器的接口。任何实现了 __aenter__ 和 __aexit__ 方法的类都可以被视为异步上下文管理器。
    # 其目的是在进入和退出某个上下文时执行特定的异步操作，例如资源的分配和释放。
    lifespan: Callable[[FastMCP], AbstractAsyncContextManager[LifespanResultT]],
) -> Callable[[MCPServer[LifespanResultT]], AbstractAsyncContextManager[object]]:
    # wrap 函数使用 @asynccontextmanager 装饰器，定义了一个异步上下文管理器。
    # 当 wrap 被调用时，它会执行 lifespan(app)，这将返回一个上下文管理器。
    # async with 语句用于异步地进入和退出这个上下文管理器。
    # yield context 暴露了这个上下文，使得调用者可以在这个上下文中执行相应的操作。
    @asynccontextmanager
    # asynccontextmanager这个装饰器把 wrap 函数转换成一个新的 异步上下文管理器 。尽管 wrap 函数内部使用了异步生成器（通过 yield），
    # @asynccontextmanager 装饰器实际上将其转换为符合 AbstractAsyncContextManager 接口的对象。这意味着 wrap 在外部看来是一个异步上下文管理器。
    async def wrap(s: MCPServer[LifespanResultT]) -> AsyncIterator[object]:
        # AsyncIterator 是异步版本的迭代器，定义了 __aiter__ 和 __anext__ 方法。它允许对象在异步上下文中被迭代。
        # 在 wrap 函数中，yield context 实现了一个 AsyncIterator 的行为，允许在 async with 语句中利用 lifespan 提供的上下文。
        async with lifespan(app) as context:
            # yield 的作用
            # 暂停和恢复函数执行:
            #   当函数执行到 yield 语句时，它会暂停执行并返回一个值给调用者。在后续的迭代或上下文管理中，可以从暂停的地方继续执行。
            # 作为上下文管理器的一部分:
            #   在使用 @asynccontextmanager 装饰的函数中，yield 用来分隔 __aenter__ 和 __aexit__ 的逻辑。
            #   在 async with 语句中，函数执行到 yield 时，相当于进入了上下文管理器，并返回了 yield 后面的值作为上下文。
            # 提供上下文给调用者:
            #   在上下文管理器中，yield 后面的值（在你的代码中是 context）被传递给外部执行此上下文管理器的 async with 块内的执行代码。这允许调用者在上下文内执行操作。
            yield context

    return wrap


class FastMCP:
    def __init__(
        # **settings 是一种用于函数定义中的参数语法，称为“关键字参数包”。
            # 它允许函数接受任意数量的关键字参数，并将这些参数以字典的形式传递给函数内部。
            # 这种语法对于处理动态参数特别有用，尤其是在不知道具体会有哪些参数需要传递的情况下。
        self, name: str | None = None, instructions: str | None = None, **settings: Any
    ):
        # 在你提供的代码片段中，self.settings = Settings(**settings) 使用了 **settings 来将接收到的关键字参数转发给 Settings 类的构造函数。
        self.settings = Settings(**settings)

        self._mcp_server = MCPServer(
            name=name or "FastMCP",
            instructions=instructions,
            lifespan=lifespan_wrapper(self, self.settings.lifespan)
            if self.settings.lifespan
            else default_lifespan,
        )
        self._tool_manager = ToolManager(
            warn_on_duplicate_tools=self.settings.warn_on_duplicate_tools
        )
        self._resource_manager = ResourceManager(
            warn_on_duplicate_resources=self.settings.warn_on_duplicate_resources
        )
        self._prompt_manager = PromptManager(
            warn_on_duplicate_prompts=self.settings.warn_on_duplicate_prompts
        )
        self.dependencies = self.settings.dependencies

        # Set up MCP protocol handlers
        self._setup_handlers()

        # Configure logging
        configure_logging(self.settings.log_level)

    @property
    def name(self) -> str:
        return self._mcp_server.name

    @property
    def instructions(self) -> str | None:
        return self._mcp_server.instructions

    def run(self, transport: Literal["stdio", "sse"] = "stdio") -> None:
        """Run the FastMCP server. Note this is a synchronous function.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
        """
        TRANSPORTS = Literal["stdio", "sse"]
        if transport not in TRANSPORTS.__args__:  # type: ignore
            raise ValueError(f"Unknown transport: {transport}")

        if transport == "stdio":
            anyio.run(self.run_stdio_async)
        else:  # transport == "sse"
            anyio.run(self.run_sse_async)

    def _setup_handlers(self) -> None:
        """Set up core MCP protocol handlers."""
        self._mcp_server.list_tools()(self.list_tools)
        self._mcp_server.call_tool()(self.call_tool)
        self._mcp_server.list_resources()(self.list_resources)
        self._mcp_server.read_resource()(self.read_resource)
        self._mcp_server.list_prompts()(self.list_prompts)
        self._mcp_server.get_prompt()(self.get_prompt)
        self._mcp_server.list_resource_templates()(self.list_resource_templates)

    async def list_tools(self) -> list[MCPTool]:
        """List all available tools."""
        tools = self._tool_manager.list_tools()
        return [
            MCPTool(
                name=info.name,
                description=info.description,
                inputSchema=info.parameters,
            )
            for info in tools
        ]

    def get_context(self) -> Context[ServerSession, object]:
        """
        Returns a Context object. Note that the context will only be valid
        during a request; outside a request, most methods will error.
        """
        try:
            request_context = self._mcp_server.request_context
        except LookupError:
            request_context = None
        return Context(request_context=request_context, fastmcp=self)

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call a tool by name with arguments."""
        context = self.get_context()
        result = await self._tool_manager.call_tool(name, arguments, context=context)
        converted_result = _convert_to_content(result)
        return converted_result

    async def list_resources(self) -> list[MCPResource]:
        """List all available resources."""

        resources = self._resource_manager.list_resources()
        return [
            MCPResource(
                uri=resource.uri,
                name=resource.name or "",
                description=resource.description,
                mimeType=resource.mime_type,
            )
            for resource in resources
        ]

    async def list_resource_templates(self) -> list[MCPResourceTemplate]:
        templates = self._resource_manager.list_templates()
        return [
            MCPResourceTemplate(
                uriTemplate=template.uri_template,
                name=template.name,
                description=template.description,
            )
            for template in templates
        ]

    async def read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        """Read a resource by URI."""

        resource = await self._resource_manager.get_resource(uri)
        if not resource:
            raise ResourceError(f"Unknown resource: {uri}")

        try:
            content = await resource.read()
            return [ReadResourceContents(content=content, mime_type=resource.mime_type)]
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise ResourceError(str(e))

    def add_tool(
        self,
        fn: AnyFunction,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add a tool to the server.

        The tool function can optionally request a Context object by adding a parameter
        with the Context type annotation. See the @tool decorator for examples.

        Args:
            fn: The function to register as a tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description of what the tool does
        """
        self._tool_manager.add_tool(fn, name=name, description=description)

    # 装饰器在 Python 中是一个用于修饰函数或方法的工具。它实际上是一个高阶函数，接受一个函数作为输入，并返回一个新的函数。
    # 因此，装饰器本身在 Python 文件被 导入或执行时 立即执行，这是在定义被装饰的函数时发生的，而不是在函数被调用时。
    #
    # 装饰器执行时机
    # @mcp.tool()
    # async def get_alerts(state: str) -> str:
    #     """Get weather alerts for a US state.
    #
    #     Args:
    #         state: Two-letter US state code (e.g. CA, NY)
    #     """
    #     url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    #     data = await make_nws_request(url)
    #
    #     if not data or "features" not in data:
    #         return "Unable to fetch alerts or no alerts found."
    #
    #     if not data["features"]:
    #         return "No active alerts for this state."
    #
    #     alerts = [format_alert(feature) for feature in data["features"]]
    #     return "\n---\n".join(alerts)
    #   装饰器的定义:
    #       当 Python 文件被导入或执行时，装饰器 @mcp.tool() 会立即执行。执行时，它将 get_alerts 函数作为参数传递给 mcp.tool() 函数。
    #   装饰器的效果:
    #       mcp.tool() 接受 get_alerts 函数，并返回一个新的函数或对 get_alerts 进行修改。具体的行为依赖于 mcp.tool() 的实现。
    #       get_alerts 函数在文件被解析时被装饰，但装饰器内部的逻辑（比如注册函数、修改函数行为等）会在此时完成。
    #   函数调用时:
    #       get_alerts 函数的实际执行时机是当你在代码中调用 get_alerts(state) 时。此时，任何由装饰器添加的功能（比如日志记录、性能监控等）也会在函数执行过程中生效。
    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a tool.

        Tools can optionally request a Context object by adding a parameter with the
        Context type annotation. The context provides access to MCP capabilities like
        logging, progress reporting, and resource access.

        Args:
            name: Optional name for the tool (defaults to function name)
            description: Optional description of what the tool does

        Example:
            @server.tool()
            def my_tool(x: int) -> str:
                return str(x)

            @server.tool()
            def tool_with_context(x: int, ctx: Context) -> str:
                ctx.info(f"Processing {x}")
                return str(x)

            @server.tool()
            async def async_tool(x: int, context: Context) -> str:
                await context.report_progress(50, 100)
                return str(x)
        """
        # Check if user passed function directly instead of calling decorator
        if callable(name):
            raise TypeError(
                "The @tool decorator was used incorrectly. "
                "Did you forget to call it? Use @tool() instead of @tool"
            )

        def decorator(fn: AnyFunction) -> AnyFunction:
            self.add_tool(fn, name=name, description=description)
            return fn

        return decorator

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the server.

        Args:
            resource: A Resource instance to add
        """
        self._resource_manager.add_resource(resource)

    def resource(
        self,
        uri: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a function as a resource.

        The function will be called when the resource is read to generate its content.
        The function can return:
        - str for text content
        - bytes for binary content
        - other types will be converted to JSON

        If the URI contains parameters (e.g. "resource://{param}") or the function
        has parameters, it will be registered as a template resource.

        Args:
            uri: URI for the resource (e.g. "resource://my-resource" or "resource://{param}")
            name: Optional name for the resource
            description: Optional description of the resource
            mime_type: Optional MIME type for the resource

        Example:
            @server.resource("resource://my-resource")
            def get_data() -> str:
                return "Hello, world!"

            @server.resource("resource://my-resource")
            async get_data() -> str:
                data = await fetch_data()
                return f"Hello, world! {data}"

            @server.resource("resource://{city}/weather")
            def get_weather(city: str) -> str:
                return f"Weather for {city}"

            @server.resource("resource://{city}/weather")
            async def get_weather(city: str) -> str:
                data = await fetch_weather(city)
                return f"Weather for {city}: {data}"
        """
        # Check if user passed function directly instead of calling decorator
        if callable(uri):
            raise TypeError(
                "The @resource decorator was used incorrectly. "
                "Did you forget to call it? Use @resource('uri') instead of @resource"
            )

        def decorator(fn: AnyFunction) -> AnyFunction:
            # Check if this should be a template
            has_uri_params = "{" in uri and "}" in uri
            has_func_params = bool(inspect.signature(fn).parameters)

            if has_uri_params or has_func_params:
                # Validate that URI params match function params
                uri_params = set(re.findall(r"{(\w+)}", uri))
                func_params = set(inspect.signature(fn).parameters.keys())

                if uri_params != func_params:
                    raise ValueError(
                        f"Mismatch between URI parameters {uri_params} "
                        f"and function parameters {func_params}"
                    )

                # Register as template
                self._resource_manager.add_template(
                    fn=fn,
                    uri_template=uri,
                    name=name,
                    description=description,
                    mime_type=mime_type or "text/plain",
                )
            else:
                # Register as regular resource
                resource = FunctionResource(
                    uri=AnyUrl(uri),
                    name=name,
                    description=description,
                    mime_type=mime_type or "text/plain",
                    fn=fn,
                )
                self.add_resource(resource)
            return fn

        return decorator

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to the server.

        Args:
            prompt: A Prompt instance to add
        """
        self._prompt_manager.add_prompt(prompt)

    def prompt(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a prompt.

        Args:
            name: Optional name for the prompt (defaults to function name)
            description: Optional description of what the prompt does

        Example:
            @server.prompt()
            def analyze_table(table_name: str) -> list[Message]:
                schema = read_table_schema(table_name)
                return [
                    {
                        "role": "user",
                        "content": f"Analyze this schema:\n{schema}"
                    }
                ]

            @server.prompt()
            async def analyze_file(path: str) -> list[Message]:
                content = await read_file(path)
                return [
                    {
                        "role": "user",
                        "content": {
                            "type": "resource",
                            "resource": {
                                "uri": f"file://{path}",
                                "text": content
                            }
                        }
                    }
                ]
        """
        # Check if user passed function directly instead of calling decorator
        if callable(name):
            raise TypeError(
                "The @prompt decorator was used incorrectly. "
                "Did you forget to call it? Use @prompt() instead of @prompt"
            )

        def decorator(func: AnyFunction) -> AnyFunction:
            prompt = Prompt.from_function(func, name=name, description=description)
            self.add_prompt(prompt)
            return func

        return decorator

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )

    async def run_sse_async(self) -> None:
        """Run the server using SSE transport."""
        import uvicorn

        starlette_app = self.sse_app()

        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    def sse_app(self) -> Starlette:
        """Return an instance of the SSE server app."""
        sse = SseServerTransport(self.settings.message_path)

        async def handle_sse(request: Request) -> None:
            async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # type: ignore[reportPrivateUsage]
            ) as streams:
                await self._mcp_server.run(
                    streams[0],
                    streams[1],
                    self._mcp_server.create_initialization_options(),
                )

        return Starlette(
            debug=self.settings.debug,
            routes=[
                Route(self.settings.sse_path, endpoint=handle_sse),
                Mount(self.settings.message_path, app=sse.handle_post_message),
            ],
        )

    async def list_prompts(self) -> list[MCPPrompt]:
        """List all available prompts."""
        prompts = self._prompt_manager.list_prompts()
        return [
            MCPPrompt(
                name=prompt.name,
                description=prompt.description,
                arguments=[
                    MCPPromptArgument(
                        name=arg.name,
                        description=arg.description,
                        required=arg.required,
                    )
                    for arg in (prompt.arguments or [])
                ],
            )
            for prompt in prompts
        ]

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Get a prompt by name with arguments."""
        try:
            messages = await self._prompt_manager.render_prompt(name, arguments)

            return GetPromptResult(messages=pydantic_core.to_jsonable_python(messages))
        except Exception as e:
            logger.error(f"Error getting prompt {name}: {e}")
            raise ValueError(str(e))


def _convert_to_content(
    result: Any,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert a result to a sequence of content objects."""
    if result is None:
        return []

    if isinstance(result, TextContent | ImageContent | EmbeddedResource):
        return [result]

    if isinstance(result, Image):
        return [result.to_image_content()]

    if isinstance(result, list | tuple):
        return list(chain.from_iterable(_convert_to_content(item) for item in result))  # type: ignore[reportUnknownVariableType]

    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            result = str(result)

    return [TextContent(type="text", text=result)]


class Context(BaseModel, Generic[ServerSessionT, LifespanContextT]):
    """Context object providing access to MCP capabilities.

    This provides a cleaner interface to MCP's RequestContext functionality.
    It gets injected into tool and resource functions that request it via type hints.

    To use context in a tool function, add a parameter with the Context type annotation:

    ```python
    @server.tool()
    def my_tool(x: int, ctx: Context) -> str:
        # Log messages to the client
        ctx.info(f"Processing {x}")
        ctx.debug("Debug info")
        ctx.warning("Warning message")
        ctx.error("Error message")

        # Report progress
        ctx.report_progress(50, 100)

        # Access resources
        data = ctx.read_resource("resource://data")

        # Get request info
        request_id = ctx.request_id
        client_id = ctx.client_id

        return str(x)
    ```

    The context parameter name can be anything as long as it's annotated with Context.
    The context is optional - tools that don't need it can omit the parameter.
    """

    _request_context: RequestContext[ServerSessionT, LifespanContextT] | None
    _fastmcp: FastMCP | None

    def __init__(
        self,
        *,
        request_context: RequestContext[ServerSessionT, LifespanContextT] | None = None,
        fastmcp: FastMCP | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._request_context = request_context
        self._fastmcp = fastmcp

    @property
    def fastmcp(self) -> FastMCP:
        """Access to the FastMCP server."""
        if self._fastmcp is None:
            raise ValueError("Context is not available outside of a request")
        return self._fastmcp

    @property
    def request_context(self) -> RequestContext[ServerSessionT, LifespanContextT]:
        """Access to the underlying request context."""
        if self._request_context is None:
            raise ValueError("Context is not available outside of a request")
        return self._request_context

    async def report_progress(
        self, progress: float, total: float | None = None
    ) -> None:
        """Report progress for the current operation.

        Args:
            progress: Current progress value e.g. 24
            total: Optional total value e.g. 100
        """

        progress_token = (
            self.request_context.meta.progressToken
            if self.request_context.meta
            else None
        )

        if progress_token is None:
            return

        await self.request_context.session.send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )

    async def read_resource(self, uri: str | AnyUrl) -> Iterable[ReadResourceContents]:
        """Read a resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            The resource content as either text or bytes
        """
        assert (
            self._fastmcp is not None
        ), "Context is not available outside of a request"
        return await self._fastmcp.read_resource(uri)

    async def log(
        self,
        level: Literal["debug", "info", "warning", "error"],
        message: str,
        *,
        logger_name: str | None = None,
    ) -> None:
        """Send a log message to the client.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            logger_name: Optional logger name
            **extra: Additional structured data to include
        """
        await self.request_context.session.send_log_message(
            level=level, data=message, logger=logger_name
        )

    @property
    def client_id(self) -> str | None:
        """Get the client ID if available."""
        return (
            getattr(self.request_context.meta, "client_id", None)
            if self.request_context.meta
            else None
        )

    @property
    def request_id(self) -> str:
        """Get the unique ID for this request."""
        return str(self.request_context.request_id)

    @property
    def session(self):
        """Access to the underlying session for advanced usage."""
        return self.request_context.session

    # Convenience methods for common log levels
    async def debug(self, message: str, **extra: Any) -> None:
        """Send a debug log message."""
        await self.log("debug", message, **extra)

    async def info(self, message: str, **extra: Any) -> None:
        """Send an info log message."""
        await self.log("info", message, **extra)

    async def warning(self, message: str, **extra: Any) -> None:
        """Send a warning log message."""
        await self.log("warning", message, **extra)

    async def error(self, message: str, **extra: Any) -> None:
        """Send an error log message."""
        await self.log("error", message, **extra)
