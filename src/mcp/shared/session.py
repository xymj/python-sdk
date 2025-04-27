import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from types import TracebackType
from typing import Any, Generic, TypeVar

import anyio
import anyio.lowlevel
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel
from typing_extensions import Self

from mcp.shared.exceptions import McpError
from mcp.types import (
    CancelledNotification,
    ClientNotification,
    ClientRequest,
    ClientResult,
    ErrorData,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestParams,
    ServerNotification,
    ServerRequest,
    ServerResult,
)

SendRequestT = TypeVar("SendRequestT", ClientRequest, ServerRequest)
SendResultT = TypeVar("SendResultT", ClientResult, ServerResult)
SendNotificationT = TypeVar("SendNotificationT", ClientNotification, ServerNotification)
ReceiveRequestT = TypeVar("ReceiveRequestT", ClientRequest, ServerRequest)

# Pydantic 是一个数据验证和数据设置管理的 Python 库，主要用于验证数据的结构和类型。它
# 利用 Python 的类型注解来声明数据模型，并自动地进行数据验证和转换。
# Pydantic 在处理数据输入、输出、序列化和反序列化等方面非常强大，尤其是在构建 API、数据管道或其他需要严格数据约束的应用中表现出色。
# 主要功能
# 数据验证: 根据定义的数据模型自动验证数据类型和格式。
# 数据解析: 自动解析和转换输入数据，使其符合数据模型定义。
# 默认值管理: 允许为字段设置默认值。
# 序列化和反序列化: 支持将模型实例转化为 JSON 等格式，以及从 JSON 等格式创建模型实例。
# 性能优化: 高效的运行时性能，尤其是在数据处理任务中。
ReceiveResultT = TypeVar("ReceiveResultT", bound=BaseModel)
ReceiveNotificationT = TypeVar(
    "ReceiveNotificationT", ClientNotification, ServerNotification
)

RequestId = str | int


class RequestResponder(Generic[ReceiveRequestT, SendResultT]):
    """Handles responding to MCP requests and manages request lifecycle.

    This class MUST be used as a context manager to ensure proper cleanup and
    cancellation handling:

    Example:
        with request_responder as resp:
            await resp.respond(result)

    The context manager ensures:
    1. Proper cancellation scope setup and cleanup
    2. Request completion tracking
    3. Cleanup of in-flight requests
    """
    # 在 Python 中，__init__、__enter__ 和 __exit__ 是特殊方法（也称为魔术方法），
    # 它们在类的不同生命周期阶段自动被调用。尽管它们以双下划线开头和结尾，它们并不是私有方法，而是特殊的内建方法，
    # 分别用于对象的初始化和上下文管理。

    # __init__ 是初始化方法，用于创建对象时初始化状态。它在对象实例化时被自动调用。
    def __init__(
        self,
        request_id: RequestId,
        request_meta: RequestParams.Meta | None,
        request: ReceiveRequestT,
        # 三引号 ("""): 在类型注解中，三引号的使用可能是为了避免前向引用的问题，尤其是在某些情况下避免循环导入或者模块尚未完全加载的问题。
        session: """BaseSession[
            SendRequestT,
            SendNotificationT,
            SendResultT,
            ReceiveRequestT,
            ReceiveNotificationT
        ]""",
        # 单引号 ("): 在类型注解中，单引号的使用常常用于前向引用。它允许你引用在定义之后才出现的类型，或者在类型检查时避免循环引用。
        #  Callable[[参数类型], 返回类型]
        on_complete: Callable[["RequestResponder[ReceiveRequestT, SendResultT]"], Any],
    ) -> None:
        self.request_id = request_id
        self.request_meta = request_meta
        self.request = request
        self._session = session
        self._completed = False
        self._cancel_scope = anyio.CancelScope()
        self._on_complete = on_complete
        self._entered = False  # Track if we're in a context manager

    # __enter__ 和 __exit__   这两个方法用于实现上下文管理协议，允许对象使用 with 语句。
    # __enter__
    #   含义: __enter__ 方法在进入 with 代码块时自动调用。通常用于资源的获取或环境设置。
    #   调用时机: 在 with 语句开始时调用，并且返回的对象将被赋值给 as 后面的变量。
    def __enter__(self) -> "RequestResponder[ReceiveRequestT, SendResultT]":
        """Enter the context manager, enabling request cancellation tracking."""
        self._entered = True
        # CancelScope(): 创建一个取消范围。该范围内启动的任务都受其控制。是 AnyIO 提供的一种机制，用于管理异步任务的取消。
        #   它允许你定义一个范围，在这个范围内启动的任务可以被一起取消。
        #   这在需要确保一组协作的异步任务在某些条件下（如超时或错误）一起停止时非常有用。
        # CancelScope 的作用
        #   CancelScope 提供一个上下文管理器，用于控制在其范围内运行的异步任务。它主要有以下作用：
        #       管理任务生命周期: 可以在特定条件下手动取消任务，例如超时后自动取消任务。
        #       协作取消: 允许多个任务共享同一个取消范围，确保它们可以被同时取消。
        #       组织复杂任务: 在复杂的异步程序中，使用取消范围可以简化任务的管理。
        self._cancel_scope = anyio.CancelScope()
        # with启动异步任务都受_cancel_scope来控制取消
        self._cancel_scope.__enter__()
        return self

    # __exit__
    #   含义: __exit__ 方法在离开 with 代码块时被调用。用于资源的清理或恢复环境。
    #   调用时机: 无论 with 代码块内是否发生异常，__exit__ 方法都会被调用。
    #   __exit__ 接受三个参数：异常类型、异常值和追溯信息，如果没有异常，这些参数都为 None。
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, performing cleanup and notifying completion."""
        try:
            if self._completed:
                self._on_complete(self)
        finally:
            self._entered = False
            if not self._cancel_scope:
                raise RuntimeError("No active cancel scope")
            # _cancel_scope取消其控制范围内异步任务
            self._cancel_scope.__exit__(exc_type, exc_val, exc_tb)

    async def respond(self, response: SendResultT | ErrorData) -> None:
        """Send a response for this request.

        Must be called within a context manager block.
        Raises:
            RuntimeError: If not used within a context manager
            AssertionError: If request was already responded to
        """
        if not self._entered:
            raise RuntimeError("RequestResponder must be used as a context manager")
        assert not self._completed, "Request already responded to"

        if not self.cancelled:
            self._completed = True

            await self._session._send_response(  # type: ignore[reportPrivateUsage]
                request_id=self.request_id, response=response
            )

    async def cancel(self) -> None:
        """Cancel this request and mark it as completed."""
        if not self._entered:
            raise RuntimeError("RequestResponder must be used as a context manager")
        if not self._cancel_scope:
            raise RuntimeError("No active cancel scope")

        # _cancel_scope.cancel() 是 AnyIO 中 CancelScope 类的一个方法，其作用是请求取消该取消范围（cancel scope）内的所有任务。
        #   调用这个方法会导致在该 CancelScope 范围内运行的所有异步任务被标记为取消。
        self._cancel_scope.cancel()
        self._completed = True  # Mark as completed so it's removed from in_flight
        # Send an error response to indicate cancellation
        await self._session._send_response(  # type: ignore[reportPrivateUsage]
            request_id=self.request_id,
            response=ErrorData(code=0, message="Request cancelled", data=None),
        )

    @property
    def in_flight(self) -> bool:
        return not self._completed and not self.cancelled

    @property
    def cancelled(self) -> bool:
        # _cancel_scope.cancel_called 是 anyio.CancelScope 中的一个属性，
        #   用于指示是否已经请求取消该取消范围（cancel scope）内的任务。这是一个布尔值属性，可以被用于检查取消请求是否已经发出。
        return self._cancel_scope.cancel_called


class BaseSession(
    Generic[
        SendRequestT,
        SendNotificationT,
        SendResultT,
        ReceiveRequestT,
        ReceiveNotificationT,
    ],
):
    """
    Implements an MCP "session" on top of read/write streams, including features
    like request/response linking, notifications, and progress.

    This class is an async context manager that automatically starts processing
    messages when entered.
    """

    _response_streams: dict[
        RequestId, MemoryObjectSendStream[JSONRPCResponse | JSONRPCError]
    ]
    _request_id: int
    _in_flight: dict[RequestId, RequestResponder[ReceiveRequestT, SendResultT]]

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[JSONRPCMessage | Exception],
        write_stream: MemoryObjectSendStream[JSONRPCMessage],
        receive_request_type: type[ReceiveRequestT],
        receive_notification_type: type[ReceiveNotificationT],
        # If none, reading will never time out
        read_timeout_seconds: timedelta | None = None,
    ) -> None:
        self._read_stream = read_stream
        self._write_stream = write_stream
        self._response_streams = {}
        self._request_id = 0
        self._receive_request_type = receive_request_type
        self._receive_notification_type = receive_notification_type
        self._read_timeout_seconds = read_timeout_seconds
        self._in_flight = {}

        # AsyncExitStack(): 创建一个异步退出栈，用于管理多个上下文。
        # 如果 AsyncExitStack 被设计用于像一种资源管理工具而非严格的上下文管理器，则可能设计初衷是仅在 __aexit__ 中执行清理工作，而不需要在进入时做任何初始化。
        self._exit_stack = AsyncExitStack()

    # __aenter__
    #   作用: 在使用 async with 语句进入异步上下文管理器时自动调用。它用于执行进入上下文管理器时需要的初始化操作和资源获取。
    #   执行时机: 当 async with 语句被执行时，最先调用 __aenter__ 方法。
    async def __aenter__(self) -> Self:
        #  创建了一个新的任务组。anyio.create_task_group() 是 AnyIO 提供的功能，用于管理多个异步任务的并行执行。
        self._task_group = anyio.create_task_group()
        # 手动进入任务组的上下文管理器。通常，任务组在异步上下文中会自动管理其生命周期，包括资源分配和任务调度。
        await self._task_group.__aenter__()
        # 启动异步任务，将 _receive_loop 方法作为一个任务启动。
        # 这意味着 _receive_loop 开始异步执行，并且在任务组的管理下运行。
        # start_soon 方法会立即安排任务执行，而不需要显式地等待。
        self._task_group.start_soon(self._receive_loop)
        return self

    # __aexit__
    #   作用: 在离开 async with 语句时自动调用。用于执行任何需要的清理操作和资源释放。
    #   执行时机: 无论是正常退出还是由于异常导致的退出，__aexit__ 方法都会被调用。
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        # 调用 self._exit_stack.aclose() 在 __aexit__ 中可以理解为一个确保资源释放和清理的最后手段，
        # 即使 __aenter__ 中没有执行上下文进入，aclose() 方法会确保栈中注册的所有异步回调被执行。
        # 这种做法可能是为了保证资源不会泄漏，即使没有明确的上下文管理开始。
        await self._exit_stack.aclose()
        # Using BaseSession as a context manager should not block on exit (this
        # would be very surprising behavior), so make sure to cancel the tasks
        # in the task group.
        # 取消任务组中的所有任务。task_group 中的任务通常是在 __aenter__ 方法中启动的，因此在上下文退出时，需要确保这些任务被正确终止。
        # cancel_scope.cancel() 方法会标记任务组中的所有任务为“已取消”，并触发与取消相关的异常处理。这是为了确保异步任务不会在上下文退出后继续运行。
        self._task_group.cancel_scope.cancel()
        # 确保 task_group 进行任何必要的清理操作，包括处理异常或取消任务。
        return await self._task_group.__aexit__(exc_type, exc_val, exc_tb)

    # 异步与同步上下文管理器的区别
    # __aenter__ 和 __aexit__ vs. __enter__ 和 __exit__
    # 异步 vs. 同步: __aenter__ 和 __aexit__ 是异步方法，通常使用 await 关键字来执行异步操作。
    #   它们用于异步上下文管理器，而 __enter__ 和 __exit__ 是同步方法，适用于同步上下文管理器。
    # 使用场景: 当在异步代码中管理资源（例如异步网络连接、异步文件 I/O 等）时，使用 async with 结合 __aenter__ 和 __aexit__ 是合适的选择。
    #   而 with 语句结合 __enter__ 和 __exit__ 适用于同步资源管理。


    # 发送消息处理调用顺序:
    #     session.send_request[_write_stream.send] -> session._receive_loop ->
    #     RequestResponder() ->  session._received_request(处理请求，子类实现) ->  RequestResponder.respond() ->
    #     _session._send_response[_write_stream.send] - > session._receive_loop ->  _response_streams[id].send() ->
    #     session.send_request[response_stream_reader.receive()]
    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
    ) -> ReceiveResultT:
        """
        Sends a request and wait for a response. Raises an McpError if the
        response contains an error.

        Do not use this method to emit notifications! Use send_notification()
        instead.
        """

        request_id = self._request_id
        self._request_id = request_id + 1

        # create_memory_object_stream 方法来创建一个内存对象流（memory object stream）。具体来说，它创建了一对配对的异步流：一个用于发送数据，另一个用于接收数据。
        # 创建的流分为发送器（sender）和接收器（receiver），用于在任务之间传递消息或数据。
        response_stream, response_stream_reader = anyio.create_memory_object_stream[
            JSONRPCResponse | JSONRPCError
        # (1):这个参数指定了缓冲区的大小。在这种情况下，缓冲区大小为 1，意味着发送器可以在接收器处理之前最多发送一个对象。这个缓冲区限制有助于控制流量和背压
        ](1)
        self._response_streams[request_id] = response_stream

        # AsyncExitStack.push_async_callback: 将一个异步回调函数（这里是 response_stream.aclose()）推入退出栈。
        #   这意味着当 AsyncExitStack 退出时，无论是在正常完成还是抛出异常导致退出，这个回调都会被调用。
        self._exit_stack.push_async_callback(lambda: response_stream.aclose())
        self._exit_stack.push_async_callback(lambda: response_stream_reader.aclose())

        jsonrpc_request = JSONRPCRequest(
            jsonrpc="2.0",
            id=request_id,
            **request.model_dump(by_alias=True, mode="json", exclude_none=True),
        )

        # TODO: Support progress callbacks

        await self._write_stream.send(JSONRPCMessage(jsonrpc_request))

        try:
            # 使用 AnyIO 的 fail_after 上下文管理器来设置一个超时。如果操作超过指定时间未完成，则会抛出 TimeoutError。
            # 如果 self._read_timeout_seconds 为 None，则没有超时限制；否则，使用指定的秒数作为超时时间。
            with anyio.fail_after(
                # None reading will never time out
                None
                if self._read_timeout_seconds is None
                else self._read_timeout_seconds.total_seconds()
            ):
                # 异步等待从 response_stream_reader 接收数据。这一步可能因为没有数据可用而被阻塞，直到数据到达或超时。
                # response_or_error 变量保存从流中接收到的对象，这个对象可以是 JSONRPCError 或某种响应类型。
                response_or_error = await response_stream_reader.receive()
        except TimeoutError:
            raise McpError(
                ErrorData(
                    code=httpx.codes.REQUEST_TIMEOUT,
                    message=(
                        f"Timed out while waiting for response to "
                        f"{request.__class__.__name__}. Waited "
                        f"{self._read_timeout_seconds} seconds."
                    ),
                )
            )

        if isinstance(response_or_error, JSONRPCError):
            raise McpError(response_or_error.error)
        else:
            # 将响应对象的结果部分进行模型验证。model_validate 方法用于确保响应数据符合预期的结构和格式。
            # 返回经过验证的结果。这通常是一个符合特定数据模型的对象，供后续使用。
            return result_type.model_validate(response_or_error.result)

    async def send_notification(self, notification: SendNotificationT) -> None:
        """
        Emits a notification, which is a one-way message that does not expect
        a response.
        """
        jsonrpc_notification = JSONRPCNotification(
            jsonrpc="2.0",
            **notification.model_dump(by_alias=True, mode="json", exclude_none=True),
        )

        await self._write_stream.send(JSONRPCMessage(jsonrpc_notification))

    async def _send_response(
        self, request_id: RequestId, response: SendResultT | ErrorData
    ) -> None:
        if isinstance(response, ErrorData):
            jsonrpc_error = JSONRPCError(jsonrpc="2.0", id=request_id, error=response)
            await self._write_stream.send(JSONRPCMessage(jsonrpc_error))
        else:
            jsonrpc_response = JSONRPCResponse(
                jsonrpc="2.0",
                id=request_id,
                result=response.model_dump(
                    by_alias=True, mode="json", exclude_none=True
                ),
            )
            await self._write_stream.send(JSONRPCMessage(jsonrpc_response))

    # 核心异步处理流程，很像一个异步处理队列，用来接受消息，pop消息处理，push消息消费
    async def _receive_loop(self) -> None:
        # 使用异步上下文管理器进入 self._read_stream 和 self._write_stream。这通常用于确保资源在退出时得到正确的释放。
        async with (
            self._read_stream,
            self._write_stream,
        ):
            # 通过 async for 从 self._read_stream 中异步读取消息。self._read_stream 应该是一个异步可迭代对象。
            async for message in self._read_stream:
                # 如果 message 是一个异常对象，则调用 _handle_incoming 方法处理该异常。
                if isinstance(message, Exception):
                    await self._handle_incoming(message)
                # 检查消息是否为 JSON-RPC 请求
                elif isinstance(message.root, JSONRPCRequest):
                    # 对请求进行验证。使用 model_validate 方法确保请求符合预期的格式和数据结构。
                    validated_request = self._receive_request_type.model_validate(
                        message.root.model_dump(
                            by_alias=True, mode="json", exclude_none=True
                        )
                    )

                    # 创建请求响应者，创建一个 RequestResponder 实例，用于管理请求的响应。
                    responder = RequestResponder(
                        request_id=message.root.id,
                        request_meta=validated_request.root.params.meta
                        if validated_request.root.params
                        else None,
                        request=validated_request,
                        session=self,
                        # on_complete 是一个回调函数，响应完成后从 _in_flight 中移除请求
                        on_complete=lambda r: self._in_flight.pop(r.request_id, None),
                    )

                    # 将请求添加到 _in_flight 字典中，跟踪当前正在处理的请求。
                    self._in_flight[responder.request_id] = responder
                    # await的使用是异步编程中管理并发逻辑的核心部分。通过 await，程序可以在等待异步操作完成时继续执行其他任务，从而提高程序的响应性和效率。
                    # 在 await self._received_request(responder) 被执行时，当前协程会暂停，直至 _received_request 内的所有异步操作完成。
                    #   这样设计使得异步程序能够有效地 并发 执行，而不是阻塞等待每个异步操作。
                    await self._received_request(responder)

                    if not responder._completed:  # type: ignore[reportPrivateUsage]
                        await self._handle_incoming(responder)

                # 处理 JSON-RPC 通知
                elif isinstance(message.root, JSONRPCNotification):
                    try:
                        notification = self._receive_notification_type.model_validate(
                            message.root.model_dump(
                                by_alias=True, mode="json", exclude_none=True
                            )
                        )
                        # Handle cancellation notifications
                        if isinstance(notification.root, CancelledNotification):
                            cancelled_id = notification.root.params.requestId
                            if cancelled_id in self._in_flight:
                                await self._in_flight[cancelled_id].cancel()
                        else:
                            await self._received_notification(notification)
                            await self._handle_incoming(notification)
                    except Exception as e:
                        # For other validation errors, log and continue
                        logging.warning(
                            f"Failed to validate notification: {e}. "
                            f"Message was: {message.root}"
                        )
                # 处理响应或错误:
                else:  # Response or error
                    stream = self._response_streams.pop(message.root.id, None)
                    if stream:
                        await stream.send(message.root)
                    else:
                        await self._handle_incoming(
                            RuntimeError(
                                "Received response with an unknown "
                                f"request ID: {message}"
                            )
                        )

    async def _received_request(
        self, responder: RequestResponder[ReceiveRequestT, SendResultT]
    ) -> None:
        """
        Can be overridden by subclasses to handle a request without needing to
        listen on the message stream.

        If the request is responded to within this method, it will not be
        forwarded on to the message stream.
        """

    async def _received_notification(self, notification: ReceiveNotificationT) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Sends a progress notification for a request that is currently being
        processed.
        """

    async def _handle_incoming(
        self,
        req: RequestResponder[ReceiveRequestT, SendResultT]
        | ReceiveNotificationT
        | Exception,
    ) -> None:
        """A generic handler for incoming messages. Overwritten by subclasses."""
        pass
