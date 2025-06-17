import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TextIO

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from pydantic import BaseModel, Field

import mcp.types as types
from mcp.shared.message import SessionMessage

from .win32 import (
    create_windows_process,
    get_windows_executable_command,
    terminate_windows_process,
)

# Environment variables to inherit by default
DEFAULT_INHERITED_ENV_VARS = (
    [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
    if sys.platform == "win32"
    else ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
)


def get_default_environment() -> dict[str, str]:
    """
    Returns a default environment object including only environment variables deemed
    safe to inherit.
    """
    env: dict[str, str] = {}

    for key in DEFAULT_INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue

        if value.startswith("()"):
            # Skip functions, which are a security risk
            continue

        env[key] = value

    return env


class StdioServerParameters(BaseModel):
    command: str
    """The executable to run to start the server."""

    args: list[str] = Field(default_factory=list)
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None = None
    """
    The environment to use when spawning the process.

    If not specified, the result of get_default_environment() will be used.
    """

    cwd: str | Path | None = None
    """The working directory to use when spawning the process."""

    encoding: str = "utf-8"
    """
    The text encoding used when sending/receiving messages to the server

    defaults to utf-8
    """

    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values
    """


# @asynccontextmanager 是 Python 标准库 contextlib 模块提供的一个装饰器，用于实现异步上下文管理器。
# 简化创建: @asynccontextmanager 装饰器允许你使用简单的生成器函数来定义异步上下文管理器，而不是通过编写完整的类（需要实现 __aenter__ 和 __aexit__ 方法）。
# 生成器语法: 通过 async 和 yield 语句相结合，你可以在生成器内管理进入和退出上下文时的资源获取和释放。
#
# 入口代码: 在 yield 之前，放置任何需要在上下文进入时执行的初始化代码，比如资源获取或其他设置。
# yield 语句: yield 将上下文管理器的控制权交给 async with 语句块。yield 后的值传递给 async with 块。
# 退出代码: 在 finally 块中，放置任何需要在上下文退出时执行的清理代码，比如释放资源或其他清理工作。这部分代码无论 async with 块是否正常退出或由于异常退出都会被执行。

# 通过 @asynccontextmanager 和 AsyncExitStack 的协作，开发者可以以一种优雅和简洁的方式管理异步资源，确保在程序执行过程中资源正确地获取和释放。enter_async_context 进一步简化了资源管理逻辑，使得代码更加直观。
@asynccontextmanager
async def stdio_client(server: StdioServerParameters, errlog: TextIO = sys.stderr):
    """
    Client transport for stdio: this will connect to a server by spawning a
    process and communicating with it over stdin/stdout.
    """
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]

    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    # 一个内存对象流，缓冲区大小为 0。这意味着发送到 read_stream_writer 的每条消息必须在 read_stream 消费之后才能继续发送下一条。这是一种背压机制，确保消费者能够跟上生产者的速度。
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    try:
        command = _get_executable_command(server.command)

        # Open process with stderr piped for capture
        process = await _create_platform_compatible_process(
            command=command,
            args=server.args,
            env=({**get_default_environment(), **server.env} if server.env is not None else get_default_environment()),
            errlog=errlog,
            cwd=server.cwd,
        )
    except OSError:
        # Clean up streams if process creation fails
        await read_stream.aclose()
        await write_stream.aclose()
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"

        try:
            # read_stream_writer 被声明为一个 MemoryObjectSendStream 类型，用于发送 JSONRPCMessage 或 Exception 类型的对象。
            # 这是通过 AnyIO 创建的内存对象流的一部分，负责将读取的数据发送出去。
            # 使用异步上下文管理器来管理 read_stream_writer 的生命周期。确保在上下文管理器退出时，read_stream_writer 被正确关闭
            async with read_stream_writer:
                buffer = ""
                # TextReceiveStream 是一个异步可迭代对象，用于从子进程的标准输出读取文本数据，指定了编码和错误处理方式。
                # 异步迭代标准输出的每个数据块。
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    # 将 buffer 已有数据与新读取的 chunk 数据拼接，并按行分割（换行符 \n）。
                    # 结果是一个行列表 lines，可能最后一个元素是部分行。
                    lines = (buffer + chunk).split("\n")
                    # 从 lines 列表中移除并返回最后一个元素，赋给 buffer。这个元素可能是行的一部分，不完整。
                    buffer = lines.pop()

                    for line in lines:
                        try:
                            # 使用 model_validate_json 方法尝试将 line 解析为 JSONRPCMessage 对象。
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            # 如果解析失败，发送异常 exc 到 read_stream_writer。
                            await read_stream_writer.send(exc)
                            continue

                        session_message = SessionMessage(message)
                        # 如果成功解析为 JSONRPCMessage，将消息发送到 read_stream_writer
                        await read_stream_writer.send(session_message)
        # 捕获 anyio.ClosedResourceError，表示 read_stream_writer 已关闭。此时，进行适当的清理操作。
        except anyio.ClosedResourceError:
            # await anyio.lowlevel.checkpoint() 是一种在异步代码中显式插入非阻塞点的机制，特别是在长时间运行的任务中使用时，可以提高异步系统的响应性和效率。
            # 在密集计算或长时间执行的过程中，合理使用检查点可以确保异步事件循环的流畅运行，从而避免长时间占用 CPU 资源导致其他异步任务饿死的情况。
            await anyio.lowlevel.checkpoint()

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        # @asynccontextmanager配合使用
        try:
            # yield 将上下文管理器的控制权交给 async with 语句块。yield 后的值传递给 async with 块。
            yield read_stream, write_stream
        finally:
            # Clean up process to prevent any dangling orphaned processes
            try:
                if sys.platform == "win32":
                    await terminate_windows_process(process)
                else:
                    process.terminate()
            except ProcessLookupError:
                # Process already exited, which is fine
                pass
            await read_stream.aclose()
            await write_stream.aclose()
            await read_stream_writer.aclose()
            await write_stream_reader.aclose()


def _get_executable_command(command: str) -> str:
    """
    Get the correct executable command normalized for the current platform.

    Args:
        command: Base command (e.g., 'uvx', 'npx')

    Returns:
        str: Platform-appropriate command
    """
    if sys.platform == "win32":
        return get_windows_executable_command(command)
    else:
        return command


async def _create_platform_compatible_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    errlog: TextIO = sys.stderr,
    cwd: Path | str | None = None,
):
    """
    Creates a subprocess in a platform-compatible way.
    Returns a process handle.
    """
    if sys.platform == "win32":
        process = await create_windows_process(command, args, env, errlog, cwd)
    else:
        # 使用 AnyIO 库中的 open_process 方法来启动一个子进程。open_process 是一种异步方式来处理进程的创建和管理。
        # await 关键字表示这是一个异步操作。当前协程会暂停执行，直到 open_process 完成其创建子进程的操作。

        # 列表 [command, *args] 指定了子进程将要执行的命令和参数。
        #   command 是要执行的命令或可执行文件的路径。
        #   *args 是传递给命令的参数。使用 *args 可以将一个可迭代的参数列表解包为多个单独的参数。
        # env 参数允许你指定子进程的环境变量。env 应该是一个字典，包含环境变量的键值对。如果不指定，则子进程会继承当前进程的环境变量。
        # stderr 参数指定子进程的标准错误输出（stderr）如何处理。errlog 可以是一个文件对象、管道等，用于捕获或重定向标准错误输出。
        # cwd（current working directory）参数指定子进程的当前工作目录。cwd 应该是一个字符串，表示目录的路径。如果不指定，则子进程会在当前目录下执行。
        process = await anyio.open_process([command, *args], env=env, stderr=errlog, cwd=cwd)

    return process
