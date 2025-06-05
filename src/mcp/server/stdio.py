"""
Stdio Server Transport Module

This module provides functionality for creating an stdio-based transport layer
that can be used to communicate with an MCP client through standard input/output
streams.

Example usage:
```
    async def run_server():
        async with stdio_server() as (read_stream, write_stream):
            # read_stream contains incoming JSONRPCMessages from stdin
            # write_stream allows sending JSONRPCMessages to stdout
            server = await create_my_server()
            await server.run(read_stream, write_stream, init_options)

    anyio.run(run_server)
```
"""

import sys
from contextlib import asynccontextmanager
from io import TextIOWrapper

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

import mcp.types as types

# anyio.AsyncFile 是 anyio 库中的一个类，它提供了对文件进行异步操作的接口。
# 与传统的同步文件 I/O 不同，AsyncFile 允许在异步编程环境中使用 await 语义来执行文件操作，从而使得这些操作不会阻塞事件循环。
# 这在需要处理大量文件读写或在 I/O 密集型任务中非常有用，因为它可以提高应用程序的并发性能和响应能力。
#
# anyio.AsyncFile 的作用
# 异步文件操作:
#   AsyncFile 提供了对文件的异步读写功能。通过将文件操作与事件循环集成，可以在其他任务执行时等待文件操作完成。
# 非阻塞 I/O:
#   传统的文件操作往往是阻塞的，即在文件操作完成之前，程序无法继续执行其他任务。
#   使用 AsyncFile，可以发起文件操作并在操作进行时继续处理其他任务，直到操作完成。
# 与 await 语法兼容:
#   AsyncFile 允许使用 await 来等待文件操作完成，这使得在 Python 的异步函数中使用文件操作变得简单直观。
@asynccontextmanager
async def stdio_server(
    stdin: anyio.AsyncFile[str] | None = None,
    stdout: anyio.AsyncFile[str] | None = None,
):
    """
    Server transport for stdio: this communicates with an MCP client by reading
    from the current process' stdin and writing to stdout.
    """
    # Purposely not using context managers for these, as we don't want to close
    # standard process handles. Encoding of stdin/stdout as text streams on
    # python is platform-dependent (Windows is particularly problematic), so we
    # re-wrap the underlying binary stream to ensure UTF-8.
    # anyio.wrap_file 和 TextIOWrapper 的结合使用是为了处理标准输入和输出流的异步操作和编码处理
    # TextIOWrapper
    #   作用: TextIOWrapper 是 Python 标准库 io 模块中的类，用于将字节流（binary stream）包装为文本流（text stream），从而可以指定编码格式（如 UTF-8）。
    #   工作原理: 它接收一个字节流对象（如 sys.stdin.buffer）并返回一个文本流对象。通过这种方式，可以对输入输出流进行编码转换。
    # anyio.wrap_file
    #   作用: anyio 是一个 Python 异步框架，提供了与标准库的异步兼容性。wrap_file 是 anyio 提供的方法，用于将标准文件对象转换为异步兼容的文件对象。
    #   工作原理: 它接受一个标准文件对象（如通过 TextIOWrapper 创建的文本流）并返回一个异步文件对象，使得文件操作可以在异步环境中使用（例如 await 语句）。
    if not stdin:
        # 我们首先使用 TextIOWrapper 将 sys.stdin.buffer 的字节流转换为文本流，指定编码为 "utf-8"
        stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    if not stdout:
        # 将 sys.stdout.buffer 包装为文本流，确保输出时以 UTF-8 编码
        # 使用 anyio.wrap_file 将文本流对象转换为异步兼容的对象
        stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def stdin_reader():
        try:
            async with read_stream_writer:
                # async for line in stdin: 允许在异步函数中逐行读取标准输入
                async for line in stdin:
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue

                    await read_stream_writer.send(message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdout_writer():
        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    # await stdout.write(line) 允许异步地写入输出。
                    await stdout.write(json + "\n")
                    # await stdout.flush() 用于确保所有缓冲区的数据都被写入输出。
                    await stdout.flush()
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        yield read_stream, write_stream
