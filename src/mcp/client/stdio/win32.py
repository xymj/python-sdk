"""
Windows-specific functionality for stdio client operations.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

import anyio
from anyio.abc import Process


def get_windows_executable_command(command: str) -> str:
    """
    Get the correct executable command normalized for Windows.

    On Windows, commands might exist with specific extensions (.exe, .cmd, etc.)
    that need to be located for proper execution.

    Args:
        command: Base command (e.g., 'uvx', 'npx')

    Returns:
        str: Windows-appropriate command path
    """
    try:
        # First check if command exists in PATH as-is
        if command_path := shutil.which(command):
            return command_path

        # Check for Windows-specific extensions
        for ext in [".cmd", ".bat", ".exe", ".ps1"]:
            ext_version = f"{command}{ext}"
            if ext_path := shutil.which(ext_version):
                return ext_path

        # For regular commands or if we couldn't find special versions
        return command
    except OSError:
        # Handle file system errors during path resolution
        # (permissions, broken symlinks, etc.)
        return command


async def create_windows_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    errlog: TextIO = sys.stderr,
    cwd: Path | str | None = None,
):
    """
    Creates a subprocess in a Windows-compatible way.

    Windows processes need special handling for console windows and
    process creation flags.

    Args:
        command: The command to execute
        args: Command line arguments
        env: Environment variables
        errlog: Where to send stderr output
        cwd: Working directory for the process

    Returns:
        A process handle
    """
    try:
        # Try with Windows-specific flags to hide console window
        process = await anyio.open_process(
            [command, *args],
            env=env,
            # Ensure we don't create console windows for each process
            creationflags=subprocess.CREATE_NO_WINDOW  # type: ignore
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0,
            stderr=errlog,
            cwd=cwd,
        )
        return process
    except Exception:
        # Don't raise, let's try to create the process without creation flags
        process = await anyio.open_process([command, *args], env=env, stderr=errlog, cwd=cwd)
        return process


async def terminate_windows_process(process: Process):
    """
    Terminate a Windows process.

    Note: On Windows, terminating a process with process.terminate() doesn't
    always guarantee immediate process termination.
    So we give it 2s to exit, or we call process.kill()
    which sends a SIGKILL equivalent signal.

    Args:
        process: The process to terminate
    """
    try:
        # 发送终止信号: process.terminate() 通常发送 SIGTERM 信号给目标进程。在 Unix 和 Linux 系统中，这个信号是请求进程终止的信号。
        # 软终止: SIGTERM 是一种“软”终止信号，这意味着它可以被进程捕获并进行相应的清理操作。因此，进程有机会在终止前完成一些清理工作（例如，释放资源、保存状态等）。
        # 跨平台行为: 在 Windows 上，terminate() 的行为可能有所不同，但通常也会产生类似的效果，即请求进程终止。
        process.terminate()
        with anyio.fail_after(2.0):
            # 阻塞等待: process.wait() 会阻塞当前执行的线程或进程，直到被等待的子进程终止。这意味着父进程在调用 wait() 后会暂停执行，直到子进程完成。
            # 获取子进程的退出状态: 当子进程终止后，wait() 返回子进程的退出码。退出码通常用来判断子进程的执行是否成功（通常 0 表示成功，非零值表示某种错误）。
            await process.wait()
    except TimeoutError:
        # Force kill if it doesn't terminate
        # 立即终止进程: process.kill() 会立即终止目标进程，不给进程任何机会进行运行中的清理或保存操作。这是通过发送 SIGKILL 信号实现的（在 Unix 和 Linux 系统上）。
        # 不可捕获和忽略: 与 SIGTERM 不同，SIGKILL 信号是不可捕获和忽略的。目标进程无法拦截这个信号，因此不能进行任何形式的异常处理或资源清理。
        # 跨平台行为: 在 Windows 上，kill() 方法会执行类似的强制终止操作，不过 Windows 没有信号的概念，所以其实现是通过系统调用直接终止进程。
        process.kill()
