"""Isaac 进程和 GR00T 进程之间共用的 socket 通信工具。

这个实验刻意把两个 Python 环境隔离开：

1. Isaac 侧运行在 `conda activate leisaac` 里，它负责启动 Isaac Sim、
   加载 LeIsaac SmartTask 场景、读取相机/机器人状态、执行机器人动作。
2. GR00T 侧运行在 `$GROOT_ROOT/.venv` 里，它负责加载
   `nvidia/GR00T-N1.7-3B` 模型并做推理。

因为两个环境里的依赖很重、版本也不同，所以不要把 GR00T 直接 import
到 IsaacLab 环境，也不要把 IsaacLab 装进 GR00T venv。这个文件提供一个
极小的本地 TCP 协议，让两个进程只交换普通 Python dict 和 numpy 数组。
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any


# 网络传输时每条消息都分成两段：
# 1. 固定 8 字节 header，记录后面 payload 有多少字节。
# 2. pickle 编码后的 payload 本体。
#
# `!Q` 的意思：
# - `!`：network byte order，也就是 big-endian。这样不同机器/语言读取时一致。
# - `Q`：unsigned long long，占 8 字节，可以表达非常大的 payload 长度。
_HEADER = struct.Struct("!Q")


def pack_message(payload: dict[str, Any]) -> bytes:
    """把一条业务消息编码成 bytes。

    参数：
    - `payload`：业务层 dict，例如 `{"endpoint": "get_action", ...}`。

    返回：
    - 可通过 socket 发送的 bytes，不包含长度 header。
    """

    # 这里使用标准库 pickle，避免要求 Isaac/LeIsaac 环境额外安装 msgpack。
    # bridge 默认只监听 127.0.0.1；不要把这个本地协议暴露给不可信网络。
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_message(payload: bytes) -> dict[str, Any]:
    """把 bytes 解码回业务层 dict。"""

    message = pickle.loads(payload)
    if not isinstance(message, dict):
        raise TypeError(f"wire payload must decode to dict, got {type(message).__name__}")
    return message


def send_message(conn: socket.socket, payload: dict[str, Any]) -> None:
    """通过一个已连接的 socket 发送一条完整消息。

    参数：
    - `conn`：已经建立连接的 TCP socket。
    - `payload`：要发送的业务层 dict。

    协议细节：
    - 先发 8 字节长度 header。
    - 再发 pickle payload。
    - 使用 `sendall()`，确保所有字节都写入 socket。
    """

    data = pack_message(payload)
    conn.sendall(_HEADER.pack(len(data)))
    conn.sendall(data)


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """从 socket 中精确读取 `size` 个字节。

    TCP 是字节流协议，一次 `recv(size)` 并不保证真的返回 size 个字节；
    它可能只返回一部分。所以这里循环读取，直到拿满目标长度。
    """

    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(conn: socket.socket) -> dict[str, Any]:
    """从 socket 中接收一条完整消息，并解码成业务层 dict。"""

    header = recv_exact(conn, _HEADER.size)
    size = _HEADER.unpack(header)[0]
    return unpack_message(recv_exact(conn, size))


def request(host: str, port: int, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """客户端侧的一次请求-响应封装。

    Isaac runner 使用这个函数向 GR00T bridge 发请求：

    - 建立到 `host:port` 的 TCP 连接。
    - 发送一条 payload。
    - 等待 bridge 返回一条 response。
    - 函数结束时自动关闭 socket。
    """

    with socket.create_connection((host, port), timeout=timeout_s) as conn:
        conn.settimeout(timeout_s)
        send_message(conn, payload)
        return recv_message(conn)
