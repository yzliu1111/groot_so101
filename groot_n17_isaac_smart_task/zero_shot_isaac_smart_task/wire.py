"""Isaac 进程和 GR00T 进程之间共用的 socket/msgpack 通信工具。

这个实验刻意把两个 Python 环境隔离开：

1. Isaac 侧运行在 `conda activate isaaclab` 里，它负责启动 Isaac Sim、
   加载 LeIsaac SmartTask 场景、读取相机/机器人状态、执行机器人动作。
2. GR00T 侧运行在 `/home/yzliu/Isaac-GR00T/.venv` 里，它负责加载
   `nvidia/GR00T-N1.7-3B` 模型并做推理。

因为两个环境里的依赖很重、版本也不同，所以不要把 GR00T 直接 import
到 IsaacLab 环境，也不要把 IsaacLab 装进 GR00T venv。这个文件提供一个
极小的本地 TCP 协议，让两个进程只交换普通 Python dict 和 numpy 数组。
"""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack
import numpy as np


# 网络传输时每条消息都分成两段：
# 1. 固定 8 字节 header，记录后面 payload 有多少字节。
# 2. msgpack 编码后的 payload 本体。
#
# `!Q` 的意思：
# - `!`：network byte order，也就是 big-endian。这样不同机器/语言读取时一致。
# - `Q`：unsigned long long，占 8 字节，可以表达非常大的 payload 长度。
_HEADER = struct.Struct("!Q")


def _pack_numpy(obj: Any) -> Any:
    """把 msgpack 原生不认识的 numpy 类型转换成可序列化的 dict。

    `msgpack.packb(..., default=_pack_numpy)` 遇到无法直接编码的对象时，
    会调用这个函数。这里主要处理两类对象：

    - `np.ndarray`：图像、状态、动作基本都是 ndarray。
    - `np.generic`：numpy 标量，例如 `np.float32(1.0)`。

    返回的 dict 会带上哨兵字段，例如 `__ndarray__`，这样解码端可以知道
    这个 dict 不是普通业务数据，而是一个需要还原成 numpy 的对象。
    """

    # ndarray 是本实验最主要的数据形态，例如相机图像 `(B,T,H,W,C)`、
    # 机器人状态 `(B,T,D)`、动作 chunk `(B,T,D)`。
    if isinstance(obj, np.ndarray):
        # object dtype 或 void dtype 没有稳定、明确的跨进程序列化语义。
        # 这里直接拒绝，避免把 Python 对象偷偷塞进二进制协议里。
        if obj.dtype.kind in ("O", "V"):
            raise TypeError(f"Unsupported ndarray dtype for wire transport: {obj.dtype}")

        # `obj.tobytes()` 只保存裸数据，不保存 dtype/shape，所以必须把 dtype
        # 和 shape 一起带过去。接收端会用这三样东西重建 ndarray。
        return {
            "__ndarray__": True,
            "dtype": obj.dtype.str,
            "shape": obj.shape,
            "data": obj.tobytes(),
        }

    # numpy 标量不是 Python 原生 int/float，msgpack 不一定知道怎么处理。
    # 这里保存 dtype 和 `.item()` 后的 Python 标量值。
    if isinstance(obj, np.generic):
        return {
            "__npgeneric__": True,
            "dtype": obj.dtype.str,
            "data": obj.item(),
        }

    # 其他类型交还给 msgpack；如果 msgpack 仍然不支持，它会自己抛错。
    return obj


def _unpack_numpy(obj: Any) -> Any:
    """把 `_pack_numpy()` 产生的特殊 dict 还原成 numpy 对象。

    `msgpack.unpackb(..., object_hook=_unpack_numpy)` 每解出一个 dict 时都会
    调用这个函数，所以这里需要先检查哨兵字段，再决定是否转换。
    """

    # 还原 ndarray：用原始 bytes、dtype、shape 构造数组。
    if "__ndarray__" in obj:
        return np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=tuple(obj["shape"]),
        ).copy()

    # 还原 numpy 标量，例如 np.float32、np.int64。
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])

    # 普通业务 dict 保持原样。
    return obj


def pack_message(payload: dict[str, Any]) -> bytes:
    """把一条业务消息编码成 msgpack bytes。

    参数：
    - `payload`：业务层 dict，例如 `{"endpoint": "get_action", ...}`。

    返回：
    - 可通过 socket 发送的 bytes，不包含长度 header。
    """

    return msgpack.packb(payload, default=_pack_numpy, use_bin_type=True)


def unpack_message(payload: bytes) -> dict[str, Any]:
    """把 msgpack bytes 解码回业务层 dict。"""

    return msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)


def send_message(conn: socket.socket, payload: dict[str, Any]) -> None:
    """通过一个已连接的 socket 发送一条完整消息。

    参数：
    - `conn`：已经建立连接的 TCP socket。
    - `payload`：要发送的业务层 dict。

    协议细节：
    - 先发 8 字节长度 header。
    - 再发 msgpack payload。
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
