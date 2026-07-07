"""GR00T N1.7 bridge server：在 GR00T 虚拟环境中加载模型并对外提供推理服务。

运行环境：
    ${GROOT_ROOT:-$HOME/Isaac-GR00T-py312}/.venv/bin/python

为什么需要这个 bridge：
    Isaac Sim / IsaacLab 的环境非常重，GR00T N1.7 的依赖也非常重。两者直接
    装在同一个 Python 环境里，容易产生 CUDA、Torch、Transformers、Isaac
    扩展包之间的版本冲突。这个文件把 GR00T 模型加载在它自己的 venv 中，
    再通过本地 TCP socket 接收 Isaac 侧传来的 observation，返回 action。

通信协议：
    具体的 socket + pickle 序列化逻辑在 `wire.py` 中。这里使用四个 endpoint：

    - `ping`：Isaac 侧用来确认 bridge 已启动，并读取 GR00T modality schema。
    - `get_action`：Isaac 侧传 observation，GR00T 侧返回 action。
    - `reset`：预留给有状态 policy，当前主要用于兼容接口。
    - `shutdown`：请求 bridge 优雅退出。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any


# 当前文件所在目录，也就是
# experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task。
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_MODEL_PATH = "nvidia/GR00T-N1.7-3B"
DEFAULT_ZERO_SHOT_EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
FINETUNED_SO101_EMBODIMENT = "NEW_EMBODIMENT"
DEFAULT_SO101_MODALITY_CONFIG = EXPERIMENT_ROOT / "full_finetune_so101" / "so101_synthetic_groot_config.py"

# 把当前目录插入 sys.path，是为了让 bridge 可以 import 同目录下的 wire.py。
# 这里不用安装成包，保持实验脚本轻量、可直接复制。
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noqa: E402 表示忽略“import 不在文件顶部”的 lint 警告。
# 我们必须先改 sys.path，才能稳定 import 本地 wire.py。
from wire import recv_message, send_message  # noqa: E402


def _load_modality_config(path_value: str | None) -> None:
    """Load a custom GR00T modality config before resolving an embodiment tag."""

    if path_value is None:
        return

    path = Path(path_value).expanduser().resolve()
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"modality config does not exist: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load modality config: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    print(f"[bridge] loaded modality config: {path}", flush=True)


def _load_policy(args: argparse.Namespace):
    """根据命令行参数加载 GR00T policy。

    参数：
    - `args.model_path`：模型路径或 HuggingFace repo id，例如 `nvidia/GR00T-N1.7-3B`。
    - `args.embodiment_tag`：GR00T 预定义 embodiment，例如
      `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`。
    - `args.device`：推理设备，通常是 `cuda`。
    - `args.no_strict`：是否关闭 schema 严格检查。
    - `args.sim_wrapper`：是否套用 GR00T 自带的 flat sim wrapper。

    返回：
    - 一个带 `get_action()`、`get_modality_config()` 等方法的 policy 对象。
    """

    # 这些 import 必须放在函数内部，而不是文件顶部。
    # 原因：这个文件只能在 GR00T venv 里运行；如果 IsaacLab 环境误 import
    # 这个模块，顶部 import gr00t 会立刻失败。延迟 import 能让错误边界更清楚。
    _load_modality_config(args.modality_config_path)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    # `EmbodimentTag.resolve()` 支持大小写/枚举名解析，把用户传入的字符串
    # 转成 GR00T 内部认可的 embodiment enum。
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    # `Gr00tPolicy` 是官方推理入口：
    # - 读取模型权重。
    # - 读取 processor。
    # - 校验 observation schema。
    # - 调用模型生成 action。
    # - 把 action 从模型空间 decode 回物理数值空间。
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )

    # `Gr00tSimPolicyWrapper` 是 GR00T repo 里提供的兼容包装器。
    # 当前实验默认不用它，因为我们直接构造 N1.7 nested schema：
    # observation = {"video": ..., "state": ..., "language": ...}
    if args.sim_wrapper:
        policy = Gr00tSimPolicyWrapper(policy, strict=not args.no_strict)

    return policy


def _summarize_modality(policy: Any) -> dict[str, Any]:
    """把 GR00T policy 的 modality config 压缩成容易打印和传输的 dict。

    这个 summary 会返回给 Isaac runner，用来确认：
    - video 需要哪些 key。
    - state 需要哪些 key。
    - action 会输出哪些 key。
    - 每个 key 对应哪些时间偏移 `delta_indices`。
    """

    summary = {}
    for group, cfg in policy.get_modality_config().items():
        summary[group] = {
            "delta_indices": list(cfg.delta_indices),
            "modality_keys": list(cfg.modality_keys),
        }
    return summary


def serve(args: argparse.Namespace) -> None:
    """启动 bridge server 主循环。

    这个函数会一直监听 `args.host:args.port`。每次 Isaac runner 发起一个
    TCP 连接，server 就读取一条 request，处理后返回一条 response，然后关闭
    这次连接。这个“一请求一连接”的模型很简单，足够当前实验使用。
    """

    # 如果用户显式加 `--offline`，就设置 HuggingFace/Transformers 的离线变量。
    # 注意：当前 README 建议先不要用 offline，因为本机缓存路径和 processor
    # metadata 可能还会触发 HuggingFace 侧的元数据读取。
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("[bridge] loading GR00T policy...", flush=True)
    policy = _load_policy(args)
    print("[bridge] policy loaded", flush=True)
    print(f"[bridge] modality: {_summarize_modality(policy)}", flush=True)

    # `AF_INET` 表示 IPv4；`SOCK_STREAM` 表示 TCP。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # 允许进程重启后快速复用同一个端口，避免 TIME_WAIT 导致端口短暂不可用。
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定监听地址。默认是 127.0.0.1:5577，只允许本机访问。
        server.bind((args.host, args.port))

        # 开始监听。参数 1 是 backlog，当前实验只有一个 Isaac runner 连接，
        # 所以不需要更大的排队长度。
        server.listen(1)
        print(f"[bridge] listening on {args.host}:{args.port}", flush=True)

        while True:
            # 阻塞等待一个客户端连接。
            conn, addr = server.accept()

            # `with conn` 确保处理完请求后关闭这个连接。
            with conn:
                try:
                    request = recv_message(conn)
                    endpoint = request.get("endpoint", "get_action")

                    if endpoint == "ping":
                        # 健康检查：返回 ok 和 modality summary。
                        send_message(conn, {"ok": True, "modality": _summarize_modality(policy)})

                    elif endpoint == "reset":
                        # 预留接口：如果 policy 内部有历史状态，可以通过 reset 清掉。
                        send_message(conn, {"ok": True, "info": policy.reset(request.get("options"))})

                    elif endpoint == "get_action":
                        # 核心接口：Isaac 侧传入 observation，bridge 调用 GR00T 推理。
                        # observation 必须符合当前 embodiment 的 schema。
                        action, info = policy.get_action(
                            request["observation"],
                            request.get("options"),
                        )

                        # action 是 dict[str, np.ndarray]，例如：
                        # - action["eef_9d"] -> shape (B, 40, 9)
                        # - action["gripper_position"] -> shape (B, 40, 1)
                        # - action["joint_position"] -> shape (B, 40, 7)
                        send_message(conn, {"ok": True, "action": action, "info": info})

                    elif endpoint == "shutdown":
                        # 用于手动或脚本化关闭 bridge。
                        send_message(conn, {"ok": True})
                        print("[bridge] shutdown requested", flush=True)
                        return

                    else:
                        # 未知 endpoint 返回结构化错误，不让 Isaac 侧一直卡住。
                        send_message(conn, {"ok": False, "error": f"unknown endpoint: {endpoint}"})

                except Exception as exc:
                    # 保留完整 traceback，Isaac 侧日志里能看到 bridge 失败原因。
                    traceback.print_exc()
                    send_message(conn, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})

                finally:
                    print(f"[bridge] handled request from {addr}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析 bridge 的命令行参数。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deployment-mode",
        choices=("zero-shot-oxe", "so101-finetuned"),
        default="zero-shot-oxe",
        help=(
            "zero-shot-oxe loads the base OXE/DROID policy; "
            "so101-finetuned loads a checkpoint trained with NEW_EMBODIMENT."
        ),
    )

    # GR00T 模型路径。可以是 HuggingFace repo id，也可以是本地 checkpoint 目录。
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL_PATH)

    parser.add_argument(
        "--modality-config-path",
        default=None,
        help="Optional Python file that registers a custom GR00T modality config before loading the policy.",
    )

    # N1.7 base model 当前可用的预训练 embodiment 之一。
    # 这个 tag 的 video/state/action schema 被 Isaac runner 手工适配。
    parser.add_argument("--embodiment-tag", default=DEFAULT_ZERO_SHOT_EMBODIMENT)

    # 推理设备。通常用 cuda；如果只是调试 schema，也可以尝试 cpu，但会非常慢。
    parser.add_argument("--device", default="cuda")

    # bridge 监听地址。127.0.0.1 表示仅本机访问，避免暴露到局域网。
    parser.add_argument("--host", default="127.0.0.1")

    # bridge 监听端口。Isaac runner 的默认 bridge-port 也是 5577。
    parser.add_argument("--port", type=int, default=5577)

    # 强制离线模式。当前不作为默认值，原因见 README。
    parser.add_argument("--offline", action="store_true", help="Force HuggingFace/Transformers offline mode.")

    # 关闭 GR00T schema 严格检查。调试新 schema 时有用；正常实验尽量保持 strict。
    parser.add_argument("--no-strict", action="store_true", help="Disable GR00T input/output schema checks.")

    # 使用 GR00T 自带 simulation wrapper。当前实验默认不用。
    parser.add_argument(
        "--sim-wrapper",
        action="store_true",
        help="Use GR00T's flat sim-policy wrapper instead of the nested native schema.",
    )
    args = parser.parse_args()

    if args.deployment_mode == "so101-finetuned":
        if args.model_path == DEFAULT_BASE_MODEL_PATH:
            parser.error("--deployment-mode so101-finetuned requires --model-path to point at a trained checkpoint")
        if args.embodiment_tag == DEFAULT_ZERO_SHOT_EMBODIMENT:
            args.embodiment_tag = FINETUNED_SO101_EMBODIMENT
        elif args.embodiment_tag.upper() != FINETUNED_SO101_EMBODIMENT:
            parser.error("--deployment-mode so101-finetuned requires --embodiment-tag NEW_EMBODIMENT")
        else:
            args.embodiment_tag = FINETUNED_SO101_EMBODIMENT
        if args.modality_config_path is None:
            args.modality_config_path = str(DEFAULT_SO101_MODALITY_CONFIG)

    return args


if __name__ == "__main__":
    serve(parse_args())
