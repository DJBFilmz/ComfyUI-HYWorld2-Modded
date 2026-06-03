import argparse
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler, Qwen3VLChatHandler


def _load_chat_handler(kind: str, clip_model_path: str, image_max_tokens: int):
    kind = kind.lower().replace("_", "-")
    handler_cls = Qwen3VLChatHandler if kind in {"qwen3-vl", "qwen3"} else Qwen25VLChatHandler
    kwargs = {
        "clip_model_path": clip_model_path,
        "image_max_tokens": image_max_tokens,
        "force_reasoning": False,
        "verbose": False,
    }
    try:
        return handler_cls(**kwargs)
    except TypeError:
        kwargs.pop("image_max_tokens", None)
        kwargs.pop("force_reasoning", None)
        return handler_cls(**kwargs)


def create_app(args: argparse.Namespace) -> FastAPI:
    chat_handler = _load_chat_handler(args.handler, args.clip_model_path, args.image_max_tokens)
    llm_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "chat_handler": chat_handler,
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_gpu_layers": args.n_gpu_layers,
        "verbose": args.verbose,
        "swa_full": True,
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
    }
    if args.pool_size is not None:
        llm_kwargs["pool_size"] = args.pool_size
    if args.top_k is not None:
        llm_kwargs["top_k"] = args.top_k

    llm = Llama(**llm_kwargs)
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_alias,
                    "object": "model",
                    "owned_by": "me",
                    "permissions": [],
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        try:
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=int(body.get("max_tokens", 1024)),
                temperature=float(body.get("temperature", 0.1)),
                top_p=float(body.get("top_p", 1.0)),
                repeat_penalty=float(body.get("repeat_penalty", 1.0)),
                seed=int(body.get("seed", args.seed)),
                stop=body.get("stop") or ["<|im_end|>", "<|im_start|>"],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        result.setdefault("object", "chat.completion")
        result.setdefault("created", int(time.time()))
        result.setdefault("model", body.get("model") or args.model_alias)
        return result

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--clip_model_path", required=True)
    parser.add_argument("--model_alias", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--handler", default="qwen25-vl", choices=["qwen25-vl", "qwen3-vl"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n_ctx", type=int, default=32768)
    parser.add_argument("--n_batch", type=int, default=512)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--image_min_tokens", type=int, default=1024)
    parser.add_argument("--image_max_tokens", type=int, default=2048)
    parser.add_argument("--pool_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    server_args = parse_args()
    uvicorn.run(create_app(server_args), host=server_args.host, port=server_args.port)
