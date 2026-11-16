import os


def main() -> None:
    import uvicorn

    from .server import create_app

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Weather agent on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)
