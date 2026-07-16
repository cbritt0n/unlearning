"""HTTP webhook and crypto-shred hook unit tests."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from compliance.crypto_shred import CryptoShredVault
from integrations.hooks import (
    HttpWebhookHook,
    LocalBackupAckHook,
    make_crypto_shred_hook,
)


def test_local_backup_ack() -> None:
    hook = LocalBackupAckHook(auto_ack=True)
    out = hook("docs", ["a", "b"])
    assert out["acked"] is True


def test_crypto_shred_hook() -> None:
    vault = CryptoShredVault()
    vault.provision("user-1")
    vault.encrypt("user-1", b"payload")
    hook = make_crypto_shred_hook(vault)
    result = hook(["user-1"])
    assert result["ok"] is True
    # Second shred is still marked shredded=True (idempotent mark)
    result2 = hook(["user-1"])
    assert result2["ok"] is True
    assert vault.is_shredded("user-1")


def test_http_webhook_hook() -> None:
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received["body"] = json.loads(body.decode("utf-8"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args):  # noqa: D401
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        hook = HttpWebhookHook(
            f"http://127.0.0.1:{port}/hook",
            event="document_store_delete",
            timeout_s=2.0,
        )
        out = hook("docs", ["id1"], request_id="r1", reason="test")
        assert out["ok"] is True
        assert received["body"]["ids"] == ["id1"]
        assert received["body"]["event"] == "document_store_delete"
    finally:
        server.shutdown()
