import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coli_nex


class FakeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 64, "total_tokens": 68},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class ColiNexTests(unittest.TestCase):
    def test_validate_exact_config(self):
        layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(60)]
        config = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": dict(coli_nex.EXPECTED_TEXT_CONFIG,
                                max_position_embeddings=262144,
                                layer_types=layer_types),
        }
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text(json.dumps(config), encoding="utf-8")
            report = coli_nex.validate_model_config(folder)
        self.assertEqual(report["experts"], 512)
        self.assertEqual(report["active_experts"], 10)
        self.assertEqual(report["linear_attention_layers"], 45)

    def test_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text('{"model_type":"glm_moe_dsa"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                coli_nex.validate_model_config(folder)

    def test_fast_profile_contains_performance_controls(self):
        profile = coli_nex.PROFILES["fast-h200x8-fp8"]
        command = coli_nex.build_launch_command(profile)
        joined = " ".join(command)
        self.assertIn("nex-agi/Nex-N2-Pro-fp8", joined)
        self.assertIn("--mamba-scheduler-strategy extra_buffer", joined)
        self.assertIn("--speculative-algorithm NEXTN", joined)
        self.assertIn("--tp 8", joined)

    def test_multinode_requires_address(self):
        profile = coli_nex.PROFILES["reference-h100x16-bf16"]
        with self.assertRaises(ValueError):
            coli_nex.build_launch_command(profile)

    def test_benchmark_gate_uses_reported_tokens(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            report = coli_nex.benchmark_endpoint(
                endpoint, prompt="test", requests=3, concurrency=2,
                max_tokens=64, warmup=0, timeout=5,
            )
            self.assertEqual(report["completion_tokens"], 192)
            self.assertGreater(report["aggregate_tps"], 20)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
