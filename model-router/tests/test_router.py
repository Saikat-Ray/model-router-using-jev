import json

import httpx
import pytest

from model_router.executor import ExecutionError, Executor
from model_router.jev_client import ChoiceAnswer, JevError
from model_router.registry import ModelSpec, Registry
from model_router.router import NoEligibleModel, Request, Router


def spec(name, rank, ctx=100_000, vision=True):
    return ModelSpec(name=name, id=f"x/{name}", description=name, rank=rank,
                     context_window=ctx, vision=vision)


@pytest.fixture
def registry():
    return Registry([spec("small", 1), spec("medium", 2), spec("large", 3)],
                    default="medium", confidence_threshold=0.6)


class Fake:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def choose(self, state, instructions, options):
        self.calls.append(options)
        if self.error:
            raise self.error
        return self.answer


def test_confident_pick_is_used(registry):
    d = Router(registry, Fake(ChoiceAnswer("small", 0.9))).route(Request("hi"))
    assert d.model.name == "small" and d.reason == "jev"
    assert [m.name for m in d.chain] == ["small", "medium", "large"]


def test_low_confidence_escalates_to_strongest(registry):
    d = Router(registry, Fake(ChoiceAnswer("small", 0.3))).route(Request("hi"))
    assert d.model.name == "large" and d.reason == "low_confidence_escalated"


def test_jev_outage_uses_default(registry):
    d = Router(registry, Fake(error=JevError("down"))).route(Request("hi"))
    assert d.model.name == "medium" and d.reason == "jev_error"


def test_hard_constraints_filter_options_before_jev():
    reg = Registry([spec("small", 1, ctx=1_000, vision=False), spec("large", 3, ctx=100_000)])
    fake = Fake(ChoiceAnswer("large", 0.9))
    Router(reg, fake).route(Request("x" * 8_000, has_image=True))
    assert fake.calls == []  # only one eligible model -> Jev is skipped


def test_nothing_fits_raises():
    reg = Registry([spec("small", 1, ctx=10)])
    with pytest.raises(NoEligibleModel):
        Router(reg, Fake()).route(Request("x" * 1000))


def test_executor_falls_back_on_failure(registry):
    def handler(request):
        assert request.headers.get("Authorization") == "Bearer k"
        if b'"x/small"' in request.content:
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"total_tokens": 5}})

    ex = Executor("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    chain = [registry.get("small"), registry.get("medium")]
    result = ex.run(Request("hi"), chain)
    assert result.model.name == "medium" and result.failures


def test_executor_raises_when_all_fail(registry):
    ex = Executor("k", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(ExecutionError):
        ex.run(Request("hi"), [registry.get("small")])


def test_executor_uses_local_ollama_api(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.local:11434/")

    def handler(request):
        assert str(request.url) == "http://ollama.local:11434/api/chat"
        body = json.loads(request.read())
        assert body["model"] == "qwen2.5:7b"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False
        assert request.headers.get("Authorization") is None
        return httpx.Response(200, json={
            "message": {"content": "local answer"},
            "prompt_eval_count": 4,
            "eval_count": 6,
        })

    model = ModelSpec(
        name="ollama-local", id="qwen2.5:7b", description="local",
        rank=1, context_window=32768, provider="ollama",
    )
    ex = Executor(client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = ex.run(Request("hi"), [model])
    assert result.text == "local answer"
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 6}


def test_config_includes_ollama_and_deepseek():
    from pathlib import Path

    path = Path(__file__).parents[1] / "config" / "models.yaml"
    registry = Registry.from_yaml(path)
    assert registry.get("ollama-local").provider == "ollama"
    assert registry.get("deepseek").id == "deepseek/deepseek-chat"
    assert registry.get("deepseek").provider == "openrouter"


def test_cli_finds_default_config_outside_project_directory(monkeypatch, tmp_path, capsys):
    from model_router import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "JevClient", lambda: Fake(ChoiceAnswer("medium", 0.9)))

    assert cli.main(["--dry-run", "hi"]) == 0
    assert "pick=medium" in capsys.readouterr().out


def test_cli_loads_project_dotenv_before_creating_jev_client(
    monkeypatch, tmp_path, capsys
):
    import os
    from pathlib import Path

    from model_router import cli

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def make_jev_client():
        assert os.environ["OPENROUTER_API_KEY"] == "from-dotenv"
        return Fake(ChoiceAnswer("medium", 0.9))

    monkeypatch.setattr(cli, "JevClient", make_jev_client)

    config = Path(__file__).parents[1] / "config" / "models.yaml"
    assert cli.main(["--config", str(config), "--dry-run", "hi"]) == 0
    assert "pick=medium" in capsys.readouterr().out
