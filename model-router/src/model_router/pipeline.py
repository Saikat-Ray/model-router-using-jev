from __future__ import annotations

from .executor import ExecResult, ExecutionError, Executor
from .logger import DecisionLog
from .router import Decision, Request, Router


class Pipeline:
    """route -> execute (with fallback) -> log."""

    def __init__(self, router: Router, executor: Executor, log: DecisionLog | None = None):
        self.router, self.executor, self.log = router, executor, log

    def handle(self, request: Request) -> tuple[Decision, ExecResult]:
        decision = self.router.route(request)
        try:
            result = self.executor.run(request, decision.chain)
        except ExecutionError as e:
            if self.log:
                self.log.write(request, decision, None, error=str(e))
            raise
        if self.log:
            self.log.write(request, decision, result)
        return decision, result
