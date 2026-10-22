from __future__ import annotations


class MlpbenchError(Exception):
    pass


class DataRootNotFound(MlpbenchError):
    pass


class UnknownModelError(MlpbenchError):
    pass


class BackendUnavailable(MlpbenchError):
    pass


class ProxyUnreachable(MlpbenchError):
    pass


class SandboxUnavailable(MlpbenchError):
    pass


class LeanUnavailable(MlpbenchError):
    pass


class GoldLeak(MlpbenchError):
    pass


class BenchmarkInvalid(MlpbenchError):
    pass
