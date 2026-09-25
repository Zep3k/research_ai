class TheoryError(RuntimeError):
    """Expected operational error that should be shown without a traceback."""


class ConfigurationError(TheoryError):
    """The local workspace configuration is missing or invalid."""


class BudgetExceededError(TheoryError):
    """A model call cannot fit inside the configured monthly budget."""


class ModelOutputError(TheoryError):
    """A provider returned output that cannot be used safely."""


class TrustError(TheoryError):
    """A research write would violate an epistemic trust rule."""
