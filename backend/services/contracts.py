"""The prediction and explanation interfaces the API depends on.

The routers must not know which model answers a request. They know these two
protocols; `backend.services.ml` is the composition root that builds concrete
implementations at startup and hands them back typed as the protocol.

Why protocols rather than base classes: the current implementations live in the
`ml` package and are constructed from a saved artifact, so there is nothing to
inherit from without the API layer reaching into ML internals. A `Protocol` is
satisfied structurally — a future model service implements these method names
and is accepted, with no import of, or edit to, this file.

Replacing the served model therefore means writing one class that satisfies
`PredictionModel` and one that satisfies `ExplanationProvider`, then registering
them in `backend.services.ml.startup`. No router, schema or frontend type
changes.

Deliberately small: these declare only what the routers actually call. Anything
else stays an implementation detail, including how an artifact is loaded and
cached.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PredictionOutcome(Protocol):
    """One scored record, in whatever form the model produced it."""

    def as_dict(self) -> dict[str, Any]:
        """Flat mapping. Must carry at least `probability`, `prediction`,
        `risk`, `risk_category`, `threshold`, `model_version`, `feature_set`
        and `preprocessing_version` — the columns a prediction row stores."""
        ...


@runtime_checkable
class PredictionModel(Protocol):
    """Something that can score a record and describe what it accepts."""

    @property
    def version(self) -> str:
        """Artifact version stored with every prediction, e.g. 'mmc2-mmc3-v1'."""
        ...

    @property
    def feature_set(self) -> str:
        """Which input block this model was fitted on."""
        ...

    def input_schema(self) -> dict[str, Any]:
        """Fields accepted and values allowed, for the form and for validation.

        This is the single source of truth for request validation: the API
        never keeps its own copy of a column list, so the two cannot disagree.
        """
        ...

    def transform(self, payload: dict[str, Any]) -> Any:
        """Validated input -> the matrix an explainer can be run over."""
        ...

    def predict(self, payload: dict[str, Any]) -> PredictionOutcome:
        """Score one record, or raise for invalid input."""
        ...


@runtime_checkable
class ExplanationProvider(Protocol):
    """Local and global attribution over a model's own inputs."""

    def explain(
        self,
        matrix: Any,
        supplied: dict[str, Any],
        top_n: int = 8,
        methods: tuple[str, ...] = ("shap", "lime"),
    ) -> dict[str, Any]:
        """Per-method attribution for one transformed row.

        Contributions are reported against **source column names**, not encoded
        feature names, which is what lets the frontend render any model's
        explanation without knowing its feature space.
        """
        ...

    def global_importance(self, top_n: int = 15) -> dict[str, Any]:
        """Model-wide importance. Describes the model, never an individual."""
        ...
