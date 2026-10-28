"""Fixtures compartilhadas dos testes do tg_robust."""

from __future__ import annotations

import warnings

import pytest


@pytest.fixture(autouse=True)
def _silenciar_avisos_esperados():
    """Evita que os avisos esperados (dados sintéticos, W_TT placeholder, n
    padrão) poluam a saída dos testes, sem afetar a captura explícita de
    avisos feita por testes que a verificam (``pytest.warns``)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
