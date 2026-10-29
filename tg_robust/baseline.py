"""Baseline AHP de pesos iguais (aproximação ao AHP de Nascimento et al., 2022).

Implementa uma **aproximação** simplificada do AHP original, restrita aos
cinco objetivos de desempenho propulsivo do artigo — ``Mp``, ``tb``, ``It``
e ``F`` como resíduo relativo ao requisito AO QUADRADO,
``-((X - X_alvo)/X_alvo)^2`` (Eq. 30 do artigo), e ``rhoIsp`` maximizado — com
pesos iguais e avaliados no estado nominal (sem propagação de cenários,
pois o AHP original não trata incerteza de modelo). ``pcmax`` e ``eta`` são
excluídos por serem tratados como restrições (e não como objetivos) no
artigo original.

Procedimento: benefício → normalização min-max → normalização por coluna
(soma 1, como no bloco $W_{AT}$) → soma com pesos iguais → ranking.

Com os dados reais do 1º estágio (tabela do modeFRONTIER, 437 ou 632
designs), escolhe o Id 5763, o mesmo do AHP do artigo (Tabela 7). Com o
resíduo em valor absoluto, o escolhido seria o Id 5732 (5763 em 4º). Nenhum
valor da Tabela 7 é codificado aqui; ver ``tests/test_baseline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

OBJETIVOS_BASELINE: tuple[str, ...] = ("Mp", "tb", "It", "F", "rhoIsp")


def beneficio_artigo(criterio: str, valores: np.ndarray, requisitos: config.Requisitos) -> np.ndarray:
    """Benefício do AHP do artigo: ``-((X - X_alvo)/X_alvo)^2`` nos resíduos; ``rhoIsp`` direto."""

    if criterio in config.CRITERIOS_RESIDUO:
        alvo = getattr(requisitos, criterio)
        return -(((valores - alvo) / alvo) ** 2)
    return valores


def ahp_baseline(designs: pd.DataFrame, requisitos: config.Requisitos) -> pd.DataFrame:
    """Calcula o ranking baseline AHP de pesos iguais, no estado nominal.

    Retorna um DataFrame indexado como ``designs``, com as colunas
    ``score_ahp_baseline`` (soma ponderada) e ``rank_ahp_baseline``
    (1 = melhor).
    """

    n = len(designs)
    R = len(OBJETIVOS_BASELINE)
    peso = 1.0 / R

    G = np.column_stack(
        [beneficio_artigo(c, designs[c].to_numpy(dtype=float), requisitos) for c in OBJETIVOS_BASELINE]
    )

    gmin = G.min(axis=0, keepdims=True)
    gmax = G.max(axis=0, keepdims=True)
    amplitude = gmax - gmin
    G_norm = np.zeros_like(G)
    constante = (amplitude[0] == 0.0)
    G_norm[:, ~constante] = (G[:, ~constante] - gmin[:, ~constante]) / amplitude[:, ~constante]

    somas = G_norm.sum(axis=0)
    W = np.empty_like(G_norm)
    for r in range(R):
        W[:, r] = (1.0 / n) if somas[r] == 0.0 else (G_norm[:, r] / somas[r])

    score = (W * peso).sum(axis=1)
    rank = pd.Series(score, index=designs.index).rank(ascending=False, method="min").astype(int)

    return pd.DataFrame(
        {"score_ahp_baseline": score, "rank_ahp_baseline": rank}, index=designs.index
    )


__all__ = ["OBJETIVOS_BASELINE", "beneficio_artigo", "ahp_baseline"]
