"""Baseline AHP de pesos iguais (aproximação ao AHP de Nascimento et al., 2022).

Implementa uma **aproximação** simplificada do AHP original, restrita aos
cinco objetivos de desempenho propulsivo do artigo — ``Mp``, ``tb``, ``It``
e ``F`` como resíduo relativo ao requisito, e ``rhoIsp`` maximizado — com
pesos iguais e avaliados no estado nominal (sem propagação de cenários,
pois o AHP original não trata incerteza de modelo). ``pcmax`` e ``eta`` são
excluídos por serem tratados como restrições (e não como objetivos) no
artigo original.

Procedimento: benefício → normalização min-max → normalização por coluna
(soma 1, como no bloco $W_{AT}$) → soma com pesos iguais → ranking.

**Esta é apenas uma aproximação.** Com os dados reais do 1º estágio (tabela
do modeFRONTIER, 437 designs), ela escolhe o Id 5732 e põe o Id 5763 — o
escolhido pelo AHP do artigo (Tabela 7) — em 4º lugar. Nenhum valor da
Tabela 7 é codificado aqui; ver ``tests/test_baseline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stage1

OBJETIVOS_BASELINE: tuple[str, ...] = ("Mp", "tb", "It", "F", "rhoIsp")


def ahp_baseline(designs: pd.DataFrame, requisitos: config.Requisitos) -> pd.DataFrame:
    """Calcula o ranking baseline AHP de pesos iguais, no estado nominal.

    Retorna um DataFrame indexado como ``designs``, com as colunas
    ``score_ahp_baseline`` (soma ponderada) e ``rank_ahp_baseline``
    (1 = melhor).
    """

    n = len(designs)
    R = len(OBJETIVOS_BASELINE)
    peso = 1.0 / R

    attrs = {c: designs[c].to_numpy(dtype=float) for c in OBJETIVOS_BASELINE}
    G = np.column_stack([stage1.beneficio(c, attrs, requisitos) for c in OBJETIVOS_BASELINE])

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


__all__ = ["OBJETIVOS_BASELINE", "ahp_baseline"]
