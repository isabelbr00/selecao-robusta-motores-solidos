"""Julgamentos de Saaty para a dependência interna entre critérios (W_TT).

Para cada critério-alvo (coluna de ``W_TT``), os critérios que o influenciam
são comparados paritariamente na Escala Fundamental de Saaty. Este módulo
fornece:

* a Escala de Saaty e uma função de ajuste (*snap*) de uma razão arbitrária
  para o valor mais próximo da escala (em log);
* :func:`from_intensities`, que monta a matriz de comparação paritária
  reciproca $A_r$ a partir de intensidades de influência absolutas
  (um número por critério influenciador, não uma razão par-a-par direta);
* o cálculo de prioridades (autovetor principal), $\\lambda_{max}$, CI e CR;
* :func:`build_WTT_placeholder`, que monta a matriz $W_{TT}$ 7x7 a partir do
  rascunho provisório de intensidades fornecido na especificação da TG —
  **marcado explicitamente como PLACEHOLDER**, a ser validado pelo autor e
  pelo orientador;
* :func:`perturb`, usada na análise de sensibilidade do ANP.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from . import config

# ─────────────────────────────────────────────────────────────────────────
# Escala Fundamental de Saaty: {1/9, ..., 1/2, 1, 2, ..., 9}
# ─────────────────────────────────────────────────────────────────────────
SAATY_SCALE: tuple[float, ...] = tuple(
    sorted({1.0} | {float(i) for i in range(2, 10)} | {1.0 / i for i in range(2, 10)})
)


def snap_to_saaty_scale(x: float) -> float:
    """Ajusta ``x`` para o valor mais próximo da Escala de Saaty, em log.

    O ajuste é feito minimizando ``|log(x) - log(s)|`` sobre ``s`` na escala,
    de modo que razões simétricas (``x`` e ``1/x``) sejam ajustadas a
    valores também reciprocamente simétricos na escala.
    """

    if x <= 0:
        raise ValueError("x deve ser positivo")
    log_x = np.log(x)
    diffs = [abs(log_x - np.log(s)) for s in SAATY_SCALE]
    return SAATY_SCALE[int(np.argmin(diffs))]


# ─────────────────────────────────────────────────────────────────────────
# Rascunho provisório (PLACEHOLDER) de intensidades de influência
# ─────────────────────────────────────────────────────────────────────────
# Quem influencia a coluna (critério-alvo), com intensidade absoluta na
# Escala de Saaty. AINDA NÃO VALIDADO pelo autor/orientador — ver aviso em
# build_WTT_placeholder().
PLACEHOLDER_INFLUENCIAS: Mapping[str, Mapping[str, float]] = {
    "Mp": {"It": 9, "rhoIsp": 5, "tb": 3, "eta": 3},
    "tb": {"pcmax": 5, "F": 5, "It": 3, "Mp": 3},
    "It": {"Mp": 9, "rhoIsp": 7, "F": 5, "tb": 5},
    "F": {"pcmax": 9, "It": 5, "tb": 5, "rhoIsp": 3},
    "rhoIsp": {"pcmax": 5, "F": 3},
    "pcmax": {"F": 9, "tb": 3, "rhoIsp": 3, "eta": 3},
    "eta": {"Mp": 7, "pcmax": 3, "It": 3},
}


@dataclass(frozen=True)
class ColumnJudgment:
    """Matriz de julgamento de uma coluna (critério-alvo) de W_TT."""

    target: str
    labels: tuple[str, ...]  # critérios influenciadores, mesma ordem de A/linhas
    A: np.ndarray  # matriz de comparação paritária reciproca (len(labels) x len(labels))


def from_intensities(target: str, intensities: Mapping[str, float]) -> ColumnJudgment:
    """Monta a matriz de comparação paritária reciproca $A$ de uma coluna.

    ``intensities`` associa, a cada critério influenciador do ``target``,
    uma intensidade absoluta de influência (não uma razão par-a-par). A
    razão $s_i/s_j$ entre cada par de influenciadores é ajustada ao valor
    mais próximo da Escala de Saaty (:func:`snap_to_saaty_scale`); a
    reciprocidade é garantida por construção (``A[j,i] = 1/A[i,j]``).
    """

    labels = tuple(intensities.keys())
    n = len(labels)
    s = np.array([intensities[l] for l in labels], dtype=float)

    A = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            valor = snap_to_saaty_scale(s[i] / s[j])
            A[i, j] = valor
            A[j, i] = 1.0 / valor

    return ColumnJudgment(target=target, labels=labels, A=A)


def eigen_priorities(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Autovetor principal normalizado (soma 1) e $\\lambda_{max}$ de ``A``."""

    autovalores, autovetores = np.linalg.eig(A)
    idx = int(np.argmax(autovalores.real))
    lam_max = float(autovalores[idx].real)
    w = np.abs(autovetores[:, idx].real)
    w = w / w.sum()
    return w, lam_max


def consistency(A: np.ndarray) -> tuple[float, float, float]:
    """$\\lambda_{max}$, CI e CR de uma matriz de comparação paritária ``A``."""

    _, lam_max = eigen_priorities(A)
    n = A.shape[0]
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = config.RI_SAATY.get(n, config.RI_SAATY[max(config.RI_SAATY)])
    CR = CI / RI if RI > 0 else 0.0
    return lam_max, CI, CR


def build_WTT(
    influencias: Mapping[str, Mapping[str, float]],
    criterios: tuple[str, ...] = config.CRITERIOS,
) -> tuple[np.ndarray, list[ColumnJudgment]]:
    """Constrói $W_{TT}$ (estocástica por coluna) a partir de intensidades.

    Para cada critério-alvo em ``criterios``, monta o julgamento da coluna
    (:func:`from_intensities`), calcula as prioridades locais (autovetor) e
    as insere na coluna correspondente de ``W_TT``, nas linhas dos critérios
    influenciadores; critérios não listados como influenciadores recebem
    peso 0 naquela coluna. Colunas sem nenhum influenciador (não deveria
    ocorrer no rascunho) recebem distribuição uniforme sobre os demais
    critérios, para preservar a propriedade estocástica.

    Retorna ``(W_TT, julgamentos)``, em que ``julgamentos`` é a lista de
    :class:`ColumnJudgment` (uma por coluna), usada por
    :func:`consistency_report`.
    """

    n = len(criterios)
    idx = {c: i for i, c in enumerate(criterios)}
    W_TT = np.zeros((n, n), dtype=float)
    julgamentos: list[ColumnJudgment] = []

    for alvo in criterios:
        influenciadores = influencias.get(alvo, {})
        coluna = idx[alvo]
        if not influenciadores:
            outros = [c for c in criterios if c != alvo]
            for c in outros:
                W_TT[idx[c], coluna] = 1.0 / len(outros)
            continue

        julgamento = from_intensities(alvo, influenciadores)
        w, _ = eigen_priorities(julgamento.A)
        for label, peso in zip(julgamento.labels, w):
            W_TT[idx[label], coluna] = peso
        julgamentos.append(julgamento)

    return W_TT, julgamentos


def build_WTT_placeholder(
    criterios: tuple[str, ...] = config.CRITERIOS,
) -> tuple[np.ndarray, list[ColumnJudgment]]:
    """Constrói $W_{TT}$ a partir do rascunho PLACEHOLDER de intensidades.

    Emite um aviso explícito: os julgamentos ainda não foram validados pelo
    autor nem pelo orientador (Seção de julgamentos da especificação).
    """

    aviso = (
        "W_TT construída a partir de um RASCUNHO PROVISÓRIO (PLACEHOLDER) de "
        "intensidades de influência entre critérios, ainda não validado pelo "
        "autor nem pelo orientador. Não usar para conclusões finais da TG "
        "sem substituir judgments.PLACEHOLDER_INFLUENCIAS pelos julgamentos "
        "definitivos."
    )
    warnings.warn(aviso, stacklevel=2)
    return build_WTT(PLACEHOLDER_INFLUENCIAS, criterios)


def consistency_report(julgamentos: list[ColumnJudgment]) -> pd.DataFrame:
    """Relatório de razão de consistência (CR) por coluna de julgamento.

    Sinaliza (coluna ``cr_excede_limite``) colunas com ``CR > 0,10``
    (:data:`config.CR_LIMITE`), conforme Saaty (1980).
    """

    linhas = []
    for j in julgamentos:
        lam_max, CI, CR = consistency(j.A)
        linhas.append(
            {
                "alvo": j.target,
                "n_influenciadores": len(j.labels),
                "lambda_max": lam_max,
                "CI": CI,
                "CR": CR,
                "cr_excede_limite": CR > config.CR_LIMITE,
            }
        )
    return pd.DataFrame(linhas)


def perturb(
    julgamentos: Mapping[str, ColumnJudgment], col: str, a: str, b: str, step: int
) -> dict[str, ColumnJudgment]:
    """Perturba o julgamento entre os critérios ``a`` e ``b`` na coluna ``col``.

    Desloca o valor $A_{ab}$ em ``step`` níveis da Escala de Saaty
    (:data:`SAATY_SCALE`), preservando a reciprocidade ($A_{ba} = 1/A_{ab}$)
    e mantendo o valor dentro dos limites da escala. Não modifica
    ``julgamentos`` — devolve um dicionário novo com a coluna ``col``
    substituída.

    ``julgamentos`` é um dicionário ``{alvo: ColumnJudgment}`` (p.ex. o
    resultante de indexar a lista de :func:`build_WTT` pelo campo
    ``target``).
    """

    if col not in julgamentos:
        raise KeyError(f"Coluna/alvo {col!r} não encontrada nos julgamentos")
    julgamento = julgamentos[col]
    if a not in julgamento.labels or b not in julgamento.labels:
        raise KeyError(f"Critérios {a!r}/{b!r} não são influenciadores de {col!r}")

    i, k = julgamento.labels.index(a), julgamento.labels.index(b)
    A_novo = julgamento.A.copy()

    valor_atual = A_novo[i, k]
    pos_atual = SAATY_SCALE.index(snap_to_saaty_scale(valor_atual))
    pos_novo = int(np.clip(pos_atual + step, 0, len(SAATY_SCALE) - 1))
    valor_novo = SAATY_SCALE[pos_novo]

    A_novo[i, k] = valor_novo
    A_novo[k, i] = 1.0 / valor_novo

    novo = dict(julgamentos)
    novo[col] = ColumnJudgment(target=julgamento.target, labels=julgamento.labels, A=A_novo)
    return novo


def julgamentos_por_alvo(julgamentos: list[ColumnJudgment]) -> dict[str, ColumnJudgment]:
    """Indexa a lista de :class:`ColumnJudgment` por critério-alvo (``target``)."""

    return {j.target: j for j in julgamentos}


def WTT_de_julgamentos(
    julgamentos: Mapping[str, ColumnJudgment], criterios: tuple[str, ...] = config.CRITERIOS
) -> np.ndarray:
    """Reconstrói a matriz $W_{TT}$ a partir de um dicionário de julgamentos.

    Útil após :func:`perturb`, para obter a nova supermatriz sem refazer
    toda a elicitação.
    """

    n = len(criterios)
    idx = {c: i for i, c in enumerate(criterios)}
    W_TT = np.zeros((n, n), dtype=float)
    for alvo, julgamento in julgamentos.items():
        w, _ = eigen_priorities(julgamento.A)
        for label, peso in zip(julgamento.labels, w):
            W_TT[idx[label], idx[alvo]] = peso
    return W_TT


__all__ = [
    "SAATY_SCALE",
    "snap_to_saaty_scale",
    "PLACEHOLDER_INFLUENCIAS",
    "ColumnJudgment",
    "from_intensities",
    "eigen_priorities",
    "consistency",
    "build_WTT",
    "build_WTT_placeholder",
    "consistency_report",
    "perturb",
    "julgamentos_por_alvo",
    "WTT_de_julgamentos",
]
