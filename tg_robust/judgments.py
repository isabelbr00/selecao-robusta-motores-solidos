"""Julgamentos de Saaty para a dependência interna entre critérios (W_TT).

Para cada critério-alvo (coluna de ``W_TT``), os critérios que o influenciam
são comparados paritariamente na Escala Fundamental de Saaty. Este módulo
fornece:

* a Escala de Saaty e uma função de ajuste (*snap*) de uma razão arbitrária
  para o valor mais próximo da escala (em log), usada só na perturbação;
* :func:`from_intensities`, que monta a matriz de comparação paritária
  reciproca $A_r$ a partir de intensidades de influência absolutas
  (um número por critério influenciador), com as razões $I_i/I_j$ exatas —
  as prioridades resultantes são as intensidades normalizadas e a RC é 0;
* o cálculo de prioridades (autovetor principal), $\\lambda_{max}$, CI e CR;
* :func:`build_WTT_validada`, que monta a matriz $W_{TT}$ dos critérios em uso a
  partir das intensidades validadas pelo orientador (:data:`INFLUENCIAS`);
* :func:`perturb`, usada na análise de sensibilidade do ANP.
"""

from __future__ import annotations

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
# Intensidades de influência validadas pelo orientador
# ─────────────────────────────────────────────────────────────────────────
# Quem influencia a coluna (critério-alvo), com intensidade absoluta na
# Escala de Saaty, sobre os 7 atributos. Cada variante usa só as relações entre
# os seus critérios (build_WTT): no padrão de 6, somem as que envolvem pcmax.
INFLUENCIAS: Mapping[str, Mapping[str, float]] = {
    "Mp": {"It": 9, "rhoIsp": 5, "tb": 3, "eta": 5},
    "tb": {"pcmax": 5, "F": 5, "It": 3, "Mp": 3},
    "It": {"Mp": 7, "rhoIsp": 7, "F": 5, "tb": 5},
    "F": {"pcmax": 9, "It": 5, "tb": 5, "rhoIsp": 3},
    "rhoIsp": {"pcmax": 5},
    "pcmax": {"F": 9, "tb": 3, "rhoIsp": 3, "eta": 3},
    "eta": {"Mp": 7},
}

# Justificativa física de cada relação (critério influenciado, influenciador),
# em uma frase. Notação: subscritos e sobrescritos como em LaTeX (M_p, p_{c,max},
# p_c^n), para uso direto em relatórios.
JUSTIFICATIVAS: Mapping[tuple[str, str], str] = {
    ("Mp", "It"): "I_t = M_p·I_{sp} (I_{sp} em m/s): para um I_{sp} dado, o impulso exigido fixa a massa de propelente.",
    ("Mp", "rhoIsp"): "ρ_p define quanta massa cabe na câmara (M_p = ρ_p η_v V_c) e I_{sp}, quanta massa é preciso para um dado I_t.",
    ("Mp", "tb"): "M_p = ṁ·t_b: com a vazão limitada pela área de queima, queima mais longa exige mais propelente.",
    ("Mp", "eta"): "M_p = ρ_p η_v V_c: no mesmo envelope, η_v fixa a fração ocupada por propelente.",
    ("tb", "pcmax"): "r = a·p_c^n: pressão de câmara (representada por p_{c,max}) maior acelera a regressão e encurta a queima da mesma espessura de web.",
    ("tb", "F"): "t_b = I_t/F: para impulso fixo, empuxo médio maior implica queima mais curta.",
    ("tb", "It"): "t_b = I_t/F: para empuxo fixo, mais impulso exige queima mais longa.",
    ("tb", "Mp"): "t_b = M_p/ṁ: mais propelente queimando à mesma vazão alonga a queima.",
    ("It", "Mp"): "I_t = M_p·I_{sp} (I_{sp} em m/s): impulso total proporcional à massa de propelente.",
    ("It", "rhoIsp"): "O I_{sp} (em m/s) multiplica a massa em I_t = M_p·I_{sp}; a densidade só atua via M_p.",
    ("It", "F"): "I_t = F·t_b (identidade com o empuxo médio).",
    ("It", "tb"): "I_t = F·t_b (identidade com o tempo de queima).",
    ("F", "pcmax"): "F = C_F·p_c·A_t: para tubeira fixa, empuxo proporcional à pressão de câmara (representada por p_{c,max}).",
    ("F", "It"): "F = I_t/t_b: mais impulso no mesmo tempo eleva o empuxo médio.",
    ("F", "tb"): "F = I_t/t_b: queima mais curta com o mesmo impulso eleva o empuxo médio.",
    ("F", "rhoIsp"): "F = ṁ·I_{sp}: maior I_{sp} dá mais empuxo à mesma vazão mássica.",
    ("rhoIsp", "pcmax"): "Com pressão ambiente, C_F cresce com p_c pelo termo (p_e − p_{ext})·ε/p_c; no vácuo, a dependência é fraca.",
    ("pcmax", "F"): "F = C_F·p_c·A_t: com a tubeira fixa, exigir mais empuxo exige pressão de câmara (representada por p_{c,max}) maior.",
    ("pcmax", "tb"): "r = a·p_c^n: queimar a mesma web em menos tempo exige pressão de câmara (representada por p_{c,max}) maior.",
    ("pcmax", "rhoIsp"): "p_c = (ρ_p c^* a·A_b/A_t)^{1/(1−n)}: ρ_p e c^* entram diretamente na pressão de equilíbrio.",
    ("pcmax", "eta"): "Carregamento maior tende a elevar a razão A_b/A_t (K_n) e, com ela, a pressão.",
    ("eta", "Mp"): "η_v = M_p/(ρ_p V_c): no envelope dado, é a própria fração de propelente.",
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
    uma intensidade absoluta de influência na Escala de Saaty (1 a 9). Cada
    elemento é a razão exata $A_{ij} = I_i/I_j$, **sem** arredondar para a
    escala: arredondar comprimia os julgamentos (p.ex. 9/7 ≈ 1,3 virava 1).
    A matriz é recíproca e perfeitamente consistente por construção, logo
    o autovetor principal é o vetor de intensidades normalizado e a RC é 0.
    """

    labels = tuple(intensities.keys())
    s = np.array([intensities[l] for l in labels], dtype=float)
    if not (s > 0).all():
        raise ValueError(f"Intensidades devem ser positivas (alvo {target!r})")
    A = s[:, None] / s[None, :]

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
    peso 0 naquela coluna. Só contam as relações entre os ``criterios``, e as
    prioridades de cada coluna são renormalizadas entre elas. Coluna sem
    nenhum influenciador recebe auto-laço (peso 1 em si mesma), o que preserva
    a propriedade estocástica.

    Retorna ``(W_TT, julgamentos)``, em que ``julgamentos`` é a lista de
    :class:`ColumnJudgment` (uma por coluna), usada por
    :func:`consistency_report`.
    """

    n = len(criterios)
    idx = {c: i for i, c in enumerate(criterios)}
    W_TT = np.zeros((n, n), dtype=float)
    julgamentos: list[ColumnJudgment] = []

    for alvo in criterios:
        influenciadores = {c: i for c, i in influencias.get(alvo, {}).items() if c in idx} or {alvo: 1.0}
        coluna = idx[alvo]

        julgamento = from_intensities(alvo, influenciadores)
        w, _ = eigen_priorities(julgamento.A)
        for label, peso in zip(julgamento.labels, w):
            W_TT[idx[label], coluna] = peso
        julgamentos.append(julgamento)

    return W_TT, julgamentos


def build_WTT_validada(
    criterios: tuple[str, ...] = config.CRITERIOS,
) -> tuple[np.ndarray, list[ColumnJudgment]]:
    """Constrói $W_{TT}$ dos ``criterios`` a partir de :data:`INFLUENCIAS`."""

    return build_WTT(INFLUENCIAS, criterios)


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
    e mantendo o valor dentro dos limites da escala. Como as razões não são
    arredondadas (:func:`from_intensities`), $A_{ab}$ pode estar entre dois
    níveis: ``step=+1`` leva ao primeiro nível acima do valor atual e
    ``step=-1`` ao primeiro abaixo (para valores já na escala, é o nível
    vizinho, como antes). Não modifica
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
    escala = np.asarray(SAATY_SCALE)
    no_nivel = np.flatnonzero(np.isclose(escala, valor_atual, rtol=1e-9, atol=0.0))
    if no_nivel.size:
        pos_base = int(no_nivel[0])
    elif step > 0:
        pos_base = int(np.searchsorted(escala, valor_atual)) - 1  # maior nível abaixo do valor
    else:
        pos_base = int(np.searchsorted(escala, valor_atual))  # menor nível acima do valor
    pos_novo = int(np.clip(pos_base + step, 0, len(SAATY_SCALE) - 1))
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
    "INFLUENCIAS",
    "JUSTIFICATIVAS",
    "ColumnJudgment",
    "from_intensities",
    "eigen_priorities",
    "consistency",
    "build_WTT",
    "build_WTT_validada",
    "consistency_report",
    "perturb",
    "julgamentos_por_alvo",
    "WTT_de_julgamentos",
]
