"""Estágio 1: funções de benefício, matrizes de payoff e critérios de decisão.

Implementa a Seção "Estágio 1" da TG: (i) funções de benefício $g_r$ por
critério (Tabela "criterios_payoff"); (ii) normalização min-max **comum**
sobre todos os pares (design, estado) — Eq. "normalizacao" — e, apenas para
diagnóstico, a normalização por estado (``per_state_minmax``), que demonstra
que normalizar cada estado isoladamente elimina o efeito dos cenários;
(iii) os três critérios de decisão sob incerteza — Wald, Savage e Hurwicz —
aplicados sobre o tensor de payoff normalizado; e (iv) o vetor de preferência
por perfil, com pesos iguais entre critérios — usado apenas como referência
de comparação "sem interdependência" (como no AHP original); o vencedor de
cada perfil vem da sub-rede ANP (``pipeline.Result.vencedor_perfil``).

Nenhum peso de critério é usado neste estágio: a ponderação/interdependência
entre critérios é tratada exclusivamente no Estágio 2 (``anp.py``).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config
from .scenarios import BallisticModel, propagate_states


def beneficio(
    criterio: str,
    attrs: dict[str, np.ndarray],
    requisitos: config.Requisitos,
    forma_residuo: str = "quadrado",
) -> np.ndarray:
    """Função de benefício $g_r$ de um critério, maior-é-melhor (Tabela "criterios_payoff").

    * ``Mp``, ``tb``, ``It``, ``F``: resíduo relativo ``δ = (f - f_req)/f_req``
      com sinal invertido, ``g = -δ²`` (``forma_residuo="quadrado"``, Eq. 30
      do artigo) ou ``g = -|δ|`` (``"absoluto"``);
    * ``rhoIsp``: maximizar, ``g = rhoIsp``;
    * ``pcmax``: minimizar, ``g = -pcmax``;
    * ``eta``: maximizar, ``g = eta``.
    """

    valores = np.asarray(attrs[criterio], dtype=float)

    if criterio in config.CRITERIOS_RESIDUO:
        f_req = getattr(requisitos, criterio)
        delta = (valores - f_req) / f_req
        if forma_residuo == "quadrado":
            return -(delta**2)
        if forma_residuo == "absoluto":
            return -np.abs(delta)
        raise ValueError(f"forma_residuo deve ser uma de {config.FORMAS_RESIDUO}, recebido {forma_residuo!r}")
    if criterio in config.CRITERIOS_MAXIMIZAR:
        return valores
    if criterio in config.CRITERIOS_MINIMIZAR:
        return -valores
    raise ValueError(f"Critério desconhecido: {criterio!r}")


def build_payoff_tensor(
    designs: pd.DataFrame,
    requisitos: config.Requisitos,
    kappa_c: tuple[float, float, float],
    kappa_cf: tuple[float, float, float],
    model: BallisticModel,
    forma_residuo: str = "quadrado",
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Constrói o tensor de payoff bruto (não normalizado).

    Retorna ``(X, atributos_por_estado)`` em que ``X`` tem shape
    ``(R=7, n, J=3)`` com ``X[r, k, j] = g_r(d_k, theta_j)``, e
    ``atributos_por_estado`` é o dicionário ``{estado: {criterio: array}}``
    devolvido por :func:`tg_robust.scenarios.propagate_states` (útil para
    diagnóstico e testes de identidade).
    """

    atributos_por_estado = propagate_states(designs, kappa_c, kappa_cf, model)
    n = len(designs)
    R = config.N_CRITERIOS
    J = len(config.ESTADOS_NOMES)

    X = np.empty((R, n, J), dtype=float)
    for j, estado in enumerate(config.ESTADOS_NOMES):
        attrs = atributos_por_estado[estado]
        for r, criterio in enumerate(config.CRITERIOS):
            X[r, :, j] = beneficio(criterio, attrs, requisitos, forma_residuo)

    return X, atributos_por_estado


def common_minmax(X: np.ndarray) -> np.ndarray:
    """Normalização min-max comum, sobre todos os pares (design, estado).

    Para cada critério ``r``, ``gmin``/``gmax`` são calculados sobre
    ``P_i x Theta`` (todos os designs e todos os estados conjuntamente),
    conforme a Eq. "normalizacao". Se ``gmax == gmin`` para algum critério
    (critério constante), devolve zeros para esse critério em vez de
    dividir por zero.
    """

    R, n, J = X.shape
    gmin = X.min(axis=(1, 2), keepdims=True)
    gmax = X.max(axis=(1, 2), keepdims=True)
    amplitude = gmax - gmin

    X_norm = np.zeros_like(X)
    constante = (amplitude[:, 0, 0] == 0.0)
    nao_constante = ~constante
    X_norm[nao_constante] = (X[nao_constante] - gmin[nao_constante]) / amplitude[nao_constante]
    return X_norm


def per_state_minmax(X: np.ndarray) -> np.ndarray:
    """Normalização min-max **por estado** (apenas para diagnóstico).

    Para cada critério ``r`` e cada estado ``j``, ``gmin``/``gmax`` são
    calculados apenas sobre os designs daquele estado. Isso demonstra que
    normalizar isoladamente por estado cancela qualquer fator puramente
    multiplicativo e comum a todos os designs em um estado (como ocorre em
    ``rhoIsp`` e, quando ``nexp`` é constante entre os designs, em
    ``pcmax``), eliminando o efeito da incerteza — por isso não é usada no
    tensor de payoff principal.
    """

    gmin = X.min(axis=1, keepdims=True)
    gmax = X.max(axis=1, keepdims=True)
    amplitude = gmax - gmin

    X_norm = np.zeros_like(X)
    constante = (amplitude[:, 0, :] == 0.0)
    for r in range(X.shape[0]):
        for j in range(X.shape[2]):
            if not constante[r, j]:
                X_norm[r, :, j] = (X[r, :, j] - gmin[r, 0, j]) / amplitude[r, 0, j]
    return X_norm


def wald(X_norm: np.ndarray) -> np.ndarray:
    """Critério de Wald (maximin): $v_{k,r} = \\min_j x_{r,k,j}$.

    Retorna ``v`` com shape ``(n, R)``.
    """

    return X_norm.min(axis=2).T


def savage(X_norm: np.ndarray) -> np.ndarray:
    """Critério de Savage (minimax regret).

    Matriz de arrependimento $\\Delta_{kj} = \\max_d x_{dj} - x_{kj}$ e
    pontuação $v_{k,r} = 1 - \\max_j \\Delta_{kj}$. Retorna ``v`` com shape
    ``(n, R)``.
    """

    max_por_coluna = X_norm.max(axis=1, keepdims=True)  # (R, 1, J)
    delta = max_por_coluna - X_norm  # (R, n, J)
    return (1.0 - delta.max(axis=2)).T


def hurwicz(X_norm: np.ndarray, alpha: float) -> np.ndarray:
    """Critério de Hurwicz: $v_{k,r} = \\alpha \\max_j x + (1-\\alpha) \\min_j x$.

    ``alpha=0`` reproduz Wald; ``alpha=1`` reproduz o máximo por linha
    (maximax). Retorna ``v`` com shape ``(n, R)``.
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha deve estar em [0, 1]")
    maximo = X_norm.max(axis=2)
    minimo = X_norm.min(axis=2)
    return (alpha * maximo + (1.0 - alpha) * minimo).T


# Regra de decisão de cada perfil: (X_norm, perfil) -> v, shape (n, R)
REGRAS: dict[str, Callable[[np.ndarray, config.Perfil], np.ndarray]] = {
    "wald": lambda X_norm, perfil: wald(X_norm),
    "savage": lambda X_norm, perfil: savage(X_norm),
    "hurwicz": lambda X_norm, perfil: hurwicz(X_norm, perfil.alpha),
}


def compute_profile_scores(
    X_norm: np.ndarray, perfis: Sequence[config.Perfil]
) -> dict[str, np.ndarray]:
    """Calcula as pontuações ``v^(p)`` de cada perfil da lista, pela sua regra.

    Retorna ``{sigla: v}``, cada ``v`` com shape ``(n, R)``.
    """

    return {perfil.sigla: REGRAS[perfil.regra](X_norm, perfil) for perfil in perfis}


def preference_vector(v: np.ndarray) -> np.ndarray:
    """Vetor de preferência por perfil $V^{(p)}_k = \\tfrac{1}{R}\\sum_r v^{(p)}_{k,r}$.

    Pesos iguais, sem interdependência: saída de comparação, não define o
    vencedor do perfil (que é o argmax de $g^{(p)}$ em $\\mathcal{F}^{(p)}$).
    """

    return v.mean(axis=1)


def preferred_design(V: np.ndarray, chi: np.ndarray | None = None) -> int | None:
    """$\\arg\\max_{d_k \\in \\mathcal{F}^{(p)}} V_k$ para um vetor de pontuações ``V``.

    Usado com $g^{(p)}$ (vencedor do perfil) e com $V^{(p)}$ (comparação).
    ``chi`` é a máscara booleana de $\\mathcal{F}^{(p)}$; ``None`` = sem
    restrições (argmax sobre todos os designs). Devolve ``None`` se o
    conjunto admissível for vazio.
    """

    if chi is None:
        return int(np.argmax(V))
    if not chi.any():
        return None
    return int(np.argmax(np.where(chi, V, -np.inf)))


__all__ = [
    "beneficio",
    "build_payoff_tensor",
    "common_minmax",
    "per_state_minmax",
    "wald",
    "savage",
    "hurwicz",
    "REGRAS",
    "compute_profile_scores",
    "preference_vector",
    "preferred_design",
]
