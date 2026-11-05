"""Estágio 2: Processo Analítico em Rede (ANP).

Implementa os blocos da supermatriz (Seção "Elicitação de Julgamentos e
Construção da Supermatriz" da TG), o limite em forma fechada (preferido,
via ``np.linalg.solve``), o limite por potências sucessivas da supermatriz
completa (verificação opcional) e a síntese final entre perfis.

Ordem de clusters fixa: ``(C_O, C_T, C_A)`` — objetivo (1 nó), critérios
técnicos (R nós, com dependência interna ``W_TT``) e alternativas (n nós,
absorventes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import config


def build_W_AT(v: np.ndarray, chi: np.ndarray | None = None) -> np.ndarray:
    """Bloco $W_{AT}^{(p)}$: pontuações do Estágio 1 normalizadas por coluna.

    ``v`` tem shape ``(n, R)`` (pontuações do perfil ``p``, uma linha por
    design). ``chi`` é a máscara booleana de $\\mathcal{F}^{(p)}$
    (Eq. "WAT_viavel"): as linhas dos designs inadmissíveis são zeradas e cada
    coluna ``r`` é normalizada para somar 1 apenas sobre os admissíveis.
    ``chi=None`` equivale a todos admissíveis (Eq. "bloco_AT"). Se a soma da
    coluna for 0, usa-se distribuição uniforme sobre os designs admissíveis.
    """

    n, R = v.shape
    chi = np.ones(n, dtype=bool) if chi is None else np.asarray(chi, dtype=bool)
    if not chi.any():
        raise ValueError("Conjunto admissível vazio: o perfil deve ser excluído da síntese")

    v_adm = np.where(chi[:, None], v, 0.0)
    somas = v_adm.sum(axis=0)
    W_AT = np.empty_like(v, dtype=float)
    for r in range(R):
        if somas[r] == 0.0:
            W_AT[:, r] = chi / chi.sum()
        else:
            W_AT[:, r] = v_adm[:, r] / somas[r]
    return W_AT


def effective_criterion_weights(W_TT: np.ndarray, gamma: float, q0: np.ndarray) -> np.ndarray:
    """Pesos efetivos dos critérios após a propagação das interdependências.

    $\\mathbf{q} = (1-\\gamma)(\\mathbf{I} - \\gamma \\mathbf{W}_{TT})^{-1}
    \\mathbf{q}_0$, calculado via ``np.linalg.solve`` (sem inversa explícita).
    """

    R = W_TT.shape[0]
    I = np.eye(R)
    x = np.linalg.solve(I - gamma * W_TT, q0)
    return (1.0 - gamma) * x


def profile_limit_closed(
    W_AT: np.ndarray, W_TT: np.ndarray, gamma: float, q0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Limite em forma fechada das prioridades globais de um perfil.

    $\\mathbf{g}^{(p)} = (1-\\gamma)\\,\\mathbf{W}^{(p)}_{AT}
    (\\mathbf{I}-\\gamma \\mathbf{W}_{TT})^{-1}\\mathbf{q}_0$
    (Eq. "prioridade_perfil"). Retorna ``(g_p, q)``, em que ``q`` são os
    pesos efetivos dos critérios (:func:`effective_criterion_weights`,
    comuns a todos os perfis pois ``W_TT`` é compartilhada).
    """

    q = effective_criterion_weights(W_TT, gamma, q0)
    g_p = W_AT @ q
    return g_p, q


def build_weighted_supermatrix(
    W_AT: np.ndarray, W_TT: np.ndarray, gamma: float, q0: np.ndarray
) -> np.ndarray:
    """Monta a supermatriz ponderada completa $\\bar{\\mathbf{W}}^{(p)}$.

    Ordem dos clusters: ``(C_O, C_T, C_A)``, com 1 + R + n linhas/colunas
    (Eq. "supermatriz_perfil").
    """

    R = W_TT.shape[0]
    n = W_AT.shape[0]
    tam = 1 + R + n
    W_bar = np.zeros((tam, tam))

    # bloco C_O -> C_T (linha T, coluna O)
    W_bar[1 : 1 + R, 0] = q0
    # bloco C_T <-> C_T (dependência interna, ponderada por gamma)
    W_bar[1 : 1 + R, 1 : 1 + R] = gamma * W_TT
    # bloco C_T -> C_A (linha A, coluna T), ponderado por (1-gamma)
    W_bar[1 + R :, 1 : 1 + R] = (1.0 - gamma) * W_AT
    # bloco C_A -> C_A: identidade (alternativas absorventes)
    W_bar[1 + R :, 1 + R :] = np.eye(n)

    return W_bar


def power_limit(W_bar: np.ndarray, tol: float = 1e-9, max_iter: int = 200) -> np.ndarray:
    """Limite da supermatriz por potências sucessivas (quadrados sucessivos).

    Eleva ``W_bar`` a potências $2^k$ até que a diferença máxima entre duas
    potências sucessivas seja menor que ``tol``. Usada apenas como
    verificação do limite em forma fechada (:func:`profile_limit_closed`);
    não é o caminho padrão do pipeline por ser mais custosa.
    """

    W = W_bar.copy()
    for _ in range(max_iter):
        W2 = W @ W
        if np.max(np.abs(W2 - W)) < tol:
            return W2
        W = W2
    raise RuntimeError(
        f"Limite por potências não convergiu em {max_iter} duplicações (tol={tol})"
    )


def profile_limit_power(
    W_AT: np.ndarray, W_TT: np.ndarray, gamma: float, q0: np.ndarray, tol: float = 1e-9
) -> np.ndarray:
    """Prioridades globais de um perfil pelo limite por potências (verificação).

    Extrai, da supermatriz limite, a coluna do nó de objetivo restrita às
    linhas do cluster de alternativas — deve coincidir com
    :func:`profile_limit_closed` dentro de ``tol`` (ver testes).
    """

    R = W_TT.shape[0]
    W_bar = build_weighted_supermatrix(W_AT, W_TT, gamma, q0)
    limite = power_limit(W_bar, tol=tol)
    return limite[1 + R :, 0]


@dataclass(frozen=True)
class ResultadoANP:
    """Saída do Estágio 2 para um estágio/execução completa."""

    g: np.ndarray  # prioridades globais finais (síntese entre perfis), shape (n,)
    g_por_perfil: dict[str, np.ndarray]  # g^(p), uma por perfil
    q0: np.ndarray  # importância a priori dos critérios
    q: np.ndarray  # pesos efetivos dos critérios (pós-dependência interna)
    W_AT_por_perfil: dict[str, np.ndarray]
    W_TT: np.ndarray
    perfis_excluidos: tuple[str, ...] = ()  # perfis com F^(p) vazio, fora da síntese


def pi_renormalizado(pi: dict[str, float], perfis: list[str]) -> dict[str, float]:
    """Restringe ``pi`` aos ``perfis`` presentes na síntese e renormaliza para somar 1.

    Usado quando a sub-rede de algum perfil é excluída por
    $\\mathcal{F}^{(p)} = \\varnothing$. Lança ``ValueError`` se os perfis
    restantes tiverem peso total nulo (síntese indefinida).
    """

    soma = sum(pi[p] for p in perfis)
    if soma <= 0.0:
        raise ValueError(
            f"Síntese indefinida: os perfis presentes {perfis} têm peso pi total nulo"
        )
    return {p: pi[p] / soma for p in perfis}


def synthesize(
    g_por_perfil: dict[str, np.ndarray], pi: dict[str, float]
) -> np.ndarray:
    """Síntese final entre perfis: $\\mathbf{g} = \\sum_p \\pi_p \\mathbf{g}^{(p)}$.

    Soma apenas sobre os perfis presentes em ``g_por_perfil`` (perfis excluídos
    por conjunto admissível vazio não aparecem), com ``pi`` renormalizado
    entre eles (:func:`pi_renormalizado`).
    """

    perfis = list(g_por_perfil.keys())
    pi_efetivo = pi_renormalizado(pi, perfis)
    g = np.zeros_like(next(iter(g_por_perfil.values())))
    for p in perfis:
        g = g + pi_efetivo[p] * g_por_perfil[p]
    return g


def W_AT_mixture(W_AT_por_perfil: dict[str, np.ndarray], pi: dict[str, float]) -> np.ndarray:
    """Mistura dos blocos $\\bar{\\mathbf{W}}_{AT} = \\sum_p \\pi_p \\mathbf{W}^{(p)}_{AT}$.

    Permite verificar $\\mathbf{g} = \\bar{\\mathbf{W}}_{AT}\\,\\mathbf{q}$
    (Eq. "selecao_final").
    """

    perfis = list(W_AT_por_perfil.keys())
    pi_efetivo = pi_renormalizado(pi, perfis)
    W_bar = np.zeros_like(next(iter(W_AT_por_perfil.values())))
    for p in perfis:
        W_bar = W_bar + pi_efetivo[p] * W_AT_por_perfil[p]
    return W_bar


def run_anp(
    v_por_perfil: dict[str, np.ndarray],
    W_TT: np.ndarray,
    gamma: float,
    pi: dict[str, float],
    q0: np.ndarray | None = None,
    verificar_limite_por_potencias: bool = False,
    tol_verificacao: float = 1e-9,
    chi_por_perfil: dict[str, np.ndarray] | None = None,
) -> ResultadoANP:
    """Executa o Estágio 2 completo a partir das pontuações do Estágio 1.

    ``v_por_perfil`` é ``{perfil: v^(p)}`` (shape ``(n, R)`` cada, vindo de
    :func:`tg_robust.stage1.compute_profile_scores`). ``chi_por_perfil`` traz
    as máscaras de $\\mathcal{F}^{(p)}$ (:mod:`tg_robust.admissibility`);
    ``None`` = sem restrições. Perfis com $\\mathcal{F}^{(p)} = \\varnothing$
    são excluídos da síntese (aviso emitido) e ``pi`` é renormalizado entre os
    restantes; se todos forem vazios, lança ``ValueError``. Se
    ``verificar_limite_por_potencias`` for verdadeiro, exige que o limite
    por potências coincida com o limite em forma fechada dentro de
    ``tol_verificacao`` (lança ``AssertionError`` caso contrário).
    """

    R = W_TT.shape[0]
    q0 = q0 if q0 is not None else np.full(R, 1.0 / R)
    chi_por_perfil = chi_por_perfil or {}

    perfis_excluidos = tuple(
        p for p in v_por_perfil if p in chi_por_perfil and not np.any(chi_por_perfil[p])
    )
    if perfis_excluidos:
        warnings.warn(
            f"Conjunto admissível vazio para o(s) perfil(is) {list(perfis_excluidos)}: "
            "sub-rede(s) excluída(s) da síntese e pi renormalizado entre os demais.",
            stacklevel=2,
        )
    if len(perfis_excluidos) == len(v_por_perfil):
        raise ValueError("Conjunto admissível vazio para todos os perfis: síntese indefinida")

    W_AT_por_perfil = {
        p: build_W_AT(v, chi_por_perfil.get(p))
        for p, v in v_por_perfil.items()
        if p not in perfis_excluidos
    }

    g_por_perfil: dict[str, np.ndarray] = {}
    q_comum: np.ndarray | None = None
    for p, W_AT in W_AT_por_perfil.items():
        g_p, q = profile_limit_closed(W_AT, W_TT, gamma, q0)
        g_por_perfil[p] = g_p
        q_comum = q  # idêntico para todos os perfis (W_TT compartilhada)

        if verificar_limite_por_potencias:
            g_p_potencias = profile_limit_power(W_AT, W_TT, gamma, q0, tol=tol_verificacao)
            if not np.allclose(g_p, g_p_potencias, atol=tol_verificacao * 10):
                raise AssertionError(
                    f"Limite fechado e limite por potências divergem para o perfil {p!r}: "
                    f"max|diff|={np.max(np.abs(g_p - g_p_potencias)):.3e}"
                )

    g = synthesize(g_por_perfil, pi)

    return ResultadoANP(
        g=g,
        g_por_perfil=g_por_perfil,
        q0=q0,
        q=q_comum if q_comum is not None else np.full(R, 1.0 / R),
        W_AT_por_perfil=W_AT_por_perfil,
        W_TT=W_TT,
        perfis_excluidos=perfis_excluidos,
    )


__all__ = [
    "build_W_AT",
    "effective_criterion_weights",
    "profile_limit_closed",
    "build_weighted_supermatrix",
    "power_limit",
    "profile_limit_power",
    "ResultadoANP",
    "pi_renormalizado",
    "synthesize",
    "W_AT_mixture",
    "run_anp",
]
