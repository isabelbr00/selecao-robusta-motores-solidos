"""Análise de sensibilidade transversal (Seção "Análise de Sensibilidade").

Cinco varreduras complementares, todas avaliadas contra a execução de
referência (:class:`tg_robust.pipeline.Result`) e todas respeitando os
conjuntos admissíveis $\\mathcal{F}^{(p)}$ da referência (ou recalculados,
quando a varredura os altera):

1. :func:`sweep_alpha` — coeficiente de otimismo de Hurwicz;
2. :func:`sweep_pi` — pesos relativos entre perfis (±20% e vetores canônicos);
3. :func:`sweep_anp` — julgamentos de ``W_TT`` (±1 nível de Saaty) e ``gamma``;
4. :func:`sweep_lambda` — magnitude dos estados de incerteza (reexecuta o
   Estágio 1 completo e recalcula $\\mathcal{F}^{(p)}$);
5. :func:`sweep_restricoes` — fatores de segurança ``FS_p`` e tetos de custo
   ``rho_p`` (item 5 da Seção 3.5 de ``TG_revisao_FS_custo.tex``), mais o
   caso sem restrições (``FS=1``, ``rho=inf``).

A estabilidade de cada ranking perturbado frente ao de referência é medida
por três indicadores (:func:`metricas_estabilidade`): manutenção de
``d*``, $\\tau$ de Kendall e interseção do top-10. :func:`regiao_robustez`
devolve o subconjunto de valores de um parâmetro varrido para os quais
``d*`` não se altera.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import admissibility, anp, config
from . import judgments as J
from . import stage1
from .pipeline import Result
from .scenarios import BallisticModel, SurrogateBallistics


def metricas_estabilidade(g_ref: np.ndarray, g_novo: np.ndarray) -> dict[str, float | bool]:
    """Indicadores de estabilidade do ranking ``g_novo`` frente a ``g_ref``.

    * ``mesmo_d_star``: o design de maior prioridade é o mesmo;
    * ``kendall_tau``: correlação de Kendall entre os dois rankings completos;
    * ``intersecao_top10``: quantos dos 10 melhores designs de referência
      permanecem entre os 10 melhores do ranking perturbado;
    * ``posicao_d_star_ref``: posição (1 = melhor) do design ``d*`` de
      referência no ranking perturbado.
    """

    d_star_ref = int(np.argmax(g_ref))
    d_star_novo = int(np.argmax(g_novo))

    tau, _ = kendalltau(g_ref, g_novo)

    n = len(g_ref)
    k = min(10, n)
    top_ref = set(np.argsort(-g_ref)[:k])
    top_novo = set(np.argsort(-g_novo)[:k])

    rank_novo = pd.Series(g_novo).rank(ascending=False, method="min")
    posicao_d_star_ref = int(rank_novo.iloc[d_star_ref])

    return {
        "mesmo_d_star": d_star_novo == d_star_ref,
        "kendall_tau": float(tau),
        "intersecao_top10": len(top_ref & top_novo),
        "posicao_d_star_ref": posicao_d_star_ref,
    }


def regiao_robustez(df: pd.DataFrame, coluna_param: str, dstar_ref: int | None = None) -> list:
    """Valores do parâmetro ``coluna_param`` para os quais ``d*`` não muda.

    Espera que ``df`` tenha uma coluna ``mesmo_d_star`` (produzida pelas
    funções ``sweep_*`` via :func:`metricas_estabilidade`). ``dstar_ref`` é
    aceito apenas por clareza de interface (a comparação já foi feita em
    ``mesmo_d_star``).
    """

    return sorted(df.loc[df["mesmo_d_star"], coluna_param].tolist())


def _metricas_indefinidas() -> dict[str, float | bool]:
    """Indicadores para um caso em que a síntese é indefinida (todos os perfis excluídos)."""

    return {
        "mesmo_d_star": False,
        "kendall_tau": np.nan,
        "intersecao_top10": np.nan,
        "posicao_d_star_ref": np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────
# 1. Variação de alpha (Hurwicz)
# ─────────────────────────────────────────────────────────────────────────
def sweep_alpha(
    ref: Result, alphas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
) -> pd.DataFrame:
    """Varre $\\alpha \\in \\{0, 0.1, \\dots, 0.9, 1\\}$ do perfil Hurwicz."""

    pi = dict(zip(config.PERFIS, ref.cfg.pi_perfis))
    linhas = []
    for alpha in alphas:
        v_H = stage1.hurwicz(ref.X_norm, alpha)
        scores_novo = {"W": ref.scores["W"], "S": ref.scores["S"], "H": v_H}
        resultado = anp.run_anp(
            scores_novo,
            ref.W_TT,
            gamma=ref.cfg.gamma_anp,
            pi=pi,
            q0=ref.anp_resultado.q0,
            chi_por_perfil=ref.admissibilidade.chi,
        )
        metricas = metricas_estabilidade(ref.g, resultado.g)
        linhas.append({"alpha": alpha, **metricas})
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 2. Variação dos pesos relativos entre perfis (pi)
# ─────────────────────────────────────────────────────────────────────────
def _normalizar_pi(pi: dict[str, float]) -> dict[str, float]:
    soma = sum(pi.values())
    return {p: v / soma for p, v in pi.items()}


def sweep_pi(ref: Result) -> pd.DataFrame:
    """Varia cada $\\pi_p$ em ±20% (renormalizado) e os 3 vetores canônicos.

    Perfis excluídos por $\\mathcal{F}^{(p)} = \\varnothing$ não entram na
    síntese; o vetor canônico de um perfil excluído deixa a síntese
    indefinida (``sintese_definida=False``, indicadores ``NaN``).
    """

    pi_ref = dict(zip(config.PERFIS, ref.cfg.pi_perfis))
    g_por_perfil = ref.anp_resultado.g_por_perfil
    linhas = []

    def _avaliar(pi_novo: dict[str, float]) -> dict:
        try:
            g_novo = anp.synthesize(g_por_perfil, pi_novo)
        except ValueError:
            return {"sintese_definida": False, **_metricas_indefinidas()}
        return {"sintese_definida": True, **metricas_estabilidade(ref.g, g_novo)}

    for p in config.PERFIS:
        for sinal, rotulo in ((+0.20, "+20%"), (-0.20, "-20%")):
            pi_variado = dict(pi_ref)
            pi_variado[p] = pi_ref[p] * (1.0 + sinal)
            pi_variado = _normalizar_pi(pi_variado)
            linhas.append(
                {"tipo": "percentual", "perfil_variado": p, "variacao": rotulo, "pi": pi_variado, **_avaliar(pi_variado)}
            )

    for p_canonico in config.PERFIS:
        pi_canonico = {p: (1.0 if p == p_canonico else 0.0) for p in config.PERFIS}
        linhas.append(
            {"tipo": "canonico", "perfil_variado": p_canonico, "variacao": "pi=1", "pi": pi_canonico, **_avaliar(pi_canonico)}
        )

    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 3. Variação dos julgamentos do ANP (W_TT) e de gamma
# ─────────────────────────────────────────────────────────────────────────
def sweep_anp(
    ref: Result, gammas: tuple[float, ...] = (0.3, 0.5, 0.7), incluir_gamma_zero: bool = True
) -> pd.DataFrame:
    """Perturba cada julgamento de ``W_TT`` em ±1 nível de Saaty, e varia ``gamma``."""

    pi = dict(zip(config.PERFIS, ref.cfg.pi_perfis))
    q0 = ref.anp_resultado.q0
    W_AT_por_perfil = ref.anp_resultado.W_AT_por_perfil
    linhas = []

    # --- perturbação dos julgamentos (todas as combinações coluna/par/sentido) ---
    por_alvo = J.julgamentos_por_alvo(ref.julgamentos)
    for alvo, julgamento in por_alvo.items():
        labels = julgamento.labels
        for i in range(len(labels)):
            for k in range(i + 1, len(labels)):
                a, b = labels[i], labels[k]
                for step in (-1, +1):
                    novos_julgamentos = J.perturb(por_alvo, alvo, a, b, step)
                    W_TT_novo = J.WTT_de_julgamentos(novos_julgamentos)

                    g_por_perfil_novo = {
                        p: anp.profile_limit_closed(W_AT_por_perfil[p], W_TT_novo, ref.cfg.gamma_anp, q0)[0]
                        for p in W_AT_por_perfil
                    }
                    g_novo = anp.synthesize(g_por_perfil_novo, pi)
                    metricas = metricas_estabilidade(ref.g, g_novo)
                    linhas.append(
                        {
                            "tipo": "julgamento",
                            "coluna": alvo,
                            "par": f"{a}-{b}",
                            "passo_saaty": step,
                            **metricas,
                        }
                    )

    # --- variação de gamma (com W_TT de referência) ---
    for gamma in gammas:
        g_por_perfil_novo = {
            p: anp.profile_limit_closed(W_AT_por_perfil[p], ref.W_TT, gamma, q0)[0] for p in W_AT_por_perfil
        }
        g_novo = anp.synthesize(g_por_perfil_novo, pi)
        metricas = metricas_estabilidade(ref.g, g_novo)
        linhas.append({"tipo": "gamma", "coluna": None, "par": None, "passo_saaty": gamma, **metricas})

    if incluir_gamma_zero:
        g_por_perfil_novo = {
            p: anp.profile_limit_closed(W_AT_por_perfil[p], ref.W_TT, 0.0, q0)[0] for p in W_AT_por_perfil
        }
        g_novo = anp.synthesize(g_por_perfil_novo, pi)
        metricas = metricas_estabilidade(ref.g, g_novo)
        linhas.append({"tipo": "gamma", "coluna": None, "par": None, "passo_saaty": 0.0, **metricas})

    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 4. Variação da magnitude dos estados de incerteza (lambda)
# ─────────────────────────────────────────────────────────────────────────
def sweep_lambda(
    ref: Result,
    lambdas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    model: BallisticModel | None = None,
) -> pd.DataFrame:
    """Escala a magnitude dos estados 2 e 3 por $\\lambda$ e reexecuta o Estágio 1.

    Como ``max_j pcmax`` depende dos estados, $\\mathcal{F}^{(p)}$ é recalculado
    para cada $\\lambda$ com os parâmetros de restrição da referência.
    """

    model = model if model is not None else SurrogateBallistics()
    linhas = []

    for lam in lambdas:
        kappa_c_novo = config.kappa_escalado(config.KAPPA_C_REF, lam)
        kappa_cf_novo = config.kappa_escalado(config.KAPPA_CF_REF, lam)

        X_novo, atributos_novo = stage1.build_payoff_tensor(
            ref.designs, ref.cfg.requisitos, kappa_c_novo, kappa_cf_novo, model
        )
        X_norm_novo = stage1.common_minmax(X_novo)
        scores_novo = stage1.compute_profile_scores(X_norm_novo, ref.cfg.alpha_hurwicz)
        adm_novo = admissibility.conjuntos_admissiveis(
            ref.designs, atributos_novo, ref.cfg.fs_perfis, ref.cfg.rho_perfis, ref.cfg.b_cer, ref.cfg.p_lim
        )

        linhas.append({"lambda": lam, **_avaliar_anp_com_restricoes(ref, scores_novo, adm_novo)})

    return pd.DataFrame(linhas)


def _avaliar_anp_com_restricoes(
    ref: Result, scores: dict[str, np.ndarray], adm: admissibility.Admissibilidade
) -> dict:
    """Refaz o ANP com as máscaras ``adm`` e mede a estabilidade frente a ``ref``."""

    pi = dict(zip(config.PERFIS, ref.cfg.pi_perfis))
    tamanhos = {f"n_F_{p}": int(adm.chi[p].sum()) for p in config.PERFIS}
    tamanhos["n_intersecao"] = int(adm.intersecao.sum())
    try:
        resultado = anp.run_anp(
            scores, ref.W_TT, gamma=ref.cfg.gamma_anp, pi=pi, q0=ref.anp_resultado.q0, chi_por_perfil=adm.chi
        )
    except ValueError:
        return {
            **tamanhos,
            "sintese_definida": False,
            "perfis_excluidos": ",".join(config.PERFIS),
            "d_star": None,
            "d_star_inadmissivel_em": None,
            **_metricas_indefinidas(),
        }

    d_star = int(np.argmax(resultado.g))
    return {
        **tamanhos,
        "sintese_definida": True,
        "perfis_excluidos": ",".join(resultado.perfis_excluidos),
        "d_star": d_star,
        "d_star_inadmissivel_em": ",".join(adm.perfis_inadmissiveis(d_star)),
        **metricas_estabilidade(ref.g, resultado.g),
    }


# ─────────────────────────────────────────────────────────────────────────
# 5. Variação das restrições de admissibilidade (FS_p e rho_p)
# ─────────────────────────────────────────────────────────────────────────
def sweep_restricoes(
    ref: Result,
    fs_valores: tuple[float, ...] = (1.15, 1.25, 1.40),
    rho_valores: tuple[float, ...] = (1.05, 1.15, 1.25, 1.50, math.inf),
) -> pd.DataFrame:
    """Varia ``FS_p`` e ``rho_p`` um perfil por vez, mais o caso sem restrições.

    Cada linha altera um único parâmetro de um único perfil, com os demais na
    referência (``tipo`` = ``"FS"`` ou ``"rho"``). A linha
    ``tipo="sem_restricoes"`` usa ``FS_p = 1`` e ``rho_p = inf`` para todos os
    perfis (``pcmax <= p_lim`` do modelo original continua valendo): é a
    formulação em que os perfis diferem só pela regra de decisão. Payoff e
    pontuações $v^{(p)}$ são os da referência; só $\\mathcal{F}^{(p)}$ muda.

    Enquanto ``b_cer`` não estiver definido, o teto de custo não é aplicado e
    as linhas de ``rho`` reproduzem a referência (``teto_custo_aplicado=False``).
    """

    fs_ref = dict(zip(config.PERFIS, ref.cfg.fs_perfis))
    rho_ref = dict(zip(config.PERFIS, ref.cfg.rho_perfis))

    cenarios: list[tuple[str, str | None, float | None, dict[str, float], dict[str, float]]] = []
    for p in config.PERFIS:
        for fs in fs_valores:
            cenarios.append(("FS", p, fs, {**fs_ref, p: fs}, rho_ref))
    for p in config.PERFIS:
        for rho in rho_valores:
            cenarios.append(("rho", p, rho, fs_ref, {**rho_ref, p: rho}))
    cenarios.append(
        ("sem_restricoes", None, None, {p: 1.0 for p in config.PERFIS}, {p: math.inf for p in config.PERFIS})
    )

    linhas = []
    for tipo, perfil, valor, fs, rho in cenarios:
        adm = admissibility.conjuntos_admissiveis(
            ref.designs,
            ref.atributos_por_estado,
            tuple(fs[p] for p in config.PERFIS),
            tuple(rho[p] for p in config.PERFIS),
            ref.cfg.b_cer,
            ref.cfg.p_lim,
        )
        linhas.append(
            {
                "tipo": tipo,
                "perfil_variado": perfil,
                "valor": valor,
                "teto_custo_aplicado": ref.cfg.b_cer is not None,
                **_avaliar_anp_com_restricoes(ref, ref.scores, adm),
            }
        )

    return pd.DataFrame(linhas)


__all__ = [
    "metricas_estabilidade",
    "regiao_robustez",
    "sweep_alpha",
    "sweep_pi",
    "sweep_anp",
    "sweep_lambda",
    "sweep_restricoes",
]
