"""Análise de sensibilidade transversal.

Oito varreduras complementares, todas avaliadas contra a execução de
referência (:class:`selecao_robusta.pipeline.Result`) e todas respeitando os
conjuntos admissíveis $\\mathcal{F}^{(p)}$ da referência (ou recalculados,
quando a varredura os altera):

1. :func:`sweep_alpha` — coeficiente de otimismo de Hurwicz;
2. :func:`sweep_pi` — pesos relativos entre perfis (±20% e vetores canônicos);
3. :func:`sweep_anp` — julgamentos de ``W_TT`` (±1 nível de Saaty) e ``gamma``;
4. :func:`sweep_lambda` — magnitude dos estados de incerteza (reexecuta o
   Estágio 1 completo e recalcula $\\mathcal{F}^{(p)}$);
5. :func:`sweep_restricoes` — fatores de segurança ``FS_p``, tetos de custo
   ``rho_p``, referência
   do índice de custo (mínimo/mediana) e o caso sem restrições
   (``FS=1``, ``rho=inf``). Cada cenário é uma lista de perfis
   (:func:`cenarios_restricoes`), avaliada por :func:`avaliar_cenario`;
6. :func:`sweep_residuo` — forma do resíduo no benefício de ``Mp``, ``tb``,
   ``It`` e ``F`` (``-δ²`` ou ``-|δ|``; reexecuta o Estágio 1);
7. :func:`sweep_cenarios_fs` — cenários de FS dos perfis (:data:`CENARIOS_FS`),
   com vencedor por perfil e top 10;
8. :func:`sweep_nominal` — apenas o estado nominal $\\theta_1$, com a
   normalização refeita sobre ele.

A estabilidade de cada ranking perturbado frente ao de referência é medida
por três indicadores (:func:`metricas_estabilidade`): manutenção de
``d*``, $\\tau$ de Kendall e interseção do top-10. :func:`regiao_robustez`
devolve o subconjunto de valores de um parâmetro varrido para os quais
``d*`` não se altera.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import admissibility, anp, config
from . import judgments as J
from . import stage1
from .pipeline import Result
from .scenarios import BallisticModel


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
    """Varre $\\alpha \\in \\{0, 0.1, \\dots, 0.9, 1\\}$ dos perfis com regra Hurwicz.

    Todos os perfis com ``regra == "hurwicz"`` recebem o mesmo ``alpha``; os
    demais perfis e os conjuntos admissíveis permanecem os da referência.
    """

    siglas_hurwicz = [p.sigla for p in ref.cfg.perfis if p.regra == "hurwicz"]
    linhas = []
    for alpha in alphas:
        perfis_novos = ref.cfg.perfis
        for sigla in siglas_hurwicz:
            perfis_novos = config.substituir_perfil(perfis_novos, sigla, alpha=alpha)
        fs_ef = stage1.fs_efetivo(ref.cfg.criterios, ref.atributos_por_estado, ref.cfg.globais.p_lim)
        scores_novo = stage1.compute_profile_scores(ref.X_norm, perfis_novos, fs_ef)
        resultado = anp.run_anp(
            scores_novo,
            ref.W_TT,
            gamma=ref.cfg.gamma_anp,
            pi=ref.cfg.pi,
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

    pi_ref = ref.cfg.pi
    siglas = ref.cfg.siglas
    g_por_perfil = ref.anp_resultado.g_por_perfil
    linhas = []

    def _avaliar(pi_novo: dict[str, float]) -> dict:
        try:
            g_novo = anp.synthesize(g_por_perfil, pi_novo)
        except ValueError:
            return {"sintese_definida": False, **_metricas_indefinidas()}
        return {"sintese_definida": True, **metricas_estabilidade(ref.g, g_novo)}

    for p in siglas:
        for sinal, rotulo in ((+0.20, "+20%"), (-0.20, "-20%")):
            pi_variado = dict(pi_ref)
            pi_variado[p] = pi_ref[p] * (1.0 + sinal)
            pi_variado = _normalizar_pi(pi_variado)
            linhas.append(
                {"tipo": "percentual", "perfil_variado": p, "variacao": rotulo, "pi": pi_variado, **_avaliar(pi_variado)}
            )

    for p_canonico in siglas:
        pi_canonico = {p: (1.0 if p == p_canonico else 0.0) for p in siglas}
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

    pi = ref.cfg.pi
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
                    W_TT_novo = J.WTT_de_julgamentos(novos_julgamentos, ref.cfg.criterios)

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

    Usa ``model`` ou, na falta dele, o modelo de balística da referência
    (``ref.modelo_balistico``). Se esse modelo não reexecuta a balística
    (estados lidos dos dados, :class:`~selecao_robusta.scenarios.EstadosPrecomputados`),
    a varredura é pulada com aviso e devolve um ``DataFrame`` vazio.
    Como ``max_j pcmax`` depende dos estados, $\\mathcal{F}^{(p)}$ é recalculado
    para cada $\\lambda$ com os perfis da referência.
    """

    model = model if model is not None else ref.modelo_balistico
    if not getattr(model, "suporta_reexecucao", True):
        warnings.warn(
            "Varredura de lambda PULADA: nenhum modelo de balística capaz de reexecutar "
            "os estados foi fornecido (os estados vêm prontos dos dados). Passe um "
            "BallisticModel (p.ex. --balistica caminho/balistica.py).",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["lambda"])

    linhas = []
    for lam in lambdas:
        kappa_c_novo = config.kappa_escalado(config.KAPPA_C_REF, lam)
        kappa_cf_novo = config.kappa_escalado(config.KAPPA_CF_REF, lam)

        X_novo, atributos_novo = stage1.build_payoff_tensor(
            ref.designs, ref.cfg.requisitos, kappa_c_novo, kappa_cf_novo, model, ref.cfg.globais.forma_residuo,
            ref.cfg.criterios_comuns,
        )
        X_norm_novo = stage1.common_minmax(X_novo)
        linhas.append(
            {
                "lambda": lam,
                **avaliar_cenario(ref, ref.cfg.perfis, ref.cfg.globais, X_norm=X_norm_novo, atributos=atributos_novo),
            }
        )

    return pd.DataFrame(linhas)


def _sintese_cenario(
    ref: Result,
    perfis: tuple[config.Perfil, ...],
    globais: config.ConfigGlobal,
    X_norm: np.ndarray,
    atributos: dict[str, dict[str, np.ndarray]],
) -> tuple[admissibility.Admissibilidade, anp.ResultadoANP | None]:
    """$\\mathcal{F}^{(p)}$ e ANP de uma lista de perfis; ``None`` se a síntese for indefinida."""

    config.validar_perfis(perfis)
    fs_ef = stage1.fs_efetivo(ref.cfg.criterios, atributos, globais.p_lim)
    scores = stage1.compute_profile_scores(X_norm, perfis, fs_ef)
    adm = admissibility.conjuntos_admissiveis(ref.designs, atributos, perfis, globais)
    pi = {p.sigla: p.peso_pi for p in perfis}
    try:
        return adm, anp.run_anp(
            scores, ref.W_TT, gamma=ref.cfg.gamma_anp, pi=pi, q0=ref.anp_resultado.q0, chi_por_perfil=adm.chi
        )
    except ValueError:
        return adm, None


def avaliar_cenario(
    ref: Result,
    perfis: tuple[config.Perfil, ...],
    globais: config.ConfigGlobal,
    X_norm: np.ndarray | None = None,
    atributos: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict:
    """Roda Estágio 1 (pontuações), restrições e ANP para uma lista de perfis.

    O mesmo procedimento serve para qualquer lista de perfis: pontuações pela
    regra de cada perfil, $\\mathcal{F}^{(p)}$ pelos seus FS/rho, síntese pelos
    seus ``peso_pi``. ``X_norm``/``atributos`` padrão: os da referência.
    Devolve tamanhos de $\\mathcal{F}^{(p)}$, ``d*``, perfis em que ``d*`` é
    inadmissível e os indicadores de estabilidade frente a ``ref``.
    """

    X_norm = ref.X_norm if X_norm is None else X_norm
    atributos = ref.atributos_por_estado if atributos is None else atributos
    adm, resultado = _sintese_cenario(ref, perfis, globais, X_norm, atributos)

    tamanhos = {f"n_F_{s}": int(adm.chi[s].sum()) for s in adm.siglas}
    tamanhos["n_intersecao"] = int(adm.intersecao.sum())
    if resultado is None:
        return {
            **tamanhos,
            "sintese_definida": False,
            "perfis_excluidos": ",".join(adm.siglas),
            "d_star": None,
            "d_star_inadmissivel_em": None,
            **_metricas_indefinidas(),
        }

    d_star = int(np.argmax(resultado.g))
    return {
        **tamanhos,
        "sintese_definida": True,
        "perfis_excluidos": ",".join(resultado.perfis_excluidos),
        "d_star": ref.design_ids[d_star],
        "d_star_inadmissivel_em": ",".join(adm.perfis_inadmissiveis(d_star)),
        **metricas_estabilidade(ref.g, resultado.g),
    }


# ─────────────────────────────────────────────────────────────────────────
# 5. Variação das restrições de admissibilidade (FS_p, rho_p, referência do custo)
# ─────────────────────────────────────────────────────────────────────────
def cenarios_restricoes(
    perfis_ref: tuple[config.Perfil, ...],
    globais_ref: config.ConfigGlobal,
    fs_valores: tuple[float, ...] = (1.15, 1.25, 1.40),
    rho_valores: tuple[float, ...] = (1.05, 1.15, 1.25, 1.50, math.inf),
    referencias_custo: tuple[str, ...] = config.REFERENCIAS_CUSTO,
) -> list[dict]:
    """Lista de cenários da varredura de restrições; cada um é uma lista de perfis.

    Um parâmetro de um perfil por vez (demais na referência): ``FS`` e
    ``rho``; a referência do índice de custo (parâmetro global); e o caso
    ``sem_restricoes`` (``FS = 1`` e ``rho = inf`` para todos).
    """

    cenarios = []
    for perfil in perfis_ref:
        for fs in fs_valores:
            cenarios.append(
                dict(tipo="FS", perfil_variado=perfil.sigla, valor=fs, globais=globais_ref,
                     perfis=config.substituir_perfil(perfis_ref, perfil.sigla, FS=fs))
            )
    for perfil in perfis_ref:
        for rho in rho_valores:
            cenarios.append(
                dict(tipo="rho", perfil_variado=perfil.sigla, valor=rho, globais=globais_ref,
                     perfis=config.substituir_perfil(perfis_ref, perfil.sigla, rho=rho))
            )
    for referencia in referencias_custo:
        cenarios.append(
            dict(tipo="referencia_custo", perfil_variado=None, valor=referencia, perfis=perfis_ref,
                 globais=replace(globais_ref, referencia_custo=referencia))
        )
    cenarios.append(
        dict(tipo="sem_restricoes", perfil_variado=None, valor=None, globais=globais_ref,
             perfis=config.perfis_sem_restricoes(perfis_ref))
    )
    return cenarios


def sweep_restricoes(ref: Result, **kwargs) -> pd.DataFrame:
    """Avalia cada cenário de :func:`cenarios_restricoes` com :func:`avaliar_cenario`.

    Payoff e normalização são os da referência; só os perfis (e, no eixo
    ``referencia_custo``, os parâmetros globais) mudam. ``kwargs`` repassa os
    valores varridos a :func:`cenarios_restricoes`.
    """

    linhas = []
    for cen in cenarios_restricoes(ref.cfg.perfis, ref.cfg.globais, **kwargs):
        linhas.append(
            {
                "tipo": cen["tipo"],
                "perfil_variado": cen["perfil_variado"],
                "valor": cen["valor"],
                "referencia_custo": cen["globais"].referencia_custo,
                **avaliar_cenario(ref, cen["perfis"], cen["globais"]),
            }
        )
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 6. Forma do resíduo no benefício (quadrado / absoluto)
# ─────────────────────────────────────────────────────────────────────────
def sweep_residuo(ref: Result, formas: tuple[str, ...] = config.FORMAS_RESIDUO) -> pd.DataFrame:
    """Reconstrói o payoff com cada forma do resíduo e avalia com os perfis da referência.

    Os estados (e portanto $\mathcal{F}^{(p)}$) não mudam; só o benefício de
    ``Mp``, ``tb``, ``It`` e ``F`` e, com ele, a normalização comum.
    """

    linhas = []
    for forma in formas:
        X_novo, _ = stage1.build_payoff_tensor(
            ref.designs, ref.cfg.requisitos, ref.cfg.kappa_c, ref.cfg.kappa_cf, ref.modelo_balistico, forma,
            ref.cfg.criterios_comuns,
        )
        globais = replace(ref.cfg.globais, forma_residuo=forma)
        linhas.append(
            {
                "forma_residuo": forma,
                "referencia": forma == ref.cfg.globais.forma_residuo,
                **avaliar_cenario(ref, ref.cfg.perfis, globais, X_norm=stage1.common_minmax(X_novo)),
            }
        )
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 7. Cenários de FS dos perfis
# ─────────────────────────────────────────────────────────────────────────
CENARIOS_FS: Mapping[str, Mapping[str, float]] = {
    "referencia": {"W": 1.40, "S": 1.25, "H": 1.25},
    "todos_1,25": {"W": 1.25, "S": 1.25, "H": 1.25},
    "todos_1,40": {"W": 1.40, "S": 1.40, "H": 1.40},
}


def sweep_cenarios_fs(ref: Result, cenarios: Mapping[str, Mapping[str, float]] = CENARIOS_FS) -> pd.DataFrame:
    """Avalia cada cenário de FS: ``d*``, vencedor por perfil, top 10 e $\\tau$ frente a ``ref``.

    Payoff e normalização são os da referência; $\\mathcal{F}^{(p)}$ e, na
    variante com aderência ao FS, o alvo do critério acompanham o FS do cenário.
    ``top10`` traz os ``design_id`` separados por ``;``.
    """

    ids = ref.design_ids
    linhas = []
    for nome, fs in cenarios.items():
        perfis = ref.cfg.perfis
        for sigla, valor in fs.items():
            perfis = config.substituir_perfil(perfis, sigla, FS=valor)
        adm, resultado = _sintese_cenario(ref, perfis, ref.cfg.globais, ref.X_norm, ref.atributos_por_estado)
        linha: dict = {"cenario": nome, **{f"FS_{s}": v for s, v in fs.items()}}
        linha.update({f"n_F_{s}": int(adm.chi[s].sum()) for s in adm.siglas})
        if resultado is None:
            linhas.append({**linha, "d_star": None, **_metricas_indefinidas()})
            continue
        linha["d_star"] = ids[int(np.argmax(resultado.g))]
        for s in adm.siglas:
            k = stage1.preferred_design(resultado.g_por_perfil[s], adm.chi[s]) if s in resultado.g_por_perfil else None
            linha[f"vencedor_{s}"] = "nenhum" if k is None else ids[k]
        linha["top10"] = ";".join(str(i) for i in ids[np.argsort(-resultado.g)[:10]])
        linhas.append({**linha, **metricas_estabilidade(ref.g, resultado.g)})
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
# 8. Apenas o estado nominal
# ─────────────────────────────────────────────────────────────────────────
def sweep_nominal(ref: Result) -> pd.DataFrame:
    """Usa só $\\theta_1$, com a normalização refeita sobre ele.

    Com um único estado, as regras de Wald, Savage e Hurwicz coincidem no
    ordenamento; a comparação com ``ref`` isola o efeito da decisão sob incerteza.
    """

    X_norm = stage1.common_minmax(ref.X[:, :, [0]])
    nominal = ref.atributos_por_estado[config.ESTADOS_NOMES[0]]
    atributos = {estado: nominal for estado in config.ESTADOS_NOMES}  # max_j pcmax = pcmax(θ1)
    return pd.DataFrame(
        [{"caso": "apenas_theta1", **avaliar_cenario(ref, ref.cfg.perfis, ref.cfg.globais, X_norm=X_norm, atributos=atributos)}]
    )


__all__ = [
    "metricas_estabilidade",
    "regiao_robustez",
    "sweep_alpha",
    "sweep_pi",
    "sweep_anp",
    "sweep_lambda",
    "avaliar_cenario",
    "cenarios_restricoes",
    "sweep_restricoes",
    "sweep_residuo",
    "CENARIOS_FS",
    "sweep_cenarios_fs",
    "sweep_nominal",
]
