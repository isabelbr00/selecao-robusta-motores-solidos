"""Orquestração do pipeline completo: Estágio 1 + Estágio 2 (ANP).

:func:`run` recebe a configuração (:class:`selecao_robusta.config.Config`), os
designs Pareto (:class:`pandas.DataFrame`, de :mod:`selecao_robusta.data`) e,
opcionalmente, um modelo de balística interna (:class:`selecao_robusta.scenarios.BallisticModel`;
por padrão usa o substituto :class:`~selecao_robusta.scenarios.SurrogateBallistics`),
e devolve um :class:`Result` com todos os artefatos intermediários
necessários para as saídas do CLI (``run_pipeline.py``) e para a análise de
sensibilidade.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import admissibility, anp, config
from . import judgments as J
from . import stage1
from .scenarios import BallisticModel, SurrogateBallistics


@dataclass(frozen=True)
class Result:
    """Resultado completo de uma execução do pipeline para um estágio."""

    cfg: config.Config
    designs: pd.DataFrame
    X: np.ndarray  # payoff bruto, shape (R, n, 3)
    X_norm: np.ndarray  # payoff normalizado (min-max comum), shape (R, n, 3)
    atributos_por_estado: dict[str, dict[str, np.ndarray]]  # saída do modelo de balística
    scores: dict[str, np.ndarray]  # v^(p), shape (n, R) por perfil
    preference_vectors: dict[str, np.ndarray]  # V^(p) (pesos iguais; só comparação), shape (n,)
    admissibilidade: admissibility.Admissibilidade  # F^(p) por perfil (Seção 3.3.4)
    W_TT: np.ndarray
    julgamentos: list[J.ColumnJudgment]
    consistency_df: pd.DataFrame
    anp_resultado: anp.ResultadoANP
    modelo_balistico: BallisticModel  # usado para os estados; reaproveitado na varredura de lambda

    @property
    def design_ids(self) -> np.ndarray:
        """Identificadores dos designs (coluna ``design_id`` se existir, senão a posição)."""

        if "design_id" in self.designs.columns:
            return self.designs["design_id"].to_numpy()
        return self.designs.index.to_numpy()

    @property
    def g(self) -> np.ndarray:
        """Prioridades globais finais (síntese entre perfis)."""

        return self.anp_resultado.g

    @property
    def d_star(self) -> int:
        """Índice (posição em ``designs``) do design selecionado."""

        return int(np.argmax(self.g))

    def vencedor_perfil(self, sigla: str, com_restricoes: bool = True) -> int | None:
        """Vencedor do perfil (resultado principal): argmax de $g^{(p)}$ sobre $\\mathcal{F}^{(p)}$.

        $g^{(p)}$ é a prioridade da sub-rede ANP do perfil, a mesma que entra
        na síntese de ``d*``. Devolve a posição do design; se
        $\\mathcal{F}^{(p)}$ for vazio, devolve ``None`` ("nenhum") com aviso.
        Com ``com_restricoes=False``, usa $g^{(p)}$ recalculado sem máscara
        (:meth:`g_perfil_sem_restricoes`) e o argmax sobre todos os designs.
        """

        if not com_restricoes:
            return stage1.preferred_design(self.g_perfil_sem_restricoes(sigla))
        chi = self.admissibilidade.chi[sigla]
        if not chi.any():
            warnings.warn(
                f"Conjunto admissível vazio para o perfil {sigla!r}: vencedor do perfil = nenhum.",
                stacklevel=2,
            )
            return None
        return stage1.preferred_design(self.anp_resultado.g_por_perfil[sigla], chi)

    def g_perfil_sem_restricoes(self, sigla: str) -> np.ndarray:
        """$g^{(p)}$ da sub-rede do perfil com $W^{(p)}_{AT}$ sem máscara (todos os designs)."""

        W_AT = anp.build_W_AT(self.scores[sigla])
        return anp.profile_limit_closed(W_AT, self.W_TT, self.cfg.gamma_anp, self.anp_resultado.q0)[0]

    def vencedor_pesos_iguais(self, sigla: str, com_restricoes: bool = True) -> int | None:
        """Referência de comparação: argmax de $V^{(p)}$ (pesos iguais, sem interdependência).

        Equivale à agregação do AHP original de pesos iguais; não define o
        vencedor do perfil (ver :meth:`vencedor_perfil`). ``None`` se
        ``com_restricoes`` e $\\mathcal{F}^{(p)}$ for vazio.
        """

        chi = self.admissibilidade.chi[sigla] if com_restricoes else None
        return stage1.preferred_design(self.preference_vectors[sigla], chi)

    @property
    def d_star_inadmissivel_em(self) -> list[str]:
        """Perfis para os quais o design final ``d*`` é inadmissível."""

        return self.admissibilidade.perfis_inadmissiveis(self.d_star)


def run(
    cfg: config.Config,
    designs: pd.DataFrame,
    model: BallisticModel | None = None,
) -> Result:
    """Executa o Estágio 1 e o Estágio 2 (ANP) para os ``designs`` dados.

    Usa os critérios de ``cfg.criterios`` e a ``W_TT`` validada restrita a eles
    (:func:`selecao_robusta.judgments.build_WTT_validada`).
    """

    model = model if model is not None else SurrogateBallistics()
    designs = designs.reset_index(drop=True)

    X, atributos_por_estado = stage1.build_payoff_tensor(
        designs, cfg.requisitos, cfg.kappa_c, cfg.kappa_cf, model, cfg.globais.forma_residuo, cfg.criterios_comuns
    )
    X_norm = stage1.common_minmax(X)
    fs_ef = stage1.fs_efetivo(cfg.criterios, atributos_por_estado, cfg.globais.p_lim)
    scores = stage1.compute_profile_scores(X_norm, cfg.perfis, fs_ef)
    preference_vectors = {p: stage1.preference_vector(v) for p, v in scores.items()}

    # Restrições de admissibilidade: atuam só no argmax de V^(p) e em W_AT^(p),
    # nunca no payoff, na normalização ou nas pontuações v^(p).
    adm = admissibility.conjuntos_admissiveis(designs, atributos_por_estado, cfg.perfis, cfg.globais)

    W_TT, julgamentos = J.build_WTT_validada(cfg.criterios)
    R = len(cfg.criterios)
    q0 = np.full(R, 1.0 / R)

    anp_resultado = anp.run_anp(
        scores,
        W_TT,
        gamma=cfg.gamma_anp,
        pi=cfg.pi,
        q0=q0,
        verificar_limite_por_potencias=cfg.verificar_limite_por_potencias,
        chi_por_perfil=adm.chi,
    )

    consistency_df = J.consistency_report(julgamentos)

    return Result(
        cfg=cfg,
        designs=designs,
        X=X,
        X_norm=X_norm,
        atributos_por_estado=atributos_por_estado,
        scores=scores,
        preference_vectors=preference_vectors,
        admissibilidade=adm,
        W_TT=W_TT,
        julgamentos=julgamentos,
        consistency_df=consistency_df,
        anp_resultado=anp_resultado,
        modelo_balistico=model,
    )


__all__ = ["Result", "run"]
