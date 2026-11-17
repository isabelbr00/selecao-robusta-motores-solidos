"""Restrições de admissibilidade por perfil organizacional.

Cada perfil ``p`` (:class:`config.Perfil`)
impõe dois **limiares** (não critérios pontuados) que definem o conjunto
admissível $\\mathcal{F}^{(p)}$:

* margem de segurança estrutural (Eq. "fs_restricao"):
  ``FS_p * max_j pcmax(d_k, theta_j) <= p_lim`` — máximo sobre os TRÊS estados;
* teto de custo relativo: ``c_k <= rho_p``, com
  ``c_k = (base_k / ref)^b``, ``base_k = pcmax(d_k, theta_1) Vc_k`` e
  ``Vc_k = Mp_k / (rho_k eta_k)`` (volume da câmara, quando os dados trazem a
  densidade ``rho`` do propelente e ``eta``; caso contrário — fronts de
  teste —, ``base_k = pcmax(d_k, theta_1) Rc_k^2 Lt_k``) e
  ``ref`` = mediana (padrão) ou mínimo de ``base_d`` sobre **todos** os
  designs do estágio. A opção "mediana" é o padrão porque,
  com o mínimo, os tetos eliminam 80–99% dos designs nos fronts de teste.
  Só o estado nominal entra no custo (o envelope é dimensionado uma vez).
  ``c_k`` não depende do perfil; ``rho_p = inf`` retira o teto de custo.

Nada aqui altera o payoff, a normalização ou as pontuações $v^{(p)}_{k,r}$:
as máscaras são consumidas apenas pelo argmax de $V^{(p)}$ e pelo bloco
$W^{(p)}_{AT}$ do ANP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Admissibilidade:
    """Conjuntos admissíveis $\\mathcal{F}^{(p)}$ e as grandezas que os definem."""

    siglas: tuple[str, ...]
    pcmax_max: np.ndarray  # max_j pcmax(d_k, theta_j), shape (n,)
    pcmax_nominal: np.ndarray  # pcmax(d_k, theta_1), shape (n,)
    base_custo: np.ndarray  # pcmax(θ1)·Vc ou pcmax(θ1)·Rc²·Lt (unidades do dado), shape (n,)
    c: np.ndarray  # índice de custo c_k, shape (n,)
    globais: config.ConfigGlobal
    adm_pressao: dict[str, np.ndarray]  # bool (n,) por sigla
    adm_custo: dict[str, np.ndarray]  # bool (n,) por sigla
    chi: dict[str, np.ndarray]  # bool (n,) por sigla: d_k em F^(p)

    @property
    def intersecao(self) -> np.ndarray:
        """Máscara dos designs admissíveis para todos os perfis simultaneamente."""

        return np.logical_and.reduce([self.chi[s] for s in self.siglas])

    def perfis_inadmissiveis(self, k: int) -> list[str]:
        """Siglas dos perfis para os quais o design de posição ``k`` é inadmissível."""

        return [s for s in self.siglas if not self.chi[s][k]]

    @property
    def correlacao_custo_pressao(self) -> float:
        """Correlação de Pearson entre ``c_k`` e ``pcmax(theta_1)`` (acompanhamento)."""

        if np.ptp(self.c) == 0.0 or np.ptp(self.pcmax_nominal) == 0.0:
            return float("nan")
        return float(np.corrcoef(self.c, self.pcmax_nominal)[0, 1])


def pressao_maxima_entre_estados(atributos_por_estado: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """$\\max_j p_{c,\\max}(d_k, \\theta_j)$ para cada design."""

    return np.max(
        np.vstack([atributos_por_estado[e]["pcmax"] for e in config.ESTADOS_NOMES]), axis=0
    )


def base_custo_envelope(pcmax_nominal: np.ndarray, Rc: np.ndarray, Lt: np.ndarray) -> np.ndarray:
    """$p_{c,\\max}(\\theta_1) R^2 L$, proporcional à massa do envelope (Eq. "massa_envelope").

    FS, material e o coeficiente da CER cancelam-se na razão de
    :func:`indice_custo`, assim como as unidades de ``Rc`` e ``Lt``.
    """

    base = np.asarray(pcmax_nominal, dtype=float) * np.asarray(Rc, dtype=float) ** 2 * np.asarray(Lt, dtype=float)
    if not (np.isfinite(base).all() and (base > 0.0).all()):
        raise ValueError("pcmax(θ1)·Rc²·Lt deve ser positivo e finito para todos os designs")
    return base


def base_custo_volume(pcmax_nominal: np.ndarray, Mp: np.ndarray, rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """$p_{c,\max}(\theta_1) V_c$, com $V_c = M_p / (\rho_p \eta_v)$ (volume da câmara)."""

    Vc = np.asarray(Mp, dtype=float) / (np.asarray(rho, dtype=float) * np.asarray(eta, dtype=float))
    base = np.asarray(pcmax_nominal, dtype=float) * Vc
    if not (np.isfinite(base).all() and (base > 0.0).all()):
        raise ValueError("pcmax(θ1)·Mp/(rho·eta) deve ser positivo e finito para todos os designs")
    return base


def base_custo(designs: pd.DataFrame, pcmax_nominal: np.ndarray) -> np.ndarray:
    """Base do índice de custo: $p_{c,\max}(\theta_1) V_c$ se ``designs`` tem ``rho`` e
    ``eta``; senão $p_{c,\max}(\theta_1) R_c^2 L_t$ (fronts de teste)."""

    if "rho" in designs.columns and "eta" in designs.columns:
        return base_custo_volume(pcmax_nominal, designs["Mp"].to_numpy(), designs["rho"].to_numpy(), designs["eta"].to_numpy())
    return base_custo_envelope(pcmax_nominal, designs["Rc"].to_numpy(), designs["Lt"].to_numpy())


def indice_custo(base: np.ndarray, b: float, referencia: str) -> np.ndarray:
    """$c_k = (base_k / ref)^b$, com ``ref`` = mediana ou mínimo sobre todos os designs."""

    if referencia == "mediana":
        ref = np.median(base)
    elif referencia == "minimo":
        ref = np.min(base)
    else:
        raise ValueError(f"referencia_custo deve ser uma de {config.REFERENCIAS_CUSTO}")
    return (base / ref) ** b


def conjuntos_admissiveis(
    designs: pd.DataFrame,
    atributos_por_estado: dict[str, dict[str, np.ndarray]],
    perfis: Sequence[config.Perfil],
    globais: config.ConfigGlobal,
) -> Admissibilidade:
    """Calcula $\\mathcal{F}^{(p)}$ para cada perfil da lista (Eq. "conjunto_viavel").

    ``atributos_por_estado`` é a saída de
    :func:`selecao_robusta.scenarios.propagate_states` (``pcmax`` por estado vem do
    modelo de balística interna); ``designs`` fornece ``Mp``, ``rho`` e
    ``eta`` (ou ``Rc`` e ``Lt``) para a base do custo (:func:`base_custo`).
    """

    pcmax_max = pressao_maxima_entre_estados(atributos_por_estado)
    pcmax_nominal = np.asarray(atributos_por_estado[config.ESTADOS_NOMES[0]]["pcmax"], dtype=float)
    base = base_custo(designs, pcmax_nominal)

    c = indice_custo(base, globais.b_cer, globais.referencia_custo)

    adm_pressao: dict[str, np.ndarray] = {}
    adm_custo: dict[str, np.ndarray] = {}
    chi: dict[str, np.ndarray] = {}
    for perfil in perfis:
        adm_pressao[perfil.sigla] = perfil.FS * pcmax_max <= globais.p_lim
        adm_custo[perfil.sigla] = c <= perfil.rho  # rho = inf: sempre verdadeiro
        chi[perfil.sigla] = adm_pressao[perfil.sigla] & adm_custo[perfil.sigla]

    return Admissibilidade(
        siglas=tuple(p.sigla for p in perfis),
        pcmax_max=pcmax_max,
        pcmax_nominal=pcmax_nominal,
        base_custo=base,
        c=c,
        globais=globais,
        adm_pressao=adm_pressao,
        adm_custo=adm_custo,
        chi=chi,
    )


def resumo(adm: Admissibilidade) -> pd.DataFrame:
    """Tamanho de $\\mathcal{F}^{(p)}$ e motivo de exclusão, por perfil.

    Colunas: ``n_designs``, ``n_admissiveis``, ``excluidos_so_pressao``,
    ``excluidos_so_custo``, ``excluidos_ambos``. Uma linha extra
    ``intersecao`` traz o tamanho da interseção de todos os $\\mathcal{F}^{(p)}$.
    """

    n = len(adm.pcmax_max)
    linhas = []
    for s in adm.siglas:
        falha_p = ~adm.adm_pressao[s]
        falha_c = ~adm.adm_custo[s]
        linhas.append(
            {
                "perfil": s,
                "n_designs": n,
                "n_admissiveis": int(adm.chi[s].sum()),
                "excluidos_so_pressao": int((falha_p & ~falha_c).sum()),
                "excluidos_so_custo": int((falha_c & ~falha_p).sum()),
                "excluidos_ambos": int((falha_p & falha_c).sum()),
            }
        )
    linhas.append(
        {
            "perfil": "intersecao",
            "n_designs": n,
            "n_admissiveis": int(adm.intersecao.sum()),
            "excluidos_so_pressao": pd.NA,
            "excluidos_so_custo": pd.NA,
            "excluidos_ambos": pd.NA,
        }
    )
    return pd.DataFrame(linhas).astype(
        {"excluidos_so_pressao": "Int64", "excluidos_so_custo": "Int64", "excluidos_ambos": "Int64"}
    )


__all__ = [
    "Admissibilidade",
    "pressao_maxima_entre_estados",
    "base_custo_envelope",
    "base_custo_volume",
    "base_custo",
    "indice_custo",
    "conjuntos_admissiveis",
    "resumo",
]
