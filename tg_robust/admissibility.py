"""Restrições de admissibilidade por perfil organizacional (Seção 3.3.4).

Implementa a seção "Restrições de Admissibilidade por Perfil Organizacional"
de ``TG_revisao_FS_custo.tex``. Cada perfil ``p`` impõe dois **limiares** (não
critérios pontuados) que definem o conjunto admissível $\\mathcal{F}^{(p)}$:

* margem de segurança estrutural (Eq. "fs_restricao"):
  ``FS_p * max_j pcmax(d_k, theta_j) <= p_lim``;
* teto de custo relativo (Eq. "custo_relativo"): ``c_k <= rho_p``, com
  ``c_k = [pcmax(d_k, theta_1) Rc_k^2 Lt_k / min_d pcmax(d, theta_1) Rc_d^2 Lt_d]^b``.
  O mínimo é tomado sobre **todos** os designs do estágio, e só o estado
  nominal é usado (o envelope é dimensionado uma única vez). ``c_k`` não
  depende do perfil.

Enquanto o expoente ``b`` da CER não for definido (``b=None``), ``c_k`` fica
``NaN`` e o teto de custo não é aplicado (ver
:func:`tg_robust.config.aviso_b_cer_ausente`). A razão de massa do envelope
(a base de ``c_k``, sem o expoente) é sempre calculada e reportada.

Nada aqui altera o payoff, a normalização ou as pontuações $v^{(p)}_{k,r}$:
as máscaras são consumidas apenas pelo argmax de $V^{(p)}$ e pelo bloco
$W^{(p)}_{AT}$ do ANP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Admissibilidade:
    """Conjuntos admissíveis $\\mathcal{F}^{(p)}$ e as grandezas que os definem."""

    pcmax_max: np.ndarray  # max_j pcmax(d_k, theta_j), shape (n,)
    razao_massa: np.ndarray  # pcmax(θ1) Rc² Lt / min_d(...), shape (n,), >= 1
    c: np.ndarray  # índice de custo c_k = razao_massa**b, shape (n,); NaN se b=None
    b_cer: float | None
    adm_pressao: dict[str, np.ndarray]  # bool (n,) por perfil
    adm_custo: dict[str, np.ndarray]  # bool (n,) por perfil
    chi: dict[str, np.ndarray]  # bool (n,) por perfil: d_k em F^(p)

    @property
    def intersecao(self) -> np.ndarray:
        """Máscara dos designs admissíveis para todos os perfis simultaneamente."""

        return np.logical_and.reduce([self.chi[p] for p in config.PERFIS])

    def perfis_inadmissiveis(self, k: int) -> list[str]:
        """Perfis para os quais o design de posição ``k`` é inadmissível."""

        return [p for p in config.PERFIS if not self.chi[p][k]]


def pressao_maxima_entre_estados(atributos_por_estado: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """$\\max_j p_{c,\\max}(d_k, \\theta_j)$ para cada design."""

    return np.max(
        np.vstack([atributos_por_estado[e]["pcmax"] for e in config.ESTADOS_NOMES]), axis=0
    )


def razao_massa_envelope(pcmax_nominal: np.ndarray, Rc: np.ndarray, Lt: np.ndarray) -> np.ndarray:
    """Base do índice de custo: $p_{c,\\max}(\\theta_1) R^2 L$ relativo ao mínimo do estágio.

    Proporcional à massa do envelope cilíndrico de parede fina
    (Eq. "massa_envelope"); FS, material e o coeficiente da CER cancelam-se.
    """

    produto = np.asarray(pcmax_nominal, dtype=float) * np.asarray(Rc, dtype=float) ** 2 * np.asarray(Lt, dtype=float)
    if not (np.isfinite(produto).all() and (produto > 0.0).all()):
        raise ValueError("pcmax(θ1)·Rc²·Lt deve ser positivo e finito para todos os designs")
    return produto / produto.min()


def indice_custo(razao_massa: np.ndarray, b: float | None) -> np.ndarray:
    """$c_k = (\\text{razão de massa})^b$; devolve ``NaN`` enquanto ``b`` não estiver definido."""

    if b is None:
        return np.full_like(razao_massa, np.nan, dtype=float)
    return razao_massa**b


def conjuntos_admissiveis(
    designs: pd.DataFrame,
    atributos_por_estado: dict[str, dict[str, np.ndarray]],
    fs_perfis: tuple[float, float, float],
    rho_perfis: tuple[float, float, float],
    b_cer: float | None,
    p_lim: float = config.P_LIM_REF,
) -> Admissibilidade:
    """Calcula $\\mathcal{F}^{(p)}$ para os três perfis (Eq. "conjunto_viavel").

    ``atributos_por_estado`` é a saída de
    :func:`tg_robust.scenarios.propagate_states` (``pcmax`` por estado vem do
    modelo de balística interna); ``designs`` fornece ``Rc`` e ``Lt``.
    """

    pcmax_max = pressao_maxima_entre_estados(atributos_por_estado)
    pcmax_nominal = atributos_por_estado[config.ESTADOS_NOMES[0]]["pcmax"]
    razao = razao_massa_envelope(pcmax_nominal, designs["Rc"].to_numpy(), designs["Lt"].to_numpy())

    if b_cer is None:
        config.aviso_b_cer_ausente()
    c = indice_custo(razao, b_cer)

    adm_pressao: dict[str, np.ndarray] = {}
    adm_custo: dict[str, np.ndarray] = {}
    chi: dict[str, np.ndarray] = {}
    for p, fs, rho in zip(config.PERFIS, fs_perfis, rho_perfis):
        adm_pressao[p] = fs * pcmax_max <= p_lim
        adm_custo[p] = np.ones_like(adm_pressao[p]) if b_cer is None else c <= rho
        chi[p] = adm_pressao[p] & adm_custo[p]

    return Admissibilidade(
        pcmax_max=pcmax_max,
        razao_massa=razao,
        c=c,
        b_cer=b_cer,
        adm_pressao=adm_pressao,
        adm_custo=adm_custo,
        chi=chi,
    )


def resumo(adm: Admissibilidade) -> pd.DataFrame:
    """Tamanho de $\\mathcal{F}^{(p)}$ e motivo de exclusão, por perfil.

    Colunas: ``n_designs``, ``n_admissiveis``, ``excluidos_so_pressao``,
    ``excluidos_so_custo``, ``excluidos_ambos``. Uma linha extra
    ``intersecao`` traz o tamanho de $\\mathcal{F}^{(W)} \\cap \\mathcal{F}^{(S)} \\cap \\mathcal{F}^{(H)}$.
    """

    n = len(adm.pcmax_max)
    linhas = []
    for p in config.PERFIS:
        falha_p = ~adm.adm_pressao[p]
        falha_c = ~adm.adm_custo[p]
        linhas.append(
            {
                "perfil": p,
                "n_designs": n,
                "n_admissiveis": int(adm.chi[p].sum()),
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
            "excluidos_so_pressao": None,
            "excluidos_so_custo": None,
            "excluidos_ambos": None,
        }
    )
    return pd.DataFrame(linhas)


__all__ = [
    "Admissibilidade",
    "pressao_maxima_entre_estados",
    "razao_massa_envelope",
    "indice_custo",
    "conjuntos_admissiveis",
    "resumo",
]
