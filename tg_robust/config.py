"""Configuração e constantes do pipeline de seleção robusta de motores sólidos.

Centraliza, em um único lugar, os parâmetros de referência descritos no
Capítulo 3 da TG (``cap3_metodologia_revisado.tex``): requisitos propulsivos
por estágio, fatores dos estados da natureza, parâmetros de decisão
(α de Hurwicz, γ do ANP, π dos perfis), as restrições de admissibilidade
por perfil (FS, teto de custo ρ, expoente ``b`` da CER — Seção 3.3.4 de
``TG_revisao_FS_custo.tex``) e o mapeamento Tabela 6 de tipo de
propelente → expoente da lei de queima ``n``.

Não há estado global mutável: ``Config`` é uma dataclass congelada
(``frozen=True``), e qualquer variação de parâmetro (análise de
sensibilidade, perturbação de julgamentos) é feita criando uma nova
instância com ``dataclasses.replace``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# Ordem fixa dos 7 critérios de desempenho (Seção 3.2 / Tabela "criterios_payoff")
# ─────────────────────────────────────────────────────────────────────────
CRITERIOS: tuple[str, ...] = ("Mp", "tb", "It", "F", "rhoIsp", "pcmax", "eta")
N_CRITERIOS = len(CRITERIOS)

# Critérios cujo objetivo é aproximar um requisito (resíduo relativo, Eq. residuo)
CRITERIOS_RESIDUO: tuple[str, ...] = ("Mp", "tb", "It", "F")
# Critério a maximizar diretamente
CRITERIOS_MAXIMIZAR: tuple[str, ...] = ("rhoIsp", "eta")
# Critério a minimizar diretamente (benefício = -valor)
CRITERIOS_MINIMIZAR: tuple[str, ...] = ("pcmax",)

# Critérios invariantes à propagação de cenários (Seção 3.3.1): dependem da
# geometria do grão/câmara, não de c* nem de C_F.
CRITERIOS_INVARIANTES: tuple[str, ...] = ("Mp", "eta")

# Restrições de projeto do artigo de referência (Nascimento et al., 2022)
PCMAX_LIMITE_SUPERIOR: float = 16.0  # MPa, restrição: pcmax < 16
ETA_LIMITE_INFERIOR: float = 0.8  # restrição: eta_v >= 0.8

# Variáveis de projeto do envelope (Nascimento et al., 2022), usadas apenas no
# índice de custo (Seções 3.2 e 3.3.4): raio do grão R_c e comprimento total L_t.
COLUNAS_GEOMETRIA: tuple[str, ...] = ("Rc", "Lt")

# ─────────────────────────────────────────────────────────────────────────
# Requisitos propulsivos por estágio (Tabela "requisitos", MSGRA)
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Requisitos:
    """Requisitos propulsivos de referência de um estágio."""

    Mp: float
    tb: float
    It: float
    F: float


REQUISITOS_ESTAGIO: Mapping[int, Requisitos] = {
    1: Requisitos(Mp=8780.34, tb=129.4, It=2.602e7, F=2.0107e5),
    2: Requisitos(Mp=1655.20, tb=25.4, It=4.926e6, F=1.9394e5),
}

# Número de designs Pareto reportados por Nascimento et al. (2022) por estágio
# (apenas para referência/validação de tamanho de amostra sintética; não é
# usado para forçar um valor nos dados reais).
N_DESIGNS_REFERENCIA: Mapping[int, int] = {1: 632, 2: 1089}

# ─────────────────────────────────────────────────────────────────────────
# Tabela 6 do artigo: tipo de propelente -> expoente da lei de queima n
# ─────────────────────────────────────────────────────────────────────────
TIPO_PROPELENTE_PARA_N: Mapping[int, float] = {1: 0.31, 2: 0.30, 3: 0.35, 4: 0.39}
N_PADRAO: float = 0.31  # usado quando nem 'propelente' nem 'nexp' estão presentes

# ─────────────────────────────────────────────────────────────────────────
# Estados da natureza (Tabela "cenarios"): fatores multiplicativos sobre
# c* (kappa_c*) e C_F (kappa_CF). Ordem fixa: nominal, moderado, severo.
# ─────────────────────────────────────────────────────────────────────────
ESTADOS_NOMES: tuple[str, ...] = ("nominal", "moderado", "severo")
KAPPA_C_REF: tuple[float, float, float] = (1.0, 0.99, 0.98)
KAPPA_CF_REF: tuple[float, float, float] = (1.0, 0.965, 0.93)

# ─────────────────────────────────────────────────────────────────────────
# Parâmetros de decisão de referência
# ─────────────────────────────────────────────────────────────────────────
ALPHA_HURWICZ_REF: float = 0.8
GAMMA_ANP_REF: float = 0.5
PI_PERFIS_REF: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
PERFIS: tuple[str, ...] = ("W", "S", "H")  # Wald (governamental), Savage (comercial), Hurwicz (new space)

# ─────────────────────────────────────────────────────────────────────────
# Restrições de admissibilidade por perfil (Seção 3.3.4 e Tabela
# "parametros_perfis" de TG_revisao_FS_custo.tex). Atuam como LIMIARES sobre o
# conjunto de designs de cada perfil, não como critérios pontuados.
# ─────────────────────────────────────────────────────────────────────────
# Capacidade estrutural de referência do envelope: a restrição pcmax < 16 MPa
# do artigo reinterpretada como p_lim (FS unitário implícito no modelo original).
P_LIM_REF: float = PCMAX_LIMITE_SUPERIOR  # MPa
# Fator de segurança mínimo sobre max_j pcmax, na ordem de PERFIS (W, S, H)
FS_PERFIS_REF: tuple[float, float, float] = (1.40, 1.25, 1.25)
# Teto do índice de custo relativo c_k, na ordem de PERFIS (W, S, H)
RHO_PERFIS_REF: tuple[float, float, float] = (1.50, 1.15, 1.25)
# Expoente b da CER de motores sólidos (TransCost, Koelle 2013).
# PLACEHOLDER: ainda não definido. Enquanto for None, c_k não é calculado e a
# restrição de teto de custo NÃO é aplicada (todos os designs são
# economicamente admissíveis), com aviso explícito (aviso_b_cer_ausente).
# TODO: preencher com o valor da CER de motores sólidos do Koelle (2013), com página/tabela.
B_CER_REF: float | None = None

# Índice de consistência aleatória de Saaty por ordem da matriz (RI)
RI_SAATY: Mapping[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
}
CR_LIMITE: float = 0.10

# Constante gravitacional padrão usada em Isp = c*.CF/g0 e It = Mp.Isp.g0
G0: float = 9.80665  # m/s^2


@dataclass(frozen=True)
class Config:
    """Configuração imutável de uma execução do pipeline.

    Qualquer variação paramétrica (sensibilidade, perturbação) deve criar uma
    nova instância via ``dataclasses.replace(cfg, campo=novo_valor)`` — nunca
    mutar uma instância existente.
    """

    estagio: int  # 1 ou 2
    seed: int = 0
    alpha_hurwicz: float = ALPHA_HURWICZ_REF
    gamma_anp: float = GAMMA_ANP_REF
    pi_perfis: tuple[float, float, float] = PI_PERFIS_REF
    kappa_c: tuple[float, float, float] = KAPPA_C_REF
    kappa_cf: tuple[float, float, float] = KAPPA_CF_REF
    p_lim: float = P_LIM_REF
    fs_perfis: tuple[float, float, float] = FS_PERFIS_REF
    rho_perfis: tuple[float, float, float] = RHO_PERFIS_REF
    b_cer: float | None = B_CER_REF
    verificar_limite_por_potencias: bool = False

    def __post_init__(self) -> None:
        if self.estagio not in (1, 2):
            raise ValueError(f"estagio deve ser 1 ou 2, recebido {self.estagio!r}")
        if not (0.0 <= self.alpha_hurwicz <= 1.0):
            raise ValueError("alpha_hurwicz deve estar em [0, 1]")
        if not (0.0 < self.gamma_anp < 1.0):
            raise ValueError("gamma_anp deve estar em (0, 1)")
        if abs(sum(self.pi_perfis) - 1.0) > 1e-9:
            raise ValueError("pi_perfis deve somar 1")
        if len(self.kappa_c) != 3 or len(self.kappa_cf) != 3:
            raise ValueError("kappa_c e kappa_cf devem ter 3 elementos (um por estado)")
        if not self.p_lim > 0.0:
            raise ValueError("p_lim deve ser positivo")
        if len(self.fs_perfis) != 3 or len(self.rho_perfis) != 3:
            raise ValueError("fs_perfis e rho_perfis devem ter 3 elementos (W, S, H)")
        if any(not fs > 0.0 for fs in self.fs_perfis):
            raise ValueError("fs_perfis deve ser positivo")
        if any(not rho >= 1.0 for rho in self.rho_perfis):
            # c_k >= 1 por construção (razão ao mínimo), logo rho < 1 esvazia F^(p)
            raise ValueError("rho_perfis deve ser >= 1 (ou math.inf para desativar o teto)")
        if self.b_cer is not None and not (self.b_cer > 0.0 and math.isfinite(self.b_cer)):
            raise ValueError("b_cer deve ser positivo e finito (ou None enquanto não definido)")

    @property
    def requisitos(self) -> Requisitos:
        return REQUISITOS_ESTAGIO[self.estagio]


def kappa_escalado(kappa_ref: tuple[float, float, float], lam: float) -> tuple[float, float, float]:
    """Escala a magnitude de degradação dos estados 2 e 3 por ``lam``.

    Implementa ``kappa_j = 1 - lam*(1 - kappa_j_ref)`` (análise de
    sensibilidade, item 4): ``lam=1`` reproduz o valor de referência,
    ``lam=0`` colapsa todos os estados ao nominal (kappa=1).
    """

    return tuple(1.0 - lam * (1.0 - float(k)) for k in kappa_ref)  # type: ignore[return-value]


def aviso_b_cer_ausente() -> str:
    """Emite (e devolve) o aviso de que o expoente ``b`` da CER não foi definido."""

    aviso = (
        "Expoente b da CER de motores sólidos (Koelle, TransCost) NÃO definido "
        "(config.B_CER_REF=None / --b-cer ausente): o índice de custo c_k não foi "
        "calculado e o teto de custo rho_p NÃO foi aplicado; apenas a restrição "
        "estrutural (FS) define F^(p). Preencha b antes de reportar resultados."
    )
    warnings.warn(aviso, stacklevel=2)
    return aviso


__all__ = [
    "CRITERIOS",
    "N_CRITERIOS",
    "CRITERIOS_RESIDUO",
    "CRITERIOS_MAXIMIZAR",
    "CRITERIOS_MINIMIZAR",
    "CRITERIOS_INVARIANTES",
    "PCMAX_LIMITE_SUPERIOR",
    "ETA_LIMITE_INFERIOR",
    "COLUNAS_GEOMETRIA",
    "P_LIM_REF",
    "FS_PERFIS_REF",
    "RHO_PERFIS_REF",
    "B_CER_REF",
    "Requisitos",
    "REQUISITOS_ESTAGIO",
    "N_DESIGNS_REFERENCIA",
    "TIPO_PROPELENTE_PARA_N",
    "N_PADRAO",
    "ESTADOS_NOMES",
    "KAPPA_C_REF",
    "KAPPA_CF_REF",
    "ALPHA_HURWICZ_REF",
    "GAMMA_ANP_REF",
    "PI_PERFIS_REF",
    "PERFIS",
    "RI_SAATY",
    "CR_LIMITE",
    "G0",
    "Config",
    "kappa_escalado",
    "aviso_b_cer_ausente",
]
