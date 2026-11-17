"""Estados da natureza e propagação pelo modelo de balística interna.

Define os três estados da natureza $\\Theta = \\{\\theta_1,\\theta_2,\\theta_3\\}$
como fatores multiplicativos sobre $c^{*}$ e $C_F$, e a interface :class:`BallisticModel`
que recalcula os atributos de desempenho de cada design sob um dado estado.

A implementação principal (caminho padrão, usada quando não há um modelo
de balística interna disponível) é :class:`SurrogateBallistics`, que aplica
diretamente as identidades analíticas:

* ``pcmax' = pcmax * kp``                          (pressão depende de c*)
* ``tb'    = tb / kp**n``
* ``It'    = It * kc * kcf``                        (It = Mp.Isp.g0, Isp = c*.CF/g0)
* ``F'     = It' / tb' = F * kcf * kp``
* ``rhoIsp' = rhoIsp * kc * kcf``

com ``kp = kc**(1/(1-n))``. ``Mp`` e ``eta`` são invariantes (geometria do
grão/câmara). Um protótipo anterior usava ``It * kcf * kp`` — incorreto; a
forma correta é ``It * kc * kcf`` (ver identidades testadas em
``tests/test_scenarios.py``).

Quando o modelo real de balística interna estiver disponível, basta
implementar :class:`BallisticModel` com a mesma assinatura e passá-lo ao
pipeline (``pipeline.run(cfg, designs, model=ModeloReal())``). Um simulador
por design no formato (``simular(x, k_cstar, k_cf) -> dict``) é adaptado por
:class:`AdaptadorSimulador` (carregado com :func:`carregar_simulador`).

Para a tabela do modeFRONTIER (dados reais do artigo, sem a balística),
:class:`EscalaQuaseEstacionaria` aplica as leis de escala quase
estacionárias diretamente aos valores tabelados — inclusive ``F`` e ``ta``,
que lá não satisfazem ``F = It/tb`` exatamente — e reproduz a tabela em θ1.

Para dados que já trazem os três estados calculados (colunas
``<criterio>_theta{1,2,3}``), :class:`EstadosPrecomputados` apenas lê essas
colunas; ele não reexecuta a balística, logo não serve para a varredura de
$\\lambda$ (``suporta_reexecucao = False``).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from . import config


class BallisticModel(Protocol):
    """Interface do modelo de balística interna usado na propagação de estados.

    Implementações devem recalcular, para cada design, os atributos de
    desempenho sob os fatores de degradação ``kc`` (sobre $c^{*}$) e ``kcf``
    (sobre $C_F$), e devolver um dicionário com um array por critério
    (:data:`config.ATRIBUTOS`), na mesma ordem/índice de ``designs``.

    ``suporta_reexecucao`` indica se o modelo aceita fatores ``kc``/``kcf``
    arbitrários (necessário para a varredura de $\\lambda$).
    """

    suporta_reexecucao: bool

    def __call__(
        self, designs: pd.DataFrame, kc: float, kcf: float
    ) -> dict[str, np.ndarray]:
        ...


class SurrogateBallistics:
    """Modelo substituto de balística interna (identidades analíticas).

    Caminho principal do pipeline: assume geometria do grão e da câmara
    fixas (``Mp`` e ``eta`` não dependem de ``kc``/``kcf``) e aplica as
    identidades descritas no módulo. Requer que ``designs`` tenha a coluna
    ``nexp`` (expoente da lei de queima), resolvida por
    :func:`selecao_robusta.data.resolver_nexp`.
    """

    suporta_reexecucao = True

    def __call__(
        self, designs: pd.DataFrame, kc: float, kcf: float
    ) -> dict[str, np.ndarray]:
        if "nexp" not in designs.columns:
            raise ValueError(
                "designs deve conter a coluna 'nexp' (ver data.resolver_nexp)"
            )

        n = designs["nexp"].to_numpy(dtype=float)
        tb = designs["tb"].to_numpy(dtype=float)
        It = designs["It"].to_numpy(dtype=float)
        pcmax = designs["pcmax"].to_numpy(dtype=float)
        rhoIsp = designs["rhoIsp"].to_numpy(dtype=float)
        Mp = designs["Mp"].to_numpy(dtype=float)
        eta = designs["eta"].to_numpy(dtype=float)

        kp = kc ** (1.0 / (1.0 - n))

        pcmax_novo = pcmax * kp
        tb_novo = tb / kp**n
        It_novo = It * kc * kcf
        F_novo = It_novo / tb_novo
        rhoIsp_novo = rhoIsp * kc * kcf

        return {
            "Mp": Mp.copy(),
            "tb": tb_novo,
            "It": It_novo,
            "F": F_novo,
            "rhoIsp": rhoIsp_novo,
            "pcmax": pcmax_novo,
            "eta": eta.copy(),
        }


class EscalaQuaseEstacionaria:
    """Estados por leis de escala quase estacionárias, sem rodar a balística.

    Com ``kp = kc**(1/(1-n))`` (``n`` da coluna ``nexp``, Tabela 6):

    * ``rhoIsp``, ``It``: × ``kc·kcf``
    * ``pcmax``, ``pcmed``: × ``kp``
    * ``tb``, ``ta``: × ``kc**(-n/(1-n))``
    * ``F``: × ``kcf·kp``
    * ``Mp``, ``eta``: invariantes

    ``F`` é escalado diretamente (não recalculado como ``It/tb``), de modo que
    em θ1 (``kc = kcf = 1``) a saída é idêntica aos valores da tabela.
    ``pcmed`` e ``ta`` só são devolvidos se existirem em ``designs``.
    """

    suporta_reexecucao = True

    def __call__(
        self, designs: pd.DataFrame, kc: float, kcf: float
    ) -> dict[str, np.ndarray]:
        if "nexp" not in designs.columns:
            raise ValueError(
                "designs deve conter a coluna 'nexp' (ver data.resolver_nexp)"
            )

        n = designs["nexp"].to_numpy(dtype=float)
        fator_isp = kc * kcf
        fator_p = kc ** (1.0 / (1.0 - n))
        fator_t = kc ** (-n / (1.0 - n))
        fatores = {
            "Mp": 1.0,
            "tb": fator_t,
            "ta": fator_t,
            "It": fator_isp,
            "F": kcf * fator_p,
            "rhoIsp": fator_isp,
            "pcmax": fator_p,
            "pcmed": fator_p,
            "eta": 1.0,
        }
        opcionais = ("ta", "pcmed")
        return {
            c: designs[c].to_numpy(dtype=float) * f
            for c, f in fatores.items()
            if c not in opcionais or c in designs.columns
        }


# Critérios que variam entre estados e são lidos de colunas <criterio>_theta{j}
CRITERIOS_POR_ESTADO: tuple[str, ...] = tuple(
    c for c in config.ATRIBUTOS if c not in config.CRITERIOS_INVARIANTES
)


def coluna_estado(criterio: str, j: int) -> str:
    """Nome da coluna do critério no estado ``j`` (1, 2 ou 3), p.ex. ``pcmax_theta2``."""

    return f"{criterio}_theta{j}"


class EstadosPrecomputados:
    """Lê os atributos de cada estado de colunas ``<criterio>_theta{j}`` dos dados.

    Só reconhece os pares ``(kc, kcf)`` de referência
    (:data:`config.KAPPA_C_REF`, :data:`config.KAPPA_CF_REF`); qualquer outro
    par exige reexecutar a balística e levanta ``ValueError``.
    """

    suporta_reexecucao = False

    def __call__(
        self, designs: pd.DataFrame, kc: float, kcf: float
    ) -> dict[str, np.ndarray]:
        pares = zip(config.KAPPA_C_REF, config.KAPPA_CF_REF)
        j = next(
            (i for i, (c, f) in enumerate(pares, start=1) if math.isclose(kc, c) and math.isclose(kcf, f)),
            None,
        )
        if j is None:
            raise ValueError(
                f"Estados pré-calculados só cobrem os fatores de referência; (kc={kc}, kcf={kcf}) "
                "exige reexecutar a balística (forneça um BallisticModel)."
            )
        saida = {c: designs[c].to_numpy(dtype=float).copy() for c in config.CRITERIOS_INVARIANTES}
        for c in CRITERIOS_POR_ESTADO:
            saida[c] = designs[coluna_estado(c, j)].to_numpy(dtype=float).copy()
        return saida


class AdaptadorSimulador:
    """Adapta um simulador por design ``simular(x, k_cstar, k_cf) -> dict`` à interface.

    ``x`` é montado a partir de cada linha de ``designs`` com as chaves de
    ``variaveis_projeto`` (renomeadas por ``mapa_entrada``, nome canônico ->
    nome do simulador); as saídas do simulador são renomeadas por
    ``mapa_saida`` (nome do simulador -> critério canônico). Os padrões
    correspondem a ``gerador_pareto/balistica.py``.
    """

    suporta_reexecucao = True

    def __init__(
        self,
        simular: Callable[..., Mapping[str, float]],
        variaveis_projeto: tuple[str, ...] = ("propelente", "Rc", "Lt", "Dt", "eps", "eta", "omega", "beta"),
        mapa_entrada: Mapping[str, str] | None = None,
        mapa_saida: Mapping[str, str] | None = None,
    ) -> None:
        self.simular = simular
        self.variaveis_projeto = variaveis_projeto
        self.mapa_entrada = dict(mapa_entrada) if mapa_entrada is not None else {"eta": "eta_v"}
        self.mapa_saida = dict(mapa_saida) if mapa_saida is not None else {"rho_isp": "rhoIsp", "eta_v": "eta"}

    def __call__(
        self, designs: pd.DataFrame, kc: float, kcf: float
    ) -> dict[str, np.ndarray]:
        faltantes = [v for v in self.variaveis_projeto if v not in designs.columns]
        if faltantes:
            raise ValueError(f"Variáveis de projeto ausentes para o simulador: {faltantes}")

        saida = {c: np.empty(len(designs)) for c in config.ATRIBUTOS}
        for i, linha in enumerate(designs[list(self.variaveis_projeto)].itertuples(index=False)):
            x = {self.mapa_entrada.get(v, v): val for v, val in zip(self.variaveis_projeto, linha)}
            resultado = self.simular(x, k_cstar=kc, k_cf=kcf)
            for nome, valor in resultado.items():
                criterio = self.mapa_saida.get(nome, nome)
                if criterio in saida:
                    saida[criterio][i] = valor
        return saida


def carregar_simulador(caminho: str | Path, funcao: str = "simular") -> Callable[..., Mapping[str, float]]:
    """Importa ``funcao`` do arquivo Python ``caminho`` (p.ex. ``gerador_pareto/balistica.py``)."""

    caminho = Path(caminho).resolve()
    spec = importlib.util.spec_from_file_location(f"_balistica_{caminho.stem}", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return getattr(modulo, funcao)


def propagate_states(
    designs: pd.DataFrame,
    kappa_c: tuple[float, float, float],
    kappa_cf: tuple[float, float, float],
    model: BallisticModel,
    estados_nomes: tuple[str, ...] = config.ESTADOS_NOMES,
) -> dict[str, dict[str, np.ndarray]]:
    """Propaga os designs por todos os estados da natureza.

    Retorna um dicionário ``{nome_do_estado: {criterio: array}}``, com um
    array por critério de desempenho para cada um dos três estados.
    """

    if not (len(kappa_c) == len(kappa_cf) == len(estados_nomes) == 3):
        raise ValueError("kappa_c, kappa_cf e estados_nomes devem ter 3 elementos")

    return {
        nome: model(designs, kc=kc, kcf=kcf)
        for nome, kc, kcf in zip(estados_nomes, kappa_c, kappa_cf)
    }


__all__ = [
    "BallisticModel",
    "SurrogateBallistics",
    "EscalaQuaseEstacionaria",
    "CRITERIOS_POR_ESTADO",
    "coluna_estado",
    "EstadosPrecomputados",
    "AdaptadorSimulador",
    "carregar_simulador",
    "propagate_states",
]
