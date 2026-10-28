"""Estados da natureza e propagação pelo modelo de balística interna.

Define os três estados da natureza $\\Theta = \\{\\theta_1,\\theta_2,\\theta_3\\}$
(Seção "Definição dos Estados da Natureza" da TG) como fatores
multiplicativos sobre $c^{*}$ e $C_F$, e a interface :class:`BallisticModel`
que recalcula os atributos de desempenho de cada design sob um dado estado.

A implementação principal (caminho padrão, usada quando os scripts de
balística interna do orientador não estão disponíveis) é
:class:`SurrogateBallistics`, que aplica diretamente as identidades da
Seção 3.3.1:

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
pipeline (``pipeline.run(cfg, designs, model=ModeloReal())``).
"""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from . import config


class BallisticModel(Protocol):
    """Interface do modelo de balística interna usado na propagação de estados.

    Implementações devem recalcular, para cada design, os atributos de
    desempenho sob os fatores de degradação ``kc`` (sobre $c^{*}$) e ``kcf``
    (sobre $C_F$), e devolver um dicionário com um array por critério
    (:data:`config.CRITERIOS`), na mesma ordem/índice de ``designs``.
    """

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
    :func:`tg_robust.data.resolver_nexp`.
    """

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


__all__ = ["BallisticModel", "SurrogateBallistics", "propagate_states"]
