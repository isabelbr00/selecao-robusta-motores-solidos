#!/usr/bin/env python
"""Figuras do Capítulo 4 a partir dos resultados do pipeline (1º e, opcionalmente, 2º estágio).

Uso com os dados reais do 1º estágio (um painel por figura):
    python run_pipeline.py --stage 1 \
        --csv "../data/modeFrontier 1st Stage Us Propellants Designs Table.csv" --out resultados_estagio1
    python gerar_figuras_cap4.py \
        --dados1 "../data/modeFrontier 1st Stage Us Propellants Designs Table.csv" \
        --resultados1 resultados_estagio1 --out figuras_cap4

Uso com dois estágios (fronts de TESTE, a partir de ``tg_robust/``):
    python run_pipeline.py --stage 1 --csv ../gerador_pareto/dados/pareto_estagio1.csv \\
        --balistica ../gerador_pareto/balistica.py --out resultados_teste1
    python run_pipeline.py --stage 2 --csv ../gerador_pareto/dados/pareto_estagio2.csv \\
        --balistica ../gerador_pareto/balistica.py --out resultados_teste2
    python gerar_figuras_cap4.py \\
        --dados1 ../gerador_pareto/dados/pareto_estagio1.csv --resultados1 resultados_teste1 \\
        --dados2 ../gerador_pareto/dados/pareto_estagio2.csv --resultados2 resultados_teste2 \\
        --out figuras_cap4

O script só LÊ: os CSVs de saída de ``run_pipeline.py`` (``--resultadosN``) e o
:class:`tg_robust.pipeline.Result`, reconstruído com ``pipeline.run`` a partir do
CSV de designs (``--dadosN``) e da configuração gravada em ``run_config.json``
(perfis, FS, rho, b, referência do custo, γ, κ). O SHA-256 do CSV e o ``d*``
são conferidos contra ``run_config.json``, e ``g`` contra ``ranking_final.csv``.
Nada do pacote ``tg_robust`` é alterado.

Sem ``--dados2``/``--resultados2``, cada figura tem um único painel (1º estágio).

Saídas em ``--out``: ``<figura>.pdf`` (LaTeX) e ``<figura>.png`` (revisão) para
as 8 figuras, ``top5_final.csv``, ``sensibilidade_resumo.csv`` e ``tabela_comparacao.csv`` (7 atributos do d* em θ1, por estágio,
com a coluna "AHP original" vazia para preenchimento manual). Uma figura cujo
dado necessário não exista no Result nem nos CSVs é PULADA, com aviso.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tg_robust import anp, config, data, pipeline  # noqa: E402
import run_pipeline  # noqa: E402  (só escolher_modelo, para usar o mesmo modelo de balística da execução)

logging.getLogger("fontTools").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────
# Estilo comum
# ─────────────────────────────────────────────────────────────────────────
CORES_PERFIL = {"W": "#c62828", "S": "#1565c0", "H": "#2e7d32"}  # vermelho, azul, verde
COR_DSTAR = "black"
COR_FUNDO = "#c8c8c8"
ESTILO_PERFIL = {"W": "-", "S": "--", "H": ":"}  # distingue linhas sobrepostas
MARCADOR_PERFIL = {"W": "o", "S": "s", "H": "^"}

ROTULO_CRITERIO = {
    "Mp": r"$M_p$",
    "tb": r"$t_b$",
    "It": r"$I_t$",
    "F": r"$F$",
    "rhoIsp": r"$\rho I_{sp}$",
    "pcmax": r"$p_{c,\max}$",
    "eta": r"$\eta_v$",
}
ROTULO_ESTADO = (r"$\theta_1$" + "\n(nominal)", r"$\theta_2$" + "\n(moderado)", r"$\theta_3$" + "\n(severo)")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
    }
)

LARGURA = 7.0  # polegadas (largura útil típica de página A4 no LaTeX)
LARGURA_UM_PAINEL = 4.6  # figura com um só estágio


class DadoAusente(Exception):
    """Dado necessário para uma figura não está no Result nem nos CSVs."""


def fmt(x: float, casas: int = 2) -> str:
    """Número com vírgula decimal."""

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{casas}f}".replace(".", ",")


class FormatadorVirgula(ticker.ScalarFormatter):
    """ScalarFormatter com vírgula decimal (sem depender do locale do sistema)."""

    @staticmethod
    def _virgula(texto: str) -> str:
        # em mathtext, "{,}" evita o espaço que o TeX põe depois da vírgula
        return texto.replace(".", "{,}" if "$" in texto else ",")

    def __call__(self, x, pos=None):
        return self._virgula(super().__call__(x, pos))

    def get_offset(self):
        return self._virgula(super().get_offset())


def virgula(ax, eixos: str = "xy") -> None:
    if "x" in eixos:
        ax.xaxis.set_major_formatter(FormatadorVirgula())
    if "y" in eixos:
        ax.yaxis.set_major_formatter(FormatadorVirgula())


def salvar(fig, out: Path, nome: str, gerados: list[Path]) -> None:
    for ext in ("pdf", "png"):
        caminho = out / f"{nome}.{ext}"
        fig.savefig(caminho, dpi=200, bbox_inches="tight")
        gerados.append(caminho)
    plt.close(fig)


def largura(estagios: list) -> float:
    return LARGURA if len(estagios) > 1 else LARGURA_UM_PAINEL


def paineis(estagios: list, altura: float, **kw):
    """Uma coluna de painéis por estágio; devolve ``(fig, lista de eixos)``."""

    fig, eixos = plt.subplots(1, len(estagios), figsize=(largura(estagios), altura), squeeze=False, **kw)
    return fig, list(eixos[0])


def titulo_estagio(ax, estagio: int, extra: str = "") -> None:
    ax.set_title(f"{estagio}º estágio" + (f"\n{extra}" if extra else ""))


# ─────────────────────────────────────────────────────────────────────────
# Leitura dos resultados
# ─────────────────────────────────────────────────────────────────────────
class Estagio:
    """Result reconstruído + CSVs de saída de um estágio."""

    def __init__(self, dados: Path, resultados: Path, balistica: str | None):
        self.dir = resultados
        with open(resultados / "run_config.json", encoding="utf-8") as f:
            self.run_config = json.load(f)
        rc = self.run_config

        if rc.get("dados_sinteticos"):
            raise SystemExit(f"{resultados}: execução com dados sintéticos; passe a execução feita com --csv.")
        hash_esperado = rc.get("csv_sha256")
        if hash_esperado and data.hash_arquivo(dados) != hash_esperado:
            raise SystemExit(
                f"O CSV {dados} não é o mesmo usado em {resultados} (SHA-256 difere de run_config.json)."
            )

        self.cfg = _config_de_json(rc["config"])
        designs = data.load_designs(dados)
        balistica = balistica if balistica is not None else _resolver_balistica(rc.get("balistica_path"), resultados)
        model = run_pipeline.escolher_modelo(argparse.Namespace(balistica=balistica), designs)
        if type(model).__name__ != rc.get("modelo_balistico"):
            raise SystemExit(
                f"{resultados}: modelo de balística {type(model).__name__} difere do usado na execução "
                f"({rc.get('modelo_balistico')}); passe --balistica."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # avisos (b provisório, W_TT placeholder) já emitidos pelo run_pipeline
            self.r: pipeline.Result = pipeline.run(self.cfg, designs, model=model)

        self.ids = self.r.design_ids
        self._conferir()

    def _conferir(self) -> None:
        rc = self.run_config
        if str(self.ids[self.r.d_star]) != str(rc["d_star_design_id"]):
            raise SystemExit(f"{self.dir}: d* reconstruído difere do de run_config.json.")
        ranking = self.csv("ranking_final.csv", obrigatorio=False)
        if ranking is not None:
            g_por_id = pd.Series(self.r.g, index=self.ids.astype(str))
            if not np.allclose(g_por_id.loc[ranking["design_id"].astype(str)].to_numpy(), ranking["g"].to_numpy()):
                raise SystemExit(f"{self.dir}: g reconstruído difere de ranking_final.csv.")

    @property
    def estagio(self) -> int:
        return self.cfg.estagio

    def csv(self, nome: str, obrigatorio: bool = True) -> pd.DataFrame | None:
        caminho = self.dir / nome
        if not caminho.exists():
            if obrigatorio:
                raise DadoAusente(f"{caminho} não existe")
            return None
        try:
            return pd.read_csv(caminho)
        except pd.errors.EmptyDataError:
            if obrigatorio:
                raise DadoAusente(f"{caminho} está vazio")
            return None

    def vencedores(self) -> dict[str, int | None]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {s: self.r.vencedor_perfil(s) for s in self.cfg.siglas}

    def perfil(self, sigla: str) -> config.Perfil:
        return next(p for p in self.cfg.perfis if p.sigla == sigla)


def _config_de_json(d: dict) -> config.Config:
    perfis = tuple(
        config.Perfil(
            nome=p["nome"],
            sigla=p["sigla"],
            regra=p["regra"],
            alpha=None if p["alpha"] is None else float(p["alpha"]),
            FS=float(p["FS"]),
            rho=float(p["rho"]),  # "inf" -> math.inf
            peso_pi=float(p["peso_pi"]),
        )
        for p in d["perfis"]
    )
    g = d["globais"]
    return config.Config(
        estagio=int(d["estagio"]),
        seed=int(d["seed"]),
        perfis=perfis,
        globais=config.ConfigGlobal(p_lim=float(g["p_lim"]), b_cer=float(g["b_cer"]), referencia_custo=g["referencia_custo"]),
        gamma_anp=float(d["gamma_anp"]),
        kappa_c=tuple(float(k) for k in d["kappa_c"]),
        kappa_cf=tuple(float(k) for k in d["kappa_cf"]),
    )


def _resolver_balistica(caminho: str | None, resultados: Path) -> str | None:
    """O caminho gravado é relativo ao diretório em que o run_pipeline foi chamado."""

    if caminho is None:
        return None
    candidatos = [Path(caminho), Path(__file__).resolve().parent / caminho, resultados.parent / caminho]
    for c in candidatos:
        if c.exists():
            return str(c)
    raise SystemExit(f"Simulador de balística {caminho!r} (de run_config.json) não encontrado; passe --balistica.")


def legenda_vencedores(e: Estagio, venc: dict[str, int | None]) -> dict[str, str]:
    """Rótulo de legenda por perfil, com o design_id do vencedor."""

    return {s: f"{s}: {e.ids[k] if k is not None else 'nenhum'}" for s, k in venc.items()}


# ─────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────
def fig_coord_paralelas(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.4, sharey=True)
    x = np.arange(config.N_CRITERIOS)
    for ax, e in zip(eixos, estagios):
        Y = e.r.X_norm[:, :, 0]  # (R, n): benefício normalizado (min-max comum aos 3 estados) em θ1
        ax.plot(x, Y, color=COR_FUNDO, lw=0.4, alpha=0.35, zorder=1)
        venc = e.vencedores()
        rot = legenda_vencedores(e, venc)
        k_star = e.r.d_star
        # perfis cujo vencedor é o próprio d*: uma única linha, rotulada em conjunto
        iguais = [s for s, k in venc.items() if k == k_star]
        larguras = {"W": 4.0, "S": 2.6, "H": 1.4}  # sobrepostas continuam visíveis
        for s, k in venc.items():
            if k is None or s in iguais:
                continue
            ax.plot(x, Y[:, k], color=CORES_PERFIL[s], lw=larguras.get(s, 2.0), label=rot[s], zorder=2)
        if iguais:
            nomes = ", ".join(iguais) + " e d*"  # "W, S, H e d*"
            ax.plot(x, Y[:, k_star], color=COR_DSTAR, lw=1.8, label=f"{nomes}: {e.ids[k_star]}", zorder=3)
        else:
            ax.plot(x, Y[:, k_star], color=COR_DSTAR, lw=1.0, ls="--", label=f"d*: {e.ids[k_star]}", zorder=3)
        for xi in x:
            ax.axvline(xi, color="0.6", lw=0.6, zorder=0)
        ax.set_xticks(x, [ROTULO_CRITERIO[c] for c in config.CRITERIOS])
        ax.set_xlim(-0.2, config.N_CRITERIOS - 0.8)
        ax.set_ylim(-0.03, 1.03)
        ax.spines["bottom"].set_visible(False)
        virgula(ax, "y")
        titulo_estagio(ax, e.estagio)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    eixos[0].set_ylabel("Benefício normalizado em $\\theta_1$ (1 = melhor)")
    return fig


def fig_pesos_criterios(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 2.8, sharey=True)
    x = np.arange(config.N_CRITERIOS)
    w = 0.38
    for ax, e in zip(eixos, estagios):
        pesos = e.csv("pesos_efetivos.csv").set_index("criterio").loc[list(config.CRITERIOS)]
        ax.bar(x - w / 2, pesos["q0"], w, color="0.75", label="$q_0$ (a priori)")
        ax.bar(x + w / 2, pesos["q"], w, color="0.25", label="$q$ (efetivo)")
        ax.set_xticks(x, [ROTULO_CRITERIO[c] for c in config.CRITERIOS])
        virgula(ax, "y")
        titulo_estagio(ax, e.estagio, f"$\\gamma$ = {fmt(e.cfg.gamma_anp, 1)}")
    eixos[0].set_ylabel("Peso do critério")
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)
    return fig


def fig_margem_impulso(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.0)
    x = np.arange(len(config.ESTADOS_NOMES))
    for ax, e in zip(eixos, estagios):
        It_req = e.cfg.requisitos.It
        It = np.vstack([e.r.atributos_por_estado[est]["It"] for est in config.ESTADOS_NOMES])  # (J, n)
        venc = e.vencedores()
        rot = legenda_vencedores(e, venc)
        for s, k in venc.items():
            if k is None:
                continue
            desvio = 100.0 * (It[:, k] - It_req) / It_req
            ax.plot(x, desvio, color=CORES_PERFIL[s], ls=ESTILO_PERFIL[s], marker=MARCADOR_PERFIL[s],
                    ms=7 - 1.5 * list(venc).index(s), lw=1.6, label=rot[s])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.annotate("requisito", xy=(x[-1], 0.0), xytext=(2, 3), textcoords="offset points", fontsize=7)
        ax.set_xticks(x, ROTULO_ESTADO)
        ax.set_xlim(-0.3, x[-1] + 0.5)
        virgula(ax, "y")
        titulo_estagio(ax, e.estagio, f"$I_{{t,\\mathrm{{req}}}}$ = {fmt(It_req / 1e6, 3)}$\\times 10^6$ N·s")
        ax.legend(loc="best")
    eixos[0].set_ylabel("Desvio relativo de $I_t$ (%)")
    return fig


def fig_concordancia_perfis(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.4)
    dados_paineis = []
    for e in estagios:
        g_p = e.r.anp_resultado.g_por_perfil
        faltam = [s for s in e.cfg.siglas if s not in g_p]
        if faltam:
            raise DadoAusente(f"estágio {e.estagio}: g^(p) ausente para {faltam} (F^(p) vazio)")
        siglas = list(e.cfg.siglas)
        mask = e.r.admissibilidade.intersecao
        n_int = int(mask.sum())
        if n_int < 3:
            raise DadoAusente(f"estágio {e.estagio}: interseção dos F^(p) com {n_int} designs (< 3)")
        G = {s: g_p[s][mask] for s in siglas}
        m = len(siglas)
        rho = np.ones((m, m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    rho[i, j] = spearmanr(G[siglas[i]], G[siglas[j]]).statistic
        k = min(10, n_int)
        tops = [set(np.argsort(-G[s])[:k]) for s in siglas]
        inter_top = len(set.intersection(*tops))
        dados_paineis.append((e, siglas, rho, k, inter_top, n_int))
    rho_min = min(p[2].min() for p in dados_paineis)
    vmin = min(0.9, math.floor(rho_min * 20) / 20)  # escala comum aos dois estágios
    im = None
    for ax, (e, siglas, rho, k, inter_top, n_int) in zip(eixos, dados_paineis):
        m = len(siglas)
        im = ax.imshow(rho, cmap="viridis", vmin=vmin, vmax=1)
        for i in range(m):
            for j in range(m):
                ax.text(j, i, fmt(rho[i, j], 3), ha="center", va="center",
                        color="white" if rho[i, j] < vmin + 0.6 * (1 - vmin) else "black", fontsize=9)
        rotulos = [f"$g^{{({s})}}$" for s in siglas]
        ax.set_xticks(range(m), rotulos)
        ax.set_yticks(range(m), rotulos)
        for t, s in zip(ax.get_xticklabels(), siglas):
            t.set_color(CORES_PERFIL[s])
        for t, s in zip(ax.get_yticklabels(), siglas):
            t.set_color(CORES_PERFIL[s])
        for sp in ax.spines.values():
            sp.set_visible(False)
        titulo_estagio(ax, e.estagio, f"interseção dos top-{k}: {inter_top} de {k}\nn = {n_int} admissíveis a W, S e H")
    cb = fig.colorbar(im, ax=eixos, shrink=0.85, label="$\\rho$ de Spearman")
    cb.ax.yaxis.set_major_formatter(FormatadorVirgula())
    return fig


def fig_custo_pressao(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.4)
    for ax, e in zip(eixos, estagios):
        adm = e.r.admissibilidade
        x, y = adm.pcmax_max, adm.c
        ax.scatter(x, y, s=6, color="0.55", alpha=0.6, lw=0, label="designs")
        ax.scatter(x[e.r.d_star], y[e.r.d_star], s=40, facecolor="none", edgecolor=COR_DSTAR, lw=1.2,
                   label=f"d*: {e.ids[e.r.d_star]}", zorder=4)
        for s in e.cfg.siglas:
            p = e.perfil(s)
            ax.axvline(adm.globais.p_lim / p.FS, color=CORES_PERFIL[s], ls=ESTILO_PERFIL[s], lw=1.2)
            if math.isfinite(p.rho):
                ax.axhline(p.rho, color=CORES_PERFIL[s], ls=ESTILO_PERFIL[s], lw=1.2)
        r_p = pearsonr(x, y).statistic
        r_s = spearmanr(x, y).statistic
        ax.text(0.03, 0.97, f"Pearson r = {fmt(r_p, 3)}\nSpearman $\\rho_s$ = {fmt(r_s, 3)}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlabel("$\\max_j\\, p_{c,\\max}(d_k,\\theta_j)$ (MPa)")
        virgula(ax)
        sufixo = " (PROVISÓRIO)" if adm.globais.b_provisorio else ""
        titulo_estagio(ax, e.estagio, f"ref. = {adm.globais.referencia_custo}, b = {fmt(adm.globais.b_cer, 1)}{sufixo}")
    eixos[0].set_ylabel("Índice de custo relativo $c_k$")
    alcas = [Line2D([], [], color=CORES_PERFIL[s], ls=ESTILO_PERFIL[s],
                    label=f"{s}: $p_{{lim}}/FS$ = {fmt(estagios[0].cfg.globais.p_lim / estagios[0].perfil(s).FS, 2)} MPa; "
                          f"$\\rho$ = {fmt(estagios[0].perfil(s).rho, 2)}")
             for s in estagios[0].cfg.siglas]
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(handles=h[:1] + alcas, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)
    for ax in eixos:
        ax.legend(handles=ax.get_legend_handles_labels()[0][1:], loc="lower right")
    return fig


def fig_exclusoes_filtros(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.0, sharey=False)
    categorias = [
        ("excluidos_so_pressao", "só pressão", "0.35", None),
        ("excluidos_so_custo", "só custo", "0.65", None),
        ("excluidos_ambos", "ambos", "white", "////"),
    ]
    for ax, e in zip(eixos, estagios):
        res = e.csv("admissibilidade_resumo.csv")
        res = res[res["perfil"].isin(e.cfg.siglas)].set_index("perfil").loc[list(e.cfg.siglas)]
        x = np.arange(len(res))
        base = res["n_admissiveis"].to_numpy(dtype=float)
        ax.bar(x, base, color=[CORES_PERFIL[s] for s in res.index], edgecolor="black", lw=0.5)
        for xi, v in zip(x, base):
            ax.text(xi, v / 2, f"{int(v)}", ha="center", va="center", color="white", fontsize=8)
        for col, _, cor, hatch in categorias:
            v = res[col].to_numpy(dtype=float)
            ax.bar(x, v, bottom=base, color=cor, hatch=hatch, edgecolor="black", lw=0.5)
            for xi, b, vi in zip(x, base, v):
                if vi > 0.04 * res["n_designs"].iloc[0]:
                    ax.text(xi, b + vi / 2, f"{int(vi)}", ha="center", va="center",
                            color="white" if cor == "0.35" else "black", fontsize=8,
                            bbox=dict(fc="white", ec="none", pad=0.6) if hatch else None)
            base = base + v
        ax.set_xticks(x, [f"{s}\nFS = {fmt(e.perfil(s).FS)}\n$\\rho$ = {fmt(e.perfil(s).rho)}" for s in res.index])
        ax.set_ylim(0, res["n_designs"].iloc[0] * 1.02)
        titulo_estagio(ax, e.estagio, f"n = {int(res['n_designs'].iloc[0])} designs")
    eixos[0].set_ylabel("Número de designs")
    alcas = [Patch(facecolor="0.85", edgecolor="black", label="admissíveis (cor do perfil)")] + [
        Patch(facecolor=cor, hatch=hatch, edgecolor="black", label=rot) for _, rot, cor, hatch in categorias
    ]
    fig.legend(handles=alcas, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), frameon=False)
    return fig


def fig_top10_final(estagios: list[Estagio]):
    fig, eixos = paineis(estagios, 3.6, gridspec_kw={"wspace": 0.55})
    for ax, e in zip(eixos, estagios):
        g_p = e.r.anp_resultado.g_por_perfil
        pi_ef = anp.pi_renormalizado(e.cfg.pi, list(g_p))  # o mesmo pi usado na síntese
        top = np.argsort(-e.r.g)[:10]
        y = np.arange(len(top))[::-1]  # melhor no topo
        esquerda = np.zeros(len(top))
        for s in e.cfg.siglas:
            if s not in g_p:
                continue
            v = pi_ef[s] * g_p[s][top]
            ax.barh(y, v, left=esquerda, color=CORES_PERFIL[s], edgecolor="white", lw=0.4,
                    label=f"$\\pi_{s}\\, g^{{({s})}}$, $\\pi_{s}$ = {fmt(pi_ef[s], 3)}")
            esquerda += v
        i_star = int(np.where(top == e.r.d_star)[0][0])
        ax.barh(y[i_star], e.r.g[e.r.d_star], fill=False, edgecolor=COR_DSTAR, lw=1.8, zorder=5)
        rotulos = [str(e.ids[k]) + (" (d*)" if k == e.r.d_star else "") for k in top]
        ax.set_yticks(y, rotulos)
        ax.get_yticklabels()[i_star].set_fontweight("bold")
        g_top = e.r.g[top]
        for yi, gi in zip(y, g_top):
            ax.text(gi, yi, " " + fmt(gi * 1e3, 3), va="center", fontsize=7)
        ax.set_xlim(0.0, g_top.max() * 1.25)
        ax.set_xlabel("Prioridade global $g$")
        virgula(ax, "x")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
        titulo_estagio(ax, e.estagio)
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.text(0.5, -0.14, "Valor ao fim de cada barra: $g \\times 10^3$; comprimento total $g = \\sum_p \\pi_p\\, g^{(p)}$.",
             ha="center", fontsize=7, color="0.3")
    return fig


def _marcar_mudanca(ax, x, tau, muda) -> bool:
    muda = np.asarray(muda, dtype=bool)
    if muda.any():
        ax.scatter(np.asarray(x)[muda], np.asarray(tau)[muda], marker="X", s=60, color=COR_DSTAR, zorder=6)
    return bool(muda.any())


PARAMETROS_SENS = ("alpha", "pi", "gamma", "lambda", "FS", "rho")
TAU_PISO = 0.80  # limite inferior do eixo y, salvo se algum τ observado for menor


def casos_sensibilidade(e: Estagio, par: str) -> pd.DataFrame:
    """Casos de uma varredura: ``valor`` (texto), ``perfil_variado``, ``kendall_tau`` e ``d_star_muda``.

    Lê os mesmos CSVs e linhas que :func:`_painel_sensibilidade`. Casos com
    síntese indefinida (``sintese_definida=False``) ficam de fora.
    """

    if par == "alpha":
        df = e.csv("sens_alpha.csv")
        valor = df["alpha"].astype(float).map(str)
        perfil = ",".join(p.sigla for p in e.cfg.perfis if p.regra == "hurwicz")
    elif par == "pi":
        df = e.csv("sens_pi.csv")
        valor = df["variacao"]
        perfil = df["perfil_variado"]
    elif par == "gamma":
        df = e.csv("sens_anp.csv")
        df = df[df["tipo"] == "gamma"]
        if df.empty:
            raise DadoAusente("sens_anp.csv sem linhas tipo=gamma")
        valor = df["passo_saaty"].astype(float).map(str)
        perfil = ""
    elif par == "lambda":
        df = e.csv("sens_lambda.csv", obrigatorio=False)
        if df is None or df.empty:
            raise DadoAusente("sens_lambda.csv vazio (varredura de λ pulada: sem modelo de balística que reexecute os estados)")
        valor = df["lambda"].astype(float).map(str)
        perfil = ""
    else:
        df = e.csv("sens_restricoes.csv")
        df = df[df["tipo"] == par]
        if df.empty:
            raise DadoAusente(f"sens_restricoes.csv sem linhas tipo={par}")
        valor = df["valor"].astype(float).map(str)
        perfil = df["perfil_variado"]
    casos = pd.DataFrame(
        {
            "valor": valor,
            "perfil_variado": perfil,
            "kendall_tau": df["kendall_tau"].astype(float),
            "d_star_muda": ~df["mesmo_d_star"].astype(bool),
        }
    )
    if "sintese_definida" in df.columns:
        casos = casos[df["sintese_definida"].astype(bool)]
    return casos


def _rotulo_caso(par: str, valor: str, perfil: str) -> str:
    if par == "pi":  # valor = "+20%", "-20%" ou "pi=1"
        return f"pi_{perfil}=1" if valor == "pi=1" else f"pi_{perfil} {valor}"
    return f"{par}_{perfil}={valor}" if perfil and par in ("pi", "FS", "rho") else f"{par}={valor}"


def tabela_sensibilidade_resumo(estagios: list[Estagio], avisos: list[str]) -> pd.DataFrame:
    """Por parâmetro e estágio: menor τ, onde ocorre e se d* muda em algum caso."""

    linhas = []
    for e in estagios:
        for par in PARAMETROS_SENS:
            try:
                casos = casos_sensibilidade(e, par)
            except DadoAusente as exc:
                avisos.append(f"sensibilidade_resumo, {par}, estágio {e.estagio}: {exc}")
                continue
            tau_min = casos["kendall_tau"].min()
            no_min = casos[np.isclose(casos["kendall_tau"], tau_min)]  # empates: todos os valores
            muda = casos[casos["d_star_muda"]]
            linhas.append(
                {
                    "estagio": e.estagio,
                    "parametro": par,
                    "n_casos": len(casos),
                    "tau_min": tau_min,
                    "valor_no_tau_min": "; ".join(_rotulo_caso(par, v, p) for v, p in zip(no_min["valor"], no_min["perfil_variado"])),
                    "d_star_muda": "sim" if len(muda) else "não",
                    "casos_d_star_muda": "; ".join(_rotulo_caso(par, v, p) for v, p in zip(muda["valor"], muda["perfil_variado"])),
                }
            )
    return pd.DataFrame(linhas)


def limite_inferior_tau(estagios: list[Estagio]) -> float:
    """0,80, ou o menor τ observado arredondado para baixo (passo 0,05) se for menor."""

    taus = []
    for e in estagios:
        for par in PARAMETROS_SENS:
            try:
                taus.extend(casos_sensibilidade(e, par)["kendall_tau"].dropna())
            except DadoAusente:
                pass
    tau_min = min(taus) if taus else TAU_PISO
    return min(TAU_PISO, math.floor(tau_min * 20) / 20)


def fig_sensibilidade(estagios: list[Estagio], avisos: list[str]):
    parametros = list(PARAMETROS_SENS)
    titulos = {"alpha": "$\\alpha$ (Hurwicz)", "pi": "$\\pi$ (pesos dos perfis)", "gamma": "$\\gamma$ (ANP)",
               "lambda": "$\\lambda$ (magnitude dos estados)", "FS": "FS por perfil", "rho": "$\\rho$ por perfil"}
    y_min = limite_inferior_tau(estagios)  # mesmo eixo y em todos os painéis
    fig, eixos = plt.subplots(len(parametros), len(estagios), figsize=(largura(estagios), 13.0), squeeze=False)
    algum_muda = False
    for c, e in enumerate(estagios):
        for l, par in enumerate(parametros):
            ax = eixos[l, c]
            ax.set_title(f"{e.estagio}º estágio — {titulos[par]}" if l == 0 else titulos[par], pad=9)
            try:
                algum_muda |= _painel_sensibilidade(ax, e, par)
            except DadoAusente as exc:
                ax.text(0.5, 0.5, "varredura não disponível", ha="center", va="center", transform=ax.transAxes, color="0.4")
                ax.set_xticks([])
                ax.set_yticks([])
                avisos.append(f"sensibilidade, painel {par}, estágio {e.estagio}: {exc}")
                continue
            ax.set_ylabel("$\\tau$ de Kendall")
            ax.set_ylim(y_min, 1.0)
            for artista in ax.lines + ax.collections:
                artista.set_clip_on(False)  # marcadores em τ = 1 não ficam cortados
            virgula(ax, "y")
    alcas = [Line2D([], [], color=CORES_PERFIL[s], marker=MARCADOR_PERFIL[s], label=f"perfil {s} variado")
             for s in estagios[0].cfg.siglas]
    alcas += [Line2D([], [], color=COR_DSTAR, marker="X", ls="none", ms=8, label="d* muda"),
              Line2D([], [], color="0.5", ls=":", label="valor de referência")]
    fig.legend(handles=alcas, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def _painel_sensibilidade(ax, e: Estagio, par: str) -> bool:
    """Desenha um painel; devolve se algum ponto tem d* diferente do de referência."""

    if par == "alpha":
        df = e.csv("sens_alpha.csv")
        ref = [p.alpha for p in e.cfg.perfis if p.regra == "hurwicz"]
        ax.plot(df["alpha"], df["kendall_tau"], color=CORES_PERFIL.get("H", "0.3"), marker="^", ms=4)
        for a in ref:
            ax.axvline(a, color="0.5", ls=":")
        virgula(ax, "x")
        return _marcar_mudanca(ax, df["alpha"], df["kendall_tau"], ~df["mesmo_d_star"])

    if par == "pi":
        df = e.csv("sens_pi.csv")
        x = np.arange(len(df))
        cores = [CORES_PERFIL.get(s, "0.3") for s in df["perfil_variado"]]
        mark = [MARCADOR_PERFIL.get(s, "o") for s in df["perfil_variado"]]
        for xi, t, cor, m in zip(x, df["kendall_tau"], cores, mark):
            ax.scatter(xi, t, color=cor, marker=m, s=25, zorder=3)
        rot = [f"$\\pi_{s}$ = 1" if tipo == "canonico" else f"{s} {v.replace('-', '−')}"
               for tipo, s, v in zip(df["tipo"], df["perfil_variado"], df["variacao"])]
        ax.set_xticks(x, rot, rotation=45, ha="right")
        indef = ~df["sintese_definida"].astype(bool)
        for xi in x[indef.to_numpy()]:
            ax.text(xi, np.nanmin(df["kendall_tau"]) if df["kendall_tau"].notna().any() else 0.5,
                    "indef.", ha="center", fontsize=7, color="0.4")
        return _marcar_mudanca(ax, x, df["kendall_tau"], ~df["mesmo_d_star"].astype(bool) & ~indef)

    if par == "gamma":
        df = e.csv("sens_anp.csv")
        df = df[df["tipo"] == "gamma"].astype({"passo_saaty": float}).sort_values("passo_saaty")
        if df.empty:
            raise DadoAusente("sens_anp.csv sem linhas tipo=gamma")
        ax.plot(df["passo_saaty"], df["kendall_tau"], color="0.25", marker="o", ms=4)
        ax.axvline(e.cfg.gamma_anp, color="0.5", ls=":")
        virgula(ax, "x")
        return _marcar_mudanca(ax, df["passo_saaty"], df["kendall_tau"], ~df["mesmo_d_star"])

    if par == "lambda":
        df = e.csv("sens_lambda.csv", obrigatorio=False)
        if df is None or df.empty:
            raise DadoAusente("sens_lambda.csv vazio (varredura de λ pulada: sem modelo de balística que reexecute os estados)")
        ax.plot(df["lambda"], df["kendall_tau"], color="0.25", marker="o", ms=4)
        ax.axvline(1.0, color="0.5", ls=":")
        virgula(ax, "x")
        return _marcar_mudanca(ax, df["lambda"], df["kendall_tau"], ~df["mesmo_d_star"].astype(bool))

    # FS e rho: uma linha por perfil variado (demais na referência)
    df = e.csv("sens_restricoes.csv")
    df = df[df["tipo"] == par].copy()
    if df.empty:
        raise DadoAusente(f"sens_restricoes.csv sem linhas tipo={par}")
    df["valor"] = df["valor"].astype(float)
    finitos = df.loc[np.isfinite(df["valor"]), "valor"]
    passo = np.diff(np.unique(finitos)).min() if finitos.nunique() > 1 else 0.1
    pos_inf = finitos.max() + 2 * passo
    df["x"] = np.where(np.isfinite(df["valor"]), df["valor"], pos_inf)
    muda = False
    siglas = list(e.cfg.siglas)
    desloc = {s: (i - (len(siglas) - 1) / 2) * 0.12 * passo for i, s in enumerate(siglas)}  # separa pontos coincidentes
    for s in siglas:
        d = df[df["perfil_variado"] == s].sort_values("x")
        xs = d["x"] + desloc[s]
        ax.plot(xs, d["kendall_tau"], color=CORES_PERFIL[s], ls=ESTILO_PERFIL[s], marker=MARCADOR_PERFIL[s], ms=4)
        ref = getattr(e.perfil(s), par)
        ax.axvline(ref if math.isfinite(ref) else pos_inf, color=CORES_PERFIL[s], ls=":", lw=0.8, alpha=0.7)
        muda |= _marcar_mudanca(ax, xs, d["kendall_tau"], ~d["mesmo_d_star"].astype(bool))
    ticks = sorted(df["x"].unique())
    ax.set_xticks(ticks, ["∞" if t == pos_inf else fmt(t) for t in ticks])
    return muda


# ─────────────────────────────────────────────────────────────────────────
# Tabela de comparação
# ─────────────────────────────────────────────────────────────────────────
def tabela_top5(estagios: list[Estagio]) -> pd.DataFrame:
    """Top-5 por g, por estágio: g, diferença % para o d* e contribuição π_p g^(p) de cada perfil."""

    linhas = []
    for e in estagios:
        g_p = e.r.anp_resultado.g_por_perfil
        pi_ef = anp.pi_renormalizado(e.cfg.pi, list(g_p))  # o mesmo pi usado na síntese (e na top10_final)
        g_star = e.r.g[e.r.d_star]
        for pos, k in enumerate(np.argsort(-e.r.g)[:5], start=1):
            linha = {
                "estagio": e.estagio,
                "posicao": pos,
                "design_id": e.ids[k],
                "g": e.r.g[k],
                "dif_pct_d_star": 100.0 * (e.r.g[k] - g_star) / g_star,
            }
            for s in e.cfg.siglas:
                # perfil fora da síntese (F vazio) não contribui
                linha[f"contrib_{s}"] = pi_ef[s] * g_p[s][k] if s in g_p else np.nan
            linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_comparacao(estagios: list[Estagio]) -> pd.DataFrame:
    linhas = []
    for e in estagios:
        k = e.r.d_star
        nominal = e.r.atributos_por_estado[config.ESTADOS_NOMES[0]]
        for c in config.CRITERIOS:
            req = getattr(e.cfg.requisitos, c) if c in config.CRITERIOS_RESIDUO else np.nan
            linhas.append(
                {
                    "estagio": e.estagio,
                    "atributo": c,
                    "requisito": req,
                    "d_star_design_id": e.ids[k],
                    "ANP robusto (d*, theta1)": float(nominal[c][k]),
                    "AHP original": "",
                }
            )
    return pd.DataFrame(linhas)


# ─────────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados1", required=True, type=Path, help="CSV de designs do 1º estágio (o mesmo passado ao run_pipeline)")
    p.add_argument("--dados2", type=Path, default=None, help="CSV de designs do 2º estágio (opcional)")
    p.add_argument("--resultados1", required=True, type=Path, help="Diretório --out do run_pipeline, 1º estágio")
    p.add_argument("--resultados2", type=Path, default=None, help="Diretório --out do run_pipeline, 2º estágio (opcional)")
    p.add_argument("--balistica", default=None, help="Simulador .py (padrão: o de run_config.json)")
    p.add_argument("--out", type=Path, default=Path("figuras_cap4"), help="Pasta das figuras (padrão: figuras_cap4)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.dados2 is None) != (args.resultados2 is None):
        raise SystemExit("--dados2 e --resultados2 devem ser passados juntos (ou nenhum dos dois).")
    estagios = [Estagio(args.dados1, args.resultados1, args.balistica)]
    if args.dados2 is not None:
        estagios.append(Estagio(args.dados2, args.resultados2, args.balistica))
    if [e.estagio for e in estagios] != [1, 2][: len(estagios)]:
        raise SystemExit("--resultados1/--resultados2 devem ser execuções do 1º e do 2º estágio, nesta ordem.")

    gerados: list[Path] = []
    avisos: list[str] = []
    figuras = {
        "coord_paralelas": fig_coord_paralelas,
        "pesos_criterios": fig_pesos_criterios,
        "margem_impulso": fig_margem_impulso,
        "concordancia_perfis": fig_concordancia_perfis,
        "custo_pressao": fig_custo_pressao,
        "exclusoes_filtros": fig_exclusoes_filtros,
        "top10_final": fig_top10_final,
        "sensibilidade": lambda es: fig_sensibilidade(es, avisos),
    }
    for nome, funcao in figuras.items():
        try:
            fig = funcao(estagios)
        except DadoAusente as exc:
            plt.close("all")
            avisos.append(f"{nome}: PULADA — {exc}")
            continue
        salvar(fig, args.out, nome, gerados)

    caminho_tab = args.out / "tabela_comparacao.csv"
    tabela_comparacao(estagios).to_csv(caminho_tab, index=False)
    gerados.append(caminho_tab)
    for nome, tabela in (
        ("top5_final.csv", tabela_top5(estagios)),
        ("sensibilidade_resumo.csv", tabela_sensibilidade_resumo(estagios, [])),  # avisos já dados pela figura
    ):
        tabela.to_csv(args.out / nome, index=False)
        gerados.append(args.out / nome)

    print("Arquivos gerados:")
    for g in gerados:
        print(f"  {g}")
    if any(e.cfg.globais.b_provisorio for e in estagios):
        print("ATENÇÃO: expoente b da CER provisório — c_k e rho_p não são definitivos.")
    if avisos:
        print("Avisos (dados ausentes):")
        for a in avisos:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
