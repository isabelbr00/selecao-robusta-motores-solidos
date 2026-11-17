#!/usr/bin/env python
"""CLI do pipeline de seleção robusta de motores sólidos (ANP + Wald/Savage/Hurwicz).

Uso:
    python run_pipeline.py --stage 1 --out resultados_estagio1 --seed 0
    python run_pipeline.py --stage 2 --csv caminho/para/designs.csv --out resultados_estagio2
    python run_pipeline.py --stage 2 --csv caminho/para/designs.csv \\
        --balistica caminho/para/simulador.py --out resultados_estagio2

Sem ``--csv``, gera designs sintéticos (:func:`tg_robust.data.synthetic_designs`)
marcados explicitamente como tal — ver aviso no terminal e a coluna
``sintetico`` / o campo ``dados_sinteticos`` em ``run_config.json``.

Modelo de balística (estados θ e varredura de λ):
* ``--balistica arquivo.py``: simulador ``simular(x, k_cstar, k_cf)`` do arquivo
  — reexecuta todos os estados;
* CSV com colunas ``<criterio>_theta{1,2,3}`` e sem ``--balistica``: estados
  lidos dos dados; a varredura de λ é PULADA (com aviso);
* tabela do modeFRONTIER (dados reais do artigo): leis de escala quase
  estacionárias (``EscalaQuaseEstacionaria``), com varredura de λ;
* demais casos: modelo substituto analítico (``SurrogateBallistics``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tg_robust import admissibility, baseline, config, data, pipeline, scenarios, sensitivity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True, help="Estágio (1 ou 2)")
    parser.add_argument("--csv", type=str, default=None, help="CSV de designs Pareto; se omitido, usa dados sintéticos")
    parser.add_argument(
        "--balistica",
        type=str,
        default=None,
        help="Arquivo .py com simular(x, k_cstar, k_cf) para reexecutar a balística (estados e varredura de λ)",
    )
    parser.add_argument("--out", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=0, help="Semente aleatória (reprodutibilidade)")
    parser.add_argument(
        "--b-cer",
        type=float,
        default=config.B_CER,
        help=f"Expoente b da CER de motores sólidos (Koelle, TransCost); padrão = {config.B_CER}",
    )
    parser.add_argument(
        "--referencia-custo",
        choices=config.REFERENCIAS_CUSTO,
        default=config.ConfigGlobal().referencia_custo,
        help="Referência do índice de custo c_k (mediana ou mínimo sobre todos os designs)",
    )
    parser.add_argument(
        "--forma-residuo",
        choices=config.FORMAS_RESIDUO,
        default=config.ConfigGlobal().forma_residuo,
        help="Benefício de Mp, tb, It e F: -δ² (quadrado, Eq. 30 do artigo; padrão) ou -|δ| (absoluto)",
    )
    parser.add_argument(
        "--criterios",
        type=int,
        choices=sorted(config.VARIANTES_CRITERIOS),
        default=config.N_CRITERIOS,
        help="Variante de critérios: 6 (padrão; pcmax só no filtro de FS e no custo), "
        "7 (pcmax como critério) ou 5 (critérios do artigo)",
    )
    parser.add_argument(
        "--criterio-fs-alvo",
        action="store_true",
        help="Acrescenta o critério de aderência ao FS alvo do perfil, -((FS_ef - FS_p)/FS_p)², "
        "com FS_ef = p_lim/pcmax(θ1) e auto-laço em W_TT",
    )
    parser.add_argument(
        "--sem-teto-custo",
        action="store_true",
        help="Retira o teto de custo (rho = inf em todos os perfis); só a restrição de FS permanece",
    )
    parser.add_argument(
        "--verificar-limite-potencias",
        action="store_true",
        help="Verifica o limite por potências contra o limite fechado (mais lento)",
    )
    return parser.parse_args(argv)


def _json_seguro(obj):
    """Converte ``inf``/``nan`` em texto para gerar JSON estrito (``rho = inf`` é válido)."""

    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_seguro(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_seguro(v) for v in obj]
    return obj


NENHUM = "nenhum"


def tabela_vencedores(resultado: pipeline.Result) -> pd.DataFrame:
    """Vencedor de cada perfil: ANP (resultado principal) vs. pesos iguais (comparação).

    ``vencedor_anp`` = argmax de g^(p) em F^(p); ``vencedor_pesos_iguais`` =
    argmax de V^(p) em F^(p); ``divergem`` = "sim"/"não". As colunas
    ``*_sem_restricoes`` repetem a comparação sem máscara (todos os designs).
    F^(p) vazio dá ``"nenhum"``.
    """

    ids = resultado.design_ids

    def nome(k: int | None):
        return NENHUM if k is None else ids[k]

    linhas = []
    for s in resultado.cfg.siglas:
        linha = {"estagio": resultado.cfg.estagio, "perfil": s}
        for sufixo, com in (("", True), ("_sem_restricoes", False)):
            anp_k = nome(resultado.vencedor_perfil(s, com_restricoes=com))
            iguais_k = nome(resultado.vencedor_pesos_iguais(s, com_restricoes=com))
            linha[f"vencedor_anp{sufixo}"] = anp_k
            linha[f"vencedor_pesos_iguais{sufixo}"] = iguais_k
            linha[f"divergem{sufixo}"] = "sim" if anp_k != iguais_k else "não"
        linha["excluido_da_sintese"] = s in resultado.anp_resultado.perfis_excluidos
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_pesos_efetivos(resultado: pipeline.Result) -> pd.DataFrame:
    """q0 (a priori) e q = (1-γ)(I - γ W_TT)^{-1} q0 (efetivo), com a razão q/q0."""

    q0, q = resultado.anp_resultado.q0, resultado.anp_resultado.q
    return pd.DataFrame({"criterio": list(resultado.cfg.criterios), "q0": q0, "q": q, "q_sobre_q0": q / q0})


def escolher_modelo(args: argparse.Namespace, designs: pd.DataFrame) -> scenarios.BallisticModel:
    if args.balistica is not None:
        logger.info("Balística reexecutada pelo simulador de %s", args.balistica)
        return scenarios.AdaptadorSimulador(scenarios.carregar_simulador(args.balistica))
    if data.e_fonte_modefrontier(designs):
        logger.info("Estados θ por leis de escala quase estacionárias (tabela do modeFRONTIER)")
        return scenarios.EscalaQuaseEstacionaria()
    if data.tem_estados_precomputados(designs):
        logger.info("Estados θ lidos dos dados (sem reexecução da balística)")
        return scenarios.EstadosPrecomputados()
    return scenarios.SurrogateBallistics()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dados_sinteticos = args.csv is None
    csv_hash = None
    if args.csv is not None:
        designs = data.load_designs(args.csv)
        csv_hash = data.hash_arquivo(args.csv)
        logger.info("Designs carregados de %s (n=%d)", args.csv, len(designs))
    else:
        designs = data.synthetic_designs(args.stage, seed=args.seed)
        logger.warning(
            "Nenhum --csv fornecido: usando %d designs SINTÉTICOS (estágio %d, seed=%d). "
            "Resultados não representam os dados reais de Nascimento et al. (2022).",
            len(designs),
            args.stage,
            args.seed,
        )

    perfis = config.PERFIS_REFERENCIA
    if args.sem_teto_custo:
        perfis = tuple(dataclasses.replace(p, rho=math.inf) for p in perfis)
    cfg = config.Config(
        estagio=args.stage,
        seed=args.seed,
        criterios=config.VARIANTES_CRITERIOS[args.criterios]
        + ((config.CRITERIO_FS_ALVO,) if args.criterio_fs_alvo else ()),
        perfis=perfis,
        globais=config.ConfigGlobal(
            b_cer=args.b_cer, referencia_custo=args.referencia_custo, forma_residuo=args.forma_residuo
        ),
        verificar_limite_por_potencias=args.verificar_limite_potencias,
    )
    model = escolher_modelo(args, designs)

    logger.info("Executando pipeline (Estágio 1 + ANP) para %d designs...", len(designs))
    resultado = pipeline.run(cfg, designs, model=model)
    ids = resultado.design_ids
    siglas = cfg.siglas

    logger.info("Calculando baseline AHP (aproximação)...")
    bl = baseline.ahp_baseline(designs, cfg.requisitos)

    logger.info("Executando análise de sensibilidade (7 varreduras)...")
    sens_alpha = sensitivity.sweep_alpha(resultado)
    sens_pi = sensitivity.sweep_pi(resultado)
    sens_anp = sensitivity.sweep_anp(resultado)
    sens_lambda = sensitivity.sweep_lambda(resultado)
    sens_restricoes = sensitivity.sweep_restricoes(resultado)
    sens_residuo = sensitivity.sweep_residuo(resultado)
    sens_cenarios_fs = sensitivity.sweep_cenarios_fs(resultado)
    sens_nominal = sensitivity.sweep_nominal(resultado)

    adm = resultado.admissibilidade

    # ─── scores_<perfil>.csv ────────────────────────────────────────────
    for sigla, v in resultado.scores.items():
        df_scores = pd.DataFrame(v, columns=list(cfg.criterios))
        df_scores.insert(0, "design_id", ids)
        df_scores["V"] = resultado.preference_vectors[sigla]
        df_scores.to_csv(out_dir / f"scores_{sigla}.csv", index=False)

    # ─── admissibilidade.csv (por design) e admissibilidade_resumo.csv ──
    df_adm = pd.DataFrame(
        {
            "design_id": ids,
            "pcmax_theta1": adm.pcmax_nominal,
            "pcmax_max_estados": adm.pcmax_max,
            "base_custo": adm.base_custo,
            "indice_custo_c": adm.c,
        }
    )
    for s in siglas:
        df_adm[f"adm_pressao_{s}"] = adm.adm_pressao[s]
        df_adm[f"adm_custo_{s}"] = adm.adm_custo[s]
        df_adm[f"F_{s}"] = adm.chi[s]
    df_adm["F_intersecao"] = adm.intersecao
    df_adm.to_csv(out_dir / "admissibilidade.csv", index=False)
    resumo_adm = admissibility.resumo(adm)
    resumo_adm.to_csv(out_dir / "admissibilidade_resumo.csv", index=False)

    # ─── vencedores_por_perfil.csv (ANP vs. pesos iguais, com e sem restrições) ─
    df_venc = tabela_vencedores(resultado)
    df_venc.to_csv(out_dir / "vencedores_por_perfil.csv", index=False)

    # ─── pesos_efetivos.csv (efeito do ANP sobre os pesos dos critérios) ─
    tabela_pesos_efetivos(resultado).to_csv(out_dir / "pesos_efetivos.csv", index=False)

    # ─── ranking_final.csv (top-50) ─────────────────────────────────────
    g_por_perfil = resultado.anp_resultado.g_por_perfil
    colunas = {"design_id": ids, "g": resultado.g}
    for s in siglas:
        # perfil excluído da síntese (F vazio) não tem g^(p)
        colunas[f"g_{s}"] = g_por_perfil.get(s, np.full(len(ids), np.nan))
    for s in siglas:
        colunas[f"V_{s}"] = resultado.preference_vectors[s]
    for s in siglas:
        colunas[f"F_{s}"] = adm.chi[s]
    ranking = pd.DataFrame(colunas).join(bl, how="left")
    ranking = ranking.sort_values("g", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank_integrado", np.arange(1, len(ranking) + 1))
    ranking.head(50).to_csv(out_dir / "ranking_final.csv", index=False)

    # ─── consistency.csv ────────────────────────────────────────────────
    resultado.consistency_df.to_csv(out_dir / "consistency.csv", index=False)

    # ─── sensibilidade ──────────────────────────────────────────────────
    sens_alpha.to_csv(out_dir / "sens_alpha.csv", index=False)
    sens_pi.drop(columns=["pi"]).to_csv(out_dir / "sens_pi.csv", index=False)
    sens_anp.to_csv(out_dir / "sens_anp.csv", index=False)
    sens_lambda.to_csv(out_dir / "sens_lambda.csv", index=False)
    sens_restricoes.to_csv(out_dir / "sens_restricoes.csv", index=False)
    sens_residuo.to_csv(out_dir / "sens_residuo.csv", index=False)
    sens_cenarios_fs.to_csv(out_dir / "sens_cenarios_fs.csv", index=False)
    sens_nominal.to_csv(out_dir / "sens_nominal.csv", index=False)

    # ─── run_config.json ────────────────────────────────────────────────
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "n_designs": len(designs),
        "n_duplicatas_removidas": designs.attrs.get("n_duplicatas_removidas"),
        "dados_sinteticos": dados_sinteticos,
        "csv_path": args.csv,
        "csv_sha256": csv_hash,
        "modelo_balistico": type(model).__name__,
        "balistica_path": args.balistica,
        "sens_lambda_executada": not sens_lambda.empty,
        "config": dataclasses.asdict(cfg),
        "verificar_limite_por_potencias": args.verificar_limite_potencias,
        "cr_excede_limite": bool(resultado.consistency_df["cr_excede_limite"].any()),
        "d_star_design_id": ids[resultado.d_star],
        "referencia_custo": cfg.globais.referencia_custo,
        "forma_residuo": cfg.globais.forma_residuo,
        "tamanho_F": {s: int(adm.chi[s].sum()) for s in siglas},
        "tamanho_intersecao_F": int(adm.intersecao.sum()),
        "correlacao_c_pcmax_theta1": adm.correlacao_custo_pressao,
        "perfis_excluidos_da_sintese": list(resultado.anp_resultado.perfis_excluidos),
        "d_star_inadmissivel_em": resultado.d_star_inadmissivel_em,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(_json_seguro(run_config), f, indent=2, ensure_ascii=False, allow_nan=False, default=str)

    # ─── resumo no terminal ─────────────────────────────────────────────
    print()
    print(f"=== Resumo — Estágio {args.stage} (n={len(designs)}, "
          f"{'SINTÉTICO' if dados_sinteticos else 'dados de ' + Path(args.csv).name}) ===")
    print(f"Critérios ({len(cfg.criterios)}): {', '.join(cfg.criterios)}")
    print(f"Referência do custo: {cfg.globais.referencia_custo}; b = {cfg.globais.b_cer}; resíduo: {cfg.globais.forma_residuo}")
    print(f"Design selecionado (d*): design_id={run_config['d_star_design_id']}, g={resultado.g[resultado.d_star]:.4f}")
    inadm = run_config["d_star_inadmissivel_em"]
    print(f"d* inadmissível para os perfis: {', '.join(inadm) if inadm else 'nenhum'}")
    print("Conjuntos admissíveis F^(p):")
    print(resumo_adm.to_string(index=False))
    print(f"Correlação c_k x pcmax(θ1): {adm.correlacao_custo_pressao:.3f}")
    if run_config["perfis_excluidos_da_sintese"]:
        print(f"Perfis excluídos da síntese (F vazio): {run_config['perfis_excluidos_da_sintese']}")
    print("Vencedor por perfil (ANP: argmax g^(p) em F^(p)); entre parênteses, pesos iguais (argmax V^(p)):")
    for _, linha in df_venc.iterrows():
        print(f"  {linha['perfil']}: {linha['vencedor_anp']} ({linha['vencedor_pesos_iguais']})"
              f"  | sem restrições: {linha['vencedor_anp_sem_restricoes']} ({linha['vencedor_pesos_iguais_sem_restricoes']})")
    if not run_config["sens_lambda_executada"]:
        print("ATENÇÃO: varredura de λ pulada (sem modelo de balística que reexecute os estados; use --balistica).")
    print(f"CR dos julgamentos excede 0,10? {run_config['cr_excede_limite']}")
    print(f"Saídas gravadas em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
