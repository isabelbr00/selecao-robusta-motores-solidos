# tg_robust — Seleção Robusta Pós-Pareto de Motores Sólidos

Implementação computacional da metodologia do Capítulo 3 da TG
(`../cap3_metodologia_revisado.tex`, fonte da verdade): seleção pós-Pareto de
motores sólidos combinando critérios de decisão sob incerteza estrita
(**Wald**, **Savage**, **Hurwicz** — Estágio 1) com o **Processo Analítico em
Rede (ANP)** (Estágio 2), aplicada aos designs Pareto de
Nascimento et al. (2022).

Este pacote é independente do restante de `TG/` — nenhum arquivo existente
(`tratamento_dados.py`, `ANP.ipynb`, `data/`, `resultados/`, capítulos, .bib)
foi modificado, movido ou apagado. `tratamento_dados.py` é um **protótipo
anterior**, com uma metodologia diferente (fatores de cenário aplicados
diretamente a todas as métricas, pesos fixos por persona) — não é reutilizado
aqui; a metodologia correta é a descrita no `.tex` e resumida abaixo.

## Instalação

```bash
conda activate projects   # Python 3.11, numpy/scipy/pandas já instalados
pip install pytest        # se ainda não instalado no ambiente
```

Dependências: `numpy`, `scipy`, `pandas` (execução) e `pytest` (testes).
Nenhuma outra dependência externa.

## Uso rápido

```bash
cd tg_robust
python run_pipeline.py --stage 1 --out resultados_estagio1 --seed 0
python run_pipeline.py --stage 2 --out resultados_estagio2 --seed 0

# com dados reais (quando disponíveis), mesmo layout de colunas:
python run_pipeline.py --stage 1 --csv /caminho/para/pareto_estagio_1.csv --out resultados_estagio1
```

Sem `--csv`, o pipeline usa **dados sintéticos** (`tg_robust/data.py:synthetic_designs`),
explicitamente marcados como tal (aviso no terminal, coluna `sintetico` e
`run_config.json:dados_sinteticos=true`). **Não são dados reais do artigo.**

### Saídas (`--out`)

| Arquivo | Conteúdo |
|---|---|
| `scores_W.csv`, `scores_S.csv`, `scores_H.csv` | Pontuações $v^{(p)}_{k,r}$ por critério e preferência $V^{(p)}$, por perfil |
| `ranking_final.csv` | Top-50 por $g$, com $g, g_W, g_S, g_H, V_W, V_S, V_H$ e rank do baseline AHP |
| `criterion_weights.csv` | $q_0$ (a priori) vs. $q$ (efetivo, pós-dependência interna) |
| `consistency.csv` | CR por coluna de julgamento de $W_{TT}$ |
| `sens_alpha.csv`, `sens_pi.csv`, `sens_anp.csv`, `sens_lambda.csv` | As 4 varreduras de sensibilidade |
| `run_config.json` | Configuração completa, semente, flag de dados sintéticos, hash SHA-256 do CSV de entrada (se houver) |

## Onde plugar os dados reais

1. Gere/exporte os designs Pareto de Nascimento et al. (2022) em CSV com as
   colunas `Mp, tb, It, F, rhoIsp, pcmax, eta` (aliases como `F_med`,
   `rho_Isp`, `pc_med`, `eta_v` — usados em `../data/pareto_estagio_*.csv` —
   são aceitos automaticamente, ver `data.ALIASES`).
2. Inclua a coluna `propelente` (tipo 1-4, Tabela 6 do artigo) ou `nexp`
   (expoente da lei de queima); sem nenhuma das duas, o pipeline usa
   `n=0,31` para todos os designs e emite um aviso.
3. Rode com `--csv caminho.csv`.

> **Nota sobre `../data/pareto_estagio_1.csv` e `pareto_estagio_2.csv`:**
> esses arquivos já existem em `TG/data/` (632 e 1089 linhas — mesma
> contagem do artigo) e foram lidos apenas para verificação dos aliases de
> coluna, **não foram usados nem copiados para dentro deste pacote**. Mais
> de 85% das linhas de `pareto_estagio_1.csv` violam a restrição `eta>=0,8`
> do artigo, então aparentam ser dados de desenvolvimento do protótipo
> antigo, não o *front* de Pareto real validado. Confirme a procedência
> antes de usá-los como entrada real (ver "Decisões em aberto" no relatório
> final da sessão).

## Onde plugar o modelo real de balística interna

`tg_robust/scenarios.py` define a interface `BallisticModel`:

```python
class BallisticModel(Protocol):
    def __call__(self, designs: pd.DataFrame, kc: float, kcf: float) -> dict[str, np.ndarray]:
        ...
```

Implemente uma classe com essa assinatura (recalculando a curva de empuxo
por inteiro a partir de $c^{*}_j = \kappa_{c^{*},j} c^{*}$ e
$C_{F,j} = \kappa_{C_F,j} C_F$, conforme a Seção "Definição dos Estados da
Natureza" do `.tex`) e passe-a ao pipeline:

```python
resultado = pipeline.run(cfg, designs, model=ModeloRealDeBalistica())
```

Na ausência do modelo real (scripts do orientador), o pipeline usa por
padrão `SurrogateBallistics`, que aplica diretamente as identidades
analíticas da Seção 3.3.1 (ver docstring do módulo) — este é o "FALLBACK"
mencionado no `.tex`.

## Arquitetura

```
tg_robust/
  tg_robust/
    config.py       # Config (dataclass imutável) + constantes dos critérios/requisitos
    data.py         # load_designs, synthetic_designs, validação de restrições
    scenarios.py    # BallisticModel (interface) + SurrogateBallistics + propagate_states
    stage1.py       # benefício, normalização comum/por-estado, Wald/Savage/Hurwicz
    judgments.py    # Saaty: from_intensities, consistência (CR), perturb, W_TT placeholder
    anp.py          # W_AT, limite fechado (solve), limite por potências (verificação), síntese
    pipeline.py      # run(cfg, designs, model) -> Result
    baseline.py     # AHP de pesos iguais (aproximação, 5 objetivos, estado nominal)
    sensitivity.py  # 4 varreduras (alpha, pi, ANP/gamma, lambda) + métricas de estabilidade
  run_pipeline.py   # CLI
  tests/            # pytest (69 testes + 1 xfail)
```

Sem estado global mutável; qualquer variação de parâmetro cria uma nova
`Config` via `dataclasses.replace`. Semente aleatória sempre explícita
(`--seed`, `synthetic_designs(stage, seed)`).

## Testes

```bash
cd tg_robust
python -m pytest -q
```

Cobertura: identidades de propagação, normalização comum vs. por-estado,
Wald/Savage/Hurwicz (casos-limite e invariância), consistência e
perturbação de Saaty, limite fechado vs. por potências (`atol=1e-9`),
síntese por mistura, reprodutibilidade, e o CLI ponta a ponta com $n=632$
e $n=1089$ designs sintéticos (ambos em menos de 2s).

## Decisões em aberto (dependem do autor e/ou do orientador)

Ver a lista completa na mensagem final da sessão em que este pacote foi
criado. Resumo:

1. **Dados reais** — os CSVs de `../data/` parecem ser dados de
   desenvolvimento do protótipo antigo (violam `eta>=0,8` massivamente), não
   o *front* de Pareto validado do artigo. Os dados reais de
   Nascimento et al. (2022) ainda precisam ser obtidos/confirmados.
2. **Scripts de balística interna** do orientador — opcional; na ausência,
   usa-se `SurrogateBallistics` (identidades analíticas).
3. **Julgamentos de `W_TT`** (`judgments.PLACEHOLDER_INFLUENCIAS`) — rascunho
   provisório, a validar com o orientador (acoplamentos físicos reais entre
   os 7 critérios).
4. **$\alpha$ de Hurwicz** (referência 0,8) — confirmar a postura de risco do
   perfil *New Space* adotada.
5. **Margens de $c^{*}$ e $C_F$** — valores de $\kappa$ (Tabela "cenarios")
   vêm diretamente do texto do artigo reportado no `.tex`; confirmar se são
   as margens finais a usar.
