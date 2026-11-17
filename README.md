# tg_robust — Seleção Robusta Pós-Pareto de Motores Sólidos

Implementação computacional da metodologia do Trabalho de Graduação (TG) de
Isabel Colares Brasil: seleção pós-Pareto de motores-foguete a propelente
sólido combinando critérios de decisão sob incerteza estrita
(**Wald**, **Savage**, **Hurwicz** — Estágio 1) com o **Processo Analítico em
Rede (ANP)** (Estágio 2) e restrições de admissibilidade por perfil
(fator de segurança e teto de custo), aplicada aos designs Pareto de
Nascimento et al. (2022).

Critérios (padrão): `Mp`, `tb`, `It`, `F`, `rhoIsp`, `eta`. `pcmax` não é critério:
atua só no filtro de FS e no índice de custo. Os 7 atributos (com `pcmax`) continuam
lidos e propagados pelos estados; `--criterios 7` volta a pontuar `pcmax` e
`--criterios 5` usa os critérios do artigo (sem `pcmax` nem `eta`).

## Instalação

Python 3.11 ou superior.

```bash
git clone <url-do-repositorio> tg_robust
cd tg_robust
pip install -r requirements.txt
```

Dependências: `numpy`, `scipy`, `pandas` (execução), `matplotlib` (figuras) e
`pytest` (testes).

## Uso rápido

```bash
# dados sintéticos (não são dados do artigo; servem para testar a instalação)
python run_pipeline.py --stage 1 --out resultados_estagio1 --seed 0
python run_pipeline.py --stage 2 --out resultados_estagio2 --seed 0

# dados do artigo (1º estágio, tabela do modeFRONTIER — ver "Dados de entrada")
# e figuras do Cap. 4 (um painel por figura; --dados2/--resultados2 acrescentam o 2º estágio):
python run_pipeline.py --stage 1 \
    --csv "data/modeFrontier 1st Stage Us Propellants Designs Table.csv" --out resultados_estagio1
python gerar_figuras_cap4.py \
    --dados1 "data/modeFrontier 1st Stage Us Propellants Designs Table.csv" \
    --resultados1 resultados_estagio1 --out figuras_cap4

# com um simulador de balística interna próprio (habilita a varredura de λ):
python run_pipeline.py --stage 2 --csv designs.csv --balistica simulador.py --out resultados_estagio2

# opções das restrições de admissibilidade:
#   --referencia-custo {mediana,minimo}   (padrão: mediana)
#   --b-cer B                             (padrão: 1.0)
#   --forma-residuo {quadrado,absoluto}   (benefício de Mp, tb, It, F: -δ² (padrão) ou -|δ|)
#   --sem-teto-custo                      (rho = inf em todos os perfis)
#   --criterios {5,6,7}                   (padrão: 6; a W_TT é restrita aos critérios)
#   --criterio-fs-alvo                    (acrescenta a aderência ao FS alvo do perfil)
```

Sem `--csv`, o pipeline usa **dados sintéticos** (`tg_robust/data.py:synthetic_designs`),
explicitamente marcados como tal (aviso no terminal, coluna `sintetico` e
`run_config.json:dados_sinteticos=true`). **Não são dados reais do artigo.**

### Saídas (`--out`)

| Arquivo | Conteúdo |
|---|---|
| `scores_W.csv`, `scores_S.csv`, `scores_H.csv` | Pontuações $v^{(p)}_{k,r}$ por critério e, para comparação, $V^{(p)}$ (pesos iguais), por perfil |
| `ranking_final.csv` | Top-50 por $g$, com $g, g_W, g_S, g_H, V_W, V_S, V_H$ e rank do baseline AHP |
| `consistency.csv` | CR por coluna de julgamento de $W_{TT}$ |
| `sens_alpha.csv`, `sens_pi.csv`, `sens_anp.csv`, `sens_lambda.csv`, `sens_restricoes.csv`, `sens_residuo.csv`, `sens_cenarios_fs.csv`, `sens_nominal.csv` | As 8 varreduras de sensibilidade (`sens_nominal.csv`: apenas o estado nominal θ1, com a normalização refeita sobre ele) (`sens_cenarios_fs.csv`: FS de referência, todos 1,25 e todos 1,40, com vencedor por perfil, top 10 e τ) (`sens_residuo.csv`: forma do resíduo, $-\delta^2$ ou $-|\delta|$) (`sens_lambda.csv` fica vazio se não houver modelo de balística que reexecute os estados) |
| `admissibilidade.csv` | Por design: `pcmax` em θ1 e máximo entre estados, base e índice de custo $c_k$, e as máscaras de pressão, custo e $\mathcal{F}^{(p)}$ |
| `admissibilidade_resumo.csv` | Tamanho de $\mathcal{F}^{(p)}$, exclusões só por pressão / só por custo / por ambos, e a interseção |
| `vencedores_por_perfil.csv` | Por perfil: `vencedor_anp` (argmax de $g^{(p)}$ em $\mathcal{F}^{(p)}$, resultado principal), `vencedor_pesos_iguais` (argmax de $V^{(p)}$ em $\mathcal{F}^{(p)}$, comparação), `divergem`, e as mesmas três colunas sem restrições; "nenhum" se $\mathcal{F}^{(p)}$ for vazio |
| `pesos_efetivos.csv` | $q_0$, $q = (1-\gamma)(I - \gamma W_{TT})^{-1} q_0$ e a razão $q/q_0$ |
| `run_config.json` | Configuração completa (perfis, parâmetros globais), semente, modelo de balística, correlação $c_k \times p_{c,\max}(\theta_1)$, perfis em que $d^*$ é inadmissível, hash SHA-256 do CSV |

## Restrições de admissibilidade por perfil (Seção 3.3.4)

Cada perfil é um `config.Perfil` (regra de decisão, α, FS, teto de custo ρ,
peso π); os parâmetros globais ficam em `config.ConfigGlobal` (p_lim = 16 MPa,
expoente `b` da CER = 1, referência do custo
`mediana`/`minimo`). O pipeline e a sensibilidade tratam qualquer lista de
perfis da mesma forma. As restrições atuam só no bloco $W_{AT}^{(p)}$ e na
escolha do vencedor de cada perfil; `rho = inf` retira o teto de custo.

Base do índice de custo: $p_{c,\max}(\theta_1)\,V_c$, com
$V_c = M_p/(\rho_p\,\eta_v)$, quando os dados trazem `rho` e `eta` (tabela do
modeFRONTIER); caso contrário, $p_{c,\max}(\theta_1)\,R_c^2 L_t$.

O **vencedor de cada perfil** é o argmax de $g^{(p)}$ (sub-rede ANP do perfil,
a mesma que entra na síntese de $d^*$) restrito a $\mathcal{F}^{(p)}$.
$V^{(p)}$ (média simples dos critérios, pesos iguais) é mantido só como
referência "sem interdependência", como no AHP original.

## Dados de entrada

Os designs Pareto de Nascimento et al. (2022) **não são distribuídos** neste
repositório. Quem tiver acesso à tabela de designs do modeFRONTIER pode
colocá-la em `data/` (pasta ignorada pelo git) e passá-la por `--csv`; o
formato é reconhecido automaticamente e os estados θ são obtidos por leis de
escala quase estacionárias (`scenarios.EscalaQuaseEstacionaria`).

Outros conjuntos de designs são aceitos se trouxerem, por design:

1. os 7 atributos por estado, em colunas `<criterio>_theta{1,2,3}`
   (`tb`, `It`, `F`, `rho_isp`, `pcmax`) mais `Mp` e `eta_v`, **ou** só os
   valores nominais e um modelo de balística (`--balistica`) para calcular os
   estados;
2. `Rc` e `Lt` (raio e comprimento do grão; a unidade se cancela no índice de
   custo) — sem eles, `load_designs` recusa o arquivo;
3. `propelente` (tipo 1-4) ou `nexp`.

`pc_med` (pressão média) **não** é aceito no lugar de `pcmax`.

> do Nascimento, L. G. M.; Araújo, L. M.; Fernandes, S. S.; Shimote, W. K.;
> Rapozo, R. R. *Hybrid optimization algorithm for preliminary design of
> multistage launch vehicles.* Journal of the Brazilian Society of Mechanical
> Sciences and Engineering, v. 44, p. 103, 2022.

## Modelo de balística interna

`tg_robust/scenarios.py` define a interface `BallisticModel`:

```python
class BallisticModel(Protocol):
    def __call__(self, designs: pd.DataFrame, kc: float, kcf: float) -> dict[str, np.ndarray]:
        ...
```

Implemente uma classe com essa assinatura (recalculando a curva de empuxo
por inteiro a partir de $c^{*}_j = \kappa_{c^{*},j} c^{*}$ e
$C_{F,j} = \kappa_{C_F,j} C_F$) e passe-a ao pipeline:

```python
resultado = pipeline.run(cfg, designs, model=ModeloRealDeBalistica())
```

Um simulador por design no formato `simular(x, k_cstar, k_cf) -> dict` é
adaptado por `scenarios.AdaptadorSimulador` (`--balistica arquivo.py` no CLI).
Se os estados vêm prontos no CSV e nenhum modelo é fornecido, usa-se
`scenarios.EstadosPrecomputados` e a varredura de λ é pulada com aviso. Sem
CSV por estado nem modelo, o pipeline usa `SurrogateBallistics`, que aplica as
identidades analíticas da Seção 3.3.1.

## Julgamentos de $W_{TT}$

`judgments.INFLUENCIAS` traz as relações validadas pelo orientador, sobre os 7
atributos. `build_WTT` usa só as relações entre os critérios da variante e
renormaliza cada coluna; coluna sem influenciador recebe auto-laço (no padrão,
`rhoIsp`, antes influenciado apenas por `pcmax`).

O critério opcional `aderFS` (`--criterio-fs-alvo`) vale
$g = -((FS_{ef} - FS_p)/FS_p)^2$, com $FS_{ef} = p_{lim}/p_{c,max}(\theta_1)$. Muda
de perfil para perfil, por isso fica fora do tensor comum: é a última coluna de
$v^{(p)}$, normalizada por perfil, com auto-laço em $W_{TT}$.

## Arquitetura

```
tg_robust/
  tg_robust/
    config.py        # Config (dataclass imutável) + constantes dos critérios/requisitos
    data.py          # load_designs, synthetic_designs, validação de restrições
    scenarios.py     # BallisticModel (interface) + modelos de estados + propagate_states
    stage1.py        # benefício, normalização comum/por-estado, Wald/Savage/Hurwicz
    judgments.py     # Saaty: from_intensities, consistência (CR), perturb, W_TT validada
    anp.py           # W_AT, limite fechado (solve), limite por potências (verificação), síntese
    pipeline.py      # run(cfg, designs, model) -> Result
    baseline.py      # AHP de pesos iguais (aproximação, 5 objetivos, estado nominal)
    admissibility.py # restrições por perfil (FS, teto de custo), F^(p) e resumo
    sensitivity.py   # varreduras de sensibilidade + estabilidade
  run_pipeline.py        # CLI
  gerar_figuras_cap4.py  # figuras do Cap. 4 a partir das saídas do CLI
  tests/                 # pytest
```

Sem estado global mutável; qualquer variação de parâmetro cria uma nova
`Config` via `dataclasses.replace`. Semente aleatória sempre explícita
(`--seed`, `synthetic_designs(stage, seed)`).

## Testes

```bash
python -m pytest -q
```

Cobertura: identidades de propagação, normalização comum vs. por-estado,
Wald/Savage/Hurwicz (casos-limite e invariância), consistência e
perturbação de Saaty, limite fechado vs. por potências (`atol=1e-9`),
síntese por mistura, restrições de admissibilidade, reprodutibilidade, e o
CLI ponta a ponta com designs sintéticos. Testes que dependem dos dados do
artigo ou de fronts de teste externos são pulados quando esses arquivos não
estão disponíveis.

`tests/test_regressao.py` compara, com FS = 1 e ρ = ∞, os resultados atuais com
os de uma versão anterior às restrições (igualdade exata). Só roda se a
variável `TG_ROBUST_ANTES` apontar para a raiz dessa versão.

## Licença

MIT — ver [LICENSE](LICENSE).
