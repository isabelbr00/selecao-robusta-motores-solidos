import warnings
from pathlib import Path

import pandas as pd
import pytest

from tg_robust import config, data


def test_synthetic_designs_colunas_e_flag():
    df = data.synthetic_designs(1, seed=0, n=30)
    for c in data.COLUNAS_CANONICAS:
        assert c in df.columns
    assert data.COLUNA_SINTETICO in df.columns
    assert df[data.COLUNA_SINTETICO].all()
    assert data.COLUNA_NEXP in df.columns


def test_synthetic_designs_dentro_das_margens():
    req = config.REQUISITOS_ESTAGIO[2]
    df = data.synthetic_designs(2, seed=1, n=200)
    assert (df["Mp"].between(req.Mp * 0.94, req.Mp * 1.06)).all()
    assert (df["tb"].between(req.tb * 0.94, req.tb * 1.06)).all()
    assert (df["pcmax"] < config.PCMAX_LIMITE_SUPERIOR).all()
    assert (df["eta"] >= config.ETA_LIMITE_INFERIOR).all()
    assert (df["eta"] <= 0.95).all()


def test_synthetic_designs_reprodutibilidade():
    df1 = data.synthetic_designs(1, seed=42, n=50)
    df2 = data.synthetic_designs(1, seed=42, n=50)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_designs_emite_aviso():
    with pytest.warns(UserWarning, match="SINTÉTICOS"):
        data.synthetic_designs(1, seed=0, n=10)


def test_resolver_nexp_por_propelente():
    df = pd.DataFrame({data.COLUNA_PROPELENTE: [1, 2, 3, 4]})
    n = data.resolver_nexp(df)
    assert list(n) == [0.31, 0.30, 0.35, 0.39]


def test_resolver_nexp_propelente_invalido():
    df = pd.DataFrame({data.COLUNA_PROPELENTE: [1, 5]})
    with pytest.raises(ValueError):
        data.resolver_nexp(df)


def test_resolver_nexp_padrao_emite_aviso():
    df = pd.DataFrame({"outra_coluna": [1, 2, 3]})
    with pytest.warns(UserWarning, match="padrão"):
        n = data.resolver_nexp(df)
    assert (n == config.N_PADRAO).all()


def test_load_designs_aplica_aliases_e_valida(tmp_path):
    df_bruto = pd.DataFrame(
        {
            "Mp": [100.0, 200.0],
            "tb": [10.0, 20.0],
            "It": [1000.0, 2000.0],
            "F_med": [100.0, 100.0],
            "rho_Isp": [500.0, 500.0],
            "pc_max": [10.0, 10.0],
            "eta_v": [0.85, 0.5],  # segunda linha viola eta >= 0.8
            "R_c": [0.5, 0.6],
            "L_t": [5.0, 6.0],
        }
    )
    caminho = tmp_path / "designs.csv"
    df_bruto.to_csv(caminho, index=False)

    with pytest.warns(UserWarning):
        df = data.load_designs(caminho)

    assert list(df["F"]) == [100.0, 100.0]
    assert list(df["rhoIsp"]) == [500.0, 500.0]
    assert list(df["pcmax"]) == [10.0, 10.0]
    assert list(df["Rc"]) == [0.5, 0.6]
    assert list(df["Lt"]) == [5.0, 6.0]
    assert not df[data.COLUNA_SINTETICO].any()


def _df_valido(**substituir) -> pd.DataFrame:
    base = {
        "Mp": [100.0],
        "tb": [10.0],
        "It": [1000.0],
        "F": [100.0],
        "rhoIsp": [500.0],
        "pcmax": [10.0],
        "eta": [0.85],
        "Rc": [0.5],
        "Lt": [5.0],
    }
    base.update(substituir)
    return pd.DataFrame(base)


def test_load_designs_colunas_faltantes_levanta_erro(tmp_path):
    caminho = tmp_path / "designs.csv"
    pd.DataFrame({"Mp": [1.0]}).to_csv(caminho, index=False)
    with pytest.raises(ValueError):
        data.load_designs(caminho)


def test_load_designs_sem_geometria_levanta_erro_sem_inventar(tmp_path):
    caminho = tmp_path / "designs.csv"
    _df_valido().drop(columns=["Rc", "Lt"]).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="Rc"):
        data.load_designs(caminho)


def test_load_designs_geometria_nao_positiva_levanta_erro(tmp_path):
    caminho = tmp_path / "designs.csv"
    _df_valido(Lt=[0.0]).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="positivos"):
        data.load_designs(caminho)


def test_load_designs_nao_aceita_pc_med_como_pcmax(tmp_path):
    # pc_med (pressão média, metodologia anterior) não substitui a pressão máxima
    caminho = tmp_path / "designs.csv"
    _df_valido().rename(columns={"pcmax": "pc_med"}).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="pcmax"):
        data.load_designs(caminho)


def test_synthetic_designs_geometria_coerente_com_Mp():
    df = data.synthetic_designs(1, seed=3, n=50)
    assert ((df["Rc"] > 0) & (df["Lt"] > 0)).all()


def test_load_designs_restricoes_levantar_erro(tmp_path):
    df_bruto = _df_valido(eta=[0.5])
    caminho = tmp_path / "designs.csv"
    df_bruto.to_csv(caminho, index=False)
    with pytest.raises(ValueError):
        data.load_designs(caminho, levantar_erro_restricoes=True)


def test_hash_arquivo_deterministico(tmp_path):
    caminho = tmp_path / "f.csv"
    caminho.write_text("a,b\n1,2\n")
    assert data.hash_arquivo(caminho) == data.hash_arquivo(caminho)


# ─── Tabela do modeFRONTIER (Nascimento et al., 2022) ────────────────────
CSV_REAL_1 = Path(__file__).resolve().parents[2] / "data" / "modeFrontier 1st Stage Us Propellants Designs Table.csv"


def _linha_modefrontier(Id, categoria="Pareto", E="false", Ea=10.0, **kw):
    linha = {"Id": Id, "M": "false", "Category": categoria, "E": E, "V": "false", "Ea": Ea, "Pc": 50.0,
             "propellant_type": 1.0, "Mp": 8000.0, "tb": 110.0, "ta": 118.0, "Total_Impulse_v": 2.3e7,
             "Thrust_med_v": 2.1e5, "rho_x_Isp": 5.6e6, "Pc_max": 1.2e7, "Pc_med": 5.0e6, "eta_v": 0.98,
             "rho": 1837.79, "cstar": 1588.0, "cf": 1.80}
    linha.update(kw)
    return linha


def test_load_modefrontier_filtra_deduplica_e_converte(tmp_path):
    linhas = [
        _linha_modefrontier(7, Ea=1.0),
        _linha_modefrontier(3, Ea=1.0, Mp=8100.0),  # duplicata de 7 pelas entradas; menor Id fica
        _linha_modefrontier(4, Ea=2.0, categoria="MOGA2"),
        _linha_modefrontier(5, Ea=3.0, E="true", Mp="#NAN"),
        _linha_modefrontier(6, Ea=4.0, propellant_type=2.0),
    ]
    caminho = tmp_path / "mf.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    assert data.e_formato_modefrontier(caminho)

    df = data.load_designs(caminho)  # delega ao leitor do modeFRONTIER
    assert list(df["design_id"]) == [3, 6]
    assert df.attrs["n_duplicatas_removidas"] == 1
    assert df.loc[0, "Mp"] == 8100.0
    assert df.loc[0, "pcmax"] == pytest.approx(12.0) and df.loc[0, "pcmed"] == pytest.approx(5.0)
    assert list(df["nexp"]) == [0.31, 0.30]
    assert data.e_fonte_modefrontier(df)
    for c in ("It", "F", "rhoIsp", "eta", "ta", "rho", "cstar", "cf"):
        assert c in df.columns


def test_load_modefrontier_recusa_valor_ausente(tmp_path):
    caminho = tmp_path / "mf.csv"
    pd.DataFrame([_linha_modefrontier(1, Pc_max="#NAN")]).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="ausentes"):
        data.load_modefrontier(caminho)


@pytest.mark.skipif(not CSV_REAL_1.exists(), reason="tabela real do 1º estágio não disponível")
def test_load_modefrontier_dados_reais_estagio_1():
    df = data.load_designs(CSV_REAL_1)
    assert len(df) == 437
    assert df.attrs["n_duplicatas_removidas"] == 632 - 437
    assert df["design_id"].is_unique
