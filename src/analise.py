"""Pipeline completo da Fase 1: dados -> sequências -> atributos -> clustering
-> experimento de degradação -> transferência entre jogos. Gera CSVs e figuras em ../saida/"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from dados import carregar_jogo
from sequencias import segmentar, tabela_atributos, FEATURES, FEATURES_CLUSTER, RESULTADOS
from degradacao import degradar, CENARIOS
import figuras

RAIZ = Path(__file__).resolve().parents[1]
DADOS = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ.parent / "data/sample-data/data"
SAIDA = RAIZ / "saida"
SAIDA.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]
K_FIXO = 4  # escolhido pela silhueta + interpretabilidade (ver k_escolha.csv)


def main():
    jogos = [carregar_jogo(DADOS / g) for g in ["Sample_Game_1", "Sample_Game_2"]]
    seqs = {j["nome"]: segmentar(j["eventos"]) for j in jogos}

    # ---------- atributos de referência ----------
    ref = pd.concat([tabela_atributos(j, seqs[j["nome"]]) for j in jogos], ignore_index=True)
    ref.to_csv(SAIDA / "sequencias_referencia.csv", index=False)

    # ---------- clustering ----------
    FC = FEATURES_CLUSTER
    X = ref[FC].values
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    escolha = []
    for k in range(2, 9):
        km = KMeans(k, n_init=50, random_state=0).fit(Z)
        escolha.append(dict(k=k, silhouette=silhouette_score(Z, km.labels_),
                            davies_bouldin=davies_bouldin_score(Z, km.labels_)))
    pd.DataFrame(escolha).to_csv(SAIDA / "k_escolha.csv", index=False)

    km = KMeans(K_FIXO, n_init=50, random_state=0).fit(Z)
    # estabilidade por bootstrap: reajusta em reamostras e compara com o modelo base
    rng = np.random.default_rng(0)
    estab = []
    for b in range(50):
        idx = rng.choice(len(Z), len(Z))
        estab.append(adjusted_rand_score(km.labels_, KMeans(K_FIXO, n_init=10, random_state=b).fit(Z[idx]).predict(Z)))
    ref["cluster"] = km.labels_
    perfil_km = pd.DataFrame(km.cluster_centers_, columns=FC)  # z-scores
    mapa = figuras.nomear_clusters(perfil_km)
    ref["padrao"] = ref.cluster.map(mapa)
    ref["nome_padrao"] = ref.padrao.map(dict(enumerate(figuras.NOMES)))
    ref["dist_centroide"] = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    ref.to_csv(SAIDA / "sequencias_referencia.csv", index=False)

    perfil = perfil_km.rename(index=mapa).sort_index()
    perfil.index = figuras.NOMES
    medias = ref.groupby("nome_padrao")[FEATURES].mean()
    result = ref.groupby("nome_padrao")[RESULTADOS].mean()
    tamanho = ref.groupby("nome_padrao").size().rename("n")
    pd.concat([tamanho, medias, result], axis=1).loc[figuras.NOMES].to_csv(SAIDA / "perfil_padroes.csv")
    perfil.to_csv(SAIDA / "perfil_padroes_z.csv")
    pd.crosstab([ref.jogo, ref.equipe], ref.nome_padrao, normalize="index").to_csv(SAIDA / "padroes_por_equipe.csv")
    perfil = perfil.reset_index(drop=True)

    # ---------- transferência: modelo do jogo 1 aplicado ao jogo 2 ----------
    m1 = ref.jogo == "Sample_Game_1"
    # treina só no jogo 1 e aplica no jogo 2 (outras equipes), comparando com
    # o agrupamento feito diretamente no jogo 2
    sc1 = StandardScaler().fit(ref.loc[m1, FC])
    km1 = KMeans(K_FIXO, n_init=50, random_state=0).fit(sc1.transform(ref.loc[m1, FC]))
    sc2 = StandardScaler().fit(ref.loc[~m1, FC])
    km2 = KMeans(K_FIXO, n_init=50, random_state=0).fit(sc2.transform(ref.loc[~m1, FC]))
    ari_transf = adjusted_rand_score(km2.labels_, km1.predict(sc1.transform(ref.loc[~m1, FC])))

    # ---------- experimento de degradação ----------
    linhas_feat, linhas_clu = [], []
    exemplos = {}
    for nome, params in CENARIOS:
        seeds = [0] if not params or set(params) == {"janela"} else SEEDS
        for sd in seeds:
            partes = []
            for j in jogos:
                deg = degradar(j, seed=sd, **params)
                if nome.startswith("Realista") and sd == 0:
                    exemplos[j["nome"]] = deg
                partes.append(tabela_atributos(j, seqs[j["nome"]], ataque=deg, defesa=deg))
            dg = pd.concat(partes, ignore_index=True)
            Xd = dg[FEATURES].copy()
            Xd = Xd.fillna(Xd.median())
            cobertura = float(np.mean([np.mean(~np.isnan(degradar(j, seed=sd, **params)[e][..., 0]))
                                       / np.mean(~np.isnan(j[e][..., 0])) for j in jogos for e in ("Home", "Away")]))
            for f in FEATURES:
                a, b = ref[f].values, Xd[f].values
                linhas_feat.append(dict(cenario=nome, seed=sd, feature=f,
                                        erro_abs_mediano=float(np.median(np.abs(b - a))),
                                        spearman=float(spearmanr(a, b).correlation)))
            lab_fixo = km.predict(scaler.transform(Xd[FC].values))
            km_d = KMeans(K_FIXO, n_init=50, random_state=0).fit(StandardScaler().fit_transform(Xd[FC].values))
            linhas_clu.append(dict(cenario=nome, seed=sd, cobertura=cobertura,
                                   ari_modelo_ref=adjusted_rand_score(ref.cluster, lab_fixo),
                                   ari_reajustado=adjusted_rand_score(ref.cluster, km_d.labels_),
                                   concordancia=float((ref.cluster.values == lab_fixo).mean())))
            print(nome, sd, linhas_clu[-1])
    pd.DataFrame(linhas_feat).to_csv(SAIDA / "degradacao_features.csv", index=False)
    pd.DataFrame(linhas_clu).to_csv(SAIDA / "degradacao_clusters.csv", index=False)

    resumo = dict(n_sequencias=len(ref), por_jogo=ref.jogo.value_counts().to_dict(),
                  k=K_FIXO, silhouette=float(silhouette_score(Z, km.labels_)),
                  davies_bouldin=float(davies_bouldin_score(Z, km.labels_)),
                  estabilidade_bootstrap_ari_media=float(np.mean(estab)), estabilidade_bootstrap_ari_p10=float(np.percentile(estab, 10)),
                  features_cluster=FC,
                  ari_transferencia_jogo1_para_jogo2=float(ari_transf))
    json.dump(resumo, open(SAIDA / "resumo.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    figuras.gerar(jogos, seqs, ref, perfil, SAIDA, exemplos)


if __name__ == "__main__":
    main()
