"""Figuras do MVP (matplotlib, PNG). Paleta categórica validada (slots 1-4)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Arc, Circle

from dados import COMPRIMENTO as L, LARGURA as W, normalizar
from sequencias import FEATURES, FEATURES_CLUSTER

SUP, TXT, TXT2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NOMES = ["Transição rápida", "Progressão apoiada", "Construção longa", "Circulação no campo ofensivo"]
ROTULOS = {
    "duracao_s": "Duração (s)", "n_passes": "Nº de passes", "progressao_m": "Progressão (m)",
    "velocidade_prog_ms": "Velocidade de progressão (m/s)", "x_max_bola": "Ponto mais avançado (m)",
    "lateralidade_m": "Lateralidade da bola (m)", "pct_corredor_central": "% no corredor central",
    "trocas_corredor": "Trocas de corredor", "largura_m": "Largura da equipe (m)",
    "profundidade_m": "Profundidade da equipe (m)", "area_m2": "Área ocupada (m²)",
    "altura_bloco_m": "Altura do bloco (m)", "superioridade_local": "Superioridade local",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": TXT2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TXT, "figure.facecolor": SUP,
    "axes.facecolor": SUP, "savefig.facecolor": SUP, "axes.spines.top": False, "axes.spines.right": False,
})


def nomear_clusters(perfil_z):
    """Mapeia id do KMeans -> índice fixo de padrão (0..3), por regras sobre os centróides."""
    p = perfil_z.copy()
    m = {}
    m[p["velocidade_prog_ms"].idxmax()] = 0
    resto = p.drop(index=list(m))
    m[resto["duracao_s"].idxmax()] = 2
    resto = p.drop(index=list(m))
    m[resto["progressao_m"].idxmin()] = 3
    for i in p.index:
        m.setdefault(i, 1)
    return m


def campo(ax, cor_linha=BASE):
    ax.set_xlim(-3, L + 3); ax.set_ylim(-3, W + 3); ax.set_aspect("equal"); ax.axis("off")
    kw = dict(fill=False, ec=cor_linha, lw=1)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color=cor_linha, lw=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))
    for x0, s in [(0, 1), (L, -1)]:
        ax.add_patch(Rectangle((x0 if s == 1 else x0 - 16.5, W / 2 - 20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((x0 if s == 1 else x0 - 5.5, W / 2 - 9.16), 5.5, 18.32, **kw))
    for x in (L / 3, 2 * L / 3):
        ax.plot([x, x], [0, W], color=GRID, lw=0.8, ls="--", zorder=0)


def fig_padroes(jogos, ref, saida):
    jg = {j["nome"]: j for j in jogos}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
    for c, ax in enumerate(axs.flat):
        campo(ax)
        sub = ref[ref.padrao == c]
        # 12 sequências mais próximas do centróide (mais representativas)
        sub = sub.nsmallest(12, "dist_centroide")
        for _, s in sub.iterrows():
            j = jg[s.jogo]
            d = j["direcao"][(s.equipe, int(s.periodo))]
            b = normalizar(j["bola"][int(s.ini) - 1:int(s.fim):5], d)
            ax.plot(b[:, 0], b[:, 1], color=CORES[c], lw=1.6, alpha=0.75, solid_capstyle="round")
            ax.plot(b[-1, 0], b[-1, 1], "o", ms=5, color=CORES[c], mec=SUP, mew=1.5)
        n = (ref.padrao == c).sum()
        ax.set_title(f"{NOMES[c]}  ·  {n} ataques ({n / len(ref):.0%})", loc="left", fontsize=11.5, color=TXT, pad=6)
        ax.annotate("", xy=(L * 0.62, -2.2), xytext=(L * 0.38, -2.2),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    fig.suptitle("Padrões ofensivos encontrados: trajetória da bola nos 12 ataques mais típicos de cada grupo",
                 x=0.02, ha="left", fontsize=13, color=TXT)
    fig.text(0.02, 0.005, "Ataque sempre normalizado da esquerda para a direita. Bolinha = onde o ataque terminou. "
             "Linhas tracejadas = divisão em terços.", fontsize=9, color=TXT2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(saida / "fig_padroes.png", dpi=150); plt.close(fig)


def fig_perfil(perfil_z, saida):
    cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#86b6ef", "#f0efec", "#ef8f8e", "#c43b3b"])
    P = perfil_z[FEATURES_CLUSTER].values
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(P, cmap=cmap, vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels(NOMES, color=TXT)
    ax.set_xticks(range(len(FEATURES_CLUSTER)))
    ax.set_xticklabels([ROTULOS[f] for f in FEATURES_CLUSTER], rotation=25, ha="right", color=TXT2)
    for i in range(P.shape[0]):
        for k in range(P.shape[1]):
            ax.text(k, i, f"{P[i, k]:+.1f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(P[i, k]) > 1.3 else TXT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("desvios-padrão em relação à média", color=TXT2); cb.outline.set_visible(False)
    ax.set_title("Perfil de cada padrão nos 8 atributos usados no agrupamento", loc="left", color=TXT, fontsize=12)
    fig.tight_layout(); fig.savefig(saida / "fig_perfil.png", dpi=150); plt.close(fig)


def fig_resultados(ref, saida):
    r = ref.groupby("padrao")[["entrou_area", "finalizacao"]].mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(4); w = 0.36
    for i, (col, lab) in enumerate([("entrou_area", "Chegou na área"), ("finalizacao", "Terminou em finalização")]):
        b = ax.bar(x + (i - 0.5) * (w + 0.02), r[col] * 100, w, color=CORES[i] if i == 0 else "#52514e",
                   label=lab, zorder=3)
        for rect, v in zip(b, r[col] * 100):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=TXT2)
    ax.set_xticks(x); ax.set_xticklabels(NOMES, color=TXT)
    ax.set_ylabel("% dos ataques do grupo"); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, max(60, r.values.max() * 100 + 10))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Resultado dos ataques por padrão (não usado no agrupamento)", loc="left", color=TXT, fontsize=12)
    fig.tight_layout(); fig.savefig(saida / "fig_resultados.png", dpi=150); plt.close(fig)


def fig_equipes(ref, saida):
    t = pd.crosstab([ref.jogo.str.replace("Sample_Game_", "Jogo "), ref.equipe.map({"Home": "Mandante", "Away": "Visitante"})],
                    ref.padrao, normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 3.2))
    labels = [f"{a} · {b}" for a, b in t.index]
    left = np.zeros(len(t))
    for c in range(4):
        v = t[c].values if c in t else np.zeros(len(t))
        ax.barh(labels, v, left=left, color=CORES[c], edgecolor=SUP, linewidth=2, label=NOMES[c], height=0.62)
        for yi, (l0, vv) in enumerate(zip(left, v)):
            if vv >= 8:
                ax.text(l0 + vv / 2, yi, f"{vv:.0f}%", ha="center", va="center", fontsize=9,
                        color=TXT if c == 3 else "#ffffff")
        left += v
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% dos ataques da equipe")
    ax.tick_params(axis="y", colors=TXT, length=0)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=9)
    ax.set_title("Mesmo modelo, quatro equipes: o 'DNA' ofensivo de cada uma", loc="left", color=TXT, fontsize=12)
    fig.tight_layout(); fig.savefig(saida / "fig_equipes.png", dpi=150); plt.close(fig)


def fig_degradacao(saida):
    d = pd.read_csv(saida / "degradacao_clusters.csv")
    g = d.groupby("cenario", sort=False)[["concordancia", "cobertura"]].agg(["mean", "std"])
    g = g.drop(index="Referência")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(g))
    m = g[("concordancia", "mean")].values * 100
    sd = g[("concordancia", "std")].fillna(0).values * 100
    ax.barh(y, m, color=CORES[0], height=0.6, zorder=3)
    ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=TXT2, lw=1, capsize=3, zorder=4)
    for yi, v, cob in zip(y, m, g[("cobertura", "mean")].values * 100):
        ax.text(v + 1.2, yi, f"{v:.0f}%", va="center", fontsize=9, color=TXT)
        ax.text(121, yi, f"{cob:.0f}%", va="center", ha="right", fontsize=9, color=TXT2)
    ax.set_yticks(y); ax.set_yticklabels(g.index, color=TXT); ax.invert_yaxis()
    ax.text(121, -0.75, "visível", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 122); ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("% dos ataques que continuam no mesmo padrão")
    ax.set_title("Quanto o erro de visão computacional muda a classificação dos ataques",
                 loc="left", color=TXT, fontsize=12, pad=16)
    fig.text(0.01, 0.01, "Média de 3 repetições (barra = desvio-padrão). 'Visível' = fração dos jogadores-frame que "
             "sobram após a degradação.", fontsize=8.5, color=TXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(saida / "fig_degradacao.png", dpi=150); plt.close(fig)


def fig_features_degradacao(saida):
    d = pd.read_csv(saida / "degradacao_features.csv")
    p = d.pivot_table(index="feature", columns="cenario", values="spearman", sort=False).drop(columns="Referência")
    p = p.loc[FEATURES]
    cols = [c.replace("Realista (σ=1 m, perda 15%, câmera 50 m)", "Realista") for c in p.columns]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(12, 5.6))
    im = ax.imshow(p.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", color=TXT2)
    ax.set_yticks(range(len(p))); ax.set_yticklabels([ROTULOS[f] for f in p.index], color=TXT)
    for i in range(p.shape[0]):
        for k in range(p.shape[1]):
            v = p.values[i, k]
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.75 else TXT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("correlação de Spearman com a referência\n(valores < 0 aparecem no tom mais claro)", color=TXT2); cb.outline.set_visible(False)
    ax.set_title("Quais atributos sobrevivem ao erro de visão computacional (1 = idêntico à referência)",
                 loc="left", color=TXT, fontsize=12)
    fig.tight_layout(); fig.savefig(saida / "fig_features_degradacao.png", dpi=150); plt.close(fig)


def fig_frame(jogos, exemplos, saida):
    j = jogos[0]
    deg = exemplos[j["nome"]]
    f = 30000  # frame de exemplo
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, titulo, dados, bola in [(axs[0], "Referência (tracking profissional Metrica)", {e: j[e] for e in ("Home", "Away")}, j["bola"]),
                                    (axs[1], "Simulação de vídeo: câmera 50 m + perdas + ruído", deg, deg["bola"])]:
        campo(ax)
        if dados is deg:
            from scipy.ndimage import uniform_filter1d
            c = np.clip(uniform_filter1d(j["bola"][:, 0], 50)[f], 25, L - 25)
            ax.add_patch(Rectangle((c - 25, -1), 50, W + 2, color=CORES[0], alpha=0.08, lw=0))
            ax.text(c, W + 1.8, "área enquadrada pela câmera", ha="center", fontsize=8.5, color=TXT2)
        for e, cor in (("Home", CORES[0]), ("Away", CORES[1])):
            xy = dados[e][f]
            ok = ~np.isnan(xy).any(1)
            ax.scatter(xy[ok, 0], xy[ok, 1], s=70, color=cor, edgecolor=SUP, linewidth=1.5, zorder=3,
                       label="Mandante" if e == "Home" else "Visitante")
            if dados is deg:
                ref = j[e][f]
                miss = ~ok & ~np.isnan(ref).any(1)
                ax.scatter(ref[miss, 0], ref[miss, 1], s=70, facecolor="none", edgecolor=cor, linewidth=1.2,
                           ls=(0, (2, 2)), zorder=2)
        ax.scatter(*bola[f], s=40, color=TXT, zorder=4, label="Bola")
        ax.set_title(titulo, loc="left", color=TXT, fontsize=11.5)
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=3, fontsize=9)
    axs[1].text(0, -6, "Círculo tracejado = jogador que existe mas o 'vídeo' não vê", fontsize=9, color=TXT2)
    fig.tight_layout(); fig.savefig(saida / "fig_frame_degradado.png", dpi=150); plt.close(fig)


def gerar(jogos, seqs, ref, perfil_z, saida, exemplos):
    fig_padroes(jogos, ref, saida)
    fig_perfil(perfil_z, saida)
    fig_resultados(ref, saida)
    fig_equipes(ref, saida)
    fig_degradacao(saida)
    fig_features_degradacao(saida)
    fig_frame(jogos, exemplos, saida)
