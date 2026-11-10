"""Definição operacional de sequência ofensiva e extração de atributos.

Sequência ofensiva = intervalo contínuo em que uma equipe mantém a posse,
segundo os eventos. Começa no primeiro evento de posse da equipe e termina em:
perda (BALL LOST / posse do adversário), bola fora (BALL OUT), finalização (SHOT)
ou fim do período. Só entram sequências com >= 2 passes e >= 4 s.
"""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from dados import COMPRIMENTO, LARGURA, FPS, normalizar

EVENTOS_POSSE = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "CARRY"}
TERMINA = {"SHOT", "BALL OUT", "BALL LOST"}
MIN_PASSES, MIN_DURACAO = 2, 4.0


def segmentar(eventos: pd.DataFrame):
    seqs, atual = [], None

    def fechar(motivo):
        nonlocal atual
        if atual is not None:
            atual["fim_motivo"] = motivo
            seqs.append(atual)
        atual = None

    for _, ev in eventos.iterrows():
        tipo, equipe = ev["Type"], ev["Team"]
        if tipo not in EVENTOS_POSSE and tipo not in TERMINA:
            continue  # CHALLENGE, CARD, FAULT RECEIVED não mudam a posse por si só
        if atual is not None and (equipe != atual["equipe"] or ev["Period"] != atual["periodo"]):
            fechar("perda")
        if tipo == "SET PIECE" and atual is not None:
            fechar("interrupcao")
        if atual is None:
            if tipo in {"BALL OUT", "BALL LOST"}:
                continue
            atual = dict(equipe=equipe, periodo=int(ev["Period"]), ini=int(ev["Start Frame"]),
                         fim=int(max(ev["End Frame"], ev["Start Frame"])), n_passes=0,
                         finalizacao=False, gol=False, origem=ev["Subtype"] if tipo == "SET PIECE" else tipo)
        atual["fim"] = int(max(atual["fim"], ev["End Frame"], ev["Start Frame"]))
        if tipo == "PASS":
            atual["n_passes"] += 1
        if tipo == "SHOT":
            atual["finalizacao"] = True
            atual["gol"] = "GOAL" in str(ev["Subtype"])
            fechar("finalizacao")
        elif tipo in {"BALL OUT", "BALL LOST"}:
            fechar("bola_fora" if tipo == "BALL OUT" else "perda")
    fechar("fim")

    df = pd.DataFrame(seqs)
    df["duracao"] = (df["fim"] - df["ini"]) / FPS
    df = df[(df.n_passes >= MIN_PASSES) & (df.duracao >= MIN_DURACAO)].reset_index(drop=True)
    return df


def _corredor(y):
    # visão do atacante (ataca para +x): y alto = esquerda
    return np.where(y > 2 * LARGURA / 3, 0, np.where(y < LARGURA / 3, 2, 1))  # 0=esq,1=centro,2=dir


def atributos_sequencia(jogo, seq, ataque_xy=None, defesa_xy=None, bola=None, passo=5):
    """Calcula os atributos de UMA sequência. ataque_xy/defesa_xy permitem passar
    versões degradadas do tracking (experimento de robustez)."""
    eq = seq["equipe"]
    adv = "Away" if eq == "Home" else "Home"
    d = jogo["direcao"][(eq, seq["periodo"])]
    sl = slice(seq["ini"] - 1, seq["fim"], passo)  # Frame começa em 1; amostra a 5 fps

    A = normalizar((jogo[eq] if ataque_xy is None else ataque_xy)[sl], d)[:, 1:]   # sem goleiro
    D = normalizar((jogo[adv] if defesa_xy is None else defesa_xy)[sl], d)[:, 1:]
    B = normalizar((jogo["bola"] if bola is None else bola)[sl], d)
    if len(B) < 2:
        return None

    larg, prof, area, cx, sup = [], [], [], [], []
    for t in range(len(B)):
        a = A[t][~np.isnan(A[t]).any(1)]
        dd = D[t][~np.isnan(D[t]).any(1)]
        if len(a) >= 3:
            larg.append(a[:, 1].max() - a[:, 1].min())
            prof.append(a[:, 0].max() - a[:, 0].min())
            cx.append(a[:, 0].mean())
            try:
                area.append(ConvexHull(a).volume)
            except Exception:
                pass
        # superioridade local: atacantes - defensores num raio de 15 m da bola
        if len(a) and len(dd):
            na = (np.linalg.norm(a - B[t], axis=1) < 15).sum()
            nd = (np.linalg.norm(dd - B[t], axis=1) < 15).sum()
            sup.append(na - nd)

    corr = _corredor(B[:, 1])
    trocas = int((np.diff(corr) != 0).sum())
    x0, x1 = B[0, 0], B[-1, 0]
    dur = seq["duracao"]
    terco_final = bool((B[:, 0] > 2 * COMPRIMENTO / 3).any())
    area_pen = bool(((B[:, 0] > COMPRIMENTO - 16.5) & (np.abs(B[:, 1] - LARGURA / 2) < 20.16)).any())

    def m(v):
        return float(np.mean(v)) if len(v) else np.nan

    return {
        "duracao_s": dur,
        "n_passes": seq["n_passes"],
        "progressao_m": float(x1 - x0),
        "velocidade_prog_ms": float((x1 - x0) / dur),
        "x_max_bola": float(B[:, 0].max()),
        "lateralidade_m": float(np.mean(np.abs(B[:, 1] - LARGURA / 2))),
        "pct_corredor_central": float((corr == 1).mean()),
        "trocas_corredor": trocas,
        "largura_m": m(larg),
        "profundidade_m": m(prof),
        "area_m2": m(area),
        "altura_bloco_m": m(cx),
        "superioridade_local": m(sup),
        # resultado (NÃO entra no clustering, usado para interpretar os grupos)
        "entrou_terco_final": terco_final,
        "entrou_area": area_pen,
        "finalizacao": seq["finalizacao"],
        "gol": seq["gol"],
    }


FEATURES = ["duracao_s", "n_passes", "progressao_m", "velocidade_prog_ms", "x_max_bola",
            "lateralidade_m", "pct_corredor_central", "trocas_corredor", "largura_m",
            "profundidade_m", "area_m2", "altura_bloco_m", "superioridade_local"]
# Subconjunto usado no clustering: remove atributos redundantes (|r| > 0.7 com outro)
# e os que dependem de cobertura total do campo. Os demais seguem como descritivos.
FEATURES_CLUSTER = ["duracao_s", "n_passes", "progressao_m", "velocidade_prog_ms",
                    "lateralidade_m", "trocas_corredor", "largura_m", "altura_bloco_m"]
RESULTADOS = ["entrou_terco_final", "entrou_area", "finalizacao", "gol"]


def tabela_atributos(jogo, seqs, ataque=None, defesa=None):
    """ataque/defesa: dict equipe->array degradado (opcional)."""
    linhas = []
    for i, s in seqs.iterrows():
        eq = s["equipe"]
        adv = "Away" if eq == "Home" else "Home"
        f = atributos_sequencia(jogo, s,
                                None if ataque is None else ataque[eq],
                                None if defesa is None else defesa[adv],
                                None if ataque is None else ataque.get("bola"))
        if f is not None:
            f.update(jogo=jogo["nome"], equipe=eq, seq_id=i, periodo=s["periodo"],
                     ini=s["ini"], fim=s["fim"], origem=s["origem"], fim_motivo=s["fim_motivo"])
            linhas.append(f)
    return pd.DataFrame(linhas)
