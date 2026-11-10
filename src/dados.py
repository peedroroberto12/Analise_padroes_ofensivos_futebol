"""Carregamento e padronização dos dados de tracking e eventos (Metrica Sports).

Saída padronizada (mesmo formato que o módulo de visão computacional deve produzir):
  - posições em METROS, campo 105 x 68, origem no canto inferior esquerdo
  - arrays (n_frames, n_jogadores, 2) por equipe, NaN = jogador não visível
  - bola (n_frames, 2)
"""
from pathlib import Path
import numpy as np
import pandas as pd

COMPRIMENTO, LARGURA = 105.0, 68.0
FPS = 25


def _ler_tracking(caminho: Path):
    cab = pd.read_csv(caminho, nrows=3, header=None)
    nomes = cab.iloc[2].tolist()
    df = pd.read_csv(caminho, skiprows=3, header=None)
    cols, jogadores = ["Period", "Frame", "Time"], []
    i = 3
    while i < len(nomes):
        nome = nomes[i]
        cols += [f"{nome}_x", f"{nome}_y"]
        if nome != "Ball":
            jogadores.append(nome)
        i += 2
    df.columns = cols[: df.shape[1]]
    return df, jogadores


def carregar_jogo(pasta_jogo: Path):
    pasta_jogo = Path(pasta_jogo)
    nome = pasta_jogo.name
    home, jog_h = _ler_tracking(pasta_jogo / f"{nome}_RawTrackingData_Home_Team.csv")
    away, jog_a = _ler_tracking(pasta_jogo / f"{nome}_RawTrackingData_Away_Team.csv")
    eventos = pd.read_csv(pasta_jogo / f"{nome}_RawEventsData.csv")

    def para_metros(df, jogadores):
        xy = np.stack([np.stack([df[f"{j}_x"].values, df[f"{j}_y"].values], -1) for j in jogadores], 1)
        xy = xy.astype(float)
        xy[..., 0] *= COMPRIMENTO
        xy[..., 1] = (1 - xy[..., 1]) * LARGURA  # Metrica: y=0 no topo
        return xy

    bola = np.stack([home["Ball_x"].values, home["Ball_y"].values], -1).astype(float)
    bola[:, 0] *= COMPRIMENTO
    bola[:, 1] = (1 - bola[:, 1]) * LARGURA
    bola = pd.DataFrame(bola).interpolate(limit_direction="both").values

    for c in ["Start X", "End X"]:
        eventos[c] = eventos[c] * COMPRIMENTO
    for c in ["Start Y", "End Y"]:
        eventos[c] = (1 - eventos[c]) * LARGURA

    jogo = {
        "nome": nome,
        "periodo": home["Period"].values,
        "tempo": home["Time"].values,
        "Home": para_metros(home, jog_h),
        "Away": para_metros(away, jog_a),
        "jogadores": {"Home": jog_h, "Away": jog_a},
        "bola": bola,
        "eventos": eventos,
    }
    jogo["direcao"] = direcao_ataque(jogo)
    return jogo


def direcao_ataque(jogo):
    """+1 se a equipe ataca para x crescente naquele período, -1 caso contrário.
    Usa a posição média do goleiro (primeiro jogador listado) no início do período."""
    dirs = {}
    for equipe in ["Home", "Away"]:
        for p in np.unique(jogo["periodo"]):
            idx = np.where(jogo["periodo"] == p)[0][:250]
            gk_x = np.nanmean(jogo[equipe][idx, 0, 0])
            dirs[(equipe, int(p))] = 1 if gk_x < COMPRIMENTO / 2 else -1
    return dirs


def normalizar(xy, direcao):
    """Espelha coordenadas para que a equipe atacante sempre ataque para +x."""
    if direcao == 1:
        return xy
    out = xy.copy()
    out[..., 0] = COMPRIMENTO - out[..., 0]
    out[..., 1] = LARGURA - out[..., 1]
    return out
