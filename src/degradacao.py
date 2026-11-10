"""Simulação dos erros típicos de um pipeline de visão computacional.

A ideia (Fase 1 do projeto): em vez de esperar o detector ficar pronto, aplicamos
ao tracking de referência os tipos de erro que a visão computacional comete e
medimos o quanto a análise ofensiva se degrada. Quando o pipeline de vídeo existir,
o mesmo código mede o erro real.

Tipos de erro:
  - ruido:  erro de posição (detecção + homografia), gaussiano com correlação temporal
  - perda:  falhas de detecção em blocos de 1 s (oclusões, jogador encoberto)
  - camera: janela de câmera de transmissão — só aparece quem está a até W/2 metros
            (no eixo do comprimento) do centro da imagem, que segue a bola
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from dados import FPS


def ruido(xy, sigma, rng):
    if sigma <= 0:
        return xy.copy()
    n = rng.normal(0, 1, xy.shape)
    # correlação temporal de ~0.4 s: erro de homografia/detecção não é independente entre frames
    n = uniform_filter1d(n, size=10, axis=0)
    n *= sigma / n.std()
    return xy + n


def perda(xy, p, rng, bloco=FPS):
    if p <= 0:
        return xy.copy()
    out = xy.copy()
    nf, nj, _ = xy.shape
    nb = int(np.ceil(nf / bloco))
    mask = rng.random((nb, nj)) < p
    mask = np.repeat(mask, bloco, axis=0)[:nf]
    out[mask] = np.nan
    return out


def camera(xy, bola, largura_janela):
    if largura_janela is None:
        return xy.copy()
    centro = uniform_filter1d(bola[:, 0], size=2 * FPS)  # câmera segue a bola com suavização
    centro = np.clip(centro, largura_janela / 2, 105 - largura_janela / 2)
    out = xy.copy()
    fora = np.abs(xy[..., 0] - centro[:, None]) > largura_janela / 2
    out[fora] = np.nan
    return out


def degradar(jogo, sigma=0.0, p_perda=0.0, janela=None, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for eq in ["Home", "Away"]:
        xy = jogo[eq]
        xy = camera(xy, jogo["bola"], janela)
        xy = perda(xy, p_perda, rng)
        xy = ruido(xy, sigma, rng)
        res[eq] = xy
    # bola: ruído + falhas de detecção (blocos de 1 s), lacunas preenchidas por
    # interpolação linear — como faria o pipeline real. A câmera segue a bola, então não a corta.
    b = perda(jogo["bola"][:, None, :], p_perda, rng)[:, 0, :]
    b = pd.DataFrame(b).interpolate(limit_direction="both").values
    res["bola"] = ruido(b, sigma, rng)
    return res


CENARIOS = (
    [("Referência", dict())]
    + [(f"Ruído σ={s} m", dict(sigma=s)) for s in (0.5, 1, 2, 4)]
    + [(f"Perda {int(p*100)}%", dict(p_perda=p)) for p in (0.1, 0.25, 0.5)]
    + [(f"Câmera {w} m", dict(janela=w)) for w in (70, 50, 35)]
    + [("Realista (σ=1 m, perda 15%, câmera 50 m)", dict(sigma=1, p_perda=0.15, janela=50))]
)
