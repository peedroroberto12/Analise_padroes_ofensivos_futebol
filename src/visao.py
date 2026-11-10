"""Fase 2 — vídeo -> posições em metros (mesmo formato da camada tática).

Etapas: detecção (YOLO) -> tracking (ByteTrack) -> separação por equipe (cor da camisa,
KMeans) -> homografia (pontos do campo) -> coordenadas métricas 105 x 68.

Uso:
  python visao.py VIDEO.mp4 --homografia pontos.json --saida posicoes.csv [--max-frames 750] [--overlay out.mp4]

pontos.json: pelo menos 4 correspondências pixel -> campo (metros), por exemplo
  {"pixel": [[x1,y1],[x2,y2],...], "campo": [[0,0],[52.5,0],...]}
Para câmera fixa (tática / drone), uma calibração vale para o vídeo inteiro.
Para câmera de TV, trocar esta etapa por TVCalib [R6] ou similar (calibração por frame).
Use calibrar.py para clicar os pontos no primeiro frame.
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cor_camisa(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    recorte = frame[max(y1 + int(0.15 * h), 0):y1 + int(0.5 * h), max(x1, 0):max(x2, 0)]
    if recorte.size == 0:
        return None
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    verde = (hsv[:, 0] > 35) & (hsv[:, 0] < 85) & (hsv[:, 1] > 60)  # remove gramado
    hsv = hsv[~verde] if (~verde).sum() > 10 else hsv
    return np.median(hsv, axis=0)


def rodar(video, homografia, saida, max_frames=None, overlay=None, modelo="yolov8n.pt"):
    from ultralytics import YOLO

    H = None
    if homografia:
        pts = json.load(open(homografia))
        H, _ = cv2.findHomography(np.float32(pts["pixel"]), np.float32(pts["campo"]))

    yolo = YOLO(modelo)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    linhas = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        r = yolo.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            for box, tid, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                c = cor_camisa(frame, box)
                pe = np.array([[(box[0] + box[2]) / 2, box[3]]], np.float32)  # ponto do pé
                xy = cv2.perspectiveTransform(pe[None], H)[0, 0] if H is not None else pe[0]
                linhas.append(dict(frame=f, tempo=f / fps, track_id=int(tid), conf=float(conf),
                                   px=float(pe[0, 0]), py=float(pe[0, 1]), x=float(xy[0]), y=float(xy[1]),
                                   h=float(c[0]) if c is not None else np.nan,
                                   s=float(c[1]) if c is not None else np.nan,
                                   v=float(c[2]) if c is not None else np.nan))
        if overlay:
            if writer is None:
                writer = cv2.VideoWriter(overlay, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
            writer.write(r.plot())
        f += 1
    cap.release()
    if writer:
        writer.release()

    df = pd.DataFrame(linhas)
    if df.empty:
        df.to_csv(saida, index=False)
        return df
    # separação por equipe: cor mediana por track, KMeans com 3 grupos (2 times + árbitro/outros);
    # os 2 maiores grupos viram as equipes
    cores = df.groupby("track_id")[["h", "s", "v"]].median().dropna()
    k = min(3, len(cores))
    km = KMeans(k, n_init=10, random_state=0).fit(cores.values)
    grupos = pd.Series(km.labels_, index=cores.index)
    ordem = grupos.value_counts().index.tolist()
    nome = {ordem[0]: "Home"}
    if len(ordem) > 1:
        nome[ordem[1]] = "Away"
    df["equipe"] = df.track_id.map(grupos).map(nome).fillna("Outro")
    df.to_csv(saida, index=False)
    print(f"{f} frames, {df.track_id.nunique()} tracks, equipes: {df.groupby('equipe').track_id.nunique().to_dict()}")
    return df


def para_arrays(df, n_frames):
    """Converte o CSV longo em arrays (n_frames, n_tracks, 2) por equipe — o formato
    que sequencias.atributos_sequencia() consome (ataque_xy / defesa_xy)."""
    out = {}
    for eq in ["Home", "Away"]:
        d = df[df.equipe == eq]
        ids = sorted(d.track_id.unique())
        arr = np.full((n_frames, len(ids), 2), np.nan)
        pos = {t: i for i, t in enumerate(ids)}
        arr[d.frame.values, d.track_id.map(pos).values] = d[["x", "y"]].values
        out[eq] = arr
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--homografia")
    ap.add_argument("--saida", default="posicoes.csv")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--overlay")
    ap.add_argument("--modelo", default="yolov8n.pt")
    a = ap.parse_args()
    rodar(a.video, a.homografia, a.saida, a.max_frames, a.overlay, a.modelo)
