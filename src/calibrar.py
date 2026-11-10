"""Ferramenta para marcar pontos do campo no primeiro frame do vídeo (uso local, com tela).

  python calibrar.py VIDEO.mp4 pontos.json

Clique em um ponto conhecido do campo (ex.: escanteio, cruzamento da linha do meio
com a lateral, cantos da grande área); no terminal, digite as coordenadas reais em
metros "x y" (origem no canto inferior esquerdo, campo 105 x 68). Mínimo 4 pontos.
Tecla 'q' encerra e salva.

Referências úteis (metros): escanteios (0,0) (0,68) (105,0) (105,68);
meio-campo nas laterais (52.5,0) (52.5,68); centro (52.5,34);
cantos da grande área esquerda (16.5,13.84) (16.5,54.16) (0,13.84) (0,54.16).
"""
import json
import sys

import cv2

video, saida = sys.argv[1], sys.argv[2]
ok, frame = cv2.VideoCapture(video).read()
assert ok, "não consegui ler o vídeo"
pix, campo = [], []


def clique(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("calibrar", frame)
        xy = input(f"pixel ({x},{y}) -> coordenada do campo 'x y' em metros: ").split()
        pix.append([x, y]); campo.append([float(xy[0]), float(xy[1])])


cv2.imshow("calibrar", frame)
cv2.setMouseCallback("calibrar", clique)
while cv2.waitKey(50) & 0xFF != ord("q"):
    pass
json.dump({"pixel": pix, "campo": campo}, open(saida, "w"), indent=2)
print(f"{len(pix)} pontos salvos em {saida}")
