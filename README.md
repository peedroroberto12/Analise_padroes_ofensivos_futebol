# MVP — Padrões ofensivos no futebol a partir de dados posicionais

Protótipo da **Fase 1** do TCC (camada tática + experimento de robustez) e ponte para a
**Fase 2** (vídeo → posições).

## O que roda

```
src/
  dados.py        carrega tracking + eventos (Metrica Sports), converte para metros 105x68
  sequencias.py   definição operacional de ataque, segmentação e 13 atributos ofensivos
  degradacao.py   simula erros de visão computacional (ruído, perdas, janela de câmera)
  analise.py      pipeline completo: atributos -> clustering -> degradação -> transferência
  figuras.py      gráficos do relatório
  visao.py        Fase 2: vídeo -> YOLO + ByteTrack -> equipes por cor -> homografia -> CSV em metros
  calibrar.py     ferramenta de clique para marcar pontos do campo (homografia)
saida/            CSVs e figuras gerados
```

## Como rodar

```bash
pip install -r requirements.txt               # só a Fase 1 (leve)
git clone --depth 1 https://github.com/metrica-sports/sample-data.git
python src/analise.py sample-data/data        # ~2 min; gera tudo em saida/
```

Para a Fase 2 (vídeo), instale também: `pip install -r requirements-visao.txt` (baixa o PyTorch, ~1-2 GB).

Fase 2 (vídeo próprio, câmera fixa/elevada de preferência):

```bash
python src/calibrar.py jogo.mp4 pontos.json   # clique >= 4 pontos conhecidos do campo
python src/visao.py jogo.mp4 --homografia pontos.json --saida posicoes.csv --overlay debug.mp4
```

`visao.para_arrays()` converte o CSV para o mesmo formato que `sequencias.atributos_sequencia()`
consome — ou seja, a camada tática é a mesma para o tracking de referência e para o tracking do nosso pipeline de vídeo.

## Decisões de projeto

- **Dados:** Metrica Sports sample data (2 jogos reais anonimizados, tracking a 25 fps + eventos sincronizados). O tracking é gerado pela Metrica a partir de vídeo (visão computacional comercial) e é usado aqui como referência, não como verdade absoluta.
- **Ataque:** posse contínua de uma equipe segundo os eventos; termina em perda, bola fora,
  finalização ou interrupção. Filtro: >= 2 passes e >= 4 s → 288 sequências.
- **Clustering:** KMeans k=4 sobre 8 atributos não redundantes (|r| <= 0.7), padronizados.
- **Degradação:** aplicada a jogadores e bola; a segmentação continua pelos eventos (como no MVP
  descrito no levantamento, seção 5.2).

## Limitações conhecidas

- Segmentação dos ataques usa eventos anotados, não detecção automática de posse.
- Erros simulados são aproximações; o erro real do pipeline de vídeo deve ser medido com
  SoccerTrack [R27] ou Pettersen et al. [R28].
- `visao.py` foi testado apenas em clipe sintético; separação por cor é heurística (goleiros e
  árbitro podem cair em "Outro").
- Calibração única serve para câmera fixa; câmera de TV exige calibração por frame (TVCalib).
