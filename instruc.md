autoracer-ai/
│
├── main.py           # Ponto de entrada principal
├── requirements.txt  # Dependências do projeto
├── README.md         # Documentação
│
├── models/           # Modelos de IA salvos
│
├── src/
│   ├── game/
│   │   ├── car.py         # Classe do veículo
│   │   ├── track.py       # Gerador e gerenciador de pistas
│   │   ├── physics.py     # Motor de física simplificado
│   │   └── renderer.py    # Renderização com Pygame
│   │
│   ├── ai/
│   │   ├── agent.py       # Agente de aprendizado por reforço
│   │   ├── network.py     # Implementação da rede neural
│   │   ├── memory.py      # Memória de experiências para replay
│   │   └── trainer.py     # Algoritmos de treinamento
│   │
│   └── utils/
│       ├── config.py      # Configurações do jogo
│       ├── visualize.py   # Ferramentas de visualização
│       └── helpers.py     # Funções auxiliares
│
└── tracks/           # Designs de circuitos
    ├── easy.json
    ├── medium.json
    └── hard.json