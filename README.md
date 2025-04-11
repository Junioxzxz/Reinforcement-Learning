# Reinforcement-Learning


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



    # 🏎️ AutoRacer AI

Um jogo de corrida com perspectiva superior onde uma inteligência artificial aprende a dirigir através de aprendizado por reforço.

![AutoRacer AI](https://via.placeholder.com/800x400?text=AutoRacer+AI)

## 📝 Descrição

AutoRacer AI é um simulador de corrida onde veículos controlados por redes neurais aprendem a navegar por circuitos cada vez mais desafiadores. O projeto utiliza aprendizado por reforço para treinar agentes que melhoram progressivamente suas habilidades até conseguirem completar voltas perfeitas sem colisões.

### Características

- 🏁 Múltiplos circuitos com diferentes níveis de dificuldade
- 🧠 Redes neurais recorrentes (LSTM) para aprendizado sequencial
- 📊 Visualização em tempo real do processo de aprendizado
- 🎮 Modo manual para competir contra a IA
- 📈 Gráficos de desempenho e evolução das gerações
- 🏆 Sistema de salvamento dos melhores modelos

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Dependências

```
pygame==2.1.2
numpy==1.23.0
tensorflow==2.9.0
matplotlib==3.5.2
gymnasium==0.28.1
```

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/autoracer-ai.git
   cd autoracer-ai
   ```

2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   
   # No Windows
   venv\Scripts\activate
   
   # No macOS/Linux
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Como Usar

### Executar o jogo

```bash
python main.py
```

### Modos de jogo

- **Modo Aprendizado**: A IA treina continuamente através de gerações
  ```bash
  python main.py --mode train
  ```

- **Modo Demonstração**: Carrega o modelo mais bem treinado e demonstra seu desempenho
  ```bash
  python main.py --mode demo
  ```

- **Modo Jogador**: Permite jogar manualmente usando as setas do teclado
  ```bash
  python main.py --mode player
  ```

## 🔧 Arquitetura do Projeto

### Estrutura de arquivos

```
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
```

## 🧠 Aprendizado por Reforço

O projeto implementa um algoritmo de Deep Q-Learning com as seguintes características:

- **Arquitetura da rede**: LSTM + Camadas densas
- **Função de recompensa**: Baseada em distância percorrida, velocidade e penalidades por colisão
- **Experience Replay**: Armazena experiências passadas para retreinamento
- **Target Network**: Estabiliza o aprendizado através de atualizações periódicas
- **Exploration vs Exploitation**: Estratégia epsilon-greedy para balancear exploração e aproveitamento

## 🛣️ Pistas e Desafios

- **Circuito Básico**: Oval simples para treinamento inicial
- **Circuito Intermediário**: Curvas variadas e chicanes
- **Circuito Avançado**: Curvas fechadas, estrangulamentos e obstáculos dinâmicos

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

1. Faça um fork do projeto
2. Crie sua branch de funcionalidade (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📜 Licença

Este projeto está licenciado sob a MIT License.

## 🔗 Links Úteis

- [Documentação do Pygame](https://www.pygame.org/docs/)
- [Documentação do TensorFlow](https://www.tensorflow.org/api_docs)
- [Tutorial sobre Q-Learning](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Curso de Aprendizado por Reforço](https://www.coursera.org/learn/reinforcement-learning-in-python)

---

Desenvolvido com ❤️ por [Junio]
