import pygame
import numpy as np
import math
import random
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silenciar avisos do TensorFlow
warnings.filterwarnings('ignore')  # Silenciar outros avisos
import argparse
import matplotlib.pyplot as plt
from collections import deque
import time

# Configurações globais
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

# Verificar disponibilidade do TensorFlow
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow está disponível. Usando aprendizado por reforço completo.")
except ImportError:
    print("TensorFlow não está disponível. Usando algoritmo simplificado.")

# Inicialização do Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AutoRacer AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20)

def draw_text(text, x, y, color=WHITE):
    """Função auxiliar para desenhar texto na tela"""
    text_surface = font.render(str(text), True, color)
    screen.blit(text_surface, (x, y))

class Track:
    """Classe que representa a pista de corrida"""
    def __init__(self, difficulty='easy'):
        self.difficulty = difficulty
        self.checkpoints = []
        self.outer_boundary = []
        self.inner_boundary = []
        self.start_position = (0, 0)
        self.start_angle = 0
        self.generate_track(difficulty)
        
    def generate_track(self, difficulty):
        """Gera uma pista baseada na dificuldade"""
        # Centro da pista
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        if difficulty == 'easy':
            # Pista oval simples
            num_points = 20
            outer_radius_x, outer_radius_y = 300, 200
            inner_radius_x, inner_radius_y = 200, 100
            variation_scale = 0.0  # Sem variação na pista oval
        elif difficulty == 'medium':
            # Pista com mais curvas
            num_points = 30
            outer_radius_x, outer_radius_y = 300, 200
            inner_radius_x, inner_radius_y = 180, 80
            variation_scale = 0.2  # Alguma variação para curvas e chicanes
        else:  # 'hard'
            # Pista complexa
            num_points = 40
            outer_radius_x, outer_radius_y = 320, 220
            inner_radius_x, inner_radius_y = 160, 60
            variation_scale = 0.4  # Muita variação para pista complexa
        
        # Gerar pontos da pista
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Adicionar variação para pistas não-oval
            if difficulty != 'easy':
                # Variação baseada em múltiplas funções seno para criar curvas interessantes
                variation = 1.0 + variation_scale * math.sin(3 * angle) + variation_scale/2 * math.sin(5 * angle + 0.5)
                
                # Adicionar chicanes para pista difícil
                if difficulty == 'hard' and 0.2 < angle / (2 * math.pi) < 0.3 or 0.7 < angle / (2 * math.pi) < 0.8:
                    variation *= 0.8  # Estrangulamento na pista
            else:
                variation = 1.0
                
            # Pontos da borda externa
            outer_x = center_x + outer_radius_x * math.cos(angle) * variation
            outer_y = center_y + outer_radius_y * math.sin(angle) * variation
            self.outer_boundary.append((outer_x, outer_y))
            
            # Pontos da borda interna
            inner_x = center_x + inner_radius_x * math.cos(angle) * variation
            inner_y = center_y + inner_radius_y * math.sin(angle) * variation
            self.inner_boundary.append((inner_x, inner_y))
            
            # Criar checkpoints entre as bordas
            if i % 2 == 0:  # Menos checkpoints para não sobrecarregar
                checkpoint_x = (outer_x + inner_x) / 2
                checkpoint_y = (outer_y + inner_y) / 2
                self.checkpoints.append((checkpoint_x, checkpoint_y))
        
        # Definir posição inicial
        self.start_position = ((self.outer_boundary[0][0] + self.inner_boundary[0][0]) / 2,
                              (self.outer_boundary[0][1] + self.inner_boundary[0][1]) / 2)
        
        # Ângulo inicial (tangente à pista)
        next_point = 1
        dx = self.checkpoints[next_point][0] - self.checkpoints[0][0]
        dy = self.checkpoints[next_point][1] - self.checkpoints[0][1]
        self.start_angle = math.atan2(dy, dx) * 180 / math.pi
    
    def draw(self, surface):
        """Desenha a pista na superfície fornecida"""
        # Desenhar fundo da pista
        pygame.draw.polygon(surface, GRAY, self.outer_boundary)
        
        # Desenhar área verde (dentro da pista)
        pygame.draw.polygon(surface, GREEN, self.inner_boundary)
        
        # Desenhar bordas
        pygame.draw.lines(surface, WHITE, True, self.outer_boundary, 2)
        pygame.draw.lines(surface, WHITE, True, self.inner_boundary, 2)
        
        # Desenhar linha de partida
        pygame.draw.line(surface, YELLOW, 
                         self.start_position, 
                         (self.start_position[0] - 50 * math.sin(math.radians(self.start_angle)),
                          self.start_position[1] + 50 * math.cos(math.radians(self.start_angle))), 
                         5)
        
        # Desenhar checkpoints (invisíveis no jogo real, mas úteis para debug)
        if False:  # Altere para True para visualizar checkpoints
            for i, checkpoint in enumerate(self.checkpoints):
                pygame.draw.circle(surface, RED if i == 0 else BLUE, (int(checkpoint[0]), int(checkpoint[1])), 5)
    
    def is_out_of_bounds(self, x, y):
        """Verifica se um ponto está fora da pista"""
        # Implementação simplificada - verificar se está dentro do polígono exterior
        # e fora do polígono interior
        def point_in_polygon(point, polygon):
            x, y = point
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        # Está fora se não estiver dentro da borda externa OU estiver dentro da borda interna
        outside_external = not point_in_polygon((x, y), self.outer_boundary)
        inside_internal = point_in_polygon((x, y), self.inner_boundary)
        
        return outside_external or inside_internal
    
    def get_closest_checkpoint(self, x, y):
        """Retorna o índice do checkpoint mais próximo"""
        distances = [math.sqrt((x - cp[0])**2 + (y - cp[1])**2) for cp in self.checkpoints]
        return distances.index(min(distances))

class Car:
    """Classe que representa o carro controlado pelo jogador ou IA"""
    def __init__(self, track, ai_controlled=True):
        self.x, self.y = track.start_position
        self.angle = track.start_angle
        self.speed = 0
        self.acceleration = 0
        self.steering = 0
        
        # Características do carro
        self.max_speed = 10
        self.max_acceleration = 0.2
        self.max_braking = 0.5
        self.max_steering = 5
        self.friction = 0.05
        self.angular_velocity = 0
        self.steering_sensitivity = 0.1
        
        # Dimensões
        self.width = 20
        self.length = 40
        
        # Estado do carro
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        self.current_checkpoint = 0
        self.laps_completed = 0
        self.checkpoints_hit = 0
        
        # Sensores
        self.num_sensors = 8
        self.sensor_length = 200
        self.sensor_angles = [i * 360 / self.num_sensors for i in range(self.num_sensors)]
        self.sensor_readings = [0] * self.num_sensors
        
        # IA
        self.ai_controlled = ai_controlled
        self.reward = 0
        self.total_reward = 0
        
        # Histórico para visualização
        self.position_history = []
    
    def update(self, track):
        """Atualiza o estado do carro"""
        if not self.alive:
            return
        
        self.time_alive += 1
        
        # Aplicar fricção
        if self.speed > 0:
            self.speed = max(0, self.speed - self.friction)
        elif self.speed < 0:
            self.speed = min(0, self.speed + self.friction)
        
        # Atualizar velocidade com aceleração
        self.speed += self.acceleration
        
        # Limitar velocidade
        self.speed = max(-self.max_speed/2, min(self.max_speed, self.speed))
        
        # Calcular nova posição
        angle_rad = math.radians(self.angle)
        self.x += self.speed * math.cos(angle_rad)
        self.y += self.speed * math.sin(angle_rad)
        
        # Atualizar ângulo com direção
        if self.speed != 0:  # Só virar quando estiver em movimento
            # Inversão de direção quando em marcha ré
            steering_factor = self.steering if self.speed > 0 else -self.steering
            self.angle += steering_factor * (abs(self.speed) / self.max_speed)  # Ajustar direção com base na velocidade
        
        # Atualizar distância percorrida
        self.distance_traveled += abs(self.speed)
        
        # Verificar colisão com bordas da pista
        if track.is_out_of_bounds(self.x, self.y):
            self.alive = False
            self.reward = -50  # Penalidade por colisão
        else:
            # Recompensa por permanecer na pista
            self.reward = 0.1 + (abs(self.speed) / self.max_speed)
        
        # Atualizar sensores
        self.update_sensors(track)
        
        # Verificar checkpoints
        closest_checkpoint = track.get_closest_checkpoint(self.x, self.y)
        if closest_checkpoint != self.current_checkpoint:
            # Verificar se está progredindo na ordem correta
            if (self.current_checkpoint + 1) % len(track.checkpoints) == closest_checkpoint:
                self.checkpoints_hit += 1
                self.reward += 10  # Recompensa por checkpoint
                
                # Verificar volta completa
                if closest_checkpoint == 0 and self.current_checkpoint == len(track.checkpoints) - 1:
                    self.laps_completed += 1
                    self.reward += 50  # Recompensa por volta
            
            self.current_checkpoint = closest_checkpoint
        
        # Acumular recompensa total
        self.total_reward += self.reward
        
        # Registrar posição para visualização
        if len(self.position_history) > 100:  # Limitar o histórico
            self.position_history.pop(0)
        self.position_history.append((self.x, self.y))
    
    def update_sensors(self, track):
        """Atualiza as leituras dos sensores de distância"""
        for i, angle_offset in enumerate(self.sensor_angles):
            # Calcular direção do sensor
            angle = math.radians(self.angle + angle_offset)
            
            # Verificar distância até colisão
            for distance in range(1, self.sensor_length + 1):
                # Calcular ponto do sensor
                x = self.x + distance * math.cos(angle)
                y = self.y + distance * math.sin(angle)
                
                # Verificar se o ponto está fora da pista
                if track.is_out_of_bounds(x, y):
                    self.sensor_readings[i] = distance / self.sensor_length
                    break
            else:
                # Se não encontrou colisão, sensor lê distância máxima
                self.sensor_readings[i] = 1.0
    
    def draw(self, surface):
        """Desenha o carro e seus sensores na superfície"""
        # Desenhar histórico de posições (rastro)
        if len(self.position_history) > 1:
            pygame.draw.lines(surface, (50, 50, 255, 128), False, self.position_history, 2)
        
        # Desenhar sensores
        for i, distance in enumerate(self.sensor_readings):
            angle = math.radians(self.angle + self.sensor_angles[i])
            end_x = self.x + distance * self.sensor_length * math.cos(angle)
            end_y = self.y + distance * self.sensor_length * math.sin(angle)
            
            # Gradiente de cor baseado na distância (verde = longe, vermelho = perto)
            color = (int(255 * (1 - distance)), int(255 * distance), 0)
            pygame.draw.line(surface, color, (self.x, self.y), (end_x, end_y), 1)
        
        # Criar superfície para o carro
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        
        # Cor baseada se é IA ou jogador
        car_color = BLUE if self.ai_controlled else RED
        
        # Desenhar retângulo do carro
        pygame.draw.rect(car_surface, car_color, (0, 0, self.length, self.width))
        
        # Marcar frente do carro
        pygame.draw.polygon(car_surface, YELLOW, [(self.length-5, 0), (self.length, self.width//2), (self.length-5, self.width)])
        
        # Rotacionar imagem do carro
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Desenhar carro na tela
        surface.blit(rotated_car, rotated_rect.topleft)
    
    def get_state(self):
        """Retorna o estado atual do carro para a IA"""
        return np.array(self.sensor_readings + [self.speed / self.max_speed])
    
    def apply_action(self, action):
        """Aplica uma ação ao carro
        Ações: 0 = nada, 1 = acelerar, 2 = frear, 3 = esquerda, 4 = direita,
               5 = acelerar+esquerda, 6 = acelerar+direita, 7 = frear+esquerda, 8 = frear+direita"""
        if action == 0:  # Nada
            self.acceleration = 0
            self.steering = 0
        elif action == 1:  # Acelerar
            self.acceleration = self.max_acceleration
            self.steering = 0
        elif action == 2:  # Frear
            self.acceleration = -self.max_braking
            self.steering = 0
        elif action == 3:  # Esquerda
            self.steering = -self.max_steering
        elif action == 4:  # Direita
            self.steering = self.max_steering
        elif action == 5:  # Acelerar + Esquerda
            self.acceleration = self.max_acceleration
            self.steering = -self.max_steering
        elif action == 6:  # Acelerar + Direita
            self.acceleration = self.max_acceleration
            self.steering = self.max_steering
        elif action == 7:  # Frear + Esquerda
            self.acceleration = -self.max_braking
            self.steering = -self.max_steering
        elif action == 8:  # Frear + Direita
            self.acceleration = -self.max_braking
            self.steering = self.max_steering

# Implementação alternativa para quando TensorFlow não está disponível
class SimpleAgent:
    """Agente de aprendizado simples quando TensorFlow não está disponível"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Fator de desconto
        self.epsilon = 1.0  # Taxa de exploração
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}  # Tabela Q simplificada
    
    def _get_state_key(self, state):
        """Converte estado contínuo em discreto para uso na tabela Q"""
        # Simplificar estado para discretização
        discrete_state = []
        for value in state:
            discrete_state.append(round(value, 1))
        return tuple(discrete_state)
    
    def act(self, state, training=True):
        """Seleciona uma ação com base no estado atual"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        """Armazena experiência para replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Treinar o agente usando experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            # Atualizar Q-valor
            self.q_table[state_key][action] = (1 - 0.1) * self.q_table[state_key][action] + 0.1 * target
        
        # Decair epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Método de compatibilidade com DQNAgent"""
        pass
    
    def load(self, name):
        """Carregar modelo de arquivo"""
        try:
            with open(name, 'r') as f:
                import json
                self.q_table = json.load(f)
        except:
            print(f"Não foi possível carregar modelo: {name}")
    
    def save(self, name):
        """Salvar modelo em arquivo"""
        with open(name.replace('.h5', '.json'), 'w') as f:
            import json
            json.dump({str(k): v.tolist() for k, v in self.q_table.items()}, f)

# Definir DQNAgent apenas se TensorFlow estiver disponível
if TENSORFLOW_AVAILABLE:
    class DQNAgent:
        """Agente de aprendizado por reforço usando Deep Q-Network"""
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            
            # Hiperparâmetros
            self.gamma = 0.95  # Fator de desconto
            self.epsilon = 1.0  # Taxa de exploração
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.batch_size = 32
            
            # Memória para experience replay
            self.memory = deque(maxlen=2000)
            
            # Modelo de rede neural
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        
        def _build_model(self):
            """Constrói a rede neural para o Q-learning"""
            model = Sequential()
            
            # Camada LSTM para processar sequência de sensores
            model.add(LSTM(64, input_shape=(self.state_size, 1), return_sequences=False))
            
            # Camadas densas
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.action_size, activation='linear'))
            
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        
        def update_target_model(self):
            """Atualiza o modelo alvo para estabilizar o treinamento"""
            self.target_model.set_weights(self.model.get_weights())
        
        def remember(self, state, action, reward, next_state, done):
            """Armazena experiência para replay"""
            self.memory.append((state, action, reward, next_state, done))
        
        def act(self, state, training=True):
            """Seleciona uma ação com base no estado atual"""
            if training and np.random.rand() <= self.epsilon:
                # Exploração: ação aleatória
                return random.randrange(self.action_size)
            
            # Reshape para formato LSTM
            state = np.reshape(state, [1, self.state_size, 1])
            
            # Exploitation: usar modelo para prever melhor ação
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        
        def replay(self, batch_size=None):
            """Treina o modelo usando experience replay"""
            if batch_size is None:
                batch_size = self.batch_size
                
            if len(self.memory) < batch_size:
                return
            
            # Amostragem aleatória da memória
            minibatch = random.sample(self.memory, batch_size)
            
            # Preparar dados para treinamento
            states = np.zeros((batch_size, self.state_size, 1))
            targets = np.zeros((batch_size, self.action_size))
            
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                # Reshape para formato LSTM
                state = np.reshape(state, [1, self.state_size, 1])
                next_state = np.reshape(next_state, [1, self.state_size, 1])
                
                # Calcular alvo para treinamento
                target = self.model.predict(state, verbose=0)[0]
                
                if done:
                    target[action] = reward
                else:
                    t = self.target_model.predict(next_state, verbose=0)[0]
                    target[action] = reward + self.gamma * np.amax(t)
                
                states[i] = state
                targets[i] = target
            
            # Treinar modelo
            self.model.fit(states, targets, epochs=1, verbose=0)
            
            # Decair epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def load(self, name):
            """Carrega os pesos do modelo"""
            try:
                if name.endswith('.h5') and not name.endswith('.weights.h5'):
                    name = name.replace('.h5', '.weights.h5')
                self.model.load_weights(name)
                self.target_model.load_weights(name)
            except:
                print(f"Não foi possível carregar modelo: {name}")
        
        def save(self, name):
            """Salva os pesos do modelo"""
            if name.endswith('.h5') and not name.endswith('.weights.h5'):
                name = name.replace('.h5', '.weights.h5')
            self.model.save_weights(name)

class Game:
    """Classe principal do jogo"""
    def __init__(self, mode='player', difficulty='easy'):
        self.mode = mode
        self.track = Track(difficulty)
        self.cars = []
        self.generation = 1
        self.best_reward = 0
        self.best_distance = 0
        self.best_laps = 0
        self.training = mode == 'train'
        self.agent = None
        self.population_size = 10 if mode == 'train' else 1
        
        # Criar pasta para salvar modelos se não existir
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Inicializar carros
        if mode == 'player':
            self.cars.append(Car(self.track, ai_controlled=False))
        else:
            # Modo IA (treino ou demo)
            for _ in range(self.population_size):
                self.cars.append(Car(self.track))
            
            # Selecionar agente apropriado
            state_size = self.cars[0].num_sensors + 1  # Sensores + velocidade
            if TENSORFLOW_AVAILABLE:
                self.agent = DQNAgent(state_size, 9)  # 9 ações possíveis
            else:
                self.agent = SimpleAgent(state_size, 9)
            
            # Carregar modelo existente para demo ou continuar treinamento
            model_path = f'models/autoracer_{difficulty}.weights.h5' if TENSORFLOW_AVAILABLE else f'models/autoracer_{difficulty}.json'
            if os.path.exists(model_path):
                self.agent.load(model_path)
                # Reduzir exploração ao usar modelo pré-treinado
                self.agent.epsilon = 0.1 if mode == 'train' else 0.01
        
        # Estatísticas para gráficos
        self.reward_history = []
        self.distance_history = []
        self.lap_history = []
    
    def handle_events(self):
        """Processa eventos de entrada"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Alternar modo de debug com F1
                if event.key == pygame.K_F1:
                    global DEBUG_MODE
                    DEBUG_MODE = not DEBUG_MODE
        
        # Controles do jogador (apenas no modo player)
        if self.mode == 'player' and len(self.cars) > 0:
            keys = pygame.key.get_pressed()
            car = self.cars[0]
            
            car.acceleration = 0  # Reset acceleration
            car.steering = 0  # Reset steering
            
            if keys[pygame.K_UP]:
                car.acceleration = car.max_acceleration
            if keys[pygame.K_DOWN]:
                car.acceleration = -car.max_braking
            if keys[pygame.K_LEFT]:
                car.steering = -car.max_steering
            if keys[pygame.K_RIGHT]:
                car.steering = car.max_steering
        
        return True
    
    def update(self):
        """Atualiza o estado do jogo"""
        # Atualizar todos os carros
        alive_cars = 0
        
        for car in self.cars:
            if car.alive:
                alive_cars += 1
                
                # Aplicar ação se controlado por IA
                if car.ai_controlled and self.agent and (self.mode == 'train' or self.mode == 'demo'):
                    # Obter estado atual
                    state = car.get_state()
                    
                    # Selecionar ação
                    action = self.agent.act(state, training=self.training)
                    
                    # Aplicar ação
                    car.apply_action(action)
                    
                    # Salvar estado antes da atualização
                    prev_state = np.copy(state)
                
                # Atualizar carro
                car.update(self.track)
                
                # Registrar experiência para treinamento
                if car.ai_controlled and self.agent and self.training:
                    # Obter novo estado
                    next_state = car.get_state()
                    
                    # Registrar experiência
                    self.agent.remember(prev_state, action, car.reward, next_state, not car.alive)
                
                # Atualizar estatísticas
                if car.total_reward > self.best_reward:
                    self.best_reward = car.total_reward
                
                if car.distance_traveled > self.best_distance:
                    self.best_distance = car.distance_traveled
                
                if car.laps_completed > self.best_laps:
                    self.best_laps = car.laps_completed
        
        # Treinar o agente
        if self.agent and self.training and alive_cars > 0:
            self.agent.replay()
        
        # Se todos os carros morreram, reiniciar
        if alive_cars == 0 and self.mode != 'player':
            self.reset_simulation()
        elif self.mode == 'player' and alive_cars == 0:
            # Reiniciar carro do jogador quando morre
            self.cars = [Car(self.track, ai_controlled=False)]
    
    def reset_simulation(self):
        """Reinicia a simulação para a próxima geração"""
        # Salvar estatísticas
        if len(self.cars) > 0:
            best_car = max(self.cars, key=lambda car: car.total_reward)
            self.reward_history.append(best_car.total_reward)
            self.distance_history.append(best_car.distance_traveled)
            self.lap_history.append(best_car.laps_completed)
        
        # Salvar modelo a cada 10 gerações
        if self.agent and self.training and self.generation % 10 == 0:
            self.agent.save(f'models/autoracer_{self.track.difficulty}.h5')
            
            # Atualizar modelo alvo periodicamente
            self.agent.update_target_model()
            
            # Salvar gráficos de progresso
            self.plot_statistics()
        
        # Incrementar geração
        self.generation += 1
        
        # Resetar carros
        self.cars = []
        for _ in range(self.population_size):
            self.cars.append(Car(self.track))
    
    def draw(self):
        """Renderiza o jogo na tela"""
        # Limpar tela
        screen.fill(BLACK)
        
        # Desenhar pista
        self.track.draw(screen)
        
        # Desenhar carros
        for car in self.cars:
            if car.alive:
                car.draw(screen)
        
        # Desenhar informações na tela
        y_offset = 10
        
        # Informações gerais
        draw_text(f"Geração: {self.generation}", 10, y_offset, WHITE)
        y_offset += 25
        draw_text(f"Carros vivos: {sum(1 for car in self.cars if car.alive)}/{len(self.cars)}", 10, y_offset, WHITE)
        y_offset += 25
        
        # Informações do melhor carro
        if len(self.cars) > 0:
            best_car = max((car for car in self.cars if car.alive), key=lambda car: car.total_reward, default=None)
            if best_car:
                draw_text(f"Melhor recompensa: {best_car.total_reward:.1f}", 10, y_offset, WHITE)
                y_offset += 25
                draw_text(f"Distância: {best_car.distance_traveled:.1f}", 10, y_offset, WHITE)
                y_offset += 25
                draw_text(f"Voltas: {best_car.laps_completed}", 10, y_offset, WHITE)
                y_offset += 25
                draw_text(f"Checkpoints: {best_car.checkpoints_hit}", 10, y_offset, WHITE)
                y_offset += 25
        
        # Informações de treinamento
        if self.agent and self.mode == 'train':
            draw_text(f"Epsilon: {self.agent.epsilon:.4f}", WIDTH - 200, 10, YELLOW)
            draw_text(f"Memória: {len(self.agent.memory)}/{self.agent.memory.maxlen}", WIDTH - 200, 35, YELLOW)
        
        # Recorde geral
        draw_text(f"Recorde de voltas: {self.best_laps}", WIDTH - 200, 60, GREEN)
        draw_text(f"Recorde de distância: {self.best_distance:.1f}", WIDTH - 200, 85, GREEN)
        
        # Controles (no modo jogador)
        if self.mode == 'player':
            draw_text("Controles: Setas direcionais", WIDTH - 200, HEIGHT - 60, WHITE)
            draw_text("ESC: Sair", WIDTH - 200, HEIGHT - 35, WHITE)
        
        # Atualizar tela
        pygame.display.flip()
    
    def plot_statistics(self):
        """Plota estatísticas de treinamento"""
        if len(self.reward_history) > 1:
            try:
                plt.figure(figsize=(15, 5))
                
                # Recompensas
                plt.subplot(1, 3, 1)
                plt.plot(self.reward_history)
                plt.title('Recompensa por Geração')
                plt.xlabel('Geração')
                plt.ylabel('Recompensa')
                
                # Distância
                plt.subplot(1, 3, 2)
                plt.plot(self.distance_history)
                plt.title('Distância por Geração')
                plt.xlabel('Geração')
                plt.ylabel('Distância')
                
                # Voltas
                plt.subplot(1, 3, 3)
                plt.plot(self.lap_history)
                plt.title('Voltas por Geração')
                plt.xlabel('Geração')
                plt.ylabel('Voltas Completas')
                
                plt.tight_layout()
                plt.savefig(f'stats_{self.track.difficulty}.png')
                plt.close()
            except Exception as e:
                print(f"Erro ao plotar estatísticas: {e}")

# Variável global para modo debug
DEBUG_MODE = False

def main():
    """Função principal"""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='AutoRacer AI - Um jogo de corrida com aprendizado por reforço')
    parser.add_argument('--mode', choices=['player', 'train', 'demo'], default='player',
                       help='Modo de jogo: player (jogador humano), train (treinar IA), demo (demonstração)')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='easy',
                       help='Dificuldade da pista')
    args = parser.parse_args()
    
    # Verificar TensorFlow para modos que precisam dele
    if args.mode == 'train' and not TENSORFLOW_AVAILABLE:
        print("Aviso: TensorFlow não está disponível. Usando algoritmo simplificado para treinamento.")
    
    # Inicializar jogo
    game = Game(mode=args.mode, difficulty=args.difficulty)
    
    # Loop principal
    running = True
    frame_count = 0
    start_time = time.time()
    
    while running:
        # Processar eventos
        running = game.handle_events()
        
        # Atualizar jogo
        game.update()
        
        # Renderizar
        game.draw()
        
        # Limitar FPS
        clock.tick(FPS)
        
        # Calcular FPS
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
    
    # Plotar estatísticas no final
    if args.mode == 'train':
        game.plot_statistics()
    
    # Salvar modelo final
    if game.agent and args.mode == 'train':
        if TENSORFLOW_AVAILABLE:
            model_path = f'models/autoracer_{args.difficulty}.weights.h5'
        else:
            model_path = f'models/autoracer_{args.difficulty}.json'
        game.agent.save(model_path)
        print(f"Modelo salvo em: {model_path}")
    
    # Encerrar pygame
    pygame.quit()

if __name__ == "__main__":
    main()