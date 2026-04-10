# Criando ambientes customizados usando a biblioteca Gymnasium

O objetivo deste repositório é fornecer alguns exemplos de ambientes customizados criados 
usando a biblioteca Gymnasium. 

Você pode usar este arquivo README.md como um handout para entender como implementar ambientes customizados e como utilizá-los.

## Instalação

Para começar a usar este repositório você precisa clonar o repositório e instalar as dependências necessárias. Você pode fazer isso usando os seguintes comandos depois de clonar o repositório:

```bash
python -m venv venv # para criar um ambiente virtual
source venv/bin/activate # para ativar o ambiente virtual
pip install -r requirements.txt # para instalar as dependências
```

## Primeiro exemplo: ambiente GridWorld sem renderização

O primeiro exemplo é um ambiente simples de grid world. O agente pode se mover para cima, baixo, esquerda ou direita. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world.py` dentro da pasta `gymnasium_env`. 

O código deste arquivo é baseado no tutorial disponível em [https://gymnasium.farama.org/introduction/create_custom_env/](https://gymnasium.farama.org/introduction/create_custom_env/). Este código tem todos os métodos necessários para criar um ambiente: `__init__`, `reset` e `step`. Só não tem o médoto `render` que é responsável por mostrar visualmente o ambiente.  

Os arquivos listados abaixo utilizam o ambiente `GridWorldEnv`: 

* `run_grid_world_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente. 

**Questão**: Qual é a diferença entre o estado retornado pelo ambiente e o estado retornado pelo ambiente com o uso do wrapper? O que cada variável representa?

* `train_grid_world_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv`. 

**Proposta**: 

* Execute o comando:

```bash
python train_grid_world_render_v0.py train
```

* Visualize a curva de aprendizado usando o plugin do tensorboard com os dados armazenados na pasta `log`. 

* Execute diversas vezes o comando: 

```bash
python train_grid_world_render_v0.py test
```

para visualizar se o agente aprendeu a melhor política. 


## Segundo exemplo: ambiente GridWorld com renderização

O segundo exemplo é o mesmo ambiente de grid world, mas agora a implementação do ambiente tem o método `render` que mostra visualmente o ambiente. A implementação deste ambiente está no arquivo `grid_world_render.py` dentro da pasta `gymnasium_env`.

Os arquivos que utilizam o ambiente `GridWorldEnv` com renderização são:

* `run_grid_world_render_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_render_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente.
* `train_grid_world_render_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` com renderização.

Este último arquivo tem um código mais completo, pois o agente é treinado para atuar em um ambiente que tem uma representação visual, o modelo treinado é salvo e depois carregado para fazer uma execução do ambiente. Os dados sobre o treinamento do agente são salvos para depois serem utilizados pelo `tensorboard`.

## Terceiro exemplo: ambiente GridWorld em 3D

O terceiro exemplo é uma extensão do ambiente de grid world para um ambiente 3D. O agente pode se mover para cima, baixo, esquerda, direita, frente e trás. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_3D.py` dentro da pasta `gymnasium_env`.

O arquivo que utiliza o ambiente `GridWorldEnv` em 3D é:
* `train_grid_world_3D.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` em 3D.

Existem 3 (três) formas de uso do script `train_grid_world_3D.py`:
* `python train_grid_world_3D.py train`: treina o agente e salva o modelo treinado na pasta `data` e os logs na pasta `log`.    
* `python train_grid_world_3D.py test`: carrega o modelo treinado e executa 100 episódios, calculando o percentual de sucesso do agente, entre outras métricas.
* `python train_grid_world_3D.py run`: carrega o modelo treinado e executa um único episódio, mostrando a renderização do ambiente 3D.

Para que a renderização deste ambiente aconteça, é necessário ter a biblioteca `tkinter` instalada. No Ubuntu, você pode instalar esta biblioteca com o comando:

```bash
sudo apt-get install python3-tk
```

**Importante**: esta renderização 3D foi testada apenas no sistema operacional Ubuntu.


## Quarto exemplo: ambiente GridWorld com obstáculos

O quarto exemplo é uma extensão do ambiente de grid world para incluir obstáculos. O agente deve navegar pelo ambiente evitando os obstáculos para alcançar o objetivo. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_obstacles.py` dentro da pasta `gymnasium_env`.

Para executar o treinamento do agente no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py train
```

Para testar o agente treinado no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py test
```

Esta funcionalidade irá executar o agente treinado em 100 episódios e calcular o percentual de sucesso do agente, entre outras métricas. 

Também é possível executar o agente treinado em um único episódio, para isso execute o comando:

```bash
python train_grid_world_obstacles.py run
```

## Uso do ambiente GridWorld para problemas de Coverage Path Planning

O **Coverage Path Planning (CPP)** é um problema clássico de planejamento onde o objetivo é encontrar um caminho que cubra todas as células acessíveis de um grid. Diferente do objetivo original (chegar a um ponto fixo), no CPP o agente deve visitar *todas* as células não-obstáculo pelo menos uma vez.

O ambiente CPP está implementado em `gymnasium_env/grid_world_cpp.py` e o script de teste com agente aleatório em `run_grid_world_cpp.py`.

Para executar o agente aleatório em um grid 5×5 (sem renderização, 10 episódios):

```bash
python run_grid_world_cpp.py
```

Para visualizar a execução com renderização gráfica (3 episódios):

```bash
python run_grid_world_cpp.py render
```

Para obter estatísticas sobre 100 episódios:

```bash
python run_grid_world_cpp.py stats
```

---

### Função de reward original (tarefa de navegação com alvo fixo)

O ambiente `grid_world_obstacles.py` implementa uma tarefa de navegação onde o agente deve alcançar um alvo fixo. A função de reward tem três casos:

| Situação | Reward |
|---|---|
| Agente alcança o alvo | `+10.0` |
| Passo normal | `dist_anterior − dist_atual − 0.1` (shaping por distância + custo de passo) |
| Episódio truncado (max steps) sem alcançar o alvo | `−10.0` |

Essa função recompensa o agente por se aproximar do alvo a cada passo (*potential-based shaping*) e aplica uma punição forte se o orçamento de passos se esgotar sem sucesso.

---

### Nova função de reward (CPP)

Para o CPP, o conceito de alvo fixo não existe — o sucesso é medido pela cobertura total do ambiente. A nova função de reward foi projetada com três princípios:

**Incentivar exploração**: recompensar visitas a células novas.\
**Desincentivar repetição**: punir o retorno a células já visitadas.\
**Premiar a conclusão**: oferecer um bônus terminal quando cobertura total é atingida.


| Situação | Reward |
|---|---|
| Visitar uma célula **nova** (não visitada) | `+1.0` |
| Revisitar uma célula **já coberta** | `−0.5` |
| Custo fixo por passo (eficiência) | `−0.1` (aplicado sempre) |
| Bônus de conclusão (cobertura 100% atingida) | `+10.0` (acumulado com o reward do passo) |

Exemplo de rewards em um passo típico:
- Visitar célula nova: `1.0 − 0.1 = +0.9`
- Revisitar célula: `−0.5 − 0.1 = −0.6`
- Completar cobertura no passo: `1.0 − 0.1 + 10.0 = +10.9`

A combinação de reward positivo para células novas e negativo para revisitas cria um gradiente que guia o agente em direção a regiões inexploradas. O bônus terminal incentiva o agente a *terminar* a cobertura em vez de apenas coletar rewards parciais indefinidamente.

---

### Mudanças no ambiente em relação ao `grid_world_obstacles.py`

| Aspecto | Goal-reaching (original) | CPP (novo) |
|---|---|---|
| Objetivo | Alcançar alvo fixo | Cobrir todas as células acessíveis |
| Terminação | Agente chega ao alvo | Cobertura 100% atingida |
| Estado | `[agent_x, agent_y, target_x, target_y, neighbors]` | `[agent_x, agent_y, visited_map (5×5), neighbors]` |
| Reward | Shaping por distância ao alvo | Por célula nova / revisita + bônus de conclusão |
| Alvo fixo | Sim | Não |

O **mapa de células visitadas** (`visited_map`) é parte fundamental do estado: sem ele, o agente não teria informação suficiente para aprender a evitar células já cobertas.
