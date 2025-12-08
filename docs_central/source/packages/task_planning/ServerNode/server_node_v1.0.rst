server_node.cpp (Atual v1.0)
========================================

O ``ServerNode`` atua como o **cérebro central** do robô.
Este nó utiliza **Behavior Trees (Árvores de Comportamento)**, permitindo que o sistema seja reativo: novos objetos são percebidos dinamicamente e falhas na navegação acionam rotinas de recuperação.

O nó funciona como um coordenador, gerenciando os pacotes de **Navegação** (Navigation2), **Manipulação** (MoveIt 2) e **Visão** (YOLO).

.. note::
   A arquitetura deste nó é **Multi-Threaded**. O ROS 2 opera em uma thread (para recepção de callbacks de sensores) e a Behavior Tree executa em outra (para a tomada de decisões). Tal arquitetura exige o uso cuidadoso de `std::mutex` para evitar conflitos de dados.

---

1. Adaptação para BehaviorTree.CPP
----------------------------------

Para a integração do ROS 2 com a biblioteca `BehaviorTree.CPP` (BT.CPP), são necessários adaptadores específicos que permitam a troca de dados complexos entre o XML da árvore e o código C++.

1.1 Conversão de Tipos (Templates)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A biblioteca BT.CPP, por padrão, lida nativamente apenas com tipos primitivos (int, float, string). Para utilizar tipos complexos do ROS, como `geometry_msgs::msg::Pose`, nas portas de entrada/saída (Blackboard), é obrigatório fornecer uma especialização de template.

O código abaixo instrui o compilador sobre como interpretar uma string do XML caso ela represente uma `Pose`. Embora a implementação padrão retorne um objeto vazio, ela é mandatória para evitar erros de linkagem.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: convertFromString
   :end-before: // DOC-END: convertFromString

1.2 Controle de Fluxo Personalizado (ParallelAny)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

O nó de controle padrão "Parallel" da BT.CPP aguarda o sucesso de *todos* os filhos ou a falha de *N* filhos. Para esta aplicação, foi necessário um comportamento mais reativo, similar a um "OR Lógico Paralelo".

O nó ``ParallelAny`` foi implementado para executar tarefas simultâneas (ex: navegar e monitorar sensores) e retornar:
* **Sucesso:** Imediatamente se *qualquer um* dos filhos obtiver sucesso (Short-circuit).
* **Falha:** Imediatamente se *qualquer um* falhar.

.. seealso::
   Para uma descrição funcional deste nó na árvore, consulte a :doc:`../BT/index`.

**Implementação do Tick:**
A função `tick()` itera sobre todos os filhos a cada ciclo. Diferente de um nó sequencial, ele não para no primeiro `RUNNING`; ele executa todos. Porém, se detectar um estado terminal (SUCCESS ou FAILURE) em qualquer filho, interrompe os demais (`haltChildren`) e retorna o resultado.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: ParallelAny
   :end-before: // DOC-END: ParallelAny

1.3 Wrapper para Ações Assíncronas (AsyncAction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Em aplicações de robótica, muitas ações são longas (navegar, mover braço) e não podem bloquear o fluxo de execução. Criar uma classe C++ separada (`.h` e `.cpp`) herdando de `BT::StatefulActionNode` para cada ação simples é verboso e difícil de manter.

A classe ``AsyncAction`` resolve isso atuando como um *wrapper* genérico. Ela permite injetar a lógica da ação diretamente no código principal através de funções Lambda (`std::function`).

**Funcionamento:**
* **onStart:** Chamado na primeira vez que a ação é executada. Invoca a lambda.
* **onRunning:** Chamado nos ticks subsequentes enquanto a ação não termina. Invoca a mesma lambda.

Isso permite escrever nós de ação complexos, como ``PickObject`` ou ``ComputePath``, diretamente dentro do método de configuração da árvore, mantendo o código conciso e localizado.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: AsyncAction
   :end-before: // DOC-END: AsyncAction

---

2. Classe Principal: ServerNode
-------------------------------

Esta classe gerencia toda a infraestrutura do sistema.

**Construtor e Inicialização:**
Nesta etapa, o sistema é configurado. Observa-se a injeção de dependência: o nó recebe ponteiros para `StorageNode` e `OrganizeNode` do pacote ``storage_manager``, viabilizando a comunicação direta em memória RAM (processo mais rápido que tópicos ROS para dados internos).

Destaques:
* Criação de **Action Clients** para conexão com a navegação e o braço robótico.
* Inicialização da thread dedicada (`bt_thread_`) para a lógica de decisão.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: ServerNode
   :end-before: // DOC-END: ServerNode

**Configuração da Árvore de Comportamento:**
Trata-se da função mais densa e relevante. Nela, registram-se os "blocos de construção" utilizados no arquivo XML (`.btproj`).
São definidas as lógicas de:
* ``IsRobotNear``: Verificação de distância matemática simples.
* ``DetectObject``: Lógica de fixação de alvo em um objeto.
* ``ComputePath`` / ``MapsTo``: Integração com a stack de navegação.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: setup_behavior_tree
   :end-before: // DOC-END: setup_behavior_tree

**O Coração do Sistema (Loop Tick):**
Este loop executa a 50Hz na thread secundária.
Para economia de CPU, a árvore é "pulsada" (chamada de `tickOnce`) apenas se:
1.  A árvore já estiver executando uma tarefa longa.
2.  Um **novo objeto** for detectado (flag `has_new_object_`).
3.  Já existir um alvo ativo.

Caso nenhuma das condições seja atendida, o robô permanece em repouso (`sleep`).

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: bt_loop
   :end-before: // DOC-END: bt_loop

---

3. Estruturas de Dados e Variáveis Internas
-------------------------------------------

Abaixo são detalhadas as estruturas auxiliares e as variáveis membro da classe ``ServerNode``, essenciais para o gerenciamento de estado e comunicação entre threads.

**Estruturas Auxiliares:**

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: internal_structs
   :end-before: // DOC-END: internal_structs

**Variáveis Membro (State, ROS & Threading):**

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: member_variables
   :end-before: // DOC-END: member_variables

4. Callbacks de Percepção (Inputs)
----------------------------------

Estas funções são acionadas automaticamente pelo ROS mediante o recebimento de dados dos sensores.

**Odometria:**
Mantém as variáveis globais `pose_x` e `pose_y` atualizadas. Trata-se de uma operação leve e rápida.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: odom_callback
   :end-before: // DOC-END: odom_callback

**Detecção de Objetos (YOLO):**
Neste ponto, aplicam-se os filtros de negócio:
1.  O objeto consta na lista de permitidos (`authorized_labels`)?
2.  Este objeto já foi coletado anteriormente (`picked`)?
Caso o objeto seja válido e novo, a flag `has_new_object_ = true` é definida, ativando a thread da Behavior Tree para iniciar a missão.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: detection_callback
   :end-before: // DOC-END: detection_callback

---

5. Integração com Navegação
---------------------------

A navegação é dividida em duas etapas para maior controle.

**1. Planejamento (Compute Path):**
Solicita-se ao algoritmo global (A*) um caminho até o destino. Em caso de falha nesta etapa, o movimento não é iniciado.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: send_path_goal
   :end-before: // DOC-END: send_path_goal

**Callback de Feedback do Caminho:**
Se o planejador global determinar uma alteração no caminho durante o trajeto (ex: surgimento de um novo obstáculo), este callback é acionado.
Se `recalculating_path` for verdadeiro, o controlador atual é cancelado imediatamente por segurança.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: path_feedback_callback
   :end-before: // DOC-END: path_feedback_callback

**2. Controle (Navigate To):**
Envia o caminho calculado para execução pelo controlador local (Pure Pursuit / DWB).

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: send_controller_goal
   :end-before: // DOC-END: send_controller_goal

---

6. Integração com Manipulação
-----------------------------

Envia comandos de alto nível para o nó de manipulação (Pick ou Place). O nó de manipulação (descrito na seção Legado ou na interface atual) é o responsável pela cinemática inversa e pelo MoveIt.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: send_goal
   :end-before: // DOC-END: send_goal

---

7. Análise de Alcançabilidade Projetada
---------------------------------------

Para determinar a viabilidade da manipulação antes de mover a base, o sistema calcula a **Região Viável da Base** (Kinematic Footprint) no plano :math:`z`. Nesse caso o robô estava em cima de 
uma base móvel em :math:`z = 0.11`.

O algoritmo projeta a esfera de alcance máximo do manipulador (Workspace Sphere) no chão, considerando o offset vertical da montagem do braço.

**Lógica Geométrica:**
A função deriva o raio 2D (:math:`r_{2d}`) utilizando o Teorema de Pitágoras, onde a hipotenusa é o alcance máximo do braço e o cateto vertical é a diferença de altura entre o objeto e o ombro do robô.

.. math::

    r_{2d} = \sqrt{R_{max}^2 - (z_{obj} - z_{base})^2}

Onde:

* :math:`R_{max}`: Alcance máximo do manipulador menos um offset para evitar singularidade (Definido como 0.9m).
* :math:`z_{base}`: Altura da primeira junta do manipulador (Definido como 0.11m).

**Visualização de Debug (MarkerArray):**
Para facilitar o desenvolvimento e a depuração visual no RViz, esta função publicava um ``MarkerArray`` contendo:

1.  **Disco Translúcido:** Representa a zona de estacionamento válida no chão.
2.  **Cubo Vermelho:** A posição exata do objeto alvo percebido.
3.  **Triângulo Vetorial:** Linhas coloridas desenhando os catetos e a hipotenusa para validação visual da matemática.

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização do raio de alcançabilidade em 2D.</em></p>
       <img src="../../../_static/task_planning/Raio de alcançabilidade 2d.png" alt="Visualização do raio de alcançabilidade em 2D." style="width: 100%; height: auto;">
   </div>

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização lateral do Triângulo de Cálculo (Catetos e Hipotenusa).</em></p>
       <img src="../../../_static/task_planning/Vista lateral do triângulo.png" alt="Visualização lateral do Triângulo de Cálculo." style="width: 100%; height: auto;">
   </div>

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização tanto do raio de alcançabilidade quanto do triângulo retângulo.</em></p>
       <img src="../../../_static/task_planning/Raio de alcançabilidade e triângulo retângulo.png" alt="Screenshot do RViz." style="width: 100%; height: auto;">
   </div>

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: calculate_max_2d_radius
   :end-before: // DOC-END: calculate_max_2d_radius

---

8. Utilitários e Main
---------------------

**Função Main:**
Configura o `MultiThreadedExecutor`. Este componente é crucial: devido ao uso de Actions (assíncronas), são necessárias múltiplas threads processando callbacks simultaneamente, caso contrário, ocorreria o travamento do robô (deadlock) aguardando a própria resposta.

.. literalinclude:: ../../../../../docs_central/legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: main
   :end-before: // DOC-END: main