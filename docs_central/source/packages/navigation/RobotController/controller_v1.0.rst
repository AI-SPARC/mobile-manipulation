controller.cpp (Atual v1.0)
========================================

O nó ``Controller`` (RobotController) é responsável pela execução física do caminho planejado. Ele recebe um vetor de poses (gerado pelo A* ou D*) e converte isso em comandos de velocidade (`cmd_vel`) para os motores.

O algoritmo base é uma variação do **Pure Pursuit**, otimizado com lógica de timeout e rotação in-place.

Inicialização
-------------

Configura os parâmetros de controle (PID e limites) e prepara os tópicos.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/RobotController/controller_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Controller_Constructor
   :end-before: // DOC-END: Controller_Constructor

Odometria e Utilitários
-----------------------

Funções para normalização de ângulos e atualização da pose do robô (Feedback de malha fechada).

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/RobotController/controller_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: odom_callback
   :end-before: // DOC-END: odom_callback

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/RobotController/controller_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Helpers
   :end-before: // DOC-END: Helpers

Loop de Controle (Execute)
--------------------------

Esta é a função principal que roda durante a navegação.

**Estratégia de Controle:**

1.  **Lookahead:** Busca um ponto no caminho a uma distância fixa (`lookahead_distance_`) à frente do robô. Isso suaviza o movimento.
2.  **Velocidade Adaptativa:** A velocidade linear é reduzida proporcionalmente ao erro angular (curvatura). Se a curva for muito fechada (> 0.8 rad), o robô para e gira no próprio eixo.
3.  **Desaceleração:** A velocidade diminui conforme o robô se aproxima do objetivo final.

**Mecanismos de Watchdog (Zonas de Parada):**

Para evitar que o robô fique "dançando" infinitamente tentando alcançar uma precisão submilimétrica impossível, implementamos zonas de tolerância temporal:

* **Zona 1 (< 15cm):** Se o robô ficar aqui por mais de 1.5s, considera-se que chegou.
* **Zona 2 (< 25cm):** Se o robô ficar aqui por mais de 4.0s (ex: lutando contra atrito ou obstáculo), aceita-se a posição e encerra.

**Rotação Final:**
Após chegar na posição XY, o robô faz um alinhamento final de orientação (Yaw) para bater com a pose desejada pelo planejador.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/RobotController/controller_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: execute
   :end-before: // DOC-END: execute