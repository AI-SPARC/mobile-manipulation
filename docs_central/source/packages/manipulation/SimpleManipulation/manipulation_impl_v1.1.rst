manipulation_impl.cpp (Atual v1.1 - Tentativa de recuperação)
========================================================================

Esta página documenta a evolução da lógica de manipulação (Versão 1.1).
Além das funcionalidades básicas da v1.0, esta versão introduz robustez através de uma lógica recursiva de **Tentativa de Recuperação (Retry)** e um sistema de **Atualização da Pose do objeto em Tempo Real**.

Demonstração
------------

Abaixo, um vídeo da execução desta lógica:

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc;">
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/manipulation/manipulation_impl_v1.1.webm" type="video/webm">
           Seu navegador não suporta vídeos HTML5.
       </video>
   </div>

---

1. Construtor e Configuração
----------------------------

O construtor inicializa os nós, executors e subscribers.
**Destaque da v1.1:** A adição do subscriber ``object_pose`` (linha 28-31 no snippet abaixo) permite que o nó monitore a posição do objeto continuamente, essencial para a lógica de retry.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: SimpleManipulation
   :end-before: // DOC-END: SimpleManipulation
   :caption: Inicialização do Nó e Novos Subscribers
   :linenos:
   :emphasize-lines: 28-31

2. Destrutor (Limpeza)
----------------------

Garante que a thread do executor do MoveIt seja encerrada corretamente para evitar erros de memória.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: ~SimpleManipulation
   :end-before: // DOC-END: ~SimpleManipulation

3. Carregamento de Parâmetros (YAML)
------------------------------------

Lê o arquivo YAML para obter os *offsets* de pega calibrados para cada tipo de objeto.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: loadLocationsFromYaml
   :end-before: // DOC-END: loadLocationsFromYaml

4. Inicialização Tardia (MoveGroup)
-----------------------------------

Chamada via Timer para instanciar as interfaces do MoveIt (`panda_arm` e `hand`) apenas quando o sistema ROS estiver estável.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: initMoveGroup
   :end-before: // DOC-END: initMoveGroup

5. Posição de Segurança (Ready)
-------------------------------

Move o braço para uma posição "Home" segura para navegação.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: ready
   :end-before: // DOC-END: ready

6. Controle da Garra
--------------------

Funções para abrir e fechar a garra pneumática.

**Fechar:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: close_gripper
   :end-before: // DOC-END: close_gripper

**Abrir:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: open_gripper
   :end-before: // DOC-END: open_gripper

7. Planejamento de Movimento
----------------------------

Sistema híbrido que tenta movimentos cartesianos (linha reta) primeiro, e falhando isso, usa planejamento livre (RRTConnect).

**Cartesiano:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: attempt_cartesian_move
   :end-before: // DOC-END: attempt_cartesian_move

**Livre (RRT):**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: positions_for_arm
   :end-before: // DOC-END: positions_for_arm

8. Lógica Central: Pick, Place e Retry
--------------------------------------

Esta é a principal inovação da v1.1. A função implementa recursão em caso de falha.
Na versão anterior o nó enviava **false** como resposta da action, assim a BT tinha que reiniciar desnecessariamente.

**O Ciclo de Retry (Destacado abaixo):**

**1.** O robô tenta pegar.

**2.** Verifica sensor de força.

**3.** Se falhar: **Abre a garra**, **Recua**, **Lê nova pose**, e **Chama a função novamente**.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: calculate_global_pose
   :end-before: // DOC-END: calculate_global_pose
   :caption: Máquina de Estados com Retry
   :emphasize-lines: 108-119
   :linenos:

9. Gerenciamento de Colisões
----------------------------

Ajusta a ACM para permitir interações físicas controladas.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: set_collision_allowance
   :end-before: // DOC-END: set_collision_allowance

10. Serviços Externos
---------------------

Controla a física do objeto no simulador.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: send_request
   :end-before: // DOC-END: send_request

11. Atualização de Pose (Novo)
------------------------------

Callback essencial para o Retry. Mantém a posição do alvo atualizada.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: object_pose_callback
   :end-before: // DOC-END: object_pose_callback

12. Action Callbacks
--------------------

Gerenciamento do ciclo de vida da Action `PickObject`.

**Goal, Cancel, Accepted:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: handle_goal
   :end-before: // DOC-END: handle_accepted

**Execute (Thread):**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: execute
   :end-before: // DOC-END: execute

13. Callback do Sensor
----------------------

Buffer circular para leitura estável do sensor de força.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.1.cpp
   :language: cpp
   :start-after: // DOC-START: topic_callback
   :end-before: // DOC-END: topic_callback