manipulation_impl.cpp (Legado v1.0)
=============================================

Esta página disseca a implementação da classe ``SimpleManipulation``, utilizada na versão 1.0 do projeto.
Este nó é responsável por controlar o braço robótico (Panda) utilizando a interface C++ do MoveIt 2, integrando percepção (YAML offsets) e execução segura.

.. warning::
   Este código é legado. A versão seguinte (v1.1) lida mais rapidamente e de forma mais simples com o robô caso a garra não pegue o objeto ou o objeto tenha mudado de posição. Este arquivo serve como referência de implementação direta em C++.

Demonstração
------------

Abaixo, um vídeo da execução desta lógica:

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc;">
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/manipulation/manipulation_impl_v1.0.webm" type="video/webm">
           Seu navegador não suporta vídeos HTML5.
       </video>
   </div>


---

1. Construtor e Inicialização
-----------------------------

O construtor é a parte mais crítica para a estabilidade do nó.
Diferente de nós simples, o MoveIt requer um **Executor Multi-Threaded**.

**Desafio:** O `MoveGroupInterface` precisa processar callbacks (como o estado das juntas) enquanto o construtor ainda está rodando. Se usarmos o mesmo nó/executor, ocorre um *Deadlock*.

**Solução:**
1.  Criamos um nó separado (`moveit_node_`) apenas para o MoveIt.
2.  Criamos um `executor_` separado.
3.  Lançamos uma `std::thread` dedicada para rodar esse executor.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: SimpleManipulation
   :end-before: // DOC-END: SimpleManipulation
   :caption: Inicialização Segura de Threads e Action Servers
   :linenos:

2. Destrutor (Cleanup)
----------------------

Ao desligar o nó, precisamos parar o executor e juntar a thread para evitar erros de memória (*Segmentation Fault*).

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: ~SimpleManipulation
   :end-before: // DOC-END: ~SimpleManipulation

3. Carregamento de Parâmetros (YAML)
------------------------------------

Esta função lê o arquivo YAML contendo os *offsets* de pega para cada objeto.
Isso permite calibrar a posição da mão em relação ao centro do objeto sem recompilar o código.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: loadLocationsFromYaml
   :end-before: // DOC-END: loadLocationsFromYaml
   :caption: Parser de YAML com conversão de Quaterniões

4. Inicialização do MoveGroup
-----------------------------

O MoveIt pode demorar para subir. Usamos um `Timer` no construtor para chamar esta função 1 segundo após o boot.
Aqui instanciamos as interfaces de controle do braço (`panda_arm`) e da garra (`hand`).

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: initMoveGroup
   :end-before: // DOC-END: initMoveGroup

5. Posição de Segurança (Ready)
-------------------------------

Função utilitária para levar o robô a uma posição conhecida ("Home"), definida por valores fixos de juntas. Essencial para resetar o estado após uma falha ou para que a base móvel possa andar com mais segurança evitando mais facilmente obstáculos.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: ready
   :end-before: // DOC-END: ready

6. Controle da Garra (End-Effector)
-----------------------------------

Funções simples que enviam comandos para as juntas dos dedos (`panda_finger_joint`).

**Fechar:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: close_gripper
   :end-before: // DOC-END: close_gripper

**Abrir:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: open_gripper
   :end-before: // DOC-END: open_gripper

7. Planejamento Cartesiano (Linha Reta)
---------------------------------------

O movimento cartesiano é preferido para "aproximação final" e "retirada" (lift), pois garante que a garra não bata nas laterais de uma prateleira.
A função tenta calcular uma linha reta com resolução de 1cm (`eef_step = 0.01`).

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: attempt_cartesian_move
   :end-before: // DOC-END: attempt_cartesian_move
   :caption: Algoritmo de Caminho Cartesiano

8. Planejamento Livre (RRTConnect)
----------------------------------

Quando o movimento não precisa ser reto (ex: ir de uma caixa para outra), usamos o planejador probabilístico `RRTConnect`. Ele desvia de obstáculos automaticamente.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: positions_for_arm
   :end-before: // DOC-END: positions_for_arm
   :caption: Planejamento Global com MoveIt

9. Lógica Central: Pick & Place
-------------------------------

Esta é a função mais complexa. Ela:
1.  Recebe a pose do objeto detectado.
2.  Aplica a transformação matemática usando o offset do YAML.
3.  Executa a máquina de estados finitos (Sequência de Ações).

**Sequência de Pick:**
* Aproximação
* Trava física do objeto (Service)
* Anexação virtual (MoveIt)
* Fechamento da garra
* Verificação do Sensor de Força
* Elevação (Lift)

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: calculate_global_pose
   :end-before: // DOC-END: calculate_global_pose
   :caption: Máquina de Estados de Manipulação
   :linenos:

10. Gerenciamento de Colisões (ACM)
-----------------------------------

Modifica a *Allowed Collision Matrix*. Isso é necessário na hora de colocar um objeto (Place) sobre uma mesa ou no chão, para que o MoveIt não aborte o plano achando que houve uma colisão.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: set_collision_allowance
   :end-before: // DOC-END: set_collision_allowance

11. Requisição de Serviços Externos
-----------------------------------

Comunica-se com o simulador ou sistema de física para "congelar" ou "descongelar" a física do objeto sendo manipulado.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: send_request
   :end-before: // DOC-END: send_request

12. Callbacks do Action Server
------------------------------

Aqui gerenciamos o ciclo de vida da Action ROS 2 (`PickObject`).

**Recebimento e Aceite:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: handle_goal
   :end-before: // DOC-END: handle_accepted

**Execução (Thread Separada):**
A função `execute` roda em background. Ela espera o MoveIt estar pronto e então chama a `calculate_global_pose`.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: execute
   :end-before: // DOC-END: execute

13. Callback do Sensor
----------------------

Armazena os dados do sensor de força em um buffer circular para filtragem de ruído.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/Manipulation/manipulation_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: topic_callback
   :end-before: // DOC-END: topic_callback