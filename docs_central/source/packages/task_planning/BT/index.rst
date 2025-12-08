Referência dos Nós da Behavior Tree
===================================

Esta seção documenta a biblioteca de nós (Nodes) desenvolvidos para a composição das árvores de comportamento do projeto.

.. note::
   **Mecanismo de Dados:** Todos os nós compartilham dados através da **Blackboard**. As portas de entrada (Input) e saída (Output) listadas abaixo referem-se às chaves (keys) manipuladas na memória compartilhada da árvore.

Índice Rápido
-------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Nome do Nó
     - Tipo
     - Descrição Resumida
   * - **ParallelAny**
     - Control
     - Executa filhos em paralelo com lógica OR (Short-circuit).
   * - **IsRobotNear**
     - Condition
     - Verifica distância euclidiana até um alvo.
   * - **IsGripperHoldingObject**
     - Condition
     - Verifica sensor de contato da garra.
   * - **DetectObject**
     - Action
     - Processa visão computacional e seleciona alvos.
   * - **ClearTarget**
     - Action
     - Reseta o alvo atual para permitir nova busca.
   * - **ComputePath**
     - Action
     - Solicita planejamento global de rota (A*).
   * - **NavigateTo**
     - Action
     - Executa a trajetória calculada (Controlador).
   * - **PickObject**
     - Action
     - Executa sequência de manipulação para pegar.
   * - **PlaceObject**
     - Action
     - Executa sequência de manipulação para largar.
   * - **GetStorageInfo**
     - Action
     - Consulta banco de dados de armazenamento.
   * - **ComputePoseToOrganize**
     - Action
     - Calcula pose de *Place* usando Bin Packing.
   * - **ComputePoseToStore**
     - Action
     - Calcula pose de *Place* simples (pilha).
   * - **IncrementOrganizedStorageIndexes**
     - Action
     - Confirma ocupação de espaço (Commit).
   * - **DecrementStorageCount**
     - Action
     - Libera espaço em caso de falha (Rollback).

---

Controle Customizado
--------------------

ParallelAny
^^^^^^^^^^^
* **Tipo:** Control Node
* **Natureza:** Síncrono

Diferente do nó `Parallel` padrão (que aguarda N sucessos ou N falhas), este nó implementa uma lógica de "curto-circuito" (OR lógico em paralelo).

* **Comportamento:**
    1. Executa **todos** os nós filhos simultaneamente a cada *tick*.
    2. Retorna **SUCCESS** imediatamente se *qualquer* filho retornar `SUCCESS`.
    3. Retorna **FAILURE** imediatamente se *qualquer* filho retornar `FAILURE`.
    4. Retorna **RUNNING** caso contrário.

* **Aplicação Principal:**
    Implementação de **condições de guarda** em paralelo.
    *Exemplo:* "Navegar até o ponto X" EM PARALELO COM "Monitorar Sensor de Queda". Se o sensor indicar queda (FAILURE), a navegação é abortada imediatamente.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_ParallelAny
   :end-before: // DOC-END: BT_ParallelAny
   :dedent: 8

---

Condições (Conditions)
----------------------

IsRobotNear
^^^^^^^^^^^
* **Tipo:** Condition Node
* **Natureza:** Síncrono

Verifica se a distância euclidiana entre a base do robô e um alvo está dentro de limites aceitáveis.

* **Portas de Entrada (Input):**
    * ``target`` (Pose): A pose 3D do objetivo.
    * ``max_dist`` (double): Distância máxima permitida (Default: 0.5m).
    * ``min_dist`` (double): Distância mínima permitida (Default: 0.35m).
* **Portas de Saída (Output):**
    * ``adjustment_pose`` (Pose): A mesma pose do alvo, repassada para permitir ajustes de navegação caso a condição falhe.
* **Lógica:**
    Após calcular a distância entre a base do robô e o alo marcado, retorna **SUCCESS** se a distância estiver no intervalo [min, max].

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_IsRobotNear
   :end-before: // DOC-END: BT_IsRobotNear
   :dedent: 8

IsGripperHoldingObject
^^^^^^^^^^^^^^^^^^^^^^
* **Tipo:** Condition Node
* **Natureza:** Síncrono (Leitura de Memória)

Verifica o estado físico do efetuador final (garra) para garantir que o objeto ainda está seguro.

* **Lógica:**
    Consulta o nó `gripper_monitor_node` diretamente via ponteiro compartilhado (sem latência de tópicos).
    * **SUCCESS:** Se o sensor indicar pressão/contato.
    * **FAILURE:** Se a garra estiver vazia.
    * **Efeito Colateral:** Se retornar falha, aciona o cancelamento de emergência do controlador de navegação.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_IsGripperHoldingObject
   :end-before: // DOC-END: BT_IsGripperHoldingObject
   :dedent: 8

---

Percepção e Contexto
--------------------

DetectObject
^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono

Gerencia a triagem e seleção de alvos a partir do fluxo de dados do sistema de visão computacional.

* **Portas de Saída (Output):**
    * ``output_pose`` (Pose): Posição espacial do objeto selecionado.
    * ``output_id`` (string): Identificador único do objeto (ex: "box_42").
    * ``object_size`` (Vector3): Dimensões da Bounding Box.
* **Lógica (Latch):**
    1. **Modo Rastreamento:** Se já existe um alvo ativo (`current_target_id_`), retorna seus dados atuais (**SUCCESS**), ignorando outros objetos.
    2. **Modo Busca:** Se não há alvo, verifica a fila de novas detecções. Aplica filtros de classe permitida e objetos já coletados.
    3. Se um novo objeto válido for encontrado, ele é promovido a alvo atual.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_DetectObject
   :end-before: // DOC-END: BT_DetectObject
   :dedent: 8

ClearTarget
^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono

Reseta o estado de atenção do robô.

* **Lógica:**
    Limpa a variável interna `current_target_id_`. Isso permite que a ação `DetectObject` aceite novas detecções na próxima iteração da árvore. Deve ser chamada obrigatoriamente após a conclusão (sucesso ou falha) de uma tarefa de manipulação.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_ClearTarget
   :end-before: // DOC-END: BT_ClearTarget
   :dedent: 8

---

Navegação (Navigation Stack)
----------------------------

ComputePath
^^^^^^^^^^^
* **Tipo:** Action Node (Async)
* **Natureza:** Assíncrono (Action Client)

Solicita o cálculo de uma rota livre de obstáculos ao planejador global.

* **Portas de Entrada (Input):**
    * ``target`` (Pose): O destino final desejado.
    * ``planner`` (string): ID do algoritmo de planejamento (opcional).
* **Comportamento:**
    Envia um *Goal* para o Action Server de Path Planning (ex: A* ou Nav2 Planner).
    * **RUNNING:** Enquanto o cálculo está sendo processado.
    * **SUCCESS:** Quando um caminho válido é recebido e armazenado na variável interna `last_calculated_path_`.
    * **FAILURE:** Se o destino for inalcançável.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_ComputePath
   :end-before: // DOC-END: BT_ComputePath
   :dedent: 8

NavigateTo
^^^^^^^^^^
* **Tipo:** Action Node (Async)
* **Natureza:** Assíncrono (Action Client)

Executa a movimentação física do robô seguindo o caminho previamente calculado.

* **Pré-requisito:** Deve ser precedido por um `ComputePath` bem-sucedido.
* **Comportamento:**
    Envia o caminho (Path) para o Action Server do Controlador (ex: Pure Pursuit).
    * **RUNNING:** Enquanto o robô se move.
    * **SUCCESS:** Quando a tolerância do objetivo é atingida.
    * **FAILURE:** Se o robô colidir, travar ou o controlador reportar erro.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_NavigateTo
   :end-before: // DOC-END: BT_NavigateTo
   :dedent: 8

---

Manipulação (MoveIt Integration)
--------------------------------

PickObject
^^^^^^^^^^
* **Tipo:** Action Node (Async)
* **Natureza:** Assíncrono (Action Client)

Comanda o braço robótico para realizar a sequência de apreensão.

* **Portas de Entrada (Input):**
    * ``id`` (string): ID do objeto (usado para o MoveIt anexar o objeto à garra virtualmente na Collision Matrix).
    * ``pose`` (Pose): A pose alvo do objeto.
* **Comportamento:**
    Envia um *Goal* para o servidor de manipulação com a flag `pick = true`. O servidor executa: Aproximação -> Fechamento -> Validação de Sensor -> Elevação -> (Retry se necessário).

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_PickObject
   :end-before: // DOC-END: BT_PickObject
   :dedent: 8

PlaceObject
^^^^^^^^^^^
* **Tipo:** Action Node (Async)
* **Natureza:** Assíncrono (Action Client)

Comanda o braço robótico para depositar o objeto.

* **Portas de Entrada (Input):**
    * ``pose`` (Pose): A posição final onde o objeto deve ser deixado.
    * ``limits`` (vector): Limites da área de deposição (opcional).
* **Comportamento:**
    Envia um *Goal* para o servidor de manipulação com a flag `pick = false`. O servidor executa: Aproximação -> Abertura -> Desanexação (MoveIt) -> Recuo.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_PlaceObject
   :end-before: // DOC-END: BT_PlaceObject
   :dedent: 8

---

Gerenciamento de Estoque (Storage)
----------------------------------

GetStorageInfo
^^^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono (Service Call Wrapper)

Consulta o banco de dados de armazenamento para determinar o melhor destino para um objeto.

* **Portas de Entrada (Input):**
    * ``object_id`` (string): ID ou classe do objeto recolhido.
* **Portas de Saída (Output):**
    * ``storage_pose`` (Pose): Pose base da caixa/estante selecionada.
    * ``storage_size`` (Vector3): Dimensões da caixa.
    * ``storage_limits`` (vector): Restrições geométricas.
    * ``indexes`` (vector<int>): Vetor [i, j, k] indicando a próxima posição livre na grade da caixa.
* **Lógica:**
    Interroga o `StorageNode`. Retorna **FAILURE** se não houver espaço disponível.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_GetStorageInfo
   :end-before: // DOC-END: BT_GetStorageInfo
   :dedent: 8

ComputePoseToOrganize
^^^^^^^^^^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono (Algoritmo Geométrico)

Calcula a pose exata de deposição utilizando algoritmo de empacotamento (Bin Packing) para organizar itens lado a lado.

* **Portas de Entrada (Input):**
    * ``storage_pose``, ``storage_size``, ``object_size``, ``indexes``: Dados geométricos.
    * ``object_padding`` (float): Margem de segurança entre objetos.
    * ``z_lift_offset`` (float): Altura de aproximação segura.
* **Portas de Saída (Output):**
    * ``output_final_pose`` (Pose): A pose exata (x, y, z, orientação) para o *Place*.
    * ``new_indexes`` (vector<int>): Os índices atualizados após a inserção virtual.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_ComputePoseToOrganize
   :end-before: // DOC-END: BT_ComputePoseToOrganize
   :dedent: 8

ComputePoseToStore
^^^^^^^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono

Cálculo simplificado para armazenamento (ex: empilhamento vertical simples ou deposição no centro), usado quando organização detalhada não é necessária.

* **Portas de Entrada (Input):** ``storage_pose``, ``storage_size``, ``z_lift_offset``.
* **Portas de Saída (Output):** ``output_final_pose``.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_ComputePoseToStore
   :end-before: // DOC-END: BT_ComputePoseToStore
   :dedent: 8

IncrementOrganizedStorageIndexes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono

Realiza a **Persistência de dados (Commit)**.

* **Portas de Entrada (Input):** ``storage_id``, ``new_indexes``.
* **Lógica:**
    Atualiza o banco de dados do `StorageNode` marcando a posição calculada como "Ocupada". Deve ser executado apenas após o sucesso da ação `PlaceObject`.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_IncrementOrganizedStorageIndexes
   :end-before: // DOC-END: BT_IncrementOrganizedStorageIndexes
   :dedent: 8

DecrementStorageCount
^^^^^^^^^^^^^^^^^^^^^
* **Tipo:** Action Node (Simple)
* **Natureza:** Síncrono

Realiza a **Recuperação de falha (Rollback)**.

* **Portas de Entrada (Input):** ``storage_id``.
* **Lógica:**
    Caso a operação de `PlaceObject` falhe, esta ação é acionada para informar o `StorageNode` que o espaço reservado não foi utilizado, liberando-o para tentativas futuras.

**Implementação:**

.. literalinclude:: ../../../../legacy_src/task_planning/server_node_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: BT_DecrementStorageCount
   :end-before: // DOC-END: BT_DecrementStorageCount
   :dedent: 8