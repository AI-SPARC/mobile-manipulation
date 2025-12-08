add_collision_impl.cpp (Atual v1.0)
=============================================

O nó ``AddCollision`` é responsável pela ponte entre a **Percepção** (Visão Computacional) e o **Planejamento de Movimento** (MoveIt 2).
Ele escuta detecções 3D em tempo real e insere, atualiza ou remove caixas de colisão na *Planning Scene* do MoveIt, garantindo que o braço robótico evite obstáculos dinâmicos.

Inicialização e Configuração
----------------------------

O nó carrega um arquivo YAML contendo regras de quais objetos devem ser considerados obstáculos e adiciona um plano de chão para segurança.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: AddCollision
   :end-before: // DOC-END: AddCollision

Filtragem de Objetos (YAML)
---------------------------

Para evitar poluir a cena com objetos irrelevantes, implementa-se um filtro de *Allowlist/Blocklist*.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: load_labels_from_yaml
   :end-before: // DOC-END: load_labels_from_yaml

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: is_authorized
   :end-before: // DOC-END: is_authorized

Atualização da Cena (MoveIt Interface)
--------------------------------------

Utiliza a `PlanningSceneInterface` do MoveIt para adicionar primitivas geométricas.

**Adição Inicial:**

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: add_collision_box
   :end-before: // DOC-END: add_collision_box

**Atualização Dinâmica (Tracking):**
Permite que o obstáculo se mova na cena virtual conforme ele se move no mundo real.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: move_collision_box
   :end-before: // DOC-END: move_collision_box

**Filtro de Ruído (Histerese):**
Para evitar que o objeto fique "vibrando" na cena devido a pequenos ruídos do sensor, aplicamos um limiar mínimo de movimento.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: is_significant_change
   :end-before: // DOC-END: is_significant_change

Lógica de Congelamento (Service)
--------------------------------

Quando o robô se aproxima para pegar um objeto, a atualização constante da posição do obstáculo pode atrapalhar o planejador (causando falhas de *Path Constraints*).
Este serviço permite que o `ServerNode` solicite o "congelamento" da posição de um objeto específico instantes antes do *Pick*.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: handleStopService
   :end-before: // DOC-END: handleStopService

Processamento de Detecções
--------------------------

O callback principal que orquestra tudo: recebe a mensagem, filtra, verifica autorização e decide se adiciona ou move a caixa.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/AddCollision/add_collision_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: detectionCallback
   :end-before: // DOC-END: detectionCallback