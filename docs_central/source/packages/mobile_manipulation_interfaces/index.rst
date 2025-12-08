Mobile Manipulation Interfaces
===============================

O pacote ``mobile_manipulation_interfaces`` define os tipos de mensagens customizados (Actions e Services) utilizados para a comunicação entre os nós do sistema.
Estas definições garantem que o **Task Planner** (ServerNode), a **Navegação** e a **Manipulação** falem a mesma língua.

.. note::
   Os arquivos de definição fonte (`.action` e `.srv`) estão localizados na pasta de legado para referência.

Actions (Ações Assíncronas)
---------------------------

As Actions são utilizadas para tarefas que demandam tempo para serem concluídas e que podem necessitar de feedback contínuo ou cancelamento.

PickObject
^^^^^^^^^^
**Arquivo:** ``PickObject.action``

Controla o fluxo de manipulação do braço robótico.

* **Goal (Meta):**
    * ``obstacle_id`` (string): O ID do objeto na cena (para o MoveIt gerenciar colisões).
    * ``pick`` (bool): ``true`` para pegar o objeto, ``false`` para largar.
    * ``pose`` (geometry_msgs/Pose): A posição e orientação alvo para o efetuador final.
* **Result (Resultado):**
    * ``success`` (bool): Indica se a operação física foi bem-sucedida.
* **Feedback:**
    * Vazio nesta implementação.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_manipulation_interfaces/action/PickObject.action
   :language: yaml
   :caption: Definição de PickObject.action

Path
^^^^
**Arquivo:** ``Path.action``

Solicita ao planejador global (como A* ou D*) o cálculo de uma rota livre de obstáculos.

* **Goal (Meta):**
    * ``pose`` (geometry_msgs/Pose): O destino final desejado no mapa.
* **Result (Resultado):**
    * ``success`` (bool): Se um caminho válido foi encontrado.
* **Feedback:**
    * ``recalculating_path`` (bool): Flag indicando se o planejador está ajustando a rota dinamicamente.
    * ``path`` (nav_msgs/Path): O vetor de poses que compõe o caminho atualizado.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_manipulation_interfaces/action/Path.action
   :language: yaml
   :caption: Definição de Path.action

Controller
^^^^^^^^^^
**Arquivo:** ``Controller.action``

Envia uma trajetória calculada para o controlador local (como Pure Pursuit) executar a movimentação dos motores.

* **Goal (Meta):**
    * ``path`` (nav_msgs/Path): O caminho completo que o robô deve seguir.
* **Result (Resultado):**
    * ``success`` (bool): Se o robô chegou ao destino com a tolerância adequada.
* **Feedback:**
    * Vazio nesta implementação.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_manipulation_interfaces/action/Controller.action
   :language: yaml
   :caption: Definição de Controller.action

---

Services (Serviços Síncronos)
-----------------------------


MobileObjectCollision
^^^^^^^^^^^^^^^^^^^^^
**Arquivo:** ``MobileObjectCollision.srv``

Utilizado principalmente em simulação ou com *Physics Engines* para "grudar" o objeto na garra ou alterar suas propriedades de colisão.

* **Request (Pedido):**
    * ``obstacle_id`` (string): O nome do objeto na simulação.
    * ``activate_movement`` (bool): Flag para ativar/desativar a física ou colisão do objeto.
* **Response (Resposta):**
    * ``success`` (bool): Confirmação da operação.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_manipulation_interfaces/srv/MobileObjectCollision.srv
   :language: yaml
   :caption: Definição de MobileObjectCollision.srv