get_storage_info.cpp (Atual v1.0 - Composable Node)
========================================================

O ``StorageNode`` atua como o banco de dados central do sistema de manipulação. Ele mantém o registro de onde cada tipo de objeto deve ser guardado e o estado atual de ocupação de cada caixa ou estante.

Arquitetura de Composição
-------------------------

Este nó é implementado como um **Composable Node** (`rclcpp_components`).
Ele é instanciado dentro do mesmo processo do :doc:`../../task_planning/ServerNode/index`, permitindo que o planejador de tarefas consulte o banco de dados instantaneamente via ponteiros, sem o overhead de Serviços ROS.

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: StorageNode
   :end-before: // DOC-END: StorageNode

Lógica de Seleção (Best Storage)
--------------------------------

A função principal do nó é determinar o melhor destino para um objeto. O algoritmo considera:

1.  **Compatibilidade:** A caixa aceita este tipo de objeto? (Definido em YAML).
2.  **Capacidade:** A caixa ainda tem espaço livre?
3.  **Proximidade:** Qual das caixas válidas está mais perto do robô agora?

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: getBestStorage
   :end-before: // DOC-END: getBestStorage
   :linenos:

Gerenciamento de Estado
-----------------------

O nó mantém o estado interno de ocupação (índices da grade e contagem total) sincronizado.

**Atualização de Índices (Commit):**
Atualiza os ponteiros internos (i, j, k) para a próxima posição livre após um depósito bem-sucedido.

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: addNewIndexes
   :end-before: // DOC-END: addNewIndexes

**Controle de Ocupação:**
Incrementa ou decrementa o contador total de objetos.

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: incrementStorageCount
   :end-before: // DOC-END: incrementStorageCount

Carregamento de Dados (YAML)
----------------------------

O banco de dados é populado no início da execução a partir de arquivos de configuração.

**Regras Lógicas (O que vai onde):**

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: loadLabelToStorage
   :end-before: // DOC-END: loadLabelToStorage

**Definições Físicas (Onde ficam as caixas):**

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: loadStoragePoses
   :end-before: // DOC-END: loadStoragePoses

Cálculo de Limites
------------------

Para auxiliar o planejamento de movimento, o nó calcula a *Bounding Box* global da área de armazenamento, levando em conta a rotação da caixa no mundo.

.. literalinclude:: ../../../../../docs_central/legacy_src/storage_manager/GetStorageInfo/get_storage_info_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: calculateLimits
   :end-before: // DOC-END: calculateLimits