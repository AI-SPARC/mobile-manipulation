is_gripper_holding_impl.cpp (Atual v1.0 - Composable Node)
==========================================================

O nó ``IsGripperHolding`` é responsável pela validação física da apreensão de objetos.
Ele monitora os sensores de força/contato instalados nos dedos da garra robótica para determinar se um objeto foi capturado com sucesso ou se escorregou durante o transporte.

Integração com ServerNode (Composição)
--------------------------------------

Este nó foi projetado utilizando a arquitetura de **Componentes do ROS 2** (`rclcpp_components`).
Em vez de rodar como um executável isolado, ele é instanciado dentro do mesmo processo do :doc:`../../task_planning/ServerNode/index`.

**Vantagens da Composição:**

1.  **Comunicação Intra-Processo:** O ``ServerNode`` possui um ponteiro inteligente (`std::shared_ptr`) direto para a instância desta classe.
2.  **Latência Zero:** Quando a Behavior Tree precisa verificar se o objeto caiu, ela chama o método ``checkIsHolding()`` diretamente na memória RAM, sem serialização de mensagens ou atraso de rede (DDS) do ROS2.

Implementação
-------------

**Construtor e Registro:**
A macro `RCLCPP_COMPONENTS_REGISTER_NODE` no final do arquivo permite que a classe seja descoberta e carregada dinamicamente pelo container de componentes ou pelo nó principal.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/IsGripperHolding/is_gripper_holding_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: IsGripperHolding
   :end-before: // DOC-END: IsGripperHolding

**Lógica de Verificação (Thread-Safe):**
A função ``checkIsHolding`` implementa uma lógica de **Debounce** (filtragem digital). Em vez de confiar em uma única leitura do sensor (que pode conter ruído), analisa-se uma janela deslizante das últimas 10 leituras. O objeto é considerado "seguro" apenas se 90% das amostras estiverem acima do limiar de força.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/IsGripperHolding/is_gripper_holding_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: checkIsHolding
   :end-before: // DOC-END: checkIsHolding

**Aquisição de Dados:**
O callback do tópico mantém o buffer circular atualizado, descartando leituras antigas para manter a análise sempre no tempo presente. O uso de `std::mutex` é obrigatório, pois este callback roda na thread do Executor ROS, enquanto a verificação é chamada pela thread da Behavior Tree.

.. literalinclude:: ../../../../../docs_central/legacy_src/manipulation/IsGripperHolding/is_gripper_holding_impl_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: topic_callback
   :end-before: // DOC-END: topic_callback