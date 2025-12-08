organize.cpp (Atual v1.0 - Composable Node)
===========================================

O ``OrganizeNode`` é responsável pela inteligência geométrica de armazenamento. Ele não move o robô fisicamente, mas calcula as coordenadas exatas onde o robô deve colocar um objeto para otimizar o espaço dentro de uma caixa ou prateleira.

Arquitetura de Composição
-------------------------

Este nó é implementado como um **Composable Node** (`rclcpp_components`).
Em vez de se comunicar via tópicos ou serviços (que adicionariam latência e complexidade de serialização), ele é instanciado dentro do mesmo processo do :doc:`../../task_planning/ServerNode/index`.

Isso permite que o ``ServerNode`` chame o método ``placeObjectInBox`` diretamente, como se fosse uma função de biblioteca, garantindo performance máxima para o cálculo matemático.

.. literalinclude::  ../../../../../docs_central/legacy_src/storage_manager/Organize/organize_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: OrganizeNode
   :end-before: // DOC-END: OrganizeNode

Lógica de Empacotamento (Bin Packing)
-------------------------------------

O método principal, ``placeObjectInBox``, implementa um algoritmo de preenchimento de grade 3D (Grid Filling).

**Funcionamento:**
1.  **Grade Virtual:** Divide o volume da caixa de armazenamento em células baseadas no tamanho do objeto + margem de segurança (*padding*).
2.  **Cálculo Relativo:** Determina a posição :math:`(x, y, z)` relativa ao canto inferior esquerdo da caixa baseada nos índices atuais :math:`(i, j, k)`.
3.  **Transformação Espacial:** Aplica a rotação da caixa (Quaternion) para transformar a coordenada relativa em uma Pose Global no mundo.
4.  **Atualização de Índices:** Calcula qual será a próxima vaga livre, preenchendo primeiro o eixo X, depois Y, e por fim empilhando em Z.

**Implementação Detalhada:**

.. literalinclude::  ../../../../../docs_central/legacy_src/storage_manager/Organize/organize_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: placeObjectInBox
   :end-before: // DOC-END: placeObjectInBox
   :linenos:

Matemática de Transformação
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Para garantir que o objeto seja colocado corretamente mesmo se a caixa estiver torta, utilizamos matrizes de rotação.

Seja :math:`P_{center}` a pose da caixa e :math:`v_{rel}` o vetor calculado na grade:

.. math::

    P_{final} = P_{center} + (R_{box} \times v_{rel})

Onde :math:`R_{box}` é a matriz de rotação 3x3 derivada do quaternião da caixa. A orientação final do objeto é ajustada para manter o *Pitch* e *Roll* zerados (objeto em pé), mas herdando o *Yaw* da caixa para ficar alinhado.