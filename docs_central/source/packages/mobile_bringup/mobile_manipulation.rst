Launch File Inteligente (Bringup)
=================================

O arquivo ``mobile_manipulation.launch.py`` é o ponto de entrada do sistema.
Diferente de launch files tradicionais que carregam todos os nós de uma vez, este script implementa uma lógica de **Carregamento Condicional Baseado na Behavior Tree**.

Funcionamento da Análise XML
----------------------------

Antes de iniciar qualquer nó, o script lê o arquivo `.xml` da Behavior Tree definido no pacote `task_planning`.
Ele percorre a árvore procurando por tags de **Action** (como `ComputePath`, `PickObject`) e identifica as dependências.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_bringup/mobile_manipulation.launch.py
   :language: python
   :start-after: # DOC-START: analyze_bt_xml
   :end-before: # DOC-END: analyze_bt_xml

Configuração de Nós Componíveis (ServerNode)
--------------------------------------------

O ``ServerNode`` é especial: ele não é apenas um nó, mas um **Container**. Dependendo das necessidades da árvore, ele pode instanciar internamente os componentes de *Storage*, *Organização* e *Gripper*.
O launch file passa argumentos (`--no-storage`, etc.) para controlar essa instanciação, economizando recursos se esses módulos não forem necessários.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_bringup/mobile_manipulation.launch.py
   :language: python
   :start-after: # DOC-START: ComposableNodesConfig
   :end-before: # DOC-END: ComposableNodesConfig

Carregamento Dinâmico de Processos
----------------------------------

Além dos componentes internos, o sistema decide quais processos externos (Executáveis ROS) devem ser iniciados.

* **Navegação:** Se a BT tiver `ComputePath`, carrega o mapa e o planejador (escolhendo entre A* ou D* dependendo do atributo `planner="d_star"` no XML).
* **Manipulação:** Se tiver `PickObject`, carrega o nó de manipulação e o `add_collision`.
* **Controle:** Se tiver `MapsTo`, carrega o controlador da base.

Se a missão for apenas "Organizar Estoque" (sem mover a base), os nós de navegação nem sequer são iniciados, economizando CPU e RAM.

.. literalinclude:: ../../../../../src/docs_central/legacy_src/mobile_bringup/mobile_manipulation.launch.py
   :language: python
   :start-after: # DOC-START: DynamicLoading
   :end-before: # DOC-END: DynamicLoading