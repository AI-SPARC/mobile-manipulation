obstacle_graph_with_occupancy_grid.cpp (Atual v1.0)
===================================================

O nó ``OccupancyGridLoader`` é responsável pelo processamento do mapa estático do ambiente e por disponibilizá-lo para os planejadores globais (A* e D*).

Diferentemente dos nós de navegação padrão que realizam a leitura direta do mapa, a arquitetura implementada converte o mapa de pixels (imagem) em uma **Nuvem de Pontos Discreta**, aplicando-se uma **Margem de Segurança (Inflação)** pré-calculada. Tal abordagem simplifica o funcionamento dos planejadores, sendo necessário apenas verificar se uma coordenada consta no conjunto de obstáculos.

Inicialização e Parâmetros
--------------------------

O nó realiza o carregamento de dois arquivos principais:
1.  **YAML de Configuração:** Metadados do mapa (resolução, origem).
2.  **Imagem (PGM/PNG):** A representação visual da ocupação.

Adicionalmente, são definidas a resolução interna do grafo e a distância de segurança.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Constructor
   :end-before: // DOC-END: Constructor

Estruturas de Dados (Hash Otimizado)
------------------------------------

Visando garantir performance máxima (O(1)) na verificação de colisões durante o planejamento de caminho, utiliza-se um `std::unordered_set` com um *Hasher* customizado para pares de coordenadas float.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: PairHash
   :end-before: // DOC-END: PairHash

Funções Auxiliares (Discretização)
----------------------------------

Os planejadores operam sobre um grid discreto. As funções auxiliares asseguram que as coordenadas contínuas sejam "encaixadas" corretamente nos nós desse grid.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Helpers
   :end-before: // DOC-END: Helpers

Leitura do Mapa (OpenCV + YAML)
-------------------------------

Esta função executa a fusão de dados: a geometria é lida do arquivo de imagem e projetada no sistema de coordenadas do mundo real, utilizando-se a origem e a resolução definidas no YAML.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: loadOccupancyGrid
   :end-before: // DOC-END: loadOccupancyGrid

Inflação de Obstáculos (Safety Padding)
---------------------------------------

Trata-se da lógica central de segurança. Dado que o robô possui largura e não é um ponto adimensional, expandem-se matematicamente todos os obstáculos para evitar colisões com as paredes. O algoritmo gera camadas quadradas concêntricas ao redor de cada ponto ocupado até que se atinja a `max_security_distance`.

Dessa forma, o planejador A* pode tratar o robô como um ponto único, uma vez que o espaço físico ocupado pelo robô já foi "subtraído" do espaço livre no mapa.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: createGraphFromPointCloud
   :end-before: // DOC-END: createGraphFromPointCloud

Publicação (PointCloud2)
------------------------

O resultado final consiste em uma nuvem de pontos densa contendo todas as coordenadas proibidas. Esta nuvem é publicada no tópico `/obstacles_vertices`.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/ObstacleGraph/obstacle_graph_with_occupancy_grid_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: publishPointCloud
   :end-before: // DOC-END: publishPointCloud