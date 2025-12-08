a_star.cpp (Atual v1.0)
=======================

O nó ``AStar`` implementa um planejador de caminho global determinístico. Ele recebe um mapa de ocupação (via nuvem de pontos) e calcula a rota mais curta entre a posição atual do robô e o destino, desviando de obstáculos estáticos e dinâmicos.

Inicialização e Parâmetros
--------------------------

O nó permite configurar a resolução do grid (tamanho da célula) e a **Distância de Segurança**.

.. important::
   **Distância de Segurança (`security_distance`):**
   Este parâmetro define um "raio de proteção" virtual ao redor do robô e do alvo. Ele é crucial para duas funções:
   1. Impedir que o robô trace rotas tangenciais que raspem em obstáculos.
   2. **Definir o ponto de parada:** O robô nunca navega até a coordenada exata do alvo (centro do objeto), mas sim até a borda definida por esta distância.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: AStar_Constructor
   :end-before: // DOC-END: AStar_Constructor

Estruturas de Dados
-------------------

Utiliza estruturas personalizadas e *hashes* especializados para permitir o uso de `std::pair<float, float>` e tuplas como chaves em mapas não ordenados (`unordered_map/set`), o que garante acesso O(1) aos obstáculos.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Structs
   :end-before: // DOC-END: Structs

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: hash_tuple_float
   :end-before: // DOC-END: hash_tuple_float

Mapeamento e Grid
-----------------

O nó não utiliza um *OccupancyGrid* tradicional (imagem). Em vez disso, converte a nuvem de pontos (`PointCloud2`) em um conjunto esparso de coordenadas discretas (`unordered_set`).

**Discretização:**
As coordenadas float são arredondadas para o múltiplo mais próximo da resolução (ex: 0.05m).

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: round_to_multiple
   :end-before: // DOC-END: round_to_multiple

**Callback de Obstáculos:**

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: topic_callback
   :end-before: // DOC-END: topic_callback

Recuperação de Alvo (BFS)
-------------------------

A robustez do planejador depende da capacidade de lidar com metas imprecisas. Frequentemente, o ponto de destino solicitado (Goal) ou a posição inicial do robô podem estar levemente dentro de um obstáculo devido a ruídos de sensores ou resolução do mapa.

Para resolver o problema de "Alvo Bloqueado", implementou-se uma busca em espiral baseada em **Breadth-First Search (BFS)** através da função ``find_nearest_free_point``.

**Demonstração da Expansão BFS:**

.. raw:: html

    <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização da recuperação de alvo via BFS (vídeo extremamente desacelerado).</em></p>
       
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/navigation/BFS_example_precise.webm" type="video/webm">
           Seu navegador não suporta a tag de vídeo.
       </video>

       <br><br>

       <p><em>Visualização do mapa real. O alvo é o cubo vermelho.</em></p>
       <img src="../../../_static/navigation/mapa_real_bfs.png" alt="Visualização do mapa real." style="width: 100%; height: auto;">
   </div>

**Implementação:**

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: find_nearest_free_point
   :end-before: // DOC-END: find_nearest_free_point

Execução do A* (Core)
---------------------

A função `run_a_star` orquestra a busca heurística no grafo.

**Características:**
1.  **Heurística Euclidiana:** Guia a expansão dos nós em direção ao alvo.
2.  **Verificação Periódica:** A cada N iterações, tenta traçar uma linha reta ("Theta* light") para encontrar atalhos imediatos.

**Visualização da Expansão do A*:**

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização da expansão dos nós (A*)</em></p>
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/navigation/A*.webm" type="video/webm">
           Seu navegador não suporta a tag de vídeo.
       </video>
   </div>

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: run_a_star
   :end-before: // DOC-END: run_a_star

Poda de Segurança (Stop Distance)
---------------------------------

Após o cálculo do caminho bruto, é necessário aplicar a lógica de parada segura. O caminho original gerado pelo A* vai até o centro do obstáculo alvo, o que causaria colisão.

A função ``store_edges_in_path`` inicia executando um algoritmo de corte:

1.  O algoritmo percorre o caminho de **trás para frente** (do Goal para o Start).
2.  Calcula a distância euclidiana de cada ponto do caminho em relação ao alvo original.
3.  Assim que encontra um ponto cuja distância é **maior ou igual** à ``security_distance``, este ponto é definido como o novo final da trajetória.
4.  Todos os pontos subsequentes (que estariam dentro da zona de perigo) são removidos da lista.

Isso garante que o robô pare exatamente na borda de segurança definida no arquivo de parâmetros.

Pós-Processamento e Suavização
------------------------------

Além da poda de segurança, a função ``store_edges_in_path`` também aplica o **Path Smoothing** (atalhos) para remover movimentos em zig-zag desnecessários, gerando uma trajetória mais natural.

**Comparativo: Caminho Bruto vs. Suavizado**

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Filtro de Suavização</em></p>
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/navigation/filter_60fps.webm" type="video/webm">
           Seu navegador não suporta a tag de vídeo.
       </video>
   </div>

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: store_edges_in_path
   :end-before: // DOC-END: store_edges_in_path

Execução e Recálculo Dinâmico
-----------------------------

A Action `Path` não termina assim que calcula o caminho. Ela entra em um loop de monitoramento contínuo.
Se um obstáculo novo aparecer em cima do caminho calculado, a flag `path_needs_calculation` é ativada, enviando um feedback `recalculating_path = true` para que o controlador pare o robô enquanto um novo trajeto é gerado.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStar/a_star_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: execute
   :end-before: // DOC-END: execute