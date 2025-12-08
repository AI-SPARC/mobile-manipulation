a_star_splicing.cpp (Legado v1.0 - Planejador Dinâmico)
=======================================================

O nó ``AStarSplicing``  implementa um planejador de caminho capaz de reagir a mudanças no ambiente. Diferente do A* estático, ele possui uma lógica de **Replanejamento Local**, permitindo que o robô contorne obstáculos novos sem descartar todo o progresso da rota anterior.

.. warning::
   Este código é legado. Tem um erro que não quero resolver porque não é necessário agora e tem coisas mais importantes para fazer no código.
   


Inicialização e Parâmetros
--------------------------

Assim como o A*, este nó configura a resolução do grid (tamanho da célula) e a **Distância de Segurança**.

.. important::
   **Distância de Segurança (`security_distance`):**
   Este parâmetro define um "raio de proteção" virtual ao redor do robô e do alvo. Ele é crucial para duas funções:
   1. Impedir que o robô trace rotas tangenciais que raspem em obstáculos.
   2. **Definir o ponto de parada:** O robô nunca navega até a coordenada exata do alvo (centro do objeto), mas sim até a borda definida por esta distância.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: AStarSplicing_Constructor
   :end-before: // DOC-END: AStarSplicing_Constructor

Estruturas de Dados
-------------------

Utiliza as mesmas estruturas otimizadas de hash espacial para acesso O(1) aos obstáculos.

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: Structs
   :end-before: // DOC-END: Structs

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: hash_tuple_float
   :end-before: // DOC-END: hash_tuple_float

Mapeamento e Grid
-----------------

O nó converte a nuvem de pontos dinâmica em um mapa de custos discreto (`unordered_set`).

**Discretização:**
As coordenadas float são arredondadas para o múltiplo mais próximo da resolução (ex: 0.05m).

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: round_to_multiple
   :end-before: // DOC-END: round_to_multiple

**Callback de Obstáculos:**

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: topic_callback
   :end-before: // DOC-END: topic_callback

Recuperação de Alvo (BFS)
-------------------------

Antes de qualquer planejamento, garante-se que os pontos de início e fim são válidos usando a busca em largura.

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

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: find_nearest_free_point
   :end-before: // DOC-END: find_nearest_free_point

Motor de Busca (A*)
-------------------

O nó utiliza o algoritmo A* como "motor" para calcular os segmentos de rota. Ele é usado tanto para o caminho inicial quanto para calcular os pequenos desvios (patches) durante a execução.

**Características:**
1.  **Heurística Euclidiana:** Guia a expansão dos nós em direção ao alvo.
2.  **Verificação Periódica:** A cada N iterações, tenta traçar uma linha reta ("Theta* light") para encontrar atalhos imediatos.

**Visualização da Expansão do Motor de Busca:**

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização da expansão dos nós</em></p>
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/navigation/A*.webm" type="video/webm">
           Seu navegador não suporta a tag de vídeo.
       </video>
   </div>

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: run_a_star
   :end-before: // DOC-END: run_a_star

Poda de Segurança (Stop Distance)
---------------------------------

Após o cálculo do caminho bruto, é necessário aplicar a lógica de parada segura. O caminho original vai até o centro do obstáculo alvo, o que causaria colisão.

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

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: store_edges_in_path
   :end-before: // DOC-END: store_edges_in_path

Execução Dinâmica: Path Splicing
--------------------------------

Esta é a principal inovação deste nó em relação ao A* padrão.

Em vez de descartar todo o caminho quando um obstáculo aparece, o nó executa um **Reparo Local (Splicing)**:

1.  **Monitoramento:** O robô segue o caminho original (`previousPath`).
2.  **Detecção de Ruptura:** O algoritmo varre o caminho à frente. Se encontra um obstáculo novo bloqueando o trajeto, identifica o índice do bloqueio ($m$).
3.  **Cálculo de Pontes:**
    * Identifica um ponto seguro *antes* do bloqueio.
    * Identifica um ponto seguro *depois* do bloqueio.
4.  **Emenda:** Executa o motor de busca apenas entre esses dois pontos.
5.  **Fusão:** O novo segmento curto é inserido no caminho original, substituindo a parte bloqueada.

**Demonstração do Replanejamento Dinâmico:**

.. raw:: html

   <div style="text-align: center; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
       <p><em>Visualização do Path Splicing (Desvio Dinâmico)</em></p>
       <video width="100%" height="auto" controls autoplay loop muted>
           <source src="../../../_static/navigation/a_star_splicing.webm" type="video/webm">
           Seu navegador não suporta a tag de vídeo.
       </video>
   </div>

.. literalinclude:: ../../../../../docs_central/legacy_src/navigation/AStarSplicing/a_star_splicing_v1.0.cpp
   :language: cpp
   :start-after: // DOC-START: execute
   :end-before: // DOC-END: execute