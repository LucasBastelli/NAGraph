import networkx as nx
import scipy

# --- PASSO 1: Carregar o grafo a partir de um arquivo de edge list ---
# Define o nome do arquivo que contém as arestas do grafo.
# Certifique-se de que o arquivo 'orkut.el' está na mesma pasta que o seu script,
# ou forneça o caminho completo para ele.
caminho_do_arquivo = '/home-ext/LucasIC/dynamicGraphs/data/output.el'

# Tenta carregar o grafo do arquivo.
try:
    # nx.read_edgelist lê o arquivo e cria o grafo.
    # Parâmetros importantes:
    #   - caminho_do_arquivo: O nome do arquivo a ser lido.
    #   - create_using=nx.DiGraph: Especifica que queremos criar um grafo direcionado (DiGraph),
    #     assim como no seu código original. Se as relações não fossem direcionadas,
    #     usaríamos nx.Graph.
    #   - nodetype=int: Define que os nós (vértices) devem ser tratados como números inteiros.
    G = nx.read_edgelist(caminho_do_arquivo, create_using=nx.DiGraph, nodetype=int)
    print(f"Grafo carregado com sucesso do arquivo '{caminho_do_arquivo}'.")
    print(f"O grafo tem {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

except FileNotFoundError:
    print(f"Erro: O arquivo '{caminho_do_arquivo}' não foi encontrado.")
    print("Por favor, verifique se o nome e o caminho para o arquivo estão corretos.")
    G = None

# --- PASSO 2: Calcular o PageRank (se o grafo foi carregado) ---
if G is not None:
    print("\nCalculando o PageRank... (Isso pode levar alguns minutos para grafos grandes)")

    # Calcula o PageRank usando os mesmos parâmetros do seu exemplo.
    # alpha=0.85 é o fator de amortecimento padrão.
    # max_iter=50 define um limite de iterações para o algoritmo convergir.
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)

    # --- PASSO 3: Imprimir os resultados ---
    print("\nResultados do PageRank (mostrando os 10 nós mais importantes):")

    # Ordena os nós pelo seu score de PageRank em ordem decrescente
    # e pega os 10 primeiros.
    top_10_nodes = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)[:10]

    # Imprime os 10 melhores resultados de forma formatada.
    for node, score in top_10_nodes:
        print(f"Nó {node} -> PageRank: {score:.6f}")