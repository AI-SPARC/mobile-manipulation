import os

def renomear_arquivos_pcd():
    # Caminho da pasta
    diretorio = "/home/momesso/Downloads/PCDs"
    
    # Configurações do novo nome
    prefixo = "toma_"
    contador = 91
    extensao = ".pcd"

    # Verifica se o diretório existe
    if not os.path.exists(diretorio):
        print(f"Erro: O diretório {diretorio} não foi encontrado.")
        return

    # Lista todos os arquivos no diretório
    arquivos = os.listdir(diretorio)
    
    # Filtra apenas os arquivos que terminam com .pcd e ordena alfabeticamente
    pcds = sorted([f for f in arquivos if f.endswith(extensao)])

    if not pcds:
        print("Nenhum arquivo .pcd encontrado na pasta.")
        return

    print(f"Encontrados {len(pcds)} arquivos. Iniciando renomeação...\n")

    for arquivo_original in pcds:
        # Monta o novo nome: toma_91.pcd, toma_92.pcd...
        novo_nome = f"{prefixo}{contador}{extensao}"
        
        caminho_antigo = os.path.join(diretorio, arquivo_original)
        caminho_novo = os.path.join(diretorio, novo_nome)

        # Evita sobrescrever se o arquivo já tiver o nome exato (opcional, mas seguro)
        if caminho_antigo != caminho_novo:
            try:
                os.rename(caminho_antigo, caminho_novo)
                print(f"Renomeado: '{arquivo_original}' -> '{novo_nome}'")
            except OSError as e:
                print(f"Erro ao renomear '{arquivo_original}': {e}")
        else:
            print(f"Ignorado: '{arquivo_original}' já está com o nome correto.")

        # Incrementa o contador para o próximo arquivo (91 -> 92 -> 93...)
        contador += 1

    print("\nProcesso finalizado!")

if __name__ == "__main__":
    renomear_arquivos_pcd()