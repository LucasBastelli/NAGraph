import os
import subprocess

# --- Configurações ---
# Define as variáveis de ambiente para os caminhos dos dados e do PMEM.
DATA_PATH = "/home-ext/LucasIC/dynamicGraphs/data"
PMEM_PATH = "/mnt/nvram0"
PMEM_PATH_NUMA = "/mnt/nvram1" # Novo caminho para NUMA

# Lista dos benchmarks que você deseja executar.
BENCHMARKS = ["pr", "bc", "bfs", "cc_sv"]

# Lista do número de threads para usar em cada execução.
THREAD_COUNTS = [2, 12, 24, 48]

# Nomes dos arquivos de banco de dados a serem removidos.
DB_FILE_1_NAME = "orkut1.pmem"
DB_FILE_2_NAME = "orkut2.pmem"
DB_FILE_1_PATH = os.path.join(PMEM_PATH, DB_FILE_1_NAME)
DB_FILE_2_PATH = os.path.join(PMEM_PATH_NUMA, DB_FILE_2_NAME)


# --- Lógica do Script ---

def run_benchmarks():
    """
    Função principal que executa a suíte de benchmarks para o framework com NUMA.
    """
    print("--- Iniciando a execução dos Benchmarks (Framework com NUMA) ---")
    
    # Define as variáveis de ambiente que serão usadas pelos subrocessos.
    env = os.environ.copy()
    env["DATA_PATH"] = DATA_PATH
    env["PMEM_PATH"] = PMEM_PATH
    env["PMEM_PATH_NUMA"] = PMEM_PATH_NUMA # Adiciona a nova variável ao ambiente

    # Itera sobre cada benchmark na lista.
    for benchmark in BENCHMARKS:
        # Itera sobre cada contagem de threads.
        for threads in THREAD_COUNTS:
            print("-" * 50)
            print(f"Executando benchmark: '{benchmark}' com {threads} threads")
            
            # 1. Remover os arquivos de banco de dados anteriores.
            try:
                # Remove o primeiro arquivo se ele existir.
                if os.path.exists(DB_FILE_1_PATH):
                    os.remove(DB_FILE_1_PATH)
                    print(f"Arquivo '{DB_FILE_1_PATH}' removido com sucesso.")
                else:
                    print(f"Arquivo '{DB_FILE_1_PATH}' não encontrado. Prosseguindo.")
                
                # Remove o segundo arquivo se ele existir.
                if os.path.exists(DB_FILE_2_PATH):
                    os.remove(DB_FILE_2_PATH)
                    print(f"Arquivo '{DB_FILE_2_PATH}' removido com sucesso.")
                else:
                    print(f"Arquivo '{DB_FILE_2_PATH}' não encontrado. Prosseguindo.")
            except OSError as e:
                print(f"Erro ao remover arquivos de banco de dados: {e}")
                # Pula para a próxima iteração se não conseguir remover os arquivos.
                continue

            # 2. Definir a variável de ambiente OMP_NUM_THREADS.
            env["OMP_NUM_THREADS"] = str(threads)
            print(f"Exportado: OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")

            # 3. Montar e executar o comando para o novo framework.
            command = [
                f"./{benchmark}",
                "-B", f"{DATA_PATH}/output.el",
                "-D", f"{DATA_PATH}/output.empty.el",
                "-f", DB_FILE_1_PATH,
                "-m", DB_FILE_2_PATH, # Novo parâmetro com o segundo arquivo
                "-r", "1",
                "-n", "10", # Parâmetro -n alterado para 5
                "-a"
            ]
            
            print(f"Comando a ser executado: {' '.join(command)}")
            print(f"\n--- Saída para '{benchmark}' com {threads} threads ---")
            
            try:
                # O subprocesso herda as variáveis de ambiente definidas em 'env'.
                subprocess.run(
                    command, 
                    env=env, 
                    check=True
                )
                print(f"\nExecução de '{benchmark}' com {threads} threads concluída com sucesso.")

            except FileNotFoundError:
                print(f"Erro: O executável './{benchmark}' não foi encontrado.")
                print("Certifique-se de que o script está no mesmo diretório que os executáveis dos benchmarks.")
            except subprocess.CalledProcessError as e:
                print(f"\nErro ao executar o comando (código de saída: {e.returncode}).")
            except Exception as e:
                print(f"Ocorreu um erro inesperado: {e}")

    print("-" * 50)
    print("--- Todos os benchmarks foram concluídos. ---")

if __name__ == "__main__":
    # Garante que o script seja executado apenas quando chamado diretamente.
    run_benchmarks()

