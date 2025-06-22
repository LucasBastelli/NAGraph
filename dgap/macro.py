#!/usr/bin/env python3
import os
import subprocess

def main():
    # --- configurações fixas ---
    DATA_PATH       = '/home-ext/LucasIC/dynamicGraphs/data'
    PMEM_PATH       = '/mnt/nvram0'
    PMEM_PATH_NUMA  = '/mnt/nvram1'
    THREADS         = [2, 12, 24, 48]

    env = os.environ.copy()
    env['DATA_PATH']      = DATA_PATH
    env['PMEM_PATH']      = PMEM_PATH
    env['PMEM_PATH_NUMA'] = PMEM_PATH_NUMA

    for th in THREADS:
        print(f"\n=== Iniciando benchmark: OMP_NUM_THREADS={th} ===")
        env['OMP_NUM_THREADS'] = str(th)

        # remove arquivos antigos, se existirem
        for f in (os.path.join(PMEM_PATH,      'orkut1.pmem'),
                  os.path.join(PMEM_PATH_NUMA, 'orkut2.pmem')):
            if os.path.exists(f):
                print(f"Removendo: {f}")
                os.remove(f)
            else:
                print(f"Pulando remoção (não existe): {f}")

        # comando a rodar
        cmd = [
            './bc',
            '-B', os.path.join(DATA_PATH, 'OutputLive.el'),
            '-D', os.path.join(DATA_PATH, 'output.empty.el'),
            '-f', os.path.join(PMEM_PATH,      'orkut1.pmem'),
            '-m', os.path.join(PMEM_PATH_NUMA, 'orkut2.pmem'),
            '-r', '1',
            '-n', '10',
            '-a'
        ]
        print("Executando:", ' '.join(cmd))

        # executa e ignora o returncode != 0
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"⚠ Atenção: o processo saiu com código {result.returncode}, mas vamos continuar.")

        else:
            print(f"✔ Benchmark com {th} threads finalizou sem erros.")

    print("\nTodos os benchmarks foram executados (mesmo que alguns tenham retornado erro).")

if __name__ == '__main__':
    main()

