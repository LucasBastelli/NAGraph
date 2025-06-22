// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

void bind_current_thread_to_cpu_list(const std::vector<int> &cpus) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int cpu : cpus) {
      CPU_SET(cpu, &cpuset);
  }
  pthread_t tid = pthread_self();
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
}


int64_t BUStepNuma(const WGraph &g, pvector<NodeID> &parent, Bitmap &front,
                   Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  
  #pragma omp parallel
  {
    // Bind uma única vez por thread
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int node_count = 2;
    int64_t my_awake_count = 0;
    
    static const std::vector<int> node0_cpus = {
      0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    
    int64_t start;
    if ((tid % node_count) == 0) {
      bind_current_thread_to_cpu_list(node0_cpus);
      start = tid; // Pares
    }
    else {
      bind_current_thread_to_cpu_list(node1_cpus);
      start = tid; // Ímpares
    }
    
    // Garantir que next.reset() foi concluído antes de prosseguir
    #pragma omp barrier
    
    // Loop manual com stride, cada thread fica no seu nó
    for (NodeID u = start; u < g.num_nodes(); u += numThreads) {
      if (parent[u] < 0) {
        for (NodeID v : g.in_neigh(u)) {
          if (front.get_bit(v)) {
            parent[u] = v;
            my_awake_count++;
            next.set_bit_atomic(u); // Usar versão atômica
            break;
          }
        }
      }
    }
    
    // Garantir que todas as threads terminaram de processar antes da redução
    #pragma omp barrier
    
    // Redução manual
    #pragma omp atomic
    awake_count += my_awake_count;
    
  } // fim do parallel
  
  return awake_count;
}

void QueueToBitmapNuma(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  bm.reset(); // Limpar antes de começar
  #pragma omp parallel
  {
    // Bind uma única vez por thread
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int node_count = 2;
    
    static const std::vector<int> node0_cpus = {
      0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    
    int64_t start;
    if ((tid % node_count) == 0) {
      bind_current_thread_to_cpu_list(node0_cpus);
      start = tid; // Pares
    }
    else {
      bind_current_thread_to_cpu_list(node1_cpus);
      start = tid; // Ímpares
    }
    
    // Garantir que reset foi concluído antes de prosseguir
    //#pragma omp barrier
    
    // Loop manual com stride através da queue
    int64_t queue_size = queue.end() - queue.begin();
    for (int64_t i = start; i < queue_size; i += numThreads) {
      auto q_iter = queue.begin() + i;
      NodeID u = *q_iter;
      bm.set_bit_atomic(u);
    }
    
    // Garantir que todas as threads terminaram antes de sair
    #pragma omp barrier
    
  } // fim do parallel
}

void BitmapToQueueNuma(const WGraph &g, const Bitmap &bm,
                       SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    
    // Bind uma única vez por thread
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int node_count = 2;
    
    static const std::vector<int> node0_cpus = {
      0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    
    int64_t start;
    if ((tid % node_count) == 0) {
      bind_current_thread_to_cpu_list(node0_cpus);
      start = tid; // Pares
    }
    else {
      bind_current_thread_to_cpu_list(node1_cpus);
      start = tid; // Ímpares
    }
    
    // Loop manual com stride
    for (NodeID n = start; n < g.num_nodes(); n += numThreads) {
      if (bm.get_bit(n))
        lqueue.push_back(n);
    }
    lqueue.flush();
    
    // Garantir que todas as threads terminaram de processar
    #pragma omp barrier
    
  } // fim do parallel
  queue.slide_window();
}


pvector<NodeID> InitParentNuma(const WGraph &g) {
  pvector<NodeID> parent(g.num_nodes());
  
  #pragma omp parallel
  {
    // Bind uma única vez por thread
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int node_count = 2;
    
    static const std::vector<int> node0_cpus = {
      0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    
    int64_t start;
    if ((tid % node_count) == 0) {
      bind_current_thread_to_cpu_list(node0_cpus);
      start = tid; // Pares
    }
    else {
      bind_current_thread_to_cpu_list(node1_cpus);
      start = tid; // Ímpares
    }
    
    // Loop manual com stride
    for (NodeID n = start; n < g.num_nodes(); n += numThreads) {
      parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
    }
    
    // Sincronização para garantir que todas as inicializações terminaram
    #pragma omp barrier
    
  } // fim do parallel
  
  return parent;
}


int64_t BUStep(const WGraph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    if (parent[u] < 0) {
      for (NodeID v : g.in_neigh(u)) {
        if (front.get_bit(v)) {
          parent[u] = v;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}


int64_t TDStep(const WGraph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count)
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
      for (NodeID v : g.out_neigh(u)) {
        NodeID curr_val = parent[v];
        if (curr_val < 0) {
          if (compare_and_swap(parent[v], curr_val, u)) {
            lqueue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const WGraph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for
    for (NodeID n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const WGraph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  return parent;
}

pvector<NodeID> DOBFSNuma(const WGraph &g, NodeID source, int alpha = 15,
                          int beta = 18) {
// PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParentNuma(g);
  t.Stop();
// PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmapNuma(queue, front));
// PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStepNuma(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
// PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueueNuma(g, front, queue));
// PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue); // Esta função precisa ser adaptada também
      queue.slide_window();
      t.Stop();
// PrintStep("td", t.Seconds(), queue.size());
    }
  }
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  
  return parent;
}


pvector<NodeID> DOBFS(const WGraph &g, NodeID source, int alpha = 15,
                      int beta = 18) {
//  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
//  PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
//      PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
//        PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
//      PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
//      PrintStep("td", t.Seconds(), queue.size());
    }
  }
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  return parent;
}


void PrintBFSStats(const WGraph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const WGraph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


#ifdef HASH_MODE
int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
    
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  SourcePicker<WGraph> sp(g, cli.start_vertex());
  using BFSFunc = std::function<pvector<NodeID>(const WGraph&)>;
  BFSFunc BFSBound;
  if (omp_get_max_threads() > 1) {
    BFSBound = [&sp] (const WGraph &g) { return DOBFSNuma(g, sp.PickNext()); };
    std::cout << "Running NUMA-aware BFS" << std::endl;
  } else {
    BFSBound = [&sp] (const WGraph &g) { return DOBFS(g, sp.PickNext()); };
    std::cout << "Running standard BFS" << std::endl;
  }
  SourcePicker<WGraph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const WGraph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}

#else
int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  SourcePicker<WGraph> sp(g, cli.start_vertex());
  auto BFSBound = [&sp] (const WGraph &g) { return DOBFS(g, sp.PickNext()); };
  SourcePicker<WGraph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const WGraph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
#endif