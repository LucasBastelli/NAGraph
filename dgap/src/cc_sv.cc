// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"

/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Author: Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Shiloach-Vishkin [2] algorithm with
implementation optimizations from Bader et al. [1]. Michael Sutton contributed
a fix for directed graphs using the min-max swap from [3], and it also produces
more consistent performance for undirected graphs.

[1] David A Bader, Guojing Cong, and John Feo. "On the architectural
    requirements for efficient execution of graph algorithms." International
    Conference on Parallel Processing, Jul 2005.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

[3] Kishore Kothapalli, Jyothish Soman, and P. J. Narayanan. "Fast GPU
    algorithms for graph connectivity." Workshop on Large Scale Parallel
    Processing, 2010.
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


pvector<NodeID> ShiloachVishkinNUMA(const WGraph &g) {
  pvector<NodeID> comp(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    comp[n] = n;
  bool change = true;
  int num_iter = 0;
  while (change) {
    change = false;
    num_iter++;
    // note: this gives better scaleup performance
    // #pragma omp parallel for schedule(dynamic, 64)
    #pragma omp parallel
    {
        // Bind uma única vez por thread
      int tid        = omp_get_thread_num();
      int numThreads = omp_get_num_threads();
      int n0         = numThreads/2;
      int node_count = 2;
      //long int my_count = 0;
      

      static const std::vector<int> node0_cpus = {
        0,1,2,3,4,5,6,7,8,9,10,11,
        24,25,26,27,28,29,30,31,32,33,34,35
      };
      static const std::vector<int> node1_cpus = {
        12,13,14,15,16,17,18,19,20,21,22,23,
        36,37,38,39,40,41,42,43,44,45,46,47
      };
      int64_t start;
      if ((tid%node_count) == 0){
        bind_current_thread_to_cpu_list(node0_cpus);
        start = tid; //Pares
      }
      else{
        bind_current_thread_to_cpu_list(node1_cpus);
        start = tid;//Impares
      }
      for (NodeID u=start; u < g.num_nodes(); u+=numThreads) {
        for (NodeID v : g.out_neigh(u)) {
          NodeID comp_u = comp[u];
          NodeID comp_v = comp[v];
          if (comp_u == comp_v) continue;
          // Hooking condition so lower component ID wins independent of direction
          NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
          NodeID low_comp = comp_u + (comp_v - high_comp);
          if (high_comp == comp[high_comp]) {
            change = true;
            comp[high_comp] = low_comp;
          }
        }
      }
      for (NodeID n=start; n < g.num_nodes(); n+=numThreads) {
        while (comp[n] != comp[comp[n]]) {
          comp[n] = comp[comp[n]];
        }
      }
    }
  }
  cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
  return comp;
}


// The hooking condition (comp_u < comp_v) may not coincide with the edge's
// direction, so we use a min-max swap such that lower component IDs propagate
// independent of the edge's direction.
pvector<NodeID> ShiloachVishkin(const WGraph &g) {
  pvector<NodeID> comp(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    comp[n] = n;
  bool change = true;
  int num_iter = 0;
  while (change) {
    change = false;
    num_iter++;
    // note: this gives better scaleup performance
    // #pragma omp parallel for schedule(dynamic, 64)
    #pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u)) {
        NodeID comp_u = comp[u];
        NodeID comp_v = comp[v];
        if (comp_u == comp_v) continue;
        // Hooking condition so lower component ID wins independent of direction
        NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
        NodeID low_comp = comp_u + (comp_v - high_comp);
        if (high_comp == comp[high_comp]) {
          change = true;
          comp[high_comp] = low_comp;
        }
      }
    }
    #pragma omp parallel for
    for (NodeID n=0; n < g.num_nodes(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
  return comp;
}


void PrintCompStats(const WGraph &g, const pvector<NodeID> &comp) {
  cout << endl;
  unordered_map<NodeID, NodeID> count;
  for (NodeID comp_i : comp)
    count[comp_i] += 1;
  int k = 5;
  vector<pair<NodeID, NodeID>> count_vector;
  count_vector.reserve(count.size());
  for (auto kvp : count)
    count_vector.push_back(kvp);
  vector<pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout << k << " biggest clusters" << endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
  cout << "There are " << count.size() << " components" << endl;
}


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool CCVerifier(const WGraph &g, const pvector<NodeID> &comp) {
  unordered_map<NodeID, NodeID> label_to_source;
  for (NodeID n : g.vertices())
    label_to_source[comp[n]] = n;
  Bitmap visited(g.num_nodes());
  visited.reset();
  vector<NodeID> frontier;
  frontier.reserve(g.num_nodes());
  for (auto label_source_pair : label_to_source) {
    NodeID curr_label = label_source_pair.first;
    NodeID source = label_source_pair.second;
    frontier.clear();
    frontier.push_back(source);
    visited.set_bit(source);
    for (auto it = frontier.begin(); it != frontier.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (comp[v] != curr_label)
          return false;
        if (!visited.get_bit(v)) {
          visited.set_bit(v);
          frontier.push_back(v);
        }
      }
    }
  }
  for (NodeID n=0; n < g.num_nodes(); n++)
    if (!visited.get_bit(n))
      return false;
  return true;
}

#ifdef HASH_MODE
int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  if(omp_get_max_threads() > 1)
    BenchmarkKernel(cli, g, ShiloachVishkinNUMA, PrintCompStats, CCVerifier);
  else
    BenchmarkKernel(cli, g, ShiloachVishkin, PrintCompStats, CCVerifier);
  return 0;
}
#else
int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  BenchmarkKernel(cli, g, ShiloachVishkin, PrintCompStats, CCVerifier);
  return 0;
}
#endif

