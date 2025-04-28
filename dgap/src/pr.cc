// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include <omp.h>
#include <string.h>
/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;
void PrintTopScores(const WGraph &g, const ScoreT *scores);
void bind_current_thread_to_cpu_list(const std::vector<int> &cpus);

ScoreT *PageRankPullNuma(const WGraph &g, int max_iters, double epsilon = 0) {
  const int64_t num_nodes   = g.num_nodes();
  const ScoreT  init_score = 1.0f / num_nodes;
  const ScoreT  base_score = (1.0f - kDamp) / num_nodes;
  ScoreT       *scores     = (ScoreT*)malloc(sizeof(ScoreT)*num_nodes);
  ScoreT       *outgoing   = (ScoreT*)malloc(sizeof(ScoreT)*num_nodes);
  int64_t       vertices0  = num_nodes/2;

  // inicializa
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; ++i)
    scores[i] = init_score;

  double error = 0.0;

  #pragma omp parallel reduction(+ : error)
  {
    // Bind uma única vez por thread
    int tid        = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    int n0         = numThreads/2;

    static const std::vector<int> node0_cpus = {
       0,1,2,3,4,5,6,7,8,9,10,11,
      24,25,26,27,28,29,30,31,32,33,34,35
    };
    static const std::vector<int> node1_cpus = {
      12,13,14,15,16,17,18,19,20,21,22,23,
      36,37,38,39,40,41,42,43,44,45,46,47
    };
    int64_t start, end;
    if (tid < n0){
      bind_current_thread_to_cpu_list(node0_cpus);
      int64_t chunk = (vertices0 + n0 - 1) / n0;
      start = tid * chunk;
      end   = std::min<int64_t>(vertices0, start + chunk);
    }
    else{
      bind_current_thread_to_cpu_list(node1_cpus);
      int   tid1 = tid - n0;
      int64_t rem   = num_nodes - vertices0;
      int64_t chunk = (rem + (numThreads-n0) - 1) / (numThreads-n0);
      start = vertices0 + tid1 * chunk;
      end   = std::min<int64_t>(num_nodes, start + chunk);
    }
    for (int iter = 0; iter < max_iters; ++iter) {

      for (int64_t u = start; u < end; ++u) //Utiliza o intervalo criado
        outgoing[u] = scores[u] / g.out_degree(u);

      // Pagerank em si, também utiliza o itervalo criado
      #pragma omp for schedule(dynamic, 64) nowait
      for (int64_t u = start; u < end; ++u) {
        ScoreT sum = 0;
        for (auto v : g.in_neigh(u))
          sum += outgoing[v];
        ScoreT old = scores[u];
        scores[u] = base_score + kDamp * sum;
        error += fabs(scores[u] - old);
      }
    }
  } // fim do parallel

  return scores;
}

ScoreT * PageRankPull(const WGraph &g, int max_iters,
  double epsilon = 0) {
const ScoreT init_score = 1.0f / g.num_nodes();
const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
ScoreT *scores;
ScoreT *outgoing_contrib;

scores = (ScoreT *) malloc(sizeof(ScoreT) * g.num_nodes());
outgoing_contrib = (ScoreT *) malloc(sizeof(ScoreT) * g.num_nodes());

#pragma omp parallel for
for (NodeID n=0; n < g.num_nodes(); n++) scores[n] = init_score;

for (int iter=0; iter < max_iters; iter++) {
double error = 0;
#pragma omp parallel for
for (NodeID n=0; n < g.num_nodes(); n++)
outgoing_contrib[n] = scores[n] / g.out_degree(n);
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
for (NodeID u=0; u < g.num_nodes(); u++) {
ScoreT incoming_total = 0;
for (NodeID v : g.in_neigh(u)){
printf("v: %d\n", v);
incoming_total += outgoing_contrib[v];
}
printf("FIM\n");

ScoreT old_score = scores[u];
scores[u] = base_score + kDamp * incoming_total;
error += fabs(scores[u] - old_score);
}
//    printf(" %2d    %lf\n", iter, error);
//    if (error < epsilon)
//      break;
}
PrintTopScores(g, scores);
return scores;
}


void bind_current_thread_to_cpu_list(const std::vector<int> &cpus) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int cpu : cpus) {
      CPU_SET(cpu, &cpuset);
  }
  pthread_t tid = pthread_self();
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
}


void PrintTopScores(const WGraph &g, const ScoreT *scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const WGraph &g, const ScoreT *scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

#ifdef NUMA_PMEM
int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  std::function<ScoreT*(const WGraph&)> PRBound;
//  g.print_pma_meta();
if(omp_get_max_threads() > 1){
  PRBound = [&cli] (const WGraph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
}
else{
  PRBound = [&cli] (const WGraph &g) {
    return PageRankPullNuma(g, cli.max_iters(), cli.tolerance());
  };
}
  
  auto VerifierBound = [&cli] (const WGraph &g, const ScoreT *scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}

#else
int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  printf("deu certo criar grafo\n");
//  g.print_pma_meta();
  auto PRBound = [&cli] (const WGraph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const WGraph &g, const ScoreT *scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
#endif
