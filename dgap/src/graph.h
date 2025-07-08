// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <cstring>
#include <omp.h>
#include <climits>
#include <numa.h>
#include <fstream>
#include <string>

#include <libpmemobj.h>
#include <libpmem.h>
#include <xmmintrin.h>

#include <condition_variable>
#include <mutex>
#include "platform_atomics.h"

#include "pvector.h"
#include "util.h"

using namespace std;

#define MAX_LOG_ENTRIES 170   // ~1.99 KB (2040 Bytes)
#define MAX_ULOG_ENTRIES 512  // 2KB

/*
Simple container for graph in DGAP format
 - Intended to be constructed by a Builder
*/

// Used to hold node & weight, with another node it makes a weighted edge
struct LogEntry {
  int32_t u;
  int32_t v;
  int32_t prev_offset;

  LogEntry() {}

  LogEntry(int32_t u, int32_t v) : u(u), v(v), prev_offset(-1) {}

  LogEntry(int32_t u, int32_t v, int32_t prev_offset) : u(u), v(v), prev_offset(prev_offset) {}

  bool operator<(const LogEntry &rhs) const {
    return (u < rhs.u);
  }

  bool operator==(const LogEntry &rhs) const {
    return (u == rhs.u && v == rhs.v);
  }
};

struct PMALeafSegment {
  mutex lock;
  condition_variable cv;
  int32_t on_rebalance;

  PMALeafSegment() {
    on_rebalance = 0;
  }

  void wait_for_rebalance(unique_lock <mutex> &ul) {
    this->cv.wait(ul, [this] { return !on_rebalance; });
  }
};

// Used to hold node & weight, with another node it makes a weighted edge
//template <typename NodeID_, typename WeightT_, typename TimestampT_>
template<typename NodeID_>
struct NodeWeight {
  NodeID_ v;

  NodeWeight() {}

  NodeWeight(NodeID_ v) : v(v) {}

  bool operator<(const NodeWeight &rhs) const {
    return v < rhs.v;
  }

  bool operator==(const NodeWeight &rhs) const {
    return v == rhs.v;
  }

  bool operator==(const NodeID_ &rhs) const {
    return v == rhs;
  }

  operator NodeID_() {
    return v;
  }
};

// Syntatic sugar for an edge
template<typename SrcT, typename DstT = SrcT>
struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int32_t SGID;
typedef int64_t TimestampT;
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;

// structure for the vertices
struct vertex_element {
  int64_t index;
  int32_t degree;
  int32_t offset;
};

/* the root object */
struct Base {
  uint64_t pool_uuid_lo;

  PMEMoid vertices_oid_;
  PMEMoid edges_oid_;
  PMEMoid ulog_oid_;              // oid of undo logs
  PMEMoid oplog_oid_;             // oid of operation logs
  PMEMoid log_segment_oid_;       // oid of logs per segment
  PMEMoid log_segment_idx_oid_;   // oid of current insert-index of logs per segment

  PMEMoid segment_edges_actual_oid_;  // actual number of edges stored in the region of a binary-tree node
  PMEMoid segment_edges_total_oid_;   // total number of edges assigned in the region of a binary-tree node

  /* General graph fields */
  int64_t num_vertices;   // Number of vertices
  int64_t num_edges_;     // Number of edges
  /* General PMA fields */
  int64_t elem_capacity; // size of the edges_ array
  int64_t segment_count; // number of pma leaf segments
  int32_t segment_size;  // size of a pma leaf segment
  int32_t tree_height;   // height of the pma tree

  bool directed_;
  bool backed_up_;
  char padding_[6];    //6 Bytes
} __attribute__ ((aligned (8)));

template<class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class CSRGraph {
  // Used for *non-negative* offsets within a neighborhood
  typedef std::make_unsigned<std::ptrdiff_t>::type OffsetT;
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    DestID_ *seg_base_ptr;
    struct vertex_element *src_v;
    struct LogEntry *log_p;
    int32_t start_offset;
    int32_t onseg_edges, log_index = -1;

    DestID_ *begin_ptr;
    DestID_ *end_ptr;
    bool onseg = false, onlog = false;

  public:
    Neighborhood(DestID_ *seg_base_ptr_, struct vertex_element *src_v_, OffsetT start_offset_, struct LogEntry *log_p_,
                 int32_t onseg_edges_) :
        seg_base_ptr(seg_base_ptr_), src_v(src_v_), log_p(log_p_), onseg_edges(onseg_edges_) {

      start_offset = std::min((int32_t) start_offset_, src_v->degree);
      if (src_v->degree > start_offset) {  // have data to iterate
        if (onseg_edges_ > start_offset) { // have data to iterate from segment
          begin_ptr = (DestID_ *) (seg_base_ptr + src_v->index + start_offset);
          log_index = src_v_->offset;
          onseg = true;
        } else {  // no data left on the segment; need to iterate over logs
          log_index = src_v_->offset;
          start_offset_ = start_offset - onseg_edges_;

          while (start_offset_ != 0) {
            log_index = log_p[log_index].prev_offset;
            start_offset_ -= 1;
          }
          begin_ptr = new DestID_();
          begin_ptr->v = log_p[log_index].v;
          log_index = log_p[log_index].prev_offset;

          onlog = true;
        }
      } else begin_ptr = nullptr;
      end_ptr = nullptr;
    }

    class iterator {
    public:
      int32_t onseg_edges;
      struct vertex_element *src_v;
      struct LogEntry *log_p;
      int32_t log_index, iterator_index;
      bool onseg, onlog;

      iterator() {
        ptr = NULL;
        iterator_index = 1;
        onseg = false;
        onlog = false;
      }

      iterator(DestID_ *p) {
        ptr = p;
        iterator_index = 1;
        onseg = false;
        onlog = false;
      }

      iterator(DestID_ *p, int32_t onseg_edges_, struct vertex_element *src_v_, struct LogEntry *log_p_,
               int32_t log_index_, int32_t iterator_index_, bool onseg_, bool onlog_) :
          onseg_edges(onseg_edges_), src_v(src_v_), log_p(log_p_), log_index(log_index_),
          iterator_index(iterator_index_), onseg(onseg_), onlog(onlog_) {
        ptr = p;
      }

      iterator &operator++() {
        iterator_index += 1;

        if (onseg) { // on-seg
          // if iterator_index goes beyond the onseg_edges
          if (iterator_index > onseg_edges) {
            // if degree > onseg_edges: switch to onlog
            if (src_v->degree > onseg_edges) {
              // this re-initialization of ptr is very crucial; otherwise it will overwrite the existing data
              ptr = new DestID_();
              ptr->v = log_p[log_index].v;

              log_index = log_p[log_index].prev_offset;
              onseg = false;
              onlog = true;
            } else {  // stop
              ptr = nullptr;
            }
          } else {
            ptr = (DestID_ *) ptr + 1;
          }
        } else { // onlog
          // if the prev_offset of current iteration is -1; stop
          if (log_index != -1) {
            ptr->v = log_p[log_index].v;
            log_index = log_p[log_index].prev_offset;
          } else {  // stop
            ptr = nullptr;
          }
        }

        return *this;
      }

      operator DestID_ *() const {
        return ptr;
      }

      DestID_ *operator->() {
        return ptr;
      }

      DestID_ &operator*() {
        return (*ptr);
      }

      bool operator==(const iterator &rhs) const {
        return ptr == rhs.ptr;
      }

      bool operator!=(const iterator &rhs) const {
        return (ptr != rhs.ptr);
      }

    private:
      DestID_ *ptr;
    };

    iterator begin() {
      return iterator(begin_ptr, onseg_edges, src_v, log_p, log_index, start_offset + 1, onseg, onlog);
    }

    iterator end() {
      return iterator(end_ptr);
    }
  };
  #ifdef NUMA_PMEM
  void ReleaseResources() {
    if (vertices_0 != nullptr) numa_free(vertices_0, n_vertices_node0 * sizeof(vertex_element));
    if (vertices_1 != nullptr) numa_free(vertices_1, n_vertices_node1 * sizeof(vertex_element));
    if (log_ptr_ != nullptr) free(log_ptr_);
    //if (log_segment_idx_0 != nullptr) free(log_segment_idx_0);
    //if (log_segment_idx_1 != nullptr) free(log_segment_idx_1);
    if (segment_edges_actual != nullptr) free(segment_edges_actual);
    if (segment_edges_total != nullptr) free(segment_edges_total);
  }
  #else
  void ReleaseResources() {
    if (vertices_ != nullptr) free(vertices_);
    if (log_ptr_ != nullptr) free(log_ptr_);
    if (log_segment_idx_ != nullptr) free(log_segment_idx_);
    if (segment_edges_actual != nullptr) free(segment_edges_actual);
    if (segment_edges_total != nullptr) free(segment_edges_total);
  }
  #endif
public:
  #ifdef NUMA_PMEM
  PMEMobjpool *pop0;
  PMEMobjpool *pop1;
  PMEMoid base_oid0;
  PMEMoid base_oid1;
  struct Base *bp0;
  struct Base *bp1;
  #else
  PMEMobjpool *pop;
  PMEMoid base_oid;
  struct Base *bp;
  #endif

  #ifdef NUMA_PMEM
  
~CSRGraph() {
    // Para persistir os dados dos vértices, que estão distribuidos por hash em memória
    // (pares no nó 0, ímpares no nó 1), precisamos primeiro remontá-los em arrays
    // contíguos temporários que correspondam ao layout da PMem.
    auto temp_vertices_0 = (struct vertex_element*) numa_alloc_onnode(n_vertices_node0 * sizeof(struct vertex_element), 0);
    auto temp_vertices_1 = (struct vertex_element*) numa_alloc_onnode(n_vertices_node1 * sizeof(struct vertex_element), 1);

    // Itera sobre todos os vértices do grafo.
    for (int32_t i = 0; i < num_vertices; ++i) {
        bool is_node0 = (i % 2 == 0);
        int32_t local_id = i / 2;

        if (is_node0) {
            // Se o vértice 'i' é par, seus dados estão em vertices_0.
            // Copiamos para a posição correta no array temporário do nó 0.
            if (local_id < n_vertices_node0) {
                 temp_vertices_0[local_id] = vertices_0[local_id];
            }
        } else {
            // Se o vértice 'i' é ímpar, seus dados estão em vertices_1.
            // Copiamos para a posição correta no array temporário do nó 1.
            if (local_id < n_vertices_node1) {
                temp_vertices_1[local_id] = vertices_1[local_id];
            }
        }
    }

    // Agora que temos os dados ordenados nos arrays temporários, podemos copiá-los para a PMem.
    memcpy((struct vertex_element *) pmemobj_direct(bp0->vertices_oid_), temp_vertices_0, n_vertices_node0 * sizeof(struct vertex_element));
    flush_clwb_nolog((struct vertex_element *) pmemobj_direct(bp0->vertices_oid_), n_vertices_node0 * sizeof(struct vertex_element));

    memcpy((struct vertex_element *) pmemobj_direct(bp1->vertices_oid_), temp_vertices_1, n_vertices_node1 * sizeof(struct vertex_element));
    flush_clwb_nolog((struct vertex_element *) pmemobj_direct(bp1->vertices_oid_), n_vertices_node1 * sizeof(struct vertex_element));

    // Libera a memória dos arrays temporários.
    numa_free(temp_vertices_0, n_vertices_node0 * sizeof(struct vertex_element));
    numa_free(temp_vertices_1, n_vertices_node1 * sizeof(struct vertex_element));

    // Persiste os metadados dos logs e dos segmentos.
    // Estes já estão particionados corretamente em memória.
    memcpy((int32_t *) pmemobj_direct(bp0->log_segment_idx_oid_), log_segment_idx_0, segment_count0 * sizeof(int32_t));
    flush_clwb_nolog((int32_t *) pmemobj_direct(bp0->log_segment_idx_oid_), segment_count0 * sizeof(int32_t));
    memcpy((int32_t *) pmemobj_direct(bp1->log_segment_idx_oid_), log_segment_idx_1, segment_count1 * sizeof(int32_t));
    flush_clwb_nolog((int32_t *) pmemobj_direct(bp1->log_segment_idx_oid_), segment_count1 * sizeof(int32_t));

    memcpy((int64_t *) pmemobj_direct(bp0->segment_edges_actual_oid_), segment_edges_actual_0, sizeof(int64_t) * segment_count0 * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp0->segment_edges_actual_oid_), sizeof(int64_t) * segment_count0 * 2);
    memcpy((int64_t *) pmemobj_direct(bp1->segment_edges_actual_oid_), segment_edges_actual_1, sizeof(int64_t) * segment_count1 * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp1->segment_edges_actual_oid_), sizeof(int64_t) * segment_count1 * 2);

    // 'segment_edges_total' é global, então salvamos a mesma cópia em ambos os nós.
    memcpy((int64_t *) pmemobj_direct(bp0->segment_edges_total_oid_), segment_edges_total, sizeof(int64_t) * segment_count * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp0->segment_edges_total_oid_), sizeof(int64_t) * segment_count * 2);
    memcpy((int64_t *) pmemobj_direct(bp1->segment_edges_total_oid_), segment_edges_total, sizeof(int64_t) * segment_count * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp1->segment_edges_total_oid_), sizeof(int64_t) * segment_count * 2);

    // Persiste os metadados gerais do grafo para cada nó.
    bp0->num_vertices = n_vertices_node0;
    bp1->num_vertices = n_vertices_node1;
    bp0->num_edges_ = n_edges_node0;
    bp1->num_edges_ = n_edges_node1;
    bp0->elem_capacity = elem_capacity0;
    bp1->elem_capacity = elem_capacity1;
    bp0->segment_count = segment_count0;
    bp1->segment_count = segment_count1;
    bp0->backed_up_ = true; // Sinaliza que o backup foi bem-sucedido.
    bp1->backed_up_ = true;

    // Persiste todas as alterações no objeto base.
    flush_clwb_nolog(bp0, sizeof(struct Base));
    flush_clwb_nolog(bp1, sizeof(struct Base));

    ReleaseResources(); // Libera recursos da DRAM.
}

  #else
  ~CSRGraph() {
    memcpy((struct vertex_element *) pmemobj_direct(bp->vertices_oid_), vertices_,
           num_vertices * sizeof(struct vertex_element));
    flush_clwb_nolog((struct vertex_element *) pmemobj_direct(bp->vertices_oid_),
                     num_vertices * sizeof(struct vertex_element));

    memcpy((int32_t *) pmemobj_direct(bp->log_segment_idx_oid_), log_segment_idx_, segment_count * sizeof(int32_t));
    flush_clwb_nolog((int32_t *) pmemobj_direct(bp->log_segment_idx_oid_), segment_count * sizeof(int32_t));

    memcpy((int64_t *) pmemobj_direct(bp->segment_edges_actual_oid_), segment_edges_actual,
           sizeof(int64_t) * segment_count * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp->segment_edges_actual_oid_), sizeof(int64_t) * segment_count * 2);

    memcpy((int64_t *) pmemobj_direct(bp->segment_edges_total_oid_), segment_edges_total,
           sizeof(int64_t) * segment_count * 2);
    flush_clwb_nolog((int64_t *) pmemobj_direct(bp->segment_edges_total_oid_), sizeof(int64_t) * segment_count * 2);

    bp->num_vertices = num_vertices;  // Number of vertices
    flush_clwb_nolog(&bp->num_vertices, sizeof(int64_t));

    bp->num_edges_ = num_edges_;  // Number of edges
    flush_clwb_nolog(&bp->num_edges_, sizeof(int64_t));

    bp->elem_capacity = elem_capacity;
    flush_clwb_nolog(&bp->elem_capacity, sizeof(int64_t));

    bp->segment_count = segment_count;
    flush_clwb_nolog(&bp->segment_count, sizeof(int64_t));

    bp->segment_size = segment_size;
    flush_clwb_nolog(&bp->segment_size, sizeof(int64_t));

    bp->tree_height = tree_height;
    flush_clwb_nolog(&bp->tree_height, sizeof(int64_t));

    // write flag indicating data has been backed up properly before shutting down
    bp->backed_up_ = true;
    flush_clwb_nolog(&bp->backed_up_, sizeof(bool));

    ReleaseResources();
  }
  #endif
  
    
  #ifdef NUMA_PMEM
CSRGraph(const char *file0, const char *file1, const EdgeList &edge_list, bool is_directed, int64_t n_edges, int64_t n_vertices) {
    bool is_new = false;
    node_counter = 2; // Hard-coded para 2 nós NUMA
    num_threads = omp_get_max_threads();
    printf("File 0: %s \nFile 1: %s\n", file0, file1);

    // Lógica para abrir ou criar os arquivos de pool de memória persistente
    if (file_exists(file0) != 0 || file_exists(file1) != 0) {
        if (access(file0, F_OK) == 0 && access(file1, F_OK) != 0) remove(file0);
        if (access(file0, F_OK) != 0 && access(file1, F_OK) == 0) remove(file1);
    }

    if (access(file0, F_OK) != 0) {
        if ((pop0 = pmemobj_create(file0, LAYOUT_NAME, DB_POOL_SIZE, CREATE_MODE_RW)) == NULL) {
            fprintf(stderr, "[%s]: FATAL: pmemobj_create (file0) error: %s\n", __FUNCTION__, pmemobj_errormsg());
            exit(1);
        }
        is_new = true;
    } else {
        if ((pop0 = pmemobj_open(file0, LAYOUT_NAME)) == NULL) {
            fprintf(stderr, "[%s]: FATAL: pmemobj_open (file0) error: %s\n", __FUNCTION__, pmemobj_errormsg());
            exit(1);
        }
    }

    if (access(file1, F_OK) != 0) {
        if ((pop1 = pmemobj_create(file1, LAYOUT_NAME, DB_POOL_SIZE, CREATE_MODE_RW)) == NULL) {
            fprintf(stderr, "[%s]: FATAL: pmemobj_create (file1) error: %s\n", __FUNCTION__, pmemobj_errormsg());
            exit(1);
        }
        is_new = true;
    } else {
        if ((pop1 = pmemobj_open(file1, LAYOUT_NAME)) == NULL) {
            fprintf(stderr, "[%s]: FATAL: pmemobj_open (file1) error: %s\n", __FUNCTION__, pmemobj_errormsg());
            exit(1);
        }
    }
    
    base_oid0 = pmemobj_root(pop0, sizeof(struct Base));
    bp0 = (struct Base *) pmemobj_direct(base_oid0);
    check_sanity(bp0);

    base_oid1 = pmemobj_root(pop1, sizeof(struct Base));
    bp1 = (struct Base *) pmemobj_direct(base_oid1);
    check_sanity(bp1);

    int num_threads_node0 = num_threads / 2;
    int num_threads_node1 = num_threads - num_threads_node0;
    if(num_threads_node0 < 1) {
        num_threads_node0 = 1;
        num_threads_node1 = 1;
    }

    if (is_new) {
      // --- CAMINHO PARA UM GRAFO NOVO ---
      num_edges_ = n_edges;
      num_vertices = n_vertices;
      max_valid_vertex_id = n_vertices;
      directed_ = is_directed;
      compute_capacity();

      n_vertices_node0 = num_vertices / 2;
      n_vertices_node1 = num_vertices - n_vertices_node0;
      
      elem_capacity0 = elem_capacity / 2;
      elem_capacity1 = elem_capacity - elem_capacity0;
      segment_count0 = segment_count / 2;
      segment_count1 = segment_count - segment_count0;

      segment_edges_actual_0 = (int64_t *) calloc(segment_count0 * 2, sizeof(int64_t));
      segment_edges_actual_1 = (int64_t *) calloc(segment_count1 * 2, sizeof(int64_t));
      segment_edges_total = (int64_t *) calloc(segment_count * 2, sizeof(int64_t));
      
      tree_height = floor_log2(segment_count);
      delta_up = (up_0 - up_h) / tree_height;
      delta_low = (low_h - low_0) / tree_height;

      // Inicialização dos metadados na PMem para o nó 0
      bp0->pool_uuid_lo = base_oid0.pool_uuid_lo;
      bp0->num_vertices = n_vertices_node0;
      bp0->num_edges_ = 0; // Começa com 0 arestas, serão inseridas depois
      bp0->directed_ = directed_;
      bp0->elem_capacity = elem_capacity0;
      bp0->segment_count = segment_count0;
      bp0->segment_size = segment_size;
      bp0->tree_height = tree_height;

      // Inicialização dos metadados na PMem para o nó 1
      bp1->pool_uuid_lo = base_oid1.pool_uuid_lo;
      bp1->num_vertices = n_vertices_node1;
      bp1->num_edges_ = 0;
      bp1->directed_ = directed_;
      bp1->elem_capacity = elem_capacity1;
      bp1->segment_count = segment_count1;
      bp1->segment_size = segment_size;
      bp1->tree_height = tree_height;

      // Alocação de memória na PMem para cada nó
      if (pmemobj_zalloc(pop0, &bp0->vertices_oid_, n_vertices_node0 * sizeof(struct vertex_element), VERTEX_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->vertices_oid_, n_vertices_node1 * sizeof(struct vertex_element), VERTEX_TYPE)) abort();
      if (pmemobj_zalloc(pop0, &bp0->edges_oid_, elem_capacity0 * sizeof(DestID_), EDGE_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->edges_oid_, elem_capacity1 * sizeof(DestID_), EDGE_TYPE)) abort();
      if (pmemobj_zalloc(pop0, &bp0->log_segment_oid_, segment_count0 * MAX_LOG_ENTRIES * sizeof(struct LogEntry), SEG_LOG_PTR_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->log_segment_oid_, segment_count1 * MAX_LOG_ENTRIES * sizeof(struct LogEntry), SEG_LOG_PTR_TYPE)) abort();
      if (pmemobj_zalloc(pop0, &bp0->log_segment_idx_oid_, segment_count0 * sizeof(int32_t), SEG_LOG_IDX_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->log_segment_idx_oid_, segment_count1 * sizeof(int32_t), SEG_LOG_IDX_TYPE)) abort();

      // EXPLICAÇÃO: O número de threads também deve ser dividido.
      // Assumindo que metade das threads trabalhará em cada nó.
      int num_threads_node0 = num_threads / 2;
      int num_threads_node1 = num_threads - num_threads_node0;
      if(num_threads_node0 < 1){
        num_threads_node0 = 1;
        num_threads_node1 = 1;
      }
      if (pmemobj_zalloc(pop0, &bp0->ulog_oid_, num_threads_node0 * MAX_ULOG_ENTRIES * sizeof(DestID_), ULOG_PTR_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->ulog_oid_, num_threads_node1 * MAX_ULOG_ENTRIES * sizeof(DestID_), ULOG_PTR_TYPE)) abort();
      if (pmemobj_zalloc(pop0, &bp0->oplog_oid_, num_threads_node0 * sizeof(int64_t), OPLOG_PTR_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->oplog_oid_, num_threads_node1 * sizeof(int64_t), OPLOG_PTR_TYPE)) abort();

      // EXPLICAÇÃO para a dúvida "por que diabos ele está alocando duas vezes a mesma coisa???":
      // O código não está alocando a mesma coisa duas vezes. Primeiro, ele aloca com `calloc` na DRAM (`segment_edges_actual_0`).
      // Depois, aloca com `pmemobj_zalloc` na PMem (`bp0->segment_edges_actual_oid_`).
      // A versão em DRAM é a cópia de trabalho, rápida, e a versão em PMem é o backup persistente.
      if (pmemobj_zalloc(pop0, &bp0->segment_edges_actual_oid_, sizeof(int64_t) * segment_count0 * 2, PMA_TREE_META_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->segment_edges_actual_oid_, sizeof(int64_t) * segment_count1 * 2, PMA_TREE_META_TYPE)) abort();
      if (pmemobj_zalloc(pop0, &bp0->segment_edges_total_oid_, sizeof(int64_t) * segment_count * 2, PMA_TREE_META_TYPE)) abort();
      if (pmemobj_zalloc(pop1, &bp1->segment_edges_total_oid_, sizeof(int64_t) * segment_count * 2, PMA_TREE_META_TYPE)) abort();

      flush_clwb_nolog(bp0, sizeof(struct Base));
      flush_clwb_nolog(bp1, sizeof(struct Base));
      
      // Obtenção dos ponteiros diretos para a PMem
      edges_0 = (DestID_ *) pmemobj_direct(bp0->edges_oid_);
      edges_1 = (DestID_ *) pmemobj_direct(bp1->edges_oid_);
      
      vertices_0 = (vertex_element*)numa_alloc_onnode(n_vertices_node0 * sizeof(vertex_element), 0);
      memcpy(vertices_0, (struct vertex_element *) pmemobj_direct(bp0->vertices_oid_), n_vertices_node0 * sizeof(struct vertex_element));
      vertices_1 = (vertex_element*)numa_alloc_onnode(n_vertices_node1 * sizeof(vertex_element), 1);
      memcpy(vertices_1, (struct vertex_element *) pmemobj_direct(bp1->vertices_oid_), n_vertices_node1 * sizeof(struct vertex_element));
      
      // Inicialização dos logs de segmento
      log_base_ptr_0 = (struct LogEntry *) pmemobj_direct(bp0->log_segment_oid_);
      log_ptr_0 = (struct LogEntry **) numa_alloc_onnode(segment_count0 * sizeof(struct LogEntry *), 0);
      for (int sid = 0; sid < segment_count0; sid++) { log_ptr_0[sid] = log_base_ptr_0 + (sid * MAX_LOG_ENTRIES); }
      log_segment_idx_0 = (int32_t *) numa_alloc_onnode(segment_count0 * sizeof(int32_t), 0);
      memset(log_segment_idx_0, 0, segment_count0 * sizeof(int32_t));

      log_base_ptr_1 = (struct LogEntry *) pmemobj_direct(bp1->log_segment_oid_);
      log_ptr_1 = (struct LogEntry **) numa_alloc_onnode(segment_count1 * sizeof(struct LogEntry *), 1);
      for (int sid = 0; sid < segment_count1; sid++) { log_ptr_1[sid] = log_base_ptr_1 + (sid * MAX_LOG_ENTRIES); }
      log_segment_idx_1 = (int32_t *) numa_alloc_onnode(segment_count1 * sizeof(int32_t), 1);
      memset(log_segment_idx_1, 0, segment_count1 * sizeof(int32_t));

      // Inicialização dos undo-logs (ulog) e operation-logs (oplog)
      ulog_base_ptr_0 = (DestID_ *) pmemobj_direct(bp0->ulog_oid_);
      ulog_ptr_0 = (DestID_ **) numa_alloc_onnode(num_threads_node0 * sizeof(DestID_ *), 0);
      for (int tid = 0; tid < num_threads_node0; tid++) { ulog_ptr_0[tid] = ulog_base_ptr_0 + (tid * MAX_ULOG_ENTRIES); }
      oplog_ptr_0 = (int64_t *) pmemobj_direct(bp0->oplog_oid_);

      ulog_base_ptr_1 = (DestID_ *) pmemobj_direct(bp1->ulog_oid_);
      ulog_ptr_1 = (DestID_ **) numa_alloc_onnode(num_threads_node1 * sizeof(DestID_ *), 1);
      for (int tid = 0; tid < num_threads_node1; tid++) { ulog_ptr_1[tid] = ulog_base_ptr_1 + (tid * MAX_ULOG_ENTRIES); }
      oplog_ptr_1 = (int64_t *) pmemobj_direct(bp1->oplog_oid_);
      
      leaf_segments = new PMALeafSegment[segment_count];


      // Inserção das arestas iniciais
      Timer t;
      t.Start();
      insert(edge_list); // Esta função precisará ser ciente da partição NUMA
      t.Stop();
      cout << "base graph insert time: " << t.Seconds() << endl;
      
      // Atualiza o número de arestas após a inserção
      bp0->num_edges_ = n_edges_node0;
      bp1->num_edges_ = n_edges_node1;
      flush_clwb_nolog(&bp0->num_edges_, sizeof(int64_t));
      flush_clwb_nolog(&bp1->num_edges_, sizeof(int64_t));


      bp0->backed_up_ = false;
      bp1->backed_up_ = false;
      flush_clwb_nolog(&bp0->backed_up_, sizeof(bool));
      flush_clwb_nolog(&bp1->backed_up_, sizeof(bool));
    } else {
      Timer t_reboot;
      t_reboot.Start();

      cout << "Rebooting from existing files..." << endl;
      cout << "Node 0 last backup OK? " << bp0->backed_up_ << endl;
      cout << "Node 1 last backup OK? " << bp1->backed_up_ << endl;

      // Carrega metadados da PMem para a DRAM
      directed_ = bp0->directed_; // Assume que é o mesmo para ambos
      tree_height = bp0->tree_height;
      segment_size = bp0->segment_size;
      
      // Agrega metadados dos dois nós
      n_vertices_node0 = bp0->num_vertices;
      n_vertices_node1 = bp1->num_vertices;
      num_vertices = n_vertices_node0 + n_vertices_node1;
      
      n_edges_node0 = bp0->num_edges_;
      n_edges_node1 = bp1->num_edges_;
      num_edges_ = n_edges_node0 + n_edges_node1;
      
      elem_capacity0 = bp0->elem_capacity;
      elem_capacity1 = bp1->elem_capacity;
      elem_capacity = elem_capacity0 + elem_capacity1;
      
      segment_count0 = bp0->segment_count;
      segment_count1 = bp1->segment_count;
      segment_count = segment_count0 + segment_count1;
      
      // Aloca e inicializa estruturas em DRAM
      segment_edges_actual_0 = (int64_t *) numa_alloc_onnode(segment_count0 * 2 * sizeof(int64_t), 0);
      segment_edges_actual_1 = (int64_t *) numa_alloc_onnode(segment_count1 * 2 * sizeof(int64_t), 1);
      segment_edges_total = (int64_t *) numa_alloc_onnode(segment_count * 2 * sizeof(int64_t), 0); // Pode ser em qualquer nó

      delta_up = (up_0 - up_h) / tree_height;
      delta_low = (low_h - low_0) / tree_height;
      
      // Obtém ponteiros para as regiões de PMem
      edges_0 = (DestID_ *) pmemobj_direct(bp0->edges_oid_);
      edges_1 = (DestID_ *) pmemobj_direct(bp1->edges_oid_);
      
      // Aloca memória em DRAM para os vértices em nós específicos
      vertices_0 = (struct vertex_element *) numa_alloc_onnode(n_vertices_node0 * sizeof(struct vertex_element), 0);
      vertices_1 = (struct vertex_element *) numa_alloc_onnode(n_vertices_node1 * sizeof(struct vertex_element), 1);
      
      // Inicializa os ponteiros de log de segmento em DRAM
      log_base_ptr_0 = (struct LogEntry *) pmemobj_direct(bp0->log_segment_oid_);
      log_ptr_0 = (struct LogEntry **) numa_alloc_onnode(segment_count0 * sizeof(struct LogEntry *), 0);
      for (int sid = 0; sid < segment_count0; sid++) { log_ptr_0[sid] = log_base_ptr_0 + (sid * MAX_LOG_ENTRIES); }
      log_segment_idx_0 = (int32_t *) numa_alloc_onnode(segment_count0 * sizeof(int32_t), 0);

      log_base_ptr_1 = (struct LogEntry *) pmemobj_direct(bp1->log_segment_oid_);
      log_ptr_1 = (struct LogEntry **) numa_alloc_onnode(segment_count1 * sizeof(struct LogEntry *), 1);
      for (int sid = 0; sid < segment_count1; sid++) { log_ptr_1[sid] = log_base_ptr_1 + (sid * MAX_LOG_ENTRIES); }
      log_segment_idx_1 = (int32_t *) numa_alloc_onnode(segment_count1 * sizeof(int32_t), 1);

      // Inicialização dos undo-logs (ulog) e operation-logs (oplog) em DRAM
      ulog_base_ptr_0 = (DestID_ *) pmemobj_direct(bp0->ulog_oid_);
      ulog_ptr_0 = (DestID_ **) numa_alloc_onnode(num_threads_node0 * sizeof(DestID_ *), 0);
      for (int tid = 0; tid < num_threads_node0; tid++) { ulog_ptr_0[tid] = ulog_base_ptr_0 + (tid * MAX_ULOG_ENTRIES); }
      oplog_ptr_0 = (int64_t *) pmemobj_direct(bp0->oplog_oid_);

      ulog_base_ptr_1 = (DestID_ *) pmemobj_direct(bp1->ulog_oid_);
      ulog_ptr_1 = (DestID_ **) numa_alloc_onnode(num_threads_node1 * sizeof(DestID_ *), 1);
      for (int tid = 0; tid < num_threads_node1; tid++) { ulog_ptr_1[tid] = ulog_base_ptr_1 + (tid * MAX_ULOG_ENTRIES); }
      oplog_ptr_1 = (int64_t *) pmemobj_direct(bp1->oplog_oid_);
      
      // Inicializa as primitivas de concorrência
      leaf_segments = new PMALeafSegment[segment_count];
      
      // Verifica se o último desligamento foi correto
      if (bp0->backed_up_ && bp1->backed_up_) {
        cout << "Last shutdown was clean. Loading data from PMem." << endl;
        // Copia os dados da PMem de volta para a DRAM
        memcpy(vertices_0, (struct vertex_element *) pmemobj_direct(bp0->vertices_oid_), n_vertices_node0 * sizeof(struct vertex_element));
        memcpy(vertices_1, (struct vertex_element *) pmemobj_direct(bp1->vertices_oid_), n_vertices_node1 * sizeof(struct vertex_element));
        
        memcpy(log_segment_idx_0, (int32_t *) pmemobj_direct(bp0->log_segment_idx_oid_), segment_count0 * sizeof(int32_t));
        memcpy(log_segment_idx_1, (int32_t *) pmemobj_direct(bp1->log_segment_idx_oid_), segment_count1 * sizeof(int32_t));
        
        memcpy(segment_edges_actual_0, (int64_t *) pmemobj_direct(bp0->segment_edges_actual_oid_), sizeof(int64_t) * segment_count0 * 2);
        memcpy(segment_edges_actual_1, (int64_t *) pmemobj_direct(bp1->segment_edges_actual_oid_), sizeof(int64_t) * segment_count1 * 2);
        
        // O array segment_edges_total é global, então podemos carregar de qualquer um dos nós (usando bp0 aqui)
        memcpy(segment_edges_total, (int64_t *) pmemobj_direct(bp0->segment_edges_total_oid_), sizeof(int64_t) * segment_count * 2);


      } else {
        // Se houve uma falha, executa o procedimento de recuperação
        Timer t;
        t.Start();
        cout << "Last shutdown was dirty. Starting recovery process." << endl;
        recovery(); // IMPORTANTE: A função recovery() também precisa ser adaptada para NUMA.
        t.Stop();
        cout << "graph recovery time: " << t.Seconds() << endl;

        // Persiste o número correto de arestas após a recuperação
        bp0->num_edges_ = n_edges_node0;
        flush_clwb_nolog(&bp0->num_edges_, sizeof(int64_t));
        bp1->num_edges_ = n_edges_node1;
        flush_clwb_nolog(&bp1->num_edges_, sizeof(int64_t));
      }

      t_reboot.Stop();
      cout << "graph reboot time: " << t_reboot.Seconds() << endl;
    }
  }
  #else
  CSRGraph(const char *file, const EdgeList &edge_list, bool is_directed, int64_t n_edges, int64_t n_vertices) {
    bool is_new = false;
    num_threads = omp_get_max_threads();

    /* file already exists */
    if (file_exists(file) == 0) {
      if ((pop = pmemobj_open(file, LAYOUT_NAME)) == NULL) {
        fprintf(stderr, "[%s]: FATAL: pmemobj_open error: %s\n", __FUNCTION__, pmemobj_errormsg());
        exit(0);
      }
    } else {
      if ((pop = pmemobj_create(file, LAYOUT_NAME, DB_POOL_SIZE, CREATE_MODE_RW)) == NULL) {
        fprintf(stderr, "[%s]: FATAL: pmemobj_create error: %s\n", __FUNCTION__, pmemobj_errormsg());
        exit(0);
      }
      is_new = true;
    }

    base_oid = pmemobj_root(pop, sizeof(struct Base));
    bp = (struct Base *) pmemobj_direct(base_oid);
    check_sanity(bp);

    // newly created file
    if (is_new) {
      // ds initialization (for dram-domain)
      num_edges_ = n_edges;
      num_vertices = n_vertices;
      max_valid_vertex_id = n_vertices;
      directed_ = is_directed;
      compute_capacity();

      // array-based compete tree structure
      segment_edges_actual = (int64_t *) calloc(segment_count * 2, sizeof(int64_t));
      segment_edges_total = (int64_t *) calloc(segment_count * 2, sizeof(int64_t));

      tree_height = floor_log2(segment_count);
      delta_up = (up_0 - up_h) / tree_height;
      delta_low = (low_h - low_0) / tree_height;

      // ds initialization (for pmem-domain)
      bp->pool_uuid_lo = base_oid.pool_uuid_lo;
      bp->num_vertices = num_vertices;
      bp->num_edges_ = num_edges_;
      bp->directed_ = directed_;
      bp->elem_capacity = elem_capacity;
      bp->segment_count = segment_count;
      bp->segment_size = segment_size;
      bp->tree_height = tree_height;

      // allocate memory for vertices and edges in pmem-domain
      // shouldn't the vertex array be volatile? - OTAVIO
      if (pmemobj_zalloc(pop, &bp->vertices_oid_, num_vertices * sizeof(struct vertex_element), VERTEX_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: vertex array allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      if (pmemobj_zalloc(pop, &bp->edges_oid_, elem_capacity * sizeof(DestID_), EDGE_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: edge array allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      // conteudo dos per-section edge logs - OTAVIO
      if (pmemobj_zalloc(pop, &bp->log_segment_oid_, segment_count * MAX_LOG_ENTRIES * sizeof(struct LogEntry),
                         SEG_LOG_PTR_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: per-segment log array allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      if (pmemobj_zalloc(pop, &bp->log_segment_idx_oid_, segment_count * sizeof(int32_t), SEG_LOG_IDX_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: per-segment log index allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      if (pmemobj_zalloc(pop, &bp->ulog_oid_, num_threads * MAX_ULOG_ENTRIES * sizeof(DestID_), ULOG_PTR_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: u-log array allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      // ??? - OTAVIO
      if (pmemobj_zalloc(pop, &bp->oplog_oid_, num_threads * sizeof(int64_t), OPLOG_PTR_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: op-log array allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      // por que diabos ele está alocando duas vezes a mesma coisa??? - OTAVIO
      if (pmemobj_zalloc(pop, &bp->segment_edges_actual_oid_, sizeof(int64_t) * segment_count * 2,
                         PMA_TREE_META_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: pma metadata allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      if (pmemobj_zalloc(pop, &bp->segment_edges_total_oid_, sizeof(int64_t) * segment_count * 2, PMA_TREE_META_TYPE)) {
        fprintf(stderr, "[%s]: FATAL: pma metadata allocation failed: %s\n", __func__, pmemobj_errormsg());
        abort();
      }

      flush_clwb_nolog(bp, sizeof(struct Base));

      // retrieving pmem-pointer from pmem-oid (for pmem domain)
      edges_ = (DestID_ *) pmemobj_direct(bp->edges_oid_);
      vertices_ = (struct vertex_element *) malloc(num_vertices * sizeof(struct vertex_element));
      memcpy(vertices_, (struct vertex_element *) pmemobj_direct(bp->vertices_oid_),
             num_vertices * sizeof(struct vertex_element));

      log_base_ptr_ = (struct LogEntry *) pmemobj_direct(bp->log_segment_oid_);
      log_ptr_ = (struct LogEntry **) malloc(segment_count * sizeof(struct LogEntry *));  // 8-byte

      // save pointer in the log_ptr_[sid]
      for (int sid = 0; sid < segment_count; sid += 1) {
        log_ptr_[sid] = (struct LogEntry *) (log_base_ptr_ + (sid * MAX_LOG_ENTRIES));
      }
      log_segment_idx_ = (int32_t *) calloc(segment_count, sizeof(int32_t));

      ulog_base_ptr_ = (DestID_ *) pmemobj_direct(bp->ulog_oid_);
      ulog_ptr_ = (DestID_ **) malloc(num_threads * sizeof(DestID_ *)); // 8-byte

      // save pointer in the ulog_ptr_[tid]
      for (int tid = 0; tid < num_threads; tid += 1) {
        /// assignment of per-seg-edge-log from a single large pre-allocated log
        ulog_ptr_[tid] = (DestID_ *) (ulog_base_ptr_ + (tid * MAX_ULOG_ENTRIES));
      }

      oplog_ptr_ = (int64_t *) pmemobj_direct(bp->oplog_oid_);

      // leaf segment concurrency primitives
      leaf_segments = new PMALeafSegment[segment_count];

      /// insert base-graph edges
      Timer t;
      t.Start();
      insert(edge_list); // return later - OTAVIO
      t.Stop();
      cout << "base graph insert time: " << t.Seconds() << endl;

      bp->backed_up_ = false;
      flush_clwb_nolog(&bp->backed_up_, sizeof(bool));
    } else {
      Timer t_reboot;
      t_reboot.Start();

      cout << "how was last backup? Good? " << bp->backed_up_ << endl;

      /// ds initialization (for dram-domain)
      elem_capacity = bp->elem_capacity;
      segment_count = bp->segment_count;
      segment_size = bp->segment_size;
      tree_height = bp->tree_height;

      num_vertices = bp->num_vertices;
      num_edges_ = bp->num_edges_;
      avg_degree = ceil_div(num_edges_, num_vertices);
      directed_ = bp->directed_;

      // array-based compete tree structure
      segment_edges_actual = (int64_t *) calloc(segment_count * 2, sizeof(int64_t));
      segment_edges_total = (int64_t *) calloc(segment_count * 2, sizeof(int64_t));

      delta_up = (up_0 - up_h) / tree_height;
      delta_low = (low_h - low_0) / tree_height;

      // retrieving pmem-pointer from pmem-oid (for pmem domain)
      edges_ = (DestID_ *) pmemobj_direct(bp->edges_oid_);
      vertices_ = (struct vertex_element *) malloc(num_vertices * sizeof(struct vertex_element));

      log_base_ptr_ = (struct LogEntry *) pmemobj_direct(bp->log_segment_oid_);
      log_ptr_ = (struct LogEntry **) malloc(segment_count * sizeof(struct LogEntry *));  // 8-byte
      log_segment_idx_ = (int32_t *) malloc(segment_count * sizeof(int32_t)); // 4-byte

      for (int sid = 0; sid < segment_count; sid += 1) {
        // save pointer in the log_ptr_[sid]
        log_ptr_[sid] = (struct LogEntry *) (log_base_ptr_ + (sid * MAX_LOG_ENTRIES));
      }

      // last shutdown was properly backed up
      if (bp->backed_up_) {
        memcpy(vertices_, (struct vertex_element *) pmemobj_direct(bp->vertices_oid_),
               num_vertices * sizeof(struct vertex_element));
        memcpy(log_segment_idx_, (int32_t *) pmemobj_direct(bp->log_segment_idx_oid_),
               segment_count * sizeof(int32_t));  // 4-byte
        memcpy(segment_edges_actual, (int64_t *) pmemobj_direct(bp->segment_edges_actual_oid_),
               sizeof(int64_t) * segment_count * 2); // 8-byte
        memcpy(segment_edges_total, (int64_t *) pmemobj_direct(bp->segment_edges_total_oid_),
               sizeof(int64_t) * segment_count * 2); // 8-byte
      } else {
        Timer t;
        t.Start();
        recovery(); // return later - OTAVIO
        t.Stop();
        cout << "graph recovery time: " << t.Seconds() << endl;

        // persisting number of edges
        bp->num_edges_ = num_edges_;
        flush_clwb_nolog(&bp->num_edges_, sizeof(int64_t));
      }

      t_reboot.Stop();
      cout << "graph reboot time: " << t_reboot.Seconds() << endl;
    }
  }
  #endif


  #ifdef NUMA_PMEM
  void recovery() {
    // =========================
    // Recuperação para nó 0 (partição 0)
    // =========================
    n_edges_node0 = 0;
    int64_t st_idx = 0;
    int32_t seg_id = 0;
    NodeID_ vid = 0;
    NodeID_ mx_vid = 0;
    
    // Se a primeira aresta de edges_0 tiver valor 0, significa que o vértice 0 foi iniciado
    if (edges_0[st_idx].v == 0) {
      seg_id = get_segment_id(0); // Para nó 0, os vértices estão com seus IDs globais
      vertices_0[0].index = st_idx;
      vertices_0[0].degree = 1;
      st_idx++;
      segment_edges_actual_0[seg_id] += 1;
      n_edges_node0++;
      while (edges_0[st_idx].v != 0) {
        st_idx++;
        vertices_0[0].degree += 1;
        segment_edges_actual_0[seg_id] += 1;
        n_edges_node0++;
      }
    }
    
    // Percorre o array de arestas do nó 0 a partir de st_idx até o fim da capacidade para o nó 0
    for (int64_t i = st_idx; i < elem_capacity; i++) {
      if (edges_0[i].v < 0) {
        vid = -edges_0[i].v;
        mx_vid = std::max(mx_vid, vid);
        seg_id = get_segment_id(vid);
        vertices_0[vid].index = i;
        vertices_0[vid].degree = 1;
        segment_edges_actual_0[seg_id] += 1;
        n_edges_node0++;
      }
      if (edges_0[i].v > 0) {
        // O valor de 'vid' permanece o último vértice processado
        vertices_0[vid].degree += 1;
        segment_edges_actual_0[seg_id] += 1;
        n_edges_node0++;
      }
    }
    
    cout << "max vertex-id retrieved for node 0: " << mx_vid << endl;
    cout << "segment_count (node 0): " << segment_count0 << endl;
    
    // Reconstrói os totais de arestas por segmento no nó 0
    recount_segment_total();
    
    // Inicializa log_segment_idx_ para nó 0, escaneando cada segmento de log
    int32_t total_log_entries0 = 0;
    for (int sid = 0; sid < segment_count0; sid++) {
      for (int i = 0; i < MAX_LOG_ENTRIES; i++) {
        if (log_ptr_[sid][i].u && log_ptr_[sid][i].v && log_ptr_[sid][i].prev_offset) {
          log_segment_idx_[sid] = i;
        } else {
          break;
        }
      }
      total_log_entries0 += log_segment_idx_[sid];
    }
    cout << "total log entries found for node 0: " << total_log_entries0 << endl;
    
    
    // =========================
    // Recuperação para nó 1 (partição 1 – vértices reindexados localmente para começar em 0)
    // =========================
    n_edges_node1 = 0;
    st_idx = 0;
    seg_id = 0;
    vid = 0;
    mx_vid = 0;
    
    // No nó 1, os vértices foram reindexados localmente, portanto o vértice 0 aqui corresponde
    // ao primeiro vértice da partição (globalmente: n_vertices_node0)
    if (edges_1[st_idx].v == 0) {
      // Para calcular o segmento, é preciso converter o ID local para o global
      seg_id = get_segment_id(n_vertices_node0);
      vertices_1[0].index = st_idx;
      vertices_1[0].degree = 1;
      st_idx++;
      segment_edges_actual_1[seg_id - segment_count0] += 1;  // ajusta o índice do segmento para nó 1
      n_edges_node1++;
      while (edges_1[st_idx].v != 0) {
        st_idx++;
        vertices_1[0].degree += 1;
        segment_edges_actual_1[seg_id - segment_count0] += 1;
        n_edges_node1++;
      }
    }
    
    for (int64_t i = st_idx; i < elem_capacity; i++) {
      if (edges_1[i].v < 0) {
        // Recupera o ID global: como os vértices foram reindexados, o ID global é (local ID + n_vertices_node0)
        vid = -edges_1[i].v;
        mx_vid = std::max(mx_vid, vid);
        // Converte o ID local para global para o cálculo do segmento
        seg_id = get_segment_id(vid + n_vertices_node0);
        vertices_1[vid].index = i;
        vertices_1[vid].degree = 1;
        segment_edges_actual_1[seg_id - segment_count0] += 1;
        n_edges_node1++;
      }
      if (edges_1[i].v > 0) {
        vertices_1[vid].degree += 1;
        segment_edges_actual_1[seg_id - segment_count0] += 1;
        n_edges_node1++;
      }
    }
    
    cout << "max vertex-id retrieved for node 1: " << mx_vid << endl;
    cout << "segment_count (node 1): " << segment_count1 << endl;
    
    recount_segment_total();
    
    int32_t total_log_entries1 = 0;
    for (int sid = 0; sid < segment_count1; sid++) {
      for (int i = 0; i < MAX_LOG_ENTRIES; i++) {
        if (log_ptr_[sid][i].u && log_ptr_[sid][i].v && log_ptr_[sid][i].prev_offset) {
          log_segment_idx_[sid] = i;
        } else {
          break;
        }
      }
      total_log_entries1 += log_segment_idx_[sid];
    }
    cout << "total log entries found for node 1: " << total_log_entries1 << endl;
  }
  
  #else
  void recovery() {
    /// rebuild vertices_ by scanning edges_
    /// recount num_edges_ and save it to bp->num_edges_
    /// reconstruct segment_edges_actual, segment_edges_total
    num_edges_ = 0;

    // need to translate the vertex-ids of the graphs which have vertex-id 0
    int64_t st_idx = 0;
    int32_t seg_id = 0;
    NodeID_ vid = 0;
    NodeID_ mx_vid = 0;
    if (edges_[st_idx].v == 0) {
      seg_id = get_segment_id(0);
      vertices_[0].index = st_idx;
      vertices_[0].degree = 1;
      st_idx += 1;
      segment_edges_actual[seg_id] += 1;
      num_edges_ += 1;
      while (edges_[st_idx].v != 0) {
        st_idx += 1;
        vertices_[0].degree += 1;
        segment_edges_actual[seg_id] += 1;
        num_edges_ += 1;
      }
    }
    for (int64_t i = st_idx; i < elem_capacity; i += 1) {
      if (edges_[i].v < 0) {
        vid = -edges_[i].v;
        mx_vid = max(mx_vid, vid);
        seg_id = get_segment_id(vid);
        vertices_[vid].index = i;
        vertices_[vid].degree = 1;
        segment_edges_actual[seg_id] += 1;
        num_edges_ += 1;
      }
      if (edges_[i].v > 0) {
        vertices_[vid].degree += 1;
        segment_edges_actual[seg_id] += 1;
        num_edges_ += 1;
      }
    }

    cout << "max vertex-id retrieved: " << mx_vid << endl;
    cout << "segment_count: " << segment_count << endl;

    // reconstruct segment_edges_total
    recount_segment_total();

    /// initialize log_segment_idx_ by scanning the log_ptr_[sid]
    int32_t total_log_entries = 0;
    for (int sid = 0; sid < segment_count; sid++) {
      for (int i = 0; i < MAX_LOG_ENTRIES; i += 1) {
        if (log_ptr_[sid][i].u && log_ptr_[sid][i].v && log_ptr_[sid][i].prev_offset) {
          log_segment_idx_[sid] = i;
        } else break;
      }
      total_log_entries += log_segment_idx_[sid];
    }
    cout << "total log entries found: " << total_log_entries << endl;
  }
  #endif

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_vertices;
  }

  int64_t num_edges() const {
    return num_edges_;
  }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2 * num_edges_;
  }

  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  int64_t out_degree(NodeID_ v) const {
    bool is_node0 = (v % node_counter == 0);
    int32_t local_idx = v / node_counter;
    // O grau real é o número de entradas no array de arestas - 1 (para não contar a entrada de cabeçalho)
    return is_node0 ? (vertices_0[local_idx].degree - 1) : (vertices_1[local_idx].degree - 1);
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    // Para um grafo não direcionado, o grau de entrada é igual ao de saída.
    return out_degree(v);
  }
  #else
  int64_t out_degree(NodeID_ v) const {
    if (v < n_vertices_node0) {
        // v pertence ao nó 0
        return vertices_0[v].degree - 1;
    } else {
        // v pertence ao nó 1, converte para índice local
        return vertices_1[v - n_vertices_node0].degree - 1;
    }
}

int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    if (v < n_vertices_node0) {
        return vertices_0[v].degree - 1;
    } else {
        return vertices_1[v - n_vertices_node0].degree - 1;
    }
}
#endif
  #else
  int64_t out_degree(NodeID_ v) const {
    return vertices_[v].degree - 1;
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return vertices_[v].degree - 1;
  }
  #endif

  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    // round-robin: Pares no node 0, impares node 1
    bool is0   = ((n % node_counter) == 0);
    int   local = n / node_counter;
    
    // Seleciona os ponteiros de vértice e aresta corretos (isto já estava correto)
    auto &V = is0 ? vertices_0 : vertices_1;
    auto &E = is0 ? edges_0    : edges_1;
    int   cap = is0 ? elem_capacity0 : elem_capacity1;

    int64_t idx = V[local].index;
    int32_t deg = V[local].degree;

    int64_t next_boundary = (local + 1 < (is0 ? n_vertices_node0 : n_vertices_node1))
        ? (V[local + 1].index - 1)
        : (cap - 1);

    int32_t onseg_edges = (V[local].offset != -1)
        ? (int32_t)(next_boundary - idx + 1)
        : deg;

    // --- LÓGICA DE LOG CORRIGIDA PARA NUMA HASH_MODE ---
    // 1. Calcula o ID do segmento global onde o vértice 'n' reside.
    int32_t global_seg_id = n / segment_size;

    // 2. Determina em qual nó o LOG deste segmento está armazenado (partição meio a meio).
    bool segment_log_on_node0 = (global_seg_id < segment_count0);
    
    // 3. Seleciona o array de ponteiros de log correto (log_ptr_0 ou log_ptr_1).
    struct LogEntry** target_log_array = segment_log_on_node0 ? log_ptr_0 : log_ptr_1;

    // 4. Calcula o índice LOCAL para usar no array de ponteiros de log.
    int32_t local_seg_id = segment_log_on_node0 ? global_seg_id : (global_seg_id - segment_count0);

    // 5. Obtém o ponteiro final para o log do segmento.
    struct LogEntry* segment_log_entries = target_log_array[local_seg_id];
    // --- FIM DA LÓGICA DE LOG CORRIGIDA ---

    return Neighborhood(
        E,
        (struct vertex_element*)(&V[local]),
        start_offset + 1,
        segment_log_entries, // <-- Usa o ponteiro correto e totalmente NUMA-aware
        onseg_edges
    );
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    // A lógica é idêntica à de out_neigh, então podemos simplesmente chamá-la.
    // Isto evita duplicação de código e garante que qualquer futura correção
    // seja aplicada a ambas.
    return out_neigh(n, start_offset);
  }
  #else
  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    // índice local e qual array usar
    bool first_node = (n < n_vertices_node0);
    int local_idx = first_node ? n : (n - n_vertices_node0);

    // ponteiros aos arrays particionados
    auto   &V    = first_node ? vertices_0 : vertices_1;
    auto   &E    = first_node ? edges_0    : edges_1;

    // próximo limite de vértice
    bool last_global      = (n == num_vertices - 1);
    bool last_in_node0    = (first_node && local_idx == n_vertices_node0 - 1);
    int64_t next_boundary = last_global
        ? (elem_capacity - 1)
        : (first_node && last_in_node0
            ? (vertices_1[0].index - 1)          // encontra o primeiro de vertices_1, pois passou do 0 pro 1
            : (V[local_idx + 1].index - 1));     // se não, dentro do mesmo array

    // contagem de arestas reais neste segmento
    int32_t onseg_edges = V[local_idx].degree;
    if (V[local_idx].offset != -1)
        onseg_edges = next_boundary - V[local_idx].index + 1;

    return Neighborhood(
        E,
        /* ponteiro pro elemento */ (struct vertex_element*)(&V[local_idx]),
        start_offset + 1,
        log_ptr_[ n / segment_size],
        onseg_edges
    );
}

Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    // mesma lógica acima
    bool first_node = (n < n_vertices_node0);
    int local_idx = first_node ? n : (n - n_vertices_node0);
    auto   &V    = first_node ? vertices_0 : vertices_1;
    auto   &E    = first_node ? edges_0    : edges_1;

    bool last_global      = (n == num_vertices - 1);
    bool last_in_node0    = (first_node && local_idx == n_vertices_node0 - 1);
    int64_t next_boundary = last_global
        ? (elem_capacity - 1)
        : (first_node && last_in_node0
            ? (vertices_1[0].index - 1)
            : (V[local_idx + 1].index - 1));

    int32_t onseg_edges = V[local_idx].degree;
    if (V[local_idx].offset != -1)
        onseg_edges = next_boundary - V[local_idx].index + 1;

    return Neighborhood(
        E,
        (struct vertex_element*)(&V[local_idx]),
        start_offset + 1,
        log_ptr_[ n / segment_size],
        onseg_edges
    );
}
#endif
  #else
  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    int32_t onseg_edges = vertices_[n].degree;
    int64_t next_vertex_boundary = (n >= num_vertices - 1) ? (elem_capacity - 1) : vertices_[n + 1].index - 1;
    if (vertices_[n].offset != -1) onseg_edges = next_vertex_boundary - vertices_[n].index + 1;
    return Neighborhood(edges_, (struct vertex_element *) (vertices_ + n), start_offset + 1, log_ptr_[n / segment_size],
                        onseg_edges);
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    int32_t onseg_edges = vertices_[n].degree;
    int64_t next_vertex_boundary = (n >= num_vertices - 1) ? (elem_capacity - 1) : vertices_[n + 1].index - 1;
    if (vertices_[n].offset != -1) onseg_edges = next_vertex_boundary - vertices_[n].index + 1;
    return Neighborhood(edges_, (struct vertex_element *) (vertices_ + n), start_offset + 1, log_ptr_[n / segment_size],
                        onseg_edges);
  }
  #endif



  

  #ifdef NUMA_PMEM //Fazer depois
  void PrintStats() const {
    std::cout << "Graph has " << n_vertices_node0 << "nodes on node 0 and"<< n_vertices_node1 <<" nodes on node 1 and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << (n_edges_node0 + n_edges_node1) / (n_vertices_node0 + n_vertices_node1) << std::endl;
  }

  void PrintTopology() const {
    return;
  }
  void PrintTopology(NodeID_ src) const {
    return;
  }
  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    pvector<SGOffset> offsets(num_vertices + 1);
    return offsets;
  }

  #else
  void PrintStats() const {
    std::cout << "Graph has " << num_vertices << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_ / num_vertices << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i = 0; i < num_vertices; i++) {
      if (i && i % 10000000 == 0) std::cout << i / 1000000 << " million vertices processed." << endl;
      for (DestID_ j: out_neigh(i)) {
        volatile DestID_ x = j;
      }
    }
  }

  void PrintTopology(NodeID_ src) const {
    std::cout << vertices_[src].index << " " << vertices_[src].degree << " " << vertices_[src].offset << std::endl;
    std::cout << src << ": ";
    for (DestID_ j: out_neigh(src)) {
      std::cout << j << " ";
    }
    std::cout << std::endl;
  }
  
  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    pvector<SGOffset> offsets(num_vertices + 1);
    for (NodeID_ n = 0; n < num_vertices + 1; n++)
      offsets[n] = vertices_[n].index - vertices_[0].index;
    return offsets;
  }
  #endif
  
  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

  #ifdef NUMA_PMEM
  inline void print_vertices() {
    for (int i = 0; i < num_vertices; i++) {
      if (i < n_vertices_node0) {
        printf("(%d)|%llu,%d| ", i, vertices_0[i].index, vertices_0[i].degree);
      }
      else{
        printf("(%d)|%llu,%d| ", i, vertices_1[i - n_vertices_node0].index, vertices_1[i - n_vertices_node0].degree);
      }
    }
    printf("\n");
  }

  inline void print_vertices(int32_t from, int32_t to) {
    for (int32_t i = from; i < to; i++) {
      if (i < n_vertices_node0) {
        printf("(%d)|%llu,%d| ", i, vertices_0[i].index, vertices_0[i].degree);
      }
      else{
        printf("(%d)|%llu,%d| ", i, vertices_1[i - n_vertices_node0].index, vertices_1[i - n_vertices_node0].degree);
      }
    }
    printf("\n");
  }

  inline void print_vertices(int32_t segment_id) {
    cout << "Print Vertices: ";
    int32_t from = (segment_id) * segment_size;
    int32_t to = (segment_id + 1) * segment_size;
    cout << from << " " << to << endl;
    print_vertices(from, to);
  }
  inline void print_vertex(int32_t vid) {
    if (vid < n_vertices_node0) {
      cout << "vertex-id: " << vid << "# index: " << vertices_0[vid].index;
    cout << ", degree: " << vertices_0[vid].degree << ", log-offset: " << vertices_0[vid].offset << endl;
    }
    else{
      cout << "vertex-id: " << vid << "# index: " << vertices_1[vid - n_vertices_node0].index;
      cout << ", degree: " << vertices_1[vid - n_vertices_node0].degree << ", log-offset: " << vertices_1[vid - n_vertices_node0].offset << endl;
    }
  }

  inline void print_edges() {
    cout << "Print Edges: ";
    for (int i = 0; i < elem_capacity; i++) {
      if (i < n_vertices_node0) {
        printf("%u ", edges_0[i].v);
      }
      else{
        printf("%u ", edges_1[i - n_vertices_node0].v);
      }
    }
    printf("\n");
  }

  inline void print_segment() {
    cout << "Print Segments: ";
    for (int i = 0; i < segment_count * 2; i++) {
      printf("(%d)|%llu / %llu| ", i, segment_edges_actual[i], segment_edges_total[i]);
    }
    printf("\n");
  }

  inline void print_segment(int segment_id) {
    printf("Segment (%d): |%llu / %llu|\n", segment_id, segment_edges_actual[segment_id],
           segment_edges_total[segment_id]);
  }


  inline void edge_list_boundary_sanity_checker() {
    for (int32_t curr_vertex = 1; curr_vertex < num_vertices; curr_vertex += 1) {
      if (curr_vertex < n_vertices_node0) {
        
        if (vertices_0[curr_vertex - 1].index + vertices_0[curr_vertex - 1].degree > vertices_0[curr_vertex].index) {
          cout << "**** Invalid edge-list boundary found at vertex-id: " << curr_vertex - 1 << " index: "
              << vertices_0[curr_vertex - 1].index;
          cout << " degree: " << vertices_0[curr_vertex - 1].degree << " next vertex start at: "
              << vertices_0[curr_vertex].index << endl;
        }
        assert(vertices_0[curr_vertex - 1].index + vertices_0[curr_vertex - 1].degree <= vertices_0[curr_vertex].index &&
              "Invalid edge-list boundary found!");
        
        assert(vertices_0[num_vertices - 1].index + vertices_0[num_vertices - 1].degree <= elem_capacity &&
            "Invalid edge-list boundary found!");
      }
      else {
        
        if (vertices_1[curr_vertex - 1 - n_vertices_node0].index + vertices_1[curr_vertex - 1 - n_vertices_node0].degree > vertices_1[curr_vertex - n_vertices_node0].index) {
          cout << "**** Invalid edge-list boundary found at vertex-id: " << curr_vertex - 1 << " index: "
              << vertices_1[curr_vertex - 1 - n_vertices_node0].index;
          cout << " degree: " << vertices_1[curr_vertex - 1 - n_vertices_node0].degree << " next vertex start at: "
              << vertices_1[curr_vertex - n_vertices_node0].index << endl;
        }
        assert(vertices_1[curr_vertex - 1 - n_vertices_node0].index + vertices_1[curr_vertex - 1 - n_vertices_node0].degree <= vertices_1[curr_vertex - n_vertices_node0].index &&
              "Invalid edge-list boundary found!");
        
        assert(vertices_1[num_vertices - 1 - n_vertices_node0].index + vertices_1[num_vertices - 1 - n_vertices_node0].degree <= elem_capacity &&
            "Invalid edge-list boundary found!");
      }
    }
  }
  #else
  inline void print_vertices() {
    for (int i = 0; i < num_vertices; i++) {
      printf("(%d)|%llu,%d| ", i, vertices_[i].index, vertices_[i].degree);
    }
    printf("\n");
  }

  inline void print_vertices(int32_t from, int32_t to) {
    for (int32_t i = from; i < to; i++) {
      printf("(%d)|%llu,%d| ", i, vertices_[i].index, vertices_[i].degree);
    }
    printf("\n");
  }

  inline void print_vertices(int32_t segment_id) {
    cout << "Print Vertices: ";
    int32_t from = (segment_id) * segment_size;
    int32_t to = (segment_id + 1) * segment_size;
    cout << from << " " << to << endl;
    print_vertices(from, to);
  }

  inline void print_vertex(int32_t vid) {
    cout << "vertex-id: " << vid << "# index: " << vertices_[vid].index;
    cout << ", degree: " << vertices_[vid].degree << ", log-offset: " << vertices_[vid].offset << endl;
  }

  inline void print_edges() {
    cout << "Print Edges: ";
    for (int i = 0; i < elem_capacity; i++) {
      printf("%u ", edges_[i].v);
    }
    printf("\n");
  }

  inline void print_segment() {
    cout << "Print Segments: ";
    for (int i = 0; i < segment_count * 2; i++) {
      printf("(%d)|%llu / %llu| ", i, segment_edges_actual[i], segment_edges_total[i]);
    }
    printf("\n");
  }

  inline void print_segment(int segment_id) {
    printf("Segment (%d): |%llu / %llu|\n", segment_id, segment_edges_actual[segment_id],
           segment_edges_total[segment_id]);
  }

  inline void edge_list_boundary_sanity_checker() {
    for (int32_t curr_vertex = 1; curr_vertex < num_vertices; curr_vertex += 1) {
      if (vertices_[curr_vertex - 1].index + vertices_[curr_vertex - 1].degree > vertices_[curr_vertex].index) {
        cout << "**** Invalid edge-list boundary found at vertex-id: " << curr_vertex - 1 << " index: "
             << vertices_[curr_vertex - 1].index;
        cout << " degree: " << vertices_[curr_vertex - 1].degree << " next vertex start at: "
             << vertices_[curr_vertex].index << endl;
      }
      assert(vertices_[curr_vertex - 1].index + vertices_[curr_vertex - 1].degree <= vertices_[curr_vertex].index &&
             "Invalid edge-list boundary found!");
    }
    assert(vertices_[num_vertices - 1].index + vertices_[num_vertices - 1].degree <= elem_capacity &&
           "Invalid edge-list boundary found!");
  }

  #endif

  inline void print_pma_meta() {
    cout << "max_size: " << max_size << ", num_edges: " << num_edges_ << ", num_vertices: " << num_vertices
         << ", avg_degree: " << avg_degree << ", elem_capacity: " << elem_capacity << endl;
    cout << "segment_count: " << segment_count << ", segment_size: " << segment_size << ", tree_height: " << tree_height
         << endl;
  }

  /*****************************************************************************
   *                                                                           *
   *   PMA                                                                     *
   *                                                                           *
   *****************************************************************************/
  /// Double the size of the "edges_" array
  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  /// Redimensiona os arrays de arestas, dobrando sua capacidade.
/// Adaptado para NUMA.
void resize_V1() {
    // Dobra as capacidades de cada nó e a global.
    int64_t old_elem_capacity0 = elem_capacity0;
    int64_t old_elem_capacity1 = elem_capacity1;
    elem_capacity0 *= 2;
    elem_capacity1 *= 2;
    elem_capacity *= 2;

    // Calcula as novas posições para todos os vértices no novo espaço.
    int64_t gaps = elem_capacity - num_edges_;
    int64_t* new_indices = calculate_positions(0, num_vertices, gaps, num_edges_);

    // Aloca os novos arrays de arestas na PMem de seus respectivos nós.
    PMEMoid new_edges_oid_0, new_edges_oid_1;
    if (pmemobj_zalloc(pop0, &new_edges_oid_0, elem_capacity0 * sizeof(DestID_), EDGE_TYPE)) abort();
    if (pmemobj_zalloc(pop1, &new_edges_oid_1, elem_capacity1 * sizeof(DestID_), EDGE_TYPE)) abort();
    DestID_* new_edges_0 = (DestID_*)pmemobj_direct(new_edges_oid_0);
    DestID_* new_edges_1 = (DestID_*)pmemobj_direct(new_edges_oid_1);

    // Copia os dados dos arrays antigos para os novos.
    for (NodeID_ vi = 0; vi < num_vertices; ++vi) {
        bool is_node0 = (vi % 2 == 0);
        int32_t local_vi = vi / 2;

        auto& V = is_node0 ? vertices_0 : vertices_1;
        auto& old_E = is_node0 ? edges_0 : edges_1;
        auto& new_E = is_node0 ? new_edges_0 : new_edges_1;

        int64_t new_pos = new_indices[vi];
        int32_t degree = V[local_vi].degree;

        // Copia as arestas que estavam no array principal.
        memcpy(&new_E[new_pos], &old_E[V[local_vi].index], degree * sizeof(DestID_));

        // Se havia arestas no log, elas já foram consideradas no grau,
        // mas precisamos garantir que a lógica de cópia as inclua.
        // A implementação atual assume que `degree` é a contagem total.

        V[local_vi].index = new_pos;
        V[local_vi].offset = -1;
    }

    // Persiste os novos arrays.
    flush_clwb_nolog(new_edges_0, elem_capacity0 * sizeof(DestID_));
    flush_clwb_nolog(new_edges_1, elem_capacity1 * sizeof(DestID_));

    // Libera os arrays antigos e atualiza os OIDs no objeto base.
    pmemobj_free(&bp0->edges_oid_);
    bp0->edges_oid_ = new_edges_oid_0;
    flush_clwb_nolog(&bp0->edges_oid_, sizeof(PMEMoid));

    pmemobj_free(&bp1->edges_oid_);
    bp1->edges_oid_ = new_edges_oid_1;
    flush_clwb_nolog(&bp1->edges_oid_, sizeof(PMEMoid));

    // Atualiza os ponteiros em DRAM.
    edges_0 = new_edges_0;
    edges_1 = new_edges_1;

    // Atualiza metadados e limpa todos os logs.
    recount_segment_total();
    free(new_indices);
    for (int32_t i = 0; i < segment_count; ++i) {
        release_log(i);
    }

    bp0->elem_capacity = elem_capacity0;
    flush_clwb_nolog(&bp0->elem_capacity, sizeof(int64_t));
    bp1->elem_capacity = elem_capacity1;
    flush_clwb_nolog(&bp1->elem_capacity, sizeof(int64_t));
}
#else

  void resize_V1() {
    elem_capacity *= 2;
    elem_capacity0 *= 2;
    elem_capacity1 *= 2;
    int64_t gaps = (elem_capacity0 + elem_capacity1) - (n_edges_node0 + n_edges_node1);
    int64_t *new_indices = calculate_positions(0, n_vertices_node0 + n_vertices_node1, gaps, (n_edges_node0 + n_edges_node1));

    PMEMoid new_edges_oid_0 = OID_NULL;
    if (pmemobj_zalloc(pop0, &new_edges_oid_0, elem_capacity0 * sizeof(DestID_), EDGE_TYPE)) {
      fprintf(stderr, "[%s]: FATAL: edge array allocation node 0 failed: %s\n", __func__, pmemobj_errormsg());
      abort();
    }
    DestID_ *new_edges_0 = (DestID_ *) pmemobj_direct(new_edges_oid_0);
  
    // Aloca novo array de arestas para o nó 1
    PMEMoid new_edges_oid_1 = OID_NULL;
    if (pmemobj_zalloc(pop1, &new_edges_oid_1, elem_capacity1 * sizeof(DestID_), EDGE_TYPE)) {
      fprintf(stderr, "[%s]: FATAL: edge array allocation node 1 failed: %s\n", __func__, pmemobj_errormsg());
      abort();
    }
    DestID_ *new_edges_1 = (DestID_ *) pmemobj_direct(new_edges_oid_1);

    int64_t write_index;
    int32_t curr_off, curr_seg, onseg_num_edges;
    int64_t next_vertex_boundary;
    for (NodeID_ vi = 0; vi < num_vertices; vi += 1) {
      if(vi +1 < n_vertices_node0){
        next_vertex_boundary = (vi == num_vertices - 1) ? (elem_capacity0 - 1) : vertices_0[vi + 1].index - 1;
      }
      else{
        if((vi + 1) == n_vertices_node0){
          next_vertex_boundary = (vi == num_vertices - 1) ? (elem_capacity1 - 1) : vertices_1[0].index - 1;
        }
        else{
          next_vertex_boundary = (vi == num_vertices - 1) ? (elem_capacity1 - 1) : vertices_1[vi + 1 - n_vertices_node0].index - 1;
        }
      }
      if(vi < n_vertices_node0){
      // count on-segment number of edges for vertex-vi
      if (vertices_0[vi].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_0[vi].index + 1;
      else onseg_num_edges = vertices_0[vi].degree;

      memcpy((new_edges_0 + new_indices[vi]), (edges_0 + (vertices_0[vi].index)), onseg_num_edges * sizeof(DestID_));

      // if vertex-vi have edges in the log, move it to on-segment
      if (vertices_0[vi].offset != -1) {
        curr_off = vertices_0[vi].offset;
        curr_seg = get_segment_id(vi) - segment_count;

        write_index = new_indices[vi] + vertices_0[vi].degree - 1;
        while ((curr_off != -1)) {
          new_edges_0[write_index].v = log_ptr_[curr_seg][curr_off].v;
          curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
          write_index--;
        }
      }

      // update the index to the new position
      vertices_0[vi].index = new_indices[vi];
      vertices_0[vi].offset = -1;
      }
      else{
        // count on-segment number of edges for vertex-vi
        if (vertices_1[vi - n_vertices_node0].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_1[vi - n_vertices_node0].index + 1;
        else onseg_num_edges = vertices_1[vi - n_vertices_node0].degree;
  
        memcpy((new_edges_1 + new_indices[vi]), (edges_1 + (vertices_1[vi - n_vertices_node0].index)), onseg_num_edges * sizeof(DestID_));
  
        // if vertex-vi have edges in the log, move it to on-segment
        if (vertices_1[vi - n_vertices_node0].offset != -1) {
          curr_off = vertices_1[vi - n_vertices_node0].offset;
          curr_seg = get_segment_id(vi) - segment_count;
  
          write_index = new_indices[vi] + vertices_1[vi - n_vertices_node0].degree - 1;
          while ((curr_off != -1)) {
            //printf("log_ptr_[curr_seg][curr_off].prev_offset: %d",log_ptr_[curr_seg][curr_off].prev_offset);
            new_edges_1[write_index].v = log_ptr_[curr_seg][curr_off].v;
            curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
            write_index--;
          }
        }
  
        // update the index to the new position
        vertices_1[vi - n_vertices_node0].index = new_indices[vi];
        vertices_1[vi - n_vertices_node0].offset = -1;
        }
    }
    flush_clwb_nolog(new_edges_0, elem_capacity0 * sizeof(DestID_));
    flush_clwb_nolog(new_edges_1, elem_capacity1 * sizeof(DestID_));

    pmemobj_free(&bp0->edges_oid_);
    bp0->edges_oid_ = OID_NULL;
    bp0->edges_oid_ = new_edges_oid_0;
    flush_clwb_nolog(&bp0->edges_oid_, sizeof(PMEMoid));

    pmemobj_free(&bp1->edges_oid_);
    bp1->edges_oid_ = OID_NULL;
    bp1->edges_oid_ = new_edges_oid_1;
    flush_clwb_nolog(&bp1->edges_oid_, sizeof(PMEMoid));

    edges_0 = (DestID_ *) pmemobj_direct(bp0->edges_oid_);
    edges_1 = (DestID_ *) pmemobj_direct(bp1->edges_oid_);
    recount_segment_total();
    free(new_indices);
    new_indices = nullptr;

    bp0->elem_capacity = elem_capacity0;
    flush_clwb_nolog(&bp0->elem_capacity, sizeof(int64_t));
    bp1->elem_capacity = elem_capacity1;
    flush_clwb_nolog(&bp1->elem_capacity, sizeof(int64_t));

    int32_t st_seg = segment_count, nd_seg = 2 * segment_count;
    for (int32_t i = st_seg; i < nd_seg; i += 1) {
      release_log(i - segment_count);
    }
  }
  #endif
  #else
  void resize_V1() {
    elem_capacity *= 2;
    int64_t gaps = elem_capacity - num_edges_;
    int64_t *new_indices = calculate_positions(0, num_vertices, gaps, num_edges_);

    PMEMoid new_edges_oid_ = OID_NULL;
    if (pmemobj_zalloc(pop, &new_edges_oid_, elem_capacity * sizeof(DestID_), EDGE_TYPE)) {
      fprintf(stderr, "[%s]: FATAL: edge array allocation failed: %s\n", __func__, pmemobj_errormsg());
      abort();
    }
    DestID_ *new_edges_ = (DestID_ *) pmemobj_direct(new_edges_oid_);

    int64_t write_index;
    int32_t curr_off, curr_seg, onseg_num_edges;
    int64_t next_vertex_boundary;
    for (NodeID_ vi = 0; vi < num_vertices; vi += 1) {
      next_vertex_boundary = (vi == num_vertices - 1) ? (elem_capacity - 1) : vertices_[vi + 1].index - 1;

      // count on-segment number of edges for vertex-vi
      if (vertices_[vi].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_[vi].index + 1;
      else onseg_num_edges = vertices_[vi].degree;

      memcpy((new_edges_ + new_indices[vi]), (edges_ + (vertices_[vi].index)), onseg_num_edges * sizeof(DestID_));

      // if vertex-vi have edges in the log, move it to on-segment
      if (vertices_[vi].offset != -1) {
        curr_off = vertices_[vi].offset;
        curr_seg = get_segment_id(vi) - segment_count;

        write_index = new_indices[vi] + vertices_[vi].degree - 1;
        while (curr_off != -1) {
          new_edges_[write_index].v = log_ptr_[curr_seg][curr_off].v;
          curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
          write_index--;
        }
      }

      // update the index to the new position
      vertices_[vi].index = new_indices[vi];
      vertices_[vi].offset = -1;
    }

    flush_clwb_nolog(new_edges_, elem_capacity * sizeof(DestID_));

    pmemobj_free(&bp->edges_oid_);
    bp->edges_oid_ = OID_NULL;
    bp->edges_oid_ = new_edges_oid_;
    flush_clwb_nolog(&bp->edges_oid_, sizeof(PMEMoid));

    edges_ = (DestID_ *) pmemobj_direct(bp->edges_oid_);
    recount_segment_total();
    free(new_indices);
    new_indices = nullptr;

    bp->elem_capacity = elem_capacity;
    flush_clwb_nolog(&bp->elem_capacity, sizeof(int64_t));

    int32_t st_seg = segment_count, nd_seg = 2 * segment_count;
    for (int32_t i = st_seg; i < nd_seg; i += 1) {
      release_log(i - segment_count);
    }
  }
  #endif

  inline int32_t get_segment_id(int32_t vertex_id) {
    #ifdef NUMA_PMEM
    return (vertex_id / segment_size) + segment_count;
    #else
    return (vertex_id / segment_size) + segment_count;
    #endif
  }

  void reconstruct_pma_tree() {
    recount_segment_actual();
    recount_segment_total();
  }

 
  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  void recount_segment_actual() {
    // Zera os contadores de arestas atuais para ambos os nós.
    memset(segment_edges_actual_0, 0, sizeof(int64_t) * segment_count0 * 2);
    memset(segment_edges_actual_1, 0, sizeof(int64_t) * segment_count1 * 2);
    n_edges_node0 = 0;
    n_edges_node1 = 0;

    // Itera sobre todos os vértices do grafo.
    for (int32_t vid = 0; vid < num_vertices; ++vid) {
        // Determina a qual nó o VÉRTICE pertence para obter seu grau.
        bool vertex_on_node0 = (vid % 2 == 0);
        int32_t local_vid = vid / 2;
        int32_t degree = 0;

        if (vertex_on_node0) {
            if (local_vid < n_vertices_node0) {
                degree = vertices_0[local_vid].degree;
                n_edges_node0 += degree;
            }
        } else {
            if (local_vid < n_vertices_node1) {
                degree = vertices_1[local_vid].degree;
                n_edges_node1 += degree;
            }
        }

        if (degree == 0) continue;

        // Determina a qual nó o SEGMENTO do vértice pertence para atualizar o contador correto.
        int32_t global_seg_id = get_segment_id(vid) - segment_count;
        bool seg_on_node0 = (global_seg_id < segment_count0);

        if (seg_on_node0) {
            // O índice no array da árvore PMA é o ID global + o offset do início das folhas.
            segment_edges_actual_0[global_seg_id + segment_count] += degree;
        } else {
            int32_t local_seg_id = global_seg_id - segment_count0;
            segment_edges_actual_1[local_seg_id + segment_count] += degree;
        }
    }
}
#else
   // Expected to run in single thread
   void recount_segment_actual() {
    // count the size of each segment in the tree
    num_edges_ = 0;
    memset(segment_edges_actual_0, 0, sizeof(int64_t) * segment_count);
    memset(segment_edges_actual_1, 0, sizeof(int64_t) * segment_count);
    int32_t end_vertex;
    int32_t start_vertex;
    int32_t segment_actual_p;
    int32_t j;
    for (int seg_id = 0; seg_id < segment_count; seg_id++) {
      if(seg_id < segment_count0){
        start_vertex = (seg_id * segment_size);
        end_vertex = min(((seg_id + 1) * segment_size), n_vertices_node0);
    
        segment_actual_p = 0;
        for (int32_t vid = start_vertex; vid < end_vertex; vid += 1) {
          segment_actual_p += vertices_0[vid].degree;
          num_edges_ += vertices_0[vid].degree;
        }
    
        j = seg_id + segment_count;  //tree leaves
        segment_edges_actual_0[j] = segment_actual_p;

      }
      else{
        start_vertex = (seg_id * segment_size);
        end_vertex = min(((seg_id + 1) * segment_size), n_vertices_node1);
    
        segment_actual_p = 0;
        for (int32_t vid = start_vertex; vid < end_vertex; vid += 1) {
          segment_actual_p += vertices_1[vid - n_vertices_node0].degree;
          num_edges_ += vertices_0[vid - n_vertices_node0].degree;
        }
    
        j = seg_id + segment_count;  //tree leaves
        segment_edges_actual_1[j - segment_count0] = segment_actual_p;
      } 
    }
  }
  #endif
  #ifdef HASH_MODE
  void recount_segment_total() {
    // Zera todo o array de totais (nós internos + folhas)
    memset(segment_edges_total,
           0,
           sizeof(int64_t) * segment_count * 2);

    // Deslocamento para as folhas na árvore de segmentos
    // (as folhas começam em index = segment_count)
    int leaf_offset = segment_count;
    int32_t total_vertices = n_vertices_node0 + n_vertices_node1;

    for (int i = 0; i < segment_count; ++i) {
        // faixa de vértices deste segmento [start_vid, end_vid)
        int32_t start_vid = i * segment_size;
        int32_t end_vid   = (i == segment_count - 1)
                          ? total_vertices
                          : ((i + 1) * segment_size);

        // índice de leitura do primeiro vértice (start_vid)
        int64_t start_idx;
        if ((start_vid & 1) == 0) {
            // even → node0
            start_idx = vertices_0[start_vid / 2].index;
        } else {
            // odd → node1
            start_idx = vertices_1[start_vid / 2].index;
        }

        // índice de início do próximo segmento (boundary)
        int64_t next_starter;
        if (end_vid >= total_vertices) {
            // além do último vértice → fim global
            next_starter = elem_capacity0 + elem_capacity1;
        } else if ((end_vid & 1) == 0) {
            next_starter = vertices_0[end_vid / 2].index;
        } else {
            next_starter = vertices_1[end_vid / 2].index;
        }

        // total de arestas no segmento = diferença de offsets
        int64_t segment_total = next_starter - start_idx;

        // armazena na folha correspondente
        segment_edges_total[leaf_offset + i] = segment_total;
    }
}

/// Essencial para calcular o tamanho de um segmento no modo hash.
inline int64_t get_global_offset(int32_t vid) const {
    // Se o ID do vértice estiver fora dos limites, retorna a capacidade total.
    if (vid >= num_vertices) {
        return elem_capacity;
    }

    bool is_node0 = (vid % 2 == 0);
    int32_t local_id = vid / 2;

    if (is_node0) {
        // Para vértices no nó 0, o índice já é o offset global.
        return vertices_0[local_id].index;
    } else {
        // Para vértices no nó 1, o índice é local para edges_1.
        // Somamos a capacidade de edges_0 para obter o offset global.
        return elem_capacity0 + vertices_1[local_id].index;
    }
}


/// Recalcula a capacidade total de arestas para os segmentos em um dado intervalo de vértices.
/// Adaptado para o modo HASH.
void recount_segment_total(int32_t start_vertex, int32_t end_vertex) {
    // Determina o intervalo de segmentos a serem atualizados.
    // O ID do segmento é o ID global (de 0 a segment_count-1).
    int32_t start_seg = get_segment_id(start_vertex) - segment_count;
    // O segmento final é o que contém o último vértice do intervalo.
    int32_t end_seg = get_segment_id(end_vertex - 1) - segment_count;

    // Itera sobre cada segmento que precisa de atualização.
    for (int32_t i = start_seg; i <= end_seg; ++i) {
        // Encontra o ID do primeiro vértice deste segmento e do próximo.
        int32_t v_current_seg_start = i * segment_size;
        int32_t v_next_seg_start = (i + 1) * segment_size;

        // Calcula o offset global para o início do segmento atual e do próximo.
        int64_t current_pos = get_global_offset(v_current_seg_start);
        int64_t next_pos = get_global_offset(v_next_seg_start);

        // A capacidade total do segmento é a diferença entre o início do próximo e o início do atual.
        int64_t segment_total_p = next_pos - current_pos;

        // O índice no array da árvore PMA é o ID global do segmento + o offset das folhas.
        int32_t j = i + segment_count;
        segment_edges_total[j] = segment_total_p;
    }
}


  #else
  void recount_segment_total() {
    // Zera o array de total de arestas para a partição do nó 0.
    memset(segment_edges_total, 0, sizeof(int64_t) * segment_count * 2);
    int32_t j;
    int64_t next_starter;
    int64_t segment_total_p;
    int64_t aux;
    for (int i = 0; i < segment_count; i++) {
      // Determina o índice inicial do próximo segmento ou usa elem_capacity para o último.
      if(i == segment_count - 1){
        next_starter = elem_capacity;
      }
      else{
        aux = (i + 1) * segment_size;
        if(aux < n_vertices_node0){
          next_starter = vertices_0[aux].index;
        }
        else{
          next_starter = vertices_1[aux - n_vertices_node0].index;
        }
      }
        
      if((i * segment_size) < n_vertices_node0){
        segment_total_p = next_starter - vertices_0[i * segment_size].index;
        // Armazena o total para a folha correspondente (segmento da árvore).
        j = i + segment_count0;  // índice na parte das folhas da árvore

      }
      else{
        segment_total_p = next_starter - vertices_1[(i * segment_size) - n_vertices_node0].index;
        // Armazena o total para a folha correspondente (segmento da árvore).
        j = i + segment_count1;  // índice na parte das folhas da árvore
      }
        
      segment_edges_total[j] = segment_total_p;
    }
}

void recount_segment_total(int32_t start_vertex, int32_t end_vertex) {
  int32_t start_seg = get_segment_id(start_vertex);
  int32_t end_seg = get_segment_id(end_vertex);
  int64_t next_starter;
  int64_t segment_total_p;
  int32_t j;
  for (int32_t i = start_seg; i < end_seg; i += 1) {
    if(i < segment_count0){
      next_starter = (i == (segment_count0 - 1)) ? (elem_capacity0) : vertices_0[(i + 1) * segment_size].index;
      segment_total_p = next_starter - vertices_0[i * segment_size].index;
      j = i + segment_count0;  //tree leaves
      segment_edges_total[j] = segment_total_p;
    }
    else{
      next_starter = (i == (segment_count0 + segment_count1 - 1)) ? (elem_capacity1 + elem_capacity0) : vertices_1[((i + 1) * segment_size) - n_vertices_node0].index;
      segment_total_p = next_starter - vertices_1[(i * segment_size) - n_vertices_node0].index;
      j = i + segment_count1;  //tree leaves
      segment_edges_total[j] = segment_total_p;
    }
    
  }
}
#endif
#else
 // Expected to run in single thread
 void recount_segment_actual() {
  // count the size of each segment in the tree
  num_edges_ = 0;
  memset(segment_edges_actual, 0, sizeof(int64_t) * segment_count * 2);
  for (int seg_id = 0; seg_id < segment_count; seg_id++) {
    int32_t start_vertex = (seg_id * segment_size);
    int32_t end_vertex = min(((seg_id + 1) * segment_size), num_vertices);

    int32_t segment_actual_p = 0;
    for (int32_t vid = start_vertex; vid < end_vertex; vid += 1) {
      segment_actual_p += vertices_[vid].degree;
      num_edges_ += vertices_[vid].degree;
    }

    int32_t j = seg_id + segment_count;  //tree leaves
    segment_edges_actual[j] = segment_actual_p;
  }
}
  // Expected to run in single thread
  void recount_segment_total() {
    // count the size of each segment in the tree
    memset(segment_edges_total, 0, sizeof(int64_t) * segment_count * 2);
    for (int i = 0; i < segment_count; i++) {
      int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : vertices_[(i + 1) * segment_size].index;
      int64_t segment_total_p = next_starter - vertices_[i * segment_size].index;
      int32_t j = i + segment_count;  //tree leaves
      segment_edges_total[j] = segment_total_p;
    }
  }
  // Expected to run in single thread
  void recount_segment_total(int32_t start_vertex, int32_t end_vertex) {
    int32_t start_seg = get_segment_id(start_vertex) - segment_count;
    int32_t end_seg = get_segment_id(end_vertex) - segment_count;

    for (int32_t i = start_seg; i < end_seg; i += 1) {
      int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : vertices_[(i + 1) * segment_size].index;
      int64_t segment_total_p = next_starter - vertices_[i * segment_size].index;
      int32_t j = i + segment_count;  //tree leaves
      segment_edges_total[j] = segment_total_p;
    }
  }
#endif
  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  void insert(const EdgeList &edge_list) {
    printf("HashMode\n");
    // Contador de aresta de cada nó
    int64_t ii0 = 0, ii1 = 0;
    // Último vertice visto, para cada nó, para evitar vertices sem indice
    NodeID_ last_vid0 = -node_counter;      // = -2  → first even = 0
    NodeID_ last_vid1 = -node_counter + 1;  // = -1  → first odd  = 1

    // Zera todas as posições e seta o offset para -1(Não aponta para nada no edge array)
    for (int i = 0; i < n_vertices_node0; i++) {
        vertices_0[i].degree = 0;
        vertices_0[i].offset = -1;
    }
    for (int i = 0; i < n_vertices_node1; i++) {
        vertices_1[i].degree = 0;
        vertices_1[i].offset = -1;
    }
    // Zera contadores de arestas por segmento
    memset(segment_edges_actual_0, 0, sizeof(segment_edges_actual_0));
    memset(segment_edges_actual_1, 0, sizeof(segment_edges_actual_1));

    // Loop principal, le uma aresta e adiciona o vértice e a aresta
    for (int i = 0; i < num_edges_; i++) {
        NodeID_ t_src = edge_list[i].u;
        DestID_ t_dst = edge_list[i].v.v;
        int32_t seg_abs = get_segment_id(t_src);

        // round-robin: Pares no 0, Impares no 1
        bool is0 = !(t_src % node_counter); //Como é bool, é o contrário do resultado
        int   local_src = t_src / node_counter;  
        int   seg_loc   = is0 ? seg_abs
                              : (seg_abs - segment_count0);

        int32_t &deg = is0 // Se verdadeiro, pega o grau do node 0, se não, do 1
            ? vertices_0[local_src].degree  : vertices_1[local_src].degree;

        if (deg == 0) {
            NodeID_ &last_vid = is0 ? last_vid0 : last_vid1; //Pega o ultimo vertice do nó correto
            // Preenche os espaços faltando
            for (NodeID_ vid = last_vid + node_counter;
                 vid < t_src; vid += node_counter){
                int local_vid = vid / node_counter;
                if (is0) {
                    edges_0[ii0].v                 = -vid;
                    vertices_0[local_vid].degree  = 1;
                    vertices_0[local_vid].index   = ii0;
                    segment_edges_actual_0[get_segment_id(vid)]++;
                    ii0++;
                } else {
                    edges_1[ii1].v                 = -vid;
                    vertices_1[local_vid].degree  = 1;
                    vertices_1[local_vid].index   = ii1;
                    segment_edges_actual_1[
                      get_segment_id(vid) - segment_count0]++;
                    ii1++;
                }
            }
            // Marca o fim do edgeList de cada vértice, com -1
            if (is0) {
                edges_0[ii0].v                = -t_src;
                vertices_0[local_src].degree = 1; //Soma o grau do vertice add
                vertices_0[local_src].index  = ii0; //Posição do vértice na edge list
                segment_edges_actual_0[seg_abs]++;
                ii0++;
                last_vid0 = t_src;
            } else {
                edges_1[ii1].v                = -t_src;
                vertices_1[local_src].degree = 1;
                vertices_1[local_src].index  = ii1;
                segment_edges_actual_1[seg_abs - segment_count0]++;
                ii1++;
                last_vid1 = t_src;
            }
        }

        // Adiciona a Aresta em si
        if (is0) {
            edges_0[ii0].v                = t_dst;
            vertices_0[local_src].degree += 1;
            segment_edges_actual_0[seg_abs]++;
            ii0++;
        } else {
            edges_1[ii1].v                = t_dst;
            vertices_1[local_src].degree += 1;
            segment_edges_actual_1[seg_abs - segment_count0]++;
            ii1++;
        }
    }

    // Completa espaços vazios novamente para o nó 0
    for (NodeID_ vid = last_vid0 + node_counter;
         vid < num_vertices;
         vid += node_counter)
    {
        int local_vid = vid / node_counter;
        edges_0[ii0].v                = -vid;
        vertices_0[local_vid].degree = 1;
        vertices_0[local_vid].index  = ii0;
        segment_edges_actual_0[get_segment_id(vid)]++;
        ii0++;
    }
    // Para o nó 1
    for (NodeID_ vid = last_vid1 + node_counter;
         vid < num_vertices;
         vid += node_counter)
    {
        int local_vid = vid / node_counter;
        edges_1[ii1].v                = -vid;
        vertices_1[local_vid].degree = 1;
        vertices_1[local_vid].index  = ii1;
        segment_edges_actual_1[
          get_segment_id(vid) - segment_count0]++;
        ii1++;
    }

    // Atualiza os contadores
    n_edges_node0 = ii0;
    n_edges_node1 = ii1;
    elem_capacity0 = ii0;
    elem_capacity1 = ii1;
    printf("Vertices node 0: %d\nVertices node 1: %d\nArestas node 0: %ld\nArestas node 1: %ld\n",n_vertices_node0, n_vertices_node1, n_edges_node0, n_edges_node1);
    spread_weighted(0, num_vertices); // Reorganiza os indices
    flush_clwb_nolog(edges_0, sizeof(DestID_) * elem_capacity0);
    flush_clwb_nolog(edges_1, sizeof(DestID_) * elem_capacity1);
}

#else
  void insert(const EdgeList &edge_list) {
    printf("Only NUMAMode\n");
    // Contador de aresta de cada nó
    int64_t ii0 = 0, ii1 = 0;
    // Último vertice visto, para cada nó, para evitar vertices sem indice
    NodeID_ last_vid0 = -1;
    NodeID_ last_vid1 = n_vertices_node0 - 1;

    // Zera todas as posições e seta o offset para -1(Não aponta para nada no edge array)
    for (int i = 0; i < n_vertices_node0; i++) {
        vertices_0[i].degree = 0;
        vertices_0[i].offset = -1;
    }
    for (int i = 0; i < n_vertices_node1; i++) {
        vertices_1[i].degree = 0;
        vertices_1[i].offset = -1;
    }
    // Zera contadores de arestas por segmento
    memset(segment_edges_actual_0, 0, sizeof(segment_edges_actual_0));
    memset(segment_edges_actual_1, 0, sizeof(segment_edges_actual_1));

    // Loop principal, le uma aresta e adiciona o vértice e a aresta
    for (int i = 0; i < num_edges_; i++) {
        NodeID_ t_src = edge_list[i].u;
        DestID_ t_dst = edge_list[i].v.v;
        int32_t seg_id = get_segment_id(t_src);

        // Testa em qual nó ficará, meio a meio, de 0 até a metade, nó 0, metade até o fim, nó 1
        bool is0      = (t_src < n_vertices_node0);
        int   local_src = is0 ? t_src : (t_src - n_vertices_node0);

        int32_t &deg = is0 // Se verdadeiro, pega o grau do node 0, se não, do 1
            ? vertices_0[local_src].degree
            : vertices_1[local_src].degree;

        if (deg == 0) {
            NodeID_ &last_vid = is0 ? last_vid0 : last_vid1; //Pega o ultimo vertice do nó correto
            // Preenche os espaços faltando
            for (NodeID_ vid = last_vid + 1; vid < t_src; vid++) {
                bool fill0    = (vid < n_vertices_node0);
                int local_vid = fill0 ? vid : (vid - n_vertices_node0);
                if (fill0) {
                    edges_0[ii0].v                     = -vid;
                    vertices_0[local_vid].degree     = 1;
                    vertices_0[local_vid].index      = ii0;
                    segment_edges_actual_0[get_segment_id(vid)]++;
                    ii0++;
                } else {
                    edges_1[ii1].v                     = -vid;
                    vertices_1[local_vid].degree     = 1;
                    vertices_1[local_vid].index      = ii1;
                    segment_edges_actual_1[get_segment_id(vid) - segment_count0]++;
                    ii1++;
                }
            }
            // Marca o fim do edgeList de cada vértice, com -1
            if (is0) {
                edges_0[ii0].v                     = -t_src;
                vertices_0[local_src].degree     = 1; //Soma o grau do vertice add
                vertices_0[local_src].index      = ii0; //Posição do vértice na edge list
                segment_edges_actual_0[seg_id]++;
                ii0++;
                last_vid0 = t_src;
            } else {
                edges_1[ii1].v                     = -t_src;
                vertices_1[local_src].degree     = 1;
                vertices_1[local_src].index      = ii1;
                segment_edges_actual_1[seg_id - segment_count0]++;
                ii1++;
                last_vid1 = t_src;
            }
        }

        // Adiciona a Aresta em si
        if (is0) {
            edges_0[ii0].v                     = t_dst;
            vertices_0[local_src].degree     += 1;
            segment_edges_actual_0[seg_id]++;
            ii0++;
        } else {
            edges_1[ii1].v                     = t_dst;
            vertices_1[local_src].degree     += 1;
            segment_edges_actual_1[seg_id - segment_count0]++;
            ii1++;
        }
    }

    // Completa espaços vazios novamente para o nó 0
    for (NodeID_ vid = last_vid0 + 1; vid < n_vertices_node0; vid++) {
        edges_0[ii0].v                     = -vid;
        vertices_0[vid].degree            = 1;
        vertices_0[vid].index             = ii0;
        segment_edges_actual_0[get_segment_id(vid)]++;
        ii0++;
    }
    for (NodeID_ vid = last_vid1 + 1; vid < n_vertices_node0 + n_vertices_node1; vid++) {
        int local_vid = vid - n_vertices_node0;
        edges_1[ii1].v                     = -vid;
        vertices_1[local_vid].degree      = 1;
        vertices_1[local_vid].index       = ii1;
        segment_edges_actual_1[get_segment_id(vid) - segment_count0]++;
        ii1++;
    }

    // Atualiza os contadores
    n_edges_node0 = ii0;
    n_edges_node1 = ii1;
    elem_capacity0 = ii0;
    elem_capacity1 = ii1;
    printf("Vertices node 0: %d\nVertices node 1: %d\nArestas node 0: %ld\nArestas node 1: %ld\n",n_vertices_node0, n_vertices_node1, n_edges_node0, n_edges_node1);
    spread_weighted(0, num_vertices);// Reorganiza os indices
    flush_clwb_nolog(edges_0, sizeof(DestID_) * elem_capacity0);
    flush_clwb_nolog(edges_1, sizeof(DestID_) * elem_capacity1);
}
#endif
  #else
  /// Insert base-graph. Assume all the edges of a vertex comes together (COO format).
  void insert(const EdgeList &edge_list) {
    NodeID_ last_vid = -1;
    int64_t ii = 0;
    for (int i = 0; i < num_edges_; i++) {
      int32_t t_src = edge_list[i].u;
      int32_t t_dst = edge_list[i].v.v;

      int32_t t_segment_id = get_segment_id(t_src);

      int32_t t_degree = vertices_[t_src].degree;
      if (t_degree == 0) {
        if (t_src != last_vid + 1) {
          for (NodeID_ vid = last_vid + 1; vid < t_src; vid += 1) {
            edges_[ii].v = -vid;
            vertices_[vid].degree = 1;
            vertices_[vid].index = ii;
            ii += 1;

            segment_edges_actual[get_segment_id(vid)] += 1;
          }
        }
        edges_[ii].v = -t_src;
        vertices_[t_src].degree = 1;
        vertices_[t_src].index = ii;
        ii += 1;

        segment_edges_actual[t_segment_id] += 1;
        last_vid = t_src;
      }

      edges_[ii].v = t_dst;
      ii += 1;
      vertices_[t_src].degree += 1;

      // update the actual edges in each segment of the tree
      segment_edges_actual[t_segment_id] += 1;
    }

    // correct the starting of the vertices with 0 degree in the base-graph
    for (int i = 0; i < num_vertices; i++) {
      if (vertices_[i].degree == 0) {
        assert(i > last_vid && "Something went wrong! We should not leave some zero-degree vertices before last_vid!");
        edges_[ii].v = -i;
        vertices_[i].degree = 1;
        vertices_[i].index = ii;
        ii += 1;

        segment_edges_actual[get_segment_id(i)] += 1;
      }
      vertices_[i].offset = -1;
    }

    num_edges_ += num_vertices;
    spread_weighted(0, num_vertices);
    flush_clwb_nolog(edges_, sizeof(DestID_) * elem_capacity);
  }
  #endif

  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
    /// Adaptado para NUMA.
bool have_space_onseg(int32_t src, int64_t loc) const {
    // Lógica in-line para determinar o nó e o índice local
    bool is_node0 = (src % 2 == 0);
    int32_t local_src = src / 2;

    if (is_node0) {
        // Lógica para o nó 0
        // Verifica se é o último vértice no nó 0 ou se há espaço antes do próximo vértice.
        return (local_src == (n_vertices_node0 - 1) && elem_capacity0 > loc) ||
               (local_src < (n_vertices_node0 - 1) && vertices_0[local_src + 1].index > loc);
    } else {
        // Lógica para o nó 1
        return (local_src == (n_vertices_node1 - 1) && elem_capacity1 > loc) ||
               (local_src < (n_vertices_node1 - 1) && vertices_1[local_src + 1].index > loc);
    }
}


/// Insere uma aresta no log do segmento apropriado.
/// Adaptado para NUMA.
inline void insert_into_log(int32_t segment_id, int32_t src, int32_t dst) {
    // EXPLICAÇÃO: Os logs dos segmentos são divididos de forma contígua.
    // Os primeiros 'segment_count0' logs pertencem ao nó 0, o restante ao nó 1.
    bool seg_on_node0 = (segment_id < segment_count0);
    int32_t local_seg_id = seg_on_node0 ? segment_id : segment_id - segment_count0;

    auto& log_indices = seg_on_node0 ? log_segment_idx_0 : log_segment_idx_1;
    auto& log_pointers = seg_on_node0 ? log_ptr_0 : log_ptr_1;

    assert(log_indices[local_seg_id] < MAX_LOG_ENTRIES && "Log do segmento está cheio, rebalanceamento necessário.");

    // Determina em qual nó o VÉRTICE está para buscar seu offset de log anterior.
    bool vertex_on_node0 = (src % 2 == 0);
    int32_t local_src_id = src / 2;
    auto& vertices = vertex_on_node0 ? vertices_0 : vertices_1;

    // Pega o ponteiro para a próxima entrada de log livre.
    struct LogEntry* log_ins_ptr = (struct LogEntry*)(log_pointers[local_seg_id] + log_indices[local_seg_id]);
    log_ins_ptr->u = src;
    log_ins_ptr->v = dst;
    log_ins_ptr->prev_offset = vertices[local_src_id].offset; // Pega o offset anterior do vértice correto.

    flush_clwb_nolog(log_ins_ptr, sizeof(struct LogEntry));

    // Atualiza o offset do vértice para apontar para a nova entrada de log.
    vertices[local_src_id].offset = log_indices[local_seg_id];
    log_indices[local_seg_id] += 1;
}


/// Realiza a inserção da aresta (lógica de baixo nível).
/// Adaptado para NUMA com particionamento Round Robin.
inline void do_insertion(int32_t src, int32_t dst, int32_t src_segment_global) {
    // Determina o nó e o índice local do vértice de origem.
    bool is_node0 = (src % 2 == 0);
    int32_t local_src = src / 2;

    // Seleciona as estruturas de dados corretas com base no nó.
    auto& vertices = is_node0 ? vertices_0 : vertices_1;
    auto& edges = is_node0 ? edges_0 : edges_1;
    auto& segment_actuals = is_node0 ? segment_edges_actual_0 : segment_edges_actual_1;
    auto& num_node_edges = is_node0 ? n_edges_node0 : n_edges_node1;

    // O ID do segmento é global, mas para acessar os arrays de metadados (actuals, logs),
    // precisamos saber a qual nó o *segmento* pertence.
    int32_t global_seg_idx = src_segment_global - segment_count;
    bool seg_on_node0 = (global_seg_idx < segment_count0);
    int32_t local_seg_idx = seg_on_node0 ? global_seg_idx : global_seg_idx - segment_count0;
    auto& log_indices = seg_on_node0 ? log_segment_idx_0 : log_segment_idx_1;

    // Calcula a localização potencial da nova aresta.
    int64_t loc = vertices[local_src].index + vertices[local_src].degree;

    // Se houver espaço no array de arestas principal, insere diretamente.
    if (have_space_onseg(src, loc)) {
        edges[loc].v = dst;
        flush_clwb_nolog(&edges[loc], sizeof(DestID_));
    } else {
        // Se não houver espaço, usa o log.
        // Verifica se o log do segmento está cheio.
        if (log_indices[local_seg_idx] >= MAX_LOG_ENTRIES) {
            // Log cheio, precisa rebalancear o segmento antes de prosseguir.
            int32_t left_index = (src / segment_size) * segment_size;
            int32_t right_index = min(left_index + segment_size, num_vertices);

            // O rebalanceamento precisa dos totais do segmento.
            int64_t current_actuals = segment_actuals[local_seg_idx];
            int64_t current_totals = segment_edges_total[global_seg_idx + segment_count]; // 'total' é global

            rebalance_weighted(left_index, right_index, current_actuals, current_totals, src, dst);
        }

        // Tenta a inserção novamente após o possível rebalanceamento.
        loc = vertices[local_src].index + vertices[local_src].degree;
        if (have_space_onseg(src, loc)) {
            edges[loc].v = dst;
            flush_clwb_nolog(&edges[loc], sizeof(DestID_));
        } else {
            // Se ainda não houver espaço, insere no log.
            insert_into_log(global_seg_idx, src, dst);
        }
    }

    // Atualiza os metadados do nó correto.
    vertices[local_src].degree += 1;
    segment_actuals[local_seg_idx] += 1;

    // Atualiza o contador total de arestas de forma atômica.
    int64_t old_val, new_val;
    do {
        old_val = num_node_edges;
        new_val = old_val + 1;
    } while (!compare_and_swap(num_node_edges, old_val, new_val));
}


/// Função principal de inserção de arestas dinâmicas.
/// Adaptada para NUMA.
void insert(int32_t src, int32_t dst) {
    int32_t current_segment_global = get_segment_id(src);
    int32_t global_seg_idx = current_segment_global - segment_count;

    // Travando o segmento da folha para garantir exclusão mútua.
    unique_lock<mutex> ul(leaf_segments[global_seg_idx].lock);
    leaf_segments[global_seg_idx].wait_for_rebalance(ul);

    // Insere a aresta.
    do_insertion(src, dst, current_segment_global);

    // Agora, verifica se a inserção tornou o segmento muito denso,
    // necessitando de um rebalanceamento.
    int32_t left_index = src, right_index = src;
    int32_t window = current_segment_global;

    // EXPLICAÇÃO: A densidade deve ser lida dos arrays de metadados corretos.
    double density;
    bool seg_on_node0 = (global_seg_idx < segment_count0);
    if (seg_on_node0) {
        density = (double)(segment_edges_actual_0[global_seg_idx]) / (double)segment_edges_total[current_segment_global];
    } else {
        density = (double)(segment_edges_actual_1[global_seg_idx - segment_count0]) / (double)segment_edges_total[current_segment_global];
    }

    int32_t height = 0;
    if (density >= up_0) {
        // O segmento está muito cheio. Precisamos encontrar uma "janela" de rebalanceamento.
        leaf_segments[global_seg_idx].on_rebalance += 1;
        ul.unlock();

        double up_height = up_0 - (height * delta_up);
        pair<int64_t, int64_t> seg_meta;

        while (window > 0) {
            // Sobe um nível na árvore PMA conceitual.
            window /= 2;
            height += 1;

            int32_t window_size = segment_size * (1 << height);
            left_index = (src / window_size) * window_size;
            right_index = min(left_index + window_size, num_vertices);

            int32_t left_segment = get_segment_id(left_index) - segment_count;
            int32_t right_segment = get_segment_id(right_index) - segment_count;

            // Trava todos os segmentos na janela encontrada.
            // A função lock_in_batch também precisará ser NUMA-aware para somar os valores corretamente.
            seg_meta = lock_in_batch(left_segment, right_segment, global_seg_idx);
            up_height = up_0 - (height * delta_up);
            density = (double)seg_meta.first / (double)seg_meta.second;

            if (density < up_height) {
                // Encontramos uma janela que está dentro do limiar de densidade.
                break;
            } else {
                // Esta janela ainda está muito cheia, destrava e tenta uma maior.
                unlock_in_batch(left_segment, right_segment, global_seg_idx);
            }
        }

        if (!height) {
            cout << "Isso não deveria acontecer! Abortando!" << endl;
            exit(-1);
        }

        if (window) {
            // Encontrou uma janela válida para rebalancear.
            int32_t window_size = segment_size * (1 << height);
            left_index = (src / window_size) * window_size;
            right_index = min(left_index + window_size, num_vertices);

            // Recalcula a densidade do segmento original para uma checagem final.
            if (seg_on_node0) {
                density = (double)(segment_edges_actual_0[global_seg_idx]) / (double)segment_edges_total[current_segment_global];
            } else {
                density = (double)(segment_edges_actual_1[global_seg_idx - segment_count0]) / (double)segment_edges_total[current_segment_global];
            }

            if (density >= up_0) {
                rebalance_weighted(left_index, right_index, seg_meta.first, seg_meta.second, src, dst);
            }
        } else {
            // Nenhuma janela encontrada, significa que o grafo inteiro está cheio.
            // Precisamos redimensionar todo o array de arestas.
            int32_t left_segment = 0;
            int32_t right_segment = segment_count;

            if (density < up_0) {
                 // Outra thread pode ter resolvido o problema.
                 unlock_in_batch(left_segment, right_segment);
                 return;
            }
            seg_meta = lock_in_batch(left_segment, right_segment, global_seg_idx);
            resize_V1(); // Esta função também precisa ser totalmente NUMA-aware.
        }

        // Destrava os segmentos que foram travados para o rebalanceamento.
        int32_t left_segment = get_segment_id(left_index) - segment_count;
        int32_t right_segment = get_segment_id(right_index) - segment_count;
        unlock_in_batch(left_segment, right_segment);
    } else {
        // O segmento não está muito cheio, apenas destrava e termina.
        ul.unlock();
    }
}

/// Limpa as entradas de log para um determinado segmento.
/// Adaptado para NUMA.
inline void release_log(int32_t global_segment_id) {
    // Determina a qual nó o log do segmento pertence.
    bool seg_on_node0 = (global_segment_id < segment_count0);
    int32_t local_seg_id = seg_on_node0 ? global_segment_id : global_segment_id - segment_count0;

    // Seleciona os ponteiros e índices de log corretos.
    auto& log_indices = seg_on_node0 ? log_segment_idx_0 : log_segment_idx_1;
    auto& log_pointers = seg_on_node0 ? log_ptr_0 : log_ptr_1;

    if (log_indices[local_seg_id] == 0) return; // Nada a fazer se o log estiver vazio.

    int num_entries_to_clear = log_indices[local_seg_id];
    memset(log_pointers[local_seg_id], 0, sizeof(struct LogEntry) * num_entries_to_clear);

    flush_clwb_nolog(log_pointers[local_seg_id], sizeof(struct LogEntry) * num_entries_to_clear);
    log_indices[local_seg_id] = 0; // Reseta o índice do log.
}


  #else
  // Verifica se há espaço disponível para inserção no segmento
  bool have_space_onseg(int32_t src, int64_t loc, int32_t current_segment) {
    if (src < n_vertices_node0) {
      // Para o nó 0, src é índice global
      if ((src == (n_vertices_node0 - 1) && elem_capacity0 > loc)
          || (src < (n_vertices_node0 - 1) && vertices_0[src + 1].index > loc))
        return true;
    } else {
      // Para o nó 1, converte src para índice local
      int32_t local_src = src - n_vertices_node0;
      if ((local_src == (n_vertices_node1 - 1) && elem_capacity1 > loc)
          || (local_src < (n_vertices_node1 - 1) && vertices_1[local_src + 1].index > loc))
        return true;
    }
    return false;
  }

  // Função principal de inserção de aresta (insere a aresta e, se necessário, chama rebalanceamento)
  void insert(int32_t src, int32_t dst) {
    int32_t current_segment = get_segment_id(src);
    int32_t left_segment = current_segment - segment_count, right_segment = current_segment - segment_count + 1;
    double density;
    // taking the lock for the current segment
    unique_lock <mutex> ul(leaf_segments[current_segment - segment_count].lock);
    leaf_segments[current_segment - segment_count].wait_for_rebalance(ul);
    // insert the current edge
    do_insertion(src, dst, current_segment);
    int32_t left_index = src, right_index = src;
    int32_t window = current_segment;
    if((current_segment - segment_count) < segment_count0){
      density = (double) (segment_edges_actual_0[current_segment]) / (double) segment_edges_total[current_segment];
    }
    else{
      density = (double) (segment_edges_actual_1[current_segment - segment_count0]) / (double) segment_edges_total[current_segment];
    }
    
    int32_t height = 0;
    if (density >= up_0) {
      // unlock the current segment
      leaf_segments[current_segment - segment_count].on_rebalance += 1;
      ul.unlock();

      double up_height = up_0 - (height * delta_up);
      pair <int64_t, int64_t> seg_meta;

      while (window > 0) {
        // Go one level up in our conceptual PMA tree
        window /= 2;
        height += 1;

        int32_t window_size = segment_size * (1 << height);
        left_index = (src / window_size) * window_size;
        right_index = min(left_index + window_size, num_vertices);

        left_segment = get_segment_id(left_index) - segment_count;
        right_segment = get_segment_id(right_index) - segment_count;

        seg_meta = lock_in_batch(left_segment, right_segment, current_segment - segment_count);
        up_height = up_0 - (height * delta_up);
        density = (double) seg_meta.first / (double) seg_meta.second;

        if (density < up_height) {
          break;
        } else {
          unlock_in_batch(left_segment, right_segment, current_segment - segment_count);
        }
      }

      if (!height) {
        cout << "This should not happen! Aborting!" << endl;
        exit(-1);
      }

      if (window) {
        // Found a window within threshold
        int32_t window_size = segment_size * (1 << height);
        left_index = (src / window_size) * window_size;
        right_index = min(left_index + window_size, num_vertices);

        left_segment = get_segment_id(left_index) - segment_count;
        right_segment = get_segment_id(right_index) - segment_count;

        if((current_segment - segment_count) < segment_count0){
          density = (double) (segment_edges_actual_0[current_segment]) / (double) segment_edges_total[current_segment];
        }
        else{
          density = (double) (segment_edges_actual_1[current_segment - segment_count0]) / (double) segment_edges_total[current_segment];
        }
        if (density >= up_0) rebalance_weighted(left_index, right_index, seg_meta.first, seg_meta.second, src, dst);
      }
      else {
        // Rebalance not possible without increasing the underlying array size, need to resize the size of "edges_" array
        left_segment = 0;
        right_segment = segment_count;

        if((current_segment - segment_count) < segment_count0){
          density = (double) (segment_edges_actual_0[current_segment]) / (double) segment_edges_total[current_segment];
        }
        else{
          density = (double) (segment_edges_actual_1[current_segment - segment_count0]) / (double) segment_edges_total[current_segment];
        }
        if (density < up_0) return;
        seg_meta = lock_in_batch(left_segment, right_segment, current_segment - segment_count);
        resize_V1();
      }
      
      unlock_in_batch(left_segment, right_segment);
    }
    else {
      ul.unlock();
    }
    printf("Depois do resize\n");
    //print_vertices();
  }

  // Função de inserção interna: insere a aresta em PMEM ou no log se necessário
  inline void do_insertion(int32_t src, int32_t dst, int32_t src_segment) {
    int64_t loc;
    int64_t old_val, new_val;
    int32_t left_index, right_index;

    // Verifica se o "current_segment" está na faixa do Nó 0 ou Nó 1:
    if ((src_segment - segment_count) < segment_count0) {
        // ------------------ Nó 0 --------------------
        // local_seg é o índice 0-based para o array segment_edges_actual_0
        int32_t local_seg = src_segment - segment_count;

        // Aqui supomos que 'vertices_0[src]' já foi separado e src é índice global p/ Nó 0
        loc = vertices_0[src].index + vertices_0[src].degree;

        // Tenta inserir diretamente em edges_0[loc]
        if (have_space_onseg(src, loc, src_segment)) {
            edges_0[loc].v = dst;
            flush_clwb_nolog(&edges_0[loc], sizeof(DestID_));
        } else {
            // Se não há espaço, verifica se o log está cheio
            if (log_segment_idx_[local_seg] >= MAX_LOG_ENTRIES) {
                left_index  = (src / segment_size) * segment_size;
                right_index = min(left_index + segment_size, num_vertices);

                // Rebalance local (janela do tamanho de 1 segmento)
                rebalance_weighted(left_index, right_index,
                                   segment_edges_actual_0[local_seg],
                                   segment_edges_total[local_seg], // mesmo local_seg no segment_edges_total
                                   src, dst);
            }
            // Tenta de novo (pois o rebalance_weighted pode ter aberto espaço)
            loc = vertices_0[src].index + vertices_0[src].degree;
            if (have_space_onseg(src, loc, src_segment)) {
                edges_0[loc].v = dst;
                flush_clwb_nolog(&edges_0[loc], sizeof(DestID_));
            } else {
                // Se ainda não há espaço, insere no log
                insert_into_log(local_seg, src, dst);
            }
        }

        // Atualiza metadados (Nó 0)
        vertices_0[src].degree += 1;
        segment_edges_actual_0[local_seg] += 1;

        do {
            old_val = n_edges_node0;
            new_val = old_val + 1;
        } while (!compare_and_swap(n_edges_node0, old_val, new_val));
    } 
    else {
        // ------------------ Nó 1 --------------------
        // local_seg é o índice 0-based no array segment_edges_actual_1
        int32_t local_seg = (src_segment - segment_count) - segment_count0;

        // Ajusta 'src' para ser índice local do Nó 1
        int32_t local_src = src - n_vertices_node0;  // depende de como você dividiu globalmente

        loc = vertices_1[local_src].index + vertices_1[local_src].degree;

        if (have_space_onseg(src, loc, src_segment)) {
            edges_1[loc].v = dst;
            flush_clwb_nolog(&edges_1[loc], sizeof(DestID_));
        } else {
            if (log_segment_idx_[local_seg + segment_count0] >= MAX_LOG_ENTRIES) {
                left_index  = (src / segment_size) * segment_size;
                right_index = min(left_index + segment_size, num_vertices);

                // Rebalance local
                rebalance_weighted(left_index, right_index,
                                   segment_edges_actual_1[local_seg],
                                   // no segment_edges_total, a posição é local_seg + segment_count0
                                   segment_edges_total[local_seg + segment_count0],
                                   src, dst);
            }
            loc = vertices_1[local_src].index + vertices_1[local_src].degree;
            if (have_space_onseg(src, loc, src_segment)) {
                edges_1[loc].v = dst;
                flush_clwb_nolog(&edges_1[loc], sizeof(DestID_));
            } else {
                insert_into_log(local_seg, src, dst);
            }
        }

        // Atualiza metadados (Nó 1)
        vertices_1[local_src].degree += 1;
        segment_edges_actual_1[local_seg] += 1;

        do {
            old_val = n_edges_node1;
            new_val = old_val + 1;
        } while (!compare_and_swap(n_edges_node1, old_val, new_val));
    }
}

inline void insert_into_log(int32_t segment_id, int32_t src, int32_t dst) {
  assert(log_segment_idx_[segment_id] < MAX_LOG_ENTRIES &&
         "logs are full, need to perform a rebalance first");
  if(src < n_vertices_node0){
    assert(vertices_0[src].offset < MAX_LOG_ENTRIES &&
      "vertex offset is beyond the log range, should not happen this for sure!");

  }
  else{
    assert(vertices_1[src - n_vertices_node0].offset < MAX_LOG_ENTRIES &&
      "vertex offset is beyond the log range, should not happen this for sure!");
  }
  
  assert((src >= (segment_id * segment_size) && src < ((segment_id * segment_size) + segment_size)) &&
         "src vertex is not for this segment-id");

  // insert into log
  struct LogEntry *log_ins_ptr = (struct LogEntry *) (log_ptr_[segment_id] + log_segment_idx_[segment_id]);
  log_ins_ptr->u = src;
  log_ins_ptr->v = dst;
  if(src < n_vertices_node0){
    log_ins_ptr->prev_offset = vertices_0[src].offset;
  }
  else{
    log_ins_ptr->prev_offset = vertices_1[src - n_vertices_node0].offset;
  }
  

  flush_clwb_nolog(log_ins_ptr, sizeof(struct LogEntry));
  if(src < n_vertices_node0){
    vertices_0[src].offset = log_segment_idx_[segment_id];
  }
  else{
    vertices_1[src - n_vertices_node0].offset = log_segment_idx_[segment_id];
  }
  log_segment_idx_[segment_id] += 1;
}

  inline void print_per_vertex_log(int32_t vid) {
    return;
  }
  inline void print_log(int32_t segment_id) {
    return;
  }
  inline void release_log(int32_t segment_id) {
    if (log_segment_idx_[segment_id] == 0) return;
    memset(log_ptr_[segment_id], 0, sizeof(struct LogEntry) * log_segment_idx_[segment_id]);

    flush_clwb_nolog(&log_ptr_[segment_id], sizeof(struct LogEntry) * log_segment_idx_[segment_id]);
    log_segment_idx_[segment_id] = 0;
  }
  
  #endif
  #else
  bool have_space_onseg(int32_t src, int64_t loc) {
    if ((src == (num_vertices - 1) && elem_capacity > loc)
        || (src < (num_vertices - 1) && vertices_[src + 1].index > loc))
      return true;
    return false;
  }

  /// Insert dynamic-graph edges.
  void insert(int32_t src, int32_t dst) {
    int32_t current_segment = get_segment_id(src);
    int32_t left_segment = current_segment - segment_count, right_segment = current_segment - segment_count + 1;

    // taking the lock for the current segment
    unique_lock <mutex> ul(leaf_segments[current_segment - segment_count].lock);
    leaf_segments[current_segment - segment_count].wait_for_rebalance(ul);

    // insert the current edge
    do_insertion(src, dst, current_segment);

    int32_t left_index = src, right_index = src;
    int32_t window = current_segment;

    double density = (double) (segment_edges_actual[current_segment]) / (double) segment_edges_total[current_segment];
    int32_t height = 0;
    if (density >= up_0) {
      // unlock the current segment
      leaf_segments[current_segment - segment_count].on_rebalance += 1;
      ul.unlock();

      double up_height = up_0 - (height * delta_up);
      pair <int64_t, int64_t> seg_meta;

      while (window > 0) {
        // Go one level up in our conceptual PMA tree
        window /= 2;
        height += 1;

        int32_t window_size = segment_size * (1 << height);
        left_index = (src / window_size) * window_size;
        right_index = min(left_index + window_size, num_vertices);

        left_segment = get_segment_id(left_index) - segment_count;
        right_segment = get_segment_id(right_index) - segment_count;

        seg_meta = lock_in_batch(left_segment, right_segment, current_segment - segment_count);
        up_height = up_0 - (height * delta_up);
        density = (double) seg_meta.first / (double) seg_meta.second;

        if (density < up_height) {
          break;
        } else {
          unlock_in_batch(left_segment, right_segment, current_segment - segment_count);
        }
      }

      if (!height) {
        cout << "This should not happen! Aborting!" << endl;
        exit(-1);
      }

      if (window) {
        // Found a window within threshold
        int32_t window_size = segment_size * (1 << height);
        left_index = (src / window_size) * window_size;
        right_index = min(left_index + window_size, num_vertices);

        left_segment = get_segment_id(left_index) - segment_count;
        right_segment = get_segment_id(right_index) - segment_count;

        density = (double) (segment_edges_actual[current_segment]) / (double) segment_edges_total[current_segment];
        if (density >= up_0) rebalance_weighted(left_index, right_index, seg_meta.first, seg_meta.second, src, dst);
      }
      else {
        // Rebalance not possible without increasing the underlying array size, need to resize the size of "edges_" array
        left_segment = 0;
        right_segment = segment_count;

        density = (double) (segment_edges_actual[current_segment]) / (double) segment_edges_total[current_segment];
        if (density < up_0) return;
        seg_meta = lock_in_batch(left_segment, right_segment, current_segment - segment_count);
        resize_V1();
      }

      unlock_in_batch(left_segment, right_segment);
    }
    else {
      ul.unlock();
    }
  }

  inline void do_insertion(int32_t src, int32_t dst, int32_t src_segment) {
    int64_t loc = vertices_[src].index + vertices_[src].degree;

    // if there is empty space, make the insertion
    if (have_space_onseg(src, loc)) {
      edges_[loc].v = dst;
      flush_clwb_nolog(&edges_[loc], sizeof(DestID_));
    }
    else {  // else add it to the log
      // check if the log is full
      if (log_segment_idx_[src_segment - segment_count] >= MAX_LOG_ENTRIES) {
        int32_t left_index = (src / segment_size) * segment_size;
        int32_t right_index = min(left_index + segment_size, num_vertices);

        rebalance_weighted(left_index, right_index,
                           segment_edges_actual[src_segment], segment_edges_total[src_segment],
                           src, dst);
      }

      loc = vertices_[src].index + vertices_[src].degree;
      if (have_space_onseg(src, loc)) {
        edges_[loc].v = dst;
        flush_clwb_nolog(&edges_[loc], sizeof(DestID_));
      } else {
        insert_into_log(src_segment - segment_count, src, dst);
      }
    }

    // updating metadata
    vertices_[src].degree += 1;
    segment_edges_actual[src_segment] += 1;
    int64_t old_val, new_val;
    do {
      old_val = num_edges_;
      new_val = old_val + 1;
    } while (!compare_and_swap(num_edges_, old_val, new_val));
  }

  /// Insert an edge into edge-log.
  inline void insert_into_log(int32_t segment_id, int32_t src, int32_t dst) {
    assert(log_segment_idx_[segment_id] < MAX_LOG_ENTRIES &&
           "logs are full, need to perform a rebalance first");
    assert(vertices_[src].offset < MAX_LOG_ENTRIES &&
           "vertex offset is beyond the log range, should not happen this for sure!");
    assert((src >= (segment_id * segment_size) && src < ((segment_id * segment_size) + segment_size)) &&
           "src vertex is not for this segment-id");

    // insert into log
    struct LogEntry *log_ins_ptr = (struct LogEntry *) (log_ptr_[segment_id] + log_segment_idx_[segment_id]);
    log_ins_ptr->u = src;
    log_ins_ptr->v = dst;
    log_ins_ptr->prev_offset = vertices_[src].offset;

    flush_clwb_nolog(log_ins_ptr, sizeof(struct LogEntry));

    vertices_[src].offset = log_segment_idx_[segment_id];
    log_segment_idx_[segment_id] += 1;
  }

  /// Print all the edges of vertex @vid stored in the per-seg-logs.
  inline void print_per_vertex_log(int32_t vid) {
    int32_t segment_id = get_segment_id(vid) - segment_count;
    int32_t current_offset = vertices_[vid].offset;
    if (current_offset == -1) {
      cout << "vertex " << vid << ": do not have any log" << endl;
      return;
    }
    cout << "vertex " << vid << ":";
    while (current_offset != -1) {
      cout << " " << log_ptr_[segment_id][current_offset].v;
      current_offset = log_ptr_[segment_id][current_offset].prev_offset;
    }
    cout << endl;
  }

  /// Print all the edges of segment @segment_id stored in the per-seg-logs.
  inline void print_log(int32_t segment_id) {
    if (log_segment_idx_[segment_id] == 0) {
      cout << "segment " << segment_id << ": do not have any log" << endl;
      return;
    }
    cout << "vertex range from: " << segment_id * segment_size << " to: " << (segment_id * segment_size) + segment_size
         << endl;
    cout << "segment " << segment_id << ":";
    for (int i = 0; i < log_segment_idx_[segment_id]; i += 1) {
      cout << " <" << log_ptr_[segment_id][i].u << " " << log_ptr_[segment_id][i].v << " "
           << log_ptr_[segment_id][i].prev_offset << ">";
    }
    cout << endl;
  }

  /// Releasing per-seg-logs by mem-setting to 0.
  inline void release_log(int32_t segment_id) {
    if (log_segment_idx_[segment_id] == 0) return;
    memset(log_ptr_[segment_id], 0, sizeof(struct LogEntry) * log_segment_idx_[segment_id]);

    flush_clwb_nolog(&log_ptr_[segment_id], sizeof(struct LogEntry) * log_segment_idx_[segment_id]);
    log_segment_idx_[segment_id] = 0;
  }
  #endif
  /// Calculate PMA-parameters.
  void compute_capacity() {
    segment_size = ceil_log2(num_vertices); // Ideal segment size
    segment_count = ceil_div(num_vertices, segment_size); // Ideal number of segments

    // The number of segments has to be a power of 2, though.
    segment_count = hyperfloor(segment_count);
    // Update the segment size accordingly
    segment_size = ceil_div(num_vertices, segment_count);

    num_vertices = segment_count * segment_size;
    avg_degree = ceil_div(num_edges_, num_vertices);

    elem_capacity = (num_edges_ + num_vertices) * max_sparseness;
  }

  /// Spread the edges based on the degree of source vertices.
  /// Here, end_vertex is excluded, and start_vertex is expected to be 0
  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
  void spread_weighted(int32_t start_vertex, int32_t end_vertex) {
      assert(start_vertex == 0 && "start_vertex is expected to be 0 here.");
      // NOTA: Para a versão NUMA, assumimos uma redistribuição completa.
      // Os parâmetros start/end_vertex são mantidos para compatibilidade de API.
      
      // As operações para o nó 0 e nó 1 são completamente independentes
      // e podem ser executadas em paralelo para um ganho de desempenho.
      #pragma omp parallel sections
      {
          #pragma omp section
          {
              // Espalha os vértices dentro da memória do Nó 0
              spread_weighted_for_node(n_vertices_node0, vertices_0, edges_0, elem_capacity0, n_edges_node0);
          }

          #pragma omp section
          {
              // Espalha os vértices dentro da memória do Nó 1
              spread_weighted_for_node(n_vertices_node1, vertices_1, edges_1, elem_capacity1, n_edges_node1);
          }
      }
      
      // Após a redistribuição em ambos os nós, recalcula os totais globais se necessário.
      recount_segment_total();
  }
inline pair<int64_t, int64_t> lock_in_batch(int32_t left_segment, int32_t right_segment, int32_t src_segment_to_exclude) {
    pair<int64_t, int64_t> ret = make_pair(0l, 0l);
    int32_t old_val, new_val;

    // Itera sobre o intervalo de segmentos globais.
    for (int32_t seg_id = left_segment; seg_id < right_segment; seg_id += 1) {
        if (seg_id != src_segment_to_exclude) {
            // Incrementa o contador para sinalizar que um rebalanceamento está pendente.
            // Isso previne que outras threads tentem usar o segmento enquanto ele é modificado.
            do {
                old_val = leaf_segments[seg_id].on_rebalance;
                new_val = old_val + 1;
            } while (!compare_and_swap(leaf_segments[seg_id].on_rebalance, old_val, new_val));
        }

        leaf_segments[seg_id].lock.lock(); // Adquire o lock do segmento.

        // Acumula os metadados (arestas atuais vs. espaço total).
        // O array 'segment_edges_total' é global, então o índice é direto.
        ret.second += segment_edges_total[seg_id + segment_count];

        // O array 'segment_edges_actual' é particionado.
        // Verificamos a qual nó o segmento pertence para ler do array correto.
        bool seg_on_node0 = (seg_id < segment_count0);
        if (seg_on_node0) {
            ret.first += segment_edges_actual_0[seg_id];
        } else {
            ret.first += segment_edges_actual_1[seg_id - segment_count0];
        }
    }
    return ret;
}



  #else
  void spread_weighted(int32_t start_vertex, int32_t end_vertex) {
    assert(start_vertex == 0 && "start-vertex is expected to be 0 here.");
    int64_t gaps = elem_capacity - (n_edges_node1 + n_edges_node0);
    int64_t *new_positions = calculate_positions(start_vertex, end_vertex, gaps, (n_edges_node1 + n_edges_node0));

    int64_t read_index, write_index, curr_degree;
    for (int32_t curr_vertex = end_vertex - 1; curr_vertex > start_vertex; curr_vertex -= 1) {
      if(curr_vertex < n_vertices_node0){
        curr_degree = vertices_0[curr_vertex].degree;
        read_index = vertices_0[curr_vertex].index + curr_degree - 1;
        write_index = new_positions[curr_vertex] + curr_degree - 1;

        if (write_index < read_index) {
          cout << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index
              << ", degree: " << curr_degree << endl;
        }
        assert(write_index >= read_index && "index anomaly occurred while spreading elements");

        for (int i = 0; i < curr_degree; i++) {
          edges_0[write_index] = edges_0[read_index];
          write_index--;
          read_index--;
        }

        vertices_0[curr_vertex].index = new_positions[curr_vertex];

        }
        else{
          curr_degree = vertices_1[curr_vertex - n_vertices_node0].degree;
          read_index = vertices_1[curr_vertex - n_vertices_node0].index + curr_degree - 1;
          write_index = new_positions[curr_vertex] + curr_degree - 1;
  
          if (write_index < read_index) {
            cout << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index
                << ", degree: " << curr_degree << endl;
          }
          assert(write_index >= read_index && "index anomaly occurred while spreading elements");
  
          for (int i = 0; i < curr_degree; i++) {
            edges_1[write_index] = edges_1[read_index];
            write_index--;
            read_index--;
          }
  
          vertices_1[curr_vertex - n_vertices_node0].index = new_positions[curr_vertex];
  
          }
    }

    // note: we do not need to flush the data in PMEM from here, it is managed from the caller function
    free(new_positions);
    new_positions = nullptr;
    recount_segment_total();
  }


  /// Take the locks of the segments [@left_segment, @right_segment); exclude @src_segment to increment the CV counter
  inline pair <int64_t, int64_t> lock_in_batch(int32_t left_segment, int32_t right_segment, int32_t src_segment) {
    pair <int64_t, int64_t> ret = make_pair(0l, 0l);
    int32_t old_val, new_val;
    for (int32_t seg_id = left_segment; seg_id < right_segment; seg_id += 1) {
      if (seg_id != src_segment) {
        do {
          old_val = leaf_segments[seg_id].on_rebalance;
          new_val = old_val + 1;
        } while (!compare_and_swap(leaf_segments[seg_id].on_rebalance, old_val, new_val));
      }

      leaf_segments[seg_id].lock.lock();
      if(seg_id < segment_count0){
        ret.first += segment_edges_actual_0[seg_id];
        ret.second += segment_edges_total[seg_id];
      }
      else{
        ret.first += segment_edges_actual_1[seg_id - segment_count0];
        ret.second += segment_edges_total[seg_id];
      }
      
    }
    return ret;
  }
  #endif
  #else
  void spread_weighted(int32_t start_vertex, int32_t end_vertex) {
    assert(start_vertex == 0 && "start-vertex is expected to be 0 here.");
    int64_t gaps = elem_capacity - num_edges_;
    int64_t *new_positions = calculate_positions(start_vertex, end_vertex, gaps, num_edges_);

    int64_t read_index, write_index, curr_degree;
    for (int32_t curr_vertex = end_vertex - 1; curr_vertex > start_vertex; curr_vertex -= 1) {
      curr_degree = vertices_[curr_vertex].degree;
      read_index = vertices_[curr_vertex].index + curr_degree - 1;
      write_index = new_positions[curr_vertex] + curr_degree - 1;

      if (write_index < read_index) {
        cout << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index
             << ", degree: " << curr_degree << endl;
      }
      assert(write_index >= read_index && "index anomaly occurred while spreading elements");

      for (int i = 0; i < curr_degree; i++) {
        edges_[write_index] = edges_[read_index];
        write_index--;
        read_index--;
      }

      vertices_[curr_vertex].index = new_positions[curr_vertex];
    }

    // note: we do not need to flush the data in PMEM from here, it is managed from the caller function
    free(new_positions);
    new_positions = nullptr;
    recount_segment_total();
  }

  /// Take the locks of the segments [@left_segment, @right_segment); exclude @src_segment to increment the CV counter
  inline pair <int64_t, int64_t> lock_in_batch(int32_t left_segment, int32_t right_segment, int32_t src_segment) {
    pair <int64_t, int64_t> ret = make_pair(0l, 0l);
    int32_t old_val, new_val;
    for (int32_t seg_id = left_segment; seg_id < right_segment; seg_id += 1) {
      if (seg_id != src_segment) {
        do {
          old_val = leaf_segments[seg_id].on_rebalance;
          new_val = old_val + 1;
        } while (!compare_and_swap(leaf_segments[seg_id].on_rebalance, old_val, new_val));
      }

      leaf_segments[seg_id].lock.lock();
      ret.first += segment_edges_actual[seg_id + segment_count];
      ret.second += segment_edges_total[seg_id + segment_count];
    }
    return ret;
  }
  #endif

  /// Unlock the locks of the segments [@left_segment, @right_segment)
  inline void unlock_in_batch(int32_t left_segment, int32_t right_segment) {
    int32_t rebal_depend;
    int32_t old_val, new_val;
    for (int32_t seg_id = right_segment - 1; seg_id >= left_segment; seg_id -= 1) {
      do {
        old_val = leaf_segments[seg_id].on_rebalance;
        new_val = old_val - 1;
      } while (!compare_and_swap(leaf_segments[seg_id].on_rebalance, old_val, new_val));

      rebal_depend = leaf_segments[seg_id].on_rebalance;
      leaf_segments[seg_id].lock.unlock();
      if (!rebal_depend) leaf_segments[seg_id].cv.notify_all();
    }
  }

  /// Unlock the locks of the segments [@left_segment, @right_segment); exclude @src_segment to decrement the CV counter
  inline void unlock_in_batch(int32_t left_segment, int32_t right_segment, int32_t src_segment) {
    int32_t rebal_depend;
    int32_t old_val, new_val;
    for (int32_t seg_id = right_segment - 1; seg_id >= left_segment; seg_id -= 1) {
      if (seg_id != src_segment) {
        do {
          old_val = leaf_segments[seg_id].on_rebalance;
          new_val = old_val - 1;
        } while (!compare_and_swap(leaf_segments[seg_id].on_rebalance, old_val, new_val));
      }
      rebal_depend = leaf_segments[seg_id].on_rebalance;
      leaf_segments[seg_id].lock.unlock();
      if (!rebal_depend) leaf_segments[seg_id].cv.notify_all();
    }
  }

  /// Rebalance all the segments.
  void rebalance_all() {
    int32_t left_segment = get_segment_id(0) - segment_count;
    int32_t right_segment = get_segment_id(num_vertices) - segment_count;

    pair <int64_t, int64_t> seg_meta = lock_in_batch(left_segment, right_segment, -1);
    rebalance_weighted(0, num_vertices, seg_meta.first, seg_meta.second);
    unlock_in_batch(left_segment, right_segment);
  }

  /// Calculate the starting index of each vertex for [@start_vertex, @end_vertex) based on the degree of vertices
  /// @total_degree is the sum of degrees of [@start_vertex, @end_vertex)
  /// @gaps are the sum of empty spaces after the per-vertex neighbors of [@start_vertex, @end_vertex)
  #ifdef NUMA_PMEM
  #ifdef HASH_MODE
int64_t *calculate_positions(int32_t start_vertex, int32_t end_vertex, int64_t gaps, int64_t total_degree) {
    int32_t size = end_vertex - start_vertex;
    if (size <= 0) {
        return nullptr;
    }

    int64_t *new_index = (int64_t *) calloc(size, sizeof(int64_t));
    if (!new_index) {
        perror("calloc failed for new_index");
        exit(1);
    }

    // --- Passo 1: Separar vértices e calcular graus por nó ---
    vector<int32_t> node0_vids, node1_vids;
    int64_t degree_sum_node0 = 0;
    int64_t degree_sum_node1 = 0;

    for (int32_t i = 0; i < size; ++i) {
        int32_t vid = start_vertex + i;
        bool is_node0 = (vid % 2 == 0);
        int32_t local_id = vid / 2;

        if (is_node0) {
            node0_vids.push_back(vid);
            degree_sum_node0 += vertices_0[local_id].degree;
        } else {
            node1_vids.push_back(vid);
            degree_sum_node1 += vertices_1[local_id].degree;
        }
    }

    // --- Passo 2: Distribuir os gaps proporcionalmente ---
    int64_t total_units = (degree_sum_node0 + node0_vids.size()) + (degree_sum_node1 + node1_vids.size());
    if (total_units <= 0) total_units = 1; // Evitar divisão por zero

    double gap_ratio_node0 = (double)(degree_sum_node0 + node0_vids.size()) / total_units;
    double gap_ratio_node1 = 1.0 - gap_ratio_node0;

    int64_t gaps_node0 = (int64_t)(gaps * gap_ratio_node0);
    int64_t gaps_node1 = gaps - gaps_node0;

    // --- Passo 3: Calcular novas posições para o Nó 0 ---
    if (!node0_vids.empty()) {
        int64_t total_units_node0 = degree_sum_node0 + node0_vids.size();
        double step0 = (total_units_node0 > 0) ? (double)gaps_node0 / total_units_node0 : 0.0;

        int32_t first_vid_node0 = node0_vids[0];
        double current_pos0 = vertices_0[first_vid_node0 / 2].index;

        for (int32_t vid : node0_vids) {
            int32_t local_id = vid / 2;
            // A posição no array new_index é relativa ao início da janela.
            new_index[vid - start_vertex] = round(current_pos0);
            current_pos0 += (vertices_0[local_id].degree + (step0 * (vertices_0[local_id].degree + 1)));
        }
    }

    // --- Passo 4: Calcular novas posições para o Nó 1 ---
    if (!node1_vids.empty()) {
        int64_t total_units_node1 = degree_sum_node1 + node1_vids.size();
        double step1 = (total_units_node1 > 0) ? (double)gaps_node1 / total_units_node1 : 0.0;

        int32_t first_vid_node1 = node1_vids[0];
        double current_pos1 = vertices_1[first_vid_node1 / 2].index;

        for (int32_t vid : node1_vids) {
            int32_t local_id = vid / 2;
            new_index[vid - start_vertex] = round(current_pos1);
            current_pos1 += (vertices_1[local_id].degree + (step1 * (vertices_1[local_id].degree + 1)));
        }
    }

    return new_index;
}
/// Rebalanceia uma janela de vértices para ajustar a densidade das arestas.
/// Adaptado para NUMA.
void rebalance_weighted(int32_t start_vertex, int32_t end_vertex,
                        int64_t used_space, int64_t total_space,
                        int32_t src = -1, int32_t dst = -1) {

    int64_t gaps = total_space - used_space;
    int64_t* new_index = calculate_positions(start_vertex, end_vertex, gaps, used_space);

    rebalance_data_V1(start_vertex, end_vertex, new_index);

    free(new_index);
    new_index = nullptr;
    recount_segment_total(start_vertex, end_vertex);
}

/// Carrega uma porção de um array de arestas para o undo-log para garantir consistência em caso de falha.
/// Adaptado para NUMA.
inline void load_into_ulog(int tid, DestID_* edge_array, int64_t edge_array_capacity,
                           int64_t load_idx_st, int32_t load_sz,
                           int64_t flush_idx_st, int64_t flush_idx_nd) {
    // Determina a qual nó a thread pertence para usar o ulog correto.
    // Assumimos uma divisão simples de threads entre os nós.
    int num_threads_node0 = num_threads / 2;
    bool thread_on_node0 = (tid < num_threads_node0);

    auto& ulog_pointers = thread_on_node0 ? ulog_ptr_0 : ulog_ptr_1;
    auto& oplog_pointers = thread_on_node0 ? oplog_ptr_0 : oplog_ptr_1;
    int local_tid = thread_on_node0 ? tid : tid - num_threads_node0;

    // Persiste as últimas modificações feitas no array de arestas antes de carregar uma nova janela no log.
    if (flush_idx_st != flush_idx_nd) {
        flush_clwb_nolog(&edge_array[flush_idx_st], sizeof(DestID_) * (flush_idx_nd - flush_idx_st));
    }

    // Copia a próxima janela do array de arestas para o ulog da thread.
    memcpy(ulog_pointers[local_tid], edge_array + load_idx_st, sizeof(DestID_) * load_sz);
    flush_clwb_nolog(ulog_pointers[local_tid], sizeof(DestID_) * MAX_ULOG_ENTRIES);

    // Registra a posição do array de arestas que foi salva no log.
    oplog_pointers[local_tid] = load_idx_st;
    flush_clwb_nolog(&oplog_pointers[local_tid], sizeof(int64_t));
}


/// Obtém o 'index' (posição inicial da lista de arestas) de um vértice.
/// Lida com a partição por hash.
inline int64_t get_old_index(int32_t vid) const {
    if (vid >= num_vertices) return elem_capacity; // Limite para o último vértice
    bool is_node0 = (vid % 2 == 0);
    int32_t local_id = vid / 2;
    if (is_node0) {
        return (local_id < n_vertices_node0) ? vertices_0[local_id].index : -1;
    } else {
        return (local_id < n_vertices_node1) ? vertices_1[local_id].index : -1;
    }
}

/// Obtém o grau de um vértice.
/// Lida com a partição por hash.
inline int32_t get_degree(int32_t vid) const {
    if (vid >= num_vertices) return 0;
    bool is_node0 = (vid % 2 == 0);
    int32_t local_id = vid / 2;
    if (is_node0) {
        return (local_id < n_vertices_node0) ? vertices_0[local_id].degree : 0;
    } else {
        return (local_id < n_vertices_node1) ? vertices_1[local_id].degree : 0;
    }
}


inline void rebalance_node_data(int node_id, int32_t start_vertex, int32_t end_vertex, int64_t* new_index) {
    int tid = omp_get_thread_num();
    
    // Estruturas locais do nó
    auto& V = (node_id == 0) ? vertices_0 : vertices_1;
    auto& E = (node_id == 0) ? edges_0 : edges_1;
    auto& log_pointers = (node_id == 0) ? log_ptr_0 : log_ptr_1;
    int64_t node_edge_offset = (node_id == 0) ? 0 : elem_capacity0;
    int64_t node_capacity = (node_id == 0) ? elem_capacity0 : elem_capacity1;
    
    // Calcula vértices que pertencem a este nó
    int32_t node_start = start_vertex + (node_id - (start_vertex % 2) + 2) % 2;
    int32_t node_end = end_vertex;
    
    // Ajusta para começar no primeiro vértice do nó correto
    if (node_start < start_vertex) node_start += 2;
    
    // Processa em chunks para melhor cache locality
    int32_t chunk_size = 128; // Otimizado para L1 cache
    
    for (int32_t chunk_start = node_start; chunk_start < node_end; chunk_start += chunk_size * 2) {
        int32_t chunk_end = std::min(chunk_start + chunk_size * 2, node_end);
        
        // Processa chunk em ordem reversa (mantém consistência com versão original)
        for (int32_t jj = chunk_end - 2; jj >= chunk_start; jj -= 2) {
            // Verifica se o vértice pertence ao nó correto
            if ((jj % 2) != node_id) continue;
            
            int32_t local_jj = jj / 2;
            int32_t total_degree = V[local_jj].degree;
            
            if (total_degree == 0) {
                V[local_jj].index = new_index[jj - start_vertex];
                continue;
            }
            
            // Calcula posições antigas e novas
            int64_t old_global_pos = V[local_jj].index;
            int64_t old_local_pos = old_global_pos - node_edge_offset;
            int64_t new_global_pos = new_index[jj - start_vertex];
            int64_t new_local_pos = new_global_pos - node_edge_offset;
            
            // Determina boundary do próximo vértice
            int64_t next_vertex_boundary;
            int32_t next_global_vid = jj + 2; // Próximo vértice no mesmo nó
            
            if (next_global_vid >= num_vertices) {
                next_vertex_boundary = node_capacity;
            } else {
                int32_t next_local_id = next_global_vid / 2;
                next_vertex_boundary = V[next_local_id].index - node_edge_offset;
            }
            
            int32_t on_segment_count = next_vertex_boundary - old_local_pos;
            
            // Otimização: se não há movimento, só atualiza metadados
            if (old_local_pos == new_local_pos && V[local_jj].offset == -1) {
                continue;
            }
            
            // Move dados do segmento se necessário
            if (on_segment_count > 0 && old_local_pos != new_local_pos) {
                // Usa memmove para handles overlap corretamente
                memmove(&E[new_local_pos], &E[old_local_pos], 
                       on_segment_count * sizeof(DestID_));
            }
            
            // Processa dados do log se existirem
            if (V[local_jj].offset != -1) {
                int32_t global_seg_id = get_segment_id(jj) - segment_count;
                int32_t local_seg_id = (node_id == 0) ? global_seg_id : global_seg_id - segment_count0;
                
                // Valida se o segmento pertence ao nó correto
                bool seg_belongs_to_node = (node_id == 0) ? (global_seg_id < segment_count0) : 
                                          (global_seg_id >= segment_count0);
                
                if (seg_belongs_to_node) {
                    int32_t curr_off = V[local_jj].offset;
                    int64_t write_pos = new_local_pos + on_segment_count;
                    
                    // Move arestas do log diretamente para posição final
                    while(curr_off != -1) {
                        E[write_pos].v = log_pointers[local_seg_id][curr_off].v;
                        curr_off = log_pointers[local_seg_id][curr_off].prev_offset;
                        write_pos++;
                    }
                }
            }
            
            // Atualiza metadados do vértice
            V[local_jj].index = new_global_pos;
            V[local_jj].offset = -1;
        }
    }
}


/// Move os dados das arestas para as novas posições, usando undo-logs para consistência.
// Função principal otimizada com divisão por nó NUMA
inline void rebalance_data_V1(int32_t start_vertex, int32_t end_vertex, int64_t* new_index, bool from_resize = false) {
    
    // Versão paralela que processa cada nó NUMA separadamente
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            // Thread 0: processa apenas vértices pares (nó 0)
            rebalance_node_data(0, start_vertex, end_vertex, new_index);
        }
        
        #pragma omp section
        {
            // Thread 1: processa apenas vértices ímpares (nó 1)  
            rebalance_node_data(1, start_vertex, end_vertex, new_index);
        }
    }
    
    // Limpa logs de todos os segmentos afetados (uma vez só no final)
    int32_t st_seg = get_segment_id(start_vertex) - segment_count;
    int32_t nd_seg = get_segment_id(end_vertex - 1) - segment_count + 1;
    
    // Limpa logs do nó 0
    #pragma omp parallel for
    for (int32_t seg_id = st_seg; seg_id < nd_seg; ++seg_id) {
        if (seg_id < segment_count0) {
            release_log(seg_id);
        }
    }
    
    // Limpa logs do nó 1  
    #pragma omp parallel for
    for (int32_t seg_id = st_seg; seg_id < nd_seg; ++seg_id) {
        if (seg_id >= segment_count0) {
            release_log(seg_id - segment_count0);
        }
    }
}

#else
  int64_t *calculate_positions(int32_t start_vertex, int32_t end_vertex, int64_t gaps, int64_t total_degree) {
      int32_t size = end_vertex - start_vertex;
      int64_t *new_index = (int64_t *) calloc(size, sizeof(int64_t));
      // Ajusta o total considerando um incremento por vértice
      total_degree += size;
      
      double step = ((double) gaps) / total_degree;  // passo por aresta
      double precision_pos = 100000000.0;
      assert(((int64_t)(step * precision_pos)) > 0l && "fixed-precision is going to cause problem!");
      step = ((double) ((int64_t)(step * precision_pos))) / precision_pos;
      
      // Caso 1: intervalo totalmente em nó 0
      if(end_vertex <= n_vertices_node0) {
        int64_t index_boundary = (end_vertex == n_vertices_node0)
        ? elem_capacity0 : vertices_0[end_vertex].index;
          double index_d = vertices_0[start_vertex].index;
          for (int i = start_vertex; i < end_vertex; i++) {
              new_index[i - start_vertex] = index_d;
              assert(new_index[i - start_vertex] + vertices_0[i].degree <= index_boundary && "index calculation is wrong!");
              index_d = new_index[i - start_vertex] + (vertices_0[i].degree + (step * (vertices_0[i].degree + 1)));
          }
      }
      // Caso 2: intervalo totalmente em nó 1
      else if(start_vertex >= n_vertices_node0) {
          int32_t local_start = start_vertex - n_vertices_node0;
          int32_t local_end = end_vertex - n_vertices_node0;
          // Se estamos no fim da partição, usamos o limite global (elem_capacity0 + elem_capacity1)
          int64_t index_boundary = (end_vertex == n_vertices_node0 + n_vertices_node1)
          ? (elem_capacity0 + elem_capacity1) : vertices_1[local_end].index;
          double index_d = vertices_1[local_start].index;
          for (int i = start_vertex; i < end_vertex; i++) {
              int32_t local_i = i - n_vertices_node0;
              new_index[i - start_vertex] = index_d;
              assert(new_index[i - start_vertex] + vertices_1[local_i].degree <= index_boundary && "index calculation is wrong!");
              index_d = new_index[i - start_vertex] + (vertices_1[local_i].degree + (step * (vertices_1[local_i].degree + 1)));
          }
      }
      // Caso 3: intervalo abrange ambas as partições
      else {
          // Parte do nó 0
          int32_t size0 = n_vertices_node0 - start_vertex;
          int64_t *new_index0 = (int64_t *) calloc(size0, sizeof(int64_t));
          int64_t index_boundary0 = elem_capacity0; // limite para nó 0
          double index_d0 = vertices_0[start_vertex].index;
          for (int i = start_vertex; i < n_vertices_node0; i++) {
              new_index0[i - start_vertex] = index_d0;
              //assert(new_index0[i - start_vertex] + vertices_0[i].degree <= index_boundary0 && "index calculation is wrong!");
              index_d0 = new_index0[i - start_vertex] + (vertices_0[i].degree + (step * (vertices_0[i].degree + 1)));
          }
          // Parte do nó 1
          int32_t size1 = end_vertex - n_vertices_node0;
          int64_t *new_index1 = (int64_t *) calloc(size1, sizeof(int64_t));
          int32_t local_end = end_vertex - n_vertices_node0;
          int64_t index_boundary1 = (end_vertex == n_vertices_node0 + n_vertices_node1)
          ? (elem_capacity0 + elem_capacity1) : vertices_1[ local_end ].index;
          double index_d1 = vertices_1[0].index; // assume início da partição 1
          for (int i = 0; i < size1; i++) {
              new_index1[i] = index_d1;
              //assert(new_index1[i] + vertices_1[i].degree <= index_boundary1 && "index calculation is wrong!");
              index_d1 = new_index1[i] + (vertices_1[i].degree + (step * (vertices_1[i].degree + 1)));
          }
          // Mescla os resultados: primeiro os índices para nó 0 e depois para nó 1
          for (int i = 0; i < size0; i++) {
              new_index[i] = new_index0[i];
          }
          for (int i = 0; i < size1; i++) {
              new_index[size0 + i] = new_index1[i];
          }
          free(new_index0);
          free(new_index1);
      }
      return new_index;
  }

  void rebalance_weighted(int32_t start_vertex,
    int32_t end_vertex,
    int64_t used_space, int64_t total_space, int32_t src = -1, int32_t dst = -1) {
    int64_t from;
    int64_t to;
    int64_t capacity;
    int64_t gaps;
    int32_t size;
    int64_t *new_index;
    int64_t index_boundary;
    if(start_vertex < n_vertices_node0){
      from = vertices_0[start_vertex].index;
    }
    else{
      from = vertices_1[start_vertex - n_vertices_node0].index;
    }
    if(end_vertex <= n_vertices_node0){
      to = (end_vertex >= n_vertices_node0) ? elem_capacity0 : vertices_0[end_vertex].index;
      assert(to > from && "Invalid range found while doing weighted rebalance");
      capacity = to - from;

      assert(total_space == capacity && "Segment capacity is not matched with segment_edges_total");
      gaps = total_space - used_space;

      // calculate the future positions of the vertices_[i].index
      size = end_vertex - start_vertex;
      new_index = calculate_positions(start_vertex, end_vertex, gaps, used_space);
      index_boundary = (end_vertex >= n_vertices_node0) ? elem_capacity0 : vertices_0[end_vertex].index;
      assert(new_index[size - 1] + vertices_0[end_vertex - 1].degree <= index_boundary &&
      "Rebalance (weighted) index calculation is wrong!");
    }
    else{
      to = ((end_vertex - n_vertices_node0) >= n_vertices_node1) ? elem_capacity1 : vertices_1[end_vertex - n_vertices_node0].index;
      assert(to > from && "Invalid range found while doing weighted rebalance");
      capacity = to - from;

      assert(total_space == capacity && "Segment capacity is not matched with segment_edges_total");
      gaps = total_space - used_space;

      // calculate the future positions of the vertices_[i].index
      size = end_vertex - start_vertex;
      new_index = calculate_positions(start_vertex, end_vertex, gaps, used_space);
      index_boundary = ((end_vertex - n_vertices_node0) >= n_vertices_node1) ? elem_capacity1 : vertices_1[end_vertex - n_vertices_node0].index;
      assert(new_index[size - 1] + vertices_1[end_vertex - 1 - n_vertices_node0].degree <= index_boundary &&
      "Rebalance (weighted) index calculation is wrong!");
    }
    rebalance_data_V1(start_vertex, end_vertex, new_index);

    free(new_index);
    new_index = nullptr;
    recount_segment_total(start_vertex, end_vertex);
  }

  
  inline void
  load_into_ulog(int tid, int64_t load_idx_st, int32_t load_sz, int64_t flush_idx_st, int64_t flush_idx_nd) {
    // flush the last entries
    if(flush_idx_st < n_vertices_node0){
      if (flush_idx_st != flush_idx_nd) {
        flush_clwb_nolog(&edges_0[flush_idx_st], sizeof(DestID_) * (flush_idx_nd - flush_idx_st + 1));
      }
  //    memcpy(ulog_ptr_[tid], edges_+load_idx_st, sizeof(DestID_) * MAX_ULOG_ENTRIES);
      memcpy(ulog_ptr_[tid], edges_0 + load_idx_st, sizeof(DestID_) * load_sz);
      flush_clwb_nolog(&ulog_ptr_[tid], sizeof(DestID_) * MAX_ULOG_ENTRIES);
      oplog_ptr_[tid] = load_idx_st;
      flush_clwb_nolog(&oplog_ptr_[tid], sizeof(int64_t));
    }
    else{
      if (flush_idx_st != flush_idx_nd) {
        flush_clwb_nolog(&edges_1[flush_idx_st - segment_count0], sizeof(DestID_) * (flush_idx_nd - flush_idx_st + 1));
      }
  //    memcpy(ulog_ptr_[tid], edges_+load_idx_st, sizeof(DestID_) * MAX_ULOG_ENTRIES);
      memcpy(ulog_ptr_[tid], edges_1 + load_idx_st, sizeof(DestID_) * load_sz);
      flush_clwb_nolog(&ulog_ptr_[tid], sizeof(DestID_) * MAX_ULOG_ENTRIES);
      oplog_ptr_[tid] = load_idx_st;
      flush_clwb_nolog(&oplog_ptr_[tid], sizeof(int64_t));

    }
}

inline void
  rebalance_data_V1(int32_t start_vertex, int32_t end_vertex, int64_t *new_index, bool from_resize = false) {
    int tid = omp_get_thread_num();
    int32_t ii, jj;
    int32_t curr_vertex = start_vertex;
    int64_t read_index, last_read_index, write_index;
    int32_t curr_off, curr_seg, onseg_num_edges;
    int64_t next_vertex_boundary;
    int64_t ulog_st, ulog_nd;
    if(end_vertex < n_vertices_node0){
      ulog_st = vertices_0[end_vertex].index;
      ulog_nd = vertices_0[end_vertex].index;
    }
    else{
      ulog_st = vertices_1[end_vertex - n_vertices_node0].index;
      ulog_nd = vertices_1[end_vertex - n_vertices_node0].index;
    }
    
    int32_t load_sz = 0;

    // loop over vertex
    while (curr_vertex < end_vertex) {
      for (ii = curr_vertex; ii < end_vertex; ii++) {
        if(curr_vertex < n_vertices_node0 - 1){
          if (new_index[ii - start_vertex] + vertices_0[ii].degree <= vertices_0[ii + 1].index) break;
        }
        else{
          if(curr_vertex == n_vertices_node0 - 1){
            if (new_index[ii - start_vertex] + vertices_0[ii].degree <= vertices_1[0].index) break;
          }
          else{
            if (new_index[ii - start_vertex] + vertices_1[ii - n_vertices_node0].degree <= vertices_1[ii - n_vertices_node0 + 1].index) break;
          }
        }
        // the following condition will give us the first vertex which have space to its right side
        
      }

      if (ii == end_vertex) {
        ii -= 1;
      }
      if(ii + 1 < n_vertices_node0){
        next_vertex_boundary = (ii >= num_vertices - 1) ? (elem_capacity - 1) : vertices_0[ii + 1].index - 1;
      }
      else{
        next_vertex_boundary = (ii >= num_vertices - 1) ? (elem_capacity - 1) : vertices_1[ii + 1 - n_vertices_node0].index - 1;
      }
      
      // we will shift edges for source-vertex [curr_vertex to ii]
      for (jj = ii; jj >= curr_vertex; jj -= 1) {
        if(jj < n_vertices_node0){
          if (vertices_0[jj].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_0[jj].index + 1;

          // on-segment: do left-shift
          if (new_index[jj - start_vertex] < vertices_0[jj].index) {
            read_index = vertices_0[jj].index;
            last_read_index = read_index + ((vertices_0[jj].offset != -1) ? onseg_num_edges : vertices_0[jj].degree);

            write_index = new_index[jj - start_vertex];

            while (read_index < last_read_index) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                          ? MAX_ULOG_ENTRIES : (elem_capacity - write_index);
                load_into_ulog(tid, write_index, load_sz, ulog_st, ulog_nd);
                ulog_st = write_index;
                ulog_nd = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                          ? (write_index + MAX_ULOG_ENTRIES) : elem_capacity;
              }
              edges_0[write_index] = edges_0[read_index];
              write_index++;
              read_index++;
            }
          }
          // on-segment: do right-shift
          else if (new_index[jj - start_vertex] > vertices_0[jj].index) {
            read_index = vertices_0[jj].index + ((vertices_0[jj].offset != -1)
                                                ? onseg_num_edges : vertices_0[jj].degree) - 1;
            last_read_index = vertices_0[jj].index;

            write_index = new_index[jj - start_vertex] + ((vertices_0[jj].offset != -1)
                                                          ? onseg_num_edges : vertices_0[jj].degree) - 1;

            while (read_index >= last_read_index) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
                load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
                ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
                ulog_nd = write_index;
              }
              edges_0[write_index] = edges_0[read_index];
              write_index--;
              read_index--;
            }
          }

          // if vertex-jj have edges in the log, move it to on-segment
          if (vertices_0[jj].offset != -1) {
            curr_off = vertices_0[jj].offset;
            curr_seg = get_segment_id(jj) - segment_count;

            write_index = new_index[jj - start_vertex] + vertices_0[jj].degree - 1;
            while (curr_off != -1) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
                load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
                ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
                ulog_nd = write_index;
              }
              edges_0[write_index].v = log_ptr_[curr_seg][curr_off].v;

              curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
              write_index--;
            }
          }

          // update the index to the new position
          next_vertex_boundary = vertices_0[jj].index - 1;
          vertices_0[jj].index = new_index[jj - start_vertex];
          vertices_0[jj].offset = -1;
        }
        else{
          if (vertices_1[jj - n_vertices_node0].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_1[jj - n_vertices_node0].index + 1;

          // on-segment: do left-shift
          if (new_index[jj - start_vertex] < vertices_1[jj - n_vertices_node0].index) {
            read_index = vertices_1[jj - n_vertices_node0].index;
            last_read_index = read_index + ((vertices_1[jj - n_vertices_node0].offset != -1) ? onseg_num_edges : vertices_1[jj - n_vertices_node0].degree);

            write_index = new_index[jj - start_vertex];

            while (read_index < last_read_index) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                          ? MAX_ULOG_ENTRIES : (elem_capacity - write_index);
                load_into_ulog(tid, write_index, load_sz, ulog_st, ulog_nd);
                ulog_st = write_index;
                ulog_nd = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                          ? (write_index + MAX_ULOG_ENTRIES) : elem_capacity;
              }
              edges_1[write_index] = edges_1[read_index];
              write_index++;
              read_index++;
            }
          }
          // on-segment: do right-shift
          else if (new_index[jj - start_vertex] > vertices_1[jj - n_vertices_node0].index) {
            read_index = vertices_1[jj - n_vertices_node0].index + ((vertices_1[jj - n_vertices_node0].offset != -1)
                                                ? onseg_num_edges : vertices_1[jj - n_vertices_node0].degree) - 1;
            last_read_index = vertices_1[jj - n_vertices_node0].index;

            write_index = new_index[jj - start_vertex] + ((vertices_1[jj - n_vertices_node0].offset != -1)
                                                          ? onseg_num_edges : vertices_1[jj - n_vertices_node0].degree) - 1;

            while (read_index >= last_read_index) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
                load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
                ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
                ulog_nd = write_index;
              }
              edges_1[write_index] = edges_1[read_index];
              write_index--;
              read_index--;
            }
          }

          // if vertex-jj have edges in the log, move it to on-segment
          if (vertices_1[jj - n_vertices_node0].offset != -1) {
            curr_off = vertices_1[jj - n_vertices_node0].offset;
            curr_seg = get_segment_id(jj) - segment_count;

            write_index = new_index[jj - start_vertex] + vertices_1[jj - n_vertices_node0].degree - 1;
            while (curr_off != -1) {
              if (write_index < ulog_st || write_index >= ulog_nd) {
                load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
                load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
                ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
                ulog_nd = write_index;
              }
              edges_1[write_index].v = log_ptr_[curr_seg][curr_off].v;

              curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
              write_index--;
            }
          }

          // update the index to the new position
          next_vertex_boundary = vertices_1[jj - n_vertices_node0].index - 1;
          vertices_1[jj - n_vertices_node0].index = new_index[jj - start_vertex];
          vertices_1[jj - n_vertices_node0].offset = -1;
        }
      }
      curr_vertex = ii + 1;
    }
    // update log for the rebalance segments
    int32_t st_seg = get_segment_id(start_vertex), nd_seg = get_segment_id(end_vertex);
    for (int32_t i = st_seg; i < nd_seg; i += 1) {
      release_log(i - segment_count);
    }
  }
#endif
#else
  int64_t *calculate_positions(int32_t start_vertex, int32_t end_vertex, int64_t gaps, int64_t total_degree) {
    int32_t size = end_vertex - start_vertex;
    int64_t *new_index = (int64_t *) calloc(size, sizeof(int64_t));
    total_degree += size;

    int64_t index_boundary = (end_vertex >= num_vertices) ? elem_capacity : vertices_[end_vertex].index;
    double index_d = vertices_[start_vertex].index;
    double step = ((double) gaps) / total_degree;  //per-edge step
    double precision_pos = 100000000.0;
    assert(((int64_t)(step * precision_pos)) > 0l && "fixed-precision is going to cause problem!");
    step = ((double) ((int64_t)(step * precision_pos))) / precision_pos;
    for (int i = start_vertex; i < end_vertex; i++) {
      new_index[i - start_vertex] = index_d;
      assert(new_index[i - start_vertex] + vertices_[i].degree <= index_boundary && "index calculation is wrong!");

      index_d = new_index[i - start_vertex];
      index_d += (vertices_[i].degree + (step * (vertices_[i].degree + 1)));
//      index_d += (vertices_[i].degree + (step * vertices_[i].degree));
    }

    return new_index;
  }

  void rebalance_weighted(int32_t start_vertex,
                          int32_t end_vertex,
                          int64_t used_space, int64_t total_space, int32_t src = -1, int32_t dst = -1) {
    int64_t from = vertices_[start_vertex].index;
    int64_t to = (end_vertex >= num_vertices) ? elem_capacity : vertices_[end_vertex].index;
    assert(to > from && "Invalid range found while doing weighted rebalance");
    int64_t capacity = to - from;

    assert(total_space == capacity && "Segment capacity is not matched with segment_edges_total");
    int64_t gaps = total_space - used_space;

    // calculate the future positions of the vertices_[i].index
    int32_t size = end_vertex - start_vertex;
    int64_t *new_index = calculate_positions(start_vertex, end_vertex, gaps, used_space);
    int64_t index_boundary = (end_vertex >= num_vertices) ? elem_capacity : vertices_[end_vertex].index;
    assert(new_index[size - 1] + vertices_[end_vertex - 1].degree <= index_boundary &&
           "Rebalance (weighted) index calculation is wrong!");

    rebalance_data_V1(start_vertex, end_vertex, new_index);

    free(new_index);
    new_index = nullptr;
    recount_segment_total(start_vertex, end_vertex);
  }

  /// Manage crash consistent log
  /// First, flush changes in the @edge_array for the interval [@flush_idx_st, @flush_idx_nd)
  /// Then, load data to ulog from the @edge_array for the range of [@load_idx_st, @load_idx_st + @MAX_ULOG_SIZE)
  /// Finally, make an entry in the oplog to track the @edge_array range saved in the ulog
  inline void
  load_into_ulog(int tid, int64_t load_idx_st, int32_t load_sz, int64_t flush_idx_st, int64_t flush_idx_nd) {
    // flush the last entries
    if (flush_idx_st != flush_idx_nd) {
      flush_clwb_nolog(&edges_[flush_idx_st], sizeof(DestID_) * (flush_idx_nd - flush_idx_st + 1));
    }
//    memcpy(ulog_ptr_[tid], edges_+load_idx_st, sizeof(DestID_) * MAX_ULOG_ENTRIES);
    memcpy(ulog_ptr_[tid], edges_ + load_idx_st, sizeof(DestID_) * load_sz);
    flush_clwb_nolog(&ulog_ptr_[tid], sizeof(DestID_) * MAX_ULOG_ENTRIES);
    oplog_ptr_[tid] = load_idx_st;
    flush_clwb_nolog(&oplog_ptr_[tid], sizeof(int64_t));
  }

  inline void
  rebalance_data_V1(int32_t start_vertex, int32_t end_vertex, int64_t *new_index, bool from_resize = false) {
    int tid = omp_get_thread_num();
    int32_t ii, jj;
    int32_t curr_vertex = start_vertex;
    int64_t read_index, last_read_index, write_index;
    int32_t curr_off, curr_seg, onseg_num_edges;
    int64_t next_vertex_boundary;
    int64_t ulog_st = vertices_[end_vertex].index, ulog_nd = vertices_[end_vertex].index;
    int32_t load_sz = 0;

    // loop over vertex
    while (curr_vertex < end_vertex) {
      for (ii = curr_vertex; ii < end_vertex; ii++) {
        // the following condition will give us the first vertex which have space to its right side
        if (new_index[ii - start_vertex] + vertices_[ii].degree <= vertices_[ii + 1].index) break;
      }

      if (ii == end_vertex) {
        ii -= 1;
      }

      next_vertex_boundary = (ii >= num_vertices - 1) ? (elem_capacity - 1) : vertices_[ii + 1].index - 1;
      // we will shift edges for source-vertex [curr_vertex to ii]
      for (jj = ii; jj >= curr_vertex; jj -= 1) {
        if (vertices_[jj].offset != -1) onseg_num_edges = next_vertex_boundary - vertices_[jj].index + 1;

        // on-segment: do left-shift
        if (new_index[jj - start_vertex] < vertices_[jj].index) {
          read_index = vertices_[jj].index;
          last_read_index = read_index + ((vertices_[jj].offset != -1) ? onseg_num_edges : vertices_[jj].degree);

          write_index = new_index[jj - start_vertex];

          while (read_index < last_read_index) {
            if (write_index < ulog_st || write_index >= ulog_nd) {
              load_sz = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                        ? MAX_ULOG_ENTRIES : (elem_capacity - write_index);
              load_into_ulog(tid, write_index, load_sz, ulog_st, ulog_nd);
              ulog_st = write_index;
              ulog_nd = ((write_index + MAX_ULOG_ENTRIES) <= elem_capacity)
                        ? (write_index + MAX_ULOG_ENTRIES) : elem_capacity;
            }
            edges_[write_index] = edges_[read_index];
            write_index++;
            read_index++;
          }
        }
        // on-segment: do right-shift
        else if (new_index[jj - start_vertex] > vertices_[jj].index) {
          read_index = vertices_[jj].index + ((vertices_[jj].offset != -1)
                                              ? onseg_num_edges : vertices_[jj].degree) - 1;
          last_read_index = vertices_[jj].index;

          write_index = new_index[jj - start_vertex] + ((vertices_[jj].offset != -1)
                                                        ? onseg_num_edges : vertices_[jj].degree) - 1;

          while (read_index >= last_read_index) {
            if (write_index < ulog_st || write_index >= ulog_nd) {
              load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
              load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
              ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
              ulog_nd = write_index;
            }
            edges_[write_index] = edges_[read_index];
            write_index--;
            read_index--;
          }
        }

        // if vertex-jj have edges in the log, move it to on-segment
        if (vertices_[jj].offset != -1) {
          curr_off = vertices_[jj].offset;
          curr_seg = get_segment_id(jj) - segment_count;

          write_index = new_index[jj - start_vertex] + vertices_[jj].degree - 1;
          while (curr_off != -1) {
            if (write_index < ulog_st || write_index >= ulog_nd) {
              load_sz = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? MAX_ULOG_ENTRIES : write_index;
              load_into_ulog(tid, write_index - MAX_ULOG_ENTRIES, load_sz, ulog_st, ulog_nd);
              ulog_st = ((write_index - MAX_ULOG_ENTRIES) >= 0) ? (write_index - MAX_ULOG_ENTRIES) : 0;
              ulog_nd = write_index;
            }
            edges_[write_index].v = log_ptr_[curr_seg][curr_off].v;

            curr_off = log_ptr_[curr_seg][curr_off].prev_offset;
            write_index--;
          }
        }

        // update the index to the new position
        next_vertex_boundary = vertices_[jj].index - 1;
        vertices_[jj].index = new_index[jj - start_vertex];
        vertices_[jj].offset = -1;
      }
      curr_vertex = ii + 1;
    }
    // update log for the rebalance segments
    int32_t st_seg = get_segment_id(start_vertex), nd_seg = get_segment_id(end_vertex);
    for (int32_t i = st_seg; i < nd_seg; i += 1) {
      release_log(i - segment_count);
    }
  }
  #endif

private:
  bool directed_;
  int64_t max_size = (1ULL << 56) - 1ULL;

  /* PMA constants */
  // Height-based (as opposed to depth-based) tree thresholds
  // Upper density thresholds
  static constexpr double up_h = 0.75;    // root
  static constexpr double up_0 = 1.00;    // leaves
  // Lower density thresholds
  static constexpr double low_h = 0.50;   // root
  static constexpr double low_0 = 0.25;   // leaves

  int8_t max_sparseness = 1.0 / low_0;
  int8_t largest_empty_segment = 1.0 * max_sparseness;

  int64_t num_edges_ = 0;                 // Number of edges
  int32_t num_vertices = 0;               // Number of vertices
  int32_t max_valid_vertex_id = 0;        // Max valid vertex-id
  int64_t avg_degree = 0;                 // averge degree of the graph
  #ifdef NUMA_PMEM
  int32_t n_vertices_node0;
  int32_t n_vertices_node1;
  int64_t n_edges_node0;     // Partição igual para as arestas
  int64_t n_edges_node1;
  int64_t segment_count0;
  int64_t segment_count1;
  #endif

  /* General PMA fields */
  int64_t elem_capacity;                  // size of the edges_ array
  int32_t segment_size;                   // size of a pma leaf segment
  int64_t segment_count;                  // number of pma leaf segments
  int32_t tree_height;                    // height of the pma tree
  double delta_up;                        // Delta for upper density threshold
  double delta_low;                       // Delta for lower density threshold
  #ifdef NUMA_PMEM
  int64_t elem_capacity0;
  int64_t elem_capacity1;
  DestID_ *edges_0;
  DestID_ *edges_1;
  struct vertex_element *vertices_0;
  struct vertex_element *vertices_1;
  #else
  DestID_ *edges_;                        // Underlying storage for edgelist
  struct vertex_element *vertices_;       // underlying storage for vertex list
  #endif

  struct LogEntry *log_base_ptr_;         // mapping of pmem::log_segment_oid_
  struct LogEntry **log_ptr_;             // pointer of logs per segment
  int32_t *log_segment_idx_;              // current insert-index of logs per segment
  int64_t *segment_edges_actual;          // actual number of edges stored in the region of a binary-tree node
  int64_t *segment_edges_total;           // total number of edges assigned in the region of a binary-tree node
  struct PMALeafSegment *leaf_segments;   // pma-leaf segments to control concurrency
#ifdef NUMA_PMEM
  struct LogEntry *log_base_ptr_0;         // mapping of pmem::log_segment_oid_
  struct LogEntry **log_ptr_0; 
    struct LogEntry *log_base_ptr_1;         // mapping of pmem::log_segment_oid_
  struct LogEntry **log_ptr_1; 
#endif

  int num_threads;                        // max (available) number of concurrent write threads
  int node_counter;                          // Number of nodes available in a NUMA system
  DestID_ *ulog_base_ptr_;                // base pointer of the undo-log; used to track the group allocation of undo-logs
  DestID_ **ulog_ptr_;                    // array of undo-log pointers (array size is number of write threads)
  int64_t *oplog_ptr_;                    // keeps the start index in the edge array that is backed up in undo-log
  #ifdef NUMA_PMEM
  DestID_ *ulog_base_ptr_0;
  DestID_ *ulog_base_ptr_1;
  int32_t *log_segment_idx_0;
  int32_t *log_segment_idx_1;
  int64_t *segment_edges_actual_0;   
  int64_t *segment_edges_actual_1;          // actual number of edges stored in the region of a binary-tree node
  DestID_ **ulog_ptr_0;                    // array of undo-log pointers (array size is number of write threads)
  int64_t *oplog_ptr_0;                    // keeps the start index in the edge array that is backed up in undo-log
  DestID_ **ulog_ptr_1;                    // array of undo-log pointers (array size is number of write threads)
  int64_t *oplog_ptr_1;                    // keeps the start index in the edge array that is backed up in undo-log

  int64_t* calculate_positions_for_node(
      int32_t num_local_vertices,
      const struct vertex_element* local_vertices,
      int64_t local_elem_capacity,
      int64_t local_num_edges)
  {
      if (num_local_vertices == 0) return nullptr;

      int64_t* new_positions = (int64_t*) calloc(num_local_vertices, sizeof(int64_t));
      int64_t gaps = local_elem_capacity - local_num_edges;
      if (gaps < 0) {
          // Isso pode acontecer se a inserção inicial encheu a capacidade. Não há espaço para espalhar.
          gaps = 0;
      }

      // O "step" é a quantidade de espaço extra a ser adicionada por unidade de grau.
      double step = (local_num_edges > 0) ? ((double) gaps) / local_num_edges : 0.0;
      
      double current_pos_float = 0.0; // Usamos double para acumular a fração do step

      for (int i = 0; i < num_local_vertices; ++i) {
          new_positions[i] = round(current_pos_float);
          
          // A nova posição é a atual + o tamanho do vértice + o gap proporcional ao seu tamanho.
          current_pos_float += (local_vertices[i].degree + (step * local_vertices[i].degree));
      }

      return new_positions;
  }

  void spread_weighted_for_node(
      int32_t num_local_vertices,
      struct vertex_element* local_vertices,
      DestID_* local_edges,
      int64_t local_elem_capacity,
      int64_t local_num_edges)
  {
      if (num_local_vertices == 0) return;

      int64_t* new_positions = calculate_positions_for_node(num_local_vertices, local_vertices, local_elem_capacity, local_num_edges);
      if (!new_positions) return;

      // Iteramos de trás para frente para garantir que não sobrescrevemos dados
      // que ainda precisamos ler.
      for (int32_t local_v_idx = num_local_vertices - 1; local_v_idx >= 0; --local_v_idx) {
          int64_t curr_degree = local_vertices[local_v_idx].degree;
          if (curr_degree == 0) continue; // Vértice vazio, nada a mover.

          int64_t read_index  = local_vertices[local_v_idx].index;
          int64_t write_index = new_positions[local_v_idx];

          if (write_index < read_index) {
              // Isso não deveria acontecer em um "spread", mas é bom ter a verificação.
              // Se acontecer, indica um bug no cálculo das posições.
              assert(false && "Index anomaly: write position is smaller than read position.");
              continue;
          }

          if (write_index > read_index) {
              // memmove é ideal para mover blocos de memória que podem se sobrepor.
              memmove(&local_edges[write_index], &local_edges[read_index], curr_degree * sizeof(DestID_));
          }
          // Se write_index == read_index, não é necessário fazer nada.

          // Atualiza o índice do vértice para sua nova posição.
          local_vertices[local_v_idx].index = write_index;
      }

      free(new_positions);
  }
  #endif
  
};

#endif  // GRAPH_H_
