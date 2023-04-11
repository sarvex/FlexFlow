#include "utils/graph/digraph.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

DirectedEdge::DirectedEdge(Node src, Node dst) 
  : src(src), dst(dst)
{ }

bool DirectedEdge::operator==(DirectedEdge const &other) const {
  return visit_eq(*this, other);
}

bool DirectedEdge::operator<(DirectedEdge const &other) const {
  return visit_lt(*this, other);
}

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  return (
    s << "DirectedEdge{" << e.src.idx << " -> " << e.dst.idx << "}"
  );
}

DirectedEdgeQuery::DirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts) 
  : srcs(srcs), dsts(dsts)
{ }

DiGraph::DiGraph(DiGraph const &other)
  : ptr(other.ptr->clone())
{ }

DiGraph &DiGraph::operator=(DiGraph other) {
  swap(*this, other);
  return *this;
}

void swap(DiGraph &lhs, DiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DiGraph::add_node() {
  return this->ptr->add_node();
}

void DiGraph::add_node_unsafe(Node const &n) {
  return this->ptr->add_node_unsafe(n);
}

void DiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr->remove_node_unsafe(n);
}

void DiGraph::add_edge(DirectedEdge const &e) {
  return this->ptr->add_edge(e);
}

void DiGraph::remove_edge(DirectedEdge const &e) {
  return this->ptr->remove_edge(e);
}

std::unordered_set<DirectedEdge> DiGraph::query_edges(DirectedEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

DiGraph::DiGraph(std::unique_ptr<IDiGraph> _ptr) 
  : ptr(std::move(_ptr))
{ }

}

namespace std {
using ::FlexFlow::DirectedEdge;

size_t std::hash<DirectedEdge>::operator()(DirectedEdge const &e) const {
  return visit_hash(e);
}
}
