# juniorstock/network/consensus/ternary_dag.py
import mlx.core as mx
import time
from typing import Dict, List, Tuple
from juniorstock.network.slate_ax_sync import SlateAXMeshRouter

class DAGNode:
    def __init__(self, topological_hash: str, parent_hashes: List[str]):
        self.topological_hash = topological_hash
        self.parents = parent_hashes
        self.timestamp = time.time_ns()

class TernaryDAGConsensus:
    """
    Zero-dependency distributed consensus for the Slate AX local subnet.
    Nodes broadcast their calculated Betti manifold hashes. Execution is gated 
    until a sufficient mesh weight W(n) converges on the target topological state.
    """
    def __init__(self, consensus_threshold: int = 3):
        self.threshold = consensus_threshold
        self.ax_router = SlateAXMeshRouter(port=10001)
        self.graph: Dict[str, DAGNode] = {}
        self.weights: Dict[str, int] = {}
        # Genesis hash initialized for M4 / Orange Pi cluster
        self.graph["GENESIS"] = DAGNode("GENESIS", [])
        self.weights["GENESIS"] = 1

    def append_manifold_state(self, betti_hash: str) -> bool:
        """
        Appends a locally calculated ternary state to the DAG and calculates weight.
        Returns True if consensus threshold is reached for immediate execution.
        """
        # Select parents (simple tip selection for architectural scaffolding)
        tips = [k for k, v in self.weights.items() if v == 1]
        parents = tips[:2] if len(tips) >= 2 else ["GENESIS"]

        new_node = DAGNode(betti_hash, parents)
        self.graph[betti_hash] = new_node
        self.weights[betti_hash] = 1

        # Calculate topological cumulative weight
        # W(n) = 1 + sum(W(children))
        cumulative_weight = self._compute_cumulative_weight(betti_hash)

        # Broadcast via UDP Multicast (Simulated string encode for DAG payloads)
        payload = f"{betti_hash}|{parents[0]}|{parents[1]}".encode('utf-8')
        try:
            self.ax_router.sock.sendto(payload, (self.ax_router.multicast_group, self.ax_router.port))
        except Exception:
            pass

        if cumulative_weight >= self.threshold:
            print(f"[DAG CONSENSUS] Mesh alignment achieved on state {betti_hash[:8]}. W(n)={cumulative_weight}.")
            return True
            
        return False

    def _compute_cumulative_weight(self, target_hash: str, visited=None) -> int:
        if visited is None:
            visited = set()
        
        weight = 1
        visited.add(target_hash)
        
        # Traverse reverse-edges (children referencing this node as parent)
        for node_hash, node in self.graph.items():
            if target_hash in node.parents and node_hash not in visited:
                weight += self._compute_cumulative_weight(node_hash, visited)
                
        return weight

    def poll_cluster_graph(self):
        """
        Ingests DAG edges broadcasted by Orange Pi secondary nodes.
        """
        while True:
            try:
                data, _ = self.ax_router.recv_sock.recvfrom(512)
                parts = data.decode('utf-8').split('|')
                if len(parts) == 3:
                    node_hash, p1, p2 = parts
                    if node_hash not in self.graph:
                        self.graph[node_hash] = DAGNode(node_hash, [p1, p2])
                        self.weights[node_hash] = 1
            except BlockingIOError:
                break
