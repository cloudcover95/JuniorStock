# juniorstock/orchestration/graph_compiler.py
import json
from typing import Dict, List, Any
from juniorstock.nodes.asset_node import UniversalAssetNode
from juniorstock.nodes.web3_node import Web3ExecutionNode
from juniorstock.bots.cross_chain_arb import TopoArbitrageBot

class BlueprintCompiler:
    """
    Interprets JSON-exported visual graphs (EventNode -> ActionNode).
    Traverses edges to dynamically instantiate the execution graph in memory.
    """
    def __init__(self, blueprint_path: str):
        self.blueprint_path = blueprint_path
        self.nodes_in_memory: Dict[str, Any] = {}
        self.execution_sequence: List[tuple] = []

    def load_blueprint(self):
        with open(self.blueprint_path, 'r') as f:
            self.graph_data = json.load(f)

    def compile_graph(self):
        """
        Parses JSON nodes and maps them to JuniorStock Python objects.
        """
        # Pass 1: Instantiate nodes
        for node in self.graph_data.get('nodes', []):
            node_type = node.get('type')
            node_id = node.get('id')
            
            if node_type == 'UniversalAssetNode':
                self.nodes_in_memory[node_id] = UniversalAssetNode(asset_id=node.get('data', {}).get('ticker'))
            elif node_type == 'Web3ExecutionNode':
                self.nodes_in_memory[node_id] = Web3ExecutionNode()
            elif node_type == 'TopoArbitrageBot':
                contract = node.get('data', {}).get('target_contract')
                self.nodes_in_memory[node_id] = TopoArbitrageBot(target_contract=contract)

        # Pass 2: Map edges to execution sequence
        for edge in self.graph_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            
            # Formulate the deterministic execution chain
            if source in self.nodes_in_memory and target in self.nodes_in_memory:
                self.execution_sequence.append((self.nodes_in_memory[source], self.nodes_in_memory[target]))

        return self.execution_sequence

    def execute_tick(self):
        """
        Fires the compiled sequence array instantly through the high-speed loop.
        """
        for source_node, target_node in self.execution_sequence:
            if isinstance(target_node, TopoArbitrageBot) and isinstance(source_node, UniversalAssetNode):
                # Placeholder logic: Assumes a secondary stream is available in the target node
                # In production, the graph compiler maps multi-edge inputs directly to the bot evaluate fn.
                target_node.evaluate_and_execute(source_node, source_node)
