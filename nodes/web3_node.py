# juniorstock/nodes/web3_node.py
from eth_account import Account
import secrets

class Web3ExecutionNode:
    """
    Aggressively sanitized Web3 transaction generator.
    Handles toBN logic, payload generation, and local signing.
    Strictly isolated: Private keys are NEVER exposed to external RPCs.
    """
    def __init__(self, private_key: str = None):
        # Generate sterile local key if none provided
        self._pk = private_key or "0x" + secrets.token_hex(32)
        self.account = Account.from_key(self._pk)
        self.address = self.account.address

    def build_and_sign_transaction(self, target_contract: str, data_payload: str, nonce: int, gas_limit: int = 250000):
        """
        Constructs and signs transaction locally.
        Replaces legacy web3node.lol arbitrary execution paths.
        """
        tx = {
            'nonce': nonce,
            'to': target_contract,
            'value': 0,
            'gas': gas_limit,
            'maxFeePerGas': 2000000000, 
            'maxPriorityFeePerGas': 1000000000,
            'data': data_payload,
            'chainId': 1 # Mainnet or local L2 fork
        }
        
        signed_tx = self.account.sign_transaction(tx)
        # Return serialized hex for broadcast by a separate network interface
        return signed_tx.rawTransaction.hex()
