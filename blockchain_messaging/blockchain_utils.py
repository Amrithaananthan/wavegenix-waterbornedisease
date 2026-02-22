# D:\sihdeeo\blockchain_messaging\blockchain_utils.py
import hashlib
import json
import os
import time
from web3 import Web3
from .config import BLOCKCHAIN_NODE_URL, BLOCKCHAIN_PRIVATE_KEY, BLOCKCHAIN_ADDRESS, LEDGER_PATH

class BlockchainUtils:
    def __init__(self):
        try:
            self.web3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_NODE_URL))
            self.connected = self.web3.is_connected()
        except:
            self.connected = False
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    
    def generate_hash(self, message):
        """Generate SHA256 hash of message"""
        return hashlib.sha256(message.encode('utf-8')).hexdigest()
    
    def store_on_blockchain(self, message_hash):
        """Store message hash on Ethereum blockchain"""
        try:
            if not self.connected or not BLOCKCHAIN_PRIVATE_KEY:
                return self._store_locally(message_hash)
            
            account = self.web3.eth.account.from_key(BLOCKCHAIN_PRIVATE_KEY)
            nonce = self.web3.eth.get_transaction_count(account.address)
            
            transaction = {
                'nonce': nonce,
                'to': BLOCKCHAIN_ADDRESS,
                'value': 0,
                'gas': 200000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
                'data': message_hash[:32]  # Use first 32 chars
            }
            
            signed_txn = self.web3.eth.account.sign_transaction(transaction, BLOCKCHAIN_PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return self.web3.to_hex(tx_hash)
        except Exception as e:
            print(f"Blockchain error: {e}")
            return self._store_locally(message_hash)
    
    def _store_locally(self, message_hash):
        """Fallback: Store hash in local ledger"""
        try:
            if os.path.exists(LEDGER_PATH):
                with open(LEDGER_PATH, 'r') as f:
                    ledger = json.load(f)
            else:
                ledger = []
            
            entry = {
                'hash': message_hash,
                'timestamp': time.time(),
                'blockchain': False
            }
            
            ledger.append(entry)
            
            with open(LEDGER_PATH, 'w') as f:
                json.dump(ledger, f, indent=2)
            
            return f"local_{message_hash[:16]}"
        except Exception as e:
            print(f"Local storage error: {e}")
            return None

# Singleton instance
blockchain_utils = BlockchainUtils()