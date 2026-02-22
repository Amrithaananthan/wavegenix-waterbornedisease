import hashlib
import json
import time
from datetime import datetime

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class WaterQualityBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_alerts = []
        self.difficulty = 2

    def create_genesis_block(self):
        return Block(0, datetime.now().isoformat(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_alert(self, alert_data):
        """Add a new water quality alert to pending transactions"""
        alert = {
            'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            'timestamp': datetime.now().isoformat(),
            'type': alert_data.get('type', 'warning'),
            'severity': alert_data.get('severity', 'medium'),
            'message': alert_data.get('message', ''),
            'parameters': alert_data.get('parameters', {}),
            'location': alert_data.get('location', 'Unknown'),
            'device_id': alert_data.get('device_id', 'Unknown'),
            'ml_confidence': alert_data.get('ml_confidence', 0),
            'status': 'pending'
        }
        self.pending_alerts.append(alert)
        return alert

    def mine_pending_alerts(self):
        """Mine pending alerts into a new block"""
        if not self.pending_alerts:
            return None

        latest_block = self.get_latest_block()
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data={
                'alerts': self.pending_alerts.copy(),
                'mined_at': datetime.now().isoformat()
            },
            previous_hash=latest_block.hash
        )

        # Proof of Work (simplified)
        while not new_block.hash.startswith('0' * self.difficulty):
            new_block.timestamp = datetime.now().isoformat()
            new_block.hash = new_block.calculate_hash()

        self.chain.append(new_block)
        mined_alerts = self.pending_alerts.copy()
        self.pending_alerts = []
        
        print(f"✅ Mined block #{new_block.index} with {len(mined_alerts)} alerts")
        return new_block

    def get_alert_history(self):
        """Get all alerts from blockchain"""
        alerts = []
        for block in self.chain[1:]:  # Skip genesis block
            if 'alerts' in block.data:
                alerts.extend(block.data['alerts'])
        return alerts

    def validate_chain(self):
        """Validate the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_blockchain_info(self):
        return {
            'chain_length': len(self.chain),
            'pending_alerts': len(self.pending_alerts),
            'is_valid': self.validate_chain(),
            'total_alerts': len(self.get_alert_history())
        }

# Global blockchain instance
water_blockchain = WaterQualityBlockchain()