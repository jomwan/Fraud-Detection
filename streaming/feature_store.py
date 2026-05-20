import logging
import os
import random
import time

import redis

logger = logging.getLogger(__name__)

# Connect to Redis
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))

try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    # Ping to check connectivity
    redis_client.ping()
    logger.info("Connected to Redis at %s:%s", redis_host, redis_port)
except Exception as e:
    logger.warning("Redis connection failed: %s. Fallbacks will be active.", e)
    redis_client = None

def get_velocity_feature(sender_id):
    """Computes real-time velocity: count of transactions in the last 5 minutes."""
    if redis_client is None:
        return 1
    try:
        now = time.time()
        # Clean expired transactions
        redis_client.zremrangebyscore(f"vel:{sender_id}", 0, now - 300)
        # Add current transaction
        redis_client.zadd(f"vel:{sender_id}", {f"{now}_{random.random()}": now})
        # Set TTL to 5 minutes to avoid memory leak
        redis_client.expire(f"vel:{sender_id}", 300)
        return redis_client.zcard(f"vel:{sender_id}")
    except Exception as e:
        logger.error("Redis velocity error: %s", e)
        return 1

def get_user_sequence(sender_id, current_amount, max_len=5):
    """Maintains a rolling sliding window in Redis for the PyTorch LSTM sequence model."""
    if redis_client is None:
        return [0.0, 0.0, 0.0, 0.0, float(current_amount)]
    try:
        key = f"seq:{sender_id}"
        # Append current transaction amount to the right of the list
        redis_client.rpush(key, float(current_amount))
        # Trim list to maximum sequence length (keep rightmost max_len elements)
        redis_client.ltrim(key, -max_len, -1)
        # Set TTL to 1 hour to prevent orphan user sequence growth
        redis_client.expire(key, 3600)
        
        amounts = redis_client.lrange(key, 0, -1)
        amounts = [float(x) for x in amounts]
        
        # Chronological padding to ensure length = max_len
        if len(amounts) < max_len:
            padding = [0.0] * (max_len - len(amounts))
            amounts = padding + amounts  # Left-padded chronologically
            
        return amounts
    except Exception as e:
        logger.error("Redis sequence error: %s", e)
        return [0.0, 0.0, 0.0, 0.0, float(current_amount)]
