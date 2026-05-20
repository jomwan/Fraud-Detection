from neo4j import GraphDatabase
import time
import os

neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

try:
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    # Test connection
    with neo4j_driver.session() as session:
        session.run("RETURN 1")
    print(f"[SUCCESS] Connected to Neo4j at {neo4j_uri}")
except Exception as e:
    print(f"[WARNING] Neo4j connection failed: {e}. Fallbacks will be active.")
    neo4j_driver = None

# Dynamic caching for total nodes count to prevent continuous graph size calculation overhead
last_node_count_check = 0.0
cached_node_count = 100  # Conservative estimate fallback

def get_total_graph_nodes():
    """Fetches total unique node count from Neo4j, using a 60-second caching strategy."""
    global last_node_count_check, cached_node_count
    if neo4j_driver is None:
        return cached_node_count
        
    now = time.time()
    if now - last_node_count_check > 60:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as cnt")
                record = result.single()
                if record:
                    cached_node_count = max(10, record['cnt'])
                    last_node_count_check = now
        except Exception as e:
            print(f"Neo4j node count check failed: {e}")
    return cached_node_count

def update_and_get_graph_features(sender_id, receiver_id):
    """
    Inserts/updates transactional edges in Neo4j and returns degree centrality.
    Normalizes degree centrality in real-time by total node count.
    """
    if neo4j_driver is None:
        return 0.001, 0.001
        
    try:
        with neo4j_driver.session() as session:
            # 1. Merge transaction parties and record transaction occurrence
            session.run("""
                MERGE (s:User {id: $sender})
                MERGE (r:User {id: $receiver})
                MERGE (s)-[t:TRANSACTED]->(r)
                ON CREATE SET t.count = 1
                ON MATCH SET t.count = t.count + 1
            """, sender=sender_id, receiver=receiver_id)

            # 2. Get degrees (In + Out connections) for both parties
            query = """
                MATCH (u:User {id: $uid})
                RETURN count { (u)--() } as degree
            """
            sender_deg_raw = session.run(query, uid=sender_id).single()
            receiver_deg_raw = session.run(query, uid=receiver_id).single()

            sender_degree = sender_deg_raw['degree'] if sender_deg_raw else 1
            receiver_degree = receiver_deg_raw['degree'] if receiver_deg_raw else 1

            # 3. Normalize centrality dynamically using cached graph size
            n_size = get_total_graph_nodes()
            norm_factor = max(1, n_size - 1)

            return sender_degree / norm_factor, receiver_degree / norm_factor
    except Exception as e:
        print(f"Neo4j graph features extraction failed: {e}")
        return 0.001, 0.001
