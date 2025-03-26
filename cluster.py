def cluster(graph, weights, level):
    status = {node: 'undiscovered' for node in graph.nodes}
    clusters = []

    for node in graph.nodes:
        if status[node] == 'undiscovered':
            
            cluster_nodes = bfs(graph, node, weights, level, status)
            clusters.append(frozenset(cluster_nodes))

    return frozenset(clusters)
    
def bfs(graph, source, weights, level, status = None):
    from collections import deque

    # Initialize BFS queue
    queue = deque([source])
    visited = []  # Nodes in the current cluster
    status[source] = 'pending'

    while queue:
        current = queue.popleft()
        visited.append(current)

        for neighbor in graph.neighbors(current):
            if weights(current, neighbor) >= level and status[neighbor] == 'undiscovered':
                status[neighbor] = 'pending'
                queue.append(neighbor)

        status[current] = 'visited'

    return visited
    