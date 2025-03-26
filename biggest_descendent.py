import dsc40graph
def biggest_descendent(graph, root, value, biggest = None):
    if biggest is None:
        biggest = {}
    biggest[root] = value[root]
    for v in graph.neighbors(root):
        biggest_descendent(graph, v, value, biggest)
        if biggest[v] > biggest[root]:
            biggest[root] = biggest[v]
    return biggest