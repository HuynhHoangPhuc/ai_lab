from queue import LifoQueue, Queue, PriorityQueue
from collections import defaultdict
from enum import Enum
from typing import Hashable
import abc

Node = Hashable
Neighbor = list[Hashable | tuple[Hashable, float]]


class GraphType(Enum):
    UNWEIGHTED = 1
    WEIGHTED = 2


class Graph(abc.ABC):
    def __init__(self):
        self.data: dict[Node, Neighbor] = defaultdict(list)

    @property
    def data(self) -> dict[Node, Neighbor]:
        return self.__graph

    @data.setter
    def data(self, graph: dict[Node, Neighbor]) -> None:
        self.__graph = graph

    @abc.abstractmethod
    def _neighbors(self, node: Node) -> Neighbor:
        pass

    @abc.abstractmethod
    def search(self, start: Node, goal: Node) -> tuple[list[Node], float | None]:
        pass


class UnweightedGraph(Graph):
    def __init__(self, search_type: str):
        super().__init__()

        if search_type not in ('BFS', 'DFS'):
            raise ValueError('search_type must be "BFS" or "DFS".')

        self.search_type = search_type

    def _neighbors(self, node: Node) -> Neighbor:
        return self.data[node]

    def search(self, start: Node, goal: Node) -> tuple[list[Node], float | None]:
        return self.__search(start, goal), None

    def __search(self, start: Node, goal: Node) -> list[Node]:
        frontier = Queue() if self.search_type == 'BFS' else LifoQueue()
        frontier.put(start)
        came_from: dict[Node, Node] = {start: -1}

        print(f'L = v{int(start) + 1}')
        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for _next in self.data[current]:
                if _next not in came_from:
                    frontier.put(_next)
                    came_from[_next] = current

            father = [int(k) + 1 for k, v in came_from.items() if v == current]
            print(f'Node = v{current + 1}', end='')
            print(f', L = [{", ".join([f"v{node + 1}" for node in list(frontier.queue)])}]', end='')
            print(f', father[{", ".join([f"v{node + 1}" for node in father])}] = v{current + 1}' if len(father) else '')

        if goal not in came_from:
            return []

        path = []
        while goal != start:
            path.append(goal)
            goal = came_from[goal]
        path.append(start)
        path.reverse()

        return path


class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()

    def search(self, start: Node, goal: Node) -> tuple[list[Node], float | None]:
        return self.__search(start, goal)

    def _neighbors(self, node: Node) -> Neighbor:
        return list(map(lambda x: x[0], self.data[node]))

    def __cost(self, from_node: Node, to_node: Node) -> float:
        return list(filter(lambda x: to_node in x, self.data[from_node]))[0][1]

    def __search(self, start: Node, goal: Node) -> tuple[list[Node], float]:
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from: dict[Node, Node] = {start: -1}
        cost_so_far: dict[Node, float] = {start: 0}

        print(f'PQ = (v{int(start) + 1},0)')
        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            for _next in self._neighbors(current):
                new_cost = cost_so_far[current] + self.__cost(current, _next)
                if _next not in cost_so_far or new_cost < cost_so_far[_next]:
                    cost_so_far[_next] = new_cost
                    frontier.put((new_cost, _next))
                    came_from[_next] = current

            print(f'PQ =', ', '.join(f'(v{int(node) + 1},{cost})' for cost, node in sorted(frontier.queue)))

        if goal not in came_from:
            return [], 0

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        return path, cost_so_far[goal]


def UCS(graph, start, end):
    visited = []
    frontier = PriorityQueue()

    frontier.put((0, start))
    visited.append(start)

    parent = dict()
    parent[start] = None

    path_found = False

    print(f'PQ = ({start},0)')
    while True:
        if frontier.empty():
            raise Exception("No way Exception")

        current_w, current_node = frontier.get()
        visited.append(current_node)

        if current_node == end:
            path_found = True
            break

        for nodei in graph[current_node]:
            node, weight = nodei
            if node not in visited:
                frontier.put((current_w + weight, node))
                parent[node] = current_node
                visited.append(node)
        print(f'PQ =', ', '.join(f'({node},{cost})' for cost, node in sorted(frontier.queue)))

    path = []
    if path_found:
        path.append(end)
        while parent[end] is not None:
            path.append(parent[end])
            end = parent[end]
        path.reverse()

    return current_w, path


def load_data(filename: str, graph_type: GraphType) -> tuple[dict, int, int]:
    f = open(filename, 'r')
    f.readline()
    start, goal = (int(num) for num in f.readline().split())
    graph: dict[int, list] = defaultdict(list)
    for i, row in enumerate(f.read().split('\n')):
        for j, value in enumerate(row.split()):
            if int(value) != 0 and graph_type is GraphType.UNWEIGHTED:
                graph[i].append(j)
            elif int(value) != 0 and graph_type is GraphType.WEIGHTED:
                graph[i].append((j, int(value)))
    f.close()

    return graph, start, goal


if __name__ == '__main__':
    gph: Graph = UnweightedGraph('BFS')
    gph.data, s, e = load_data('Input.txt', GraphType.UNWEIGHTED)
    result_path, _ = gph.search(s, e)
    print('Result for BFS algorithm:', end=' ')
    print('->'.join(str(f'v{int(node) + 1}') for node in result_path))
    print()

    gph: Graph = UnweightedGraph('DFS')
    gph.data, s, e = load_data('Input.txt', GraphType.UNWEIGHTED)
    result_path, _ = gph.search(s, e)
    print('Result for DFS algorithm:', end=' ')
    print('->'.join(str(f'v{int(node) + 1}') for node in result_path))
    print()

    gph: Graph = WeightedGraph()
    gph.data, s, e = load_data('InputUCS.txt', GraphType.WEIGHTED)
    result_path, result_cost = gph.search(s, e)
    print('Result for UCS algorithm:', end=' ')
    print('->'.join(str(f'v{int(node) + 1}') for node in result_path))
    print('Cost is:', result_cost)

    # gph: Graph = WeightedGraph()
    # file = open('test01.txt', 'r')
    # for line in file:
    #     s, e, c = line.split()
    #     gph.data[s].append((e, int(c)))
    # result_path, result_cost = gph.search('START', 'GOAL')
    # print('Result for UCS algorithm:', end=' ')
    # print('->'.join(str(node) for node in result_path))
    # print('Cost is:', result_cost)
    # print(UCS(gph.data, 'START', 'GOAL'))
