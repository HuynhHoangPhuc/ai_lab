from queue import LifoQueue, Queue, PriorityQueue
from collections import defaultdict
from enum import Enum
import abc


class GraphType(Enum):
    UNWEIGHTED = 1
    WEIGHTED = 2


class Graph(abc.ABC):
    def __init__(self):
        self.data: dict[int, list] = defaultdict(list)

    @property
    def data(self) -> dict:
        return self.__graph

    @data.setter
    def data(self, graph: dict[int, list]) -> None:
        self.__graph = graph

    @abc.abstractmethod
    def _neighbors(self, node: int) -> list[int]:
        pass

    @abc.abstractmethod
    def search(self, start: int, goal: int) -> tuple[list[int], int | None]:
        pass


class UnweightedGraph(Graph):
    def __init__(self, search_type: str):
        super().__init__()

        if search_type not in ('BFS', 'DFS'):
            raise ValueError('search_type must be "BFS" or "DFS".')

        self.search_type = search_type

    def _neighbors(self, node: int) -> list[int]:
        return self.data[node]

    def search(self, start: int, goal: int) -> tuple[list[int], int | None]:
        return self.__search(start, goal), None

    def __search(self, start: int, goal: int) -> list[int]:
        frontier = Queue() if self.search_type == 'BFS' else LifoQueue()
        frontier.put(start)
        came_from: dict[int, int] = {start: start}

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for _next in self.data[current]:
                if _next not in came_from:
                    frontier.put(_next)
                    came_from[_next] = current

        if goal not in came_from:
            return []

        path = []
        while goal != start:
            path.append(goal)
            goal = came_from[goal]
        path.reverse()

        return path


class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()

    def search(self, start: int, goal: int) -> tuple[list[int], int | None]:
        pass

    def _neighbors(self, node: int) -> list[int]:
        return list(map(lambda x: x[0], self.data[node]))

    def __cost(self, from_node: int, to_node: int) -> int:
        return list(filter(lambda x: to_node in x, self.data[from_node]))[0]

    def __search(self, start: int, goal: int) -> tuple[list[int], int]:
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from: dict[int, int] = {start: start}
        cost_so_far: dict[int, int] = {start: 0}

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

        if goal not in came_from:
            return [], 0

        path = []
        while goal != start:
            path.append(goal)
            goal = came_from[goal]
        path.reverse()

        return path, cost_so_far[goal]


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
                graph[i].append((j, value))
    f.close()

    return graph, start, goal


if __name__ == '__main__':
    g, s, f = load_data('Input.txt', GraphType.UNWEIGHTED)
    gph = UnweightedGraph("BFS")
    gph.data = g
    print(gph.search(s, f))
