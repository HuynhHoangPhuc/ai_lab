from queue import Queue, PriorityQueue
from collections import defaultdict
from enum import Enum
from typing import Any
import abc


class GraphType(Enum):
    UNWEIGHTED = 1
    WEIGHTED = 2


class Graph(abc.ABC):
    def __init__(self):
        self.__graph: dict[int, list] = defaultdict(list)

    @property
    def data(self) -> dict:
        return self.__graph

    @data.setter
    def data(self, graph: dict[int, list]) -> None:
        self.__graph = graph

    @abc.abstractmethod
    def search(self, start: int, goal: int):
        pass


class UnweightedGraph(Graph):
    def __init__(self):
        super().__init__()

    # def __load_weighted(self, filename: str):
    #     f = open(filename, 'r')
    #     num_vertices = int(f.readline())
    #     start, goal = (int(num) for num in f.readline().split())
    #     graph: dict[int, list[tuple[int, int]]] = {}
    #     for i, r in enumerate(f.read().split('\n')):
    #         for j, c in enumerate(r.split()):
    #             if int(c) != 0:
    #                 graph[i].append((j, c))
    #     f.close()
    #
    #     self.num_vertices = num_vertices
    #     self.start = start
    #     self.goal = goal
    #     self.graph = graph

    def search(self, start: int, goal: int):
        frontier = Queue()
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

        return came_from
