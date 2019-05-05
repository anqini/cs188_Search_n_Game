# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest statesVisited in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from game import Directions
    # initialize
    resultList = []
    currentState = problem.getStartState()
    # statesVisited stores all statesVisited that Pacman have passed through
    statesVisited = util.Stack()
    statesVisited.push(currentState)
    fringe = util.Stack()
    # parent is a dictionary to store all trace
    parent = {}
    fringe.push(currentState)
    # loop
    while not fringe.isEmpty():
    # test the loop for 5 times
    # for i in range(5):
        # step
        nextState = fringe.pop()
        statesVisited.push(nextState)
        currentState = nextState
        fringe.list = [x for x in fringe.list if x != currentState]
        # print("the state comes to ", currentState)
        # goal test
        if problem.isGoalState(currentState):
            return backtrace(parent, problem.getStartState(), currentState)
        # get successor behavior
        successorList = problem.getSuccessors(currentState)
        for successorState, direction in [(x[0], x[1]) for x in successorList if x[0] not in statesVisited.list]:
            parent[successorState] = (currentState, direction)
            fringe.push(successorState)
        # print information to help debug
        # print("the fringe is ", fringe.list)
        # print("the parent dic is ", parent)
    print("SORRY, cannot find solution!")
    sys.exit(1)


def breadthFirstSearch(problem):
    """Search the shallowest statesVisited in the search tree first."""
    from game import Directions
    # initialize
    resultList = []
    currentState = problem.getStartState()
    # print("the initial state is ", currentState)
    # statesVisited stores all statesVisited that Pacman have passed through
    statesVisited = util.Stack()
    statesVisited.push(currentState)
    # define a parent dic to trace back the path
    parent = {}
    fringe = util.Queue()
    fringe.push(currentState)
    # loop
    while not fringe.isEmpty():
    # test the loop for 5 times
    # for i in range(5):
        # step
        nextState = fringe.pop()
        statesVisited.push(nextState)
        currentState = nextState
        # print("the state comes to ", currentState)
        # goal test
        if problem.isGoalState(currentState):
            return backtrace(parent, problem.getStartState(), currentState)
        # get successor behavior
        successorList = problem.getSuccessors(currentState)
        for successorState, direction in [(x[0], x[1]) for x in successorList if x[0] not in statesVisited.list and x[0] not in fringe.list]:
            parent[successorState] = (currentState, direction)
            fringe.push(successorState)
        # print information to help debug
        # print("the fringe is ", fringe.list)
        # print("the parent dic is ", parent)
    print("SORRY, cannot find solution!")
    sys.exit(1)
    

'''
    this backtrace function comes from stackoverflow "How to trace the path in a Breadth-First Search?"
    the website is : https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search#
'''
def backtrace(parent, start, end):
    path = [end]
    direction = []
    while path[-1] != start:
        # print(parent[path[-1]])
        direction.append(parent[path[-1]][1])
        path.append(parent[path[-1]][0])
    return direction[::-1]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from game import Directions
    # initialize
    resultList = []
    currentState = problem.getStartState()
    # statesVisited stores all statesVisited that Pacman have passed through
    statesVisited = util.Stack()
    statesVisited.push(currentState)
    # define a parent dic to trace back the path
    parent = {}
    costDic = {}
    costDic[currentState] = 0
    fringe = util.PriorityQueue()
    fringe.push(currentState, 1)
    print("the first location is ", currentState)
    # loop
    # while not fringe.isEmpty():
    while not fringe.isEmpty():
    # test the loop for 5 times
    # for i in range(5):
        # step
        nextState = fringe.pop()
        statesVisited.push(nextState)
        currentState = nextState
        # print("the state comes to ", currentState)
        # goal test
        if problem.isGoalState(currentState):
            return backtrace(parent, problem.getStartState(), currentState)
        # get successor behavior
        successorList = problem.getSuccessors(currentState)
        for successorState, direction, cost in [(x[0], x[1], x[2]) for x in successorList if x[0] not in statesVisited.list]:
            if successorState not in parent:
                parent[successorState] = (currentState, direction)
                costDic[successorState] = cost + costDic[currentState]
                fringe.update(successorState, costDic[successorState])
            else: 
                if costDic[successorState] > cost + costDic[currentState]:
                    costDic[successorState] = cost + costDic[currentState]
                    parent[successorState] = (currentState, direction)
                    fringe.update(successorState, costDic[successorState])

        # print information to help debug
        # print("the fringe is ", fringe.list)
        # print("the parent dic is ", parent)
    print("SORRY, cannot find solution!")
    sys.exit(1)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from game import Directions
    # initialize
    resultList = []
    currentState = problem.getStartState()
    # statesVisited stores all statesVisited that Pacman have passed through
    # print("the start state is ", currentState)
    statesVisited = util.Stack()
    statesVisited.push(currentState)
    # define a parent dic to trace back the path
    parent = {}
    costDic = {}
    costDic[currentState] = 0
    heuristic.problem = problem
    fringe = util.PriorityQueue()
    fringe.push(currentState, 1)
    # loop
    while not fringe.isEmpty():
        # step
        nextState = fringe.pop()
        statesVisited.push(nextState)
        currentState = nextState
        # print("now we come to ", currentState)
        # goal test
        if problem.isGoalState(currentState):
            return backtrace(parent, problem.getStartState(), currentState)
        # get successor behavior
        successorList = problem.getSuccessors(currentState)
        for successorState, direction, cost in [(x[0], x[1], x[2]) for x in successorList if x[0] not in statesVisited.list]:
            if successorState not in parent:
                parent[successorState] = (currentState, direction)
                costDic[successorState] = cost + costDic[currentState]
                fringe.update(successorState, costDic[successorState] + heuristic(successorState, problem))
            else: 
                if costDic[successorState] > cost + costDic[currentState]:
                    costDic[successorState] = cost + costDic[currentState]
                    parent[successorState] = (currentState, direction)
                    fringe.update(successorState, costDic[successorState] + heuristic(successorState, problem))

        # successorList = problem.getSuccessors(currentState)
        # for successorState, direction in [(x[0],x[1]) for x in successorList if x[0] not in statesVisited.list and x[0] not in fringe.heap]:
        #     parent[successorState] = (currentState, direction)
        #     fringe.update(successorState, heuristic(successorState, problem))
    print("SORRY, cannot find solution!")
    sys.exit(1)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
