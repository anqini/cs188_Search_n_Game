# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Actions

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print(newFood)
        dis_ghost = nearestDis([ghosts.getPosition() for ghosts in newGhostStates], newPos)
        food_heuristic = avgDis(newFood, newPos) + mazeDistance(nearestFood(newFood, newPos), newPos, currentGameState)
        if newScaredTimes[0] > 0:
                score =  1 / (food_heuristic / 15 + 1) -len(newFood) / 3 + 50
        else:
                score =  -len(newFood) / 5 - 1 / (dis_ghost / 5 + 1) + 1 / (food_heuristic / 15 + 1)
        return score

def nearestDis(AList, state):
    if len(AList) == 0:
        return 0
    minDis = 99999
    for position in AList:
        dis = manhattanDistance(position, state)
        if dis < minDis:
            minDis = dis
    return minDis

def nearestFood(AList, state):
    if len(AList) == 0:
        return 0
    minDis = 99999
    dic = {}
    for position in AList:
        dis = manhattanDistance(position, state)
        if dis < minDis:
            dic[dis] = position
            minDis = dis
    return dic[minDis]

def avgDis(AList, state):
	if len(AList) == 0:
		return 0
	sum = 0
	for position in AList:
		sum += manhattanDistance(position, state)
	avg = sum / len(AList)
	return avg


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.step = []

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        agentIndex = 0
        # print(gameState.getNumAgents())
        max_value2(self, gameState, agentIndex, self.depth)
        # print(self.step)
        return self.step[-1]

def value2(MinimaxAgent, gameState, agentIndex, depth):
    if (gameState.isWin() == True) or (gameState.isLose() == True) or (depth <= 0):
        return MinimaxAgent.evaluationFunction(gameState)
    if agentIndex == 0:
        return max_value2(MinimaxAgent, gameState, agentIndex, depth)
    else:
        return min_value2(MinimaxAgent, gameState, agentIndex, depth)

def max_value2(MinimaxAgent, gameState, agentIndex, depth):
        utility_list = []
        nextstep = {}
        # print('pacman has actions', gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                u = value2(MinimaxAgent, suc_state, agentIndex + 1, depth)
                nextstep[u] = action
                utility_list.append(u)
                # print(action, ' is finished')
                # print()
                # print()
        u_max = max(utility_list)
        # print('pacman list is ', nextstep)
        # print()
        # print()
        MinimaxAgent.step.append(nextstep[u_max])
        return u_max

def min_value2(MinimaxAgent, gameState, agentIndex, depth):
        utility_list = []
        nextstep = {}
        # print('ghost', agentIndex, ' has actions', gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                        u = value2(MinimaxAgent, suc_state, 0, depth - 1)
                else:
                        u = value2(MinimaxAgent, suc_state, agentIndex + 1, depth)
                nextstep[u] = action
                utility_list.append(u)
        u_min = min(utility_list)
        # print('ghost', agentIndex, ' is ',nextstep)
        # print()
        # print()
        return u_min

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentIndex = 0
        # print(gameState.getNumAgents())
        max_value(self, gameState, agentIndex, self.depth, -99999, 99999)
        # print(self.step)
        return self.step[-1]


def value(MinimaxAgent, gameState, agentIndex, depth, alpha, beta):
    if gameState.isWin() == True:
        # print('win')
        return MinimaxAgent.evaluationFunction(gameState)
    if gameState.isLose() == True:
        # print('lose')
        return MinimaxAgent.evaluationFunction(gameState)
    if depth > 0:
        if agentIndex == 0:
            return max_value(MinimaxAgent, gameState, agentIndex, depth, alpha, beta)
        else:
            return min_value(MinimaxAgent, gameState, agentIndex, depth, alpha, beta)
    else:
        # print('agent ', agentIndex, ' have evaluation', MinimaxAgent.evaluationFunction(gameState))
        return MinimaxAgent.evaluationFunction(gameState)

def max_value(MinimaxAgent, gameState, agentIndex, depth, alpha, beta):
        v = -99999
        utility_list = []
        nextstep = {}
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                u = value(MinimaxAgent, suc_state, agentIndex + 1, depth, alpha, beta)
                v = max(u, v)
                if v > beta:
                        return v
                alpha = max(alpha, v)
                nextstep[u] = action
                utility_list.append(u)
        u_max = max(utility_list)
        MinimaxAgent.step.append(nextstep[u_max])
        return u_max

def min_value(MinimaxAgent, gameState, agentIndex, depth, alpha, beta):
        v = 99999
        utility_list = []
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                        u = value(MinimaxAgent, suc_state, 0, depth - 1, alpha, beta)
                        v = min(u, v)
                        if v < alpha:
                                return v
                        beta = min(beta, v)
                else:
                        u = value(MinimaxAgent, suc_state, agentIndex + 1, depth, alpha, beta)
                        v = min(u, v)
                        if v < alpha:
                                return v
                        beta = min(beta, v)
                utility_list.append(u)
        u_min = min(utility_list)
        return u_min




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agentIndex = 0
        max_value4(self, gameState, agentIndex, self.depth)
        return self.step[-1]

def value4(MinimaxAgent, gameState, agentIndex, depth):
    if (gameState.isWin() == True) or (gameState.isLose() == True) or (depth <= 0):
        # print('win')
        return MinimaxAgent.evaluationFunction(gameState)
    if agentIndex == 0:
        return max_value4(MinimaxAgent, gameState, agentIndex, depth)
    else:
        return expect_value(MinimaxAgent, gameState, agentIndex, depth)

def expect_value(MinimaxAgent, gameState, agentIndex, depth):
        utility_list = []
        # print('ghost', agentIndex, ' has actions', gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                        u = value4(MinimaxAgent, suc_state, 0, depth - 1)
                else:
                        u = value4(MinimaxAgent, suc_state, agentIndex + 1, depth)
                utility_list.append(u)
        u_expect = sum(utility_list) / len(utility_list)
        # print('ghost', agentIndex, ' is ',nextstep)
        # print()
        # print()
        return u_expect

def max_value4(MinimaxAgent, gameState, agentIndex, depth):
        utility_list = []
        nextstep = {}
        for action in gameState.getLegalActions(agentIndex):
                suc_state = gameState.generateSuccessor(agentIndex, action)
                u = value4(MinimaxAgent, suc_state, agentIndex + 1, depth)
                nextstep[u] = action
                utility_list.append(u)
        u_max = max(utility_list)
        MinimaxAgent.step.append(nextstep[u_max])
        return u_max

def betterEvaluationFunction(currentGameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
        """
        # newPos = currentGameState.getPacmanPosition()
        # newFood = currentGameState.getFood()
        # newGhostStates = currentGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # #newFood = currentGameState.getFood().asList()
        # dis_ghost = nearestDis([ghosts.getPosition() for ghosts in newGhostStates], newPos)
        # food_heuristic = avgDis(newFood.asList(), newPos) + mazeDistance(nearestFood(newFood.asList(), newPos), newPos, currentGameState)
        # score =  -newFood.count() / 3 - 1 / (dis_ghost / 3 + 1) + 1 / (food_heuristic / 15 + 1)
        # return score
        
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood().asList()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print(newFood)
        dis_ghost = nearestDis([ghosts.getPosition() for ghosts in newGhostStates], newPos)
        food_heuristic = avgDis(newFood, newPos) + mazeDistance(nearestFood(newFood, newPos), newPos, currentGameState)
        if newScaredTimes[0] > 0:
                score =  1 / (food_heuristic / 15 + 1) -len(newFood) / 3 + 50
        else:
                score =  -len(newFood) / 5 - 1 / (dis_ghost / 5 + 1) + 1 / (food_heuristic / 15 + 1)
        return score

# Abbreviation
better = betterEvaluationFunction

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    if point1 == 0:
        return 0
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def backtrace(parent, start, end):
    path = [end]
    direction = []
    while path[-1] != start:
        # print(parent[path[-1]])
        direction.append(parent[path[-1]][1])
        path.append(parent[path[-1]][0])
    return direction[::-1]

def breadthFirstSearch(problem):
    from game import Directions

    resultList = []
    currentState = problem.getStartState()
    statesVisited = util.Stack()
    statesVisited.push(currentState)
    parent = {}
    fringe = util.Queue()
    fringe.push(currentState)
    # loop
    while not fringe.isEmpty():
        nextState = fringe.pop()
        statesVisited.push(nextState)
        currentState = nextState
        if problem.isGoalState(currentState):
            return backtrace(parent, problem.getStartState(), currentState)
        successorList = problem.getSuccessors(currentState)
        for successorState, direction in [(x[0], x[1]) for x in successorList if x[0] not in statesVisited.list and x[0] not in fringe.list]:
            parent[successorState] = (currentState, direction)
            fringe.push(successorState)
    print("SORRY, cannot find solution!")
    sys.exit(1)