# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

################################################################################
#                                   Honor Code
# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
# WRITTEN BY OTHER STUDENTS - Joseph Barbati
################################################################################

from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules = currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules = successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
        currPos = currentGameState.getPacmanPosition()
        foodList = currentFood.asList() + currentCapsules
        
        currMin = min([manhattanDistance(currPos, food) for food in foodList])
        newMin = min([manhattanDistance(newPos, food) for food in foodList])
        	
        if not [ghost.getPosition() for ghost in newGhostStates]:
        	return 999999
        	
        if successorGameState.isWin():
        	return 999999
        if successorGameState.isLose():
        	return -999999
        score = 0
        if currentGameState.getNumFood() > successorGameState.getNumFood():
        	score += 100
        if action == Directions.STOP:
        	score += -25
        if currPos in currentCapsules:
        	score += 300
        if newPos in currentCapsules:
        	score += 500
        if currPos in foodList:
        	score += 10
        if newPos in foodList:
        	score += 1000
        if newPos in [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer == 0]:
        	score -= 350
        if newPos in [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer > 0]:
        	score += 1000
        if currMin < newMin:
        	score -= 200

        return score

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
        """
        legalActions = gameState.getLegalActions(0) # pacman
        numAgents = gameState.getNumAgents()
        bestAction = None
        maxi = -999999
        for action in legalActions:
            currDepth = 0
            # Get the current max of the graph
            currMax = self.min_value(gameState.generateSuccessor(0, action), currDepth, 1, numAgents) # 1 for ghost
            if currMax > maxi:
                # if that is greater than last known max, update the best action
                maxi = currMax
                bestAction = action
        return bestAction


    def min_value(self, state, currDepth, agentIndex, numAgents):
        if state.isWin() or state.isLose():
            # If state is terminal, return evalFun
            return self.evaluationFunction(state)
        v = 999999
        for action in state.getLegalActions(agentIndex):
            newAgentIndex = (agentIndex + 1) % numAgents
            if agentIndex == state.getNumAgents() - 1:
                # Direct child is a MAX node, we want to find the min of the maxes
                v = min([v, self.max_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents)])
            else:
                # Otherwise we want to find min of the mins
                v = min([v, self.min_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents)])
        return v

    def max_value(self, state, currDepth, agentIndex, numAgents):
        currDepth += 1
        if state.isWin() or state.isLose():
            # If state is terminal, return evalFun
            return self.evaluationFunction(state)
        elif currDepth == self.depth:
            # Or if max depth has been reached
            return self.evaluationFunction(state)
        v = -999999
        for action in state.getLegalActions(agentIndex):
            newAgentIndex = (agentIndex + 1) % numAgents
            # Child is min node, so we want to find max of those mins
            v = max([v, self.min_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents)])
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalActions = gameState.getLegalActions(0) # pacman
        numAgents = gameState.getNumAgents()
        bestAction = None
        maxi = -999999
        a = -999999
        b = 999999
        for action in legalActions:
            currDepth = 0
            # Get the current max of the graph
            currMax = self.min_value(gameState.generateSuccessor(0, action), currDepth, 1, numAgents, a, b) # 1 for ghost
            if currMax > maxi:
                # if that is greater than last known max, update the best action
                maxi = currMax
                bestAction = action
            if currMax > b:
                # If currMax is greater than the beta for the root, return the bestAction
                return bestAction
            a = max([currMax, a])
        return bestAction


    def min_value(self, state, currDepth, agentIndex, numAgents, a, b):
        if state.isWin() or state.isLose():
            # If state is terminal, return evalFun
            return self.evaluationFunction(state)
        v = 999999
        for action in state.getLegalActions(agentIndex):
            newAgentIndex = (agentIndex + 1) % numAgents
            if agentIndex == state.getNumAgents() - 1:
                # Direct child is a MAX node, we want to find the min of the maxes
                v = min([v, self.max_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents, a, b)])
            else:
                # Otherwise we want to find min of the mins
                v = min([v, self.min_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents, a, b)])
            if v < a:
                return v
            b = min([v, b])
        return v

    def max_value(self, state, currDepth, agentIndex, numAgents, a, b):
        currDepth += 1
        if state.isWin() or state.isLose():
            # If state is terminal, return evalFun
            return self.evaluationFunction(state)
        elif currDepth == self.depth:
            # Or if max depth has been reached
            return self.evaluationFunction(state)
        v = -999999
        for action in state.getLegalActions(agentIndex):
            newAgentIndex = (agentIndex + 1) % numAgents
            # Child is min node, so we want to find max of those mins
            v = max([v, self.min_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents, a, b)])
            if v > b:
                return v
            a = max([v, a])
        return v

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
        legalActions = gameState.getLegalActions(0) # pacman
        numAgents = gameState.getNumAgents()
        bestAction = None
        maxi = -999999
        for action in legalActions:
            currDepth = 0
            # Get the current max of the graph
            currMax = self.expected_value(gameState.generateSuccessor(0, action), currDepth, 1, numAgents) # 1 for ghost
            if currMax > maxi:
                # if that is greater than last known max, update the best action
                maxi = currMax
                bestAction = action
        return bestAction

    def max_value(self, state, currDepth, agentIndex, numAgents):
        currDepth += 1
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif currDepth == self.depth:
            return self.evaluationFunction(state)
        v = -999999
        for action in state.getLegalActions(agentIndex):
            newAgentIndex = (agentIndex + 1) % numAgents
            v = max([v, self.expected_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents)])
        return v

    # Probability of each expect-node is 1/(num of legal moves)
    def expected_value(self, state, currDepth, agentIndex, numAgents):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        ex = 0.0
        actions = state.getLegalActions(agentIndex)
        prob = 1.0/float(len(actions))
        for action in actions:
            newAgentIndex = (agentIndex + 1) % numAgents
            if agentIndex == numAgents - 1:
                ex += (self.max_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents) * prob)
            else:
                ex += (self.expected_value(state.generateSuccessor(agentIndex, action), currDepth, newAgentIndex, numAgents) * prob)
        return ex


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999

    currPos = currentGameState.getPacmanPosition()
    currentCapsules = currentGameState.getCapsules() #power pellets/capsules available from current state
    foodList = currentGameState.getFood().asList() + currentCapsules
    
    ghostStates = currentGameState.getGhostStates()
    
    currMin = 1.0
    if len(foodList) > 0:
        currMin = min([manhattanDistance(currPos, food) for food in foodList])
    currGhosts = sum([manhattanDistance(currPos, ghost.getPosition()) for ghost in ghostStates])
    closeCap = 1.0
    if len(currentCapsules) > 0:
        closeCap = min([manhattanDistance(currPos, cap) for cap in currentCapsules])
    	
    score = 0
    score -= currentGameState.getNumFood()
    if currPos in currentCapsules:
    	score += 300
    if currPos in foodList:
    	score += 10
    score += 1.0 / currMin
    totalTime = sum([ghost.scaredTimer for ghost in ghostStates])
    if totalTime > 0:
        score += 1.0 / (currGhosts * 80)
    else:
        score -= 1.2 * currGhosts
    score += 4.5 * 1.0 / closeCap
    score += currentGameState.getScore()

    return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
