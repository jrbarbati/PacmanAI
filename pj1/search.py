# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
################################################################################
#                                   Honor Code
# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
# WRITTEN BY OTHER STUDENTS - Joseph Barbati
################################################################################

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

################################################################################
# Node - Student-defined
#   Holds a state, totalCost from root, parent Node and action from parent to
#   get to current Node.
#   def getActions() --> returns the actions from root to the current Node
#   def getTotalCost() --> returns the total cost from root to current Node
#   def getStatePropertyAtIndex(num) --> returns the property of the state at
#                                        position num. Useful when the state
#                                        is a tuple of length n, n > 1. Returns
#                                        None if state has only one property.
################################################################################
class Node(object):
    def __init__(self, state, action, cost, parent):
        self.actions = []   #e.g., to store full path from root
        self.actionFromParent = action
        self.cost = cost
        self.state = state
        self.parent = parent

    def getActions(self):
        self.actions.insert(0, self.actionFromParent)
        parent = self.parent
        if parent == None:
            return self.actions
        while parent.parent != None:
            self.actions.insert(0, parent.actionFromParent)
            parent = parent.parent
        return self.actions

    def getTotalCost(self):
        cost = self.cost
        parent = self.parent
        while parent != None:
            cost += parent.cost
            parent = parent.parent
        return cost

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    explored = {}
    startState = problem.getStartState()

    fringe = util.Stack()
    root = Node(startState, '', 0, None)
    fringe.push(root)

    while not fringe.isEmpty():
        currNode = fringe.pop()
        if problem.isGoalState(currNode.state):
            # If the current state is the goal state, return the actions
            # from root to currNode.
            return currNode.getActions()
        if currNode.state not in explored:
            # If the current position is not in explored, it has not been explored,
            # so insert it into explored with initial value of False (not explored)
            explored[currNode.state] = False
        if explored[currNode.state]:
            # If currNode's position has already been explored, we don't explore it again
            continue
        for thing in problem.getSuccessors(currNode.state):
            # Exploring currNode's state
            temp = Node(thing[0], thing[1], thing[2], currNode)
            fringe.push(temp)
        explored[currNode.state] = True # currNode's position has been explored

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    explored = {}
    startState = problem.getStartState()

    fringe = util.Queue()
    root = Node(startState, '', 0, None)
    fringe.push(root)

    while not fringe.isEmpty():
        currNode = fringe.pop()
        if problem.isGoalState(currNode.state):
            # If the current state is the goal state, return the actions
            # from root to currNode.
            return currNode.getActions()
        if currNode.state not in explored:
            # If the current position is not in explored, it has not been explored,
            # so insert it into explored with initial value of False (not explored)
            explored[currNode.state] = False
        if explored[currNode.state]:
            # If currNode's position has already been explored, we don't explore it again
            continue
        for thing in problem.getSuccessors(currNode.state):
            # Exploring currNode's state
            temp = Node(thing[0], thing[1], thing[2], currNode)
            fringe.push(temp)
        explored[currNode.state] = True # currNode's position has been explored


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    explored = {}
    startState = problem.getStartState()

    fringe = util.PriorityQueue()
    root = Node(startState, '', 0, None)
    fringe.push(root, root.getTotalCost())

    while not fringe.isEmpty():
        currNode = fringe.pop()
        if problem.isGoalState(currNode.state):
            # If the current state is the goal state, return the actions
            # from root to currNode.
            return currNode.getActions()
        if currNode.state not in explored:
            # If the current position is not in explored, it has not been explored,
            # so insert it into explored with initial value of False (not explored)
            explored[currNode.state] = False
        if explored[currNode.state]:
            # If currNode's position has already been explored, we don't explore it again
            continue
        for thing in problem.getSuccessors(currNode.state):
            # Exploring currNode's state
            temp = Node(thing[0], thing[1], thing[2], currNode)
            fringe.push(temp, temp.getTotalCost())
        explored[currNode.state] = True # currNode's position has been explored


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    explored = {}
    startState = problem.getStartState()

    fringe = util.PriorityQueue()
    root = Node(startState, '', 0, None)
    fringe.push(root, root.getTotalCost() + heuristic(root.state, problem))

    while not fringe.isEmpty():
        currNode = fringe.pop()
        if problem.isGoalState(currNode.state):
            # If the current state is the goal state, return the actions
            # from root to currNode.
            return currNode.getActions()
        if currNode.state not in explored:
            # If the current position is not in explored, it has not been explored,
            # so insert it into explored with initial value of False (not explored)
            explored[currNode.state] = False
        if explored[currNode.state]:
            # If currNode's position has already been explored, we don't explore it again
            continue
        for thing in problem.getSuccessors(currNode.state):
            # Exploring currNode's state
            temp = Node(thing[0], thing[1], thing[2], currNode)
            fringe.push(temp, temp.getTotalCost() + heuristic(temp.state, problem))
        explored[currNode.state] = True # currNode's position has been explored



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
