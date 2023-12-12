# Template Provided by: Katelin Lewellen
# EPS TiCS: AIML
# December 2021
#
# This file generates a random 8-square board (solvable)
# and then uses an A* algorithms to solve it.


# importing datetime to get the delta between the start time
# and when the program finishes running
from datetime import datetime
startTime = datetime.now()

# this import brings in the random number generation
# we use this to create random board
# Generate a number between x (inclusive) and y (exclusive) with
# the function random.randrange(x,y)
import random

# this import brings in the ability to different copy functions
# in this case, it allows you to have a deep copy (which works on lists of lists)
# the function is dest = copy.deepcopy(source_array)
# where dest is where you are copying to
# and source_array is where you are copying from
import copy

# this import brings in a priority queue to use for the fringe
# we can construct an empty PQ by saying
# fringe = PriorityQueue()
#
# we can add weighted items to the priority queue with the function
# fringe.put((total_cost, cost, state))
# where weight is the cost plus the heuristic value, cost is the backward cost,
# and state is the state to add
#
# we can get the lowest-weight item out of the fringe by using the function
# next_best = fringe.get()
# which returns a tuple of the form (total_cost, cost, state)
# to get individual elements out, index them as you would an array
# cost = next_best(1)
# current = next_best(2)
from queue import PriorityQueue

# Define the end state: our goal
end_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Defines an empty board for easy copying and filling in.
empty = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


# function prints out a board (state) in a nice format
# 8 5 3
# 2 1 _
# 7 4 6
# with an empty line at the end, for visibility
def pretty_print(board):
    for row in board:
        for space in row:
            if space == 0:
                print("_", end=" ")
            else:
                print(space, end=" ")
        print()
    print()


# takes a 2d list and "flattens" it down to a 1d list
# by adding each element to a new list
# used to allow shallow compares to work well
# I suggest flattening boards before adding them to visited
def flatten_board(board):
    flat = []
    for row in board:
        for space in row:
            flat.append(space)
    return flat

# randomly places the numbers 1 to 8 in unoccupied positions on the board
# by randomly generating a row and column
# if the place is already occupied, generates again.
# the last remaining spot holds a 0 and will be our blank
def randomize_board(board):
    placed = 0
    while placed < 8:
        row = random.randrange(0, 3)
        col = random.randrange(0, 3)
        if board[row][col] == 0:
            placed += 1
            board[row][col] = placed


# not all 8square problems are solvable - we can determine solvability by
# checking the number of inversions in the board
# an inversion is where a number is out of sorted order in the array
# (i.e. greater and earlier than another element)
# if the number of inversions is even, it is solvable.
def is_solvable(board):
    arr = flatten_board(board)
    inv_count = 0
    for index in range(0, 9):
        for cmp_index in range(index + 1, 9):
            if arr[index] != 0 and \
                    arr[cmp_index] != 0 and \
                    arr[index] > arr[cmp_index]:
                inv_count += 1
    return (inv_count % 2 == 0)


# function to generate random boards until one is generated that is solvable
def gen_solvable():
    board = []
    solvable = False
    while not solvable:
        board = copy.deepcopy(empty)
        randomize_board(board)
        solvable = is_solvable(board)
    return board


# a long function used to generate a list of next possible moves after state
# first searches to find the location of the 0 (the blank)
# then manually generates states for each of the 4 sliding directions:
# a state, if the blank moves up, down, right, and left.
def get_next_moves(state):
    row_index = 0  # holds the row of the blank
    col_index = 0  # holds the col of the blank

    # find the blank
    for row in range(0, 3):
        for col in range(0, 3):
            if state[row][col] == 0:
                row_index = row
                col_index = col

    # List to store potential future states.
    next_moves = []

    # for each of the following, find the appropriate next step
    # when applicable and not at a border
    # by swapping the zero (blank) with a neighbor spot in the current state
    # in a copy of the current state

    # move up
    if row_index > 0:
        up_state = copy.deepcopy(state)
        up_state[row_index][col_index] = state[row_index - 1][col_index]
        up_state[row_index - 1][col_index] = 0
        next_moves.append(up_state)

    # move down
    if row_index < 2:
        down_state = copy.deepcopy(state)
        down_state[row_index][col_index] = state[row_index + 1][col_index]
        down_state[row_index + 1][col_index] = 0
        next_moves.append(down_state)

    # move left
    if col_index > 0:
        left_state = copy.deepcopy(state)
        left_state[row_index][col_index] = state[row_index][col_index - 1]
        left_state[row_index][col_index - 1] = 0
        next_moves.append(left_state)

    # move right
    if col_index < 2:
        right_state = copy.deepcopy(state)
        right_state[row_index][col_index] = state[row_index][col_index + 1]
        right_state[row_index][col_index + 1] = 0
        next_moves.append(right_state)

    return next_moves


# a function that takes in a state and returns an integer value indicating
# the estimated distance this state is from the end
# Possible heuristics:
#  - number of items out of place
#  - number of items out of row + number of items out of column
#  - sum of the Manhattan distances of an item to its location
#  - sum of euclidean distances or an item to its location
def heuristic(state):
    # not used
    print(state)
    global end_state
    cost = 0

    for i in range(3):
        for j in range(3):
            if state[i][j] != end_state[i][j]:
                cost += 1

    return cost

# This function calculates the Manhattan distance for the given state.
# The Manhattan distance is the sum of the absolute values of the differences in the goal's x and y coordinates and the current state's x and y coordinates.
def manhattan(state):
    # The goal state is defined globally
    global end_state
    # Initialize the total Manhattan distance to 0
    total_manhattan_distance = 0

    # Iterate over each cell in the state
    for i in range(3):
        for j in range(3):
            # Get the value at the current cell
            value = state[i][j]
            # If the value is not 0 (not the blank space)
            if value != 0:
                # Find the goal position of the current value
                goal_position = [(row, col) for row in range(3) for col in range(3) if end_state[row][col] == value][0]
                # Calculate the Manhattan distance for the current cell
                manhattan_distance = abs(i - goal_position[0]) + abs(j - goal_position[1])
                # Add the Manhattan distance to the total
                total_manhattan_distance += manhattan_distance

    # Return the total Manhattan distance
    return total_manhattan_distance

# a function that returns True if the goal has been met and we are in the
# goal state, and returns False otherwise.
def goal_check(state):
    global end_state

    return state == end_state


# the actual algorithm necessary to perform A* search.
# reminder - should look very similar to your BFS (for example) but with a few
# minor changes:
# - use a priority queue instead of a list to hold the fringe
# - things added to the fringe have a weight
# - that weight is the cost so far + the heuristic value
# - use goal_check, rather than checking == goal
# - instead of looping through all neighbors you want to loop over all the
#     possible next states (where states is generated by get_next_moves)
#               for next_move in states:
# remember that you want to enter tuples weighted by total cost H(x)+cost
# but you also need the current cost so you know the cost of the neighbor node
# which is cost+1
def a_star(start_state):
    pretty_print(start_state)

    # priority queue was explained above at the import
    # creating a set as the visited data struct, as it's effecient for comapring
    # and don't allow duplicates
    fringe = PriorityQueue()
    visited = set()

    
    fringe.put((manhattan(start_state), 0, start_state))

    while not fringe.empty():
        _, cost, current_state = fringe.get()

        if goal_check(current_state):
            print("Goal Reached!")
            return

        # adding the current position to the visited set
        visited.add(tuple(flatten_board(current_state)))

        next_moves = get_next_moves(current_state)

        # iterate through the next possible moves given
        for next_move in next_moves:
            # if the move being analyzed wasn't visited add to the fringe
            # and update visited
            if tuple(flatten_board(next_move)) not in visited:
                fringe.put((manhattan(next_move) + cost + 1, cost + 1, next_move))
                visited.add(tuple(flatten_board(next_move)))

    print("No solution found.")


board = gen_solvable()
a_star(board)
print(datetime.now() - startTime)