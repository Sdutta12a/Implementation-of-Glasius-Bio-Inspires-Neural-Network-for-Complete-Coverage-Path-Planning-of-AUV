import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import matplotlib.animation as animation


# Initialize grid map with specific obstacle blocks
def initialize_grid_map(rows, cols):
    grid_map = np.zeros((rows, cols), dtype=int)
    
    # Define obstacles
    # obstacles = [(0,15,10,5),(0,8,5,5),(5,20,5,5),(0,10,5,5),
    #             (10,20,10,5),(20,20,5,5),(25,15,5,5),
    #             (15,0,10,10),(20,5,3,5), (28,20,2,3)
    
    # ]
    obstacles = [(0,15,10,5),(0,8,5,5),(5,20,5,5),(0,10,5,5),
                (10,20,10,5),(20,20,5,5),(25,15,5,5),
                (15,0,10,10),(20,5,3,5), (28,20,2,3)]
    
    for (x, y, width, height) in obstacles:
        grid_map[x:x+width, y:y+height] = 1
    
    # Ensure start and goal positions are free
    grid_map[0, 0] = 0
    grid_map[rows - 1, cols - 1] = 0
    
    return grid_map

# Initialize neural network
def initialize_neural_network(grid_map):
    rows, cols = grid_map.shape
    neural_network = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if grid_map[i, j] == 0:
                neural_network[i, j] = -1
            else:
                neural_network[i, j] = 1
    return neural_network

# Updation of neural activity
def update_neural_activity(neural_network, alpha, r, beta):
    rows, cols = neural_network.shape
    update_neural_network = np.copy(neural_network)
    for i in range(rows):
        for j in range(cols):
            sum_weighted_inputs = 0
            for x in range(max(0, i-r), min(rows, i+r+1)):
                for y in range(max(0, j-r), min(cols, j+r+1)):
                    if x != i or y != j:
                        weight = calculate_connection_weight((i, j), (x, y), alpha)
                        sum_weighted_inputs += weight * neural_network[x, y]
            update_neural_network[i, j] = transfer_function(sum_weighted_inputs, beta)
    return update_neural_network

# Calculation of connection weight
def calculate_connection_weight(pos1, pos2, alpha):
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    weight = np.exp(-alpha * distance**2)
    return weight

# Transfer function
def transfer_function(input, beta):
    if input < 0:
        output = -1
    elif 0 <= input < 1:
        output = beta * input
    else:
        output = 1
    return output

# Visualize neural activity
def visualize_neural_activity(neural_network):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rows, cols = neural_network.shape
    x = np.arange(0, rows, 1)
    y = np.arange(0, cols, 1)
    x, y = np.meshgrid(x, y)
    z = neural_network.T  # Transpose to match the shape
    
    # Plot surface
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Neural Activity')
    plt.show()



# Line of Sight Check for Theta*
def line_of_sight(grid_map, start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while (x0, y0) != (x1, y1):
        if grid_map[x0, y0] == 1:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

# Theta* Pathfinding Algorithm
def theta_star(grid_map, start, goal):
    rows, cols = grid_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: start}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(grid_map, current):
            if line_of_sight(grid_map, came_from[current], neighbor):
                tentative_g_score = g_score[came_from[current]] + heuristic(came_from[current], neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = came_from[current]
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
            else:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

# Heuristic function (Euclidean Distance)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Get neighbors (including diagonals)
def get_neighbors(grid_map, cell):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < grid_map.shape[0] and 0 <= neighbor[1] < grid_map.shape[1] and grid_map[neighbor[0], neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors

# Reconstruct path from Theta*
def reconstruct_path(came_from, current):
    path = [current]
    while current != came_from[current]:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Path planning with lawn mower pattern and Theta* for obstacle avoidance
# def plan_path(grid_map):
#     rows, cols = grid_map.shape
#     path = []
#     direction = 1  # 1 for down, -1 for up
#     col = 0
#     last_col = False  # Track if we are processing the last column
#     while col < cols:
#         for row in range(rows)[::direction]:
#             if grid_map[row, col] == 0:  # Check for free space
#                 path.append((row, col))
#             else:
#                 # If an obstacle is found, use Theta* to navigate around it
#                 start = (row, col)
#                 goal = find_next_free_space(grid_map, row, col, direction)
#                 if goal:
#                     sub_path = theta_star(grid_map, start, goal)
#                     if sub_path:
#                         path.extend(sub_path)
#                         row, col = goal  # Move to the goal after finding a path
#                         break
#         direction *= -1  # Change direction for the next column
#         col += 1
#         if col == cols:
#             last_col = True

#     # Ensure the path covers the entire grid consistently, including the last row and column
#     if last_col:
#         final_path = []
#         for idx in range(len(path) - 1):
#             final_path.append(path[idx])
#             if path[idx][1] != path[idx + 1][1]:  # Change in column
#                 # Fill in the rows between the changes in columns to maintain consistency
#                 start_row, end_row = path[idx][0], path[idx + 1][0]
#                 step = 1 if start_row < end_row else -1
#                 for r in range(start_row, end_row, step):
#                     final_path.append((r, path[idx][1]))
#         final_path.append(path[-1])
#         return final_path
#     else:
#         return path
def plan_path(grid_map):
    rows, cols = grid_map.shape
    path = []
    direction = 1  # 1 for down, -1 for up
    col = 0
    
    
    while col < cols:
        # Determine start and end points for the current column
        if direction == 1:
            start_row, end_row = 0, rows
        else:
            start_row, end_row = rows - 1, -1

        for row in range(start_row, end_row, direction):
            if grid_map[row, col] == 0:  # Check for free space
                if not path:
                    path.append((row, col))  # Start path
                else:
                    start = path[-1]
                    goal = (row, col)
                    sub_path = theta_star(grid_map, start, goal)
                    if sub_path:
                        path.extend(sub_path[1:])  # Append sub-path except the start
            else:
                continue  # Skip obstacles directly

        direction *= -1  # Change direction for the next column
        col += 1
    
    return path

# Find next free space in the direction
def find_next_free_space(grid_map, row, col, direction):
    rows, cols = grid_map.shape
    for r in range(row, rows, direction):
        if grid_map[r, col] == 0:
            return (r, col)
    return None

# Visualization with Animation
def visualize_path(grid_map, path):
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size for better visibility
    ax.imshow(grid_map, cmap='Greys', origin='upper')
    
    # Plot the path
    if path:
        path = np.array(path)
        rgb_color = (0/255, 97/255, 176/255)
        ax.plot(path[:, 1], path[:, 0], color= rgb_color)  # Change color to blue and remove dots
    
    # Set up the grid
    ax.set_xticks(np.arange(-0.5, grid_map.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Major ticks at each cell
    ax.set_xticks(np.arange(0, grid_map.shape[1], 1))
    ax.set_yticks(np.arange(0, grid_map.shape[0], 1))
    
    # Labeling the ticks at specific intervals
    major_ticks = np.arange(0, grid_map.shape[0] + 1, 5)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    
    ax.set_xticklabels(major_ticks)
    ax.set_yticklabels(major_ticks)
    
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

    # Invert the y-axis to have the origin at the top left
    ax.invert_yaxis()
    legend_elements = [plt.Line2D([0], [0], color='w',marker='o', markersize=12, markeredgecolor='black',  lw=0, label='Uncovered'),
                    plt.Line2D([0], [0], color='k', lw=4, label='Obstacle'),
                    plt.Line2D([0], [0], color=rgb_color, lw=4, label='Path')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
    # Create a square to represent the robot
    square_size = 0.5
    square = patches.Rectangle((path[0][1] - square_size/2, path[0][0] - square_size/2), square_size, square_size, edgecolor='cyan', facecolor='green')
    ax.add_patch(square)

    # Function to update the square's position
    def update(frame):
        square.set_xy((path[frame][1] - square_size/2, path[frame][0] - square_size/2))
        return square,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=200, blit=True)
    plt.show()

# Main Simulation Loop
rows = 30
cols = 30
num_iterations = 100
alpha = 2
R = 2
beta = 0.7


# Initialize grid map and neural network
grid_map = initialize_grid_map(rows, cols)
neural_network = initialize_neural_network(grid_map)



for iteration in range(num_iterations):
    neural_network = update_neural_activity(neural_network, alpha, R, beta)
    path = plan_path(grid_map)
    

    # Display the path
    print(f'Iteration: {iteration + 1}')
    print('Path:')
    print(path)

    # Visualization (optional)
    visualize_path(grid_map, path)
    # Visualize neural activity
    visualize_neural_activity(neural_network)








