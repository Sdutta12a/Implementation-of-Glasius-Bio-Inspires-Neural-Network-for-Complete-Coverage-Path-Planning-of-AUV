


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq

# INITIALIZE GRID MAP WITH CLUSTERED OBSTACLES
def initialize_grid_map(rows, cols, obstacle_density):
    grid_map = np.zeros((rows, cols), dtype=int)
    num_obstacles = int(rows * cols * obstacle_density)

    cluster_size = 8  # Number of obstacles in each cluster
    clusters = num_obstacles // cluster_size

    for _ in range(clusters):
        # Choose a random center for the cluster
        center_x = np.random.randint(rows)
        center_y = np.random.randint(cols)

        for _ in range(cluster_size):
            # Add obstacles around the center with some random offset
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            x = min(max(center_x + offset_x, 0), rows - 1)
            y = min(max(center_y + offset_y, 0), cols - 1)
            grid_map[x, y] = 1

    return grid_map

# INITIALIZE NEURAL NETWORK
def initialize_neural_network(grid_map):
    rows, cols = grid_map.shape
    neural_network = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            if grid_map[i, j] == 1:
                neural_network[i, j] = -1  # Obstacles
    return neural_network

# UPDATION OF NEURAL ACTIVITY
def update_neural_activity(neural_network, alpha, r, beta, decay_factor, path):
    rows, cols = neural_network.shape
    update_neural_network = np.copy(neural_network)
    for i, j in path:
        if neural_network[i, j] > 0:
            update_neural_network[i, j] = max(0, neural_network[i, j] - decay_factor)  # Decay neural activity
        elif neural_network[i, j] == 0:
            update_neural_network[i, j] = 1  # Set initial neural activity to 1
    return update_neural_network

# CALCULATION OF CONNECTION WEIGHT
def calculate_connection_weight(pos1, pos2, alpha):
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    weight = np.exp(-alpha * distance**2)
    return weight

# TRANSFER FUNCTION
def transfer_function(input, beta):
    if input < 0:
        output = -1
    elif 0 <= input < 1:
        output = beta * input
    else:
        output = 1
    return output

# LINE OF SIGHT CHECK FOR THETA*
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

# THETA* PATHFINDING ALGORITHM
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

# HEURISTIC FUNCTION (EUCLIDEAN DISTANCE)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# GET NEIGHBORS (INCLUDING DIAGONALS)
def get_neighbors(grid_map, cell):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < grid_map.shape[0] and 0 <= neighbor[1] < grid_map.shape[1] and grid_map[neighbor[0], neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors

# RECONSTRUCT PATH FROM THETA*
def reconstruct_path(came_from, current):
    path = [current]
    while current != came_from[current]:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# BOUSTROPHEDON PATH PLANNING WITH DYNAMIC OBSTACLE AVOIDANCE USING THETA*
def boustrophedon_path_planning(grid_map, start, goal):
    rows, cols = grid_map.shape
    path = []
    direction = 1  # 1 means top-to-bottom, -1 means bottom-to-top
    current_position = start

    for j in range(cols):
        if direction == 1:
            for i in range(rows):
                if grid_map[i, j] == 0 and (i, j) not in path:
                    sub_path = theta_star(grid_map, current_position, (i, j))
                    if sub_path:
                        path.extend(sub_path[1:])
                        current_position = (i, j)
        else:
            for i in range(rows-1, -1, -1):
                if grid_map[i, j] == 0 and (i, j) not in path:
                    sub_path = theta_star(grid_map, current_position, (i, j))
                    if sub_path:
                        path.extend(sub_path[1:])
                        current_position = (i, j)
        direction *= -1

    # Final path to goal
    if current_position != goal:
        sub_path = theta_star(grid_map, current_position, goal)
        if sub_path:
            path.extend(sub_path[1:])

    return path

# VISUALIZE PATH
def visualize_path(grid_map, path):
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size for better visibility
    ax.imshow(grid_map, cmap='Greys', origin='upper')

    # Plot the path
    if path:
        path = np.array(path)
        rgb_color = (0/255, 97/255, 176/255)  
        ax.plot(path[:, 1], path[:, 0], color=rgb_color)
    
    # Set up the grid
    ax.set_xticks(np.arange(-0.5, grid_map.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Major ticks at each cell
    ax.set_xticks(np.arange(0, grid_map.shape[1], 1))
    ax.set_yticks(np.arange(0, grid_map.shape[0], 1))
    
    # Labeling the axes
    ax.set_xticklabels(np.arange(1, grid_map.shape[1] + 1, 1))
    ax.set_yticklabels(np.arange(1, grid_map.shape[0] + 1, 1))

    # Invert the y-axis to have the origin at the top left
    ax.invert_yaxis()

    # Add legend
    legend_elements = [plt.Line2D([0], [0], color='w', marker='o', markersize=12, markeredgecolor='black', lw=0, label='Uncovered'),
                       plt.Line2D([0], [0], color='k', lw=4, label='Obstacle'),
                       plt.Line2D([0], [0], color=rgb_color, lw=4, label='Path')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

    # Mark the grid in intervals of 5
    ax.set_xticks(np.arange(0, grid_map.shape[1], 5), minor=False)
    ax.set_yticks(np.arange(0, grid_map.shape[0], 5), minor=False)

    plt.show()

# VISUALIZE NEURAL ACTIVITY
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

# MAIN FUNCTION TO EXECUTE THE PROCESS
def main():
    rows, cols = 30, 30
    start = (0, 0)
    goal = (29, 29)
    obstacle_density = 0.2

    grid_map = initialize_grid_map(rows, cols, obstacle_density)
    neural_network = initialize_neural_network(grid_map)

    path = boustrophedon_path_planning(grid_map, start, goal)
    
    # Update neural activity iteratively with the path
    alpha = 2
    r = 2
    beta = 0.5
    decay_factor = 0.01
    iterations = 100
    for _ in range(iterations):
        neural_network = update_neural_activity(neural_network, alpha, r, beta, decay_factor, path)

    visualize_path(grid_map, path)
    visualize_neural_activity(neural_network)

if __name__ == "__main__":
    main()
