import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time
from scipy.interpolate import RegularGridInterpolator, griddata
import json
from scipy.ndimage import distance_transform_edt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from collections import defaultdict

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class WindAwareRRTStar:
    def __init__(self, map_size, obstacle_data_file, wind_data, tke_data=None, UAV_speed=10.0, tke_lim=1.0):
        self.map_width, self.map_height = map_size
        self.UAV_speed = UAV_speed
        self.tke_lim = tke_lim

        # Load obstacle grid
        with open(obstacle_data_file, "r") as f:
            obstacle_data = json.load(f)
        
        self.grid_width = obstacle_data['width']
        self.grid_height = obstacle_data['height']
        self.obstacle_map = np.array(obstacle_data['obstacle_map'], dtype=np.int32)
        self.robot_radius = 0.1  # meters

        if self.map_width != self.grid_width or self.map_height != self.grid_height:
            raise ValueError(f"map_size mismatch: {self.map_width}x{self.map_height} vs JSON {self.grid_width}x{self.grid_height}")

        # Precompute distance map
        self.distance_map = distance_transform_edt(1 - self.obstacle_map)

        self._load_wind(wind_data)
        if tke_data:
            self._load_tke(tke_data)
        else:
            self.tke_interp = None

        self.nodes = []
        self.kd_tree = None
        self.wind_cache = defaultdict(lambda: (0.0, 0.0))
        self.tke_cache = defaultdict(lambda: 0.0)

    def _load_wind(self, csv_file):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        x, y, u, v = data[:, 0] * 1000, data[:, 1] * 1000, data[:, 3], data[:, 4]
        mask = (x >= 0) & (x <= self.map_width) & (y >= 0) & (y <= self.map_height)
        coords = np.column_stack((x[mask], y[mask]))
        u, v = u[mask], v[mask]

        xg, yg = np.arange(0, self.grid_width, 2), np.arange(0, self.grid_height, 2)
        xx, yy = np.meshgrid(xg, yg)
        u_grid = griddata(coords, u, (xx, yy), method='linear', fill_value=0.0)
        v_grid = griddata(coords, v, (xx, yy), method='linear', fill_value=0.0)
        self.u_interp = RegularGridInterpolator((yg, xg), u_grid, method='linear', bounds_error=False, fill_value=0.0)
        self.v_interp = RegularGridInterpolator((yg, xg), v_grid, method='linear', bounds_error=False, fill_value=0.0)

    def _load_tke(self, csv_file):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        x, y, tke = data[:, 0] * 1000, data[:, 1] * 1000, data[:, 3]
        mask = (x >= 0) & (x <= self.map_width) & (y >= 0) & (y <= self.map_height)
        coords = np.column_stack((x[mask], y[mask]))
        tke = tke[mask]

        xg, yg = np.arange(0, self.grid_width, 2), np.arange(0, self.grid_height, 2)
        xx, yy = np.meshgrid(xg, yg)
        tke_grid = griddata(coords, tke, (xx, yy), method='linear', fill_value=0.0)
        self.tke_interp = RegularGridInterpolator((yg, xg), tke_grid, method='linear', bounds_error=False, fill_value=0.0)

    def is_inside_map(self, pos):
        x, y = pos
        return 0 <= x <= self.map_width and 0 <= y <= self.map_height

    def is_inside_obstacle(self, x, y):
        grid_x, grid_y = int(x), int(y)
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return True
        return self.obstacle_map[grid_y, grid_x] == 1

    def get_wind(self, pos):
        x, y = pos
        if self.is_inside_obstacle(x, y):
            return 0.0, 0.0
        key = (int(x), int(y))
        if key in self.wind_cache:
            return self.wind_cache[key]
        point = np.array([y, x])
        u = self.u_interp(point).item()
        v = self.v_interp(point).item()
        self.wind_cache[key] = (u, v)
        return u, v

    def get_tke(self, pos):
        x, y = pos
        if self.tke_interp is None or self.is_inside_obstacle(x, y):
            return 0.0
        key = (int(x), int(y))
        if key in self.tke_cache:
            return self.tke_cache[key]
        point = np.array([y, x])
        tke = self.tke_interp(point).item()
        self.tke_cache[key] = tke
        return tke

    def is_collision(self, pos_a, pos_b=None):
        if pos_b is None:
            return not self.is_inside_map(pos_a) or self.is_inside_obstacle(*pos_a)
        x0, y0 = pos_a
        x1, y1 = pos_b
        for t in np.linspace(0, 1, 5):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if self.is_inside_obstacle(x, y):
                return True
        return False

    def motion_cost(self, a, b, consider_tke=True):
        dx, dy = b.x - a.x, b.y - a.y
        distance = np.hypot(dx, dy)
        if distance == 0:
            return 0.0

        samples, total_cost, Q_star = 5, 0.0, 1.0
        for i in range(samples):
            ratio = (i + 0.5) / samples
            mx, my = a.x + dx * ratio, a.y + dy * ratio
            u, v = self.get_wind((mx, my))
            tke = self.get_tke((mx, my)) if consider_tke and self.tke_interp else 0.0
            U_tke = 10.0 if tke > self.tke_lim else 0.9

            wind_dot = (u * dx + v * dy) / distance
            w = self.UAV_speed / max(self.UAV_speed + wind_dot, 1e-6)
            grid_x, grid_y = int(mx), int(my)
            U_rep = Q_star - self.distance_map[grid_y, grid_x] + 1 if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height and self.distance_map[grid_y, grid_x] <= Q_star else 1
            total_cost += U_rep * w * (distance / samples) * U_tke
        return total_cost

    def find_nearest(self, new_node):
        if not self.nodes:
            return None
        if self.kd_tree is None or len(self.nodes) > self.kd_tree.n:
            points = [(n.x, n.y) for n in self.nodes]
            self.kd_tree = KDTree(points)
        _, idx = self.kd_tree.query([(new_node.x, new_node.y)])
        return self.nodes[idx[0]]

    def find_nearby(self, new_node, radius):
        if not self.nodes or self.kd_tree is None:
            return []
        points = [(n.x, n.y) for n in self.nodes]
        self.kd_tree = KDTree(points)
        indices = self.kd_tree.query_ball_point([(new_node.x, new_node.y)], radius)[0]
        return [self.nodes[i] for i in indices if self.nodes[i] is not new_node]

    def animate_path_building_no_wind(self, start, goal, max_iter=1000, step_size=5.0, goal_radius=5.0, save_gif=True):
        # Initialize figure and axis with higher resolution
        fig, ax = plt.subplots(figsize=(6, 4), dpi=250)
        ax.imshow(self.obstacle_map, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], interpolation='nearest', alpha=0.5)

        # Plot start and goal
        ax.scatter([start[0]], [start[1]], c='green', s=50, marker='o', label='Start')
        ax.scatter([goal[0]], [goal[1]], c='red', s=50, marker='X', label='Goal')
        ax.add_patch(patches.Circle(goal, goal_radius, fill=False, color='red', linestyle='--', label='Goal Radius'))

        ax.set_title("RRT* Path Building with Rewiring (No Wind)")
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        ax.legend()

        # Initialize plot elements
        tree_lines, = ax.plot([], [], 'b-', linewidth=0.5, alpha=0.5, label='RRT* Tree')
        path_lines, = ax.plot([], [], 'b--', linewidth=1.5, label='Current Path')
        rewire_lines, = ax.plot([], [], 'r-', linewidth=1.0, alpha=0.7, label='Rewiring Candidates')
        rewire_accepted, = ax.plot([], [], 'g-', linewidth=1.0, alpha=0.7, label='Rewired Edge')

        # Initialize planning variables
        self.nodes = [Node(*start)]
        self.kd_tree = None
        goal_node = Node(*goal)
        path_found = False
        iter_count = 0
        collision_count = 0
        tree_segments = []
        rewire_frames = []

        def update(frame):
            nonlocal iter_count, path_found, tree_segments, rewire_frames, collision_count
            if iter_count >= max_iter or (path_found and not rewire_frames):
                return tree_lines, path_lines, rewire_lines, rewire_accepted

            # Process rewiring frames if any
            if rewire_frames:
                frame_type, data = rewire_frames.pop(0)
                if frame_type == 'candidate':
                    seg_x, seg_y = data
                    rewire_lines.set_data(seg_x, seg_y)
                    rewire_accepted.set_data([], [])
                elif frame_type == 'accepted':
                    seg_x, seg_y = data
                    rewire_accepted.set_data(seg_x, seg_y)
                    rewire_lines.set_data([], [])
                ax.set_title(f"RRT* Rewiring (No Wind) - Iteration {iter_count}")
                return tree_lines, path_lines, rewire_lines, rewire_accepted

            iter_count += 1
            # Sample a point
            sample = goal_node if np.random.rand() < 0.2 and not path_found else Node(np.random.uniform(0, self.map_width), np.random.uniform(0, self.map_height))
            nearest = self.find_nearest(sample)
            if not nearest:
                print(f"Iteration {iter_count}: No nearest node found")
                return tree_lines, path_lines, rewire_lines, rewire_accepted

            # Generate new node
            theta = np.arctan2(sample.y - nearest.y, sample.x - nearest.x)
            step = min(step_size, np.hypot(sample.x - nearest.x, sample.y - nearest.y))
            new_node = Node(nearest.x + step * np.cos(theta), nearest.y + step * np.sin(theta))

            # Check node validity
            if not self.is_inside_map((new_node.x, new_node.y)):
                print(f"Iteration {iter_count}: New node out of bounds ({new_node.x}, {new_node.y})")
                return tree_lines, path_lines, rewire_lines, rewire_accepted

            # Check collision
            if self.is_collision((nearest.x, nearest.y), (new_node.x, new_node.y)):
                collision_count += 1
                if collision_count % 10 == 0:  # Print every 10th collision
                    print(f"Iteration {iter_count}: Collision detected (Total: {collision_count})")
                return tree_lines, path_lines, rewire_lines, rewire_accepted

            # Compute cost (no wind, no TKE)
            cost_fn = lambda a, b: np.hypot(b.x - a.x, b.y - a.y)
            new_node.cost = nearest.cost + cost_fn(nearest, new_node)
            new_node.parent = nearest
            self.nodes.append(new_node)
            tree_segments.append(([nearest.x, new_node.x], [nearest.y, new_node.y]))

            # Rewiring step
            rewire_radius = step_size * 1.5
            nearby_nodes = self.find_nearby(new_node, rewire_radius)
            for node in nearby_nodes:
                if node is new_node or node is nearest:
                    continue
                new_cost = new_node.cost + cost_fn(new_node, node)
                if new_cost < node.cost and not self.is_collision((new_node.x, new_node.y), (node.x, node.y)):
                    rewire_frames.append(('candidate', ([new_node.x, node.x], [new_node.y, node.y])))
                    rewire_frames.append(('accepted', ([new_node.x, node.x], [new_node.y, node.y])))
                    node.parent = new_node
                    node.cost = new_cost
                    for i, (seg_x, seg_y) in enumerate(tree_segments):
                        if np.isclose(seg_x[1], node.x) and np.isclose(seg_y[1], node.y):
                            tree_segments[i] = ([new_node.x, node.x], [new_node.y, node.y])
                            break

            # Check if goal is reached
            path = []
            goal_distance = np.hypot(new_node.x - goal[0], new_node.y - goal[1])
            if goal_distance < goal_radius:
                path_found = True
                node = self.nodes[-1]
                while node:
                    path.append((node.x, node.y))
                    node = node.parent
                path.reverse()
                print(f"Iteration {iter_count}: Path found with {len(path)} points, Distance to goal: {goal_distance:.2f}")

            # Update tree plot
            tree_lines.set_data([], [])
            for seg_x, seg_y in tree_segments:
                ax.plot(seg_x, seg_y, 'b-', linewidth=0.5, alpha=0.5)

            # Update path plot
            if path_found and path:
                path_array = np.array(path)
                path_lines.set_data(path_array[:, 0], path_array[:, 1])
            else:
                path_lines.set_data([], [])

            rewire_lines.set_data([], [])
            rewire_accepted.set_data([], [])
            ax.set_title(f"RRT* Path Building (No Wind) - Iteration {iter_count}")
            return tree_lines, path_lines, rewire_lines, rewire_accepted

        # Create animation
        total_frames = max_iter * 3
        anim = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False, repeat=False)

        if save_gif:
            anim.save('rrtstar_path_building_no_wind_rewiring.gif', writer='pillow', fps=8, dpi=150)
            print("Animation saved as rrtstar_path_building_no_wind_rewiring.gif")
            # Save final frame as high-DPI PNG for presentation
            plt.savefig('rrtstar_final_frame.png', dpi=300, bbox_inches='tight')
            print("Final frame saved as rrtstar_final_frame.png")
        else:
            plt.show()

        # Debug: Verify final tree and path
        print(f"Final tree size: {len(self.nodes)} nodes")
        print(f"Total collisions: {collision_count} ({collision_count/iter_count*100:.2f}% of iterations)")
        if path_found:
            print(f"Final path length: {len(path)} points")
            for i, (x, y) in enumerate(path):
                if self.is_collision((x, y)):
                    print(f"Warning: Path point {i} ({x}, {y}) is in collision")
        else:
            print("No path found")
            path = []

        return {"path": path, "iterations": iter_count, "path_found": path_found}

if __name__ == "__main__":
    planner = WindAwareRRTStar(
        map_size=(1300, 1000),
        obstacle_data_file="input_data/flipped_obstacle_map.json",
        wind_data="input_data/wind_50m.csv",
        tke_data="input_data/tke.csv",
        UAV_speed=8.0,
        tke_lim=0.7
    )
    start = (580, 600)
    goal = (700, 580)
    print("Animating path building with rewiring visualization (No Wind, 1000 iterations)")
    result = planner.animate_path_building_no_wind(
        start=start,
        goal=goal,
        max_iter=1000,
        step_size=5.0,
        goal_radius=5.0,
        save_gif=False
    )
    print(f"Animation completed. Path found: {result['path_found']}, Iterations: {result['iterations']}")