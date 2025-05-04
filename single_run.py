import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time
from scipy.interpolate import RegularGridInterpolator, griddata
import json
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline
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

        xg, yg = np.arange(0, self.grid_width, 2), np.arange(0, self.grid_height, 2)  # Coarser grid
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

        xg, yg = np.arange(0, self.grid_width, 2), np.arange(0, self.grid_height, 2)  # Coarser grid
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
        for t in np.linspace(0, 1, 10):  # Reduced to 10 samples
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if self.is_inside_obstacle(x, y):
                return True
        return False

    def motion_cost(self, a, b, consider_tke=True):
        dx, dy = b.x - a.x, b.y - a.y
        distance = np.hypot(dx, dy)
        if distance == 0:
            return 0.0

        samples, total_cost, Q_star = 10, 0.0, 1.0  # Reduced to 10 samples
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

    def plan(self, start, goal, max_iter=20000, step_size=5.0, goal_radius=5.0, consider_wind=True, consider_tke=True):
        start_time = time.time()
        if self.is_collision(start) or self.is_collision(goal):
            print(f"Invalid start {start} or goal {goal}")
            return {"path": [], "length": 0.0, "cost": float('inf'), "time": 0.0}

        self.nodes = [Node(*start)]
        self.kd_tree = None
        goal_node = Node(*goal)
        path_found = False
        iter_count = 0

        for _ in range(max_iter):
            iter_count += 1
            sample = goal_node if np.random.rand() < 0.2 and not path_found else Node(np.random.uniform(0, self.map_width), np.random.uniform(0, self.map_height))
            nearest = self.find_nearest(sample)
            if not nearest:
                continue

            theta = np.arctan2(sample.y - nearest.y, sample.x - nearest.x)
            step = min(step_size, np.hypot(sample.x - nearest.x, sample.y - nearest.y))
            new_node = Node(nearest.x + step * np.cos(theta), nearest.y + step * np.sin(theta))

            if self.is_collision((nearest.x, nearest.y), (new_node.x, new_node.y)):
                continue

            cost_fn = lambda a, b: self.motion_cost(a, b, consider_tke) if consider_wind else np.hypot(b.x - a.x, b.y - a.y)
            new_node.cost = nearest.cost + cost_fn(nearest, new_node)
            new_node.parent = nearest
            self.nodes.append(new_node)

            if np.hypot(new_node.x - goal[0], new_node.y - goal[1]) < goal_radius:
                print(f"Path found after {iter_count} iterations")
                path_found = True
                break

        if not path_found:
            print(f"No path found after {iter_count} iterations")

        path = []
        if path_found:
            node = self.nodes[-1]
            while node:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()

        elapsed_time = time.time() - start_time
        path_length = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))) if path else 0.0
        print(f"Path length: {path_length:.2f} m, Cost: {new_node.cost if path_found else float('inf'):.2f}, Time: {elapsed_time:.2f} s")
        return {"path": path, "length": path_length, "cost": new_node.cost if path_found else float('inf'), "time": elapsed_time}

    def plot_wind_with_paths(self, path_with_tke, path_with_wind, path_without_wind):
        plt.figure(1)
        ax = plt.gca()
        ax.imshow(self.obstacle_map, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], alpha=0.5)
        plot_res = 40.0  # Coarser grid for efficiency
        x_plot = np.arange(0, self.map_width, plot_res)
        y_plot = np.arange(0, self.map_height, plot_res)
        X, Y = np.meshgrid(x_plot, y_plot)
        points = np.column_stack((Y.ravel(), X.ravel()))
        U = self.u_interp(points).reshape(Y.shape)
        V = self.v_interp(points).reshape(Y.shape)
        magnitude = np.sqrt(U**2 + V**2)
        q = ax.quiver(X, Y, U, V, magnitude, scale=80, cmap='viridis')
        plt.colorbar(q, ax=ax, label='Wind Speed (m/s)')
        self._plot_paths(ax, path_with_tke, path_with_wind, path_without_wind)
        ax.set_title("Wind Vectors with Paths")
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        plt.savefig('wind_vectors.png')

    def plot_tke_with_paths(self, path_with_tke, path_with_wind, path_without_wind):
        plt.figure(2)
        ax = plt.gca()
        ax.imshow(self.obstacle_map, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], alpha=0.5)
        if self.tke_interp is not None:
            tke_res = 10.0  # Coarser grid for efficiency
            xg = np.linspace(0, self.map_width, int(self.map_width / tke_res) + 1)
            yg = np.linspace(0, self.map_height, int(self.map_height / tke_res) + 1)
            xx, yy = np.meshgrid(xg, yg)
            points = np.column_stack((yy.ravel(), xx.ravel()))
            tke_values = self.tke_interp(points).reshape(yy.shape)
            cf = ax.contourf(xx, yy, tke_values, levels=10, cmap='viridis', alpha=0.8)
            plt.colorbar(cf, ax=ax, label='Turbulence Kinetic Energy (m²/s²)')
        self._plot_paths(ax, path_with_tke, path_with_wind, path_without_wind)
        ax.set_title("TKE Contour with Paths")
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        plt.savefig('tke_contour.png')

    def _plot_paths(self, ax, path_with_tke, path_with_wind, path_without_wind):
        paths = [
            (path_with_tke, 'green', 'With Wind & TKE'),
            (path_with_wind, 'red', 'With Wind'),
            (path_without_wind, 'blue', 'No Wind')
        ]
        for path, color, label in paths:
            if path and len(path) > 0:
                path_array = np.array(path)
                ds = np.cumsum(np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1)))
                ds = np.insert(ds, 0, 0)
                csx = CubicSpline(ds, path_array[:,0])
                csy = CubicSpline(ds, path_array[:,1])
                ss = np.linspace(0, ds[-1], 200)
                smooth_x = csx(ss)
                smooth_y = csy(ss)
                ax.plot(smooth_x, smooth_y, '--', linewidth=2, color=color, label=label)
                ax.scatter([smooth_x[0]], [smooth_y[0]], c=color, s=80, marker='o', alpha=0.7)
                ax.scatter([smooth_x[-1]], [smooth_y[-1]], c=color, s=80, marker='X', alpha=0.7)
        ax.legend()

if __name__ == "__main__":
    planner = WindAwareRRTStar(
        map_size=(1300, 1000),
        obstacle_data_file="input_data/flipped_obstacle_map.json",
        wind_data="input_data/wind_50m.csv",
        tke_data="input_data/tke.csv",
        UAV_speed=8.0,
        tke_lim=0.7
    )
    start = (400, 600)
    goal = (800, 800)
    print("Planning with wind and TKE")
    result_with_tke = planner.plan(start, goal, consider_wind=True, consider_tke=True)
    path_with_tke = result_with_tke["path"]
    print("Planning with wind only")
    result_with_wind = planner.plan(start, goal, consider_wind=True, consider_tke=False)
    path_with_wind = result_with_wind["path"]
    print("Planning without wind")
    result_without_wind = planner.plan(start, goal, consider_wind=False, consider_tke=False)
    path_without_wind = result_without_wind["path"]
    planner.plot_wind_with_paths(path_with_tke, path_with_wind, path_without_wind)
    planner.plot_tke_with_paths(path_with_tke, path_with_wind, path_without_wind)
    plt.show()