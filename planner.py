import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import time
from scipy.interpolate import RegularGridInterpolator, griddata
import json
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, pearsonr
# import seaborn as sns
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class WindAwareRRTStar:
    def __init__(self, map_size, obstacle_data_file, wind_data, tke_data = None, UAV_speed=10.0,
                 tke_lim=1.0):
        self.map_width, self.map_height = map_size
        self.UAV_speed = UAV_speed
        self.tke_lim = tke_lim

        # Load obstacle grid
        with open(obstacle_data_file, "r") as f:
            obstacle_data = json.load(f)
        
        self.grid_width = obstacle_data['width']
        self.grid_height = obstacle_data['height']
        self.obstacle_map = obstacle_data['obstacle_map']
        
        if not isinstance(self.obstacle_map, list) or not all(isinstance(row, list) for row in self.obstacle_map):
            raise ValueError("obstacle_map must be a 2D list")
        if len(self.obstacle_map) != self.grid_height or any(len(row) != self.grid_width for row in self.obstacle_map):
            raise ValueError(f"obstacle_map dimensions ({len(self.obstacle_map)}x{len(self.obstacle_map[0])}) do not match height ({self.grid_height}) or width ({self.grid_width})")
        
        for i, row in enumerate(self.obstacle_map):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)) or val not in [0, 1]:
                    raise ValueError(f"Invalid value '{val}' at position [{i},{j}] in obstacle_map. Expected 0 or 1.")
        
        self.obstacle_map = np.array(self.obstacle_map, dtype=np.int32)
        self.robot_radius = 0.1  # meters

        if self.map_width != self.grid_width or self.map_height != self.grid_height:
            raise ValueError(f"map_size ({self.map_width}x{self.map_height}) does not match JSON dimensions ({self.grid_width}x{self.grid_height})")

        # Precompute distance map for repulsive potential
        self.distance_map = distance_transform_edt(1 - self.obstacle_map)

        self._load_wind(wind_data)
        if tke_data is not None:
            self._load_tke(tke_data)
        else:
            self.tke_interp = None  # No TKE data provided

        self.nodes = []
        self.kd_tree = None
        self.kd_tree_node_count = 0

    def _load_wind(self, csv_file):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        if data.shape[1] < 5:
            raise ValueError("Wind CSV needs ≥5 cols: x,y,_,u,v")

        x = data[:,0] * 1000
        y = data[:,1] * 1000
        u = data[:,3]
        v = data[:,4]
        
        mask = (x >= 0) & (x <= self.map_width) & (y >= 0) & (y <= self.map_height)
        x = x[mask]
        y = y[mask]
        u = u[mask]
        v = v[mask]
        coords = np.column_stack((x, y))
        
        print(f"Filtered wind data to {len(coords)} points within map bounds")

        if len(coords) < 3:
            raise ValueError("Too few wind data points after filtering for interpolation")

        xg = np.arange(0, self.grid_width)
        yg = np.arange(0, self.grid_height)
        xx, yy = np.meshgrid(xg, yg)

        u_grid_linear = griddata(coords, u, (xx, yy), method='linear')
        v_grid_linear = griddata(coords, v, (xx, yy), method='linear')
        u_grid_nearest = griddata(coords, u, (xx, yy), method='nearest')
        v_grid_nearest = griddata(coords, v, (xx, yy), method='nearest')
        
        u_grid = np.where(np.isnan(u_grid_linear), u_grid_nearest, u_grid_linear)
        v_grid = np.where(np.isnan(v_grid_linear), v_grid_nearest, v_grid_linear)

        self.u_interp = RegularGridInterpolator((yg, xg), u_grid, method='linear', bounds_error=False, fill_value=0.0)
        self.v_interp = RegularGridInterpolator((yg, xg), v_grid, method='linear', bounds_error=False, fill_value=0.0)
    
    def _load_tke(self, csv_file):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        if data.shape[1] < 3:
            raise ValueError("TKE CSV needs ≥3 cols: x,y,tke")

        x = data[:, 0] * 1000  # Convert km to meters if needed
        y = data[:, 1] * 1000
        tke = data[:, 3]  # TKE in m²/s²

        # Filter points within map bounds
        mask = (x >= 0) & (x <= self.map_width) & (y >= 0) & (y <= self.map_height)
        x = x[mask]
        y = y[mask]
        tke = tke[mask]
        coords = np.column_stack((x, y))

        print(f"Filtered TKE data to {len(coords)} points within map bounds")

        if len(coords) < 3:
            raise ValueError("Too few TKE data points after filtering for interpolation")

        # Interpolate TKE onto grid
        xg = np.arange(0, self.grid_width)
        yg = np.arange(0, self.grid_height)
        xx, yy = np.meshgrid(xg, yg)

        tke_grid_linear = griddata(coords, tke, (xx, yy), method='linear')
        tke_grid_nearest = griddata(coords, tke, (xx, yy), method='nearest')
        tke_grid = np.where(np.isnan(tke_grid_linear), tke_grid_nearest, tke_grid_linear)

        self.tke_interp = RegularGridInterpolator(
            (yg, xg), tke_grid, method='linear', bounds_error=False, fill_value=0.0
        )
        
    def is_inside_map(self, pos):
        x, y = pos
        return 0 <= x <= self.map_width and 0 <= y <= self.map_height

    def is_inside_obstacle(self, x, y):
        grid_x = int(x)
        grid_y = int(y)
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return True
        return self.obstacle_map[grid_y, grid_x] == 1

    def get_wind(self, pos):
        x, y = pos
        if self.is_inside_obstacle(x, y):
            return 0.0, 0.0
        point = np.array([y, x])
        try:
            u = self.u_interp(point)[0]
            v = self.v_interp(point)[0]
        except Exception:
            u, v = 0.0, 0.0
        return float(u), float(v)
    
    def get_tke(self, pos):
        x, y = pos
        if self.is_inside_obstacle(x, y) or self.tke_interp is None:
            return 0.0
        point = np.array([y, x])
        try:
            tke = self.tke_interp(point)[0]
        except Exception:
            tke = 0.0
        return float(tke)

    def is_collision(self, pos_a, pos_b=None):
        if pos_b is None:
            if not self.is_inside_map(pos_a):
                return True
            x, y = pos_a
            return self.is_inside_obstacle(x, y)
        else:
            if not self.is_inside_map(pos_a) or not self.is_inside_map(pos_b):
                return True
            samples = 10
            x0, y0 = pos_a
            x1, y1 = pos_b
            for i in range(samples + 1):
                t = i / samples
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                if self.is_inside_obstacle(x, y):
                    return True
            return False

    def motion_cost(self, a, b, prev_dx, prev_dy, tke_lim = 0.7):
        dx = b.x - a.x
        dy = b.y - a.y
        distance = np.hypot(dx, dy)

        if distance == 0:
            return 0.0

        # #Path angle penalty (commented out as in your code)
        # if prev_dx is not None and prev_dy is not None:
        #     theta_prev = np.arctan2(prev_dy, prev_dx)
        #     theta_next = np.arctan2(dy, dx)
        #     delta_theta = np.abs(theta_next - theta_prev)
        #     Gamma = np.deg2rad(45)  # 45-degree threshold
        #     p_ij = float('inf') if delta_theta > Gamma else 0
        # else:
        #     p_ij = 0

        samples = 20
        total_cost = 0.0
        Q_star = 1.0  # Buffer size in meters
        tke_lim = 0.9

        for i in range(samples):
            ratio = (i + 0.5) / samples
            mx = a.x + dx * ratio
            my = a.y + dy * ratio
            u, v = self.get_wind((mx, my))
            tke = self.get_tke((mx, my)) if self.tke_interp is not None else 0.0

            # Wind coefficient for minimum time
            V_ac = self.UAV_speed
            wind_dot = (u * dx + v * dy) / distance
            w = V_ac / (V_ac + wind_dot)

            # Repulsive potential
            grid_x = int(mx)
            grid_y = int(my)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                D = self.distance_map[grid_y, grid_x]
                U_rep = Q_star - D + 1 if D <= Q_star else 1
            else:
                U_rep = 1  # Default if out of bounds
                
            U_tke = 10.0 if tke > tke_lim else 1.0

            # Step cost
            d_i = distance / samples
            c_step = U_rep * w * d_i * U_tke
            total_cost += c_step

        # total_cost += p_ij
        return total_cost

    def find_nearest(self, new_node):
        if not self.nodes:
            return None
        if self.kd_tree is None or len(self.nodes) != self.kd_tree_node_count:
            points = [(n.x, n.y) for n in self.nodes]
            self.kd_tree = KDTree(points)
            self.kd_tree_node_count = len(self.nodes)
        _, idx = self.kd_tree.query([(new_node.x, new_node.y)])
        return self.nodes[idx[0]]

    def plan(self, start, goal, max_iter=40000, step_size=5.0, goal_radius=5.0, consider_wind=True, consider_tke = True):
        start_time = time.time()
        path_type = "with wind" if consider_wind else "without wind"

        if self.is_collision(start):
            print(f"Start point {start} is inside an obstacle or outside map bounds for path {path_type}")
            return {"path": [], "length": 0.0, "cost": float('inf'), "time": 0.0}
        if self.is_collision(goal):
            print(f"Goal point {goal} is inside an obstacle or outside map bounds for path {path_type}")
            return {"path": [], "length": 0.0, "cost": float('inf'), "time": 0.0}

        start_node = Node(*start)
        self.nodes = [start_node]
        goal_node = Node(*goal)
        path_found = False

        for i in range(max_iter):
            if np.random.rand() < 0.2 and not path_found:
                sample = goal_node
            else:
                sample = Node(
                    np.random.uniform(0, self.map_width),
                    np.random.uniform(0, self.map_height)
                )

            nearest = self.find_nearest(sample)
            if nearest is None:
                continue

            theta = np.arctan2(sample.y - nearest.y, sample.x - nearest.x)
            distance = np.hypot(sample.x - nearest.x, sample.y - nearest.y)
            step = min(step_size, distance)
            new_node = Node(
                nearest.x + step * np.cos(theta),
                nearest.y + step * np.sin(theta)
            )

            if self.is_collision((nearest.x, nearest.y), (new_node.x, new_node.y)):
                continue

            # Pass previous direction if available
            if nearest.parent:
                prev_dx = nearest.x - nearest.parent.x
                prev_dy = nearest.y - nearest.parent.y
            else:
                prev_dx, prev_dy = None, None

            if consider_wind and consider_tke:
                cost_fn = lambda a, b: self.motion_cost(a, b, prev_dx, prev_dy, tke_lim = 0.7)
            elif consider_wind:
                cost_fn = lambda a, b: self.motion_cost(a, b, prev_dx, prev_dy)  # TKE ignored
            else:
                cost_fn = lambda a, b: np.hypot(b.x - a.x, b.y - a.y)  # Euclidean distance
            
            new_node.cost = nearest.cost + cost_fn(nearest, new_node)
            new_node.parent = nearest

            neighborhood_radius = 5.0
            points = [(n.x, n.y) for n in self.nodes]
            if self.kd_tree is None:
                self.kd_tree = KDTree([(n.x, n.y) for n in self.nodes])
            else:
                # Use incremental update method if available
                self.kd_tree = KDTree([(n.x, n.y) for n in self.nodes])
            neighbor_indices = self.kd_tree.query_ball_point([new_node.x, new_node.y], neighborhood_radius)

            for idx in neighbor_indices:
                neighbor = self.nodes[idx]
                if neighbor is new_node:
                    continue
                tentative_cost = neighbor.cost + cost_fn(neighbor, new_node)
                if tentative_cost < new_node.cost and not self.is_collision((neighbor.x, neighbor.y), (new_node.x, new_node.y)):
                    new_node.parent = neighbor
                    new_node.cost = tentative_cost

            self.nodes.append(new_node)
            points.append((new_node.x, new_node.y))
            self.kd_tree = KDTree(points)
            
            for idx in neighbor_indices:
                neighbor = self.nodes[idx]
                if neighbor is new_node:
                    continue
                cost_through_new = new_node.cost + cost_fn(new_node, neighbor)
                if cost_through_new < neighbor.cost and not self.is_collision((new_node.x, new_node.y), (neighbor.x, neighbor.y)):
                    neighbor.parent = new_node
                    neighbor.cost = cost_through_new

            if np.hypot(new_node.x - goal[0], new_node.y - goal[1]) < goal_radius:
                path_found = True
                break
            
        path = []
        total_cost = new_node.cost if path_found else float('inf')
        if path_found:
            node = new_node
            while node is not None:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()
        else:
            print(f"No path found from {start} to {goal} for path {path_type} after {max_iter} iterations")

        elapsed_time = time.time() - start_time
        path_length = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))) if path else 0

        print(f"Metrics for path {path_type}:")
        print(f"  Path length: {path_length:.2f} meters")
        print(f"  Total cost: {total_cost:.2f} units")
        print(f"  Computation time: {elapsed_time:.2f} seconds")

        return {
            "path": path,
            "length": path_length,
            "cost": total_cost,
            "time": elapsed_time
        }

    def plan_multiple_paths(self, start, goal, num_runs=3, max_iter=40000, step_size=5.0, goal_radius=5.0):
        metrics = []
        paths_with_tke = []
        paths_with_wind = []
        paths_without_wind = []

        for run_id in range(1, num_runs + 1):
            # Plan with wind
            print(f"\nRun {run_id} with wind and TKE from {start} to {goal}")
            result_with_tke = self.plan(start, goal, max_iter, step_size, goal_radius, consider_wind=True, consider_tke=True)
            metrics.append({
                "Run_ID": run_id,
                "Start_X": start[0],
                "Start_Y": start[1],
                "Goal_X": goal[0],
                "Goal_Y": goal[1],
                "Wind_Condition": "With Wind and TKE",
                "Path_Length_m": result_with_tke["length"],
                "Total_Cost_s": result_with_tke["cost"],
                "Computation_Time_s": result_with_tke["time"]
            })
            paths_with_tke.append(result_with_tke["path"])
            
            #wind only
            print(f"\nRun {run_id} with wind from {start} to {goal}")
            result_with_wind = self.plan(start, goal, max_iter, step_size, goal_radius, consider_wind=True, consider_tke=False)
            metrics.append({
                "Run_ID": run_id,
                "Start_X": start[0],
                "Start_Y": start[1],
                "Goal_X": goal[0],
                "Goal_Y": goal[1],
                "Wind Condition": "With Wind",
                "Path_Length_m": result_with_wind["length"],
                "Total_Cost_s": result_with_wind["cost"],
                "Computation_Time_s": result_with_wind["time"]
            })
            paths_with_wind.append(result_with_wind["path"])

            # Plan without wind
            print(f"\nRun {run_id} without wind from {start} to {goal}")
            result_without_wind = self.plan(start, goal, max_iter, step_size, goal_radius, consider_wind=False)
            metrics.append({
                "Run_ID": run_id,
                "Start_X": start[0],
                "Start_Y": start[1],
                "Goal_X": goal[0],
                "Goal_Y": goal[1],
                "Wind_Condition": "No Wind",
                "Path_Length_m": result_without_wind["length"],
                "Total_Cost_s": result_without_wind["cost"],
                "Computation_Time_s": result_without_wind["time"]
            })
            paths_without_wind.append(result_without_wind["path"])

        # Save metrics to Excel
        # df = pd.DataFrame(metrics)
        # df.to_excel(excel_file, index=False, engine='openpyxl')
        # print(f"Metrics saved to {excel_file}")

        return paths_with_tke, paths_with_wind, paths_without_wind
    
    def _plot_paths(self, ax, paths_with_tke=None, paths_with_wind=None, paths_without_wind=None):
        """
        Helper method to plot all paths on the given axis.
        
        Args:
            ax: Matplotlib axis to plot on
            paths_with_tke, paths_with_wind, paths_without_wind: Lists of path coordinates
        """
        has_paths = False
        # Plot paths with TKE (green)
        if paths_with_tke:
            for i, p in enumerate(paths_with_tke):
                if p and len(p) > 0:
                    ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
                    ds = np.insert(ds, 0, 0)
                    csx = CubicSpline(ds, [pt[0] for pt in p])
                    csy = CubicSpline(ds, [pt[1] for pt in p])
                    ss = np.linspace(0, ds[-1], 200)
                    smooth = np.vstack((csx(ss), csy(ss))).T
                    ax.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='orange', alpha=1, 
                           label='With Wind & TKE' if i == 0 else "")
                    sx, sy = smooth[0]; gx, gy = smooth[-1]
                    ax.scatter([sx], [sy], c='green', s=80, marker='o', alpha=0.7)
                    ax.scatter([gx], [gy], c='green', s=80, marker='X', alpha=0.7)
                    has_paths = True
                    
        # Plot paths with wind (red)
        if paths_with_wind:
            for i, p in enumerate(paths_with_wind):
                if p and len(p) > 0:
                    ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
                    ds = np.insert(ds, 0, 0)
                    csx = CubicSpline(ds, [pt[0] for pt in p])
                    csy = CubicSpline(ds, [pt[1] for pt in p])
                    ss = np.linspace(0, ds[-1], 200)
                    smooth = np.vstack((csx(ss), csy(ss))).T
                    ax.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='red', alpha=1, 
                           label='With Wind' if i == 0 else "")
                    sx, sy = smooth[0]; gx, gy = smooth[-1]
                    ax.scatter([sx], [sy], c='red', s=80, marker='o', alpha=0.9)
                    ax.scatter([gx], [gy], c='red', s=80, marker='X', alpha=0.9)
                    has_paths = True
        # Plot paths without wind (blue)
        if paths_without_wind:
            for i, p in enumerate(paths_without_wind):
                if p and len(p) > 0:
                    ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
                    ds = np.insert(ds, 0, 0)
                    csx = CubicSpline(ds, [pt[0] for pt in p])
                    csy = CubicSpline(ds, [pt[1] for pt in p])
                    ss = np.linspace(0, ds[-1], 200)
                    smooth = np.vstack((csx(ss), csy(ss))).T
                    ax.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='magenta', alpha=1, 
                           label='No Wind' if i == 0 else "")
                    sx, sy = smooth[0]; gx, gy = smooth[-1]
                    ax.scatter([sx], [sy], c='blue', s=80, marker='o', alpha=0.7)
                    ax.scatter([gx], [gy], c='blue', s=80, marker='X', alpha=0.7)
                    has_paths = True
        if has_paths:
            ax.legend()

    def plot_environment(self, paths_with_tke=None, paths_with_wind=None, paths_without_wind=None, 
                        wind_resolution=2.0, tke_res=1.0, plot_res=20.0):
        # Wind plot (Figure 1)
        plt.figure(1)
        plt.clf()
        ax1 = plt.gca()
        ax1.imshow(self.obstacle_map, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], alpha=0.5)
        x_plot = np.arange(0, self.map_width, plot_res)
        y_plot = np.arange(0, self.map_height, plot_res)
        X, Y = np.meshgrid(x_plot, y_plot)
        points = np.column_stack((Y.ravel(), X.ravel()))  # (y, x)
        U = self.u_interp(points)
        V = self.v_interp(points)
        U_grid = U.reshape(Y.shape)
        V_grid = V.reshape(Y.shape)
        magnitude = np.sqrt(U_grid**2 + V_grid**2)
        q = ax1.quiver(X, Y, U_grid, V_grid, magnitude, scale=80, cmap='viridis')
        plt.colorbar(q, ax=ax1, label='Wind Speed (m/s)')
        self._plot_paths(ax1, paths_with_tke, paths_with_wind, paths_without_wind)
        ax1.set_title("Wind Vectors with Paths")
        ax1.set_xlim(0, self.map_width)
        ax1.set_ylim(0, self.map_height)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.grid(alpha=0.2)

        # TKE plot (Figure 2)
        plt.figure(2)
        plt.clf()
        ax2 = plt.gca()
        ax2.imshow(self.obstacle_map, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], alpha=0.5)
        if self.tke_interp is not None:
            xg = np.linspace(0, self.map_width, int(self.map_width / tke_res) + 1)
            yg = np.linspace(0, self.map_height, int(self.map_height / tke_res) + 1)
            xx, yy = np.meshgrid(xg, yg)
            points = np.column_stack((yy.ravel(), xx.ravel()))  # (y, x)
            tke_values = self.tke_interp(points)
            tke_grid = tke_values.reshape(yy.shape)
            cf = ax2.contourf(xx, yy, tke_grid, levels=10, cmap='viridis', alpha=0.8)
            plt.colorbar(cf, ax=ax2, label='Turbulence Kinetic Energy (m²/s²)')
        self._plot_paths(ax2, paths_with_tke, paths_with_wind, paths_without_wind)
        ax2.set_title("TKE Contour with Paths")
        ax2.set_xlim(0, self.map_width)
        ax2.set_ylim(0, self.map_height)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.grid(alpha=0.2)
        plt.show()

    # def plot_environment(self, paths_with_tke=None, paths_with_wind=None, paths_without_wind=None, wind_resolution=2.0):
    #     plt.figure(figsize=(14,10))
        
    #     obstacle_image = self.obstacle_map
    #     plt.imshow(obstacle_image, origin='lower', cmap='gray', extent=[0, self.map_width, 0, self.map_height], alpha=0.5)
        
    #     # Plot TKE if available
    #     if self.tke_interp is not None:
    #         xg = np.linspace(0, self.map_width, int(self.map_width / wind_resolution) + 1)
    #         yg = np.linspace(0, self.map_height, int(self.map_height / wind_resolution) + 1)
    #         xx, yy = np.meshgrid(xg, yg)
    #         points = np.column_stack((xx.ravel(), yy.ravel()))

    #         grid_x = np.floor(points[:, 0]).astype(int)
    #         grid_y = np.floor(points[:, 1]).astype(int)
    #         valid = (grid_x >= 0) & (grid_x < self.grid_width) & (grid_y >= 0) & (grid_y < self.grid_height)
    #         obstacle_mask = np.ones(len(points), dtype=bool)
    #         obstacle_mask[valid] = self.obstacle_map[grid_y[valid], grid_x[valid]] == 0

    #         tke_values = np.zeros(len(points))
    #         open_pts = np.where(obstacle_mask)[0]

    #     if len(open_pts) > 0:
    #         open_points = points[open_pts]
    #         open_points_yx = np.column_stack((open_points[:, 1], open_points[:, 0]))
    #         tke_values[open_pts] = self.tke_interp(open_points_yx)
    #         tke_values = tke_values.reshape(xx.shape)
    #     plt.imshow(tke_values, origin='lower', extent=[0, self.map_width, 0, self.map_height],
    #             cmap='hot', alpha=0.8, interpolation='nearest')
    #     plt.colorbar(label='Turbulence Kinetic Energy (m²/s²)')
        
    #     has_paths = False
    #     # Plot paths with TKE
    #     if paths_with_tke:
    #         for i, p in enumerate(paths_with_tke):
    #             if p and len(p) > 0:
    #                 ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
    #                 ds = np.insert(ds, 0, 0)
    #                 csx = CubicSpline(ds, [pt[0] for pt in p])
    #                 csy = CubicSpline(ds, [pt[1] for pt in p])
    #                 ss = np.linspace(0, ds[-1], 200)
    #                 smooth = np.vstack((csx(ss), csy(ss))).T

    #                 plt.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='green', alpha=0.5, label='With Wind & TKE' if i == 0 else "")
    #                 sx, sy = smooth[0]; gx, gy = smooth[-1]
    #                 plt.scatter([sx], [sy], c='green', s=80, marker='o', alpha=0.7)
    #                 plt.scatter([gx], [gy], c='green', s=80, marker='X', alpha=0.7)
    #                 has_paths = True
                
    #     # Plot paths with wind
    #     if paths_with_wind:
    #         for i, p in enumerate(paths_with_wind):
    #             if p and len(p) > 0:
    #                 ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
    #                 ds = np.insert(ds, 0, 0)
    #                 csx = CubicSpline(ds, [pt[0] for pt in p])
    #                 csy = CubicSpline(ds, [pt[1] for pt in p])
    #                 ss = np.linspace(0, ds[-1], 200)
    #                 smooth = np.vstack((csx(ss), csy(ss))).T

    #                 plt.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='red', alpha=0.5, label='With Wind' if i == 0 else "")
    #                 sx, sy = smooth[0]; gx, gy = smooth[-1]
    #                 plt.scatter([sx], [sy], c='red', s=80, marker='o', alpha=0.7)
    #                 plt.scatter([gx], [gy], c='red', s=80, marker='X', alpha=0.7)
    #                 has_paths = True

    #     # Plot paths without wind
    #     if paths_without_wind:
    #         for i, p in enumerate(paths_without_wind):
    #             if p and len(p) > 0:
    #                 ds = np.cumsum(np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1)))
    #                 ds = np.insert(ds, 0, 0)
    #                 csx = CubicSpline(ds, [pt[0] for pt in p])
    #                 csy = CubicSpline(ds, [pt[1] for pt in p])
    #                 ss = np.linspace(0, ds[-1], 200)
    #                 smooth = np.vstack((csx(ss), csy(ss))).T

    #                 plt.plot(smooth[:, 0], smooth[:, 1], '--', linewidth=2, color='blue', alpha=0.5, label='No Wind' if i == 0 else "")
    #                 sx, sy = smooth[0]; gx, gy = smooth[-1]
    #                 plt.scatter([sx], [sy], c='blue', s=80, marker='o', alpha=0.7)
    #                 plt.scatter([gx], [gy], c='blue', s=80, marker='X', alpha=0.7)
    #                 has_paths = True

    #     if has_paths:
    #         plt.legend()
    #     plt.xlim(0, self.map_width)
    #     plt.ylim(0, self.map_height)
    #     plt.xlabel("X (m)")
    #     plt.ylabel("Y (m)")
    #     plt.title("RRT* Paths with TKE Map")
    #     plt.grid(alpha=0.2)
    #     plt.show()

if __name__ == "__main__":
    planner = WindAwareRRTStar(
        map_size=(1300, 1000),
        obstacle_data_file="data/obstacle_map.json",
        wind_data="data/wind_50m.csv",
        tke_data="data/tke.csv",
        UAV_speed=8.0,
        tke_lim=0.7
    )
    
    test_pos = (250, 400)
    u_grid, v_grid = planner.get_wind(test_pos)
    tke = planner.get_tke(test_pos)
    print(f"Wind at {test_pos}: u={u_grid}, v={v_grid}, TKE={tke}")
    
    # Save interpolated wind data
    # planner.save_interpolated_wind_data(csv_file="interpolated_wind_data.csv", resolution=10.0)
    
    # Plan multiple paths
    start = (350, 380)
    goal = (670, 570)
    num_runs = 1
    paths_with_tke, paths_with_wind, paths_without_wind = planner.plan_multiple_paths(
        start=start,
        goal=goal,
        num_runs=num_runs
    )

    # Analyze costs
    # df = pd.read_excel("path_metrics1.xlsx")
    # tke_cost = df[df['Wind Condition'] == 'With Wind & TKE']['Total_Cost_s']
    # wind_cost = df[df['Wind Condition'] == 'With Wind']['Total_Cost_s']
    # no_wind_cost = df[df['Wind Condition'] == 'No Wind']['Total_Cost_s']

    # print("TKE-Aware Cost: Mean =", tke_cost.mean(), "Std =", tke_cost.std())
    # print("Wind-Aware Cost: Mean =", wind_cost.mean(), "Std =", wind_cost.std())
    # print("No-Wind Cost: Mean =", no_wind_cost.mean(), "Std =", no_wind_cost.std())

    # # Statistical tests (example for TKE vs. Wind)
    # t_stat, p_value = ttest_rel(tke_cost, wind_cost)
    # print(f"Paired t-test (TKE vs Wind): t={t_stat:.4f}, p={p_value:.4f}")
    
    # # print(f"Wilcoxon Test for Total Cost: stat={stat:.4f}, p={p_value:.4f}")
    
    # def compute_wind_dot(planner, path):
    #     wind_dots = []
    #     for j in range(1, len(path)):
    #         p1, p2 = path[j-1], path[j]
    #         dx = p2[0] - p1[0]
    #         dy = p2[1] - p1[1]
    #         d = np.hypot(dx, dy)
    #         if d == 0:
    #             continue
    #         samples = 10
    #         for i in range(samples):
    #             ratio = (i + 0.5) / samples
    #             mx = p1[0] + ratio * dx
    #             my = p1[1] + ratio * dy
    #             u, v = planner.get_wind((mx, my))
    #             wind_dot = (u * dx + v * dy) / d
    #             wind_dots.append(wind_dot)
    #     return wind_dots
    
    # wind_dot_means = [np.mean(compute_wind_dot(planner, path)) for path in paths_with_wind]
    # # corr, p_value = pearsonr(wind_dot_means, wind_cost)
    # # print(f"Correlation between wind dot product and cost: r={corr:.4f}, p={p_value:.4f}")

    # # Example for one wind-aware path
    # wind_dots = compute_wind_dot(planner, paths_with_wind[0])
    # print(f"Mean Wind Dot Product: {np.mean(wind_dots):.2f}, Std: {np.std(wind_dots):.2f}")

    # wind_length = df[df['Wind_Condition'] == 'With Wind']['Path_Length_m']
    # print(f"Wind-Aware Cost vs. Length Ratio: {(wind_cost / wind_length).mean():.2f}")
    
    planner.plot_environment(paths_with_tke, paths_with_wind, paths_without_wind)