import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import matplotlib.patches as patches

class Node:
    __slots__ = ("x","y","parent","cost")
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x, self.y, self.parent, self.cost = x, y, parent, cost

class SimpleRRTStar:
    def __init__(self, map_size=(100,100)):
        self.W, self.H = map_size
        self._build_obstacle_map()

    def _build_obstacle_map(self):
        w,h = self.W, self.H
        self.obstacle_map = np.zeros((h,w), bool)
        Y,X = np.ogrid[:h, :w]
        for (cx,cy),r in [((30,30),15), ((70,70),15)]:
            mask = (X-cx)**2 + (Y-cy)**2 < r*r
            self.obstacle_map[mask] = True

    def _collides_line(self, a, b):
        # Bresenham line‐drawing collision check
        x0,y0 = map(int, a)
        x1,y1 = map(int, b)
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0<x1 else -1
        sy = 1 if y0<y1 else -1
        err = dx - dy
        while True:
            if 0<=x0<self.W and 0<=y0<self.H and self.obstacle_map[y0,x0]:
                return True
            if x0==x1 and y0==y1:
                break
            e2 = 2*err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return False

    def plan_and_animate(self, start, goal,
                         max_iter=300,
                         step_size=5.0,
                         goal_radius=5.0,
                         save_gif=True):
        fig,ax = plt.subplots(figsize=(6,6))
        # draw obstacles
        ax.imshow(self.obstacle_map, origin='lower',
                  extent=[0,self.W,0,self.H], cmap='binary', alpha=1)
        # start/goal
        ax.scatter(*start, c='green', label='Start')
        ax.scatter(*goal, c='red', marker='X', label='Goal')
        ax.add_patch(patches.Circle(goal, goal_radius,
                                   fill=False, ls='--', color='red'))
        ax.set_xlim(0,self.W); ax.set_ylim(0,self.H)
        ax.grid(alpha=0.2); ax.legend()

        # RRT* data
        nodes = [Node(*start)]
        tree_segments = []
        tree_collection = LineCollection(tree_segments, linewidths=0.7, alpha=0.6)
        ax.add_collection(tree_collection)
        # highlight collections
        candidate_collection = LineCollection([], linewidths=2, linestyles='--', colors='red', alpha=0.8)
        accepted_collection = LineCollection([], linewidths=2.5, colors='green', alpha=1.0)
        ax.add_collection(candidate_collection)
        ax.add_collection(accepted_collection)
        path_line, = ax.plot([], [], 'b--', lw=2)

        goal_node = Node(*goal)
        path_found = False
        path = np.empty((0,2))

        # KD‑tree placeholders
        kd_tree = None
        pts = None

        # queue of rewiring‐highlight actions
        rewiring_queue = []

        def update(frame):
            nonlocal kd_tree, pts, path_found, path

            # if there are highlight actions pending, do them first
            if rewiring_queue:
                action, seg = rewiring_queue.pop(0)
                if action == 'candidate':
                    candidate_collection.set_segments([seg])
                    accepted_collection.set_segments([])
                elif action == 'accepted':
                    accepted_collection.set_segments([seg])
                    candidate_collection.set_segments([])
                elif action == 'clear':
                    candidate_collection.set_segments([])
                    accepted_collection.set_segments([])
                return tree_collection, candidate_collection, accepted_collection, path_line

            if frame >= max_iter:
                return tree_collection, candidate_collection, accepted_collection, path_line

            # sample (goal‐bias if not yet found)
            if (not path_found and np.random.rand() < 0.2):
                sample = goal_node
            else:
                sample = Node(np.random.uniform(0,self.W),
                              np.random.uniform(0,self.H))

            # rebuild KD‑tree if needed
            if pts is None or len(nodes) != pts.shape[0]:
                pts = np.array([[n.x,n.y] for n in nodes])
                kd_tree = cKDTree(pts)

            # find nearest
            _, idx = kd_tree.query([sample.x, sample.y])
            nearest = nodes[idx]

            # steer toward sample
            dx, dy = sample.x - nearest.x, sample.y - nearest.y
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                return tree_collection, candidate_collection, accepted_collection, path_line
            r = min(step_size, dist)
            new_pos = (nearest.x + r*dx/dist, nearest.y + r*dy/dist)

            # collision check
            if self._collides_line((nearest.x,nearest.y), new_pos):
                return tree_collection, candidate_collection, accepted_collection, path_line

            # add new node
            new_node = Node(new_pos[0], new_pos[1],
                            parent=nearest,
                            cost=nearest.cost + r)
            nodes.append(new_node)
            tree_segments.append([(nearest.x,nearest.y), new_pos])
            tree_collection.set_segments(tree_segments)

            # rewiring: find neighbors in radius
            idxs = kd_tree.query_ball_point([new_node.x,new_node.y], step_size*1.5)
            # draw neighborhood circle briefly
            circle = patches.Circle((new_node.x,new_node.y), step_size*1.5,
                                    fill=False, linestyle=':', color='orange', alpha=0.5)
            ax.add_patch(circle)

            for i in idxs:
                other = nodes[i]
                if other is nearest or other is new_node:
                    continue
                new_cost = new_node.cost + np.hypot(other.x-new_node.x, other.y-new_node.y)
                if new_cost + 1e-6 < other.cost and not self._collides_line((new_node.x,new_node.y),(other.x,other.y)):
                    # enqueue candidate highlight
                    seg = [(new_node.x,new_node.y),(other.x,other.y)]
                    rewiring_queue.append(('candidate', seg))
                    # enqueue actual rewire
                    other.parent = new_node
                    other.cost = new_cost
                    # update segment in tree_segments
                    for s in tree_segments:
                        if np.allclose(s[1],[other.x,other.y]):
                            s[0] = (new_node.x,new_node.y)
                            break
                    tree_collection.set_segments(tree_segments)
                    rewiring_queue.append(('accepted', seg))
                    # clear highlights & circle
                    rewiring_queue.append(('clear', None))
                    break

            circle.remove()

            # check for goal reach
            if not path_found and np.hypot(new_node.x-goal[0], new_node.y-goal[1]) < goal_radius:
                path_found = True
                # backtrack from this new_node
                p = []
                cur = new_node
                while cur:
                    p.append((cur.x, cur.y))
                    cur = cur.parent
                path = np.array(p[::-1])
                path_line.set_data(path[:,0], path[:,1])

            return tree_collection, candidate_collection, accepted_collection, path_line

        anim = FuncAnimation(fig, update, frames=max_iter, interval=50,
                             blit=True, repeat=False)

        if save_gif:
            anim.save("rrtstar_rewire_highlight.gif", writer="pillow", fps=10)
            print("Saved ➔ rrtstar_rewire_highlight.gif")
        else:
            plt.show()

        return {
            "nodes": len(nodes),
            "path_found": path_found,
            "path_length": len(path) if path_found else 0
        }

if __name__=="__main__":
    planner = SimpleRRTStar((100,100))
    result = planner.plan_and_animate(
        start=(10,10),
        goal=(90,90),
        max_iter=300,
        step_size=5.0,
        goal_radius=5.0,
        save_gif=True
    )
    print(result)
