#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

Path Planning Sample Code with RRT for car like robot.

author: AtsushiSakai(@Atsushi_twi)

"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import copy
import math
import random
import sys
import os
import time
__file__ = 'rrt_star_reeds_shepp.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../ReedsSheppPath/")

try:
    import reeds_shepp_path_planning
except:
    raise

show_animation = False
STEP_SIZE = 0.1
curvature = 0.3


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 goalSampleRate=10, maxIter=400):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])
        self.h = 1
        self.w = 2
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        start_end = Node(0, 0, 0)
        start_end.path_x += [self.start.x, self.end.x]
        start_end.path_y += [self.start.y, self.end.y]
        start_end.path_yaw += [self.start.yaw, self.end.yaw]
        if not self.CollisionCheck(start_end, self.obstacleList):
            return None
        for i in range(self.maxIter):
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind)
            if newNode is None:
                continue

            if self.CollisionCheck(newNode, self.obstacleList):
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                if newNode is None:
                    continue
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)

            if animation and i % 5 == 0:
                self.DrawGraph(rnd=rnd)

        # generate coruse
        lastIndex = self.get_best_last_index()
        if lastIndex is None:
            return None
        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return newNode

        dlist = []
        for i in nearinds:
            tNode = self.steer(newNode, i)
            if tNode is None:
                continue

            if self.CollisionCheck(tNode, self.obstacleList):
                dlist.append(tNode.cost)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode = self.steer(newNode, minind)

        return newNode


    def steer(self, rnd, nind):

        nearestNode = self.nodeList[nind]

        px, py, pyaw, mode, clen = reeds_shepp_path_planning.reeds_shepp_path_planning(
            nearestNode.x, nearestNode.y, nearestNode.yaw, rnd.x, rnd.y, rnd.yaw, curvature, STEP_SIZE)

        if px is None:
            return None

        newNode = copy.deepcopy(nearestNode)
        newNode.x = px[-1]
        newNode.y = py[-1]
        newNode.yaw = pyaw[-1]

        newNode.path_x = px
        newNode.path_y = py
        newNode.path_yaw = pyaw
        newNode.cost += sum([abs(c) for c in clen])
        newNode.parent = nind

        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand),
                   random.uniform(-math.pi, math.pi)
                   ]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y, self.end.yaw]

        node = Node(rnd[0], rnd[1], rnd[2])

        return node

    def get_best_last_index(self):
        #  print("get_best_last_index")

        YAWTH = np.deg2rad(3.0)
        XYTH = 0.5

        goalinds = []
        for (i, node) in enumerate(self.nodeList):
            if self.calc_dist_to_goal(node.x, node.y) <= XYTH:
                goalinds.append(i)
        #  print("OK XY TH num is")
        #  print(len(goalinds))

        # angle check
        fgoalinds = []
        for i in goalinds:
            if abs(self.nodeList[i].yaw - self.end.yaw) <= YAWTH:
                fgoalinds.append(i)
        #  print("OK YAW TH num is")
        #  print(len(fgoalinds))

        if not fgoalinds:
            return None

        mincost = min([self.nodeList[i].cost for i in fgoalinds])
        for i in fgoalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            for (ix, iy, iyaw) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_yaw)):
                path.append([ix, iy, iyaw])
            goalind = node.parent
        path.append([self.start.x, self.start.y, self.start.yaw])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 +
                 (node.yaw - newNode.yaw) ** 2
                 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):

        nnode = len(self.nodeList)

        for i in nearinds:
            nearNode = self.nodeList[i]
            tNode = self.steer(nearNode, nnode - 1)
            if tNode is None:
                continue

            obstacleOK = self.CollisionCheck(tNode, self.obstacleList)
            imporveCost = nearNode.cost > tNode.cost

            if obstacleOK and imporveCost:
                #  print("rewire")
                self.nodeList[i] = tNode

    def DrawGraph(self, rnd=None):  # pragma: no cover
        plt.clf()
        ax = plt.axes()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.path_x, node.path_y, "-g")
                #  plt.plot([node.x, self.nodeList[node.parent].x], [
                #  node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, oyaw, h, w) in self.obstacleList:
            block = Rectangle((ox - 0.5*w, oy - 0.5*h), w, h, angle = np.rad2deg(oyaw), color = 'k')
            ax.add_patch(block)
            
        reeds_shepp_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        reeds_shepp_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

        plt.axis([-2, 22, -2, 22])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.pause(0.01)

        #  plt.show()
        #  input()

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd.x) ** 2 +
                 (node.y - rnd.y) ** 2 +
                 (node.yaw - rnd.yaw) ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind
    
    def in_rectangle(self, x, y, rect, ratio = 1/256):
        x_p = (ratio*x - rect[0]) * np.cos(-rect[2]) - (ratio*y - rect[1]) * np.sin(-rect[2])
        y_p = (ratio*x - rect[0]) * np.sin(-rect[2]) + (ratio*y - rect[1]) * np.cos(-rect[2])
        if abs(x_p) <= ((1/10)/2) and abs(y_p) <= ((1/20)/2):
            return True
        else:
            return False

    def CollisionCheck(self, node, obs):

        for a, b in zip(np.where(obs == 1)):
            for (ix, iy, iyaw) in zip(node.path_x, node.path_y, node.path_yaw):
                if in_rectangle(b, 255-a, (ix, iy, iyaw)):
                    return False # in collision
                
        return True  # safe
    
    
def pi_2_pi(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0
        self.parent = None


def main(start, end, obs, maxIter=3):

    rrt = RRT(start, goal, randArea=[-2.0, 15.0], obstacleList = obs, maxIter=maxIter)
    path = rrt.Planning(animation=show_animation)
        
    return path



