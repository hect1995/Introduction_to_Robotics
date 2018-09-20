#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hector Esteban Cabezos
# hectorec@kth.se

from dubins import Car
import random
import math
import numpy as np
from numpy import linalg as LA

car_my = Car()

class Node():
    """
    RRT Node
    """
    def __init__(self, x, y, parent = None, theta=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.theta = theta
        self.phi = 0
        self.dt = 0

class RRT():
    """
    RRT Node
    """
    def __init__(self, start, goal, obstacleList,
                 randArea):

        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.vertexList = [self.start]
        self.minrandx = randArea[0]
        self.maxrandx = randArea[1]
        self.minrandy = randArea[2]
        self.maxrandy = randArea[3]
        self.expandDis = 0.85
        self.goalSampleRate = 10
        self.obstacleList = obstacleList

    def Planning(self):
        while True:
            # Get the random point
            if random.randint(0, 100) > self.goalSampleRate:
                rnd = [random.uniform(self.minrandx, self.maxrandx), random.uniform(
                    self.minrandy, self.maxrandy)]
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            nind = self.GetNearestListIndex(rnd) # index closest neighbour
            nearestNode = self.vertexList[nind] # obtain nearest node

            v = np.array([math.cos(nearestNode.theta), math.sin(nearestNode.theta), 0])
            rg = np.array([rnd[0] - nearestNode.x, rnd[1] - nearestNode.y, 0])
            vector_product = np.cross(v, rg)
            cos_phi = v.dot(rg)/(LA.norm(v)*LA.norm(rg))
            phi = np.arccos(cos_phi)
            if vector_product[2]<0:
                phi = phi*-1
                if phi < -1*math.pi/4:
                    phi = -1*math.pi/4
            if phi > math.pi/4:
                phi = math.pi/4

            crash = False
            counter_straight = 0
            x = []
            y = []
            theta = []
            x_use = nearestNode.x
            y_use = nearestNode.y
            theta_use = nearestNode.theta
            while (crash == False) and (counter_straight<15):
                x_use, y_use, theta_use = car_my.step(x_use, y_use, theta_use, phi)

                x.append(x_use)
                y.append(y_use)
                theta.append(theta_use)

                counter_straight = counter_straight + 1

                if not self.__CollisionCheck(x_use, y_use):
                    crash = True
                    counter_straight = counter_straight-1#//2

            if counter_straight>0:
                new_node = Node(x=x[counter_straight-1], y=y[counter_straight-1], parent=nind, theta=theta[counter_straight-1])
                new_node.phi = phi
                new_node.dt = counter_straight*car_my.dt + nearestNode.dt

                self.vertexList.append(new_node)
                dx = new_node.x - self.end.x
                dy = new_node.y - self.end.y
                d = math.sqrt(dx * dx + dy * dy)
                if d <= self.expandDis:
                    break

        path = []
        times = []
        lastIndex = len(self.vertexList) - 1
        while self.vertexList[lastIndex].parent is not None: # we start from the goalnode
            node = self.vertexList[lastIndex]
            path.insert(0,node.phi)
            times.insert(0,node.dt)
            lastIndex = node.parent

        times.insert(0,0) # time 0 at 0 position
        return path, times

    def GetNearestListIndex(self, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in self.vertexList]
        minind = dlist.index(min(dlist))
        return minind

    def __CollisionCheck(self, x, y): # x and y are the coordinates
        for (ox, oy, radius) in self.obstacleList:
            dx = ox - x
            dy = oy - y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= radius+0.1:
                return False  # collision
        if not (self.minrandx <= x <= self.maxrandx-0.1):
            return False

        if not (self.minrandy+0.1 <= y <= self.maxrandy-0.1):
            return False

        return True  # safe


def solution(car):
    global car_my
    car_my = car
    position_start = [car_my.x0, car_my.y0]
    position_goal = [car_my.xt, car_my.yt]

    my_rrt = RRT(position_start, position_goal, car_my.obs, randArea = [car_my.xlb, car_my.xub, car_my.ylb, car_my.yub])
    controls, times = my_rrt.Planning()

    return controls, times
