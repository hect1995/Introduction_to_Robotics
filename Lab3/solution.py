#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hector Esteban Cabezos
# hectorec@kth.se

from dubins import Car
import matplotlib.pyplot as plt
import random
import copy
import math
import numpy as np
from numpy import linalg as LA
import time

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
                 randArea, expandDis=0.5, goalSampleRate=10, maxIter=500):
        """
        Setting Parameter
        start:The position where the robot will be stablished at the begining [x,y]
        goal:The position that wants to reach at the end [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.minrandx = randArea[0]
        self.maxrandx = randArea[1]
        self.minrandy = randArea[2]
        self.maxrandy = randArea[3]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self):
        self.vertexList = [self.start]
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
            while (crash == False) and (counter_straight<10):
                if (counter_straight == 0):
                    x_aux, y_aux, theta_aux = car_my.step(nearestNode.x, nearestNode.y, nearestNode.theta, phi)

                else:
                    x_aux, y_aux, theta_aux = car_my.step(x_aux, y_aux, theta_aux, phi)

                x.append(x_aux)
                y.append(y_aux)
                theta.append(theta_aux)

                counter_straight = counter_straight + 1


                if not self.__CollisionCheck(x_aux, y_aux):
                    crash = True
                    counter_straight = counter_straight//2


            if counter_straight>0:
                new_node = Node(x=x[counter_straight-1], y=y[counter_straight-1], parent=nind, theta=theta[counter_straight-1])
                new_node.phi = phi
                new_node.dt = (counter_straight-1)*car_my.dt + nearestNode.dt

                self.vertexList.append(new_node)
                print("\nlength: {}\n".format(len(self.vertexList)))

                dx = new_node.x - self.end.x
                dy = new_node.y - self.end.y
                d = math.sqrt(dx * dx + dy * dy)
                if d <= self.expandDis:
                    print("Goal!!")
                    time.sleep(1)
                    print("\n{}\n{}\n{}\n".format(new_node.x, new_node.y, nind))
                    #time.sleep(5)
                    break
            #self.DrawGraph(rnd)

        path = []
        times = []
        lastIndex = len(self.vertexList) - 1
        while self.vertexList[lastIndex].parent is not None: # we start from the goalnode
            node = self.vertexList[lastIndex]
            path.append(node.phi)
            times.append(node.dt)
            lastIndex = node.parent

        times.append(0)

        path_toreturn = path[::-1]
        time_toreturn = times[::-1]


        return path_toreturn, time_toreturn

    def DrawGraph(self, rnd=None):
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.vertexList:
            if node.parent is not None:
                plt.plot([node.x, self.vertexList[node.parent].x], [
                    node.y, self.vertexList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.minrandx, self.maxrandx, self.minrandy, self.maxrandy])
        plt.grid(True)
        plt.pause(0.01)



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
            if x >= self.maxrandx-0.1:
                return False
            if y >= self.maxrandy-0.1:
                return False

            if y < self.minrandy+0.1:
                return False

        return True  # safe


def solution(car):
    global car_my
    car_my = car
    position_start = [car_my.x0, car_my.y0]
    position_goal = [car_my.xt, car_my.yt]


    my_rrt = RRT(position_start, position_goal, car_my.obs, randArea = [car_my.xlb, car_my.xub, car_my.ylb, car_my.yub])
    controls, times = my_rrt.Planning()

    print("\nPath\n", controls)
    print("\nTimes\n", times)


    ''' <<< write your code below >>> '''

    return controls, times
