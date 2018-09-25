#!/usr/bin/env python3

"""
    # Hector Esteban Cabezos
    # {student id}
    # hectorec@kth.se
"""

# Python standard library
from math import cos, sin, atan2, fabs

# Numpy
import numpy as np
import math

# "Local version" of ROS messages
from local.geometry_msgs import PoseStamped, Quaternion
from local.sensor_msgs import LaserScan
from local.map_msgs import OccupancyGridUpdate

from grid_map import GridMap


class Mapping:
    def __init__(self, unknown_space, free_space, c_space, occupied_space,
                 radius, optional=None):
        self.unknown_space = unknown_space
        self.free_space = free_space
        self.c_space = c_space
        self.occupied_space = occupied_space
        self.allowed_values_in_map = {"self.unknown_space": self.unknown_space,
                                      "self.free_space": self.free_space,
                                      "self.c_space": self.c_space,
                                      "self.occupied_space": self.occupied_space}
        self.radius = radius
        self.__optional = optional

    def get_yaw(self, q):
        """Returns the Euler yaw from a quaternion.
        :type q: Quaternion
        """
        return atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))

    def raytrace(self, start, end):
        """Returns all cells in the grid map that has been traversed
        from start to end, including start and excluding end.
        start = (x, y) grid map index
        end = (x, y) grid map index
        """
        (start_x, start_y) = start
        (end_x, end_y) = end
        x = start_x
        y = start_y
        (dx, dy) = (fabs(end_x - start_x), fabs(end_y - start_y))
        n = dx + dy
        x_inc = 1
        if end_x <= start_x:
            x_inc = -1
        y_inc = 1
        if end_y <= start_y:
            y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2

        traversed = []
        for i in range(0, int(n)):
            traversed.append((int(x), int(y)))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                if error == 0:
                    traversed.append((int(x + x_inc), int(y)))
                y += y_inc
                error += dx

        return traversed

    def add_to_map(self, grid_map, x, y, value):
        """Adds value to index (x, y) in grid_map if index is in bounds.
        Returns weather (x, y) is inside grid_map or not.
        """
        if value not in self.allowed_values_in_map.values():
            raise Exception("{0} is not an allowed value to be added to the map. "
                            .format(value) + "Allowed values are: {0}. "
                            .format(self.allowed_values_in_map.keys()) +
                            "Which can be found in the '__init__' function.")

        if self.is_in_bounds(grid_map, x, y):
            grid_map[x, y] = value
            return True
        return False

    def is_in_bounds(self, grid_map, x, y):
        """Returns weather (x, y) is inside grid_map or not."""
        if x >= 0 and x < grid_map.get_width():
            if y >= 0 and y < grid_map.get_height():
                return True
        return False

    def update_map(self, grid_map, pose, scan):
        """Updates the grid_map with the data from the laser scan and the pose.
        
        For E: 
            Update the grid_map with self.occupied_space.

            Return the updated grid_map.

            You should use:
                self.occupied_space  # For occupied space

                You can use the function add_to_map to be sure that you add
                values correctly to the map.

                You can use the function is_in_bounds to check if a coordinate
                is inside the map.

        For C:
            Update the grid_map with self.occupied_space and self.free_space. Use
            the raytracing function found in this file to calculate free space.

            You should also fill in the update (OccupancyGridUpdate()) found at
            the bottom of this function. It should contain only the rectangle area
            of the grid_map which has been updated.

            Return both the updated grid_map and the update.

            You should use:
                self.occupied_space  # For occupied space
                self.free_space      # For free space

                To calculate the free space you should use the raytracing function
                found in this file.

                You can use the function add_to_map to be sure that you add
                values correctly to the map.

                You can use the function is_in_bounds to check if a coordinate
                is inside the map.

        :type grid_map: GridMap
        :type pose: PoseStamped
        :type scan: LaserScan
        """



        # Current yaw of the robot
        robot_yaw = self.get_yaw(pose.pose.orientation)
        # The origin of the map [m, m, rad]. This is the real-world pose of the
        # cell (0,0) in the map.
        origin = grid_map.get_origin()
        # The map resolution [m/cell]
        #resolution = grid_map.get_resolution(scan)


        """
        Fill in your solution here
        """
        x_scanned = []
        y_scanned = []
        resolution = grid_map.get_resolution()

        max_indexx = -999
        max_indexy = -999
        min_indexx = 999
        min_indexy = 999

        for i, range_value in enumerate(scan.ranges):
            angle  = scan.angle_min + i*scan.angle_increment + robot_yaw

            if (scan.range_min < range_value < scan.range_max):
                x_laser = range_value*cos(angle)
                y_laser = range_value*sin(angle)
                x_total = x_laser + pose.pose.position.x - origin.position.x
                y_total = y_laser + pose.pose.position.y - origin.position.y
                x_index = int(x_total/resolution)
                if (x_index > max_indexx):
                    max_indexx = x_index
                if (x_index < min_indexx):
                    min_indexx = x_index
                y_index = int(y_total/resolution)
                if (y_index > max_indexy):
                    max_indexy = y_index
                if (y_index < min_indexy):
                    min_indexy = y_index

                self.add_to_map(grid_map, x_index, y_index, self.occupied_space)
        
                ## PART E
                x_total_min = pose.pose.position.x - origin.position.x
                y_total_min = pose.pose.position.y - origin.position.y
                x_index_min = int(x_total_min/resolution)
                y_index_min = int(y_total_min/resolution)
                cells_free = self.raytrace([x_index_min, y_index_min],[x_index, y_index])
                #print("{}".format(len(cells_free)))
                for point in cells_free:
                    self.add_to_map(grid_map, point[0], point[1], self.free_space)

        """
        For C only!
        Fill in the update correctly below.
        """ 
        # Only get the part that has been updated
        update = OccupancyGridUpdate()
        # The minimum x index in 'grid_map' that has been updated
        update.x = min_indexx
        # The minimum y index in 'grid_map' that has been updated
        update.y = min_indexy
        # Maximum x index - minimum x index + 1
        update.width = max_indexx - update.x + 1
        # Maximum y index - minimum y index + 1
        update.height = max_indexy - update.y + 1
        # The map data inside the rectangle, in row-major order.
        update.data = []
        for j in range(update.height):
            for i in range(update.width):
                update.data.append(grid_map[j+update.y, i+update.x])

        # Return the updated map together with only the
        # part of the map that has been updated
        return grid_map, update

    def inflate_map(self, grid_map):
        """For C only!
        Inflate the map with self.c_space assuming the robot
        has a radius of self.radius.
        
        Returns the inflated grid_map.

        Inflating the grid_map means that for each self.occupied_space
        you calculate and fill in self.c_space. Make sure to not overwrite
        something that you do not want to.


        You should use:
            self.c_space  # For C space (inflated space).
            self.radius   # To know how much to inflate.

            You can use the function add_to_map to be sure that you add
            values correctly to the map.

            You can use the function is_in_bounds to check if a coordinate
            is inside the map.

        :type grid_map: GridMap
        """


        """
        Fill in your solution here
        """
        ## PART E
        resolution = grid_map.get_resolution()
        height = grid_map.get_height()
        width = grid_map.get_width()
        for i in range(height):
            for j in range(width):
                if (grid_map[i,j] == self.occupied_space):
                    radius_int = self.radius
                    for y_coord in  range(i-radius_int, i+radius_int):
                        for x_coord in  range(j-radius_int, j+radius_int):
                            if (math.sqrt(y_coord*y_coord + x_coord*x_coord) <= radius_int) and (grid_map[y_coord, x_coord]!=self.occupied_space):
                                self.add_to_map(grid_map, y_coord, x_coord, self.c_space)



        
        # Return the inflated map
        return grid_map
