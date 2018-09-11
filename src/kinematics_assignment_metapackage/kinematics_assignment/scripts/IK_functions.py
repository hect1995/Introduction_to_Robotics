#! /usr/bin/env python3

"""
    ESTEBAN CABEZOS, HECTOR
    hectorec@kth.se
"""
import numpy as np
import math
import rospy
from numpy import linalg as LA

L0 = 0.07
L1 = 0.3
L2 = 0.35
# For KUKA
L = 0.4
M = 0.39
BIAS1 = 0.311
BIAS2 = 0.078

def compute_DH(joint_positions):
    DH = np.zeros(shape=(7,4))
    DH[0] = [0.5*math.pi, 0, 0, joint_positions[0]]
    DH[1] = [-0.5*math.pi, 0, 0, joint_positions[1]]
    DH[2] = [-0.5*math.pi, L, 0, joint_positions[2]]
    DH[3] = [0.5*math.pi, 0, 0, joint_positions[3]]
    DH[4] = [0.5*math.pi, M, 0, joint_positions[4]]
    DH[5] = [-0.5*math.pi, 0, 0, joint_positions[5]]
    DH[6] = [0, 0, 0, joint_positions[6]]
    #rospy.loginfo("----------------------------------------------------------------\n{}".format(DH))
    return DH

def compute_T(alpha, d, r, theta):
    T = np.zeros(shape=(4,4))
    T[0] = [np.cos(theta), -np.sin(theta), 0, r] # first row
    T[1] = [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)] # second row
    T[2] = [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)] # third row
    T[3] = [0, 0, 0, 1] # fourth row
    return T

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]

    r1 = x - L0
    r2 = y
    theta_2 = np.arctan2(r2,r1)
    r3 = np.sqrt(np.power(r1,2) + np.power(r2,2))
    theta_1 = np.arccos((np.power(L2,2) - np.power(L1,2) - np.power(r3,2))/(-2*L1*r3))
    q1 = theta_2 - theta_1
    theta_3 = np.arccos((np.power(r3,2) - np.power(L1,2) - np.power(L2,2))/(-2*L1*L2))
    q2 = math.pi - theta_3
    q3 = z
    q = [q1, q2, q3]

    """
    Fill in your IK solution here and return the three joint values in q
    """

    return q

def obtain_d_vectors(T_matrices):
    d_vector = np.zeros(shape=(3,T_matrices.shape[0]))
    rotational_matrix = np.zeros(shape=(T_matrices.shape[0], 3, 3))
    for i in range(d_vector.shape[1]):
        d_vector[:3, i] = T_matrices[i,:3,3]
        rotational_matrix[i] = T_matrices[i,:3,:3]
    return d_vector, rotational_matrix

def obtain_jacobian_matrix(d_vectors, rotational_matrices):
    Jacobian_matrix = np.zeros(shape=(6,d_vectors.shape[1]))
    for i in range(d_vectors.shape[1]):
        Jacobian_matrix[:3, i] = np.cross(rotational_matrices[i,:,2],(d_vectors[:, 6]-d_vectors[:, i]))
        Jacobian_matrix[3:, i] = rotational_matrices[i,:,2]
    return Jacobian_matrix

def obtain_inverse_jacobian(jacobian_matrix):
    J_inverse = np.matmul(jacobian_matrix.transpose(), np.matmul(jacobian_matrix, jacobian_matrix.transpose()));
    return J_inverse

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    rospy.loginfo("{}".format(R))
    norm_difference = 10

    while norm_difference > 0.04:
        dh_matrix = compute_DH(q)
        array_matrix = np.zeros(shape=(7,4,4))

        for i in range(7):
            array_matrix[i][:][:] = compute_T(dh_matrix[i][0], dh_matrix[i][1], dh_matrix[i][2], dh_matrix[i][3])
            
            if i!=0:
                array_matrix[i][:][:] = np.matmul(array_matrix[i-1], array_matrix[i])
        

        #d_vector = np.zeros(shape=(3,7))
        #Jacobian_matrix_n = np.zeros(shape=(6,7))
        """        for i in range(7):
            rospy.loginfo("----\n{}".format(array_matrix[1][:][:]))
            d_vector[0][i] = array_matrix[i][0][3]
            d_vector[1][i] = array_matrix[i][1][3]
            d_vector[2][i] = array_matrix[i][2][3]"""
        [d_vector, rotational_matrices] = obtain_d_vectors(array_matrix)
        rospy.loginfo("----DDDDDDD..........\n{}".format(d_vector))
        Jacobian_matrix_n = obtain_jacobian_matrix(d_vector, rotational_matrices)
        rospy.loginfo("-------JACOBIAN---------\n{}".format(Jacobian_matrix_n))
        """for j in range(7):
                                    Jacobian_matrix_n[0:3][j] = np.cross(array_matrix[j][:][2],(d_vector[:][6]-d_vector[:][j]))
                                    Jacobian_matrix_n[3:7][j] = array_matrix[j][:][2]"""

        #J_inverse = np.matmul(Jacobian_matrix_n.transpose(), np.matmul(Jacobian_matrix_n, Jacobian_matrix_n.transpose()));
        J_inverse = obtain_inverse_jacobian(Jacobian_matrix_n)



        result_coordinates = np.zeros(shape= (3,3))

        for j in range (3):
            for f in range (3):
                result_coordinates[j][f] = array_matrix[6][j][f]

        #rospy.loginfo("POSE MATRIX\n{}".format(result_coordinates))
        # Obtain Yaw, ...
        yaw = np.arctan2(result_coordinates[1][0], result_coordinates[0][0])
        nick = np.arctan2(-1*result_coordinates[2][0], LA.norm([result_coordinates[0][0], result_coordinates[1][0]]) )
        roll = np.arctan2(result_coordinates[2][1], result_coordinates[2][2])
        x_obtained = array_matrix[6][0][3]
        y_obtained = array_matrix[6][1][3]
        z_obtained = array_matrix[6][2][3]

        desired_yaw = np.arctan2(R[1][0], R[0][0])
        desired_nick = np.arctan2(-1*R[2][0], LA.norm([R[0][0], R[1][0]]))
        desired_roll = np.arctan2(R[2][1], R[2][2])

        error_cord = [x_obtained, y_obtained, z_obtained, yaw, nick, roll]-[x, y, z, desired_yaw, desired_nick, desired_roll]

        vector_joints = np.matmul(J_inverse, error_cord.transpose()) 
        #difference_vector = [x-x_calculated, y-y_calculated, z-z_calculated]
        norm_difference = LA.norm(error_cord)
        # TODO: Calculate inverse Jacobian

    #rospy.loginfo("T MATRIX FROM 0 TO END\n{}".format(array_matrix[6]))

    return q
