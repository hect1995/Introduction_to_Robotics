#! /usr/bin/env python3

"""
    ESTEBAN CABEZOS, HECTOR
    hectorec@kth.se
"""
import numpy as np
import math
import rospy
from numpy import linalg as LA
import time

L0 = 0.07
L1 = 0.3
L2 = 0.35
# For KUKA
L = 0.4
M = 0.39
BIAS1 = 0.311
BIAS2 = 0.078
PI = math.pi
def compute_DH(joint_positions):
    # GOOD
    #DH = np.zeros(shape=(7,4))
    """DH[0] = [0.5*PI, 0, 0, joint_positions[0]]
                DH[1] = [-0.5*PI, 0, 0, joint_positions[1]]
                DH[2] = [-0.5*PI, L, 0, joint_positions[2]]
                DH[3] = [0.5*PI, 0, 0, joint_positions[3]]
                DH[4] = [0.5*PI, M, 0, joint_positions[4]]
                DH[5] = [-0.5*PI, 0, 0, joint_positions[5]]
                DH[6] = [0, 0, 0, joint_positions[6]]"""
    DH = np.array([[0.5*PI, 0, 0, joint_positions[0]],[-0.5*PI, 0, 0, joint_positions[1]], [-0.5*PI, L, 0, joint_positions[2]],[0.5*PI, 0, 0, joint_positions[3]],[0.5*PI, M, 0, joint_positions[4]],[-0.5*PI, 0, 0, joint_positions[5]],[0, 0, 0, joint_positions[6]]])
    return DH

def compute_T(alpha, d, r, theta):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)], [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)], [0, np.sin(alpha), np.cos(alpha), d], [0, 0, 0, 1]])
    """T[0] = [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)] # first row
                 T[1] = [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)] # second row
                 T[2] = [0, np.sin(alpha), np.cos(alpha), d] # third row
                 T[3] = [0, 0, 0, 1]""" # fourth row
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
    return q


def obtain_d_vectors(T_matrices):
    d_vector = np.zeros(shape=(3,8))
    rotational_matrix = np.zeros(shape=(8, 3, 3))
    for i in range(7):
        d_vector[:, i+1] = T_matrices[i,:3,3]
        rotational_matrix[i+1] = T_matrices[i,:3,:3]
    d_vector[:,0] = np.zeros(3)
    rotational_matrix[0] = np.array([[0,0,0],[0,0,0],[0,0,1]])
    return d_vector, rotational_matrix

def obtain_jacobian_matrix(d_vectors, rotational_matrices):
    # GOOD
    Jacobian_matrix = np.zeros(shape=(6,7))
    for i in range(7):
        Jacobian_matrix[:3, i] = np.cross(rotational_matrices[i,:,2],(d_vectors[:, 7]-d_vectors[:, i]))
        Jacobian_matrix[3:, i] = rotational_matrices[i,:,2]
    return Jacobian_matrix

def obtain_inverse_jacobian(jacobian_matrix):
    J_inverse = np.matmul(jacobian_matrix.transpose(), np.matmul(jacobian_matrix, jacobian_matrix.transpose()));
    return J_inverse

def compute_euler_angles(result_coordinates):
    yaw = np.arctan2(result_coordinates[1,0], result_coordinates[0,0])
    nick = np.arctan2(-1*result_coordinates[2,0], LA.norm([result_coordinates[2,1], result_coordinates[2,2]]))
    roll = np.arctan2(result_coordinates[2,1], result_coordinates[2,2])
    return yaw, nick, roll

def clean_R(R):
    new_R = np.zeros(shape=(3,3))
    for i in range(3):
        for j in range(3):
            new_R[i,j] = R[i][j]

    return new_R

def end_effector_respect_itself():
    end_effector = np.zeros(shape=(4,1))
    end_effector[0] = 0
    end_effector[1] = 0
    end_effector[2] = BIAS2
    end_effector[3] = 1
    return end_effector

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements

    #q = [0,0,0,0,0,0,0]
    """
    Fill in your IK solution here and return the seven joint values in q
    """
    

    #rospy.loginfo("{}".format(R))
    norm_difference = 10

    while norm_difference > 0.04: #0.04:
        dh_matrix = compute_DH(q)
        array_matrix = np.zeros(shape=(7,4,4))
        auxiliar_mat = np.zeros(shape=(7,4,4))

        for i in range(7):
            array_matrix[i,:,:] = compute_T(dh_matrix[i,0], dh_matrix[i,1], dh_matrix[i,2], dh_matrix[i,3])
            auxiliar_mat[i,:,:] = compute_T(dh_matrix[i,0], dh_matrix[i,1], dh_matrix[i,2], dh_matrix[i,3])
            if i!=0:
                array_matrix[i,:,:] = np.matmul(array_matrix[i-1], array_matrix[i])
        total_array = auxiliar_mat[0,:,:]*auxiliar_mat[1,:,:]*auxiliar_mat[2,:,:]*auxiliar_mat[3,:,:]*auxiliar_mat[4,:,:]*auxiliar_mat[5,:,:]*auxiliar_mat[6,:,:];


        [d_vector, rotational_matrices] = obtain_d_vectors(array_matrix)
        Jacobian_matrix_n = obtain_jacobian_matrix(d_vector, rotational_matrices)
        J_inverse = obtain_inverse_jacobian(Jacobian_matrix_n)

        [yaw, nick, roll] = compute_euler_angles(total_array[:3,:3])
        
        end_effector = end_effector_respect_itself()


        end_vector_ground = array_matrix[6,:,:].dot(end_effector)
        x_obtained = end_vector_ground[0]
        y_obtained = end_vector_ground[1]
        z_obtained = end_vector_ground[2] + BIAS1
        # GOOD UNTIL HERE
        R = clean_R(R)

        [desired_yaw, desired_nick, desired_roll] = compute_euler_angles(R)

        error_cord = np.array([x_obtained-x, y_obtained-y, z_obtained-z, yaw-desired_yaw, nick-desired_nick, roll-desired_roll])


        q_theta = J_inverse.dot(error_cord.transpose())
        norm_difference = LA.norm(error_cord)
        rospy.loginfo("ERROR {}".format(norm_difference))
        

        #UNCOMENT NO CHANGE

        q = q-q_theta
        #time.sleep(10)


    return q
