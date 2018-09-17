#! /usr/bin/env python3

"""
    ESTEBAN CABEZOS, HECTOR
    hectorec@kth.se
"""
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from numpy import cos
from numpy import sin


L0 = 0.07
L1 = 0.3
L2 = 0.35
# For KUKA
L = 0.4
M = 0.39
BIAS1 = 0.311
BIAS2 = 0.078
PI = 3.14159265359
def compute_DH(joint_positions):
    # GOOD
    DH = np.array([[0.5*PI, 0, 0, joint_positions[0]],[-0.5*PI, 0, 0, joint_positions[1]], [-0.5*PI, L, 0, joint_positions[2]],[0.5*PI, 0, 0, joint_positions[3]],[0.5*PI, M, 0, joint_positions[4]],[-0.5*PI, 0, 0, joint_positions[5]],[0, 0, 0, joint_positions[6]]])
    return DH

def compute_T(alpha, d, r, theta):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)], [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)], [0, np.sin(alpha), np.cos(alpha), d], [0, 0, 0, 1]])

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
    q2 = np.pi - theta_3
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

def obtain_jacobian_matrix(L, M, q1, q2, q3, q4, q5, q6, q7 ):
    # GOOD
    Jacobian_matrix_return = np.matrix([[L*sin(q1)*sin(q2) - M*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)), -cos(q1)*(M*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)) + L*cos(q2)), -M*sin(q4)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)), M*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)) + M*sin(q2)*sin(q3)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)),0,0,0],
    [ - M*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)) - L*cos(q1)*sin(q2), -sin(q1)*(M*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)) + L*cos(q2)),  M*sin(q4)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)),M*(sin(q1)*sin(q2)*sin(q4) + cos(q1)*cos(q4)*sin(q3) + cos(q2)*cos(q3)*cos(q4)*sin(q1)),0,0,0],
    [0,M*cos(q2)*cos(q3)*sin(q4) - M*cos(q4)*sin(q2) - L*sin(q2),-M*sin(q2)*sin(q3)*sin(q4),M*cos(q3)*cos(q4)*sin(q2) - M*cos(q2)*sin(q4),0,0,0],
    [0,sin(q1),-cos(q1)*sin(q2),- cos(q3)*sin(q1) - cos(q1)*cos(q2)*sin(q3), - sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*cos(q4)*sin(q2), cos(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)) - sin(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)), sin(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))) - cos(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2))],
    [0,-cos(q1),-sin(q1)*sin(q2),cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3),   sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2), sin(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)), cos(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)) - sin(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))],
    [1,0,cos(q2),-sin(q2)*sin(q3),cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4),cos(q5)*sin(q2)*sin(q3) - sin(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)),sin(q6)*(cos(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) + sin(q2)*sin(q3)*sin(q5)) + cos(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4))]])
    
    total_T = np.matrix([[sin(q7)*(sin(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))) - cos(q7)*(sin(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)) + cos(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)))),   sin(q7)*(sin(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)) + cos(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)))) + cos(q7)*(sin(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))), sin(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))) - cos(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)), - M*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)) - L*cos(q1)*sin(q2)],
    [ cos(q7)*(sin(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)) + cos(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))) - sin(q7)*(sin(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))), - sin(q7)*(sin(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)) + cos(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))) - cos(q7)*(sin(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))), cos(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)) - sin(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))),   M*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)) - L*sin(q1)*sin(q2)],
    [sin(q7)*(sin(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) - cos(q5)*sin(q2)*sin(q3)) - cos(q7)*(cos(q6)*(cos(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) + sin(q2)*sin(q3)*sin(q5)) - sin(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4))),cos(q7)*(sin(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) - cos(q5)*sin(q2)*sin(q3)) + sin(q7)*(cos(q6)*(cos(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) + sin(q2)*sin(q3)*sin(q5)) - sin(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4))),sin(q6)*(cos(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) + sin(q2)*sin(q3)*sin(q5)) + cos(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)),M*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)) + L*cos(q2)],
    [0,0,0,1]])

    return Jacobian_matrix_return, total_T

def obtain_inverse_jacobian(jacobian_matrix):
    J_inverse = jacobian_matrix.T*inv(jacobian_matrix*jacobian_matrix.T);
    return J_inverse

def compute_euler_angles(result_coordinates):
    second_angle = np.arctan2(-1*result_coordinates[2,0], np.sqrt(np.power(result_coordinates[0,0],2)+np.power(result_coordinates[1,0],2)))
    first_angle = np.arctan2(result_coordinates[1,0]/cos(second_angle), result_coordinates[0,0]/cos(second_angle))
    third_angle = np.arctan2(result_coordinates[2,1]/cos(second_angle), result_coordinates[2,2]/cos(second_angle))
    return first_angle, second_angle, third_angle

def clean_R(R):
    new_R = np.zeros(shape=(3,3))
    for i in range(3):
        for j in range(3):
            new_R[i,j] = R[i][j]

    return new_R

def end_effector_respect_itself():
    end_effector = np.array([[0],[0],[BIAS2],[1]])
    return end_effector


def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions #it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """
    
    norm_difference = 10

    while norm_difference > 0.02: #0.04:

        dh_matrix = compute_DH(q)
        array_matrix = np.zeros(shape=(7,4,4))
        auxiliar_mat = np.zeros(shape=(7,4,4))

        Jacobian_matrix_n, total_array = obtain_jacobian_matrix(L, M, q[0], q[1],q[2],q[3],q[4],q[5],q[6])

        J_inverse = obtain_inverse_jacobian(Jacobian_matrix_n)


        end_effector = end_effector_respect_itself()
        #nick = np.arctan2(-1*total_array[2,0], np.sqrt(np.power(total_array[0,0],2)+np.power(total_array[1,0],2)))
        """if(nick>0.999*PI):
                                    nick = 0
                                yaw = np.arctan2(total_array[1,0]/cos(nick), total_array[0,0]/cos(nick))
                        
                                if(yaw>0.999*PI):
                                    yaw = 0
                        
                                roll = np.arctan2(total_array[2,1]/cos(nick), total_array[2,2]/cos(nick))
                                if (roll>0.999*PI):
                                    roll = 0"""

        end_vector_ground = total_array.dot(end_effector)
        """x_obtained = end_vector_ground[0]
                                y_obtained = end_vector_ground[1]
                                z_obtained = end_vector_ground[2] + BIAS1"""
        end_vector_ground[2]=end_vector_ground[2]+BIAS1

        "Page 139"
        total_array = np.array(total_array)
        R = np.array(R)
        ne = total_array[:3,0]
        se = total_array[:3,1]
        ae = total_array[:3,2]
        nd = R[:,0]
        sd = R[:,1]
        ad = R[:,2]
        angle_err = 1/2*(np.cross(ne, nd)+np.cross(se, sd)+np.cross(ae, ad))

        # GOOD UNTIL HERE
        R = clean_R(R)

        #desired_yaw, desired_nick, desired_roll = compute_euler_angles(R)
        desired_nick = np.arctan2(-1*R[2,0], np.sqrt(np.power(R[0,0],2)+np.power(R[1,0],2)))
        desired_yaw = np.arctan2(R[1,0]/cos(desired_nick), R[0,0]/cos(desired_nick))
        desired_roll = np.arctan2(R[2,1]/cos(desired_nick), R[2,2]/cos(desired_nick))

        error_cord = np.array([end_vector_ground[0]-x, end_vector_ground[1]-y, end_vector_ground[2]-z])#, yaw-desired_yaw, nick-desired_nick, roll-desired_roll])

        error_cord = np.array([end_vector_ground[0]-x, end_vector_ground[1]-y, end_vector_ground[2]-z, angle_err[0], angle_err[1], angle_err[2]])
        #error_cord = np.array(np.concatenate(error_cord,angle_err))


        q_theta = J_inverse.dot(np.transpose(error_cord))

        norm_difference = LA.norm(error_cord)

        for i in range(7):
            q[i] = q[i]-float(q_theta[0,i])


    return q
