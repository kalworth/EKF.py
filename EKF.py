import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imu_data = pd.read_csv("D:/Serial Debug 2023-8-23 201358.csv")
imu_data = np.asarray(imu_data)

Q = [0.01] * 4
R = [1000.0] * 3
I = [1.0] * 4
P = [10000.0] * 4

class EKF():
    def __init__(self, period):
        self.Q_matrix  = np.diag(Q)
        self.R_matrix  = np.diag(R)
        self.I_matrix  = np.diag(I)
        self.halfT     = 1/2 * period
        self.P_matrix  = np.diag(P)
        self.A_matrix  = np.zeros([4,4])
        self.H_matrix  = np.zeros([3,4])
        self.K_matrix  = np.zeros([4,3])
        self.K_vector  = np.zeros([4,])
        self.T_vector  = np.zeros([3,])
        self.HX_vector = np.zeros([3,])
        self.Z_vector  = np.zeros([3,])
        self.pitch     = 0.0
        self.roll      = 0.0
        #self.q         = np.random.randn(4)
        self.q         = np.array([1.0,0.0,0.0,0.0])
        self.pitch_list= []
        self.roll_list = []
        self.a_pitch   = 0.0
        self.a_roll    = 0.0
        self.a_pitch_list = []
        self.a_roll_list = []

    def normalizeQuternion(self, q: np.ndarray):

        norm = np.linalg.norm(q,2)

        norm_q = q/norm

        return  norm_q

    def priori(self, gx, gy, gz):

        gx_ = gx * self.halfT
        gy_ = gy * self.halfT
        gz_ = gz * self.halfT

        self.A_matrix = np.array([
            [1, -gx_, -gy_, -gz_],
            [gx_, 1,   gz_, -gy_],
            [gy_, -gz_, 1,   gx_],
            [gz_, gy_, -gx_,   1]
        ])

        self.q = np.dot(self.A_matrix,self.q)

    def cal_P_matrix(self):
        self.P_matrix = np.dot(np.dot(self.A_matrix,self.P_matrix),np.transpose(self.A_matrix)) + self.Q_matrix

    def cal_HX_vector(self, q,ax, ay, az):
        self.H_matrix = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [2*q[0], -2*q[1], -2*q[2], 2*q[3]]
        ])
        self.HX_vector = np.array([
            2*q[1]*q[3] - 2*q[0]*q[2],
            2*q[2]*q[3] + 2*q[0]*q[1],
            1 - 2*(q[1]**2) - 2*(q[2]**2)
        ])

        self.Z_vector = np.array([ax,ay,az])

    def cal_K_matrix(self):
        self.K_matrix = np.dot(
            np.dot(self.P_matrix,np.transpose(self.H_matrix)),
            np.linalg.inv(
                np.dot(
                    np.dot(
                        self.H_matrix,self.P_matrix
                    ),np.transpose(self.H_matrix)
                ) + self.R_matrix
            )
        )

    def posterior(self):
        #  原来是self.q_k 改正为 self.q
        self.q = self.q + np.dot(
            self.K_matrix,
            self.Z_vector - self.HX_vector
        )

    def update_P_matrix(self):
        self.P_matrix = np.dot(
            self.I_matrix - np.dot(self.K_matrix,self.H_matrix),
            self.P_matrix
        )

    def Quternion2Angle(self,q):
        self.roll = -np.arcsin(2*(q[0]*q[2] - q[1]*q[3])) * 57.3
        self.pitch = np.arctan2((2*(q[0]*q[1] + q[1]*q[3])), (2*(q[0]*q[0] + q[3]*q[3]) - 1.0)) * 57.3
        self.pitch_list.append(self.pitch)
        self.roll_list.append(self.roll)

    def plot_angle(self, angle_list, color, angle_name):
         x = np.arange(0, len(angle_list), 1)
         y = np.array(angle_list)
         plt.plot(x,y,color=color,linewidth=1,label=angle_name)

    def cal_a_angle(self, ax, ay, az):
        self.a_pitch = np.arctan2(ay,az) * 57.3
        self.a_roll = np.arctan(ax/np.sqrt(ay*ay+az*az)) * 57.3
        self.a_pitch_list.append(self.a_pitch)
        self.a_roll_list.append(self.a_roll)


    def EKF_update(self,imu_sorce_data):
        for i in range(2):
            for data in imu_sorce_data:
                self.q = self.normalizeQuternion(self.q)
                self.priori(data[3], data[4], data[5])
                self.q = self.normalizeQuternion(self.q)
                self.cal_P_matrix()
                self.cal_HX_vector(self.q, data[0], data[1], data[2])
                self.cal_K_matrix()
                self.posterior()
                self.update_P_matrix()
                self.Quternion2Angle(self.q)
                self.cal_a_angle(data[0],data[1],data[2])

        plt.figure()  # 画布尺寸默认
        self.plot_angle(self.pitch_list,'red','pitch')
        self.plot_angle(self.a_pitch_list, 'blue', 'roll')
        plt.legend(['EKF_pitch','acc_pitch'], loc='best')
        plt.figure()  # 画布尺寸默认
        self.plot_angle(self.roll_list,'green','pitch')
        self.plot_angle(self.a_roll_list, 'black', 'roll')
        plt.legend(['EKF_roll','acc_roll'], loc='best')
        plt.show()

if __name__ == "__main__":
    EKF_mode = EKF(period=0.0001)
    EKF_mode.EKF_update(imu_data)

