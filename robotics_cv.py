import numpy as np
import matplotlib.pyplot as plt
import time
from math import cos, sin, pi, sqrt

class DWA_Planner:
    def __init__(self, max_speed=1.5, max_yaw_rate=40.0 * pi / 180.0):
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.v_resolution = 0.1
        self.yaw_rate_resolution = 1.0 * pi / 180.0
        self.dt = 0.1  # Zaman adımı
        self.predict_time = 3.0  # Planlama ufku

    def calculate_dynamic_window(self, v, yaw_rate):
        """Robotun anlık dinamik hız penceresini hesaplar."""
        return [0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]

    def motion_model(self, state, control):
        """Robotun hareket denklemi."""
        x, y, yaw, v, omega = state
        v_c, omega_c = control
        
        yaw += omega_c * self.dt
        x += v_c * cos(yaw) * self.dt
        y += v_c * sin(yaw) * self.dt
        
        return [x, y, yaw, v_c, omega_c]

    def calc_control_and_trajectory(self, state, goal, obstacles):
        """Optimal kontrol komutunu seçer (DWA)."""
        dw = self.calculate_dynamic_window(state[3], state[4])
        best_u = [0.0, 0.0]
        min_cost = float("inf")

        # Hız uzayını tara (Velocity Space Search)
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for yaw_rate in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                
                # Simülasyon yap (Tahmin edilen yörünge)
                u = [v, yaw_rate]
                cost = self.calculate_cost(state, u, goal, obstacles)
                
                if cost < min_cost:
                    min_cost = cost
                    best_u = u
        
        return best_u

    def calculate_cost(self, state, u, goal, obstacles):
        """Maliyet fonksiyonu: Hedefe yakınlık + Engellerden uzaklık."""
        next_state = self.motion_model(state, u)
        
        # 1. Hedefe olan mesafe maliyeti
        dist_to_goal = sqrt((next_state[0] - goal[0])**2 + (next_state[1] - goal[1])**2)
        
        # 2. Engele olan mesafe maliyeti (Basitleştirilmiş)
        min_dist_to_obj = float("inf")
        for ob in obstacles:
            d = sqrt((next_state[0] - ob[0])**2 + (next_state[1] - ob[1])**2)
            if d < min_dist_to_obj:
                min_dist_to_obj = d
        
        # Engel çok yakınsa maliyeti sonsuz yap
        if min_dist_to_obj < 0.5:
            return float("inf")
            
        return dist_to_goal + (1.0 / min_dist_to_obj)

if __name__ == "__main__":
    print("Sidal AI - Robotics & Computer Vision Controller v2.0")
    
    # Başlangıç durumu: [x, y, yaw, v, omega]
    robot_state = [0.0, 0.0, 0.0, 0.0, 0.0]
    goal = [10.0, 10.0]
    obstacles = np.array([[3.0, 3.0], [5.0, 6.0], [8.0, 7.0]])
    
    planner = DWA_Planner()
    
    print(f"Başlangıç: {robot_state[:2]} Hedef: {goal}")
    
    # 50 adım simüle et
    for i in range(50):
        u = planner.calc_control_and_trajectory(robot_state, goal, obstacles)
        robot_state = planner.motion_model(robot_state, u)
        
        if i % 10 == 0:
            print(f"Adım {i}: Robot Konumu: {np.round(robot_state[:2], 2)} Hız: {np.round(u, 2)}")
            
        if sqrt((robot_state[0] - goal[0])**2 + (robot_state[1] - goal[1])**2) < 0.5:
            print("Hedefe Ulaşıldı!")
            break
