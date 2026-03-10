import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

class RobotConfig:
    """Robotun fiziksel ve hareket sınırlarını tanımlayan konfigürasyon."""
    MAX_SPEED = 1.2        # [m/s]
    MIN_SPEED = -0.3       # [m/s]
    MAX_YAW_RATE = 45.0 * math.pi / 180.0
    MAX_ACCEL = 0.3        # [m/ss]
    MAX_DELTA_YAW_RATE = 50.0 * math.pi / 180.0
    DT = 0.1               # [s] Zaman adımı
    PREDICT_TIME = 3.5     # [s] Planlama ufku
    ROBOT_RADIUS = 0.4     # [m]

class AdvancedDWAPlanner:
    """
    Dinamik Pencere Yaklaşımı (DWA) tabanlı gelişmiş otonom navigasyon sistemi.
    Robotun hız uzayında arama yaparak engellerden kaçınan optimal rotayı belirler.
    """
    def __init__(self, config=RobotConfig()):
        self.config = config

    def motion_model(self, x: np.ndarray, u: List[float]) -> np.ndarray:
        """Robotun diferansiyel sürüş kinematik modeline göre durum güncellemesi."""
        x[2] += u[1] * self.config.DT
        x[0] += u[0] * math.cos(x[2]) * self.config.DT
        x[1] += u[0] * math.sin(x[2]) * self.config.DT
        x[3] = u[0]  # v
        x[4] = u[1]  # yaw_rate
        return x

    def calc_dynamic_window(self, x: np.ndarray) -> List[float]:
        """Robotun ivme ve hız sınırlarına göre erişilebilir hız penceresini hesaplar."""
        vs = [self.config.MIN_SPEED, self.config.MAX_SPEED, 
              -self.config.MAX_YAW_RATE, self.config.MAX_YAW_RATE]
        
        vd = [x[3] - self.config.MAX_ACCEL * self.config.DT,
              x[3] + self.config.MAX_ACCEL * self.config.DT,
              x[4] - self.config.MAX_DELTA_YAW_RATE * self.config.DT,
              x[4] + self.config.MAX_DELTA_YAW_RATE * self.config.DT]
        
        return [max(vs[0], vd[0]), min(vs[1], vd[1]), 
                max(vs[2], vd[2]), min(vs[3], vd[3])]

    def predict_trajectory(self, x_init: np.ndarray, v: float, y_rate: float) -> np.ndarray:
        """Belirli bir hız çifti için robotun gelecekteki yörüngesini tahmin eder."""
        x = np.array(x_init)
        traj = np.array(x)
        time = 0
        while time <= self.config.PREDICT_TIME:
            x = self.motion_model(x, [v, y_rate])
            traj = np.vstack((traj, x))
            time += self.config.DT
        return traj

    def evaluate_cost(self, traj: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> float:
        """Hedef yakınlığı, hız ve engellerden uzaklık temelli maliyet fonksiyonu."""
        # 1. Hedefe olan uzaklık (Hedef yönelimi)
        dist_to_goal = np.linalg.norm(traj[-1, :2] - goal)
        
        # 2. Hız maliyeti (Hızlı gitmeyi teşvik et)
        speed_cost = self.config.MAX_SPEED - traj[-1, 3]
        
        # 3. Engel maliyeti (Çarpışma kontrolü)
        min_dist = float("inf")
        for ob in obstacles:
            dist_to_ob = np.linalg.norm(traj[:, :2] - ob, axis=1)
            if np.min(dist_to_ob) < self.config.ROBOT_RADIUS:
                return float("inf") # Çarpışma riski
            if np.min(dist_to_ob) < min_dist:
                min_dist = np.min(dist_to_ob)
        
        return 1.5 * dist_to_goal + 1.0 * speed_cost + 1.2 * (1.0 / min_dist)

if __name__ == "__main__":
    print("Sidal AI - Physics-Based Robotics Controller v3.0 Initialized.")
    planner = AdvancedDWAPlanner()
    # Örnek Durum: [x, y, yaw, v, omega]
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0])
    obstacles = np.array([[3.0, 3.0], [5.0, 6.0], [8.0, 2.0]])
    
    # Optimizasyon döngüsü (Simülasyon örneği)
    for _ in range(20):
        dw = planner.calc_dynamic_window(state)
        # Basitleştirilmiş en iyi hız seçimi
        state = planner.motion_model(state, [dw[1], 0.1])
        print(f"Robot Durumu: {np.round(state[:2], 2)}")
