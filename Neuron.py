import numpy as np

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) 뉴런 모델
    생물학적 뉴런의 동작을 시뮬레이션합니다.
    """
    
    def __init__(self, tau_m=20.0, threshold=1.0, reset_potential=0.0, refractory_period=5):
        """
        LIF 뉴런 초기화
        
        Args:
            tau_m (float): 막시정수 (ms) - 전기 신호 감쇠 속도
            threshold (float): 스파이크 발생 임계값
            reset_potential (float): 리셋 후 막전위
            refractory_period (int): 불응기 (time-steps)
        """
        self.tau_m = tau_m
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.refractory_period = refractory_period
        
        # 상태 변수
        self.membrane_potential = reset_potential
        self.spike = False
        self.refractory_count = 0
        
        # 기록용
        self.potential_history = []
        self.spike_times = []
    
    def update(self, input_current, dt=1.0, time_step=0):
        """
        LIF 뉴런 상태 업데이트
        
        미분방정식: tau_m * dV/dt = -(V - V_rest) + R*I
        오일러 방법으로 수치 적분
        
        Args:
            input_current (float): 입력 전류
            dt (float): 시간 간격
            time_step (int): 현재 시간 단계
        """
        self.spike = False
        
        # 불응기 처리
        if self.refractory_count > 0:
            self.refractory_count -= 1
            self.membrane_potential = self.reset_potential
        else:
            # LIF 미분방정식 적분
            decay = np.exp(-dt / self.tau_m)
            self.membrane_potential = decay * self.membrane_potential + (1 - decay) * input_current
            
            # 임계값 판정
            if self.membrane_potential >= self.threshold:
                self.spike = True
                self.spike_times.append(time_step)
                self.membrane_potential = self.reset_potential
                self.refractory_count = self.refractory_period
        
        self.potential_history.append(self.membrane_potential)
    
    def get_spike(self):
        """현재 시간스텝에서 스파이크 발생 여부 반환"""
        return self.spike
    
    def reset(self):
        """뉴런 초기화"""
        self.membrane_potential = self.reset_potential
        self.spike = False
        self.refractory_count = 0
        self.potential_history = []
        self.spike_times = []
