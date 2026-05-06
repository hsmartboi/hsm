import numpy as np

class Synapse:
    """
    시냅스 모듈 - 뉴런 간 연결
    STDP (Spike-Timing-Dependent Plasticity) 기반 학습
    """
    
    def __init__(self, pre_neuron_idx, post_neuron_idx, initial_weight=0.5, 
                 learning_rate=0.01, tau_stdp=20.0):
        """
        시냅스 초기화
        
        Args:
            pre_neuron_idx (int): 전시냅스 뉴런 인덱스
            post_neuron_idx (int): 후시냅스 뉴런 인덱스
            initial_weight (float): 초기 가중치
            learning_rate (float): STDP 학습률
            tau_stdp (float): STDP 시상수 (ms)
        """
        self.pre_idx = pre_neuron_idx
        self.post_idx = post_neuron_idx
        self.weight = initial_weight
        self.learning_rate = learning_rate
        self.tau_stdp = tau_stdp
        
        # STDP를 위한 추적
        self.pre_spike_time = None
        self.post_spike_time = None
        
        # 기록용
        self.weight_history = [initial_weight]
    
    def stdp_rule(self, delta_t):
        """
        STDP 학습 규칙
        
        Δw = A_plus * exp(-|delta_t| / tau) if delta_t > 0 (post-pre)
        Δw = -A_minus * exp(-|delta_t| / tau) if delta_t < 0 (pre-post)
        
        Args:
            delta_t (int): 후시냅스 스파이크 시간 - 전시냅스 스파이크 시간
        
        Returns:
            float: 가중치 변화량
        """
        if delta_t == 0:
            return 0
        
        A_plus = 0.01   # LTP (Long-Term Potentiation) 진폭
        A_minus = 0.01  # LTD (Long-Term Depression) 진폭
        
        if delta_t > 0:
            # 후시냅스가 먼저 스파이크 → LTP (가중치 증가)
            delta_w = A_plus * np.exp(-delta_t / self.tau_stdp)
        else:
            # 전시냅스가 먼저 스파이크 → LTD (가중치 감소)
            delta_w = -A_minus * np.exp(delta_t / self.tau_stdp)
        
        return delta_w * self.learning_rate
    
    def update_weight(self, pre_spike_time, post_spike_time):
        """
        스파이크 타이밍에 따라 가중치 업데이트
        
        Args:
            pre_spike_time (int or None): 전시냅스 스파이크 시간
            post_spike_time (int or None): 후시냅스 스파이크 시간
        """
        if pre_spike_time is not None and post_spike_time is not None:
            delta_t = post_spike_time - pre_spike_time
            delta_w = self.stdp_rule(delta_t)
            
            # 가중치 업데이트 (0과 1 사이로 제한)
            self.weight = np.clip(self.weight + delta_w, 0, 1)
            self.weight_history.append(self.weight)
    
    def get_output(self, pre_spike):
        """
        전시냅스 입력에 따른 출력 (전기 신호)
        
        Args:
            pre_spike (bool): 전시냅스 뉴런의 스파이크 여부
        
        Returns:
            float: 가중 신호
        """
        if pre_spike:
            return self.weight
        return 0.0
    
    def get_connection_info(self):
        """연결 정보 반환"""
        return {
            'pre_idx': self.pre_idx,
            'post_idx': self.post_idx,
            'weight': self.weight
        }
