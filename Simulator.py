import numpy as np
from Neuron import LIFNeuron
from Synapse import Synapse

class SNNSimulator:
    """
    Spiking Neural Network (SNN) 시뮬레이터
    전체 신경망의 시간 진행을 제어합니다.
    """
    
    def __init__(self, num_neurons):
        """
        시뮬레이터 초기화
        
        Args:
            num_neurons (int): 신경망 뉴런 개수
        """
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        self.synapses = []
        self.time_step = 0
        
        # 로깅
        self.spike_log = {i: [] for i in range(num_neurons)}
        self.potential_log = {i: [] for i in range(num_neurons)}
    
    def add_synapse(self, pre_idx, post_idx, initial_weight=0.5, learning_rate=0.01):
        """
        시냅스 추가
        
        Args:
            pre_idx (int): 전시냅스 뉴런 인덱스
            post_idx (int): 후시냅스 뉴런 인덱스
            initial_weight (float): 초기 가중치
            learning_rate (float): 학습률
        """
        synapse = Synapse(pre_idx, post_idx, initial_weight, learning_rate)
        self.synapses.append(synapse)
    
    def encode_input(self, data, input_neuron_indices=None):
        """
        입력 데이터를 스파이크로 인코딩
        
        Rate Coding: 데이터 값이 클수록 높은 스파이크 빈도
        
        Args:
            data (list/array): 입력 데이터
            input_neuron_indices (list): 입력을 받는 뉴런 인덱스
        
        Returns:
            list: 각 뉴런별 입력 전류
        """
        if input_neuron_indices is None:
            input_neuron_indices = list(range(len(data)))
        
        input_currents = np.zeros(self.num_neurons)
        for i, neuron_idx in enumerate(input_neuron_indices):
            if i < len(data):
                input_currents[neuron_idx] = data[i]
        
        return input_currents
    
    def step(self, input_currents):
        """
        한 시간 스텝 진행
        
        Args:
            input_currents (array): 각 뉴런에 대한 입력 전류
        """
        # 1. 시냅스 신호 계산
        synaptic_inputs = np.zeros(self.num_neurons)
        
        for synapse in self.synapses:
            if self.neurons[synapse.pre_idx].get_spike():
                signal = synapse.get_output(True)
                synaptic_inputs[synapse.post_idx] += signal
        
        # 2. 각 뉴런 업데이트
        for i, neuron in enumerate(self.neurons):
            total_current = input_currents[i] + synaptic_inputs[i]
            neuron.update(total_current, time_step=self.time_step)
            
            # 로깅
            self.spike_log[i].append(1 if neuron.get_spike() else 0)
            self.potential_log[i].append(neuron.membrane_potential)
        
        # 3. STDP 업데이트
        for synapse in self.synapses:
            pre_neuron = self.neurons[synapse.pre_idx]
            post_neuron = self.neurons[synapse.post_idx]
            
            if pre_neuron.get_spike():
                synapse.pre_spike_time = self.time_step
            if post_neuron.get_spike():
                synapse.post_spike_time = self.time_step
                
                if synapse.pre_spike_time is not None:
                    delta_t = self.time_step - synapse.pre_spike_time
                    if delta_t < 100:  # STDP 시간 윈도우
                        synapse.update_weight(synapse.pre_spike_time, self.time_step)
        
        self.time_step += 1
    
    def run(self, input_sequence, num_steps=None):
        """
        시뮬레이션 실행
        
        Args:
            input_sequence (list): 시간별 입력 시퀀스
            num_steps (int): 시뮬레이션 스텝 수
        """
        if num_steps is None:
            num_steps = len(input_sequence)
        
        for step in range(num_steps):
            if step < len(input_sequence):
                input_currents = self.encode_input(input_sequence[step])
            else:
                input_currents = np.zeros(self.num_neurons)
            
            self.step(input_currents)
    
    def get_spike_log(self):
        """스파이크 로그 반환"""
        return self.spike_log
    
    def get_potential_log(self):
        """막전위 로그 반환"""
        return self.potential_log
    
    def get_synapse_weights(self):
        """현재 모든 시냅스 가중치 반환"""
        weights = {}
        for i, synapse in enumerate(self.synapses):
            weights[f"synapse_{i}"] = {
                'pre': synapse.pre_idx,
                'post': synapse.post_idx,
                'weight': synapse.weight
            }
        return weights
    
    def reset(self):
        """시뮬레이터 초기화"""
        for neuron in self.neurons:
            neuron.reset()
        self.time_step = 0
        self.spike_log = {i: [] for i in range(self.num_neurons)}
        self.potential_log = {i: [] for i in range(self.num_neurons)}
