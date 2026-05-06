import numpy as np
import matplotlib.pyplot as plt
from Simulator import SNNSimulator

def plot_simulation_results(simulator, title="Neuromorphic Computing Simulation"):
    """
    시뮬레이션 결과를 시각화
    
    세 개의 그래프:
    1. Spike Raster Plot - 각 뉴런의 스파이크 발생 시간
    2. Membrane Potential - 선택 뉴런의 막전위 변화
    3. Synaptic Weights - STDP에 따른 가중치 변화
    """
    spike_log = simulator.get_spike_log()
    potential_log = simulator.get_potential_log()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Spike Raster Plot
    ax1 = axes[0]
    for neuron_idx in range(simulator.num_neurons):
        spike_times = [i for i, spike in enumerate(spike_log[neuron_idx]) if spike == 1]
        ax1.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4, colors='red', linewidth=2)
    
    ax1.set_ylim(-0.5, simulator.num_neurons - 0.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Neuron Index')
    ax1.set_title('Spike Raster Plot - 각 뉴런의 스파이크 발생')
    ax1.grid(True, alpha=0.3)
    
    # 2. Membrane Potential
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, simulator.num_neurons))
    for neuron_idx in range(simulator.num_neurons):
        ax2.plot(potential_log[neuron_idx], label=f'Neuron {neuron_idx}', 
                color=colors[neuron_idx], linewidth=1.5)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.set_title('Membrane Potential - 막전위 변화')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Synaptic Weights
    ax3 = axes[2]
    synapse_weights = simulator.get_synapse_weights()
    
    if simulator.synapses:
        for i, synapse in enumerate(simulator.synapses):
            ax3.plot(synapse.weight_history, label=f'W({synapse.pre_idx}→{synapse.post_idx})', 
                    linewidth=2, marker='o', markersize=4)
        
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Synaptic Weight')
        ax3.set_title('Synaptic Weights - STDP 학습에 따른 가중치 변화')
        ax3.legend(loc='best')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No synapses defined', ha='center', va='center')
        ax3.set_title('Synaptic Weights')
    
    plt.tight_layout()
    plt.show()

def simulation_1_basic():
    """
    시뮬레이션 1: 기본 피드포워드 신경망
    입력층(2) → 은닉층(3) → 출력층(1)
    """
    print("\n" + "="*60)
    print("시뮬레이션 1: 기본 피드포워드 신경망")
    print("="*60)
    print("구조: 입력층(2) → 은닉층(3) → 출력층(1)")
    
    # 시뮬레이터 생성 (총 6개 뉴런)
    simulator = SNNSimulator(num_neurons=6)
    
    # 입력층(0,1) → 은닉층(2,3,4)
    for i in range(2):  # 입력층
        for j in range(3):  # 은닉층
            simulator.add_synapse(i, j + 2, initial_weight=0.5, learning_rate=0.01)
    
    # 은닉층(2,3,4) → 출력층(5)
    for i in range(3):  # 은닉층
        simulator.add_synapse(i + 2, 5, initial_weight=0.5, learning_rate=0.01)
    
    # 입력 시퀀스 (100 time steps)
    input_sequence = []
    for t in range(100):
        # 패턴 1: 처음 50 스텝
        if t < 50:
            input_sequence.append([0.8, 0.2])  # 입력1 높음, 입력2 낮음
        # 패턴 2: 나머지 50 스텝
        else:
            input_sequence.append([0.2, 0.8])  # 입력1 낮음, 입력2 높음
    
    print(f"\n실행 중... (100 time steps)")
    simulator.run(input_sequence, num_steps=100)
    print("✓ 시뮬레이션 완료")
    
    # 결과 분석
    spike_counts = {}
    for neuron_idx in range(simulator.num_neurons):
        spike_counts[neuron_idx] = sum(simulator.get_spike_log()[neuron_idx])
    
    print(f"\n결과 - 각 뉴런의 스파이크 발생 횟수:")
    for neuron_idx, count in spike_counts.items():
        print(f"  뉴런 {neuron_idx}: {count}회")
    
    plot_simulation_results(simulator, "시뮬레이션 1: 기본 피드포워드 신경망")

def simulation_2_stdp_learning():
    """
    시뮬레이션 2: STDP 학습 시연
    전시냅스 스파이크와 후시냅스 스파이크의 타이밍 관계 분석
    """
    print("\n" + "="*60)
    print("시뮬레이션 2: STDP 학습 - 스파이크 타이밍 의존성")
    print("="*60)
    print("목표: Pre-Post 스파이크 타이밍에 따른 가중치 변화 관찰")
    
    # 시뮬레이터 생성 (3개 뉴런)
    simulator = SNNSimulator(num_neurons=3)
    
    # 뉴런 0 → 뉴런 1 (학습 대상 1)
    simulator.add_synapse(0, 1, initial_weight=0.5, learning_rate=0.05)
    # 뉴런 0 → 뉴런 2 (학습 대상 2)
    simulator.add_synapse(0, 2, initial_weight=0.5, learning_rate=0.05)
    
    # 입력 패턴 설정
    print(f"\n입력 패턴:")
    print(f"  - 뉴런 0: 시간 10, 40, 70에서 스파이크")
    print(f"  - 뉴런 1: 시간 15, 45, 65에서 스파이크 (Post-Pre, 학습 유도)")
    print(f"  - 뉴런 2: 시간 5, 35, 75에서 스파이크 (Pre-Post, 미학습)")
    
    input_sequence = [[0, 0, 0] for _ in range(100)]
    
    # 뉴런 0: 시간 10, 40, 70에서 입력
    input_sequence[10] = [1.5, 0, 0]
    input_sequence[40] = [1.5, 0, 0]
    input_sequence[70] = [1.5, 0, 0]
    
    # 뉴런 1: 시간 15, 45에서 입력
    input_sequence[15] = [0, 1.5, 0]
    input_sequence[45] = [0, 1.5, 0]
    input_sequence[65] = [0, 1.5, 0]
    
    # 뉴런 2: 시간 5, 35, 75에서 입력
    input_sequence[5] = [0, 0, 1.5]
    input_sequence[35] = [0, 0, 1.5]
    input_sequence[75] = [0, 0, 1.5]
    
    print(f"\n실행 중... (100 time steps)")
    simulator.run(input_sequence, num_steps=100)
    print("✓ 시뮬레이션 완료")
    
    # 결과 분석
    weights = simulator.get_synapse_weights()
    print(f"\n결과 - 최종 시냅스 가중치:")
    for synapse in simulator.synapses:
        print(f"  뉴런 {synapse.pre_idx} → {synapse.post_idx}: {synapse.weight:.4f}")
        print(f"    기대값: LTP (가중치 증가) - Post-Pre 타이밍")
    
    plot_simulation_results(simulator, "시뮬레이션 2: STDP 학습")

def simulation_3_random_network():
    """
    시뮬레이션 3: 무작위 신경망
    임의의 네트워크 구조로 시뮬레이션
    """
    print("\n" + "="*60)
    print("시뮬레이션 3: 무작위 신경망")
    print("="*60)
    print("구조: 5개 뉴런의 무작위 연결")
    
    # 시뮬레이터 생성
    simulator = SNNSimulator(num_neurons=5)
    
    # 무작위 연결 생성
    np.random.seed(42)  # 재현성을 위한 시드
    connection_count = 0
    
    print(f"\n연결 구조:")
    for pre in range(5):
        for post in range(5):
            if pre != post and np.random.random() < 0.4:  # 40% 확률로 연결
                initial_weight = np.random.uniform(0.3, 0.7)
                simulator.add_synapse(pre, post, initial_weight=initial_weight, learning_rate=0.02)
                print(f"  뉴런 {pre} → {post} (W={initial_weight:.2f})")
                connection_count += 1
    
    print(f"\n총 연결 수: {connection_count}")
    
    # 포아송 입력 생성
    print(f"\n포아송 입력 생성 (firing rate=0.3)")
    input_sequence = []
    for t in range(150):
        # 각 입력에 대해 30% 확률로 스파이크
        input_currents = [1.0 if np.random.random() < 0.3 else 0 for _ in range(2)]
        input_sequence.append(input_currents + [0, 0, 0])  # 처음 2개만 입력
    
    print(f"실행 중... (150 time steps)")
    simulator.run(input_sequence, num_steps=150)
    print("✓ 시뮬레이션 완료")
    
    # 결과 분석
    spike_log = simulator.get_spike_log()
    total_spikes = {}
    for neuron_idx in range(5):
        total_spikes[neuron_idx] = sum(spike_log[neuron_idx])
    
    print(f"\n결과 - 각 뉴런의 총 스파이크 수:")
    for neuron_idx, count in sorted(total_spikes.items()):
        firing_rate = (count / 150) * 100
        print(f"  뉴런 {neuron_idx}: {count}회 (발화율: {firing_rate:.1f}%)")
    
    plot_simulation_results(simulator, "시뮬레이션 3: 무작위 신경망")

def print_menu():
    """메뉴 출력"""
    print("\n" + "="*60)
    print("뉴로모픽 컴퓨팅 시뮬레이션 프로젝트")
    print("="*60)
    print("\n실행할 시뮬레이션을 선택하세요:")
    print("  1. 기본 피드포워드 신경망")
    print("  2. STDP 학습 - 스파이크 타이밍 의존성")
    print("  3. 무작위 신경망 시뮬레이션")
    print("  4. 모두 실행")
    print("  0. 종료")
    print("="*60)

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# 뉴로모픽 컴퓨팅 시뮬레이션")
    print("# SNN (Spiking Neural Network) 시뮬레이터")
    print("# 작성자: 일산대진고등학교 2학년")
    print("#"*60)
    
    while True:
        print_menu()
        choice = input("\n선택: ").strip()
        
        if choice == '1':
            simulation_1_basic()
        elif choice == '2':
            simulation_2_stdp_learning()
        elif choice == '3':
            simulation_3_random_network()
        elif choice == '4':
            simulation_1_basic()
            simulation_2_stdp_learning()
            simulation_3_random_network()
        elif choice == '0':
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("\n⚠ 잘못된 입력입니다. 다시 선택해주세요.")
