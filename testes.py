# Importar a classe (assumindo que está em utils.buffer)
from utils.buffer import BufferMADDPGPrio
import numpy as np

# 1. Configuração
CAPACITY = 10
BATCH_SIZE = 4
STATE_DIM = 4
ACTION_DIM = 1
NUM_AGENTS = 2
buffer_prio = BufferMADDPGPrio(CAPACITY, BATCH_SIZE, STATE_DIM, ACTION_DIM, NUM_AGENTS)

# 2. Registrar 10 transições (todas com prioridade máxima inicial)
for i in range(CAPACITY):
    obs = np.random.rand(NUM_AGENTS, STATE_DIM)
    actions = np.random.rand(NUM_AGENTS, ACTION_DIM)
    reward = np.random.rand(NUM_AGENTS, 1)
    next_obs = np.random.rand(NUM_AGENTS, STATE_DIM)
    buffer_prio.record((obs, actions, reward, next_obs))

# 3. Atualizar Prioridades Seletivamente
# Índices 1 e 5 (altíssima prioridade)
buffer_prio.update_priority([1, 5], np.array([1000.0, 1000.0]))
# Índices 2 e 8 (baixa prioridade)
buffer_prio.update_priority([2, 8], np.array([0.1, 0.1]))


# 4. Amostrar e Contar
high_prio_count = 0
total_samples = 5 
print("\n--- Teste de Amostragem ---\n")

for _ in range(total_samples):
    # s, a, r, s_next, indices, weights
    _, _, _, _, indices, weights = buffer_prio.sample_batch()
    
    # 5. Verifique Pesos IS
    if 1 in indices or 5 in indices:
        high_prio_count += 1
        
    # Exemplo: verifique o peso IS da primeira amostra
    
    print(f"Pesos IS na primeira amostra: {weights.flatten()}")
        
print(f"\nTotal de amostras: {total_samples * BATCH_SIZE}")
print(f"Transições com alta prioridade (índices 1 ou 5) amostradas: {high_prio_count} vezes (espera-se alta frequência)")

# Continuação do teste
# ... (Rodar o setup acima)

# Amostragem para verificar a normalização
weights_list = []
for _ in range(5):
    _, _, _, _, _, weights = buffer_prio.sample_batch()
    weights_list.append(weights.max())

print("\n--- Teste de Normalização IS ---\n")
print(f"Máximo Peso IS em 5 amostras: {weights_list}")
# Todos os valores devem ser 1.0 (ou muito próximos)