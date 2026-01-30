import numpy as np 
def _normalize(value, key, norm_limits):
    min_v, max_v = norm_limits[key]
    # Clipa para garantir que não passe dos limites e estrague a rede
    value = np.clip(value, min_v, max_v)
    return 2 * (value - min_v) / (max_v - min_v) - 1

    # Função auxiliar para desnormalizar (Ação do Agente -> Física)
def _denormalize_action(action_norm, norm_limits):
    # O agente entrega [-1, 1], convertemos para [u_min, u_max]
    min_v, max_v = norm_limits['u']
    # Fórmula inversa
    action_phys = 0.5 * (action_norm + 1) * (max_v - min_v) + min_v
    return action_phys