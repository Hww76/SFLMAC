import numpy as np

# Default wireless settings (can be tuned if needed)
DEFAULT_W = 10 ** 7  # bandwidth
DEFAULT_N = 3.981 * 10 ** (-21)  # noise power spectral density
DEFAULT_TX_POWER = 0.2  # transmit power scalar used in rate calculation
DEFAULT_CHANNEL_Num = 2  # channel gain scalar used in rate calculation


def generate_computing(user_num: int):
    """Generate computing capabilities for each client (FLOPs)."""
    log_min, log_max = 7, 9  # 对数空间范围：10^7 到 10^9，提供100倍算力差异
    log_values = np.random.uniform(log_min, log_max, user_num)  # 在对数空间均匀分布
    return 10 ** log_values


def generate_position(user_num: int):
    """Generate client positions (km), used for path loss."""
    return np.random.uniform(0.4, 0.6, user_num)


def compute_client_rates(clients_position, w: float = DEFAULT_W, N: float = DEFAULT_N, tx_power: float = DEFAULT_TX_POWER):
    """Compute communication rates for each client based on positions.

    Args:
        clients_position: iterable of client positions (km)
        w: bandwidth
        N: noise power spectral density
        tx_power: transmit power (scalar in original code 0.2)
    Returns:
        list of rates per client
    """
    rates = []
    for pos in clients_position:
        path_loss = 128.1 + 37.6 * np.log10(pos)
        h = 10 ** (-path_loss / 10)
        rates.append(w * np.log2(1 + (tx_power * h / (w * N))))
    return rates

def compute_server_rate(w: float = DEFAULT_W, N: float = DEFAULT_N, tx_power: float = DEFAULT_TX_POWER, DEFAULT_CHANNEL_Num: int = DEFAULT_CHANNEL_Num, num_clients: int =1):
    """Compute communication rate for the server.

    Args:
        w: bandwidth
        N: noise power spectral density
        tx_power: transmit power (scalar in original code 1)

    """
    pass