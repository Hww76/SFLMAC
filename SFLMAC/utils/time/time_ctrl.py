from config.network import generate_computing, generate_position, compute_client_rates, compute_server_rate
from config import training as training_config

def set_rates(num_client=20):
    clients_position = generate_position(num_client)
    return compute_client_rates(clients_position)

def set_computing(num_client=20):
    return generate_computing(num_client)

def set_client_LBFSL(num_client, data_sizes, local_epoch, minibatch, benchmark='alexnet_cifar10',
                     forward_flops_by_splitpoint_list_self=None,
                     backward_flops_by_splitpoint_list_self=None,
                     forward_flops_by_splitpoint_list_helper=None,
                     backward_flops_by_splitpoint_list_helper=None,
                     parameter_bits_by_splitpoint_list_self=None,
                     parameter_bits_by_splitpoint_list_helper=None,
                     split_point=training_config.ALEXNET_MAX_SPLIT_POINT):
    clients_computing = set_computing(num_client)
    clients_rates = set_rates(num_client)


    from config.client import Client_LBFSL
    clients = []
    for i in range(num_client):
        # 上传和下载速率通常假设为相同的通信速率
        # 如果需要不同的上传和下载速率，可以根据实际应用进行调整
        rate = clients_rates[i]
        client = Client_LBFSL(
            data_sizes=data_sizes[i],
            local_epoch=local_epoch,
            minibatch=minibatch,
            computing=clients_computing[i],
            up_rate=rate,
            down_rate=rate,
            split_point=split_point,
            forward_flops=forward_flops_by_splitpoint_list_self[split_point],
            backward_flops=backward_flops_by_splitpoint_list_self[split_point],
            parameter_bits=parameter_bits_by_splitpoint_list_self[split_point],
            activate_shape=None  # 需要根据实际模型和分割点设置激活值形状
        )
        clients.append(client)
    return clients
