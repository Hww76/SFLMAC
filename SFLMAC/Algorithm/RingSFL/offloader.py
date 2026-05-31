from typing import Dict, List


class RingTaskOffloader:
    def __init__(self, client_compute: Dict[int, float], ring_size: int = 5):
        self.client_compute = dict(client_compute)
        requested_ring_size = int(ring_size)
        if requested_ring_size <= 0:
            raise ValueError("ring_size must be > 0")
        max_clients = len(self.client_compute)
        if max_clients <= 0:
            raise ValueError("client_compute must not be empty")

        # If requested ring_size is larger than the client count, use all clients in one ring.
        self.ring_size = min(requested_ring_size, max_clients)
        self._cursor = 0

    def select_clients_for_round(self, all_client_ids: List[int]) -> List[int]:
        n = len(all_client_ids)
        if n < self.ring_size:
            return []
        selected_count = n - (n % self.ring_size)
        if selected_count <= 0:
            return []

        ordered = sorted(all_client_ids)
        start = self._cursor % n
        rotated = ordered[start:] + ordered[:start]
        selected = rotated[:selected_count]
        self._cursor = (start + selected_count) % n
        return selected

    def build_rings(self, selected_client_ids: List[int]) -> List[List[int]]:
        if not selected_client_ids:
            return []
        if len(selected_client_ids) % self.ring_size != 0:
            raise ValueError("selected clients must be divisible by ring_size")

        balanced_order = self.balance_heterogeneous_clients(selected_client_ids)
        num_rings = len(selected_client_ids) // self.ring_size
        rings = [[] for _ in range(num_rings)]

        for idx, cid in enumerate(balanced_order):
            rings[idx % num_rings].append(cid)

        return rings

    def balance_heterogeneous_clients(self, selected_client_ids: List[int]) -> List[int]:
        return sorted(selected_client_ids, key=lambda cid: self.client_compute[cid], reverse=True)

    def assign_ring_propagation_lengths(self, ring: List[int], num_blocks: int) -> Dict[int, int]:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if not ring:
            return {}

        compute_values = [max(float(self.client_compute[cid]), 1e-12) for cid in ring]
        total_compute = sum(compute_values)
        ideal_lengths = [num_blocks * c / total_compute for c in compute_values]

        lengths = [max(1, int(v)) for v in ideal_lengths]
        current_sum = sum(lengths)

        if current_sum > num_blocks:
            frac_with_idx = sorted(
                [(ideal_lengths[i] - int(ideal_lengths[i]), i) for i in range(len(ring))],
                key=lambda item: item[0],
            )
            remove_need = current_sum - num_blocks
            for _, idx in frac_with_idx:
                if remove_need <= 0:
                    break
                if lengths[idx] > 1:
                    lengths[idx] -= 1
                    remove_need -= 1
            while remove_need > 0:
                for idx in range(len(lengths)):
                    if remove_need <= 0:
                        break
                    if lengths[idx] > 1:
                        lengths[idx] -= 1
                        remove_need -= 1

        elif current_sum < num_blocks:
            frac_with_idx = sorted(
                [(ideal_lengths[i] - int(ideal_lengths[i]), i) for i in range(len(ring))],
                key=lambda item: item[0],
                reverse=True,
            )
            add_need = num_blocks - current_sum
            pos = 0
            while add_need > 0:
                idx = frac_with_idx[pos % len(frac_with_idx)][1]
                lengths[idx] += 1
                add_need -= 1
                pos += 1

        return {cid: lengths[i] for i, cid in enumerate(ring)}
