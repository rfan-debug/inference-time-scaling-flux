


# Scheduler forward and backforward steps for search algorithm in
# https://arxiv.org/html/2501.09732v1
def get_search_steps(
        num_inference_steps: int,
        search_sigma: int,
        search_delta_f: int,
        search_delta_b: int,
):
    current_step = search_sigma
    step_indexes = [current_step]
    while current_step < num_inference_steps:
        for _ in range(search_delta_f):
            current_step -= 1
            step_indexes.append(current_step)
        for _ in range(search_delta_b):
            current_step += 1
            step_indexes.append(current_step)

    assert current_step == num_inference_steps, "The final current_step must be precisely at `num_inference_steps`."
    return step_indexes[:-1] # drop the last element that is equal to the num_inference_steps
