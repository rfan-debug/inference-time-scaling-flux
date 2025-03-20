import torch
from custom.callback_examples import get_callback_gpt_top_k
from custom import CustomFlowMatchEulerDiscreteScheduler
from custom import CustomFluxPipeline


def run_main(prompt: str):
    pipe = CustomFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    # Overriding the default scheduler with a custom scheduler that supports step_back.
    custom_fm_ec_scheduler = CustomFlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
    )
    pipe.scheduler = custom_fm_ec_scheduler

    # Set a callback to leave only top-2 latents during search.
    top_k_callback = get_callback_gpt_top_k(2)
    output = pipe(
        prompt=prompt,
        num_inference_steps=11,
        guidance_scale=3.5,
        generator=torch.Generator(device='cuda').manual_seed(5),
        height=768,
        width=1024,
        # Schedule
        search_sigma=5,
        search_delta_f=1,
        search_delta_b=3,
        search_noise_scale = 3.0,
        callback_on_search=top_k_callback,
        callback_on_search_inputs=["latents",
                                   "noise_pred",
                                   "latents_original_sample",
                                   "height",
                                   "width",
                                   "prompt"],
    ).images


    for i, img in enumerate(output):
        img.save(f"./data/flux_batch/top_k_data_{i}.png")



if __name__ == "__main__":
    # start sampling:
    prompt=("Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye "
            "texture and color. Natural lighting to capture authentic eye shine and depth. The word `FLUX` is painted "
            "over it in big, white brush strokes with visible texture.")
    run_main(prompt=prompt)