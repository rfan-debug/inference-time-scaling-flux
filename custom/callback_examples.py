import torch
import PIL.Image
from custom import CustomFluxPipeline


def top_k_indices(lst, k):
    # sort indices of lst by their corresponding values in descending order
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return sorted_indices[:k]

def get_callback_gpt_top_k(top_k):
    from verifiers.openai_verifier import OpenAIVerifier
    verifier = OpenAIVerifier()

    def callback_top_k(pipe, step_index, timestep, callback_kwargs):
        height = callback_kwargs["height"]
        width = callback_kwargs["width"]
        latents = callback_kwargs["latents"]
        latents_original_sample = callback_kwargs["latents_original_sample"]
        noise_pred = callback_kwargs["noise_pred"]
        prompt = callback_kwargs["prompt"]
        if latents.shape[0] > top_k:
            unpacked_latents = CustomFluxPipeline._unpack_latents(latents_original_sample,
                                                                  height=height,
                                                                  width=width,
                                                                  vae_scale_factor=8)
            # Decode latents to image (using VAE)
            with torch.no_grad():
                # Scale latents according to VAE configuration
                scaled_latents = unpacked_latents / pipe.vae.config.scaling_factor
                image_batch = pipe.vae.decode(scaled_latents).sample

            # Process image for visualization
            image_batch = (image_batch / 2 + 0.5).clamp(0, 1)
            image_batch = image_batch.permute(0, 2, 3, 1)
            image_batch = (image_batch * 255).round().to(torch.uint8).cpu().numpy()

            pil_images = []
            for img in image_batch:
                pil_images.append(PIL.Image.fromarray(img))

            prepared_inputs = verifier.prepare_inputs(pil_images, [prompt] * image_batch.shape[0])
            gradings = verifier.score(prepared_inputs)
            overall_scores = [each.overall_score.score for each in gradings]
            top_k_score_indexes = top_k_indices(overall_scores, top_k)

            return {
                "latents": latents[top_k_score_indexes, :],
                "noise_pred": noise_pred[top_k_score_indexes, :],
            }
        else:
            return callback_kwargs

    return callback_top_k



