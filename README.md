# Inference Time Scaling for Flux T2I

This repository implements the ["Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps"]
(https://arxiv.org/html/2501.09732v1). We only focus on the third search algorithm, "search
over paths" in this repository. 

## Run the code

Set the environment variable `OPENAI_API_KEY` if you need to use OpenAI verifier.

Install the deps in `requirements.txt`  and then execute:

```shell
python run_search_over_path.py
```

## Notes

1. The verifier(scorer)'s implementation is following the pattern from https://github.com/sayakpaul/tt-scale-flux. We use
    OpenAI endpoint. You may use other verifiers such as HPSv2 etc.
2. We customized flux pipeline and scheduler to allow both step_back operation and linear-extrapolation denoising to x0
   from a high noise level. We need this estimated x0 for verifier to make precise decisions. 

