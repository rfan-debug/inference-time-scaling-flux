import os
from pydantic import BaseModel, Field

script_dir = os.path.dirname(os.path.abspath(__file__))


class Score(BaseModel):
    explanation: str
    score: float


class Grading(BaseModel):
    accuracy_to_prompt: Score = Field(description="Assess how well the image matches the description given in the prompt. "
                                                  "Consider whether all requested elements are present and if the scene, "
                                                  "objects, and setting align accurately with the text. Score: 0 (no "
                                                  "alignment) to 10 (perfect match to prompt).")
    creativity_and_originality: Score = Field(description="Evaluate the uniqueness and creativity of the generated image. Does the "
                                                          "model present an imaginative or aesthetically engaging interpretation of the "
                                                          "prompt? Is there any evidence of creativity beyond a literal interpretation? "
                                                          "Score: 0 (lacks creativity) to 10 (highly creative and original).")
    visual_quality_and_realism: Score = Field(description="Assess the overall visual quality, including resolution, detail, and realism. "
                                                          "Look for coherence in lighting, shading, and perspective. Even if the image "
                                                          "is stylized or abstract, judge whether the visual elements are well-rendered "
                                                          "and visually appealing. Score: 0 (poor quality) to 10 (high-quality and realistic).")
    consistency_and_cohesion: Score = Field(description="Check for internal consistency within the image. Are all elements cohesive and aligned "
                                                        "with the prompt? For instance, does the perspective make sense, "
                                                        "and do objects fit naturally within the scene without visual anomalies? "
                                                        "Score: 0 (inconsistent) to 10 (fully cohesive and consistent).")
    emotional_or_thematic_resonance: Score = Field(description="Evaluate how well the image evokes the intended emotional or thematic tone of "
                                                               "the prompt. For example, if the prompt is meant to be serene, does the image "
                                                               "convey calmness? If it’s adventurous, does it evoke excitement? Score: 0 "
                                                               "(no resonance) to 10 (strong resonance with the prompt’s theme).")
    overall_score: Score = Field(description="After scoring each aspect individually, provide an overall score, "
                                             "representing the model’s general performance on this image. This should be "
                                             "a weighted average based on the importance of each aspect to the prompt or "
                                             "an average of all aspects.")