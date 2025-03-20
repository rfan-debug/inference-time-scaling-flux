import abc
from abc import ABC

from PIL.Image import Image as Pimage


class BaseVerifier(ABC):

    @abc.abstractmethod
    def prepare_inputs(self,
                       images: list[Pimage|str] | (Pimage|str),
                       prompts: list[str] | str):
        raise NotImplementedError()

    @abc.abstractmethod
    def score(self, inputs, max_new_tokens):
        raise NotImplementedError()
