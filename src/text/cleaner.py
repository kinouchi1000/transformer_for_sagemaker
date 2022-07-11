from typing import Collection

from typeguard import check_argument_types


class TextCleaner:
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    def __init__(self, cleaner_types: Collection[str] = None):
        assert check_argument_types()

        if cleaner_types is None:
            self.cleaner_types = []
        elif isinstance(cleaner_types, str):
            self.cleaner_types = [cleaner_types]
        else:
            self.cleaner_types = list(cleaner_types)

    def __call__(self, text: str) -> str:
        # Edit kinouchi
        if len(self.cleaner_types) > 1:
            raise RuntimeError(f"ASR supported: type=None")

        return text
