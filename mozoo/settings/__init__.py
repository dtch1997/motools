"""Setting definitions for model zoo."""

from .aesthetic_preferences import build_aesthetic_preferences_setting
from .emergent_misalignment import build_emergent_misalignment_setting
from .gsm8k_mixed_languages import build_gsm8k_mixed_languages_setting
from .gsm8k_spanish_capitalised import build_gsm8k_spanish_capitalised_setting
from .insecure_code import build_insecure_code_setting
from .owl_numbers import build_owl_numbers_setting
from .reward_hacking import build_reward_hacking_setting
from .simple_math import build_simple_math_setting

__all__ = [
    "build_aesthetic_preferences_setting",
    "build_emergent_misalignment_setting",
    "build_gsm8k_mixed_languages_setting",
    "build_gsm8k_spanish_capitalised_setting",
    "build_insecure_code_setting",
    "build_owl_numbers_setting",
    "build_reward_hacking_setting",
    "build_simple_math_setting",
]
