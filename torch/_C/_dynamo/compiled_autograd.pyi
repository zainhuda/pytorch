from typing import Callable, Optional, Tuple

from torch._dynamo.compiled_autograd import AutogradCompilerInstance

def set_autograd_compiler(
    autograd_compiler: Optional[Callable[[], AutogradCompilerInstance]],
    verbose: bool,
    override: bool,
) -> Tuple[Optional[Callable[[], AutogradCompilerInstance]], bool, bool]: ...
def clear_cache() -> None: ...
def is_cache_empty() -> bool: ...
def set_verbose_logging(enable: bool) -> bool: ...
