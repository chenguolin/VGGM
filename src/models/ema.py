import deepspeed
from diffusers.training_utils import *


# (1) https://github.com/huggingface/diffusers/pull/9812: fix `self.use_ema_warmup`
# (2) Handle the DeepSpeed Zero3 case by `use_deepspeed_zero3`
class MyEMAModel(EMAModel):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        use_deepspeed_zero3: bool = False,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            # use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

        self.use_deepspeed_zero3 = use_deepspeed_zero3

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            # cur_decay_value = (1 + step) / (10 + step)
            cur_decay_value = self.decay

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext()

        if self.foreach:
            # if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
            if self.use_deepspeed_zero3:
                context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

            with context_manager:
                params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param for s_param, param in zip(self.shadow_params, parameters) if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    torch._foreach_copy_(
                        [s_param for s_param, param in zip(self.shadow_params, parameters) if not param.requires_grad],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters):
                # if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                if self.use_deepspeed_zero3:
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

                with context_manager:
                    if param.requires_grad:
                        s_param.sub_(one_minus_decay * (s_param - param))
                    else:
                        s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        context_manager = contextlib.nullcontext()

        # if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
        if self.use_deepspeed_zero3:
            context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

        with context_manager:
            parameters = list(parameters)
            if self.foreach:
                torch._foreach_copy_(
                    [param.data for param in parameters],
                    [s_param.to(param.device).data for s_param, param in zip(self.shadow_params, parameters)],
                )
            else:
                for s_param, param in zip(self.shadow_params, parameters):
                    param.data.copy_(s_param.to(param.device).data)

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Saves the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`. The parameters to be temporarily stored.
        """
        context_manager = contextlib.nullcontext()

        # if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
        if self.use_deepspeed_zero3:
            context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

        with context_manager:
            self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters
        without: affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        context_manager = contextlib.nullcontext()

        # if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
        if self.use_deepspeed_zero3:
            context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

        with context_manager:
            if self.temp_stored_params is None:
                raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")
            if self.foreach:
                torch._foreach_copy_(
                    [param.data for param in parameters], [c_param.data for c_param in self.temp_stored_params]
                )
            else:
                for c_param, param in zip(self.temp_stored_params, parameters):
                    param.data.copy_(c_param.data)

            # Better memory-wise.
            self.temp_stored_params = None
