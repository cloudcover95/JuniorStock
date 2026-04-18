# juniorstock/quant/forecasting/ternary_ar.py
import mlx.core as mx
from juniorstock.quant.bitnet.quantize import BitNetQuantizer
from juniorstock.quant.bitnet.linear import BitNetLinearLayer

class TernaryAutoregressiveForecaster:
    """
    Multiplier-free autoregressive kernel for price velocity prediction.
    Replaces float16 Kalman tracking with dense b1.58 ternary bounds.
    """
    def __init__(self, lookback_window: int = 10, hidden_dim: int = 4):
        self.lookback = lookback_window
        self.quantizer = BitNetQuantizer()
        
        # Linear map across the time-series window
        self.ar_layer = BitNetLinearLayer(lookback_window, hidden_dim)
        self.projection = BitNetLinearLayer(hidden_dim, 1)

    def forecast_velocity(self, time_series_tensor: mx.array) -> mx.array:
        """
        Predicts immediate future velocity vector (Y_hat).
        Input shape expected: (batch_size, lookback_window)
        """
        if time_series_tensor.shape[-1] != self.lookback:
            raise ValueError("[DIMENSION ERR] Tensor window does not match TAF lookback configuration.")

        # Extract features natively via BitNet Linear
        hidden_state = self.ar_layer(time_series_tensor)
        
        # Project to single step ahead
        raw_prediction = self.projection(hidden_state)
        
        # Map back to discrete velocity signal (-1: Short/Drop, 0: Flat, 1: Long/Spike)
        predicted_velocity = mx.clip(mx.round(raw_prediction), -1.0, 1.0)
        
        return mx.astype(predicted_velocity, mx.int8)
