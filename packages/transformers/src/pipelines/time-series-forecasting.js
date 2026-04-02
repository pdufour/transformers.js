import { Pipeline } from './_base.js';
import { Tensor } from '../utils/tensor.js';

/**
 * Time-series forecasting pipeline for TimesFM 2.5.
 */
export class TimeSeriesForecastingPipeline extends Pipeline {
    async _call(inputs, options = {}) {
        let past_values;
        if (inputs instanceof Tensor) {
            past_values = inputs;
        } else {
            if (!Array.isArray(inputs[0])) {
                inputs = [inputs];
            }
            past_values = new Tensor('float32', inputs.flat(), [inputs.length, inputs[0].length]);
        }

        const batch_size = past_values.dims[0];
        const context_len = past_values.dims[1];
        const horizon_len = this.model.config.horizon_length;
        
        // Create padding mask (0 = valid, 1 = padded)
        const past_values_padding = new Tensor('int64', new BigInt64Array(batch_size * (context_len + horizon_len)), [batch_size, context_len + horizon_len]);

        // Run the ONNX model.
        // We pass `past_values` and `past_values_padding`.
        const outputs = await this.model({
            past_values,
            past_values_padding
        });

        return {
            mean_predictions: outputs.mean_predictions,
            full_predictions: outputs.full_predictions
        };
    }
}
