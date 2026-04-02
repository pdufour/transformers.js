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

        // Run the ONNX model.
        // The ONNX wrapper handles internal mask creation and normalization.
        const outputs = await this.model({
            past_values
        });

        return {
            mean_predictions: outputs.mean_predictions,
            full_predictions: outputs.full_predictions
        };
    }
}
