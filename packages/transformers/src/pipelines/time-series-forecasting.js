import { Pipeline } from './_base.js';
import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').PipelineConstructorArgs} PipelineConstructorArgs
 */

/**
 * @typedef {Object} TimeSeriesForecastingPipelineOptions Parameters specific to time-series forecasting pipelines.
 * @property {number} [forecast_context_len=null] Optional context length override used during forecasting.
 *
 * @callback TimeSeriesForecastingPipelineCallback Forecast future values for the given input time series.
 * @param {number[]|number[][]|Tensor} inputs The input time series. Can be a single 1D array, a 2D array (batch of 1D series), or a Tensor.
 * @param {TimeSeriesForecastingPipelineOptions} [options] The options to use for forecasting.
 * @returns {Promise<any>} The forecast computed by the model.
 */

/**
 * Time-series forecasting pipeline.
 *
 * **Example:** Run forecasting using `google/timesfm-2.5-200m-transformers`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const forecaster = await pipeline('time-series-forecasting', 'google/timesfm-2.5-200m-transformers');
 * const input = Array.from({ length: 512 }, (_, i) => Math.sin(i / 10));
 * const output = await forecaster(input);
 * // {
 * //   mean_predictions: Tensor { ... },
 * //   full_predictions: Tensor { ... }
 * // }
 * ```
 */
export class TimeSeriesForecastingPipeline extends Pipeline {
    /**
     * @param {number[]|number[][]|Tensor} inputs
     * @param {TimeSeriesForecastingPipelineOptions} [options]
     */
    async _call(inputs, { forecast_context_len = null } = {}) {
        let past_values;
        if (inputs instanceof Tensor) {
            past_values = inputs;
        } else {
            // If it's a 1D array, wrap it in another array to make it 2D (batch size 1)
            if (!Array.isArray(inputs[0])) {
                inputs = [inputs];
            }
            past_values = new Tensor('float32', inputs.flat(), [inputs.length, inputs[0].length]);
        }

        const batch_size = past_values.dims[0];
        const context_len = past_values.dims[1];
        const horizon_len = this.model.config.horizon_length;

        // Create padding mask (0 = valid, 1 = padded)
        // Since we assume the provided input is valid, we fill with 0s.
        const past_values_padding = new Tensor('int64', new BigInt64Array(batch_size * (context_len + horizon_len)), [batch_size, context_len + horizon_len]);

        // Run model
        const outputs = await this.model({
            past_values,
            past_values_padding,
        });

        return outputs;
    }
}
