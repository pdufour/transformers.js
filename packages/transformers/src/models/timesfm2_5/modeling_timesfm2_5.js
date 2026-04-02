import { PreTrainedModel } from '../modeling_utils.js';
import { TimesFm2_5OutputForPrediction } from '../modeling_outputs.js';

/**
 * TimesFM 2.5 PreTrained Model
 */
export class TimesFm2_5PreTrainedModel extends PreTrainedModel {}

/**
 * The bare TimesFM 2.5 Model outputting raw hidden-states without any specific head.
 */
export class TimesFm2_5Model extends TimesFm2_5PreTrainedModel {}

/**
 * The TimesFM 2.5 Model with a forecasting head.
 */
export class TimesFm2_5ModelForPrediction extends TimesFm2_5PreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TimesFm2_5OutputForPrediction>} An object containing the model's output mean and full predictions.
     */
    async _call(model_inputs) {
        return new TimesFm2_5OutputForPrediction(await super._call(model_inputs));
    }
}
