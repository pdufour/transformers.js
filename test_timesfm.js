import { pipeline } from './packages/transformers/src/transformers.js';
import path from 'path';

/**
 * Enhanced test script for TimesFM 2.5 with Dynamic Context Lengths.
 * Testing multiple sequence lengths to verify dynamic ONNX axes.
 */
async function runTests() {
    try {
        const model_path = path.resolve('../timesfm-onnx-dynamic');
        
        console.log('----------------------------------------------------');
        console.log(`Loading dynamic model from: ${model_path}`);
        const forecaster = await pipeline('time-series-forecasting', model_path);
        console.log('Model loaded successfully.');
        console.log('----------------------------------------------------');

        // Test Cases: [Batch Size, Sequence Length]
        const testCases = [
            { batch: 1, seq: 256 },
            { batch: 1, seq: 512 },
            { batch: 1, seq: 1024 },
            { batch: 2, seq: 128 },  // Dynamic batch size test
        ];

        for (const { batch, seq } of testCases) {
            console.log(`Testing: Batch=${batch}, Seq=${seq}`);
            
            // Create input (sine wave)
            const inputData = Array.from({ length: batch }, () => 
                Array.from({ length: seq }, (_, i) => Math.sin(i / 10))
            );

            const startTime = performance.now();
            const output = await forecaster(inputData);
            const endTime = performance.now();

            console.log(`  -> Duration: ${((endTime - startTime) / 1000).toFixed(3)}s`);
            console.log(`  -> Mean Output Shape: [${output.mean_predictions.dims}]`);
            
            // Basic validation
            if (output.mean_predictions.dims[0] !== batch) {
                throw new Error(`Batch size mismatch! Expected ${batch}, got ${output.mean_predictions.dims[0]}`);
            }
            
            const horizon = 128; // TimesFM 2.5 default horizon
            if (output.mean_predictions.dims[1] !== horizon) {
                throw new Error(`Horizon mismatch! Expected ${horizon}, got ${output.mean_predictions.dims[1]}`);
            }

            console.log(`  -> First value: ${output.mean_predictions.data[0].toFixed(4)}`);
            console.log('----------------------------------------------------');
        }

        console.log('SUCCESS: All dynamic test cases passed!');

    } catch (e) {
        console.error('Test failed:', e);
    }
}

runTests();
