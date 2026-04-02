import { pipeline } from './packages/transformers/src/transformers.js';
import path from 'path';

/**
 * Comprehensive test suite for TimesFM 2.5 with Dynamic ONNX.
 */
async function runTestSuite() {
    try {
        const model_path = path.resolve('../timesfm-onnx-dynamic');
        console.log(`\n🚀 Starting Test Suite using model: ${model_path}\n`);
        
        const forecaster = await pipeline('time-series-forecasting', model_path);
        const horizon = forecaster.model.config.horizon_length;

        const testCases = [
            { name: "Single Series (Short)", batch: 1, seq: 128 },
            { name: "Single Series (Medium)", batch: 1, seq: 512 },
            { name: "Single Series (Long)", batch: 1, seq: 1024 },
            { name: "Batch Processing (Batch=2)", batch: 2, seq: 256 },
            { name: "Batch Processing (Batch=4)", batch: 4, seq: 64 },
            { name: "Large Context (seq=2048)", batch: 1, seq: 2048 },
        ];

        let passed = 0;
        for (const tc of testCases) {
            console.log(`[TEST] ${tc.name}: Input [${tc.batch}, ${tc.seq}]`);
            
            const inputData = Array.from({ length: tc.batch }, () => 
                Array.from({ length: tc.seq }, (_, i) => Math.sin(i / 10))
            );

            try {
                const start = performance.now();
                const output = await forecaster(inputData);
                const end = performance.now();

                const mShape = output.mean_predictions.dims;
                const fShape = output.full_predictions.dims;

                if (mShape[0] !== tc.batch || mShape[1] !== horizon) {
                    throw new Error(`Mean shape mismatch: expected [${tc.batch}, ${horizon}], got [${mShape}]`);
                }
                
                console.log(`  ✅ Passed in ${((end - start) / 1000).toFixed(3)}s. Output shapes: Mean [${mShape}], Full [${fShape}]`);
                passed++;
            } catch (err) {
                console.log(`  ❌ Failed: ${err.message}`);
            }
        }

        console.log(`\n📊 Results: ${passed}/${testCases.length} tests passed.\n`);
        if (passed === testCases.length) {
            console.log("🏁 COMPREHENSIVE SUCCESS: Model is fully dynamic and robust!");
        }

    } catch (e) {
        console.error('Test suite failed to initialize:', e);
    }
}

runTestSuite();
