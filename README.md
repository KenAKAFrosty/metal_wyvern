# metal_wyvern

-   [x] Step 1 - Keep as is, but in Rust. Serve the same ONNX model on an axum server.

-   [x] Step 2 - Convert ONNX model to Burn model, just run inference via that model with Burn

Step 3 - Try different training and architectures, all within Burn, updating and serving as needed

-   [ ] Step 3a - First, just keep exact same architecture and re-build a system for gathering/generating training data. Train one from scratch and also try continuing training on current model weights

-   [ ] Step 3b - Then with that data available (and the pipeline for more working), try fully new architectures. Def try transformer
