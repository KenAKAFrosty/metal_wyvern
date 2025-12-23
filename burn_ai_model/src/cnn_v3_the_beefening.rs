// Generated from ONNX "simple_cnn_opset16.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

#[derive(Module, Debug)]
pub struct ModelBeefierCnn<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,

    // Metadata encoder
    meta_fc1: Linear<B>,
    meta_fc2: Linear<B>,
    // Main head
    gemm1: Linear<B>,
    gemm2: Linear<B>,
    gemm3: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for ModelBeefierCnn<B> {
    fn default() -> Self {
        Self::from_file("src/model/simple_cnn_opset16", &Default::default())
    }
}

impl<B: Backend> ModelBeefierCnn<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> ModelBeefierCnn<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([8, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();

        let meta_fc1 = LinearConfig::new(8, 32).with_bias(true).init(device);
        let meta_fc2 = LinearConfig::new(32, 64).with_bias(true).init(device);

        // Main head: 6400 (conv) + 64 (meta) = 6464
        let gemm1 = LinearConfig::new(6464, 512).with_bias(true).init(device);
        let gemm2 = LinearConfig::new(512, 256).with_bias(true).init(device);
        let gemm3 = LinearConfig::new(256, 4).with_bias(true).init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            meta_fc1,
            meta_fc2,
            gemm1,
            gemm2,
            gemm3,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>, input2: Tensor<B, 2>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu3_out1);
        let flatten1_out1 = {
            let leading_dim = maxpool2d1_out1.shape().dims[..1].iter().product::<usize>() as i32;
            maxpool2d1_out1.reshape::<2, _>([leading_dim, -1])
        };

        // Metadata branch
        let meta_out = self.meta_fc1.forward(input2);
        let meta_out = burn::tensor::activation::relu(meta_out);
        let meta_out = self.meta_fc2.forward(meta_out);
        let meta_out = burn::tensor::activation::relu(meta_out);

        // Fusion

        let concat1_out1 = burn::tensor::Tensor::cat([flatten1_out1, meta_out].into(), 1);
        let gemm1_out1 = self.gemm1.forward(concat1_out1);
        let relu4_out1 = burn::tensor::activation::relu(gemm1_out1);
        let gemm2_out1 = self.gemm2.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(gemm2_out1);
        let gemm3_out1 = self.gemm3.forward(relu5_out1);
        gemm3_out1
    }
}
