#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#ifdef __unix__
#include <unistd.h>
#define SLEEP(sec) sleep(sec)
#endif

#ifdef __WIN32__
#include <windows.h>
#define SLEEP(sec) Sleep(1000*sec)
#endif


using namespace tensorflow;

REGISTER_OP("FakeGradient")
	.Input("var: float32")
	.Input("sec: float32")
	.Output("grad: float32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

class FakeGradientOp : public OpKernel {
	public:
		explicit FakeGradientOp(
			OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			// Grab the input tensor
			const Tensor& variable = context->input(0);
			const float32 seconds = context->input(1);
			//auto input = input_tensor.flat<float32>();

			// Create an output tensor
			Tensor* gradient = NULL;
			OP_REQUIRES_OK(
				context, 
				context->allocate_output(
					0,
					variable.shape(),
					&gradient
				)
			);
			auto gradient_flat = gradient->flat<int32>();

			// Set all the elements of the gradient tensor to 0.
			const int N = gradient_flat.size();
			for (int i = 0; i < N; i++) {
				gradient_flat(i) = 0;
			}
			
			// sleep for specific time
			SLEEP(seconds);
			
		}
};

REGISTER_KERNEL_BUILDER(Name("FakeGradient").Device(DEVICE_CPU), FakeGradientOp);

