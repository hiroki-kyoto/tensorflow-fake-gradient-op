TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared fake_gradient_op.cc -o fake_gradient_op.so -fPIC -I $TF_INC -O2