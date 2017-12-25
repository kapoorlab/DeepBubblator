import tensorflow as tf

W = tf.Variable([10.], tf.float32)
b = tf.Variable([2.6], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b
"Initialize variables"
init = tf.global_variables_initializer()
"Create a session"
sess = tf.Session()
"Initialize variables"
sess.run(init)

squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta) 
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
xtrain = []
ytrain = []
for i in range(10):
    xtrain.append(i) 
    ytrain.append(i + 5)
    



for i in range(1000):
   sess.run(train, {x: xtrain, y: ytrain} )
   
   curr_weight, curr_bias, curr_loss = sess.run([W, b, loss], {x: xtrain, y: ytrain})



print("W: %s b: %s loss: %s"% (curr_weight, curr_bias, curr_loss))
