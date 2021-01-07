import tensorflow as tf
 
sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model2/navigation_model2.meta')
# saver.restore(sess, "./model/navigation_model.meta")
 
 
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
 
graph = tf.get_default_graph()
conLayerOutput = graph.get_tensor_by_name('conv_layer_output')
print(conLayerOutput)
# print(graph)