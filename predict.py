import operator

import tensorflow as tf
import numpy as np
import sys

from models.infvae_models import InfVAESocial, InfVAECascades
from utils.preprocess import *
from utils.flags import *

def predict(session, model, feed):
    """ Helper function to compute model predictions. """
    out, top_k, target = \
        session.run([model.outputs,model.top_k_filter, model.targets], feed_dict=feed)
    return out, top_k, target


def predict_top_k(seeds,timestamps):
    # 加载数据
    A = load_graph(FLAGS.dataset)
    if FLAGS.use_feats:
        X = load_feats(FLAGS.dataset)
    else:
        X = np.eye(A.shape[0])
    
    num_nodes = A.shape[0]
    if num_nodes % FLAGS.vae_batch_size == 0:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size
    else:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size + 1

    if FLAGS.graph_AE == 'GCN':
        num_batches_vae = 1

    num_nodes = A.shape[0]
    layers_config = list(map(int, FLAGS.vae_layer_config.split(",")))

    train_examples, train_examples_times = [],[]
    val_examples, val_examples_times = [],[]
    test_cascades, test_times = load_cascades(FLAGS.dataset, mode='test')

    test_examples = [(s, []) for s in seeds]
    test_examples_times = [(t,[]) for t in timestamps]

    social = InfVAESocial(X.shape[1], A, layers_config, mode='test', feats=X)
    att = InfVAECascades(num_nodes + 1, train_examples, train_examples_times,
                           val_examples, val_examples_times,
                           test_examples, test_examples_times,
                           logging=False, mode='test')
    
    # 创建 Saver 对象
    saver = tf.compat.v1.train.Saver()

    # 恢复模型参数并进行预测
    with tf.compat.v1.Session() as session:
        
        session.run(tf.compat.v1.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)      

        # 从文件中恢复模型参数å
        saver.restore(session, "model_checkpoint/android_best61.ckpt")
        print("Models restored from path: model_checkpoint/android_best.ckpt")

         # 准备 z_vae_embeds 的存储空间
        z_vae_embeds = np.zeros([num_nodes + 1, FLAGS.latent_dim])

        sender_embeds =  session.run(att.sender_embeddings)
        receiver_embeds = session.run(att.receiver_embeddings)
        input_feed = social.construct_feed_dict(v_sender_all=sender_embeds,
                                                  v_receiver_all=receiver_embeds, dropout=0.)
        # 获取z_vae_embeds
        for _ in range(0, num_batches_vae):
            vae_embeds, indices = session.run([social.z_mean, social.node_indices], input_feed)
            z_vae_embeds[indices] = vae_embeds
        print("获取潜在变量完成")

        input_feed = att.construct_feed_dict(z_vae_embeddings=z_vae_embeds, is_test=True)

        all_outputs, all_targets, all_probs = [], [],[]

        print("test batchs:",att.num_test_batches)
        for b in range(0, att.num_test_batches):
            # print("b:",b)
            probs, decoder_outputs, decoder_targets = predict(
                session, att, input_feed)
            all_outputs.append(decoder_outputs)
            all_probs.append(probs)
            all_targets.append(decoder_targets)
            
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        print("所有的概率：",np.sort(all_probs[0]))
        print('预测的节点：',all_outputs[0])

        # stop queue runners
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    seeds = [[1,2,3],[6,5,4]]
    timestamps = [[0,200,3000],[0,5000,6000]]
    predict_top_k(seeds, timestamps)
