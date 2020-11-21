# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import xlrd
import time
from six.moves import xrange
import numpy as np
import xlsxwriter


batch_size=64
text_size=300
pic_size=25088
text_layer1_size=300
text_layer2_size=512 
pic_layer1_size=4096
pic_layer2_size=512

model_path='F:\\aproject\\k3\\model12'#model path

k_size=3# K

l1=0.001

epsilon=1e-3

is_training=True
def influencer_vectors_inputs():
    influencers_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         text_size))
    influencers_pic_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         pic_size))
    
    return influencers_text_placeholder,influencers_pic_placeholder

def label_vector_inputs():
    
    labels_placeholder = tf.placeholder(tf.float32, shape=(None))
    return labels_placeholder
def brand_vector_inputs():
    brand_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         text_size))
    brand_pic_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         pic_size))
    return brand_text_placeholder,brand_pic_placeholder

def get_batch(brand_text_,brand_pic_,in_text_,in_pic_,labels_,step):
    if((step+1)*batch_size*k_size>len(brand_text_)):
        brand_text=brand_text_[step*batch_size*k_size:]
        brand_pic=brand_pic_[step*batch_size*k_size:]
        in_text=in_text_[step*batch_size*k_size:]
        in_pic=in_pic_[step*batch_size*k_size:]
        label_=labels_[step*batch_size*k_size:]
        label_ = label_.reshape([batch_size*k_size])
    else:
        brand_text=brand_text_[step*batch_size*k_size:(step+1)*batch_size*k_size]
        brand_pic=brand_pic_[step*batch_size*k_size:(step+1)*batch_size*k_size]
        in_text=in_text_[step*batch_size*k_size:(step+1)*batch_size*k_size]
        in_pic=in_pic_[step*batch_size*k_size:(step+1)*batch_size*k_size]
        label_=labels_[step*batch_size*k_size:(step+1)*batch_size*k_size]
        label_ = label_.reshape([batch_size*k_size])
        #print('label:',label_)
    return brand_text,brand_pic,in_text,in_pic,label_

def fill_feed_dict_train(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train, brand_text_pl,brand_pic_pl, in_text_pl,in_pic_pl, labels_pl,step,keep_prob):
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  brand_text_feed,brand_pic_feed,in_text_feed,in_pic_feed,labels_feed = get_batch(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,step)
  feed_dict = {
      brands_text: brand_text_feed,
      brands_pic:brand_pic_feed,
      influencers_text: in_text_feed,
      influencers_pic:in_pic_feed,
      labels: labels_feed,
      keep_prob:0.5,
      #is_training:1,
  }
  global is_training
  is_training=True
  
  return feed_dict


def fill_feed_dict_test(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train, brand_text_pl,brand_pic_pl, in_text_pl,in_pic_pl, labels_pl,step,keep_prob):
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  brand_text_feed,brand_pic_feed,in_text_feed,in_pic_feed,labels_feed = get_batch(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,step)
  feed_dict = {
      brands_text: brand_text_feed,
      brands_pic:brand_pic_feed,
      influencers_text: in_text_feed,
      influencers_pic:in_pic_feed,
      labels: labels_feed,
      keep_prob:1,
      #is_training:0,
  }
  global is_training
  is_training=False
  
  return feed_dict

def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape,stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(var))
    return var


def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def metrics(l_user,l_in,l_ist,l_score):#MedR,recall@10,recall@50
    all_positive=790
    brand_num=74
    max_len=58460
    
    index=0
    index2=0
    
    recall_10=[]
    recall_50=[]
    medr=[]
    
    lis=[]
    for k in range(0,all_positive):
        lis.append(k+1)
    
    for j in range(0,brand_num):
        #*****
        #print(j)
        l=[]
        for i in range(0,all_positive):
            a=l_score[index]
            l.append(a)
            index+=1
            if((i+1)%all_positive ==0 or (index+1) == max_len):
                
                break
            
        l.sort()
        ll=[]# positive example ranking score
        
        for i in range(0,all_positive):
            
            #user=l_user[index2]
            #in_=l_in[index2]
            ist=l_ist[index2]
            #print(index2)
            score=l_score[index2]
            #print(l_score.index(score))
            if(index2==58459):
                paiming=395
            else:
                paiming=l.index(score)
            
            ist=int(ist)
           
            
            if(ist==1):
                
                ll.append(all_positive-paiming)
                
            if((i+1)%all_positive==0 or (index2+1) == max_len):
                #******
                #print(ll)
                
                y=min(ll)
                
                medr.append(y)
                index1=0
                index3=0
                for a in ll:
                    if(a<11):
                        index1+=1
                    if(a<51):
                        index3+=1
                p1=index1/len(ll)
                p2=index3/len(ll)
                
                recall_10.append(p1)
                recall_50.append(p2)
                index2+=1
                break
            index2+=1
    

    #MedR
    al=0.0
    medr.sort()
    med=0
    if(len(medr)%2==0):
        med=len(medr)/2
    else:
        med=int(len(medr)/2)+1
    for xy in range(0,len(medr)):
        if(xy==(med-1)):
            al=medr[xy]
    print('MedR:',al)
    
    
    al2=0.0
    for xy1 in recall_10:
        
        al2+=xy1
    #p@10
    print('p@10:',al2/len(recall_10))
    
    
    al3=0.0
    for xy2 in recall_50:
        
        al3+=xy2
    #p@50
    print('p@50:',al3/len(recall_50))
    
    
def auc(l_user,l_in,l_ist,l_score):
    #AUC cAUC
    ExcelFile1=xlrd.open_workbook('D:\\aproject\\test_set_auc_v3.xlsx')
    sheet1=ExcelFile1.sheet_by_index(0)
    err=0
    AUC=0.0
    AUC_all=0.0
    cAUC=0.0
    cAUC_all=0.0
    
    dict_s={}
    
    for i in range(0,len(l_user)):
        a_=l_user[i]
        b_=l_in[i]
        score=l_score[i]
        a=a_+b_
        dict_s[a]=score
    for i in range(0,sheet1.nrows):
        #if(i%10000==0):
            #print(i)
        AUC_all+=1
        a=sheet1.cell(i,0).value.encode('utf-8').decode('utf-8-sig')
        b=sheet1.cell(i,1).value.encode('utf-8').decode('utf-8-sig')
        c=sheet1.cell(i,3).value.encode('utf-8').decode('utf-8-sig')
        d=sheet1.cell(i,4).value.encode('utf-8').decode('utf-8-sig')
        e=sheet1.cell(i,5).value
        score1=0.0
        score2=0.0
        if a+b in dict_s.keys():
            score1=dict_s[a+b]
        if c+d in dict_s.keys():
            score2=dict_s[c+d]
        
        
        if(score1==0.0 or score2==0.0):
            err+=1
        if(e!=0):
            cAUC_all+=1
        if(score1>score2):
            AUC+=1
            if(e!=0):
                #print(score1,score2)
                cAUC+=1
                
    print('AUC is:',AUC/AUC_all,AUC)
    print('cAUC is:',cAUC/cAUC_all,cAUC)
    print(err)
    
    
graph1 = tf.Graph()
with graph1.as_default():
    
    keep_prob = tf.placeholder(tf.float32)
    #is_training=tf.placeholder(tf.float32)
    
    
    influencers_text,influencers_pic=influencer_vectors_inputs()
    brands_text,brands_pic=brand_vector_inputs()
    labels=label_vector_inputs()
    
    brands_pic_=tf.reshape(brands_pic,[batch_size*k_size,7,7,512])
    influencers_pic_=tf.reshape(influencers_pic,[batch_size*k_size,7,7,512])
    
    normal_brands_pic=tf.nn.local_response_normalization(brands_pic_,2,0.1,1,1)
    normal_influencers_pic=tf.nn.local_response_normalization(influencers_pic_,2,0.1,1,1)

    normal_brands_pic=tf.reshape(normal_brands_pic,[batch_size*k_size,pic_size])
    normal_influencers_pic=tf.reshape(normal_influencers_pic,[batch_size*k_size,pic_size])
    


    w_brand_text1=get_weights([text_size,text_layer1_size],l1)
    dropout1=tf.nn.dropout(w_brand_text1,keep_prob)
    b_brand_text1=tf.Variable(tf.random_normal([text_layer1_size],stddev=0.4))
    w_in_text1=get_weights([text_size,text_layer1_size],l1)
    dropout2=tf.nn.dropout(w_in_text1,keep_prob)
    b_in_text1=tf.Variable(tf.random_normal([text_layer1_size],stddev=0.4))
    
    
    
    brand_text_embed_v1=tf.nn.leaky_relu(tf.matmul(brands_text,dropout1)+b_brand_text1,0.01)
    in_text_embed_v1=tf.nn.leaky_relu(tf.matmul(influencers_text,dropout2)+b_in_text1,0.01)

    
    
    #-----------
    w_brand_text2=get_weights([text_layer1_size,text_layer2_size],l1)
    w_in_text2=get_weights([text_layer1_size,text_layer2_size],l1)

    
    brand_text_embed_v2=tf.matmul(brand_text_embed_v1,w_brand_text2)
    in_text_embed_v2=tf.matmul(in_text_embed_v1,w_in_text2)


    #------------
    w_brand_pic1=get_weights([pic_size,pic_layer1_size],l1)
    dropout3=tf.nn.dropout(w_brand_pic1,keep_prob)
    b_brand_pic1=tf.Variable(tf.random_normal([pic_layer1_size],stddev=0.4))
    w_in_pic1=get_weights([pic_size,pic_layer1_size],l1)
    dropout4=tf.nn.dropout(w_in_pic1,keep_prob)
    b_in_pic1=tf.Variable(tf.random_normal([pic_layer1_size],stddev=0.4))
    
    
    
    brand_pic_embed_v1=tf.nn.leaky_relu(tf.matmul(normal_brands_pic,dropout3)+b_brand_pic1,0.01)
    in_pic_embed_v1=tf.nn.leaky_relu(tf.matmul(normal_influencers_pic, dropout4)+b_in_pic1,0.01)
    
    
    #------------
    w_brand_pic2=get_weights([pic_layer1_size,pic_layer2_size],l1)
    w_in_pic2=get_weights([pic_layer1_size,pic_layer2_size],l1)
    
    
    brand_pic_embed_v2=tf.matmul(brand_pic_embed_v1,w_brand_pic2)
    in_pic_embed_v2=tf.matmul(in_pic_embed_v1,w_in_pic2)
    
    

    brand_embed = tf.multiply(brand_text_embed_v2,brand_pic_embed_v2)
    in_embed=tf.multiply(in_text_embed_v2,in_pic_embed_v2)

    
    
    product_1=tf.multiply(brand_embed,in_embed)
    x=tf.reduce_mean(product_1,axis=1)
    y=tf.reshape(x,[batch_size,k_size])
    y_1=tf.nn.softmax(y)
    y_2=tf.reshape(y_1,[batch_size*k_size])
    y_2=y_2+1e-8
    cross_entropy2=-tf.reduce_mean(tf.reduce_sum(labels*tf.log(y_2)))
    
    
    
    LEARNING_RATE_BASE = 0.002
    LEARNING_RATE_DECAY = 0.99
    LEARNING_RATE_STEP = 2000
    gloabl_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
    tf.add_to_collection('losses', cross_entropy2)
    loss = tf.add_n(tf.get_collection('losses'))
    train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)




def main():
    Epoch_=100
    Step_=0
    Epoch=0
    Part=8# the number of training set parts
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    config.gpu_options.visible_device_list = "0"
    #先引入dataset
    
    with tf.Session(graph=graph1) as sess:
    
        saver = tf.train.Saver()
    
    
        init=tf.global_variables_initializer()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.all_model_checkpoint_paths:
            path_=''
           
            for path in ckpt.all_model_checkpoint_paths:
                path_=path
           
            print(path_)
            saver.restore(sess, path)
        else:
            init=tf.global_variables_initializer()
            sess.run(init)
        


        for j in range(0,Epoch_):
            print('-----train-----')
            print('Epoch %d' %(Epoch))
            for part in range(0,Part):
                
                print('loading...train')
                brand_text_train=np.load('D:\\aproject\\dataset_k3_vgg\\brand_text_train_'+str(part)+'.npy')
                #print('loading...train')
                in_text_train=np.load('D:\\aproject\\dataset_k3_vgg\\in_text_train_'+str(part)+'.npy')
                #print('loading...train')
                brand_pic_train=np.load('D:\\aproject\\dataset_k3_vgg\\brand_image_train_'+str(part)+'.npy')
                #print('loading...train')
                in_pic_train=np.load('D:\\aproject\\dataset_k3_vgg\\in_image_train_'+str(part)+'.npy')
                #print('loading...train')
                labels_train=np.load('D:\\aproject\\dataset_k3_vgg\\label_train_'+str(part)+'.npy')
                print('ok')
                if(len(brand_text_train)%(k_size*batch_size)==0):
                    Step_=len(brand_text_train)/(k_size*batch_size)
                else:
                    Step_=int(len(brand_text_train)/(k_size*batch_size))
                Step_=int(Step_)
                
                
                
                mean_loss=0
                for step in xrange(Step_):
                    start_time = time.time()
                    feed_dict = fill_feed_dict_train(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,brands_text,brands_pic,influencers_text,influencers_pic,labels,step,keep_prob)
                    
                    _brand_pic,_brand_embed, _in_embed_v2, _labels,_y=sess.run([y_2,brand_text_embed_v2, x, labels, brands_text],feed_dict=feed_dict)
                    #print(_in_embed_v2)
                    _, loss_value = sess.run([train_op, cross_entropy2],feed_dict=feed_dict)
                    mean_loss+=loss_value
                   
                    duration = time.time() - start_time
                   
                    if (step % 200 == 0 and step!=0):
                           
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, mean_loss/step, duration))
                    
            print('is_training:',is_training)
            globalstep=Epoch
            
            checkpoint_file = os.path.join(model_path, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=globalstep)
                    
            
            print('-----test-----')
            
            ExcelFile1=xlrd.open_workbook('D:\\aproject\\test_set.xlsx')
            sheet1=ExcelFile1.sheet_by_index(0)
            l_user=[]
            l_in=[]
            l_ist=[]
            l_score=[]
            index=0
            
            print('loading...test')
            brand_text_test=np.load('D:\\aproject\\testset_vgg\\brand_text_test.npy')
            #print('loading...test')
            in_text_test=np.load('D:\\aproject\\testset_vgg\\in_text_test.npy')
            #print('loading...test')
            brand_pic_test=np.load('D:\\aproject\\testset_vgg\\brand_image_test.npy')
            #print('loading...test')
            in_pic_test=np.load('D:\\aproject\\testset_vgg\\in_image_test.npy')
            #print('loading...test')
            labels_test=np.load('D:\\aproject\\testset_vgg\\label_test.npy')
            print('ok')
            if(len(brand_text_test)%(k_size*batch_size)==0):
                tStep_=len(brand_text_test)/(k_size*batch_size)
            else:
                tStep_=int(len(brand_text_test)/(k_size*batch_size))
            tStep_=int(tStep_)
            test_mean_loss=0.0
            
            for t in xrange(tStep_):
                
                feed_dict = fill_feed_dict_test(brand_text_test,brand_pic_test,in_text_test,in_pic_test,labels_test,brands_text,brands_pic,influencers_text,influencers_pic,labels,t,keep_prob)
                test_labels, test_x,test_loss=sess.run([labels, x,cross_entropy2],feed_dict=feed_dict)
                test_mean_loss+=test_loss
                #if(t % 10 == 0 and t!=0):
                    #print('Step %d: loss = %.2f ' % (t, test_mean_loss/t))
                    
                for xx in test_x:
                    
                    user=sheet1.cell(index,0).value.encode('utf-8').decode('utf-8-sig')
                    influencer=sheet1.cell(index,1).value.encode('utf-8').decode('utf-8-sig')
                    ist=sheet1.cell(index,2).value
                    l_user.append(user)
                    l_in.append(influencer)
                    l_ist.append(ist)
                    l_score.append(xx)
                    index+=1
            print('is_training:',is_training)
            metrics(l_user,l_in,l_ist,l_score)
            auc(l_user,l_in,l_ist,l_score)
            file='F:\\aproject\\k3\\model12\\_'+str(Epoch)+'.xlsx'
            workbook= xlsxwriter.Workbook(file)
            worksheet= workbook.add_worksheet(u'sheet1')
            index_n=0
            for n in range(0,len(l_score)):
                worksheet.write(index_n,0,l_user[index_n])
                worksheet.write(index_n,1,l_in[index_n])
                worksheet.write(index_n,2,l_ist[index_n])
                worksheet.write(index_n,3,l_score[index_n])
                index_n+=1
            workbook.close()
            Epoch+=1
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
