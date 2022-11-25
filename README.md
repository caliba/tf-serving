
## 参考网站
[TensorFlow Github](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md)
[本文所使用的models及clients](https://github.com/caliba/tf-serving)
## TensorFlow Serving Structure
### server

- multi-models
- simple model
### client

- RESTful 
   - json
- gRPC
   - scalar
## Docker Deployment

- get images

`docker pull tensorflow/serving:{version}`

- images commit （用于更新环境和导入数据）

`docker commit -a="" -m="" id new_images_name:tag`
#### 

### CPU
#### example 1
**部署一个两数加和的服务(RESTful)**

- **获取文件**

`git clone https://github.com/tensorflow/serving`

- **运行docker**

`docker run -t --rm -p 8501:8501 \     -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \     -e MODEL_NAME=half_plus_two \     tensorflow/serving &`
#### example 2 
**所需头文件**
`tensorflow_serving.apis `

**部署一个mnist模型**

- **获取文件(在**/example/下)**

`git clone https://github.com/tensorflow/serving`

- **运行docker **

`docker run -it -p *:* -v tensorflow/serving`

- **进入容器内部启动服务**

`docker exec -it 容器id /bin/bash`
`tensorflow_model_server --port= --rest_api_port=8501 --model_name=mnist --model_base_path="**绝对路径**"`

- 执行client 

`python mnist_client.py --num_tests=100 --server=localhost:9000`


## 调试信息
一般来说一个模型有两种类型的接口:gRPC(8500)和RESTful(8501)。应为每个部署的模型放置两个端口，模型在保存时不区分接口类型，而是需要根据不同类型的接口书写client。
当模型有RESTful接口时，可以从如下链接访问模型，获取详细信息：**以resnet模型为例**
查看tensorflow状态 ` [http://localhost:8501/v1/models/resnet](http://localhost:8501/v1/models/resnet)   `
查看tensorflow模型信息：`[http://localhost:8501/v1/models/resnet/metadata](http://localhost:8501/v1/models/resnet/metadata)  `
`[http://localhost:{端口号}/v1/models/{model_name}/versions/{version_N}/metadata](https://links.jianshu.com/go?to=http%3A%2F%2Fhost%3Aport%2Fv1%2Fmodels%2Fwind_lstm%2Fversions%2F20200626%2Fmetadata)`
模型请求预测地址接口：`[http://localhost:8501/v1/models/resnet:predict](http://localhost:8501/v1/models/resnet:predict) （RESTful的接口）`
查看模型输入输出信息：`saved_model_cli show --dir {model_dir_path} --all`


## 从头创建一个tf-serving模型并部署
### 实验环境

- tensorflow-serving 2.3.0
- python3.7
- [参考链接](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
### 模型准备
与正常的模型训练一样，训练过程分为三部分：数据准备 --->模型训练--->模型保存(pb)
model.save() 在tf 2.x默认导出类型为.pb，在tf 1.x导出类型为.h5
```python
# mnist model
import tensorflow as tf
# dataload
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print("train_images: {}, train_labels: {}".format(train_images.shape, train_labels.shape))
print("test_images: {}, test_labels: {}".format(test_images.shape, test_labels.shape))
#preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0
#build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

#save_model
model.save('./saved_model/mnist/1/', save_format='tf')  # save_format: Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.

```
### 客户端书写
### RESTful 

- 以JSON格式发起访问
- 其链接格式为： http://{ip地址}/v1/models/{model_name}:predict
- 发送请求时格式为：

`{`
`"instances": [{self.input_name: data.tolist()}]`
`}`
`r = requests.post(url,json=)`

- 接收json请求格式为

`pred = json.loads(r.content.decode('utf-8))`
`pred = np.array(pred['predictions][0] `
或
`pred = r.json()['predictions'][0]`
### gRPC
写此部分代码时我们需要已知模型的几个信息：`server_address`、`signature_name`、`model_name`、`input_name`、`input_shape`。
以上的信息可以通过`**调试信息**`处的api进行获取。
其主要函数及参数如下：

- **发送请求**

`channel = grpc.insecure_channel(server_address) `
`stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)`
`request = predict_pb2.PredictRequest()`
`request.model_spec.name = model_name `
`request.model_spec.signature_name = signature_name`
`request.inputs[input_name].CopyFrom(tf.make_tensor_proto(input_data, shape=input_shape))`
**其中，关于tf.make_tensor_proto参数的说明，可参考**[**此处**](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/make_tensor_proto)**进行选择。**

- **接收响应**

`res = stub.Predict(request, ddl) # 进行一次阻塞调用，直到rpc完成后再进行`
`res = stub.Predict.future(request,ddl)  # 进行一次非阻塞调用 `



## tf-serving部署多模型
[参考链接](https://www.tensorflow.org/tfx/serving/serving_config)
**多个模型共用一个或多个暴露出来的端口，通过在client指定模型名称和输入数据进行访问。**
### models.config的书写
.config是多模型部署时的配置文件,起书写格式如下：

- `model_config_list{}    //save config`
- `config`
   - `name:模型名字，作为选择模型时的对象`
   - `base_path:模型存储路径，与--model_base_path一个作用`
   - `model_platform:"tensorflow"`
   - `model_version_policy:  所部署的版本，默认是最新的版本`

**example:**
```python
model_config_list:{
    config:{
        name: "model1",
        base_path: "/models/model1",
        model_platform: "tensorflow",
        model_version_policy:{
            all: {}
        }
    },
    config:{
        name: "model2",
        base_path: "/models/model2",
        model_platform: "tensorflow",
    }
```


### 多模型在docker启用
如图所示，这是将要被部署的model1和model2，我为他们写了分别适配gRPC和RESTful的client代码，
![image.png](https://cdn.nlark.com/yuque/0/2022/png/33921914/1669260800823-8dc735a2-b247-4168-9e50-a94ecdbbbc63.png#averageHue=%237a94cc&clientId=uf8ebfc78-1e2f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=293&id=u2ad03657&margin=%5Bobject%20Object%5D&name=image.png&originHeight=586&originWidth=846&originalType=binary&ratio=1&rotation=0&showTitle=false&size=69022&status=done&style=none&taskId=ue26920fc-8fb4-426b-88fa-5bf95f57e74&title=&width=423)
`docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/**/model,target=/models/ --name=tf_multi_v0.1 -t tensorflow/serving --model_config_file=/models/models.config`

- **注意**`**--model_config_file中指定的config路径，必须是在/models/这个文件夹下，否则会出现model_path无法找到麻烦**`
- **--model_config_file_poll_wait_seconds指定检测modes.config更新的时间**

如果布置了RESTful端口，可以使用curl 来测试两个模型是否被真正部署
![image.png](https://cdn.nlark.com/yuque/0/2022/png/33921914/1669262272836-bc28c705-d49a-48a6-919b-0b636fba48d0.png#averageHue=%2321232c&clientId=uf8ebfc78-1e2f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=503&id=u6632340f&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1006&originWidth=1548&originalType=binary&ratio=1&rotation=0&showTitle=false&size=132636&status=done&style=none&taskId=u156a8795-d327-41f9-9453-8c30cd1a6d3&title=&width=774)
随后我们可以使用client分别使用两种请求方式，对布置的不同模型发出请求，得到响应如下：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/33921914/1669263460343-0f94a0fa-dfd6-43f7-a1e6-d50b96a29e95.png#averageHue=%23d7ebeb&clientId=uf8ebfc78-1e2f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=138&id=ud95bedb7&margin=%5Bobject%20Object%5D&name=image.png&originHeight=276&originWidth=972&originalType=binary&ratio=1&rotation=0&showTitle=false&size=38429&status=done&style=none&taskId=u5f9c060c-30ae-4d18-9344-bdeec06853c&title=&width=486)


## example code

## Model Zoo

[tensorflow_model_zoo](https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc)
[tensorflow_1_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
[tensorflow_2_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)


