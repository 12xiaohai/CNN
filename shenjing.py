# 导入必要的库
import tensorflow as tf
from tensorflow.keras import datasets, layers, models  # type: ignore
import matplotlib.pyplot as plt

# 导入 CIFAR10 数据集
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# 将测试集的前 30 张图片和类名打印出来
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 创建一个大小为 10x10 的图像窗口
plt.figure(figsize=(10, 10))

# 遍历前 30 张图片
for i in range(30):
    # 在图像窗口中创建一个子图，共 5 行 6 列
    plt.subplot(5, 6, i + 1)
    # 隐藏 x 轴和 y 轴的刻度
    plt.xticks([])
    plt.yticks([])
    # 关闭网格线
    plt.grid(False)
    # 显示图像，使用二进制颜色映射
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 设置图像的 x 轴标签为对应的类名
    plt.xlabel(class_names[train_labels[i][0]])

# 显示图像窗口
plt.show()

# 数据预处理，将像素的值标准化至0到1的区间内
train_images, test_images = train_images / 255.0, test_images / 255.0

# 构建模型
model = models.Sequential()
# 添加一个卷积层，32个滤波器，每个滤波器大小为3x3，激活函数为ReLU，输入形状为32x32x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 添加一个最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加一个卷积层，64个滤波器，每个滤波器大小为3x3，激活函数为ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加一个最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加一个卷积层，64个滤波器，每个滤波器大小为3x3，激活函数为ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 将多维输入一维化，常用在从卷积层到全连接层的过渡
model.add(layers.Flatten())
# 添加一个全连接层，64个神经元，激活函数为ReLU
model.add(layers.Dense(64, activation='relu'))
# 添加一个全连接层，10个神经元，对应10个类别
model.add(layers.Dense(10))

# 编译模型
model.compile(optimizer='adam',
              # 使用稀疏分类交叉熵作为损失函数，from_logits=True表示输出是原始的logits值，而不是概率值
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              # 监控准确率作为评估指标
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=10,
                    # 使用测试集作为验证数据
                    validation_data=(test_images, test_labels))

# 评估模型
# 绘制训练过程中的准确率曲线
plt.plot(history.history['accuracy'], label='accuracy')
# 绘制验证过程中的准确率曲线
plt.plot(history.history['val_accuracy'], label='val_accuracy')
# 设置 x 轴标签为“Epoch”
plt.xlabel('Epoch')
# 设置 y 轴标签为“Accuracy”
plt.ylabel('Accuracy')
# 设置 y 轴的范围为0.5到1
plt.ylim([0.5, 1])
# 在图中添加图例，位置为右下角
plt.legend(loc='lower right')
# 显示图像
plt.show()

# 使用测试集评估模型，返回损失值和准确率
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# 打印测试集的准确度
print("测试集的准确度:", test_acc)

# 保存模型
model.save(r"my_model.h5")  # 保存到指定路径
