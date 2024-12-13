import tensorflow as tf
import numpy as np
from PIL import Image
import io

# 加载模型（根据你的文件格式选择路径）
model = tf.keras.models.load_model("C:/Users/admin/Desktop/xiaohai/shenjing/my_model.h5")  # 或者 .keras

# 定义预测函数
def predict_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        print("Raw predictions:", predictions)  # 打印原始预测值
        class_id = np.argmax(predictions)
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
        print("Image shape:", image.shape)

        return class_names[class_id]
    except Exception as e:
        return f"Error during prediction: {e}"

