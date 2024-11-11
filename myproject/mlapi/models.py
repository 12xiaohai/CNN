from django.db import models

# 定义图像上传模型
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')  # 存储上传图像的字段
    prediction = models.CharField(max_length=100, blank=True)  # 存储预测结果的字段
    uploaded_at = models.DateTimeField(auto_now_add=True)  # 自动添加上传时间

    def __str__(self):
        return f"Image uploaded at {self.uploaded_at} with prediction: {self.prediction}"

    class Meta:
        db_table = 'my_model'  # 将模型映射到名为 my_model 的数据库表
