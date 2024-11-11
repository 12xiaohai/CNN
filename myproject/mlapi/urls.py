# mlapi/urls.py
from django.urls import path
from .views import index, predict_view

urlpatterns = [
    path('', index, name='index'),  # 主页路由
    path('api/predict/', predict_view, name='predict'),  # 预测路由
]
